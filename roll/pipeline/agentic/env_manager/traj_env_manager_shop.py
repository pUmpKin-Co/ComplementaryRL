import copy
import json
import os
from collections import defaultdict
from contextlib import nullcontext
from threading import Lock
from typing import TYPE_CHECKING, Optional

import gem
import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import PreTrainedTokenizer

from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.rollout_scheduler import GroupQueueManager
from roll.pipeline.agentic.agentic_config import AgenticConfig, EnvManagerConfig
from roll.pipeline.agentic.env_manager.base_env_manager import BaseEnvManager, RolloutCache
from roll.pipeline.agentic.env_manager.memory_integration_mixin import MemoryIntegrationMixin
from roll.pipeline.agentic.env_manager.token_mask_utils import (
    custom_apply_chat_template,
    ensure_stripped_tensor_keys,
)
from roll.pipeline.agentic.llm_proxy import BaseLLMProxy, create_llm_proxy
from roll.pipeline.agentic.utils import write_data_json
from roll.utils.constants import GenerateStopReason
from roll.utils.env_action_limiter import get_global_limiter
from roll.utils.functionals import pad_to_length
from roll.utils.logging import get_logger
from roll.utils.str_utils import contains_renderable_field

if TYPE_CHECKING:
    from roll.pipeline.agentic.memory.memory_manager import MemoryManager


class TrajEnvManager(BaseEnvManager, MemoryIntegrationMixin):
    def __init__(
        self,
        worker_config: EnvManagerConfig,
        pipeline_config: AgenticConfig,
        env_config: DictConfig,
        tokenizer: PreTrainedTokenizer,
        generate_scheduler,
        output_queue: GroupQueueManager,
        thread_lock: Lock,
        mode="train",
        memory_manager: "MemoryManager" = None,
        *args,
        **kwargs,
    ):
        """ """
        super().__init__()
        self.logger = get_logger()
        self.worker_config: EnvManagerConfig = worker_config
        self.pipeline_config = pipeline_config
        self.env_config: DictConfig = env_config
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.output_queue = output_queue
        self.mode = mode
        self.generate_scheduler: RequestScheduler = generate_scheduler

        # EnvManager states
        self.rollout_cache: Optional[RolloutCache] = None
        self.group_seed = None
        self.episode_id = None
        self.running = False
        self.use_thread_lock = self.env_config.get(
            "use_thread_lock", False
        )  # 避免同时执行大量cpu操作, 可以通过env_config配置
        self.thread_lock = thread_lock if self.use_thread_lock else nullcontext()

        # Set environment step concurrency limit
        self.max_env_step_concurrent = self.env_config.get("max_env_step_concurrent", 0)
        self.env_step_limiter = None
        if self.max_env_step_concurrent > 0:
            env_tag = self.env_config.get("tag", "default")
            self.env_step_limiter = get_global_limiter(tag=env_tag, max_concurrent_calls=self.max_env_step_concurrent)

        self.cfg_template = self.pipeline_config.custom_envs[self.env_config["tag"]]
        self.agent_system_template = self.cfg_template["agent_system_template"]
        self.agent_template = self.cfg_template["agent_template"]
        self.memory_template = getattr(self.cfg_template, "memory_template", None)
        if self.memory_template is None:
            self.memory_template = """
以下是来自先前经验的提炼知识。
用它来指导你的行动，但不要过度依赖该内容。
某些见解可能过于具体，因此请仅识别最相关的内容。
在每一步选择行动之前，请在推理中明确引用有用的知识。
知识： 
{memory_message}
                """

        # 将shopper相关参数传递给环境配置
        if "shopper_config" in self.cfg_template:
            shopper_config = self.cfg_template["shopper_config"]
            self.env_config["shopper_system_template"] = shopper_config["shopper_system_template"]
            self.env_config["shopper_user_template_init"] = shopper_config["shopper_user_template_init"]
            self.env_config["shopper_model_name"] = shopper_config["shopper_model_name"]
            self.env_config["shopper_base_url"] = shopper_config["base_url"]
            self.env_config["shopper_authorization"] = shopper_config["authorization"]
            self.env_config["shopper_xrl_ant"] = shopper_config["xrl_ant"]

        self.initialize_memory_integration(memory_manager, self.env_config, self.mode)

        with self.thread_lock:
            if "seed" in self.env_config["config"]:
                self.env_config["config"]["seed"] = self.env_config["group_seed"]
            self.env = gem.make(env_id=self.env_config["env_type"], **self.env_config["config"])
            self.env = self.wrap_env_with_tools(self.env, self.env_config, memory_manager)

        if self.env_config["env_id"] == 0:
            self.logger.info(f"agent_system_template: {self.agent_system_template}")
            self.logger.info(f"agent_template: {self.agent_template}")

        # TODO: add rewards_scheduler for local ray reward workers
        self.llm_proxy: BaseLLMProxy = create_llm_proxy(
            generate_scheduler=self.generate_scheduler,
            llm_proxy_config=self.worker_config.llm_proxy,
            tokenizer=self.tokenizer,
            env=self.env,
        )

    def run_rollout_loop(self, data: DataProto):
        """
        1. Each time run_rollout_loop is called,
           it will continuously play episodes until it receives a command that data collection is complete.
           The seed needs to be reset to ensure consistency across all groups.

        Seed update logic:
           group_seed = base_seed + group_id
           episode_seed = group_seed + episode_id

        trajectory_id: f"{group_id}_{episode_id}_{episode_seed}"
        """
        assert "seed" in data.meta_info
        self.running = True
        self.group_seed = data.meta_info["seed"] + self.env_config["group_seed"]
        rollout_cache: RolloutCache = self.reset()
        start_step = self.current_step

        log_stats = defaultdict(list)

        while self.running and rollout_cache is not None:
            self.hook_on_episode_start(rollout_cache, log_stats)
            self.hook_on_turn_start(rollout_cache, log_stats)
            with Timer(name="generate", logger=None) as generate_timer:
                lm_output: DataProto = self.make_decision(rollout_cache)
                stop_reason = lm_output.meta_info.pop("stop_reason")
            log_stats["current_step"].append(self.current_step)
            log_stats["generate_time"].append(generate_timer.last)

            with Timer(name="step", logger=None) as step_timer:
                if stop_reason == GenerateStopReason.FINISH:
                    # 开始step
                    rollout_cache: RolloutCache = self.step(lm_output)
            log_stats["step_time"].append(step_timer.last)

            self.hook_on_turn_end(rollout_cache, log_stats)

            if self.running and (rollout_cache.terminated or stop_reason == GenerateStopReason.MAX_LENGTH):
                self.hook_on_episode_end(rollout_cache, log_stats)
                self.logger.debug(
                    f"group_id: {self.env_config['group_id']} env_id: {self.env_config['env_id']} episode_id: {self.episode_id} start_step {start_step} gen_stats: {log_stats}"
                )
                rollout: DataProto = self.formulate_rollouts(rollout_cache, log_stats)
                log_stats = defaultdict(list)

                traj_group_id = self.group_seed
                traj_id = f"{traj_group_id}_{self.rollout_cache.env_id}"
                rollout.non_tensor_batch["traj_group_id"] = np.array(
                    [traj_group_id] * rollout.batch.batch_size[0], dtype=object
                )
                rollout.non_tensor_batch["traj_id"] = np.array([traj_id] * rollout.batch.batch_size[0], dtype=object)
                if self.memory_manager is not None:
                    traj_group_id_w_or_wo_memory = (
                        f"{traj_group_id}_{'w_memory' if self._should_evolve_with_memory else 'wo_memory'}"
                    )
                else:
                    traj_group_id_w_or_wo_memory = f"{traj_group_id}_wo_memory"
                rollout.non_tensor_batch["traj_group_id_w_or_wo_memory"] = np.array(
                    [traj_group_id_w_or_wo_memory] * rollout.batch.batch_size[0],
                    dtype=object,
                )
                ray.get(
                    self.output_queue.put.remote(self.env_config["group_id"], self.episode_id, start_step, rollout)
                )

                rollout_cache = self.reset()
                start_step = self.current_step

        ray.get(self.output_queue.put.remote(self.env_config["group_id"], self.episode_id, start_step, None))

    def reset(self) -> RolloutCache:
        self.rollout_cache = RolloutCache(
            env_id=self.env_config["env_id"], group_id=self.env_config["group_id"], tag=self.env_config["tag"]
        )
        self.episode_id = ray.get(self.output_queue.get_episode_id.remote(self.env_config["group_id"]))
        if self.episode_id is None:
            assert not self.running
            return None
        seed = self.group_seed + self.episode_id

        with self.thread_lock:
            observation, info = self.env.reset(seed=seed, step=self.current_step)
            if observation is None:
                return None

        if "goal" not in info:
            info["goal"] = observation

        self.rollout_cache.history.append(
            {
                "observation": observation,
                "actions_left": self.env_config.max_steps - self.rollout_cache.step,
                **info,
            }
        )

        self.reset_memory_state()
        return self.rollout_cache

    def step(self, llm_output: DataProto):
        # responses = llm_output.batch['response_text']
        # TODO: 改回解码
        responses = self.tokenizer.batch_decode(llm_output.batch["responses"], skip_special_tokens=True)
        observation, reward, terminated, truncated, info = self.env.step(action=responses[0])
        suffix = info.pop("suffix", None)

        self.rollout_cache.step += 1
        self.rollout_cache.terminated = terminated
        self.rollout_cache.truncated = truncated
        if self.rollout_cache.step >= self.env_config.max_steps:
            self.rollout_cache.terminated = True
            if not terminated:
                self.rollout_cache.truncated = True
        self.rollout_cache.history[-1]["reward"] = reward
        self.rollout_cache.history[-1]["penalty"] = 0
        metrics = info.get("metrics", {})
        if not metrics.get("action_is_valid", True):
            self.rollout_cache.history[-1]["penalty"] = self.worker_config.format_penalty
        self.rollout_cache.history[-1]["llm_response"] = responses[0]
        if info is not None:
            self.rollout_cache.history[-1].update(info)

        self.rollout_cache.history.append(
            {
                "observation": observation,
                "actions_left": self.env_config.max_steps - self.rollout_cache.step,
            }
        )
        if suffix is not None:
            self.rollout_cache.history[-1]["suffix"] = suffix

        if self.mode == "val" and self.pipeline_config.render_save_dir and hasattr(self.env, "render"):
            frame = self.env.render(mode="rgb_array")
            if isinstance(frame, np.ndarray):
                self.rollout_cache.frames.append(frame)

        return self.rollout_cache

    def make_decision(self, rollout_cache: RolloutCache):
        content = self.rollout_cache.history[-1]
        render_dict = {"observation": content["observation"]}
        if contains_renderable_field(self.agent_template, "turn_idx"):
            render_dict["turn_idx"] = self.rollout_cache.step + 1
        if contains_renderable_field(self.agent_template, "suffix"):
            render_dict["suffix"] = content.get("suffix", "")
        if contains_renderable_field(self.agent_template, "actions_left"):
            render_dict["actions_left"] = content["actions_left"]
        if contains_renderable_field(self.agent_template, "max_response_length"):
            render_dict["max_response_length"] = self.env_config["max_tokens_per_step"]
        if contains_renderable_field(self.agent_template, "memory_message"):
            render_dict["memory_message"] = content.get("memory_message", "")

        messages = []
        system_token_length = 0
        if self.rollout_cache.step == 0:
            injection_mode = self.get_trajectory_memory_injection_mode()
            raw_memory_message = self.get_first_turn_retrieved_memory_message()

            system_message = self.agent_system_template
            if injection_mode == "system" and self._is_valid_memory_message(raw_memory_message):
                system_message = system_message + "\n" + self.format_retrieved_memory_for_prompt(raw_memory_message)

            messages.append({"role": "system", "content": system_message})

            if injection_mode == "user" and self._is_valid_memory_message(raw_memory_message):
                messages.append(
                    {"role": "user", "content": self.format_retrieved_memory_for_prompt(raw_memory_message)}
                )

            system_tokens = custom_apply_chat_template(
                messages=messages,
                tokenizer=self.tokenizer,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            system_token_length = len(system_tokens)

        if "user_persona" in content:
            use_messages = [
                {
                    "role": "user",
                    "content": f"\n用户个人文档：{json.dumps(content['user_persona'], ensure_ascii=False)}",
                },
                {"role": "assistant", "content": "ok"},
            ]
            user_tokens = custom_apply_chat_template(
                messages=use_messages,
                tokenizer=self.tokenizer,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            system_token_length += len(user_tokens)

            messages.extend(use_messages)

        if len(self.rollout_cache.history) > 1 and self.rollout_cache.history[-2].get("use_tool", False):
            messages.append({"role": "tool", "content": content["observation"]})
        else:
            messages.append({"role": "user", "content": self.agent_template.format(**render_dict)})

        content["messages"] = messages
        content["system_token_length"] = system_token_length

        prompt_ids = custom_apply_chat_template(
            messages=messages,
            tokenizer=self.tokenizer,
            add_generation_prompt=True,
            enable_thinking=self.env_config.get("enable_thinking", True),
        )
        history_token_ids = []
        for items in self.rollout_cache.history[:-1]:
            history_token_ids.extend(items["prompt_ids"])
            history_token_ids.extend(items["response_ids"])
        input_ids = history_token_ids + prompt_ids
        if len(input_ids) >= self.pipeline_config.sequence_length:
            self.logger.warning(
                f"sequence_length = {self.pipeline_config.sequence_length} input_ids length = {len(input_ids)},"
                f"maybe you should increase the response_length"
            )
            return DataProto(meta_info={"stop_reason": GenerateStopReason.MAX_LENGTH})

        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        position_ids = attention_mask.cumsum(dim=-1)
        lm_input = DataProto()
        lm_input.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0],
        )
        max_new_tokens = min(
            self.env_config["max_tokens_per_step"],
            self.worker_config.generating_args.max_new_tokens,
            self.pipeline_config.sequence_length - input_ids.shape[1],
        )

        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["max_new_tokens"] = min(max_new_tokens, self.pipeline_config.sequence_length)
        lm_input.meta_info["src_rank"] = self.env_config["env_id"]

        input_messages = [item for items in self.rollout_cache.history for item in items["messages"]]

        lm_output: DataProto = self.llm_proxy.generate(
            messages=input_messages, lm_input=lm_input, generation_config=generation_config
        )
        if lm_output is None:
            return DataProto(meta_info={"stop_reason": GenerateStopReason.ABORT})

        response_ids = lm_output.batch["responses"][0]
        response_ids = response_ids.tolist()
        content["prompt_ids"] = prompt_ids
        content["response_ids"] = response_ids
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        content["messages"].append({"role": "assistant", "content": response_text})
        lm_output.meta_info["stop_reason"] = GenerateStopReason.FINISH

        return lm_output

    def formulate_rollouts(self, rollout_cache: RolloutCache, log_stats: dict = None):
        if "observation" in rollout_cache.history[-1]:
            rollout_cache.history.pop(-1)
        history = rollout_cache.history[:-1]
        last_cache = copy.deepcopy(rollout_cache.history[-1])
        history.append(last_cache)

        episode_score = last_cache["reward"]
        scores = [episode_score for i in self.rollout_cache.history]

        token_ids = []
        prompt_masks = []
        response_masks = []
        question_masks = []
        for idx, items in enumerate(self.rollout_cache.history):
            token_ids.extend(items["prompt_ids"])
            token_ids.extend(items["response_ids"])
            prompt_masks.extend([1] * len(items["prompt_ids"]) + [0] * len(items["response_ids"]))
            response_masks.extend([0] * len(items["prompt_ids"]) + [1] * len(items["response_ids"]))

            if idx == 0:
                # First turn: exclude system message tokens
                system_token_length = items.get("system_token_length", 0)
                question_masks.extend(
                    [0] * system_token_length
                    + [1] * (len(items["prompt_ids"]) - system_token_length)
                    + [0] * len(items["response_ids"])
                )
            elif getattr(self.pipeline_config, "question_on_all_turns", False):
                question_masks.extend([1] * len(items["prompt_ids"]) + [0] * len(items["response_ids"]))
            else:
                question_masks.extend([0] * len(items["prompt_ids"]) + [0] * len(items["response_ids"]))

        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * len(token_ids), dtype=torch.long).unsqueeze(0)
        response_mask = torch.tensor(response_masks, dtype=torch.bool).unsqueeze(0)
        question_mask = torch.tensor(question_masks, dtype=torch.bool).unsqueeze(0)

        first_response_idx = response_masks.index(1)
        prompt_masks = [1] * first_response_idx + [0] * (len(token_ids) - first_response_idx)
        prompt_mask = torch.tensor(prompt_masks, dtype=torch.bool).unsqueeze(0)
        score_tensor = torch.tensor([0] * len(token_ids), dtype=torch.float).unsqueeze(0)
        score_tensor[0][-1] = episode_score
        position_ids = attention_mask.cumsum(dim=-1)

        lm_input = DataProto()
        lm_input.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0],
        )

        response_length = response_mask.sum(dim=-1).float().mean().item()

        # TODO: move pad to pipeline
        input_ids = pad_to_length(
            input_ids, length=self.pipeline_config.sequence_length, pad_value=self.tokenizer.pad_token_id
        )
        attention_mask = pad_to_length(attention_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        position_ids = pad_to_length(position_ids, length=self.pipeline_config.sequence_length, pad_value=0)
        response_mask = pad_to_length(response_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        prompt_mask = pad_to_length(prompt_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        question_mask = pad_to_length(question_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        score_tensor = pad_to_length(score_tensor, length=self.pipeline_config.sequence_length, pad_value=0)

        batch_update = ensure_stripped_tensor_keys(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "response_mask": response_mask,
                "prompt_mask": prompt_mask,
                "question_mask": question_mask,
                "scores": score_tensor,
            }
        )
        lm_input.batch.update(batch_update)
        lm_input.non_tensor_batch.update(
            {
                "env_ids": np.array([self.rollout_cache.env_id], dtype=object),
                "group_ids": np.array([self.rollout_cache.group_id], dtype=object),
                "tags": np.array([self.rollout_cache.tag], dtype=object),
                "frames": np.array([self.rollout_cache.frames], dtype=object),
                "step_scores": np.array([scores], dtype=object),
                "episode_scores": np.array([episode_score], dtype=object),
            }
        )

        metrics = self.rollout_cache.history[-1].get("metrics", {})
        env_metric = {
            "success": float(metrics.get("success", episode_score > 0)),
            "num_actions": rollout_cache.step,
        }
        custom_metric = {}
        for turn in self.rollout_cache.history:
            for k, v in turn.get("metrics", {}).items():
                if k == "success" or k == "task_idx":
                    continue
                if k not in custom_metric:
                    custom_metric[k] = []
                custom_metric[k].append(float(v))

        # TODO: 是否保留每一回合的score
        for k, v in custom_metric.items():
            # env_metric[k] = np.sum(v) / len(self.rollout_cache.history)
            env_metric[k] = last_cache["metrics"][k]

        env_metric = {f"env/{rollout_cache.tag}/{k}": v for k, v in env_metric.items()}
        env_metric["env/response_length"] = response_length
        lm_input.meta_info = {"metrics": env_metric}
        env_metric = self.collect_memory_metrics(log_stats, env_metric)
        self.add_memory_metadata_to_rollout(lm_input, rollout_cache, self.episode_id, self.group_seed)

        task_idx = metrics.get("task_idx", 0)
        log_metric = copy.deepcopy(env_metric)
        log_metric["task_idx"] = task_idx
        log_metric["episode_score"] = episode_score

        traj_messages = [item for items in self.rollout_cache.history for item in items["messages"]]
        log_metric["traj_messages"] = traj_messages

        log_dir = os.path.join(self.pipeline_config.base_dir, "env_metric", str(self.current_step))
        log_path = os.path.join(log_dir, f'rollout-{self.env_config["tag"]}_task{task_idx}_re{episode_score}.json')
        try:
            os.makedirs(log_dir, exist_ok=True)
            write_data_json(log_metric, log_path)
        except (OSError, IOError) as e:
            self.logger.warning(f"Failed to write log to {log_path}: {e}. Continuing without logging to disk.")
        return lm_input
