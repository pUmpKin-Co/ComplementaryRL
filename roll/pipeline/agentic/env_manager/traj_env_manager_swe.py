import copy
import json
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from threading import Lock
from typing import TYPE_CHECKING, Dict, Optional

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
from roll.pipeline.agentic.env.swe_env.utils import MultiprocessSafeLogger
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
from roll.utils.str_utils import contains_renderable_field

if TYPE_CHECKING:
    from roll.pipeline.agentic.memory.memory_manager import MemoryManager

try:
    DEBUG = os.environ["ROLL_LOG_LEVEL"] == "DEBUG"
    LOG = os.environ["ROLL_LOG_LEVEL"] == "DEBUG"
except:
    DEBUG = False
    LOG = False

loggers = {}


def format_template_with_memory(template: str, memory_message: str) -> str:
    if contains_renderable_field(template, "memory_message") or "{{memory_message}}" in template:
        formatted_template = template.replace("{{memory_message}}", "{memory_message}")
        return formatted_template.format(memory_message=memory_message)
    else:
        return template


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
        super().__init__()
        self.worker_config: EnvManagerConfig = worker_config
        self.pipeline_config = pipeline_config
        self.env_config: DictConfig = env_config

        # tag_group_id_group_seed_env_id
        logger_name = f"{self.env_config['tag']}_group{self.env_config['group_id']}_seed{self.env_config['group_seed']}_env{self.env_config['env_id']}"
        self.logger = None  # 初始化 logger 为 None
        self.logger = MultiprocessSafeLogger(
            path=os.path.join(
                self.pipeline_config.base_dir,
                "log/traj_manager",
                f"traj_manager-{logger_name}.log",
            )
        )

        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.output_queue = output_queue
        self.mode = mode
        self.generate_scheduler: RequestScheduler = generate_scheduler

        self.initialize_memory_integration(memory_manager, self.env_config, self.mode)

        # EnvManager states
        self.rollout_cache: Optional[RolloutCache] = None
        self.group_seed = None
        self.episode_id = None
        self.running = False
        self.use_thread_lock = self.env_config.get(
            "use_thread_lock", False
        )  # 避免同时执行大量cpu操作, 可以通过env_config配置
        self.thread_lock = thread_lock if self.use_thread_lock else nullcontext()
        # with self.thread_lock:
        if "seed" in self.env_config["config"]:
            self.env_config["config"]["seed"] = self.env_config["group_seed"]
        self.env = gem.make(env_id=self.env_config["env_type"], **self.env_config["config"])
        self.env = self.wrap_env_with_tools(self.env, self.env_config, memory_manager)

        # Set environment step concurrency limit
        self.max_env_step_concurrent = self.env_config.get("max_env_step_concurrent", 0)
        self.env_step_limiter = None
        if self.max_env_step_concurrent > 0:
            env_tag = self.env_config.get("tag", "default")
            self.env_step_limiter = get_global_limiter(
                tag=env_tag,
                max_concurrent_calls=self.max_env_step_concurrent,
            )

        if DEBUG:
            print("====== traj_env_manager_swe init ======")
        cfg_template = self.pipeline_config.custom_envs[self.env_config["tag"]]
        self.agent_system_template = cfg_template["agent_system_template"]
        self.agent_instance_template = cfg_template["agent_instance_template"]  # added for swe_env
        self.agent_obs_template = cfg_template["agent_obs_template"]  # added for swe_env
        self.agent_last_step_template = cfg_template["agent_last_step_template"]  # added for swe_env
        self.agent_template = self.agent_obs_template
        self.memory_template = self.env_config["config"].pop("memory_template", None)
        if self.memory_template is None:
            self.memory_template = """
Below is distilled knowledge from earlier experiences.
Use it to guide your actions without overfitting.
Some insights may be overly specific, so identify only what is most relevant.
Knowledge: 
{memory_message}
"""

        self.cur_seq_length = 0
        self.mode = self.pipeline_config.custom_envs[self.env_config["tag"]]["env_config"]["mode"]  # fix

        # TODO: add rewards_scheduler for local ray reward workers
        self.llm_proxy: BaseLLMProxy = create_llm_proxy(
            generate_scheduler=self.generate_scheduler,
            llm_proxy_config=self.worker_config.llm_proxy,
            tokenizer=self.tokenizer,
            env=self.env,
        )

    def build_memory_query(self, last_history: Dict, task_goal: Optional[str] = None) -> str:
        """
        Build a memory query from the current state.

        This is the default implementation from BaseEnvManager. Override this
        method to customize query building for your environment.

        Args:
            last_history: The last entry in rollout_cache.history
            task_goal: Optional task goal extracted from the environment

        Returns:
            Query string for memory search
        """
        return self.env.problem_statement

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
        if DEBUG:
            print(f"[DEBUG][RUN_ROLLOUT_LOOP] group_seed: {self.group_seed}")
        rollout_cache: RolloutCache = self.reset()
        start_step = self.current_step
        log_stats = defaultdict(list)

        if LOG:
            self.logger.info(
                f"[RUN_ROLLOUT_LOOP][ROLLOUT START][group_id: {self.env_config['group_id']}, env_id: {self.env_config['env_id']}, episode_id: {self.episode_id}, start_step: {start_step}]"
            )

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
                    rollout_cache: RolloutCache = self.step(lm_output)
            log_stats["step_time"].append(step_timer.last)

            self.hook_on_turn_end(rollout_cache, log_stats)

            if stop_reason == GenerateStopReason.MAX_LENGTH:
                rollout_cache.stop_reason = "max_length"
            elif stop_reason == GenerateStopReason.FINISH:
                rollout_cache.stop_reason = "finish"
            elif stop_reason == GenerateStopReason.ABORT:
                rollout_cache.stop_reason = "abort"
            elif rollout_cache.terminated:
                rollout_cache.stop_reason = "finish"
            else:
                rollout_cache.stop_reason = "unknown"
                if DEBUG:
                    print("[DEBUG][Attention]rollout_cache.stop_reason is unknown")
                    print(f"[DEBUG]rollout_cache: \n{rollout_cache}")

            # self.logger.info(f'[RUN_ROLLOUT_LOOP]rollout_cache.stop_reason: {rollout_cache.stop_reason}')
            if self.running and (
                rollout_cache.stop_reason == "max_length" or rollout_cache.terminated or rollout_cache.truncated
            ):  # change
                if LOG:
                    self.logger.info(
                        f"[RUN_ROLLOUT_LOOP][STOP][group_id: {self.env_config['group_id']}, env_id: {self.env_config['env_id']}, episode_id: {self.episode_id}, start_step: {start_step}, gen_stats: {log_stats}, stop_reason: {rollout_cache.stop_reason}], terminated: {rollout_cache.terminated}, truncated: {rollout_cache.truncated}"
                    )
                self.hook_on_episode_end(rollout_cache, log_stats)
                self.logger.debug(
                    f"group_id: {self.env_config['group_id']} env_id: {self.env_config['env_id']} episode_id: {self.episode_id} start_step {start_step} gen_stats: {log_stats}"
                )
                rollout: DataProto = self.formulate_rollouts(rollout_cache, log_stats)

                log_stats = defaultdict(list)
                traj_group_id = (
                    f"{self.rollout_cache.tag}_{self.rollout_cache.group_id}_{self.episode_id}_{self.group_seed}"
                )
                traj_id = f"{traj_group_id}_{self.rollout_cache.env_id}"
                rollout.non_tensor_batch["traj_group_id"] = np.array(
                    [traj_group_id] * rollout.batch.batch_size[0],
                    dtype=object,
                )
                rollout.non_tensor_batch["traj_id"] = np.array([traj_id] * rollout.batch.batch_size[0], dtype=object)
                if self.memory_manager is not None:
                    traj_group_id_w_or_wo_memory = (
                        f"{traj_group_id}_{'w_memory' if self._should_evolve_with_memory else 'wo_memory'}"
                    )
                else:
                    # Default value when memory is not configured
                    traj_group_id_w_or_wo_memory = f"{traj_group_id}_wo_memory"
                rollout.non_tensor_batch["traj_group_id_w_or_wo_memory"] = np.array(
                    [traj_group_id_w_or_wo_memory] * rollout.batch.batch_size[0],
                    dtype=object,
                )
                ray.get(
                    self.output_queue.put.remote(
                        self.env_config["group_id"],
                        self.episode_id,
                        start_step,
                        rollout,
                    )
                )

                rollout_cache = self.reset()
                start_step = self.current_step

        ray.get(
            self.output_queue.put.remote(
                self.env_config["group_id"],
                self.episode_id,
                start_step,
                None,
            )
        )

    def reset(self) -> RolloutCache:
        self.rollout_cache = RolloutCache(
            env_id=self.env_config["env_id"],
            group_id=self.env_config["group_id"],
            tag=self.env_config["tag"],
        )
        self.episode_id = ray.get(self.output_queue.get_episode_id.remote(self.env_config["group_id"]))
        if self.episode_id is None:
            assert not self.running
            return None
        seed = self.group_seed + self.episode_id

        logger_name = (
            f"{self.env_config['tag']}_group{self.env_config['group_id']}_seed{seed}_env{self.env_config['env_id']}"
        )

        if LOG:
            log_path = os.path.join(
                self.pipeline_config.base_dir,
                "log/traj_manager",
                f"{logger_name}.log",
            )
            if self.logger is None:
                self.logger = MultiprocessSafeLogger(path=log_path)
            else:
                self.logger.update_log_path(path=log_path)

            self.logger.info(
                f"[manager.reset] tag: {self.env_config['tag']}, seed: {seed} (group_seed: {self.group_seed}, env_id: {self.env_config['env_id']}, episode_id: {self.episode_id})"
            )
        if DEBUG:
            print(
                f"\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}[manager.reset] tag: {self.env_config['tag']}, seed: {seed} (group_seed: {self.group_seed}, env_id: {self.env_config['env_id']}, episode_id: {self.episode_id})"
            )

        observation, info = self.env.reset(seed=seed, step=self.current_step)
        if observation is None:
            return None
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
        responses = self.tokenizer.batch_decode(llm_output.batch["responses"], skip_special_tokens=True)
        if DEBUG:
            print(
                f"\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}[manager.step] tag: {self.env_config['tag']}, seed: {self.group_seed + self.episode_id} (group_seed: {self.group_seed}, env_id: {self.env_config['env_id']}, episode_id: {self.episode_id})"
            )
        if responses == ["max_length"]:
            if DEBUG:
                print(f"[DEBUG][max_length]responses: {responses}")
        if LOG:
            self.logger.info(
                f"[manager.step] tag: {self.env_config['tag']}, seed: {self.group_seed + self.episode_id} (group_seed: {self.group_seed}, env_id: {self.env_config['env_id']}, episode_id: {self.episode_id})"
            )
        observation, reward, terminated, truncated, info = self.env.step(action=responses[0])
        suffix = info.pop("suffix", None)

        self.rollout_cache.step += 1
        self.rollout_cache.terminated = terminated
        self.rollout_cache.truncated = truncated
        if self.rollout_cache.step >= self.env_config.max_steps:
            self.rollout_cache.terminated = True
            self.rollout_cache.truncated = True
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
        if contains_renderable_field(self.agent_obs_template, "turn_idx"):
            render_dict["turn_idx"] = self.rollout_cache.step + 1
        if contains_renderable_field(self.agent_obs_template, "suffix"):
            render_dict["suffix"] = content.get("suffix", "")
        if contains_renderable_field(self.agent_obs_template, "actions_left"):
            render_dict["actions_left"] = content["actions_left"]
        if contains_renderable_field(self.agent_obs_template, "max_response_length"):
            render_dict["max_response_length"] = self.env_config["max_tokens_per_step"]
        if contains_renderable_field(self.agent_obs_template, "memory_message"):
            render_dict["memory_message"] = content.get("memory_message", "")
        # current messages
        messages = []
        system_token_length = 0

        if self.rollout_cache.step == 0:
            injection_mode = self.get_trajectory_memory_injection_mode()
            raw_memory_message = self.get_first_turn_retrieved_memory_message() or "[No Memory Available]"

            system_message = self.agent_system_template
            user_first = self.agent_instance_template.format(problem_statement=content.get("observation", ""))
            if content.get("suffix"):
                system_message = system_message.replace("/testbed", content["suffix"])
                user_first = user_first.replace("/testbed", content["suffix"])

            if injection_mode == "system" and self._is_valid_memory_message(raw_memory_message):
                system_message = system_message + "\n" + self.format_retrieved_memory_for_prompt(raw_memory_message)

            messages = [{"role": "system", "content": system_message}]
            if injection_mode == "user" and self._is_valid_memory_message(raw_memory_message):
                messages.append(
                    {"role": "user", "content": self.format_retrieved_memory_for_prompt(raw_memory_message)}
                )

            # Calculate "prefix" token length for question mask.
            # If we inject memory as a user message, we still exclude it from question tokens.
            prefix_messages = messages if injection_mode == "user" else [{"role": "system", "content": system_message}]
            system_tokens = custom_apply_chat_template(
                messages=prefix_messages,
                tokenizer=self.tokenizer,
                add_generation_prompt=False,
            )
            system_token_length = len(system_tokens)

            messages.append(
                {
                    "role": "user",
                    "content": user_first,
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": self.agent_obs_template.format(**render_dict),
                }
            )
            messages[-1]["content"] = self.agent_last_step_template.format(observation=messages[-1]["content"])
        content["messages"] = messages

        # Store system token length for question mask computation
        content["system_token_length"] = system_token_length

        prompt_ids = custom_apply_chat_template(
            messages=messages,
            tokenizer=self.tokenizer,
            add_generation_prompt=True,
        )

        history_token_ids = []
        for items in self.rollout_cache.history[:-1]:
            history_token_ids.extend(items["prompt_ids"])
            history_token_ids.extend(items["response_ids"])
        input_ids = history_token_ids + prompt_ids
        self.cur_seq_length = len(input_ids)

        # sequence length warining
        if len(input_ids) >= self.pipeline_config.sequence_length:
            print(
                f"sequence_length = {self.pipeline_config.sequence_length} input_ids length = {len(input_ids)},"
                f"maybe you should increase the response_length"
            )
            return DataProto(meta_info={"stop_reason": GenerateStopReason.MAX_LENGTH})

        # convert to tensor for lm_input
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

        # compute max_new_tokens
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
            messages=input_messages,
            lm_input=lm_input,
            generation_config=generation_config,
        )

        if lm_output is None:
            return DataProto(meta_info={"stop_reason": GenerateStopReason.ABORT})

        response_ids = lm_output.batch["responses"][0]
        response_ids = response_ids.tolist()
        lm_output.meta_info["stop_reason"] = GenerateStopReason.FINISH
        content["prompt_ids"] = prompt_ids
        content["response_ids"] = response_ids
        content["messages"].append(
            {
                "role": "assistant",
                "content": self.tokenizer.decode(response_ids, skip_special_tokens=True),
            }
        )

        return lm_output

    def formulate_rollouts(self, rollout_cache: RolloutCache, log_stats: dict = None):
        """ """
        if "observation" in rollout_cache.history[-1]:
            rollout_cache.history.pop(-1)
        history = rollout_cache.history[:-1]
        last_cache = copy.deepcopy(rollout_cache.history[-1])

        traj_messages = [item for items in self.rollout_cache.history for item in items["messages"]]
        if DEBUG:
            print(f'\n**********[METRIC]**********\n{last_cache["metrics"]}')

        last_cache.pop("reward", None)
        history.append(last_cache)

        scores = [i["reward"] for i in self.rollout_cache.history]
        unittest_output, final_reward, info = self.env.calculate_reward(
            stop_reason=rollout_cache.stop_reason, mode=self.mode
        )
        scores[-1] = max(final_reward, max(scores))
        episode_score = scores[-1]
        if self.mode == "train" and (rollout_cache.stop_reason in ["max_length"]):
            episode_score = min(final_reward, 0.5)
            scores[-1] = episode_score
            unittest_output = f"{unittest_output}\n\nAttention: (Train Mode) Reach max turn or max length, episode_score will be set to 0/0.5. stop_reason: {rollout_cache.stop_reason}, final_reward: {final_reward}, episode_score: {episode_score}"
            print(
                f"[FORMULATE_ROLLOUTS]⚠️ Attention: (Train Mode) Reach max turn or max length), episode_score will be set to 0/0.5, stop_reason: {rollout_cache.stop_reason}, final_reward: {final_reward}, episode_score: {episode_score}"
            )

        token_ids = []
        prompt_masks = []
        response_masks = []
        question_masks = []
        step_response_length_list = []
        step_prompt_length_list = []

        for idx, items in enumerate(self.rollout_cache.history):
            prompt_ids_len = len(items["prompt_ids"])
            response_ids_len = len(items["response_ids"])

            token_ids.extend(items["prompt_ids"])
            token_ids.extend(items["response_ids"])
            prompt_masks.extend([1] * prompt_ids_len + [0] * response_ids_len)
            response_masks.extend([0] * prompt_ids_len + [1] * response_ids_len)

            # Build question mask: only mark user prompt tokens (exclude system message in first turn)
            if idx == 0:
                # First turn: exclude system message tokens
                system_token_length = items.get("system_token_length", 0)
                question_masks.extend(
                    [0] * system_token_length + [1] * (prompt_ids_len - system_token_length) + [0] * response_ids_len
                )
            elif getattr(self.pipeline_config, "question_on_all_turns", False):
                question_masks.extend([1] * prompt_ids_len + [0] * response_ids_len)
            else:
                question_masks.extend([0] * prompt_ids_len + [0] * response_ids_len)

            step_response_length = len(items["response_ids"])
            step_response_length_list.append(step_response_length)
            step_prompt_length = len(items["prompt_ids"])
            step_prompt_length_list.append(step_prompt_length)

        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * len(token_ids), dtype=torch.long).unsqueeze(0)
        response_mask = torch.tensor(response_masks, dtype=torch.bool).unsqueeze(0)
        question_mask = torch.tensor(question_masks, dtype=torch.bool).unsqueeze(0)
        step_response_length_tensor = torch.tensor(step_response_length_list, dtype=torch.float)
        step_prompt_length_tensor = torch.tensor(step_prompt_length_list, dtype=torch.float)

        prompt_masks_sys_obs = copy.deepcopy(prompt_masks)
        prompt_masks_sys_obs_tensor = torch.tensor(prompt_masks_sys_obs, dtype=torch.bool).unsqueeze(0)

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
            input_ids,
            length=self.pipeline_config.sequence_length,
            pad_value=self.tokenizer.pad_token_id,
        )
        attention_mask = pad_to_length(
            attention_mask,
            length=self.pipeline_config.sequence_length,
            pad_value=0,
        )
        position_ids = pad_to_length(
            position_ids,
            length=self.pipeline_config.sequence_length,
            pad_value=0,
        )
        response_mask = pad_to_length(
            response_mask,
            length=self.pipeline_config.sequence_length,
            pad_value=0,
        )
        prompt_mask = pad_to_length(
            prompt_mask,
            length=self.pipeline_config.sequence_length,
            pad_value=0,
        )
        question_mask = pad_to_length(
            question_mask,
            length=self.pipeline_config.sequence_length,
            pad_value=0,
        )
        score_tensor = pad_to_length(
            score_tensor,
            length=self.pipeline_config.sequence_length,
            pad_value=0,
        )

        metrics = self.rollout_cache.history[-1].get("metrics", {})
        metrics["reward"] = episode_score

        if DEBUG:
            print(
                f"[EnvManager] closing env ... because of {self.rollout_cache.stop_reason}, task_idx: {metrics.get('task_idx','')}"
            )
        self.env.close(stop_reason=self.rollout_cache.stop_reason)
        if DEBUG:
            print(f"[EnvManager] env closed ....")

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
                "traj_rollout_time": np.array(
                    [float(metrics.get("traj_rollout_time", 0))],
                    dtype=object,
                ),
                "traj_env_time": np.array([float(metrics.get("traj_total_time", 0))], dtype=object),
            }
        )

        # length
        avg_step_response_length = round(step_response_length_tensor.mean().item(), 2)
        avg_step_prompt_length = round(step_prompt_length_tensor.mean().item(), 2)
        max_step_response_length = round(step_response_length_tensor.max().item(), 2)
        max_step_prompt_length = round(step_prompt_length_tensor.max().item(), 2)
        min_step_response_length = round(step_response_length_tensor.min().item(), 2)
        min_step_prompt_length = round(step_prompt_length_tensor.min().item(), 2)

        # traj-level metric
        env_metric = {
            "success": float(metrics.get("success", episode_score > 0)),
            "reward": float(metrics.get("reward", episode_score)),
            "truncated": float(metrics.get("truncated", 0)),  # max steps, calculate reward
            "env_timeout": float(metrics.get("env_timeout", 0)),  # interaction time exceeds, not calculate reward
            "env_failed": float(metrics.get("env_failed", 0)),  # reset failed, not calculate reward
            "reach_max_length": (
                1.0 if rollout_cache.stop_reason in ["max_length"] else 0.0
            ),  # reach max length, not calculate reward
            "num_actions": rollout_cache.step,
            "turn_count": float(metrics.get("turn_count", 0)),
            "retry_times": float(metrics.get("retry_times", 0)),
            "action_is_valid": float(metrics.get("action_is_valid", 0)),
            "action_is_effective": float(metrics.get("action_is_effective", 0)),
            "traj_reset_time": float(metrics.get("traj_reset_time", 0)),
            "traj_reward_time": float(metrics.get("traj_reward_time", 0)),
            "traj_env_time": float(metrics.get("traj_total_time", 0)),
            "traj_rollout_time": float(metrics.get("traj_rollout_time", 0)),
            "avg_step_response_length": avg_step_response_length,
            "avg_step_prompt_length": avg_step_prompt_length,
            "max_step_response_length": max_step_response_length,
            "max_step_prompt_length": max_step_prompt_length,
            "min_step_response_length": min_step_response_length,
            "min_step_prompt_length": min_step_prompt_length,
        }
        traj_keys = list(env_metric.keys())

        # step-level metric
        custom_metric = {}
        for turn in self.rollout_cache.history:
            for k, v in turn.get("metrics", {}).items():
                if k in traj_keys:
                    continue
                if k == "task_idx":
                    continue
                if k not in custom_metric:
                    custom_metric[k] = []
                custom_metric[k].append(float(v))
        for k, v in custom_metric.items():
            env_metric[k] = np.sum(v) / len(self.rollout_cache.history)

        # add tag
        env_metric = {f"env/{rollout_cache.tag}/{k}": v for k, v in env_metric.items()}
        # response_length
        env_metric["env/response_length"] = response_length
        lm_input.meta_info = {"metrics": env_metric}
        if DEBUG:
            print(
                f"\n[formulate_rollouts][env_metric]({self.rollout_cache.tag}_group{self.rollout_cache.group_id}_seed{self.group_seed}_env{self.rollout_cache.env_id}){env_metric}"
            )
        # self.logger.info(f'[FORMULATE_ROLLOUTS]env_metric: {env_metric}')

        prompt_length = torch.tensor(prompt_masks_sys_obs).sum(dim=-1).float().mean().item()
        length = prompt_length + response_length
        max_seq_length = self.pipeline_config.sequence_length
        step_count = (len(traj_messages) - 1) // 2
        if DEBUG:
            print(
                f"[DEBUG] length: {length}, max_seq_length: {max_seq_length}, prompt_length: {prompt_length}, response_length: {response_length}"
            )

        if length > max_seq_length:
            stop_reason = "max_length"
        elif metrics.get("success", True):
            stop_reason = "finish"
        elif metrics.get("env_timeout"):
            stop_reason = "env_timeout"
        elif metrics.get("env_failed"):
            stop_reason = "env_failed"
        elif metrics.get("truncated"):
            stop_reason = "truncated"
        elif rollout_cache.stop_reason in ["max_length", "abort"]:
            stop_reason = rollout_cache.stop_reason
        else:
            stop_reason = "unknown"

        task_idx = metrics.get("task_idx", 0)
        tag = self.env_config["tag"]
        save = {
            "task_idx": task_idx,
            "env_id": self.env_config["env_id"],
            "group_id": self.env_config["group_id"],
            "tag": self.env_config["tag"],
            "length": length,
            "step_count": step_count,
            "stop_reason": stop_reason,
            "episode_score": episode_score,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "max_seq_length": max_seq_length,
            "traj_messages": traj_messages,
            "metrics": metrics,
            "env_metric": env_metric,
            "problem_statement": self.env.problem_statement,
        }

        # columns_config keys will be removed from data_proto after dump
        lm_input.non_tensor_batch["model_name"] = np.array(
            [os.path.basename(self.pipeline_config.base_dir)], dtype=object
        )
        lm_input.non_tensor_batch["save_content"] = np.array([json.dumps(save)], dtype=object)
        lm_input.non_tensor_batch["step"] = np.array([self.current_step], dtype=object)
        lm_input.non_tensor_batch["task_idx"] = np.array([task_idx], dtype=object)
        lm_input.non_tensor_batch["stop_reason"] = np.array([stop_reason], dtype=object)
        lm_input.non_tensor_batch["mode"] = np.array([self.mode], dtype=object)
        lm_input.non_tensor_batch["episode_scores"] = np.array([episode_score], dtype=object)
        colummns_config = [
            ["task_idx", "bigint"],
            ["model_name", "string"],
            ["stop_reason", "string"],
            ["episode_score", "double"],
            ["mode", "string"],
            ["save_content", "string"],
        ]
        lm_input.meta_info["COLUMMNS_CONFIG"] = colummns_config
        self.add_memory_metadata_to_rollout(lm_input, rollout_cache, self.episode_id, self.group_seed)

        base_dir = self.pipeline_config.base_dir
        if base_dir:
            log_path = os.path.join(
                base_dir,
                "rollouts",
                f'{time.strftime("%Y%m%d_%H%M%S", time.localtime())}-{self.rollout_cache.tag}_{self.rollout_cache.group_id}_{self.rollout_cache.env_id}_re{episode_score}_step{self.rollout_cache.step}.json',
            )
            write_data_json(save, log_path)
        return lm_input
