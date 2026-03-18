import copy
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
from roll.pipeline.agentic.env_manager.base_env_manager import BaseEnvManager, RolloutCache
from roll.pipeline.agentic.env_manager.memory_integration_mixin import MemoryIntegrationMixin
from roll.pipeline.agentic.env_manager.token_mask_utils import (
    compute_conversation_end_token_id,
    custom_apply_chat_template,
)
from roll.pipeline.agentic.llm_proxy import BaseLLMProxy, create_llm_proxy
from roll.pipeline.agentic.utils import write_data_json
from roll.utils.constants import GenerateStopReason
from roll.utils.env_action_limiter import get_global_limiter
from roll.utils.functionals import aggregate_metrics, pad_to_length
from roll.utils.logging import get_logger
from roll.utils.str_utils import contains_renderable_field

if TYPE_CHECKING:
    from roll.pipeline.agentic.memory.memory_manager import MemoryManager


class TrajEnvManager(MemoryIntegrationMixin, BaseEnvManager):
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
        # Set environment step concurrency limit
        self.max_env_step_concurrent = self.env_config.get("max_env_step_concurrent", 0)
        self.env_step_limiter = nullcontext()
        if self.max_env_step_concurrent > 0:
            env_tag = self.env_config.get("tag", "default")
            self.env_step_limiter = get_global_limiter(
                tag=env_tag,
                max_concurrent_calls=self.max_env_step_concurrent,
            )

        self.memory_template = self.env_config["config"].pop("memory_template", None)
        with self.thread_lock, self.env_step_limiter:
            if "seed" in self.env_config["config"]:
                self.env_config["config"]["seed"] = self.env_config["group_seed"]
            self.env = gem.make(
                env_id=self.env_config["env_type"],
                **self.env_config["config"],
            )
            self.env = self.wrap_env_with_tools(self.env, self.env_config, memory_manager)

        self.cfg_template = self.pipeline_config.custom_envs[self.env_config["tag"]]
        self.agent_system_template = self.cfg_template["agent_system_template"]
        self.agent_template = self.cfg_template["agent_template"]
        if self.memory_template is None:
            self.memory_template = """
Below is distilled knowledge from earlier experiences.
Use it to guide your actions without overfitting.
Some insights may be overly specific, so identify only what is most relevant.
Knowledge: 
{memory_message}
                """

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

    def build_turn_memory_value(self, action: str, action_result: str, **kwargs) -> str:
        value = f"{action}\n"
        value += f"Feedback: {action_result}"
        return value

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
        task_tag = self.env_config["tag"]

        if "webshop" in task_tag.lower():
            assert task_goal is not None
            return f"Goal: {task_goal}"
        elif "minihack" in task_tag.lower():
            raw_obs = last_history.get("raw_obs", "")
            return f"Observation: {raw_obs}"
        else:
            observation = last_history.get("observation", "")
            if task_goal is not None:
                return f"Goal: {task_goal}\nObservation: {observation}"
            elif "goal" in last_history:
                goal = last_history["goal"]
                return f"Goal: {goal}\nObservation: {observation}"
            else:
                return f"Observation: {observation}"

    def run_rollout_loop(self, data: DataProto):
        """
        1. Each time run_rollout_loop is called,
           it will continuously play episodes until it receives a command that data collection is complete.
           The seed needs to be reset to ensure consistency across all groups.
           episode_id is reset to 0.

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
                        f"{traj_group_id}_{'w_memory' if self._should_search_trajectory_memory() else 'wo_memory'}"
                    )
                else:
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
        seed = (self.group_seed + self.episode_id) % (2**32)

        with self.thread_lock, self.env_step_limiter:
            # `observation` describes the current game-state prompt;
            # `info["suffix"]` carries the current environment-specific state string.
            observation, info = self.env.reset(seed=seed)
            if observation is None:
                return None
        self.rollout_cache.history.append(
            {
                "observation": observation,
                "actions_left": self.env_config.max_steps - self.rollout_cache.step,
                "messages": None,  # agent input messages
                **info,
            }
        )

        self.reset_memory_state()
        return self.rollout_cache

    def step(self, llm_output: DataProto):
        responses = self.tokenizer.batch_decode(llm_output.batch["responses"], skip_special_tokens=True)

        with self.thread_lock, self.env_step_limiter:
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
        self.rollout_cache.history[-1]["llm_response"] = responses[0]
        if info is not None:
            self.rollout_cache.history[-1].update(info)

        self.rollout_cache.history.append(
            {
                "observation": observation,
                "actions_left": self.env_config.max_steps - self.rollout_cache.step,
                "messages": None,
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
        lm_input = self.format_messages(rollout_cache)
        input_ids = lm_input.batch["input_ids"]

        if input_ids.shape[1] >= self.pipeline_config.sequence_length:
            self.logger.warning(
                f"sequence_length = {self.pipeline_config.sequence_length} input_ids length = {input_ids.shape[1]},"
                f"maybe you should increase the response_length"
            )
            return DataProto(meta_info={"stop_reason": GenerateStopReason.MAX_LENGTH})

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
        content = self.rollout_cache.history[-1]
        content["response_ids"] = response_ids
        content["messages"].append(
            {
                "role": "assistant",
                "content": self.tokenizer.decode(response_ids, skip_special_tokens=True),
            }
        )
        lm_output.meta_info["stop_reason"] = GenerateStopReason.FINISH
        return lm_output

    def format_messages(self, history: RolloutCache) -> DataProto:
        content = self.rollout_cache.history[-1]

        messages = []
        user_content = ""
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

            # Calculate system message token length for question mask
            system_tokens = custom_apply_chat_template(
                messages=messages,
                tokenizer=self.tokenizer,
                add_generation_prompt=False,
            )
            system_token_length = len(system_tokens)

            if "env_instruction" in history.history[0]:
                user_content = f"{history.history[0]['env_instruction']}\n"

        if len(self.rollout_cache.history) > 1 and self.rollout_cache.history[-2].get("use_tool", False):
            messages.append({"role": "tool", "content": content["observation"]})
        else:
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
            user_content += self.agent_template.format(**render_dict)
            messages.append({"role": "user", "content": user_content})

        prompt_ids = custom_apply_chat_template(
            messages=messages,
            tokenizer=self.tokenizer,
            add_generation_prompt=True,
        )

        # Store system token length for question mask computation
        content["system_token_length"] = system_token_length
        history_token_ids = []
        for items in self.rollout_cache.history[:-1]:
            history_token_ids.extend(items["prompt_ids"])
            history_token_ids.extend(items["response_ids"])
        if len(history_token_ids):
            end_token_ids = compute_conversation_end_token_id(self.tokenizer)
            if len(end_token_ids) > 0 and history_token_ids[-1] != end_token_ids[0]:
                prompt_ids = end_token_ids + prompt_ids
        input_ids = history_token_ids + prompt_ids

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
        content["prompt_ids"] = prompt_ids
        content["messages"] = messages
        return lm_input

    def formulate_rollouts(self, rollout_cache: RolloutCache, log_stats: dict = None):
        """ """
        if "observation" in rollout_cache.history[-1]:
            rollout_cache.history.pop(-1)
        history = rollout_cache.history[:-1]
        last_cache = copy.deepcopy(rollout_cache.history[-1])
        last_cache.pop("reward", None)
        history.append(last_cache)

        scores = [i["reward"] for i in self.rollout_cache.history]
        episode_score = sum(scores)

        token_ids = []
        prompt_masks = []
        response_masks = []
        question_masks = []

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

        # Also build stripped version without memory if memory was used
        # This is used for training on successful memory trajectories without the memory tokens
        # Only create stripped tensors if memory is used AND extraction is enabled
        should_create_stripped = (
            self._should_search_trajectory_memory()
            and getattr(self.pipeline_config, "extract_successful_memory", False)
        )
        
        if should_create_stripped:
            stripped_token_ids = []
            stripped_prompt_masks = []
            stripped_response_masks = []
            
            for idx, items in enumerate(self.rollout_cache.history):
                # Rebuild messages without memory
                original_messages = items.get("messages", [])
                if idx == 0 and original_messages:
                    # First turn: replace system message with base template (no memory)
                    stripped_messages = []
                    for msg in original_messages:
                        if msg.get("role") == "system":
                            stripped_messages.append({"role": "system", "content": self.agent_system_template})
                        else:
                            stripped_messages.append(msg)
                    
                    # Re-tokenize stripped messages
                    stripped_prompt_ids = custom_apply_chat_template(
                        messages=stripped_messages,
                        tokenizer=self.tokenizer,
                        add_generation_prompt=True,
                    )
                    
                    # Calculate system token length for base template (no memory)
                    base_system_tokens = custom_apply_chat_template(
                        messages=[{"role": "system", "content": self.agent_system_template}],
                        tokenizer=self.tokenizer,
                        add_generation_prompt=False,
                    )
                    base_system_token_length = len(base_system_tokens)
                else:
                    # Subsequent turns: use same as original (no memory in these turns anyway)
                    stripped_prompt_ids = items["prompt_ids"]
                    base_system_token_length = 0
                
                response_ids = items["response_ids"]
                stripped_prompt_ids_len = len(stripped_prompt_ids)
                response_ids_len = len(response_ids)
                
                stripped_token_ids.extend(stripped_prompt_ids)
                stripped_token_ids.extend(response_ids)
                stripped_prompt_masks.extend([1] * stripped_prompt_ids_len + [0] * response_ids_len)
                stripped_response_masks.extend([0] * stripped_prompt_ids_len + [1] * response_ids_len)
            
            # Create stripped tensors
            stripped_input_ids = torch.tensor(stripped_token_ids, dtype=torch.long).unsqueeze(0)
            stripped_attention_mask = torch.tensor([1] * len(stripped_token_ids), dtype=torch.long).unsqueeze(0)
            stripped_response_mask = torch.tensor(stripped_response_masks, dtype=torch.bool).unsqueeze(0)
            
            stripped_first_response_idx = stripped_response_masks.index(1)
            stripped_prompt_masks_list = [1] * stripped_first_response_idx + [0] * (len(stripped_token_ids) - stripped_first_response_idx)
            stripped_prompt_mask = torch.tensor(stripped_prompt_masks_list, dtype=torch.bool).unsqueeze(0)
            stripped_position_ids = stripped_attention_mask.cumsum(dim=-1)
        else:
            # When stripped tensors are not needed, create them as copies of original tensors
            # This ensures all batches have consistent keys for concatenation
            stripped_input_ids = input_ids.clone()
            stripped_attention_mask = attention_mask.clone()
            stripped_prompt_mask = prompt_mask.clone()
            stripped_response_mask = response_mask.clone()
            stripped_position_ids = position_ids.clone()

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

        # Pad stripped tensors if they exist
        if stripped_input_ids is not None:
            stripped_input_ids = pad_to_length(
                stripped_input_ids,
                length=self.pipeline_config.sequence_length,
                pad_value=self.tokenizer.pad_token_id,
            )
            stripped_attention_mask = pad_to_length(
                stripped_attention_mask,
                length=self.pipeline_config.sequence_length,
                pad_value=0,
            )
            stripped_position_ids = pad_to_length(
                stripped_position_ids,
                length=self.pipeline_config.sequence_length,
                pad_value=0,
            )
            stripped_response_mask = pad_to_length(
                stripped_response_mask,
                length=self.pipeline_config.sequence_length,
                pad_value=0,
            )
            stripped_prompt_mask = pad_to_length(
                stripped_prompt_mask,
                length=self.pipeline_config.sequence_length,
                pad_value=0,
            )

        # Build batch update dict
        batch_update = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
            "prompt_mask": prompt_mask,
            "question_mask": question_mask,
            "scores": score_tensor,
        }
        
        # Add stripped tensors if they exist (stripped_question_mask not included per user request)
        if stripped_input_ids is not None:
            batch_update.update({
                "stripped_input_ids": stripped_input_ids,
                "stripped_attention_mask": stripped_attention_mask,
                "stripped_position_ids": stripped_position_ids,
                "stripped_response_mask": stripped_response_mask,
                "stripped_prompt_mask": stripped_prompt_mask,
            })
        
        lm_input.batch.update(batch_update)
        # Build traj_messages for potential memory extraction
        traj_messages = [item for items in self.rollout_cache.history for item in items["messages"]]

        lm_input.non_tensor_batch.update(
            {
                "env_ids": np.array([self.rollout_cache.env_id], dtype=object),
                "group_ids": np.array([self.rollout_cache.group_id], dtype=object),
                "tags": np.array([self.rollout_cache.tag], dtype=object),
                "frames": np.array([self.rollout_cache.frames], dtype=object),
                "step_scores": np.array([scores], dtype=object),
                "episode_scores": np.array([episode_score], dtype=object),
                "traj_messages": np.array([traj_messages], dtype=object),
                "base_system_template": np.array([self.agent_system_template], dtype=object),
            }
        )

        self.add_memory_metadata_to_rollout(lm_input, rollout_cache, self.episode_id, self.group_seed)

        metrics_agg_mode = self.rollout_cache.history[-1].get("metrics_agg_mode", {})
        history_metrics = [item.get("metrics", {}) for item in self.rollout_cache.history]
        env_metric = aggregate_metrics(
            history_metrics=history_metrics,
            metrics_agg_mode=metrics_agg_mode,
        )
        env_metric["num_actions"] = rollout_cache.step

        env_metric = {f"env/{rollout_cache.tag}/{k}": v for k, v in env_metric.items()}
        env_metric["env/response_length"] = response_length
        env_metric = self.collect_memory_metrics(log_stats, env_metric)
        lm_input.meta_info = {"metrics": env_metric}

        base_dir = self.pipeline_config.base_dir
        if base_dir:
            traj_messages = [item for items in self.rollout_cache.history for item in items["messages"]]
            save_data = {
                "env_id": self.rollout_cache.env_id,
                "group_id": self.rollout_cache.group_id,
                "tag": self.rollout_cache.tag,
                "traj_messages": traj_messages,
                "episode_score": episode_score,
                "step_count": self.rollout_cache.step,
                "metrics": env_metric,
            }
            log_path = os.path.join(
                base_dir,
                "rollouts",
                f'{time.strftime("%Y%m%d_%H%M%S", time.localtime())}-{self.rollout_cache.tag}_{self.rollout_cache.group_id}_{self.rollout_cache.env_id}_re{episode_score}_step{self.rollout_cache.step}.json',
            )
            write_data_json(save_data, log_path)
        return lm_input
