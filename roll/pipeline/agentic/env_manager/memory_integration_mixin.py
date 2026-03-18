import copy
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch
from codetiming import Timer
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.env_manager.base_env_manager import RolloutCache
from roll.pipeline.agentic.env_manager.memory_builders import MemoryBuildersMixin
from roll.pipeline.agentic.env_manager.memory_hooks import MemoryHooksMixin, extract_memory_metrics
from roll.pipeline.agentic.env_manager.token_mask_utils import custom_apply_chat_template
from roll.pipeline.agentic.memory.memory_config import MemoryIntegrationStrategy
from roll.utils.logging import get_logger
from roll.utils.str_utils import contains_renderable_field

if TYPE_CHECKING:
    from roll.pipeline.agentic.memory.memory_manager import MemoryManager


class MemoryIntegrationMixin(MemoryBuildersMixin, MemoryHooksMixin):
    def initialize_memory_integration(
        self,
        memory_manager: Optional["MemoryManager"],
        env_config: Dict,
        mode: str = "train",
    ) -> None:
        self.memory_manager: Optional["MemoryManager"] = memory_manager
        self.memory_env_config = env_config
        self.memory_mode = mode
        self.env_name = self.memory_env_config["env_class"]
        if "env_type" in self.memory_env_config["config"]:
            self.env_name = f"{self.env_name}_{self.memory_env_config['config']['env_type']}"

        # Memory state variables
        self._first_turn_memory_message: Optional[str] = None
        self._first_turn_memory_uids: Optional[list] = None
        self._first_turn_memory_triggered_interactions: Optional[list] = None
        self._turn_based_triggered_interactions: Optional[list] = None

        # Determine integration strategy
        self._memory_strategy: Optional[MemoryIntegrationStrategy] = None
        self._should_evolve_with_memory: bool = False
        self.logger = get_logger()

        if self.memory_manager is not None:
            self._memory_strategy = self.memory_manager.memory_config.memory_integration_strategy
            self._should_evolve_with_memory = env_config.get("evolve_with_memory", True)
            self._memory_should_update = self.memory_manager.memory_config.memory_should_update
            self._enable_actor_critic: bool = self.memory_manager.memory_config.memory_model.enable_actor_critic

    def reset_memory_state(self) -> None:
        """
        Reset memory state for a new episode.

        Call this in your reset() method.
        """
        self._first_turn_memory_message = None
        self._first_turn_memory_uids = None
        self._first_turn_memory_triggered_interactions = None
        self._turn_based_triggered_interactions = None

    def hook_on_episode_start(self, rollout_cache: RolloutCache, log_stats: Optional[Dict] = None) -> None:
        """
        Perform trajectory-based memory search at episode start.

        Call this at the beginning of your episode loop (step == 0).

        Args:
            rollout_cache: The current rollout cache
            log_stats: Optional dict to collect timing statistics
        """
        if log_stats is None:
            log_stats = defaultdict(list)

        should_use = self._should_search_trajectory_memory()
        should_update = self._should_update_trajectory_memory()
        if not should_use:
            self.logger.debug(
                f"[MemoryDebug] env_id={getattr(self.memory_env_config, 'env_id', 'unknown')} "
                f"should_search_trajectory_memory=False: "
                f"evolve_with_memory={self._should_evolve_with_memory}, "
                f"memory_manager={self.memory_manager is not None}, "
                f"strategy={self._memory_strategy}"
            )

        if should_use and rollout_cache.step == 0:
            task_goal = self._extract_task_goal_from_history(rollout_cache) or ""
            memory_query = self._build_trajectory_memory_query(rollout_cache, task_goal)
            with Timer(name="memory-search-trajectory-init", logger=None) as memory_search_timer:
                memory_message, memory_uids, triggered_interactions = self.memory_manager.search(
                    memory_query, self.env_name
                )
            log_stats["memory_search_time"].append(memory_search_timer.last)

            memory_uids = self._normalize_memory_uids(memory_uids)

            if len(memory_uids) > 0 and self._enable_actor_critic:
                with Timer(name="memory-actor-critic", logger=None) as actor_critic_timer:
                    new_memory_message, critic_score = self.actor_critic(memory_message, rollout_cache)
                    if critic_score is None:
                        # If critic process failed, we skip the memory message
                        memory_message = ""
                        memory_uids = []
                        self._first_turn_memory_critic_score = []
                    else:
                        memory_message = new_memory_message
                        self._first_turn_memory_critic_score = [critic_score]
                log_stats["memory_actor_critic_time"].append(actor_critic_timer.last)
            else:
                self._first_turn_memory_critic_score = []

            print(
                f"[MemoryDebug] env_id={getattr(self.memory_env_config, 'env_id', 'unknown')} "
                f"searched memory: query='{memory_query[:50]}...', "
                f"found_uids={len(memory_uids)}, message_preview='{memory_message[:100]}...'"
            )

            # Cache results for first turn and episode end
            self._first_turn_memory_message = memory_message
            self._first_turn_memory_uids = memory_uids
            self._first_turn_memory_triggered_interactions = triggered_interactions

            if rollout_cache.history:
                rollout_cache.history[0]["memory_message"] = memory_message
                rollout_cache.history[0]["memory_uids"] = memory_uids

    def _extract_task_goal_from_history(self, rollout_cache: RolloutCache) -> Optional[str]:
        """
        Extract task goal robustly from any turn.
        """
        if not rollout_cache.history:
            return None

        for h in reversed(rollout_cache.history):
            if "suffix" in h and isinstance(h["suffix"], dict) and "goal" in h["suffix"]:
                return h["suffix"]["goal"]
            if "goal" in h:
                return h["goal"]
        return None

    def _build_trajectory_memory_query(self, rollout_cache: RolloutCache, task_goal: Optional[str]) -> str:
        """
        Build a trajectory query in a stable way (prefer episode-start observation).
        """
        if rollout_cache.history:
            return self.build_memory_query(rollout_cache.history[0], task_goal)
        return self.build_memory_query({"observation": ""}, task_goal)

    def _render_system_template_without_memory(self, template: str) -> str:
        """
        Render a system template that may contain `{memory_message}` placeholders,
        but force memory to be empty.
        """
        if not isinstance(template, str):
            return str(template)
        if contains_renderable_field(template, "memory_message") or "{{memory_message}}" in template:
            try:
                return template.replace("{{memory_message}}", "{memory_message}").format(memory_message="")
            except Exception:
                return template.replace("{memory_message}", "").replace("{{memory_message}}", "")
        return template

    def _build_full_trajectory_messages(
        self, rollout_cache: RolloutCache, strip_retrieved_memory_from_system: bool = True
    ) -> List[Dict]:
        """
        Build a full trajectory message list from rollout history.

        """
        full_messages: List[Dict] = []
        for turn in rollout_cache.history:
            turn_messages = turn.get("messages") or []
            if turn_messages:
                full_messages.extend([copy.deepcopy(m) for m in turn_messages])

        if strip_retrieved_memory_from_system and full_messages:
            injection_mode = self.get_trajectory_memory_injection_mode()
            if injection_mode == "system":
                base_system = self.agent_system_template
                base_system = self._render_system_template_without_memory(base_system)
                for m in full_messages:
                    if m.get("role") == "system":
                        m["content"] = base_system
                        break

        return full_messages

    def hook_on_turn_start(self, rollout_cache: RolloutCache, log_stats: Optional[Dict] = None) -> None:
        """
        Perform turn-based (episodic) memory search before decision.

        Call this before make_decision() in your episode loop.

        Args:
            rollout_cache: The current rollout cache
            log_stats: Optional dict to collect timing statistics
        """
        if log_stats is None:
            log_stats = defaultdict(list)

        # Only search if turn-based or both strategy
        if self._should_use_turn_memory():
            with Timer(name="memory-search", logger=None) as memory_search_timer:
                task_goal = self.extract_task_goal(rollout_cache)
                memory_query = self.build_memory_query(rollout_cache.history[-1], task_goal)

                # Perform search
                memory_message, memory_uids, _ = self.memory_manager.search(memory_query, self.env_name)
                memory_uids = self._normalize_memory_uids(memory_uids)

                # Store in history for prompt injection
                rollout_cache.history[-1]["memory_message"] = memory_message
                rollout_cache.history[-1]["memory_uids"] = memory_uids

            log_stats["memory_search_time"].append(memory_search_timer.last)

    def hook_on_turn_end(self, rollout_cache: RolloutCache, log_stats: Optional[Dict] = None) -> None:
        """
        Perform turn-based (episodic) memory update after step.

        Call this after step() in your episode loop.

        Args:
            rollout_cache: The current rollout cache
            log_stats: Optional dict to collect timing statistics
        """
        if log_stats is None:
            log_stats = defaultdict(list)

        # Collect triggered interactions from the turn
        if rollout_cache.history and len(rollout_cache.history) >= 2:
            if "triggered_interactions" in rollout_cache.history[-2]:
                interactions = rollout_cache.history[-2]["triggered_interactions"]
                if interactions:
                    if self._turn_based_triggered_interactions is None:
                        self._turn_based_triggered_interactions = []
                    self._turn_based_triggered_interactions.extend(interactions)

        # Only update if turn-based or both strategy
        if self._should_use_turn_memory() and self._memory_should_update:
            with Timer(name="memory-update", logger=None) as memory_update_timer:
                # Extract data from the turn
                action_data = self.extract_turn_action_data(rollout_cache)

                # Build memory query and value
                task_goal = self.extract_task_goal(rollout_cache)
                if len(rollout_cache.history) >= 2:
                    memory_query = self.build_memory_query(rollout_cache.history[-2], task_goal)
                    memory_uids = self._normalize_memory_uids(rollout_cache.history[-2].get("memory_uids", []))
                else:
                    memory_query = self.build_memory_query(rollout_cache.history[-1], task_goal)
                    memory_uids = []

                memory_value = self.build_turn_memory_value(
                    action=action_data["action"],
                    action_is_valid=action_data["action_is_valid"],
                    action_is_effective=action_data["action_is_effective"],
                    action_result=action_data["action_result"],
                )

                # Build memory dict
                memory_dict = {
                    "last_turn_uids": memory_uids,
                    self.memory_manager.memory_key_field: memory_query,
                    self.memory_manager.memory_value_field: memory_value,
                    "task": self.env_name,
                }

                self.memory_manager.update(memory_dict, self.memory_mode)

            log_stats["memory_update_time"].append(memory_update_timer.last)

    def hook_on_episode_end(self, rollout_cache: RolloutCache, log_stats: Optional[Dict] = None) -> None:
        """
        Perform trajectory-based (procedural) memory update at episode end.

        Call this when episode terminates.

        Args:
            rollout_cache: The completed rollout cache
            log_stats: Optional dict to collect timing statistics
        """
        if log_stats is None:
            log_stats = defaultdict(list)

        # Only update if trajectory-based or both strategy
        if self._should_update_trajectory_memory() and self._memory_should_update:
            with Timer(name="memory-update-trajectory", logger=None) as memory_update_timer:
                # Calculate episode score
                episode_score = sum([t.get("reward", 0) for t in rollout_cache.history[:-1]])

                task_goal = self._extract_task_goal_from_history(rollout_cache) or ""
                memory_query = self._build_trajectory_memory_query(rollout_cache, task_goal)

                # Build trajectory memory value
                trajectory_value = self.build_trajectory_memory_value(rollout_cache, task_goal)

                # Build memory dict using the full rollout history (includes terminal observation)
                full_traj_messages = self._build_full_trajectory_messages(
                    rollout_cache, strip_retrieved_memory_from_system=True
                )
                trajectory_memory = {
                    self.memory_manager.memory_key_field: memory_query,
                    self.memory_manager.memory_value_field: trajectory_value,
                    "reward": episode_score,
                    "messages": full_traj_messages,
                    "task_goal": task_goal,
                    "outcome": episode_score > 0,
                    "last_turn_uids": self._normalize_memory_uids(self._first_turn_memory_uids),
                    "task": self.env_name,
                }

                self.memory_manager.update(trajectory_memory, self.memory_mode)

            log_stats["memory_update_trajectory_time"].append(memory_update_timer.last)

            self._first_turn_memory_message = None

    def add_memory_metadata_to_rollout(
        self,
        rollout: "DataProto",
        rollout_cache: RolloutCache,
        episode_id: int,
        group_seed: int,
    ) -> None:
        """
        Add memory-related metadata to the rollout output.

        Call this in formulate_rollouts() before returning.

        Args:
            rollout: The DataProto to add metadata to
            rollout_cache: The rollout cache
            episode_id: The episode ID
            group_seed: The group seed
        """
        if self.memory_manager is None or not self.memory_manager.memory_config.memory_actor_enable_training:
            rollout.non_tensor_batch["evolve_with_memory"] = np.array([False], dtype=object)
            return

        # Add evolve_with_memory flag
        rollout.non_tensor_batch["evolve_with_memory"] = np.array(
            [self._should_search_trajectory_memory()], dtype=object
        )

        # Add triggered interactions if available
        if self._should_search_trajectory_memory():
            all_triggered_interactions = []
            if self._first_turn_memory_triggered_interactions:
                all_triggered_interactions.extend(self._first_turn_memory_triggered_interactions)

            if (
                getattr(self, "_first_turn_memory_critic_score", None) is not None
                and len(self._first_turn_memory_critic_score) > 0
            ):
                critic_score = self._first_turn_memory_critic_score[0]
            else:
                critic_score = None

            if self._turn_based_triggered_interactions:
                all_triggered_interactions.extend(self._turn_based_triggered_interactions)

            if len(all_triggered_interactions) > 0:
                traj_group_id = f"{rollout_cache.tag}_{rollout_cache.group_id}_{episode_id}_{group_seed}"

                # Get the base episode_score
                base_episode_scores = rollout.non_tensor_batch.get("episode_scores", np.array([0], dtype=object))

                for interaction in all_triggered_interactions:
                    # Check if this is a first turn memory interaction
                    is_first_turn_interaction = (
                        self._first_turn_memory_triggered_interactions is not None
                        and len(self._first_turn_memory_triggered_interactions) > 0
                        and interaction in self._first_turn_memory_triggered_interactions
                    )

                    # Adjust episode_score with critic_score for first turn memory interactions
                    if is_first_turn_interaction and critic_score is not None:
                        # Extract the base score (it's wrapped in np.array)
                        base_score = (
                            base_episode_scores[0]
                            if isinstance(base_episode_scores, np.ndarray)
                            else base_episode_scores
                        )

                        # Adjust score with critic_score
                        base_score = float(base_score)
                        critic_score = float(critic_score)
                        if critic_score < 1:
                            adjusted_score = critic_score
                        else:
                            adjusted_score = base_score + critic_score
                        interaction.non_tensor_batch["episode_score"] = np.array([adjusted_score], dtype=object)
                    else:
                        interaction.non_tensor_batch["episode_score"] = base_episode_scores

                    interaction.non_tensor_batch["group_id"] = np.array([rollout_cache.group_id], dtype=object)
                    interaction.non_tensor_batch["traj_group_id"] = np.array([traj_group_id], dtype=object)
                    interaction.non_tensor_batch["env_id"] = np.array([rollout_cache.env_id], dtype=object)

                rollout.non_tensor_batch["triggered_interactions"] = all_triggered_interactions

    def collect_memory_metrics(self, log_stats: Dict, env_metrics: Dict) -> Dict:
        """
        Add memory timing metrics to environment metrics.

        Call this in formulate_rollouts() when building metrics dict.

        Args:
            log_stats: Dictionary containing timing statistics
            env_metrics: Dictionary of environment metrics to update

        Returns:
            Updated env_metrics dictionary
        """
        if self.memory_manager is None:
            return env_metrics

        memory_metrics = extract_memory_metrics(log_stats)
        env_metrics.update(memory_metrics)

        return env_metrics

    def get_first_turn_memory_message(self) -> Optional[str]:
        """
        Get the memory message from the first turn (trajectory-based search).

        Use this in make_decision() for first turn prompt injection.

        Returns:
            Memory message string, or None if not available
        """
        return self._first_turn_memory_message

    def get_trajectory_memory_injection_mode(self) -> str:
        """
        Controls how retrieved trajectory memory is injected into the *actor* prompt.

        Values:
        - "system": append memory to the system message (previous default for most env managers)
        - "user": add memory as a separate user message before the task/observation
                 (still excluded from question mask because we count its tokens in system_token_length)
        - "none": do not inject retrieved memory into the actor prompt
        """
        mode = None
        if self.pipeline_config.memory_config is None or not hasattr(
            self.pipeline_config.memory_config, "trajectory_memory_injection"
        ):
            return "system"
        mode = str(self.pipeline_config.memory_config.trajectory_memory_injection).lower()
        return mode

    def _is_valid_memory_message(self, memory_message: Optional[str]) -> bool:
        if not self._should_evolve_with_memory:
            return False
        if memory_message is None:
            return False
        if not isinstance(memory_message, str):
            return False
        if memory_message.strip() == "":
            return False
        if memory_message.strip() == "[No Memory Available]":
            return False
        return True

    def _normalize_memory_uids(self, uids: Any) -> List[str]:
        """
        Normalize memory UID containers coming from distributed components into a plain `List[str]`.
        """
        if uids is None:
            return []

        if isinstance(uids, np.ndarray):
            try:
                uids = uids.tolist()
            except Exception:
                uids = [uids]

        if not isinstance(uids, list):
            if isinstance(uids, (tuple, set)):
                uids = list(uids)
            else:
                uids = [uids]

        flattened: List[str] = []
        for u in uids:
            if u is None:
                continue
            if isinstance(u, np.ndarray):
                try:
                    u = u.tolist()
                except Exception:
                    u = [u]
            if isinstance(u, list):
                for uu in u:
                    if uu is None:
                        continue
                    s = str(uu).strip()
                    if s:
                        flattened.append(s)
            else:
                s = str(u).strip()
                if s:
                    flattened.append(s)

        return flattened

    def format_retrieved_memory_for_prompt(self, memory_message: str) -> str:
        """
        Format a retrieved memory message for prompt injection.
        """
        template = getattr(self, "memory_template", None)
        if template:
            try:
                return template.format(memory_message=memory_message)
            except Exception:
                return memory_message
        return memory_message

    def get_first_turn_retrieved_memory_message(self) -> Optional[str]:
        """
        Best-effort retrieval of the first-turn trajectory memory message.
        Falls back to `env.memory_message` if present.
        """
        memory_message = self.get_first_turn_memory_message()
        if memory_message is None and hasattr(self, "env") and hasattr(self.env, "memory_message"):
            if self.env.memory_message is not None:
                memory_message = self.env.memory_message
        return memory_message

    def build_initial_task_prompt(self, system_message: str, first_obs: str) -> str:
        """
        Build the initial task prompt.
        """
        env_instruction = f"Environment Instruction: {system_message}"
        user_content = f"Observation: {first_obs}"
        return env_instruction + "\n" + user_content

    def _should_log_actor_critic_response(self) -> bool:
        """
        Check if the actor-critic response should be logged.
        """
        if getattr(self, "last_actor_critic_response_logging", None) is None:
            self.last_actor_critic_response_logging = time.time()
            return True
        if time.time() - self.last_actor_critic_response_logging > 60:
            self.last_actor_critic_response_logging = time.time()
            return True
        return False

    def actor_critic(self, memory_message: str, rollout_cache: RolloutCache) -> str:
        """
        Actor-critic the memory message.

        This method evaluates the retrieved memory message and decides whether to:
        - Reject: The memory is not useful (return empty string)
        - Accept: The memory is helpful as-is (return original memory)
        - Refine: The memory needs improvement (return refined memory)

        Args:
            memory_message: The retrieved memory message from the knowledge base
            rollout_cache: The current rollout cache containing task context

        Returns:
            The processed memory message (empty if rejected, original if accepted, refined if improved)
        """
        # Build context for the actor critic
        system_message = self.agent_system_template
        env_instruction = (
            rollout_cache.history[-1]["env_instruction"] if "env_instruction" in rollout_cache.history[-1] else ""
        )

        content = rollout_cache.history[-1]
        render_dict = {"observation": content["observation"]}
        if contains_renderable_field(self.agent_template, "turn_idx"):
            render_dict["turn_idx"] = self.rollout_cache.step + 1
        if contains_renderable_field(self.agent_template, "suffix"):
            render_dict["suffix"] = content.get("suffix", "")
        if contains_renderable_field(self.agent_template, "actions_left"):
            render_dict["actions_left"] = content["actions_left"]
        if contains_renderable_field(self.agent_template, "max_response_length"):
            render_dict["max_response_length"] = self.env_config["max_tokens_per_step"]
        user_content = env_instruction + self.agent_template.format(**render_dict)
        initial_task_prompt = self.build_initial_task_prompt(system_message, user_content)

        # Get historical performance
        initial_score = self.memory_manager.get_tag_performance(rollout_cache.tag)
        avg_score = initial_score.get("score_sum", 0) / initial_score.get("traj_count", 1)
        traj_count = initial_score.get("traj_count", 0)

        sys_prompt = self.pipeline_config.memory_config.actor_critic_sys_prompt
        user_prompt_template = self.pipeline_config.memory_config.actor_critic_user_prompt

        # Build user prompt with context
        user_prompt = user_prompt_template.format(
            initial_task_prompt=initial_task_prompt,
            memory_message=memory_message,
            avg_score=avg_score,
            traj_count=traj_count,
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt_ids = custom_apply_chat_template(
            messages=messages,
            tokenizer=self.tokenizer,
            add_generation_prompt=True,
        )

        attention_mask = torch.tensor([1] * len(prompt_ids), dtype=torch.long).unsqueeze(0)
        position_ids = attention_mask.cumsum(dim=-1) - 1
        input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)

        lm_input = DataProto()
        lm_input.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=1,
        )

        max_new_tokens = min(
            self.env_config["max_tokens_per_step"],
            self.worker_config.generating_args.max_new_tokens,
            self.pipeline_config.sequence_length - input_ids.shape[1],
        )
        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["max_new_tokens"] = min(max_new_tokens, self.pipeline_config.sequence_length)
        lm_input.meta_info["src_rank"] = self.env_config["env_id"]
        lm_output: DataProto = self.llm_proxy.generate(
            messages=messages,
            lm_input=lm_input,
            generation_config=generation_config,
        )
        response = lm_output.batch["responses"][0].tolist()
        response_text = self.tokenizer.decode(response, skip_special_tokens=True)
        new_memory_message, critic_score = self._parse_actor_critic_response(response_text, memory_message)
        print("[DEBUG] critic_score: ", critic_score)
        self._log_actor_critic_response(messages, response_text, critic_score, new_memory_message)
        return new_memory_message, critic_score

    def _log_actor_critic_response(
        self, messages: List[Dict], response_text: str, critic_score: int, new_memory_message: str
    ) -> None:
        if not self._should_log_actor_critic_response():
            return

        save_dir = self.pipeline_config.memory_config.memory_model.memory_model_interaction_output_dir
        if not save_dir:
            save_dir = self.pipeline_config.output_dir

        actor_critic_dir = os.path.join(save_dir, "actor_critic")
        os.makedirs(actor_critic_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"actor_critic_{timestamp}.json"
        filepath = os.path.join(actor_critic_dir, filename)

        actor_critic_data = {
            "timestamp": datetime.now().isoformat(),
            "messages": messages,
            "response_text": response_text,
            "critic_score": critic_score,
            "new_memory_message": new_memory_message,
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(actor_critic_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            pass

    def _parse_actor_critic_response(self, response: str, memory_message: str) -> [str, int]:
        """
        Parse the actor-critic LLM response and return the appropriate memory message.

        Args:
            response: The LLM response containing the decision in JSON format

        Returns:
            - Empty string if rejected
            - Original memory if accepted
            - Refined memory if refined
        """

        # Extract JSON from <answer> tags
        answer_pattern = r"<answer>(.*?)</answer>"
        matches = re.findall(answer_pattern, response, re.DOTALL)

        if not matches:
            self.logger.warning("No <answer> tags found in actor-critic response")
            return "", None

        json_str = matches[-1].strip()

        try:
            decision = json.loads(json_str)

            function_name = decision.get("function_name", "")
            parameters = decision.get("parameters", {})
            if not isinstance(parameters, dict):
                return "", None

            if function_name == "reject":
                self.logger.info("Actor-critic rejected the memory message")
                return "", -1
            elif function_name == "accept":
                self.logger.info("Actor-critic accepted the memory message")
                # Return the original memory (we need to track it somehow)
                # For now, return empty and let the caller handle it
                return memory_message, 1
            elif function_name == "refine":
                refined_knowledge = parameters.get("refined_knowledge", "")
                if refined_knowledge:
                    self.logger.info("Actor-critic refined the knowledge")
                    return refined_knowledge, 0.5
                else:
                    self.logger.warning("Refine operation missing refined_knowledge parameter")
                    return "", None
            else:
                self.logger.warning(f"Unknown function_name: {function_name}")
                return "", None

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse actor-critic JSON response: {e}")
            return "", None

    def _should_search_trajectory_memory(self) -> bool:
        """Check if trajectory-based memory should be used."""
        if (
            not self._should_evolve_with_memory
            or self.memory_manager is None
            or not self.memory_manager.could_begin_interaction()
        ):
            return False

        return self._memory_strategy in [
            MemoryIntegrationStrategy.trajectory_based,
            MemoryIntegrationStrategy.both,
        ]

    def _should_update_trajectory_memory(self) -> bool:
        """Check if trajectory-based memory should be updated."""
        if not self._should_evolve_with_memory or self.memory_manager is None:
            return False

        return self._memory_strategy in [
            MemoryIntegrationStrategy.trajectory_based,
            MemoryIntegrationStrategy.both,
        ]

    def _should_use_turn_memory(self) -> bool:
        """Check if turn-based memory should be used."""
        if not self._should_evolve_with_memory or self.memory_manager is None:
            return False

        return self._memory_strategy in [
            MemoryIntegrationStrategy.turn_based,
            MemoryIntegrationStrategy.both,
        ]
