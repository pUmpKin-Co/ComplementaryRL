from typing import Dict, Optional

from roll.pipeline.agentic.env_manager.base_env_manager import RolloutCache


class MemoryBuildersMixin:
    """
    Mixin providing default implementations for memory query and value building.

    Subclasses can override these methods to customize memory content for their
    specific environment needs.
    """

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
        observation = last_history.get("observation", "")

        if task_goal is not None:
            return f"goal: {task_goal}, observation: {observation}"
        elif "goal" in last_history:
            goal = last_history["goal"]
            return f"goal: {goal}, observation: {observation}"
        else:
            return observation

    def build_turn_memory_value(
        self, action: str, action_is_valid: bool, action_is_effective: bool, action_result: str
    ) -> str:
        """
        Build memory value for turn-based (episodic) memory.

        This captures a single action and its outcome.

        Args:
            action: The action taken
            action_is_valid: Whether the action was valid
            action_is_effective: Whether the action was effective
            action_result: The result/feedback from the action

        Returns:
            Memory value string
        """
        value = f"{action}\n"
        value += f"Action info: [action_is_valid: {action_is_valid}, action_is_effective: {action_is_effective}]\n"
        value += f"Feedback: {action_result}"
        return value

    def build_trajectory_memory_value(self, rollout_cache: RolloutCache, task_goal: str) -> str:
        """
        Build memory value for trajectory-based (procedural) memory.

        This captures the entire trajectory as a procedure/solution.

        Args:
            rollout_cache: The completed trajectory
            task_goal: The task goal for this trajectory

        Returns:
            Memory value string containing the full procedure
        """
        actions = []

        # Build step-by-step summary
        for i, turn in enumerate(rollout_cache.history[:-1]):  # Exclude last observation-only entry
            action = turn.get("last_action_content", turn.get("llm_response", ""))

            # Extract validity/effectiveness if available
            action_is_valid = True
            action_is_effective = True
            if hasattr(self.env, "action_is_valid_lst") and i < len(self.env.action_is_valid_lst):
                action_is_valid = self.env.action_is_valid_lst[i]
            if hasattr(self.env, "action_is_effective_lst") and i < len(self.env.action_is_effective_lst):
                action_is_effective = self.env.action_is_effective_lst[i]

            reward = turn.get("reward", 0)

            # Build step summary
            step_summary = f"Step {i+1}: {action}"
            if not action_is_valid:
                step_summary += " [INVALID]"
            elif not action_is_effective:
                step_summary += " [INEFFECTIVE]"
            if reward != 0:
                step_summary += f" (reward: {reward})"

            actions.append(step_summary)

        # Calculate episode score and status
        episode_score = sum([t.get("reward", 0) for t in rollout_cache.history[:-1]])
        success_status = "SUCCESS" if episode_score > 0 else "FAILED"

        # Build full trajectory summary
        trajectory_summary = f"Task: {task_goal}\n"
        trajectory_summary += f"Result: {success_status} (total reward: {episode_score})\n"
        trajectory_summary += f"Procedure ({len(actions)} steps):\n"
        trajectory_summary += "\n".join(actions)

        return trajectory_summary

    def extract_task_goal(self, rollout_cache: RolloutCache) -> Optional[str]:
        """
        Extract the task goal from the rollout cache.

        Override this to customize how task goals are extracted for your environment.

        Args:
            rollout_cache: The current rollout cache

        Returns:
            Task goal string, or None if not available
        """
        if not rollout_cache.history:
            return None

        last_history = rollout_cache.history[-1]

        # Try common patterns
        if "suffix" in last_history and isinstance(last_history["suffix"], dict):
            if "goal" in last_history["suffix"]:
                return last_history["suffix"]["goal"]

        if "goal" in last_history:
            return last_history["goal"]

        return None

    def extract_turn_action_data(self, rollout_cache: RolloutCache) -> Dict:
        """
        Extract action data from the last completed turn.

        Override this to customize action data extraction for your environment.

        Args:
            rollout_cache: The current rollout cache (should have at least 2 entries)

        Returns:
            Dictionary containing action data with keys:
            - action: The action taken
            - action_is_valid: Whether action was valid
            - action_is_effective: Whether action was effective
            - action_result: Result/observation from the action
        """
        if len(rollout_cache.history) < 2:
            return {"action": "", "action_is_valid": True, "action_is_effective": True, "action_result": ""}

        # Extract action from previous turn
        action = rollout_cache.history[-2].get(
            "last_action_content", rollout_cache.history[-2].get("llm_response", "")
        )

        # Extract validity/effectiveness from environment
        action_is_valid = True
        action_is_effective = True
        if hasattr(self.env, "action_is_valid_lst") and len(self.env.action_is_valid_lst) > 0:
            action_is_valid = self.env.action_is_valid_lst[-1]
        if hasattr(self.env, "action_is_effective_lst") and len(self.env.action_is_effective_lst) > 0:
            action_is_effective = self.env.action_is_effective_lst[-1]

        # Extract result from current observation
        action_result = rollout_cache.history[-1].get("observation", "")

        return {
            "action": action,
            "action_is_valid": action_is_valid,
            "action_is_effective": action_is_effective,
            "action_result": action_result,
        }

    def inject_memory_into_prompt(
        self, rollout_cache: RolloutCache, messages: list, is_first_turn: bool, memory_message: Optional[str] = None
    ) -> list:
        """
        Inject memory into the prompt messages.

        Override this to customize how memory is injected into prompts for your environment.

        Args:
            rollout_cache: The current rollout cache
            messages: The current messages list
            is_first_turn: Whether this is the first turn
            memory_message: The memory message to inject (or None)

        Returns:
            Modified messages list with memory injected
        """
        # Default: no modification (memory should be added by make_decision)
        return messages
