from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.env_manager.base_env_manager import RolloutCache


@dataclass
class MemorySearchResult:
    """Container for memory search results."""

    message: str
    uids: List[str]
    triggered_interactions: Optional[List[DataProto]] = None


@dataclass
class MemoryUpdateData:
    """Container for data needed to update memory."""

    memory_dict: Dict
    mode: str


@dataclass
class TurnMemoryData:
    """Data extracted from a turn for episodic memory."""

    action: str
    action_is_valid: bool
    action_is_effective: bool
    action_result: str
    memory_query: str
    memory_uids: List[str]


@dataclass
class TrajectoryMemoryData:
    """Data for trajectory-based (procedural) memory."""

    query: str
    task_goal: str
    messages: List[Dict]
    episode_score: float
    trajectory_value: str
    outcome: bool
    first_turn_uids: Optional[List[str]] = None


class MemoryHooksMixin:
    """
    Mixin providing hook methods for memory integration.

    These hooks define the lifecycle of memory operations but don't implement
    environment-specific logic. Subclasses should override abstract methods
    to provide environment-specific behavior.
    """

    def hook_on_episode_start(self, rollout_cache: RolloutCache) -> None:
        """
        Called at the start of each episode.

        Performs trajectory-based memory search if configured.

        Args:
            rollout_cache: The current rollout cache
        """
        pass

    def hook_on_turn_start(self, rollout_cache: RolloutCache) -> None:
        """
        Called at the start of each turn before making a decision.

        Performs turn-based (episodic) memory search if configured.

        Args:
            rollout_cache: The current rollout cache
        """
        pass

    def hook_on_turn_end(self, rollout_cache: RolloutCache) -> None:
        """
        Called at the end of each turn after stepping the environment.

        Performs turn-based (episodic) memory update if configured.

        Args:
            rollout_cache: The current rollout cache
        """
        pass

    def hook_on_episode_end(self, rollout_cache: RolloutCache) -> None:
        """
        Called at the end of each episode.

        Performs trajectory-based (procedural) memory update if configured.

        Args:
            rollout_cache: The current rollout cache
        """
        pass

    def hook_collect_trajectory_messages(self, rollout_cache: RolloutCache) -> None:
        """
        Called after each turn to collect messages for trajectory memory.

        Args:
            rollout_cache: The current rollout cache
        """
        return


def extract_memory_metrics(log_stats: Dict) -> Dict[str, float]:
    """
    Extract memory-related timing metrics from log statistics.

    Args:
        log_stats: Dictionary containing timing statistics

    Returns:
        Dictionary of memory metrics suitable for logging
    """
    import numpy as np

    metrics = {}

    if "memory_search_time" in log_stats and log_stats["memory_search_time"]:
        metrics["memory_search_time_total"] = float(np.sum(log_stats["memory_search_time"]))
        metrics["memory_search_time_avg"] = float(np.mean(log_stats["memory_search_time"]))
        metrics["memory_search_count"] = float(len(log_stats["memory_search_time"]))

    if "memory_actor_critic_time" in log_stats and log_stats["memory_actor_critic_time"]:
        metrics["memory_actor_critic_time_total"] = float(np.sum(log_stats["memory_actor_critic_time"]))
        metrics["memory_actor_critic_time_avg"] = float(np.mean(log_stats["memory_actor_critic_time"]))
        metrics["memory_actor_critic_count"] = float(len(log_stats["memory_actor_critic_time"]))

    if "memory_update_time" in log_stats and log_stats["memory_update_time"]:
        metrics["memory_update_time_total"] = float(np.sum(log_stats["memory_update_time"]))
        metrics["memory_update_time_avg"] = float(np.mean(log_stats["memory_update_time"]))
        metrics["memory_update_count"] = float(len(log_stats["memory_update_time"]))

    if "memory_update_trajectory_time" in log_stats and log_stats["memory_update_trajectory_time"]:
        metrics["memory_update_trajectory_time"] = float(np.sum(log_stats["memory_update_trajectory_time"]))

    return metrics


def should_use_memory(memory_manager, env_config: Dict) -> bool:
    """
    Check if memory should be used based on configuration.

    Args:
        memory_manager: The memory manager instance (or None)
        env_config: Environment configuration

    Returns:
        True if memory should be used
    """
    return memory_manager is not None and env_config.get("evolve_with_memory", True)
