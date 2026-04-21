from typing import Optional


def should_flush_pending_updates_before_rollout(has_pending_memory_train: bool) -> bool:
    return not has_pending_memory_train


def fence_memory_updates_before_async_train(memory_manager, flush_timeout: Optional[float], global_step: int) -> int:
    if memory_manager is None:
        return 0

    flushed_count = memory_manager.flush_pending_updates(timeout=flush_timeout)
    memory_manager.suspend(global_step)
    return flushed_count
