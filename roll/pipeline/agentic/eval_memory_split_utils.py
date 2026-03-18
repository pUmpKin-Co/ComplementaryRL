import random


def build_deterministic_eval_memory_env_ids(total_envs: int, with_memory_ratio: float, seed: int) -> set[int]:
    """Build a deterministic env-id set that should evolve with memory during eval."""
    if total_envs < 0:
        raise ValueError(f"total_envs must be non-negative, got {total_envs}")
    if not 0.0 <= float(with_memory_ratio) <= 1.0:
        raise ValueError(f"with_memory_ratio must be in [0, 1], got {with_memory_ratio}")
    if total_envs == 0:
        return set()

    target_with_memory = int(round(total_envs * float(with_memory_ratio)))
    target_with_memory = max(0, min(total_envs, target_with_memory))

    env_ids = list(range(total_envs))
    rng = random.Random(int(seed))
    rng.shuffle(env_ids)
    return set(env_ids[:target_with_memory])
