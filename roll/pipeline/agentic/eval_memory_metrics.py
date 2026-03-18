from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

from roll.utils.bool_utils import to_bool_numpy
from roll.utils.logging import get_logger

logger = get_logger()


def _build_score_stats(prefix: str, scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {
            f"{prefix}/mean": 0.0,
            f"{prefix}/max": 0.0,
            f"{prefix}/min": 0.0,
            f"{prefix}/count": 0.0,
        }

    arr = np.asarray(scores, dtype=np.float32)
    return {
        f"{prefix}/mean": float(np.mean(arr)),
        f"{prefix}/max": float(np.max(arr)),
        f"{prefix}/min": float(np.min(arr)),
        f"{prefix}/count": float(arr.size),
    }


def _extract_memory_flag(traj_batch: Any) -> Tuple[bool, str]:
    evolve_with_memory = traj_batch.non_tensor_batch.get("evolve_with_memory", None)
    if evolve_with_memory is not None and len(evolve_with_memory) > 0:
        try:
            return bool(to_bool_numpy(evolve_with_memory[:1], strict=False)[0]), "evolve_with_memory"
        except Exception:
            pass

    group_memory_key = traj_batch.non_tensor_batch.get("traj_group_id_w_or_wo_memory", None)
    if group_memory_key is not None and len(group_memory_key) > 0:
        group_key = str(group_memory_key[0]).lower()
        if "_wo_memory" in group_key:
            return False, "traj_group_id_w_or_wo_memory"
        if "_w_memory" in group_key:
            return True, "traj_group_id_w_or_wo_memory"

    return False, "default_false"


def compute_eval_memory_split_metrics(eval_batch: Any) -> Dict[str, float]:
    """
    Compute eval score statistics split by memory usage.

    Returns metric keys:
      - total_score/{mean,max,min,count}
      - w_memory_score/{mean,max,min,count}
      - wo_memory_score/{mean,max,min,count}
    """
    batch_group_by_traj = eval_batch.group_by(keys="traj_id")
    total_scores: List[float] = []
    with_memory_scores: List[float] = []
    without_memory_scores: List[float] = []
    source_counter: Counter[str] = Counter()

    for _, traj_batch in batch_group_by_traj.items():
        episode_scores = traj_batch.non_tensor_batch.get("episode_scores", None)
        if episode_scores is None or len(episode_scores) == 0:
            continue

        score = float(episode_scores[0])
        total_scores.append(score)

        use_memory, source = _extract_memory_flag(traj_batch)
        source_counter[source] += 1
        if use_memory:
            with_memory_scores.append(score)
        else:
            without_memory_scores.append(score)

    if not total_scores:
        logger.warning("No trajectory scores found while computing eval memory split metrics.")
    if not with_memory_scores:
        logger.warning("No with-memory trajectories found in eval batch.")
    if not without_memory_scores:
        logger.warning("No without-memory trajectories found in eval batch.")
    if source_counter.get("default_false", 0) > 0:
        logger.warning(
            "Defaulted %s trajectories to without-memory because no memory flag metadata was found.",
            source_counter["default_false"],
        )

    metrics: Dict[str, float] = {}
    metrics.update(_build_score_stats("total_score", total_scores))
    metrics.update(_build_score_stats("w_memory_score", with_memory_scores))
    metrics.update(_build_score_stats("wo_memory_score", without_memory_scores))

    return metrics
