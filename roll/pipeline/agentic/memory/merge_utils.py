from typing import Dict, Iterable, List, Optional, Tuple

GLOBAL_MERGE_SCOPE_ID = "__global__"
GLOBAL_MERGING_SIZE_KEYS = ("__global__", "_global_", "global", "shared", "all")


def _to_positive_int(value: object) -> Optional[int]:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def resolve_tabular_merging_threshold(merging_size: Optional[Dict[str, int]]) -> Optional[int]:
    """
    Resolve a global merge threshold for shared tabular memory.

    Priority:
    1. Explicit global keys (e.g. "global", "__global__")
    2. Single-entry dict fallback
    3. Otherwise return None (ambiguous configuration)
    """
    if not merging_size:
        return None

    for key in GLOBAL_MERGING_SIZE_KEYS:
        if key in merging_size:
            return _to_positive_int(merging_size[key])

    if len(merging_size) == 1:
        return _to_positive_int(next(iter(merging_size.values())))

    return None


def build_multitask_merge_targets(
    task_to_uid: Dict[str, Iterable[str]],
    task_thresholds: Optional[Dict[str, int]] = None,
) -> List[Tuple[str, List[str]]]:
    """
    Build merge scopes for multi-task memory.

    - If `task_thresholds` is None, merge each task with >1 entries.
    - If set, merge task only when threshold exists and `len(uids) >= threshold`.
    """
    targets: List[Tuple[str, List[str]]] = []
    for task_id, uids_iter in task_to_uid.items():
        uids = list(uids_iter)
        if len(uids) <= 1:
            continue
        if task_thresholds is None:
            targets.append((task_id, uids))
            continue
        if task_id not in task_thresholds:
            continue
        threshold = task_thresholds[task_id]
        try:
            threshold = int(threshold)
        except (TypeError, ValueError):
            continue
        if len(uids) >= threshold:
            targets.append((task_id, uids))
    return targets


def build_tabular_merge_targets(
    uids: Iterable[str],
    threshold: Optional[int] = None,
    scope_id: str = GLOBAL_MERGE_SCOPE_ID,
) -> List[Tuple[str, List[str]]]:
    """
    Build merge scope for shared tabular memory.

    - If `threshold` is None, merge when >1 entries.
    - If set, merge when `len(uids) >= threshold` and >1 entries.
    """
    uid_list = list(uids)
    if len(uid_list) <= 1:
        return []

    if threshold is None:
        return [(scope_id, uid_list)]

    threshold = _to_positive_int(threshold)
    if threshold is None:
        return []

    if len(uid_list) >= threshold:
        return [(scope_id, uid_list)]
    return []
