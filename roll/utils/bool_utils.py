from typing import Iterable, Optional, Sequence

import numpy as np
import torch

_TRUE_STRINGS = {"true", "1", "yes", "y", "t"}
_FALSE_STRINGS = {"false", "0", "no", "n", "f", "none", "null", ""}


def parse_bool_like(value: object, *, strict: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, (float, np.floating)):
        return bool(float(value))
    if isinstance(value, str):
        s = value.strip().lower()
        if s in _TRUE_STRINGS:
            return True
        if s in _FALSE_STRINGS:
            return False
        if strict:
            raise ValueError(f"Unsupported bool-like string: {value!r}")
        return False

    if strict:
        raise ValueError(f"Unsupported bool-like value type: {type(value)} ({value!r})")
    return bool(value)


def to_bool_numpy(values: Sequence[object] | np.ndarray, *, strict: bool = False) -> np.ndarray:
    """Convert a sequence (or numpy array) of bool-like objects into a `np.bool_` array."""
    if isinstance(values, np.ndarray):
        seq: Iterable[object] = values.tolist()
    else:
        seq = values
    return np.asarray([parse_bool_like(v, strict=strict) for v in seq], dtype=np.bool_)


def to_bool_tensor(
    values: Sequence[object] | np.ndarray,
    *,
    strict: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert bool-like objects to a torch.bool tensor on the specified device."""
    mask_np = to_bool_numpy(values, strict=strict)
    return torch.as_tensor(mask_np, dtype=torch.bool, device=device)
