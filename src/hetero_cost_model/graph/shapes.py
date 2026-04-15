"""Tensor-shape utilities shared across graph modules."""
from typing import Sequence, Tuple


Shape = Tuple[int, ...]


def numel(shape: Sequence[int]) -> int:
    """Total element count of a static shape; 0 if empty or dynamic."""
    if not shape:
        return 0
    n = 1
    for d in shape:
        if isinstance(d, int) and d > 0:
            n *= d
    return n


__all__ = ["Shape", "numel"]
