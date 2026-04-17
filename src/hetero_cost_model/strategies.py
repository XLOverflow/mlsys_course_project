"""Inference configuration: batch size and sequence length.

A sample is (model graph G, inference config s, hardware h) → latency.
The "strategy" is now the serving configuration, not CPU/GPU op placement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


CONFIG_FEATURE_DIM: int = 2  # [batch_size_norm, seq_len_norm]

_MAX_BATCH: float = 16.0
_MAX_SEQ: float = 256.0


@dataclass(frozen=True)
class InferenceConfig:
    """A single (batch_size, seq_len) serving configuration."""

    batch_size: int = 1
    seq_len: int = 128

    def to_vector(self) -> List[float]:
        return [self.batch_size / _MAX_BATCH, self.seq_len / _MAX_SEQ]


def config_grid(
    batch_sizes: Sequence[int] = (1, 4, 8),
    seq_lens: Sequence[int] = (64, 128, 256),
) -> List[InferenceConfig]:
    """Return the Cartesian product of batch sizes and sequence lengths."""
    return [
        InferenceConfig(b, s)
        for b in batch_sizes
        for s in seq_lens
    ]


def config_pairs(configs: List[InferenceConfig]) -> List[Tuple[InferenceConfig, InferenceConfig]]:
    """All ordered pairs of distinct configs — used to build ranking loss pairs."""
    return [
        (a, b)
        for i, a in enumerate(configs)
        for b in configs[i + 1:]
    ]


__all__ = [
    "InferenceConfig",
    "CONFIG_FEATURE_DIM",
    "config_grid",
    "config_pairs",
]
