"""Inference configuration: batch size and sequence length.

A sample is (model graph G, inference config s, hardware h) → latency.
The "strategy" is now the serving configuration, not CPU/GPU op placement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


CONFIG_FEATURE_DIM: int = 2  # [batch_size_norm, seq_len_norm]

# Normalization upper bounds — must cover the largest value in the profiling
# config grid (see data_collection_plan.md §2.3). Keep components in [0, 1]
# so they share scale with the normalized hardware vector ``h``.
_MAX_BATCH: float = 16.0
_MAX_SEQ: float = 1024.0


@dataclass(frozen=True)
class InferenceConfig:
    """A single (batch_size, seq_len) serving configuration."""

    batch_size: int = 1
    seq_len: int = 128

    def to_vector(self) -> List[float]:
        return [self.batch_size / _MAX_BATCH, self.seq_len / _MAX_SEQ]


def config_grid(
    batch_sizes: Sequence[int] = (1, 2, 3, 4, 6, 8, 12, 16),
    seq_lens: Sequence[int] = (32, 64, 128, 256, 512, 1024),
) -> List[InferenceConfig]:
    """Return the Cartesian product of batch sizes and sequence lengths.

    Defaults match ``data_collection_plan.md §2.3``: 8 batch × 6 seq = 48
    configs per (model, GPU). The non-power-of-2 batches (3, 6, 12) give
    the model non-trivial interpolation targets between the log2 sweep,
    and seq extends to 32 (short-prompt interactive case) and 1024
    (long-context case, OOM-prone on 16-24 GB cards).

    Smaller grids can be passed explicitly for large models that OOM on
    bigger (batch, seq) combinations (the profiler records NaN rows on
    OOM and continues, but caller can prune upfront to save wall time).
    """
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
