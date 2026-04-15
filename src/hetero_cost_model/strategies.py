"""Execution strategy enumeration: per-op CPU/GPU placement.

A full 2^N enumeration over graph nodes is intractable, so we only vary
placement on the top-K most compute-intensive ops and pin the rest to a
default device — matching the plan's "关键子图的所有 CPU/GPU 放置组合".
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import List, Sequence

from hetero_cost_model.graph import GraphRepr


CPU: int = 0
GPU: int = 1

# Per-node strategy feature = one-hot over {CPU, GPU}.
STRATEGY_FEATURE_DIM: int = 2


@dataclass
class Strategy:
    """An execution plan over a computation graph."""

    placements: List[int]
    batch_size: int = 1
    seq_length: int = 128

    def num_cpu_ops(self) -> int:
        return sum(1 for p in self.placements if p == CPU)

    def num_gpu_ops(self) -> int:
        return sum(1 for p in self.placements if p == GPU)


def identify_strategic_nodes(graph: GraphRepr, k: int = 4) -> List[int]:
    """Return indices of the top-k nodes by estimated FLOPs."""
    return sorted(
        range(graph.num_nodes()),
        key=lambda i: graph.nodes[i].flops,
        reverse=True,
    )[:k]


def enumerate_strategies(
    graph: GraphRepr,
    strategic_k: int = 4,
    default_device: int = GPU,
    batch_sizes: Sequence[int] = (1,),
    seq_lengths: Sequence[int] = (128,),
) -> List[Strategy]:
    """Enumerate 2^k placements over the top-k hot nodes × batch × seq grid."""
    n = graph.num_nodes()
    hot = identify_strategic_nodes(graph, strategic_k)
    strategies: List[Strategy] = []
    for combo in itertools.product((CPU, GPU), repeat=len(hot)):
        base = [default_device] * n
        for idx, device in zip(hot, combo):
            base[idx] = device
        for bs in batch_sizes:
            for sl in seq_lengths:
                strategies.append(
                    Strategy(placements=base[:], batch_size=bs, seq_length=sl)
                )
    return strategies


def random_strategies(graph: GraphRepr, n_samples: int, seed: int = 0) -> List[Strategy]:
    rng = random.Random(seed)
    n = graph.num_nodes()
    return [
        Strategy(placements=[rng.randint(CPU, GPU) for _ in range(n)])
        for _ in range(n_samples)
    ]


def full_device(graph: GraphRepr, device: int = GPU) -> Strategy:
    return Strategy(placements=[device] * graph.num_nodes())


__all__ = [
    "Strategy",
    "CPU",
    "GPU",
    "STRATEGY_FEATURE_DIM",
    "identify_strategic_nodes",
    "enumerate_strategies",
    "random_strategies",
    "full_device",
]
