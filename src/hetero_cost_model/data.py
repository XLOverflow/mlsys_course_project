"""PyG dataset construction from (graph, strategy, hardware, latency) samples.

Each sample becomes one PyG ``Data`` object whose node features are
``[op features | strategy one-hot | hardware vector broadcast]``. The
hardware vector is broadcast to every node so message passing can modulate
node representations by the target device's characteristics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch_geometric.data import Data, Dataset

from hetero_cost_model.graph import GraphRepr
from hetero_cost_model.hardware import Hardware
from hetero_cost_model.strategies import CPU, GPU, STRATEGY_FEATURE_DIM, Strategy


@dataclass
class Sample:
    """A single training example."""

    graph: GraphRepr
    strategy: Strategy
    hardware: Hardware
    latency_ms: float
    model_name: str = ""


def sample_to_pyg(sample: Sample) -> Data:
    """Convert one :class:`Sample` into a PyG ``Data`` graph."""
    node_feats = torch.tensor(sample.graph.node_feature_matrix(), dtype=torch.float)
    n = node_feats.shape[0]

    strategy_onehot = torch.zeros(n, STRATEGY_FEATURE_DIM)
    for i, placement in enumerate(sample.strategy.placements):
        strategy_onehot[i, GPU if placement == GPU else CPU] = 1.0

    hw_vec = torch.tensor(sample.hardware.to_vector(), dtype=torch.float)
    hw_broadcast = hw_vec.unsqueeze(0).expand(n, -1)

    x = torch.cat([node_feats, strategy_onehot, hw_broadcast], dim=-1)

    if sample.graph.edges:
        edge_index = torch.tensor(sample.graph.edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long)

    y = torch.tensor([sample.latency_ms], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)


class LatencyDataset(Dataset):
    """In-memory PyG dataset — sufficient for the project's ~1500 samples."""

    def __init__(self, samples: List[Sample]):
        super().__init__()
        self.samples = samples
        self._cache = [sample_to_pyg(s) for s in samples]

    # PyG's abstract interface
    def len(self) -> int:
        return len(self._cache)

    def get(self, idx: int) -> Data:
        return self._cache[idx]

    # Python sequence protocol
    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> Data:
        return self._cache[idx]


__all__ = ["Sample", "LatencyDataset", "sample_to_pyg"]
