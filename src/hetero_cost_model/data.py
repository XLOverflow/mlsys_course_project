"""PyG dataset construction from (graph, config, hardware, latency) samples.

Each sample becomes one PyG ``Data`` object:
  - ``x``  : node feature matrix [N, NODE_FEATURE_DIM] — op structure only
  - ``s``  : inference config vector [CONFIG_FEATURE_DIM] — graph-level
  - ``h``  : hardware feature vector [HARDWARE_FEATURE_DIM] — graph-level
  - ``y``  : measured latency in ms — scalar

s and h are graph-level attributes concatenated in the model head after
graph readout, so node features stay architecture-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch_geometric.data import Data, Dataset

from hetero_cost_model.graph import GraphRepr
from hetero_cost_model.hardware import Hardware
from hetero_cost_model.strategies import InferenceConfig


@dataclass
class Sample:
    """A single training example."""

    graph: GraphRepr
    config: InferenceConfig
    hardware: Hardware
    latency_ms: float
    model_name: str = ""


def sample_to_pyg(sample: Sample) -> Data:
    """Convert one :class:`Sample` into a PyG ``Data`` object."""
    x = torch.tensor(sample.graph.node_feature_matrix(), dtype=torch.float)
    s = torch.tensor(sample.config.to_vector(), dtype=torch.float)
    h = torch.tensor(sample.hardware.to_vector(), dtype=torch.float)
    y = torch.tensor([sample.latency_ms], dtype=torch.float)

    if sample.graph.edges:
        edge_index = torch.tensor(sample.graph.edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, s=s, h=h, y=y)


class LatencyDataset(Dataset):
    """In-memory PyG dataset — sufficient for the project's ~1500 samples."""

    def __init__(self, samples: List[Sample]):
        super().__init__()
        self._cache = [sample_to_pyg(s) for s in samples]

    def len(self) -> int:
        return len(self._cache)

    def get(self, idx: int) -> Data:
        return self._cache[idx]

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> Data:
        return self._cache[idx]


__all__ = ["Sample", "LatencyDataset", "sample_to_pyg"]
