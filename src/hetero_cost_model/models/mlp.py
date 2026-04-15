"""Flat-MLP ablation model (drops graph structure entirely).

Used in the ablation table to isolate the contribution of message passing.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from hetero_cost_model.graph import NODE_FEATURE_DIM
from hetero_cost_model.hardware import HARDWARE_FEATURE_DIM
from hetero_cost_model.strategies import STRATEGY_FEATURE_DIM


class MLPCostModel(nn.Module):
    """Mean-pool the node features, then run a plain MLP regressor."""

    def __init__(
        self,
        hidden_dim: int = 256,
        node_in_dim: int = NODE_FEATURE_DIM,
        strategy_dim: int = STRATEGY_FEATURE_DIM,
        hardware_dim: int = HARDWARE_FEATURE_DIM,
    ):
        super().__init__()
        in_dim = node_in_dim + strategy_dim + hardware_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data) -> torch.Tensor:
        pooled = global_mean_pool(data.x, data.batch)
        return self.net(pooled).squeeze(-1)


__all__ = ["MLPCostModel"]
