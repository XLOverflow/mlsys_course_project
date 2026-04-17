"""Flat-MLP ablation model (drops graph structure entirely).

Used in the ablation table to isolate the contribution of message passing.
Mirrors the GNN design: mean-pool node features, then concatenate s and h.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from hetero_cost_model.graph import NODE_FEATURE_DIM
from hetero_cost_model.hardware import HARDWARE_FEATURE_DIM
from hetero_cost_model.strategies import CONFIG_FEATURE_DIM


class MLPCostModel(nn.Module):
    """Mean-pool node features, concatenate s+h, run plain MLP regressor."""

    def __init__(
        self,
        hidden_dim: int = 256,
        node_in_dim: int = NODE_FEATURE_DIM,
        config_dim: int = CONFIG_FEATURE_DIM,
        hardware_dim: int = HARDWARE_FEATURE_DIM,
    ):
        super().__init__()
        in_dim = node_in_dim + config_dim + hardware_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data) -> torch.Tensor:
        pooled = global_mean_pool(data.x, data.batch)
        b = pooled.size(0)
        combined = torch.cat([pooled, data.s.view(b, -1), data.h.view(b, -1)], dim=-1)
        return self.net(combined).squeeze(-1)


__all__ = ["MLPCostModel"]
