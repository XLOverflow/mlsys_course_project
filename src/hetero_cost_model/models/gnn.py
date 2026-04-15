"""GNN-based cost model with pluggable GAT / Graph-Transformer backbone.

Given a PyG ``Data`` batch whose nodes carry
``[op features | strategy one-hot | hardware vector]`` the model predicts a
single scalar end-to-end latency per graph.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    TransformerConv,
    global_max_pool,
    global_mean_pool,
)

from hetero_cost_model.graph import NODE_FEATURE_DIM
from hetero_cost_model.hardware import HARDWARE_FEATURE_DIM
from hetero_cost_model.strategies import STRATEGY_FEATURE_DIM


Backbone = Literal["gat", "transformer"]

_BACKBONES = {"gat": GATConv, "transformer": TransformerConv}


class CostModel(nn.Module):
    """Predict end-to-end latency ``T̂`` from a (graph, strategy, hardware) triple."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        backbone: Backbone = "gat",
        dropout: float = 0.1,
        node_in_dim: int = NODE_FEATURE_DIM,
        strategy_dim: int = STRATEGY_FEATURE_DIM,
        hardware_dim: int = HARDWARE_FEATURE_DIM,
    ):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"unknown backbone: {backbone}")

        self.backbone = backbone
        self.dropout = dropout

        in_dim = node_in_dim + strategy_dim + hardware_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        conv_cls = _BACKBONES[backbone]
        head_dim = hidden_dim // heads
        self.convs = nn.ModuleList([
            conv_cls(hidden_dim, head_dim, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x = F.relu(self.input_proj(data.x))
        for conv in self.convs:
            x = F.relu(conv(x, data.edge_index))
        pooled = torch.cat(
            [global_mean_pool(x, data.batch), global_max_pool(x, data.batch)],
            dim=-1,
        )
        return self.head(pooled).squeeze(-1)


__all__ = ["CostModel", "Backbone"]
