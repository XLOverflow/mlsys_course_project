"""GNN-based cost model with pluggable GAT / Graph-Transformer backbone.

Architecture:
  1. Node feature projection (op structure only, no s/h per node)
  2. N-layer GNN (message passing over compute graph edges)
  3. Graph readout: mean + max pooling → concatenated
  4. Concatenate graph-level s (config) and h (hardware) vectors
  5. MLP head → predicted latency scalar T̂

s and h are injected after readout so node representations stay
architecture-specific; the global features modulate the final prediction.
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
from hetero_cost_model.strategies import CONFIG_FEATURE_DIM


Backbone = Literal["gat", "transformer"]

_BACKBONES = {"gat": GATConv, "transformer": TransformerConv}


class CostModel(nn.Module):
    """Predict end-to-end latency T̂ from a (graph, config, hardware) triple."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        backbone: Backbone = "gat",
        dropout: float = 0.1,
        node_in_dim: int = NODE_FEATURE_DIM,
        config_dim: int = CONFIG_FEATURE_DIM,
        hardware_dim: int = HARDWARE_FEATURE_DIM,
    ):
        # Defaults sized for 1500-2500 sample regime (data_collection_plan.md
        # §2.3). Earlier draft used hidden=128/num_layers=3 (~155 K params);
        # at the training-data scale this likely over-parameterizes the model
        # and lets the hardware branch memorize the few training anchors.
        # hidden=64/num_layers=2 gives ~78 K params — still large for 1.5 K
        # samples but markedly healthier. Can be overridden via CLI for
        # ablation (e.g. ``--hidden 128`` to reproduce the old capacity).
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"unknown backbone: {backbone}")

        self.dropout = dropout
        self.input_proj = nn.Linear(node_in_dim, hidden_dim)

        conv_cls = _BACKBONES[backbone]
        head_dim = hidden_dim // heads
        self.convs = nn.ModuleList([
            conv_cls(hidden_dim, head_dim, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # readout produces hidden_dim*2; then we append s and h
        head_in = hidden_dim * 2 + config_dim + hardware_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
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
        )  # [B, hidden_dim*2]

        # PyG concatenates graph-level attrs along dim=0 when batching,
        # so data.s is [B*config_dim] and must be reshaped to [B, config_dim].
        b = pooled.size(0)
        combined = torch.cat([
            pooled,
            data.s.view(b, -1),
            data.h.view(b, -1),
        ], dim=-1)
        return self.head(combined).squeeze(-1)


__all__ = ["CostModel", "Backbone"]
