"""GNN-based cost model with pluggable GAT / Graph-Transformer backbone.

Default (v2) architecture, rewritten 2026-04-17 to beat a strong tabular
XGBoost baseline:

  1. **Per-node s/h injection** — the inference config and hardware vector
     are broadcast to every node and concatenated into node features BEFORE
     the GAT layers. Message passing can then learn hardware-dependent
     patterns (e.g. "attention on low-bandwidth GPUs runs disproportionately
     slow"). This follows Akhauri & Abdelfattah (MLSys'24)'s OPHW design.

  2. **Sum readout** instead of mean+max pool. Latency is physically
     additive over ops (prefill executes kernels sequentially), so the
     readout should preserve totals, not normalize them away. mean_pool
     = avg per-op latency loses the "gpt2-large has 3× more ops than
     gpt2-small" signal.

  3. **Global-summary skip connection** — [log1p(total_flops),
     log1p(total_memory_bytes), log1p(num_nodes), log1p(num_edges)] is
     concatenated to the head input. These are the Roofline inputs that
     XGBoost already gets directly; with this skip, the GNN head doesn't
     have to reconstruct totals from per-node features, and message
     passing only needs to learn the *residual* on top of the Roofline
     approximation.

Each of the three changes is independently toggleable so Table 3's
"v1 vs v2" can be broken down per-component in an ablation.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    TransformerConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from hetero_cost_model.graph import GRAPH_GLOBAL_FEATURE_DIM, NODE_FEATURE_DIM
from hetero_cost_model.hardware import HARDWARE_FEATURE_DIM
from hetero_cost_model.strategies import CONFIG_FEATURE_DIM


Backbone = Literal["gat", "transformer"]
Readout = Literal["sum", "mean_max"]

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
        # ---- v2 defaults (v1 to revert: node_level_sh=False, readout="mean_max", global_skip=False) ----
        node_level_sh: bool = True,
        readout: Readout = "sum",
        global_skip: bool = True,
    ):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"unknown backbone: {backbone}")
        if readout not in ("sum", "mean_max"):
            raise ValueError(f"unknown readout: {readout}")

        self.dropout = dropout
        self.node_level_sh = node_level_sh
        self.readout = readout
        self.global_skip = global_skip

        # (a) Input projection — input dim grows if we inject s/h per-node
        in_dim = node_in_dim + (config_dim + hardware_dim if node_level_sh else 0)
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # (b) GAT / Transformer conv stack
        conv_cls = _BACKBONES[backbone]
        head_dim = hidden_dim // heads
        self.convs = nn.ModuleList([
            conv_cls(hidden_dim, head_dim, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # (c) Head input = readout(node_repr) + s + h + [optional] global_skip
        readout_dim = hidden_dim if readout == "sum" else hidden_dim * 2
        head_in = readout_dim + config_dim + hardware_dim
        if global_skip:
            head_in += GRAPH_GLOBAL_FEATURE_DIM
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    # --- helpers -------------------------------------------------------------

    def _broadcast_to_nodes(self, data, graph_attr: torch.Tensor) -> torch.Tensor:
        """Given a graph-level attribute of shape [B*D] (PyG flattens across
        the batch dim when it concatenates Data objects), reshape to [B, D]
        and index with ``data.batch`` to produce a per-node [N, D] tensor."""
        b = int(data.batch.max().item()) + 1 if data.batch.numel() > 0 else 1
        return graph_attr.view(b, -1)[data.batch]

    # --- forward -------------------------------------------------------------

    def forward(self, data) -> torch.Tensor:
        # (a) Per-node s/h injection BEFORE the GAT stack
        if self.node_level_sh:
            s_per_node = self._broadcast_to_nodes(data, data.s)
            h_per_node = self._broadcast_to_nodes(data, data.h)
            x = torch.cat([data.x, s_per_node, h_per_node], dim=-1)
        else:
            x = data.x

        x = F.relu(self.input_proj(x))
        for conv in self.convs:
            x = F.relu(conv(x, data.edge_index))

        # (b) Readout — sum preserves additive scale; mean_max is legacy v1
        if self.readout == "sum":
            pooled = global_add_pool(x, data.batch)
        else:
            pooled = torch.cat(
                [global_mean_pool(x, data.batch), global_max_pool(x, data.batch)],
                dim=-1,
            )

        # (c) Build head input — always include graph-level s and h; optionally
        # concat the Roofline-style global summary so the head has a direct
        # (non-reconstructed) view of total_flops / total_memory / counts.
        b = pooled.size(0)
        parts = [pooled, data.s.view(b, -1), data.h.view(b, -1)]
        if self.global_skip:
            parts.append(data.g.view(b, -1))
        combined = torch.cat(parts, dim=-1)
        return self.head(combined).squeeze(-1)


__all__ = ["CostModel", "Backbone", "Readout"]
