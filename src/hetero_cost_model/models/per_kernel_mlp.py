"""NeuSight-style per-kernel MLP cost model.

For each node in the computation graph, an MLP predicts that node's
per-op latency from [op_features | s | h]. Total graph latency is the
scatter-sum over per-op predictions. No message passing.

Role: the "GNN w/o edges" ablation — isolates the marginal contribution
of graph structure (message passing) vs. per-op features alone. Same
forward-signature as CostModel so it slots into the existing training
loops and DataLoader pipeline without change.

Optionally concat a graph-global skip connection to each node's input
(the four Roofline-style log1p totals). This keeps the baseline's
inductive bias parallel to CostModel v2's ``global_skip``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from hetero_cost_model.graph import GRAPH_GLOBAL_FEATURE_DIM, NODE_FEATURE_DIM
from hetero_cost_model.hardware import HARDWARE_FEATURE_DIM
from hetero_cost_model.strategies import CONFIG_FEATURE_DIM


class PerKernelMLPCostModel(nn.Module):
    """Per-node MLP, scatter-sum to one prediction per graph."""

    def __init__(
        self,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        node_in_dim: int = NODE_FEATURE_DIM,
        config_dim: int = CONFIG_FEATURE_DIM,
        hardware_dim: int = HARDWARE_FEATURE_DIM,
        global_skip: bool = True,
    ):
        super().__init__()
        self.global_skip = global_skip
        per_node_in = node_in_dim + config_dim + hardware_dim
        if global_skip:
            per_node_in += GRAPH_GLOBAL_FEATURE_DIM
        self.per_node = nn.Sequential(
            nn.Linear(per_node_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data) -> torch.Tensor:
        b = int(data.batch.max().item()) + 1 if data.batch.numel() > 0 else 1
        s_per_node = data.s.view(b, -1)[data.batch]
        h_per_node = data.h.view(b, -1)[data.batch]
        parts = [data.x, s_per_node, h_per_node]
        if self.global_skip:
            g_per_node = data.g.view(b, -1)[data.batch]
            parts.append(g_per_node)
        x = torch.cat(parts, dim=-1)
        per_op_ms = self.per_node(x).squeeze(-1)
        return scatter(per_op_ms, data.batch, dim=0, reduce="sum")


__all__ = ["PerKernelMLPCostModel"]
