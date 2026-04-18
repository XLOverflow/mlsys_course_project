"""NeuSight-style per-kernel MLP cost model.

For each node in the computation graph, an MLP predicts that node's
per-op latency from [op_features | s | h]. Total graph latency is the
scatter-sum over the per-op predictions. No message passing.

Role: this is the "GNN w/o edges" ablation that isolates the marginal
contribution of graph structure (message passing) vs. per-op features
alone. Same parameter regime as the GNN head (similar capacity) so that
any gap between the two is attributable to message passing, not width.

The forward signature matches ``CostModel``'s so it can slot into the
existing ``training.train``/``predict`` loops and PyG DataLoader pipeline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from hetero_cost_model.graph import NODE_FEATURE_DIM
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
    ):
        super().__init__()
        per_node_in = node_in_dim + config_dim + hardware_dim
        self.per_node = nn.Sequential(
            nn.Linear(per_node_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data) -> torch.Tensor:
        # Number of graphs in the batch. data.batch is [N_total_nodes], an
        # int tensor that maps each node to its graph index.
        b = int(data.batch.max().item()) + 1 if data.batch.numel() > 0 else 1

        # Graph-level config ``s`` and hardware ``h`` are stored concatenated
        # by PyG (one flat tensor of length B*dim). Reshape back to [B, dim]
        # then broadcast to each node by indexing with data.batch.
        s_per_graph = data.s.view(b, -1)      # [B, config_dim]
        h_per_graph = data.h.view(b, -1)      # [B, hardware_dim]
        s_per_node = s_per_graph[data.batch]  # [N_total, config_dim]
        h_per_node = h_per_graph[data.batch]  # [N_total, hardware_dim]

        x = torch.cat([data.x, s_per_node, h_per_node], dim=-1)
        per_op_ms = self.per_node(x).squeeze(-1)   # [N_total]
        # sum per-op latencies over each graph → [B]
        return scatter(per_op_ms, data.batch, dim=0, reduce="sum")


__all__ = ["PerKernelMLPCostModel"]
