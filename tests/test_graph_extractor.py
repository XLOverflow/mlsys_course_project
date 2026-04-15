import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from hetero_cost_model.graph import NODE_FEATURE_DIM, extract_graph


class _TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 8)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_extract_tiny_mlp():
    g = extract_graph(_TinyMLP(), (torch.randn(2, 16),), name="tiny")
    assert g.num_nodes() >= 4
    assert any(n.op_type == "linear" for n in g.nodes)
    assert len(g.node_feature_matrix()[0]) == NODE_FEATURE_DIM
    assert len(g.edges) >= g.num_nodes() - 1


def test_flops_nonzero_for_linears():
    g = extract_graph(_TinyMLP(), (torch.randn(2, 16),))
    assert sum(n.flops for n in g.nodes if n.op_type == "linear") > 0
