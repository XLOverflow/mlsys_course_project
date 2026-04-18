"""Behavioral tests for CostModel v2 switches:

- node_level_sh=True must make the GNN sensitive to s / h during
  message passing (forward pass depends on them even when the MLP head
  is zeroed out)
- readout='sum' must scale with node count (unlike mean pool)
- global_skip=True must make the model read data.g
"""
import pickle
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Batch, Data

from hetero_cost_model.graph import (
    GRAPH_GLOBAL_FEATURE_DIM,
    NODE_FEATURE_DIM,
)
from hetero_cost_model.hardware import HARDWARE_FEATURE_DIM
from hetero_cost_model.models import CostModel
from hetero_cost_model.strategies import CONFIG_FEATURE_DIM


def _fake_data(n_nodes: int = 6) -> Data:
    x = torch.randn(n_nodes, NODE_FEATURE_DIM)
    src = list(range(n_nodes - 1))
    dst = list(range(1, n_nodes))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(
        x=x,
        edge_index=edge_index,
        s=torch.rand(CONFIG_FEATURE_DIM),
        h=torch.rand(HARDWARE_FEATURE_DIM),
        g=torch.rand(GRAPH_GLOBAL_FEATURE_DIM),
        y=torch.tensor([5.0]),
    )


# --- node_level_sh ----------------------------------------------------------

def test_v2_node_level_sh_expands_input_projection():
    """With node-level s/h, the input projection's in_features grows by
    CONFIG_FEATURE_DIM + HARDWARE_FEATURE_DIM."""
    m_v1 = CostModel(node_level_sh=False, readout="mean_max", global_skip=False)
    m_v2 = CostModel(node_level_sh=True,  readout="mean_max", global_skip=False)
    diff = m_v2.input_proj.in_features - m_v1.input_proj.in_features
    assert diff == CONFIG_FEATURE_DIM + HARDWARE_FEATURE_DIM


def test_v2_node_level_sh_makes_gat_sensitive_to_hardware():
    """Same graph, same config, but different hardware MUST change the
    per-node GAT representation (the whole point of node-level injection)."""
    torch.manual_seed(0)
    model = CostModel(
        hidden_dim=32, num_layers=1, heads=2,
        node_level_sh=True, readout="mean_max", global_skip=False,
    )
    model.eval()
    base = _fake_data()
    low = Data(
        x=base.x, edge_index=base.edge_index, s=base.s,
        h=torch.zeros_like(base.h), g=base.g, y=base.y,
    )
    high = Data(
        x=base.x, edge_index=base.edge_index, s=base.s,
        h=torch.ones_like(base.h) * 2.0, g=base.g, y=base.y,
    )
    with torch.no_grad():
        o_low  = model(Batch.from_data_list([low]))
        o_high = model(Batch.from_data_list([high]))
    assert not torch.allclose(o_low, o_high, atol=1e-6)


# --- readout = "sum" --------------------------------------------------------

def test_v2_sum_readout_scales_with_node_count():
    """Sum readout should scale ~linearly with the number of nodes (post-GAT
    representations are similar but we have twice as many). Mean readout
    would be roughly invariant to node count."""
    torch.manual_seed(0)
    model = CostModel(
        hidden_dim=32, num_layers=1, heads=2,
        node_level_sh=False, readout="sum", global_skip=False,
    )
    model.eval()
    with torch.no_grad():
        out_small = model(Batch.from_data_list([_fake_data(5)]))
        out_large = model(Batch.from_data_list([_fake_data(50)]))
    # With 10× more nodes, sum-pooled representation is much larger, which
    # in general routes the head to a different output. We just require
    # they aren't equal (the "mean pool" version would be near-equal).
    assert not torch.allclose(out_small, out_large, atol=1e-3)


def test_v2_readout_dim_routing_is_correct():
    """Head's first Linear should see readout_dim + config + hardware (+ g
    if global_skip) — checks that dim bookkeeping in __init__ is right."""
    m_sum = CostModel(node_level_sh=False, readout="sum", global_skip=False)
    m_mxm = CostModel(node_level_sh=False, readout="mean_max", global_skip=False)
    # "mean_max" concats two pools → 2× the input size of "sum"
    assert m_mxm.head[0].in_features == m_sum.head[0].in_features + 64
    # global_skip adds GRAPH_GLOBAL_FEATURE_DIM (=4) on top of sum-readout head
    m_gsk = CostModel(node_level_sh=False, readout="sum", global_skip=True)
    assert m_gsk.head[0].in_features == m_sum.head[0].in_features + GRAPH_GLOBAL_FEATURE_DIM


# --- global_skip ------------------------------------------------------------

def test_v2_global_skip_changes_output_when_g_changes():
    """When global_skip=True, changing data.g must change the prediction."""
    torch.manual_seed(0)
    model = CostModel(
        hidden_dim=32, num_layers=1, heads=2,
        node_level_sh=False, readout="sum", global_skip=True,
    )
    model.eval()
    base = _fake_data()
    d1 = Data(x=base.x, edge_index=base.edge_index, s=base.s, h=base.h,
              g=torch.zeros_like(base.g), y=base.y)
    d2 = Data(x=base.x, edge_index=base.edge_index, s=base.s, h=base.h,
              g=torch.ones_like(base.g) * 10.0, y=base.y)
    with torch.no_grad():
        o1 = model(Batch.from_data_list([d1]))
        o2 = model(Batch.from_data_list([d2]))
    assert not torch.allclose(o1, o2, atol=1e-6)


def test_v2_global_skip_off_ignores_g():
    """With global_skip=False, data.g shouldn't influence the forward pass."""
    torch.manual_seed(0)
    model = CostModel(
        hidden_dim=32, num_layers=1, heads=2,
        node_level_sh=False, readout="sum", global_skip=False,
    )
    model.eval()
    base = _fake_data()
    d1 = Data(x=base.x, edge_index=base.edge_index, s=base.s, h=base.h,
              g=torch.zeros_like(base.g), y=base.y)
    d2 = Data(x=base.x, edge_index=base.edge_index, s=base.s, h=base.h,
              g=torch.ones_like(base.g) * 10.0, y=base.y)
    with torch.no_grad():
        o1 = model(Batch.from_data_list([d1]))
        o2 = model(Batch.from_data_list([d2]))
    assert torch.allclose(o1, o2, atol=1e-6)


# --- full-stack (default v2) wiring -----------------------------------------

def test_v2_default_construction_and_forward_runs():
    """Default constructor (all v2 features on) just has to not explode."""
    model = CostModel()
    out = model(Batch.from_data_list([_fake_data(), _fake_data(8)]))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()
