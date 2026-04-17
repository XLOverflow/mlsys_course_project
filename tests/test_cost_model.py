import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Batch, Data

from hetero_cost_model.graph import NODE_FEATURE_DIM
from hetero_cost_model.hardware import HARDWARE_FEATURE_DIM
from hetero_cost_model.models import CostModel, MLPCostModel
from hetero_cost_model.strategies import CONFIG_FEATURE_DIM


def _fake_data(n_nodes: int = 6) -> Data:
    """Fake graph: node features only; s and h are graph-level."""
    x = torch.randn(n_nodes, NODE_FEATURE_DIM)
    src = list(range(n_nodes - 1))
    dst = list(range(1, n_nodes))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    s = torch.rand(CONFIG_FEATURE_DIM)
    h = torch.rand(HARDWARE_FEATURE_DIM)
    return Data(x=x, edge_index=edge_index, s=s, h=h, y=torch.tensor([5.0]))


def test_gat_forward_shape():
    model = CostModel(hidden_dim=32, num_layers=2, heads=2, backbone="gat")
    out = model(Batch.from_data_list([_fake_data(), _fake_data(8)]))
    assert out.shape == (2,)


def test_transformer_forward_shape():
    model = CostModel(hidden_dim=32, num_layers=2, heads=2, backbone="transformer")
    assert model(Batch.from_data_list([_fake_data()])).shape == (1,)


def test_mlp_ablation_forward_shape():
    model = MLPCostModel()
    assert model(Batch.from_data_list([_fake_data(), _fake_data()])).shape == (2,)


def test_different_hardware_gives_different_output():
    """Same graph + config but different h should produce different predictions."""
    model = CostModel(hidden_dim=32, num_layers=1, heads=2)
    model.eval()
    base = _fake_data()
    d1 = Data(x=base.x, edge_index=base.edge_index,
               s=base.s, h=torch.zeros(HARDWARE_FEATURE_DIM), y=base.y)
    d2 = Data(x=base.x, edge_index=base.edge_index,
               s=base.s, h=torch.ones(HARDWARE_FEATURE_DIM), y=base.y)
    with torch.no_grad():
        o1 = model(Batch.from_data_list([d1]))
        o2 = model(Batch.from_data_list([d2]))
    assert not torch.allclose(o1, o2)


def test_backward_produces_finite_grads():
    model = CostModel(hidden_dim=16, num_layers=1, heads=2)
    batch = Batch.from_data_list([_fake_data(), _fake_data()])
    loss = torch.nn.functional.mse_loss(model(batch), batch.y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)
