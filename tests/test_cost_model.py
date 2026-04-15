import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Batch, Data

from hetero_cost_model.graph import NODE_FEATURE_DIM
from hetero_cost_model.hardware import HARDWARE_FEATURE_DIM
from hetero_cost_model.models import CostModel, MLPCostModel
from hetero_cost_model.strategies import STRATEGY_FEATURE_DIM


def _fake_data(n_nodes: int = 6) -> Data:
    in_dim = NODE_FEATURE_DIM + STRATEGY_FEATURE_DIM + HARDWARE_FEATURE_DIM
    x = torch.randn(n_nodes, in_dim)
    src = list(range(n_nodes - 1))
    dst = list(range(1, n_nodes))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=torch.tensor([5.0]))


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


def test_backward_produces_finite_grads():
    model = CostModel(hidden_dim=16, num_layers=1, heads=2)
    batch = Batch.from_data_list([_fake_data(), _fake_data()])
    loss = torch.nn.functional.mse_loss(model(batch), batch.y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)
