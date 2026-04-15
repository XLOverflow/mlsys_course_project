import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from hetero_cost_model.training import pairwise_ranking_loss


def test_ranking_loss_zero_when_ordered():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 3.0])
    assert pairwise_ranking_loss(pred, target).item() == 0.0


def test_ranking_loss_positive_when_inverted():
    pred = torch.tensor([3.0, 2.0, 1.0])
    target = torch.tensor([1.0, 2.0, 3.0])
    assert pairwise_ranking_loss(pred, target).item() > 0.0
