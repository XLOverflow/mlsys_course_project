"""Forward/backward shape + invariance tests for PerKernelMLPCostModel."""
import pickle
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hetero_cost_model.data import LatencyDataset, Sample, sample_to_pyg
from hetero_cost_model.graph import GraphRepr
from hetero_cost_model.hardware import HARDWARE_REGISTRY
from hetero_cost_model.models.per_kernel_mlp import PerKernelMLPCostModel
from hetero_cost_model.strategies import InferenceConfig

GRAPH_DIR = Path("data/graphs")


def _mini_samples() -> list:
    """Build 4 real samples across 2 models and 2 GPUs for batched testing."""
    from torch_geometric.loader import DataLoader
    models = ["bert-base", "gpt2-small"]
    gpus = ["h100", "a100"]
    samples = []
    for m in models:
        with open(GRAPH_DIR / f"{m}.pkl", "rb") as f:
            g: GraphRepr = pickle.load(f)
        for gk in gpus:
            samples.append(Sample(
                graph=g,
                config=InferenceConfig(batch_size=2, seq_len=128),
                hardware=HARDWARE_REGISTRY[gk],
                latency_ms=10.0,
                model_name=m,
            ))
    return samples


def test_per_kernel_mlp_forward_shape_matches_batch_size():
    from torch_geometric.loader import DataLoader
    samples = _mini_samples()
    loader = DataLoader(LatencyDataset(samples), batch_size=len(samples))
    batch = next(iter(loader))

    model = PerKernelMLPCostModel()
    out = model(batch)
    assert out.shape == (len(samples),), (
        f"expected one prediction per graph ({len(samples)},), got {tuple(out.shape)}"
    )
    assert torch.isfinite(out).all()


def test_per_kernel_mlp_is_permutation_invariant_over_nodes():
    """Scatter-sum must be invariant to node ordering within a graph.

    Previous sources of flakiness (resolved in this version):
    - model weights varied with global torch state → seed before
      instantiation
    - torch.randperm used torch's default stream → manually seed first
    - batch.clone() with torch_geometric 2.7+ can share tensors → do
      explicit ``.detach().clone()`` on the relevant fields.
    """
    from torch_geometric.loader import DataLoader

    # Seed before *anything* that touches torch's random state.
    torch.manual_seed(1234)
    samples = _mini_samples()[:1]
    loader = DataLoader(LatencyDataset(samples), batch_size=1)
    batch = next(iter(loader))

    model = PerKernelMLPCostModel()
    model.eval()

    with torch.no_grad():
        baseline = model(batch).item()

        N = batch.x.size(0)
        perm = torch.randperm(N)
        shuffled = batch.clone()
        shuffled.x = batch.x.detach().clone()[perm]
        # data.batch unchanged: every node belongs to graph 0 either way
        shuffled_out = model(shuffled).item()

    # Summation is exactly permutation-invariant; tolerance just covers
    # any fp rounding from summing in a different order.
    assert abs(baseline - shuffled_out) < 1e-4, (
        f"baseline={baseline}, shuffled={shuffled_out}, diff={abs(baseline-shuffled_out)}"
    )


def test_per_kernel_mlp_backward_produces_finite_grads():
    from torch_geometric.loader import DataLoader
    samples = _mini_samples()
    loader = DataLoader(LatencyDataset(samples), batch_size=len(samples))
    batch = next(iter(loader))

    model = PerKernelMLPCostModel()
    out = model(batch)
    loss = torch.nn.functional.mse_loss(out, batch.y)
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_per_kernel_mlp_params_are_positive_and_smaller_than_gnn():
    """Per-kernel MLP has no pooling head and shares weights across nodes, so
    it naturally has fewer params than GNN. Assert it's non-trivially sized
    (not a smoke test / empty network) but lighter than GNN (consistent with
    NeuSight-style per-kernel predictor)."""
    from hetero_cost_model.models.gnn import CostModel
    gnn = CostModel()
    pk  = PerKernelMLPCostModel()
    n_gnn = sum(p.numel() for p in gnn.parameters())
    n_pk  = sum(p.numel() for p in pk.parameters())
    assert n_pk > 1000, f"per-kernel MLP too small ({n_pk} params) — suspected bug"
    assert n_pk < n_gnn, f"per-kernel MLP ({n_pk}) shouldn't exceed GNN ({n_gnn})"
