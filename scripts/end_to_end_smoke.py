"""End-to-end CPU smoke test: verify every pipeline interface lines up.

Runs the full training data flow without touching a GPU, so interface
mismatches get caught here (for free) instead of on the critical-path
GPU session.

Phases:
  1. Load pre-extracted graph pkls for all 6 models
  2. Write a synthetic profiling CSV (30 rows = 5 GPUs × 6 models)
  3. Parse CSV → Sample via the shared ``load_samples_from_csv`` helper
  4. Build ``LatencyDataset`` → PyG ``DataLoader`` → batched ``Data``
  5. ``CostModel`` forward pass, check output shape / finiteness
  6. MSE loss, backward, verify all gradients are finite
  7. Run 2 epochs of the real training loop, assert no NaN
  8. XGBoost + Roofline baselines fit/predict over the same samples

This is the "gate" before Day 3 GPU profiling: if everything here
passes, the profiler (Xinhao) and training (Zhikai) can plug into
existing code without re-discovering broken interfaces.
"""
from __future__ import annotations

import csv
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch_geometric.loader import DataLoader

from hetero_cost_model.baselines import XGBoostBaseline, roofline_latency
from hetero_cost_model.data import LatencyDataset, Sample, load_samples_from_csv
from hetero_cost_model.graph import GraphRepr
from hetero_cost_model.hardware import HARDWARE_REGISTRY
from hetero_cost_model.metrics import mape, spearman
from hetero_cost_model.model_zoo import MODELS
from hetero_cost_model.models.gnn import CostModel
from hetero_cost_model.strategies import InferenceConfig
from hetero_cost_model.training import TrainConfig, train


GRAPH_DIR = Path("data/graphs")
CSV_PATH = Path("/tmp/smoke_profiling.csv")
TOTAL_PHASES = 8


def _phase(i: int, total: int, title: str) -> None:
    print(f"\n[{i}/{total}] {title}")


def load_graphs() -> Dict[str, GraphRepr]:
    out: Dict[str, GraphRepr] = {}
    for spec in MODELS:
        with open(GRAPH_DIR / f"{spec.name}.pkl", "rb") as f:
            out[spec.name] = pickle.load(f)
        assert isinstance(out[spec.name], GraphRepr)
        print(f"    {spec.name:14s} {out[spec.name].num_nodes():4d} nodes")
    return out


def write_synthetic_csv(graphs: Dict[str, GraphRepr]) -> int:
    """Roofline-flavored synthetic latencies so the GNN has a coherent target."""
    rows: List[dict] = []
    for gpu_key, hw in HARDWARE_REGISTRY.items():
        for spec in MODELS:
            g = graphs[spec.name]
            # Rough roofline: total_flops / peak + total_bytes / bw
            compute_ms = g.total_flops() / (hw.fp16_tflops * 1e12) * 1e3
            memory_ms = g.total_memory() / (hw.bandwidth_gbs * 1e9) * 1e3
            latency = max(compute_ms + memory_ms, 0.1)
            rows.append({
                "model_name": spec.name,
                "batch_size": 1,
                "seq_len": 64,
                "p50_ms": f"{latency:.4f}",
                "actual_gpu_name": gpu_key,
                "n_runs": 100,
                "noisy": False,
            })
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"    wrote {len(rows)} rows → {CSV_PATH}")
    return len(rows)


def load_samples(graphs: Dict[str, GraphRepr]) -> List[Sample]:
    """Use the shared helper (data.load_samples_from_csv) + pre-loaded cache."""
    samples = load_samples_from_csv(CSV_PATH, GRAPH_DIR, graphs=graphs)
    print(f"    constructed {len(samples)} Sample objects")
    return samples


def check_forward(model: CostModel, batch) -> torch.Tensor:
    pred = model(batch)
    expected_shape = (batch.num_graphs,)
    assert pred.shape == expected_shape, f"pred shape {tuple(pred.shape)} != expected {expected_shape}"
    assert torch.isfinite(pred).all(), "prediction contains NaN/Inf"
    print(
        f"    pred.shape={tuple(pred.shape)}  "
        f"range=[{pred.min().item():.3f}, {pred.max().item():.3f}]  "
        f"target range=[{batch.y.min().item():.3f}, {batch.y.max().item():.3f}]"
    )
    return pred


def check_backward(model: CostModel, pred: torch.Tensor, batch) -> None:
    loss = torch.nn.functional.mse_loss(pred, batch.y)
    loss.backward()
    total, have_grad = 0, 0
    for p in model.parameters():
        total += 1
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "gradient contains NaN/Inf"
            have_grad += 1
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    loss={loss.item():.3f}  trainable_tensors={have_grad}/{total}  total_params={n_params:,}")


def main() -> int:
    print("=" * 68)
    print("End-to-end CPU smoke test (pre-GPU gate)")
    print("=" * 68)

    _phase(1, TOTAL_PHASES, "Loading pre-extracted graph pkls...")
    graphs = load_graphs()

    _phase(2, TOTAL_PHASES, "Writing synthetic profiling CSV...")
    write_synthetic_csv(graphs)

    _phase(3, TOTAL_PHASES, "Parsing CSV → Sample objects (shared helper)...")
    samples = load_samples(graphs)
    assert len(samples) == len(MODELS) * len(HARDWARE_REGISTRY), "row count mismatch"

    _phase(4, TOTAL_PHASES, "Building LatencyDataset + DataLoader...")
    ds = LatencyDataset(samples)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    print(
        f"    batch: {batch.num_graphs} graphs  "
        f"x={tuple(batch.x.shape)}  edge_index={tuple(batch.edge_index.shape)}  "
        f"s={tuple(batch.s.shape)}  h={tuple(batch.h.shape)}  y={tuple(batch.y.shape)}"
    )

    _phase(5, TOTAL_PHASES, "CostModel (GAT) forward pass...")
    model = CostModel(hidden_dim=64, num_layers=2, heads=4)   # small for speed
    pred = check_forward(model, batch)

    _phase(6, TOTAL_PHASES, "MSE backward pass, gradient finiteness...")
    check_backward(model, pred, batch)

    _phase(7, TOTAL_PHASES, "Training loop, 2 epochs on full dataset...")
    fresh = CostModel(hidden_dim=64, num_layers=2, heads=4)
    cfg = TrainConfig(epochs=2, batch_size=8, lr=1e-3, ranking_lambda=0.1, device="cpu")
    history = train(fresh, ds, cfg)
    assert all(math.isfinite(h) for h in history), f"NaN in loss history: {history}"
    print(f"    epoch losses: {[f'{h:.3f}' for h in history]}")

    _phase(8, TOTAL_PHASES, "Baselines: XGBoost + Roofline fit/predict...")
    xgb = XGBoostBaseline(n_estimators=50, max_depth=4).fit(samples)
    xgb_pred = xgb.predict(samples).tolist()
    xgb_targets = [s.latency_ms for s in samples]
    roof_pred = [roofline_latency(s.graph, s.config, s.hardware) for s in samples]
    print(
        f"    XGBoost   — MAPE={mape(xgb_pred,  xgb_targets) * 100:6.2f}%  "
        f"Spearman={spearman(xgb_pred,  xgb_targets):.3f}   (in-sample fit)"
    )
    print(
        f"    Roofline  — MAPE={mape(roof_pred, xgb_targets) * 100:6.2f}%  "
        f"Spearman={spearman(roof_pred, xgb_targets):.3f}   (no training)"
    )

    print("\n" + "=" * 68)
    print("END-TO-END SMOKE TEST PASSED")
    print("All pipeline interfaces aligned. Ready for Day 3 GPU profiling.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
