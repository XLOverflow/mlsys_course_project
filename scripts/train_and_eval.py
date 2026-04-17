"""Train the GNN cost model + XGBoost baseline + Roofline, compare on hold-out.

Reads the CSV produced by ``scripts/run_profiling.py``, builds ``Sample``
objects, trains each method, and reports MAPE / Spearman ρ on the
chosen evaluation split.

Splits available via ``--split``:

  * ``random``       — random 80/20 (quickest; over-reports because of
                       graph/hardware leakage). Sanity check only.
  * ``leave-gpu=<k>``— leave-one-GPU-out CV on training GPUs
                       (k ∈ v100/a100/h100). This is the ``main claim``
                       per the Week 1 milestone in two_week_execution_plan.md.
  * ``leave-model=<name>`` — leave-one-model-out.
  * ``zero-shot=<k>``— train on everything else, test on GPU k.
                       Use k=b200 for the hero experiment.

Example:

  python scripts/train_and_eval.py --csv data/raw/all.csv --split leave-gpu=v100
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch

from hetero_cost_model.baselines import XGBoostBaseline, roofline_latency
from hetero_cost_model.data import LatencyDataset, Sample, load_samples_from_csv
from hetero_cost_model.metrics import mape, spearman, top_k_accuracy
from hetero_cost_model.models.gnn import CostModel
from hetero_cost_model.training import TrainConfig, predict, train


# --- CLI ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True, help="Profiling CSV path")
    p.add_argument("--graph-dir", type=Path, default=Path("data/graphs"))
    p.add_argument("--split", default="random",
                   help="random | leave-gpu=<key> | leave-model=<name> | zero-shot=<key>")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone", default="gat", choices=["gat", "transformer"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


# --- Splits ------------------------------------------------------------------

Split = Tuple[List[Sample], List[Sample]]


def make_split(samples: List[Sample], spec: str, seed: int) -> Split:
    if spec == "random":
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(samples))
        cut = int(0.8 * len(samples))
        train_ix = set(idx[:cut].tolist())
        tr = [s for i, s in enumerate(samples) if i in train_ix]
        te = [s for i, s in enumerate(samples) if i not in train_ix]
        return tr, te

    if spec.startswith("leave-gpu="):
        held = spec.split("=", 1)[1]
        tr = [s for s in samples if s.hardware.name.lower() != held.lower()]
        te = [s for s in samples if s.hardware.name.lower() == held.lower()]
        return tr, te

    if spec.startswith("leave-model="):
        held = spec.split("=", 1)[1]
        tr = [s for s in samples if s.model_name != held]
        te = [s for s in samples if s.model_name == held]
        return tr, te

    if spec.startswith("zero-shot="):
        held = spec.split("=", 1)[1]
        tr = [s for s in samples if s.hardware.name.lower() != held.lower()]
        te = [s for s in samples if s.hardware.name.lower() == held.lower()]
        return tr, te

    raise ValueError(f"unknown split: {spec}")


# --- Evaluation --------------------------------------------------------------

@dataclass
class Report:
    name: str
    pred: Sequence[float]
    true: Sequence[float]

    @property
    def mape(self) -> float:
        return mape(self.pred, self.true)

    @property
    def spearman(self) -> float:
        return spearman(self.pred, self.true)

    @property
    def top1(self) -> float:
        return top_k_accuracy(self.pred, self.true, k=1)


def run_gnn(
    train_samples: Sequence[Sample], test_samples: Sequence[Sample],
    backbone: str, cfg: TrainConfig,
) -> Report:
    torch.manual_seed(0)
    model = CostModel(backbone=backbone)
    train(model, LatencyDataset(list(train_samples)), cfg)
    pred, true = predict(model, LatencyDataset(list(test_samples)), device=cfg.device)
    return Report(f"GNN ({backbone})", pred, true)


def run_xgboost(
    train_samples: Sequence[Sample], test_samples: Sequence[Sample],
) -> Report:
    bl = XGBoostBaseline().fit(list(train_samples))
    pred = bl.predict(list(test_samples)).tolist()
    true = [s.latency_ms for s in test_samples]
    return Report("XGBoost", pred, true)


def run_roofline(test_samples: Sequence[Sample]) -> Report:
    pred = [roofline_latency(s.graph, s.config, s.hardware) for s in test_samples]
    true = [s.latency_ms for s in test_samples]
    return Report("Roofline", pred, true)


# --- Main --------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    print("=" * 72)
    print(f"train_and_eval.py  split={args.split}  backbone={args.backbone}")
    print("=" * 72)

    print(f"\nLoading samples from {args.csv} + graphs from {args.graph_dir}...")
    samples = load_samples_from_csv(args.csv, args.graph_dir)
    if not samples:
        print("No usable samples found (CSV empty or all rows filtered). Aborting.")
        return 1
    print(f"  {len(samples)} total samples, "
          f"{len({s.model_name for s in samples})} distinct models, "
          f"{len({s.hardware.name for s in samples})} distinct GPUs")

    tr, te = make_split(samples, args.split, args.seed)
    print(f"  train: {len(tr)}  test: {len(te)}")
    if not tr or not te:
        print("empty split half. Aborting.")
        return 1

    cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        ranking_lambda=0.1, device=args.device,
    )

    reports: List[Report] = []
    print("\nTraining/evaluating ...")
    reports.append(run_roofline(te))
    reports.append(run_xgboost(tr, te))
    reports.append(run_gnn(tr, te, args.backbone, cfg))

    print("\n" + "-" * 72)
    print(f"{'Method':<18}  {'MAPE':>8}  {'Spearman':>10}  {'Top-1':>8}")
    print("-" * 72)
    for r in reports:
        mape_str = f"{r.mape * 100:.2f}%" if not math.isnan(r.mape) else "nan"
        print(f"{r.name:<18}  {mape_str:>8}  {r.spearman:>10.3f}  {r.top1:>8.3f}")
    print("-" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
