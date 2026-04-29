"""Demo CLI: predict latency for a single (model, batch, seq, gpu) input.

Operationalizes the project's central finding (RQ1 vs RQ2 extrapolation
regimes) as an interactive prediction tool. The router decides per-call
whether to call XGBoost or the GNN based on whether the queried model
architecture is in the training fold.

Two regimes
-----------

  ``--regime in-distribution``
      Train on all 6 models in the dataset. The queried model is "seen",
      so the router picks XGBoost (the strong tabular baseline).

  ``--regime leave-out`` (default)
      Train on the 5 models other than the queried one. The queried
      model is "unseen", so the router picks the GNN (the architecture-
      extrapolation regime).

Both regimes reuse the existing pipeline (``data.load_samples_from_csv``
+ ``baselines.XGBoostBaseline`` + ``models.gnn.CostModel``). Training
runs on each invocation; expect ~30-60s for the GNN. For repeated
queries, prefer caching the trained models offline.

Examples
--------
    # Same input, two regimes — for poster screenshots side by side:
    python scripts/predict.py --model gpt2-large --batch 4 --seq 256 --gpu h100
    python scripts/predict.py --model gpt2-large --batch 4 --seq 256 --gpu h100 --regime in-distribution
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

from hetero_cost_model.baselines import XGBoostBaseline
from hetero_cost_model.data import LatencyDataset, Sample, load_samples_from_csv
from hetero_cost_model.hardware import HARDWARE_REGISTRY
from hetero_cost_model.models.gnn import CostModel
from hetero_cost_model.router import Router
from hetero_cost_model.strategies import InferenceConfig
from hetero_cost_model.training import TrainConfig, predict, train


# --- CLI ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="Model name (must match a graph in --graph-dir, e.g. gpt2-large)")
    p.add_argument("--batch", type=int, required=True, help="Batch size")
    p.add_argument("--seq", type=int, required=True, help="Sequence length")
    p.add_argument("--gpu", required=True,
                   help="GPU registry key (t4, a10, a100, l4, h100, h200, b200)")
    p.add_argument("--regime", choices=["leave-out", "in-distribution"], default="leave-out",
                   help="leave-out: train on other 5 models (router picks GNN). "
                        "in-distribution: train on all 6 (router picks XGBoost).")
    p.add_argument("--csv", type=Path, default=Path("data/raw"))
    p.add_argument("--graph-dir", type=Path, default=Path("data/graphs"))
    p.add_argument("--epochs", type=int, default=30,
                   help="GNN training epochs. Lower = faster demo, higher = more accurate.")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    return p.parse_args()


# --- Helpers -----------------------------------------------------------------

def _load_all_samples(csv_arg: Path, graph_dir: Path) -> List[Sample]:
    files = sorted(csv_arg.glob("*.csv")) if csv_arg.is_dir() else [csv_arg]
    out: List[Sample] = []
    for f in files:
        out.extend(load_samples_from_csv(f, graph_dir))
    return out


def _select_train_fold(samples: Sequence[Sample], queried_model: str, regime: str) -> List[Sample]:
    if regime == "leave-out":
        return [s for s in samples if s.model_name != queried_model]
    return list(samples)  # in-distribution: train on everything


def _build_query_sample(samples: Sequence[Sample], model: str, batch: int, seq: int, gpu: str) -> Sample:
    """Construct a Sample for the user's query, sourcing graph from the dataset
    and hardware from the registry. We do NOT need a true latency value — the
    field is only used as a placeholder for the GNN forward pass."""
    graph = next((s.graph for s in samples if s.model_name == model), None)
    if graph is None:
        raise SystemExit(
            f"error: model '{model}' has no extracted graph. "
            f"Available: {sorted({s.model_name for s in samples})}"
        )

    hardware = HARDWARE_REGISTRY.get(gpu.lower())
    if hardware is None:
        raise SystemExit(
            f"error: gpu '{gpu}' not in HARDWARE_REGISTRY. "
            f"Available: {sorted(HARDWARE_REGISTRY.keys())}"
        )

    return Sample(
        graph=graph,
        config=InferenceConfig(batch_size=batch, seq_len=seq),
        hardware=hardware,
        latency_ms=0.0,
        model_name=model,
    )


# --- Main --------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    print(f"[Loading] samples from {args.csv} + graphs from {args.graph_dir}")
    samples = _load_all_samples(args.csv, args.graph_dir)
    print(f"  → {len(samples)} samples, "
          f"{len({s.model_name for s in samples})} models, "
          f"{len({s.hardware.name for s in samples})} GPUs")

    train_fold = _select_train_fold(samples, args.model, args.regime)
    print(f"[Regime] {args.regime} → train fold has "
          f"{len({s.model_name for s in train_fold})} models, "
          f"{len(train_fold)} samples")

    query = _build_query_sample(samples, args.model, args.batch, args.seq, args.gpu)
    print(f"[Query]  model={args.model}  batch={args.batch}  seq={args.seq}  "
          f"gpu={query.hardware.name}")

    # Two-tier router: tier 1 = architecture identity, tier 2 = SHAP-driven
    # feature OOD on (log1p(total_flops), log1p(total_memory_bytes)).
    router = Router.fit(train_fold)
    result = router.route_one(query)
    print(f"[Router] tier:     {result.tier.value}")
    print(f"[Router] decision: {result.decision.value.upper()}")
    print(f"[Router] reason:   {result.reason}")

    # Train both models and predict (GNN is the slow part).
    print(f"[Train]  XGBoost + GNN (epochs={args.epochs}, device={args.device}) ...")
    xgb = XGBoostBaseline().fit(train_fold)
    xgb_pred = float(xgb.predict([query])[0])

    cfg = TrainConfig(epochs=args.epochs, batch_size=32, lr=1e-3,
                      ranking_lambda=0.1, device=args.device)
    import torch
    torch.manual_seed(0)
    gnn_model = CostModel()
    train(gnn_model, LatencyDataset(list(train_fold)), cfg)
    gnn_pred_arr, _ = predict(gnn_model, LatencyDataset([query]), device=cfg.device)
    gnn_pred = float(gnn_pred_arr[0])

    is_xgb = result.decision.value == "xgboost"
    chosen, chosen_pred = ("XGBoost", xgb_pred) if is_xgb else ("GNN", gnn_pred)
    other, other_pred = ("GNN", gnn_pred) if is_xgb else ("XGBoost", xgb_pred)

    print()
    print("─" * 56)
    print(f"  Predicted latency:  {chosen_pred:7.2f} ms   (routed: {chosen})")
    print(f"  Reference:          {other_pred:7.2f} ms   (other: {other})")
    print("─" * 56)
    if args.regime == "leave-out":
        print("  Caveat: GNN prediction is in the architecture-extrapolation")
        print("          regime (unseen graph). Empirical MAPE on this regime")
        print("          is 20-33% (vs XGBoost 39-126%) — see Table 1.")
    else:
        print("  Note: queried architecture was in the training fold;")
        print("        XGBoost dominates this regime (4-14% MAPE on RQ2 splits).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
