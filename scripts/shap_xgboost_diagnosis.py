"""SHAP-based diagnosis of why XGBoost fails on architecture extrapolation.

Generates two figures for the poster:

  1. ``results/shap_importance.png`` — global SHAP feature importance bar
     plot from XGBoost fitted on a leave-model-out training fold. The
     dominant feature is expected to be a graph aggregate magnitude
     (``log1p(total_flops)``).

  2. ``results/ood_distribution.png`` — histograms of the dominant
     feature on the train fold vs the held-out test fold, showing that
     the test distribution falls *outside* the training range. This is
     the mechanism behind the 39-126% MAPE blow-up: tree models cannot
     extrapolate beyond cell boundaries.

Run on the most extreme split (``leave-model=bert-base``, where XGBoost
hits 125.9% MAPE) by default. Override with ``--held-model``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from hetero_cost_model.baselines import (
    XGBoostBaseline,
    samples_to_feature_matrix,
)
from hetero_cost_model.data import Sample, load_samples_from_csv


# Feature names — must match the order in baselines.sample_to_global_features.
FEATURE_NAMES = [
    "log1p(total_flops)",
    "log1p(total_memory_bytes)",
    "num_nodes",
    "num_edges",
    "batch_size",
    "seq_len",
    "fp16_tflops",
    "memory_gb",
    "bandwidth_gbs",
    "l2_cache_mb",
    "sm_count",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=Path("data/raw"))
    p.add_argument("--graph-dir", type=Path, default=Path("data/graphs"))
    p.add_argument("--held-model", default="bert-base",
                   help="Model to leave out (default: bert-base — XGBoost's worst case)")
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    return p.parse_args()


def _load_all_samples(csv_arg: Path, graph_dir: Path) -> List[Sample]:
    files = sorted(csv_arg.glob("*.csv")) if csv_arg.is_dir() else [csv_arg]
    out: List[Sample] = []
    for f in files:
        out.extend(load_samples_from_csv(f, graph_dir))
    return out


def _plot_feature_importance(shap_values: np.ndarray, out_path: Path) -> None:
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    sorted_names = [FEATURE_NAMES[i] for i in order]
    sorted_vals = mean_abs[order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(range(len(sorted_names)), sorted_vals, color="#4C78A8")
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| (impact on XGBoost prediction, log-ms scale)")
    ax.set_title("XGBoost feature importance — dominated by graph aggregates")
    for bar, val in zip(bars, sorted_vals):
        ax.text(val, bar.get_y() + bar.get_height() / 2, f" {val:.2f}",
                va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → wrote {out_path}")


def _plot_ood_distribution(
    train_X: np.ndarray, test_X: np.ndarray,
    feature_idx: int, feature_name: str, held_model: str,
    out_path: Path,
) -> None:
    train_vals = train_X[:, feature_idx]
    test_vals = test_X[:, feature_idx]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(train_vals, bins=30, alpha=0.6, label=f"Train ({len(train_vals)} samples)",
            color="#4C78A8")
    ax.hist(test_vals, bins=30, alpha=0.6, label=f"Test (held-out: {held_model})",
            color="#F58518")

    train_lo, train_hi = train_vals.min(), train_vals.max()
    ax.axvline(train_lo, color="#4C78A8", ls="--", lw=1)
    ax.axvline(train_hi, color="#4C78A8", ls="--", lw=1)
    ax.axvspan(train_lo, train_hi, alpha=0.05, color="#4C78A8")

    ax.set_xlabel(feature_name)
    ax.set_ylabel("Count")
    ax.set_title("Train vs held-out distribution of dominant feature\n"
                 "(test values often fall outside training range — tree extrapolation breaks)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → wrote {out_path}")


def main() -> int:
    import shap
    args = parse_args()

    print(f"[Loading] samples from {args.csv}")
    samples = _load_all_samples(args.csv, args.graph_dir)
    print(f"  → {len(samples)} samples, {len({s.model_name for s in samples})} models")

    train_samples = [s for s in samples if s.model_name != args.held_model]
    test_samples = [s for s in samples if s.model_name == args.held_model]
    if not test_samples:
        raise SystemExit(f"error: no samples found for --held-model={args.held_model}")
    print(f"[Split] leave-model={args.held_model}: "
          f"train={len(train_samples)}, test={len(test_samples)}")

    print("[Train] XGBoost on train fold ...")
    xgb = XGBoostBaseline().fit(train_samples)

    print("[SHAP] computing TreeExplainer SHAP values ...")
    explainer = shap.TreeExplainer(xgb.model)
    train_X = samples_to_feature_matrix(train_samples)
    test_X = samples_to_feature_matrix(test_samples)
    shap_values = explainer.shap_values(train_X)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    importance_path = args.out_dir / "shap_importance.png"
    _plot_feature_importance(shap_values, importance_path)

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    print("[Top features by mean |SHAP|]")
    for rank, idx in enumerate(order[:5]):
        tr_lo, tr_hi = train_X[:, idx].min(), train_X[:, idx].max()
        te_lo, te_hi = test_X[:, idx].min(), test_X[:, idx].max()
        outside = ((test_X[:, idx] < tr_lo) | (test_X[:, idx] > tr_hi)).sum()
        pct = 100.0 * outside / len(test_X)
        marker = "  ← OOD" if pct > 0 else ""
        print(f"  #{rank+1}  {FEATURE_NAMES[idx]:<25}  mean|SHAP|={mean_abs[idx]:7.3f}  "
              f"train=[{tr_lo:.2g},{tr_hi:.2g}]  test=[{te_lo:.2g},{te_hi:.2g}]  "
              f"OOD={pct:.0f}%{marker}")

    # For the OOD plot, prefer the most-important feature that's actually OOD.
    # If none of the top features are OOD, fall back to the overall dominant
    # feature (still informative — the SHAP plot shows the same).
    dominant_idx = int(order[0])
    for idx in order[:5]:
        tr_lo, tr_hi = train_X[:, idx].min(), train_X[:, idx].max()
        outside = ((test_X[:, idx] < tr_lo) | (test_X[:, idx] > tr_hi)).sum()
        if outside > 0:
            dominant_idx = int(idx)
            break
    dominant_name = FEATURE_NAMES[dominant_idx]
    print(f"[Plot] OOD distribution chart will use: {dominant_name}")

    ood_path = args.out_dir / "ood_distribution.png"
    _plot_ood_distribution(
        train_X, test_X, dominant_idx, dominant_name, args.held_model, ood_path,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
