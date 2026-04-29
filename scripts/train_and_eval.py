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

from hetero_cost_model.baselines import (
    PerGraphMeanBaseline,
    XGBoostBaseline,
    roofline_latency,
)
from hetero_cost_model.data import LatencyDataset, Sample, load_samples_from_csv
from hetero_cost_model.metrics import grouped_top_k_accuracy, mape, spearman, top_k_accuracy
from hetero_cost_model.models.gnn import CostModel
from hetero_cost_model.models.mlp import MLPCostModel
from hetero_cost_model.router import RouteResult, routed_predictions, tier_breakdown
from hetero_cost_model.training import TrainConfig, predict, train


# --- CLI ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, nargs="+", required=True,
                   help="Profiling CSV paths. Accepts one file, multiple files, or "
                        "a directory (all *.csv under it are loaded).")
    p.add_argument("--graph-dir", type=Path, default=Path("data/graphs"))
    p.add_argument("--split", default="random",
                   help="random | leave-gpu=<key> | leave-model=<name> | "
                        "zero-shot=<key> | mixed=<model1,model2,...>")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone", default="gat", choices=["gat", "transformer"])
    p.add_argument("--hidden-dim", type=int, default=64,
                   help="GNN / Per-kernel-MLP / Pooled-MLP hidden size")
    p.add_argument("--num-layers", type=int, default=2,
                   help="Number of GNN message-passing layers (GNN only)")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout rate used in GNN + ablation MLPs")
    p.add_argument("--ranking-lambda", type=float, default=0.1,
                   help="Weight of the pairwise ranking loss added to MSE")
    # --- GNN v2 architecture switches ---
    p.add_argument("--gnn-node-level-sh", type=int, default=1,
                   choices=[0, 1],
                   help="Inject s/h into node features before GAT (default 1). "
                        "Set to 0 for v1 behavior (s/h only at head).")
    p.add_argument("--gnn-readout", default="sum", choices=["sum", "mean_max"],
                   help="Graph readout: sum preserves latency's additive "
                        "structure (default). mean_max is legacy v1.")
    p.add_argument("--gnn-global-skip", type=int, default=1,
                   choices=[0, 1],
                   help="Concat Roofline-style global summary (log flops/mem/"
                        "nodes/edges) into head input (default 1).")
    p.add_argument("--ablation-global-skip", type=int, default=1,
                   choices=[0, 1],
                   help="Whether Pooled MLP / Per-kernel MLP baselines also "
                        "receive the global summary skip (default 1). "
                        "Set to 0 for a fair 'pre-v2' ablation comparison.")
    p.add_argument("--constant-h", action="store_true",
                   help="Constant-h ablation: replace each sample's hardware "
                        "vector with the train-split mean. Diagnoses whether "
                        "the h branch is learning hardware scaling or just "
                        "memorizing per-GPU offsets.")
    p.add_argument("--few-shot-samples", type=int, default=0,
                   help="If > 0, sample N rows from the test split and append "
                        "to training (for Table 2 H200/B200 few-shot rows).")
    p.add_argument("--filter-noisy", action="store_true",
                   help="Drop rows with noisy=True from BOTH train and test.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def load_samples(csv_paths: Sequence[Path], graph_dir: Path) -> List[Sample]:
    """Load and merge samples from one or more profiling CSVs.

    ``csv_paths`` can be file paths, directory paths (which are recursively
    expanded to their ``*.csv`` children), or a mix. Duplicate rows across
    files (same model × bs × sl × actual_gpu_name) are deduplicated to the
    first occurrence so that overlapping test/sanity CSVs don't over-count.
    """
    files: List[Path] = []
    for p in csv_paths:
        if p.is_dir():
            files.extend(sorted(p.glob("*.csv")))
        elif p.is_file():
            files.append(p)
        else:
            raise SystemExit(f"--csv path does not exist: {p}")
    if not files:
        raise SystemExit("no CSV files found from --csv arguments")

    samples: List[Sample] = []
    seen_keys: set = set()
    for f in files:
        loaded = load_samples_from_csv(f, graph_dir)
        for s in loaded:
            key = (s.model_name, s.config.batch_size, s.config.seq_len, s.hardware.name)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            samples.append(s)
    return samples


# --- Data transforms (pre-split or pre-training) -----------------------------

def filter_noisy(samples: List[Sample], csv_noisy_lookup: Dict[tuple, bool]) -> List[Sample]:
    """Drop samples whose CSV row had noisy=True.

    The noisy flag is not stored on :class:`Sample` (we keep Sample minimal),
    so we look it up by the sample's (model, batch, seq, gpu_name) key.
    """
    return [
        s for s in samples
        if not csv_noisy_lookup.get(
            (s.model_name, s.config.batch_size, s.config.seq_len, s.hardware.name),
            False,
        )
    ]


def _mean_hardware(samples: Sequence[Sample]) -> "Hardware":
    """Return a synthetic Hardware with each dim set to the samples' mean (raw,
    not normalized — the ``to_vector`` downstream will re-apply normalization)."""
    from hetero_cost_model.hardware import Hardware
    assert samples, "cannot compute mean hardware from empty samples"
    vs = np.array([s.hardware.to_vector(normalize=False) for s in samples], dtype=np.float64)
    mean = vs.mean(axis=0)
    return Hardware(
        name="MEAN_H",
        fp16_tflops=float(mean[0]),
        memory_gb=float(mean[1]),
        bandwidth_gbs=float(mean[2]),
        l2_cache_mb=float(mean[3]),
        sm_count=int(round(mean[4])),
    )


def apply_constant_h(samples: Sequence[Sample], mean_h: "Hardware") -> List[Sample]:
    """Replace every sample's ``hardware`` with ``mean_h``, preserving graph /
    config / latency. Used for the constant-h ablation (Table 3 row 2).

    ``mean_h`` is computed ONLY from the training split to avoid test leakage.
    """
    return [
        Sample(
            graph=s.graph, config=s.config, hardware=mean_h,
            latency_ms=s.latency_ms, model_name=s.model_name,
        )
        for s in samples
    ]


def apply_few_shot(
    train_samples: List[Sample],
    test_samples: List[Sample],
    n: int,
    seed: int,
) -> Tuple[List[Sample], List[Sample]]:
    """Move ``n`` random samples from ``test_samples`` into ``train_samples``.

    Used for Table 2 H200 / B200 few-shot experiments: the model gets ``n``
    target-hardware samples to fine-tune on; MAPE is measured on the remaining
    test_samples. Sampling is deterministic given ``seed``.
    """
    if n <= 0:
        return list(train_samples), list(test_samples)
    if n >= len(test_samples):
        raise ValueError(
            f"few_shot_samples={n} leaves empty test set (test has {len(test_samples)})"
        )
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(test_samples))
    fs_indices = set(int(i) for i in idx[:n])
    added = [test_samples[i] for i in sorted(fs_indices)]
    remaining = [s for i, s in enumerate(test_samples) if i not in fs_indices]
    return list(train_samples) + added, remaining


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

    if spec.startswith("mixed="):
        # Heterogeneous test set that exercises per-sample routing.
        #
        #   Train fold = (samples from "trained" models) × 80% of (model, batch,
        #                seq) tuples
        #   Test fold  = (samples from "held-out" models, all configs)
        #              ∪ (samples from "trained" models, the held-out 20%)
        #
        # Held-out tuples are sampled with the same RNG seed as the rest of the
        # script so the split is reproducible. The seen-vs-unseen architecture
        # mix in the test set is the whole point: it forces the router to make
        # per-sample decisions that aren't all-GNN or all-XGBoost.
        held_models_str = spec.split("=", 1)[1]
        held_models = {m.strip() for m in held_models_str.split(",") if m.strip()}
        if not held_models:
            raise ValueError("mixed=<model1,model2,...> requires at least one model")
        all_models = {s.model_name for s in samples}
        unknown = held_models - all_models
        if unknown:
            raise ValueError(f"mixed: unknown model(s): {unknown}; "
                             f"available: {sorted(all_models)}")

        unseen = [s for s in samples if s.model_name in held_models]
        trained = [s for s in samples if s.model_name not in held_models]

        # Hold out 20% of the (model, batch, seq) tuples for trained models,
        # so test set has both unseen-architecture and held-out-config samples.
        rng = np.random.default_rng(seed)
        tuples = sorted({(s.model_name, s.config.batch_size, s.config.seq_len)
                         for s in trained})
        cut = int(0.2 * len(tuples))
        held_tuple_ix = set(int(i) for i in rng.permutation(len(tuples))[:cut])
        held_tuples = {tuples[i] for i in held_tuple_ix}

        tr = [s for s in trained
              if (s.model_name, s.config.batch_size, s.config.seq_len) not in held_tuples]
        te_held_config = [s for s in trained
                          if (s.model_name, s.config.batch_size, s.config.seq_len) in held_tuples]
        te = unseen + te_held_config
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
    hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1,
    node_level_sh: bool = True, readout: str = "sum", global_skip: bool = True,
) -> Report:
    torch.manual_seed(0)
    model = CostModel(
        hidden_dim=hidden_dim, num_layers=num_layers,
        backbone=backbone, dropout=dropout,
        node_level_sh=node_level_sh, readout=readout, global_skip=global_skip,
    )
    train(model, LatencyDataset(list(train_samples)), cfg)
    pred, true = predict(model, LatencyDataset(list(test_samples)), device=cfg.device)
    return Report(f"GNN ({backbone})", pred, true)


def run_pooled_mlp(
    train_samples: Sequence[Sample], test_samples: Sequence[Sample],
    cfg: TrainConfig,
    hidden_dim: int = 256,
    global_skip: bool = True,
) -> Report:
    """Pooled-MLP baseline: mean-pool node features, then [pool | s | h] → MLP.
    Loses both node-level granularity and graph structure; serves as the
    weakest learned baseline in Table 1."""
    torch.manual_seed(0)
    model = MLPCostModel(hidden_dim=hidden_dim, global_skip=global_skip)
    train(model, LatencyDataset(list(train_samples)), cfg)
    pred, true = predict(model, LatencyDataset(list(test_samples)), device=cfg.device)
    return Report("Pooled MLP", pred, true)


def run_per_kernel_mlp(
    train_samples: Sequence[Sample], test_samples: Sequence[Sample],
    cfg: TrainConfig,
    hidden_dim: int = 64, dropout: float = 0.1,
    global_skip: bool = True,
) -> Report:
    """Per-kernel sum MLP: GNN w/o edges (Table 3 row 1).
    Each node's latency predicted independently by MLP([op|s|h]); graph
    latency = scatter-sum over nodes. Isolates the marginal value of graph
    structure vs. per-op features."""
    from hetero_cost_model.models.per_kernel_mlp import PerKernelMLPCostModel
    torch.manual_seed(0)
    model = PerKernelMLPCostModel(
        hidden_dim=hidden_dim, dropout=dropout, global_skip=global_skip,
    )
    train(model, LatencyDataset(list(train_samples)), cfg)
    pred, true = predict(model, LatencyDataset(list(test_samples)), device=cfg.device)
    return Report("Per-kernel MLP", pred, true)


def run_per_graph_mean(
    train_samples: Sequence[Sample], test_samples: Sequence[Sample],
) -> Report:
    bl = PerGraphMeanBaseline().fit(list(train_samples))
    pred = bl.predict(list(test_samples)).tolist()
    true = [s.latency_ms for s in test_samples]
    return Report("Per-graph mean", pred, true)


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


def run_router(
    train_samples: Sequence[Sample], test_samples: Sequence[Sample],
    xgb_pred: Sequence[float], gnn_pred: Sequence[float],
) -> Tuple[Report, "List[RouteResult]"]:
    """Two-tier SHAP-driven router.

    Tier 1 catches architecture extrapolation via metadata
    (``model_name`` ∉ training fold). Tier 2 catches feature-OOD on the
    SHAP-identified architecture-distinguishing features
    (``log1p(total_flops)``, ``log1p(total_memory_bytes)``). Otherwise
    defaults to XGBoost.

    Reuses XGB/GNN predictions already produced upstream — no model
    re-invocation. Returns the :class:`Report` plus the per-sample
    :class:`RouteResult` list for downstream tier breakdown.
    """
    pred_arr, decisions = routed_predictions(
        test_samples, train_samples, xgb_pred, gnn_pred,
    )
    true = [s.latency_ms for s in test_samples]
    return Report("Router", pred_arr.tolist(), true), decisions


# --- Main --------------------------------------------------------------------

def _read_noisy_flags(csv_paths: Sequence[Path]) -> Dict[tuple, bool]:
    """Read all CSVs' (model, bs, sl, actual_gpu_name) → noisy flag mapping,
    so ``filter_noisy`` can query it without re-parsing CSVs at filter time."""
    import csv as _csv
    out: Dict[tuple, bool] = {}
    for p in csv_paths:
        files = sorted(p.glob("*.csv")) if p.is_dir() else [p]
        for f in files:
            with open(f) as fh:
                for row in _csv.DictReader(fh):
                    try:
                        key = (
                            row["model_name"],
                            int(row["batch_size"]),
                            int(row["seq_len"]),
                            # Sample.hardware.name is set by HARDWARE_REGISTRY
                            # lookup (e.g. "H100"); this mirrors the key format
                            # the filter_noisy function constructs.
                            _registry_name_for(row.get("actual_gpu_name", ""),
                                               row.get("gpu", "")),
                        )
                        out[key] = row.get("noisy", "").lower() == "true"
                    except (KeyError, ValueError):
                        continue
    return out


def _registry_name_for(actual: str, declared: str) -> str:
    """Mirror Sample-building logic: map an ``actual_gpu_name`` to the
    ``Hardware.name`` used in Sample (needed for the noisy-flag key)."""
    from hetero_cost_model.hardware import HARDWARE_REGISTRY
    from hetero_cost_model.runtime_info import gpu_name_to_registry_key
    key = gpu_name_to_registry_key(actual) or declared
    hw = HARDWARE_REGISTRY.get(key)
    return hw.name if hw else ""


def main() -> int:
    args = parse_args()

    print("=" * 72)
    print(f"train_and_eval.py  split={args.split}  backbone={args.backbone}")
    print("=" * 72)

    print(f"\nLoading samples from {[str(p) for p in args.csv]}")
    print(f"   + graphs from {args.graph_dir}")
    samples = load_samples(args.csv, args.graph_dir)
    if not samples:
        print("No usable samples found. Aborting.")
        return 1
    print(f"  {len(samples)} total samples, "
          f"{len({s.model_name for s in samples})} distinct models, "
          f"{len({s.hardware.name for s in samples})} distinct GPUs")

    if args.filter_noisy:
        noisy_lookup = _read_noisy_flags(args.csv)
        before = len(samples)
        samples = filter_noisy(samples, noisy_lookup)
        print(f"  --filter-noisy: {before} → {len(samples)} samples "
              f"({before - len(samples)} rows with CoV > 5% dropped)")

    tr, te = make_split(samples, args.split, args.seed)
    print(f"  split '{args.split}': train={len(tr)}  test={len(te)}")
    if not tr or not te:
        print("empty split half. Aborting.")
        return 1

    if args.few_shot_samples > 0:
        tr_before, te_before = len(tr), len(te)
        tr, te = apply_few_shot(tr, te, args.few_shot_samples, args.seed)
        print(f"  --few-shot-samples={args.few_shot_samples}: "
              f"train {tr_before}→{len(tr)}, test {te_before}→{len(te)}")

    if args.constant_h:
        from hetero_cost_model.hardware import Hardware  # re-exposed for type
        mean_h = _mean_hardware(tr)
        tr = apply_constant_h(tr, mean_h)
        te = apply_constant_h(te, mean_h)
        print(f"  --constant-h: replaced hardware with train-split mean: "
              f"{[f'{v:.1f}' for v in mean_h.to_vector(normalize=False)]}")

    cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        ranking_lambda=args.ranking_lambda, device=args.device,
    )
    print(f"\n  hyperparams: hidden={args.hidden_dim} layers={args.num_layers} "
          f"dropout={args.dropout} ranking_lambda={args.ranking_lambda}")

    reports: List[Report] = []
    print("\nTraining/evaluating ...")
    reports.append(run_roofline(te))
    reports.append(run_per_graph_mean(tr, te))
    reports.append(run_pooled_mlp(
        tr, te, cfg, global_skip=bool(args.ablation_global_skip),
    ))
    xgb_report = run_xgboost(tr, te)
    reports.append(xgb_report)
    reports.append(run_per_kernel_mlp(
        tr, te, cfg, hidden_dim=args.hidden_dim, dropout=args.dropout,
        global_skip=bool(args.ablation_global_skip),
    ))
    gnn_report = run_gnn(
        tr, te, args.backbone, cfg,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout,
        node_level_sh=bool(args.gnn_node_level_sh),
        readout=args.gnn_readout,
        global_skip=bool(args.gnn_global_skip),
    )
    reports.append(gnn_report)
    router_report, route_decisions = run_router(tr, te, xgb_report.pred, gnn_report.pred)
    reports.append(router_report)

    print("\n" + "-" * 72)
    print(f"{'Method':<18}  {'MAPE':>8}  {'Spearman':>10}  {'Top-1':>8}")
    print("-" * 72)
    for r in reports:
        mape_str = f"{r.mape * 100:.2f}%" if not math.isnan(r.mape) else "nan"
        print(f"{r.name:<18}  {mape_str:>8}  {r.spearman:>10.3f}  {r.top1:>8.3f}")
    print("-" * 72)

    # RQ3 — Decision effectiveness: within each (model, batch, seq) group
    # the candidate set is the held-out GPUs. "GPU selection top-1" =
    # fraction of (model, batch, seq) groups where the predictor's
    # fastest GPU is also the actual fastest GPU. Operationalizes the
    # proposal's RQ3 ("rank candidate strategies and select near-optimal
    # without exhaustive profiling") at the granularity of GPU choice.
    gpu_groups = [(s.model_name, s.config.batch_size, s.config.seq_len) for s in te]
    print("\nRQ3: GPU-selection accuracy (per-(model,batch,seq) groups, "
          f"{len(set(gpu_groups))} groups):")
    print(f"  {'Method':<18}  {'top-1':>6}  {'top-2':>6}")
    for r in reports:
        t1 = grouped_top_k_accuracy(r.pred, r.true, gpu_groups, k=1)
        t2 = grouped_top_k_accuracy(r.pred, r.true, gpu_groups, k=2)
        t1s = f"{t1:.3f}" if not math.isnan(t1) else "n/a"
        t2s = f"{t2:.3f}" if not math.isnan(t2) else "n/a"
        print(f"  {r.name:<18}  {t1s:>6}  {t2s:>6}")
    print()

    # Router tier breakdown — shows how many samples each tier caught.
    counts = tier_breakdown(route_decisions)
    n = len(route_decisions)
    print("\nRouter tier breakdown (per-sample routing decisions):")
    print(f"  tier1 (architecture identity)        : "
          f"{counts['tier1']:4d}  ({100 * counts['tier1'] / n:5.1f}%) → GNN")
    print(f"  tier2 (SHAP feature OOD)             : "
          f"{counts['tier2']:4d}  ({100 * counts['tier2'] / n:5.1f}%) → GNN")
    print(f"  default (in distribution)            : "
          f"{counts['default']:4d}  ({100 * counts['default'] / n:5.1f}%) → XGBoost")

    # Per-tier MAPE — shows on each subset whether the routed model is
    # actually the better choice. Validates the "GNN-weak ↔ Router uses XGB"
    # hypothesis with concrete numbers.
    xgb_pred_arr = np.asarray(xgb_report.pred, dtype=np.float64)
    gnn_pred_arr = np.asarray(gnn_report.pred, dtype=np.float64)
    true_arr = np.asarray(xgb_report.true, dtype=np.float64)

    def _mape_subset(pred: np.ndarray, mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return float("nan")
        return float(np.mean(np.abs((pred[mask] - true_arr[mask]) / true_arr[mask])))

    tier_masks = {
        "tier1":   np.array([d.tier.value == "tier1"   for d in route_decisions]),
        "tier2":   np.array([d.tier.value == "tier2"   for d in route_decisions]),
        "default": np.array([d.tier.value == "default" for d in route_decisions]),
    }
    print("\nPer-tier MAPE breakdown (validates routing decisions):")
    print(f"  {'subset':<32}  {'n':>4}  {'XGBoost':>8}  {'GNN':>8}  {'Router uses':<12}  {'Right?':<6}")
    for tier_name, mask in tier_masks.items():
        if mask.sum() == 0:
            continue
        xgb_m = _mape_subset(xgb_pred_arr, mask)
        gnn_m = _mape_subset(gnn_pred_arr, mask)
        uses = "GNN" if tier_name in ("tier1", "tier2") else "XGBoost"
        better = "XGBoost" if xgb_m < gnn_m else "GNN"
        right_choice = "✓" if better == uses else "✗"
        label = {
            "tier1":   "tier1: unseen architecture",
            "tier2":   "tier2: SHAP feature OOD",
            "default": "default: in distribution",
        }[tier_name]
        print(f"  {label:<32}  {int(mask.sum()):>4}  "
              f"{xgb_m * 100:>7.2f}%  {gnn_m * 100:>7.2f}%  {uses:<12}  {right_choice:<6}")

    print("\nDecision rules (plan §5.2):")
    print("  GNN > Pooled MLP + Per-graph mean + Roofline  → C1 holds")
    print("  GNN − Per-kernel MLP gap ≥ 3 pts              → graph structure (C3)")
    print("  GNN − XGBoost gap ≥ 3-5 pts                   → graph > tabular features")

    return 0


if __name__ == "__main__":
    sys.exit(main())
