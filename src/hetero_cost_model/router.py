"""SHAP-driven two-tier router that picks XGBoost or GNN per sample.

The router operationalizes the project's central finding: tabular and
graph-aware cost models occupy *different* extrapolation regimes.

  * **Hardware extrapolation** (unseen GPU; same architectures): tabular
    methods (XGBoost) interpolate across the smooth (FLOPs × bandwidth)
    manifold and dominate (4.4–14% MAPE).
  * **Architecture extrapolation** (unseen graph): aggregate features
    extrapolate poorly outside the training range, so XGBoost fails
    (39–126% MAPE) while a graph-aware GNN holds at 20–33%.

Routing logic
-------------

The router runs two checks in order; first match wins. Both tiers route
to GNN when triggered (architecture extrapolation regime); otherwise
the sample defaults to XGBoost.

  **Tier 1 — architecture identity (metadata)**
      If ``sample.model_name`` is not in the training fold, the
      architecture is unseen — the canonical RQ1 case. Cheapest possible
      check; available at deployment time without any feature
      computation.

  **Tier 2 — feature-level OOD (SHAP-driven)**
      Even when the architecture is "seen", a sample's graph-aggregate
      features may still fall outside the training range (e.g. a
      fine-tuned variant with different op counts). We check exactly the
      architecture-distinguishing features that XGBoost relies on
      according to SHAP analysis (see ``ARCHITECTURE_FEATURES``).

      *Why these features?* Running SHAP on XGBoost over a
      leave-model-out fold (see ``scripts/shap_xgboost_diagnosis.py``)
      reveals that ``log1p(total_flops)`` and
      ``log1p(total_memory_bytes)`` are the architecture-distinguishing
      features XGBoost actually uses. The other top-SHAP features
      (``seq_len``, ``batch_size``, hardware specs) are *not*
      architecture-distinguishing, so OOD on them does not signal
      architecture extrapolation. The choice of which features to check
      in Tier 2 is therefore *driven by the SHAP analysis on the
      tabular baseline*, not picked by hand.

  **Default — XGBoost**
      Sample is in distribution on every check. XGBoost is the strong
      baseline for in-distribution + hardware extrapolation regimes.

The two tiers are complementary: Tier 1 is deployment-cheap; Tier 2 is
robust to subtle within-architecture drift. On our experimental splits,
graph-aggregate features are graph-level constants, so Tier 2 fires
*iff* Tier 1 fires — but their independent agreement is itself
validation: the SHAP-identified OOD signal coincides with the metadata
notion of "unseen architecture", supporting the design.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from hetero_cost_model.data import Sample


# Feature names matching baselines.sample_to_global_features. Kept in
# sync with that function — adding a feature there requires an entry
# here.
FEATURE_NAMES: Tuple[str, ...] = (
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
)

# Architecture-distinguishing features that drive XGBoost's predictions
# on unseen graphs (per SHAP analysis on leave-model=gpt2-large; see
# results/SHAP_FINDINGS.md). Tier 2 OOD detection runs over exactly
# these features. ``num_nodes`` / ``num_edges`` are technically
# architecture-distinguishing too, but SHAP shows XGBoost effectively
# never uses them (mean |SHAP| ≈ 0), so OOD on them does not predict
# XGBoost failure and we skip them.
ARCHITECTURE_FEATURES: Tuple[str, ...] = (
    "log1p(total_flops)",
    "log1p(total_memory_bytes)",
)


class RouteDecision(str, Enum):
    XGBOOST = "xgboost"
    GNN = "gnn"


class RouteTier(str, Enum):
    TIER1_ARCHITECTURE_IDENTITY = "tier1"
    TIER2_FEATURE_OOD = "tier2"
    DEFAULT_XGBOOST = "default"


@dataclass(frozen=True)
class RouteResult:
    decision: RouteDecision
    tier: RouteTier
    reason: str


def _arch_features(sample: Sample) -> Dict[str, float]:
    """Compute the architecture-distinguishing features for one sample.

    Mirrors the relevant slice of
    :func:`hetero_cost_model.baselines.sample_to_global_features`. We do
    *not* call that function directly to avoid a circular import and to
    document explicitly which features the router consults.
    """
    g = sample.graph
    return {
        "log1p(total_flops)": math.log1p(g.total_flops()),
        "log1p(total_memory_bytes)": math.log1p(g.total_memory()),
    }


@dataclass(frozen=True)
class Router:
    """Two-tier OOD-aware router. Cheap to build (one pass over the
    training fold), cheap to query (per-sample, no model invocation
    needed for the routing decision itself)."""

    train_model_names: Set[str]
    arch_feature_ranges: Dict[str, Tuple[float, float]]

    @classmethod
    def fit(cls, train_samples: Sequence[Sample]) -> "Router":
        if not train_samples:
            raise ValueError("Router.fit: empty training fold")

        per_feature: Dict[str, List[float]] = {f: [] for f in ARCHITECTURE_FEATURES}
        for s in train_samples:
            af = _arch_features(s)
            for f in ARCHITECTURE_FEATURES:
                per_feature[f].append(af[f])

        ranges = {
            f: (float(min(vals)), float(max(vals)))
            for f, vals in per_feature.items()
        }
        return cls(
            train_model_names={s.model_name for s in train_samples},
            arch_feature_ranges=ranges,
        )

    def route_one(self, sample: Sample) -> RouteResult:
        # --- Tier 1: architecture identity ---------------------------------
        if sample.model_name not in self.train_model_names:
            return RouteResult(
                decision=RouteDecision.GNN,
                tier=RouteTier.TIER1_ARCHITECTURE_IDENTITY,
                reason=f"architecture '{sample.model_name}' not in training fold",
            )

        # --- Tier 2: SHAP-driven feature OOD --------------------------------
        af = _arch_features(sample)
        for feat in ARCHITECTURE_FEATURES:
            val = af[feat]
            lo, hi = self.arch_feature_ranges[feat]
            if val < lo or val > hi:
                return RouteResult(
                    decision=RouteDecision.GNN,
                    tier=RouteTier.TIER2_FEATURE_OOD,
                    reason=(f"{feat}={val:.3f} outside training range "
                            f"[{lo:.3f}, {hi:.3f}]"),
                )

        # --- Default: in-distribution → XGBoost -----------------------------
        return RouteResult(
            decision=RouteDecision.XGBOOST,
            tier=RouteTier.DEFAULT_XGBOOST,
            reason="in distribution on architecture identity and SHAP-driven features",
        )

    def route(self, samples: Sequence[Sample]) -> List[RouteResult]:
        return [self.route_one(s) for s in samples]


def routed_predictions(
    test_samples: Sequence[Sample],
    train_samples: Sequence[Sample],
    xgb_pred: Sequence[float],
    gnn_pred: Sequence[float],
) -> Tuple[np.ndarray, List[RouteResult]]:
    """Combine XGB and GNN predictions per sample using :class:`Router`.

    Returns
    -------
    predictions
        Routed prediction vector aligned with ``test_samples``.
    decisions
        Per-sample :class:`RouteResult` (decision + tier + reason),
        useful for breakdown statistics and demo output.
    """
    if len(xgb_pred) != len(test_samples) or len(gnn_pred) != len(test_samples):
        raise ValueError(
            f"prediction length mismatch: xgb={len(xgb_pred)} gnn={len(gnn_pred)} "
            f"test={len(test_samples)}"
        )

    router = Router.fit(train_samples)
    decisions = router.route(test_samples)

    out = np.empty(len(test_samples), dtype=np.float32)
    for i, d in enumerate(decisions):
        out[i] = gnn_pred[i] if d.decision == RouteDecision.GNN else xgb_pred[i]
    return out, decisions


def tier_breakdown(decisions: Sequence[RouteResult]) -> Dict[str, int]:
    """Count routing decisions by tier — used in train_and_eval reports."""
    counts: Dict[str, int] = {t.value: 0 for t in RouteTier}
    for d in decisions:
        counts[d.tier.value] += 1
    return counts


__all__ = [
    "ARCHITECTURE_FEATURES",
    "FEATURE_NAMES",
    "RouteDecision",
    "RouteTier",
    "RouteResult",
    "Router",
    "routed_predictions",
    "tier_breakdown",
]
