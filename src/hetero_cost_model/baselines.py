"""Non-GNN baselines for cost-model comparison.

Three reference methods, each playing a distinct role:

1. ``roofline_latency`` — pure analytical formula, no training. Proves the
   GNN learns something beyond peak compute/bandwidth bounds.
2. ``PerGraphMeanBaseline`` — data-leakage diagnostic. Categorical two-way
   mean (per-model + per-GPU offset) with **no access to graph structure,
   per-op features, or (batch, seq)**. If the GNN's leave-one-GPU-out MAPE
   doesn't beat this, the GNN is effectively a (model_id, gpu_id) lookup
   and graph structure / hardware features contribute nothing. Mandatory
   companion for any generalization claim.
3. ``XGBoostBaseline`` — gradient-boosted trees on hand-crafted global
   graph + config + hardware features. This is the **strong** baseline the
   GNN has to beat (target: 3–5 MAPE points gap); if it doesn't, the graph
   structure isn't adding value.

Features used by XGBoost (``NUM_GLOBAL_FEATURES`` total):

  - Graph-level: log1p(total_flops), log1p(total_memory_bytes), num_nodes,
    num_edges
  - Config:      batch_size, seq_len
  - Hardware:    the full 5-dim ``HARDWARE_FEATURE_DIM`` vector (normalized)

log1p is used for the graph-level magnitudes because they span ~3 orders of
magnitude across the 6 target models.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from hetero_cost_model.data import Sample
from hetero_cost_model.graph import GraphRepr
from hetero_cost_model.hardware import HARDWARE_FEATURE_DIM, Hardware
from hetero_cost_model.strategies import InferenceConfig


NUM_GLOBAL_FEATURES: int = 4 + 2 + HARDWARE_FEATURE_DIM


# --- Analytical: Roofline ----------------------------------------------------

def roofline_latency(
    graph: GraphRepr,
    config: InferenceConfig,
    hardware: Hardware,
) -> float:
    """GPU roofline latency estimate in milliseconds.

    Each op is bounded by either compute (FLOPs / peak_TFLOPS) or memory
    bandwidth (bytes / peak_BW). FLOPs are scaled by batch_size since the
    graph stores per-token/per-sample estimates.
    """
    peak_flops = hardware.fp16_tflops * 1e12   # ops/s
    peak_bw    = hardware.bandwidth_gbs * 1e9  # bytes/s

    total_ms = 0.0
    for node in graph.nodes:
        scaled_flops = node.flops * config.batch_size
        compute_ms  = scaled_flops / peak_flops * 1000.0 if peak_flops else 0.0
        memory_ms   = node.memory_bytes / peak_bw * 1000.0 if peak_bw else 0.0
        total_ms += max(compute_ms, memory_ms)
    return total_ms


# --- Diagnostic: Per-graph mean + per-GPU offset -----------------------------

@dataclass
class PerGraphMeanBaseline:
    """Categorical two-way mean: per-model intercept + per-GPU residual offset.

    No graph structure, no per-op features, no (batch, seq). Purely a lookup
    on (``model_name``, ``hardware.name``). The point is to force the GNN to
    prove it learned something beyond that lookup.

    Prediction rule::

        ŷ(m, g) = graph_mean[m] + gpu_offset[g]

    For held-out GPUs (leave-one-GPU-out, B200 zero-shot), ``gpu_offset``
    falls back to 0 — representing the "no information about this device"
    case. This is the correct handicap because the whole point of the
    hardware feature vector is to carry that information in the GNN.
    """

    graph_mean: Dict[str, float] = field(default_factory=dict)
    gpu_offset: Dict[str, float] = field(default_factory=dict)
    overall_mean: float = 0.0

    def fit(self, samples: Sequence[Sample]) -> "PerGraphMeanBaseline":
        if not samples:
            raise ValueError("cannot fit on empty sample list")

        self.overall_mean = float(np.mean([s.latency_ms for s in samples]))

        # Stage 1: per-model mean latency
        by_model: Dict[str, List[float]] = defaultdict(list)
        for s in samples:
            by_model[s.model_name].append(s.latency_ms)
        self.graph_mean = {m: float(np.mean(v)) for m, v in by_model.items()}

        # Stage 2: per-GPU residual offset
        by_gpu: Dict[str, List[float]] = defaultdict(list)
        for s in samples:
            residual = s.latency_ms - self.graph_mean[s.model_name]
            by_gpu[s.hardware.name].append(residual)
        self.gpu_offset = {g: float(np.mean(v)) for g, v in by_gpu.items()}

        return self

    def predict(self, samples: Sequence[Sample]) -> np.ndarray:
        out: List[float] = []
        for s in samples:
            base = self.graph_mean.get(s.model_name, self.overall_mean)
            offset = self.gpu_offset.get(s.hardware.name, 0.0)
            out.append(base + offset)
        return np.array(out, dtype=np.float32)


# --- Learned: XGBoost on global features -------------------------------------

def sample_to_global_features(sample: Sample) -> List[float]:
    """Project a ``Sample`` to a flat feature vector for tabular regressors."""
    g = sample.graph
    feats: List[float] = [
        math.log1p(g.total_flops()),
        math.log1p(g.total_memory()),
        float(g.num_nodes()),
        float(len(g.edges)),
        float(sample.config.batch_size),
        float(sample.config.seq_len),
    ]
    feats.extend(sample.hardware.to_vector(normalize=True))
    assert len(feats) == NUM_GLOBAL_FEATURES
    return feats


def samples_to_feature_matrix(samples: Sequence[Sample]) -> np.ndarray:
    return np.array([sample_to_global_features(s) for s in samples], dtype=np.float32)


def samples_to_targets(samples: Sequence[Sample]) -> np.ndarray:
    return np.array([s.latency_ms for s in samples], dtype=np.float32)


@dataclass
class XGBoostBaseline:
    """Gradient-boosted trees on flat features; the strong baseline for GNN.

    Thin wrapper over ``xgboost.XGBRegressor`` with project-appropriate
    defaults and the ``(graph + config + hardware) → latency`` feature map
    pinned in this module so every comparison uses the same features.
    """

    n_estimators: int = 400
    max_depth: int = 6
    learning_rate: float = 0.05
    min_child_weight: float = 1.0
    subsample: float = 0.9
    reg_lambda: float = 1.0
    random_state: int = 0

    model: Optional["xgboost.XGBRegressor"] = None   # type: ignore[name-defined]

    def fit(self, samples: Sequence[Sample]) -> "XGBoostBaseline":
        import xgboost
        X = samples_to_feature_matrix(samples)
        y = samples_to_targets(samples)
        self.model = xgboost.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )
        self.model.fit(X, y)
        return self

    def predict(self, samples: Sequence[Sample]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call .fit() before .predict()")
        X = samples_to_feature_matrix(samples)
        return self.model.predict(X)


__all__ = [
    "roofline_latency",
    "PerGraphMeanBaseline",
    "XGBoostBaseline",
    "sample_to_global_features",
    "samples_to_feature_matrix",
    "samples_to_targets",
    "NUM_GLOBAL_FEATURES",
]
