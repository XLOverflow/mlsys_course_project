# Methods

> Technical approach end-to-end: from "what's a data point" to "how the
> router consumes the trained models". Implementation pointers throughout.

## 1. Data

### 1.1 Sample definition

One sample = one transformer forward-pass (= LLM prefill stage) measurement, indexed by `(model_name, gpu, batch_size, seq_len)`. The cost-model triplet `f(G, s, h) → T̂` maps onto the dataset as:

| Cost-model variable | Dataset representation |
|---|---|
| G (computation graph) | extracted from `model_name` at training time via HF `fx`; not stored in CSV |
| s (inference config) | `batch_size` + `seq_len` columns |
| h (hardware vector) | looked up from `HARDWARE_REGISTRY` keyed by **`actual_gpu_name`** (the runtime-queried device name, not the dispatch label — prevents Modal SKU upgrades from polluting h) |
| T̂ (latency target) | **`p50_ms`** — robust to shared-host long-tail noise |

Decode-mode samples follow the same schema with one extra column: `mode ∈ {prefill, decode}`. They live in a separate directory (`data/raw/decode/*.csv`) so the non-recursive glob in `train_and_eval.py` keeps them physically separated from prefill data.

Implementation: [src/hetero_cost_model/data.py](../src/hetero_cost_model/data.py), [src/hetero_cost_model/profiling.py](../src/hetero_cost_model/profiling.py).

### 1.2 Final dataset (5 GPUs)

| Hardware | Family / SM | FP16 TFLOPS | HBM GB | HBM BW (GB/s) | L2 (MB) | SMs |
|---|---|---:|---:|---:|---:|---:|
| A10G | Ampere consumer (sm_86) | 70 | 24 | 600 | 6 | 80 |
| A100-SXM4-40GB | Ampere datacenter (sm_80) | 312 | 40 | 1555 | 40 | 108 |
| L4 | Ada (sm_89) | 121 | 24 | 300 | 48 | 58 |
| H100 | Hopper (sm_90) | 1979 | 80 | 3350 | 50 | 132 |
| B200 | Blackwell (sm_100) | 4500 | 180 | 8000 | 96 | 160 |

**Coverage**: 4 architecture generations (Ampere → Ada → Hopper → Blackwell), 27× HBM bandwidth range. Profiled in FP16 with `attn_implementation="eager"` for cross-arch consistency.

| Mode | Models | Configs | Total samples |
|---|---|---|---:|
| Prefill | 6 (gpt2-{small,medium,large}, bert-{base,large}, t5-small) | 8 batches × 6 seqs = 48 | **1360** |
| Decode | 3 (gpt2 family only — BERT no autoregressive decode, T5 enc-dec TODO) | 8 batches × 5 seqs = 40 (skip seq=1024 for next-token headroom) | **600** |

V100 is registered in `HARDWARE_REGISTRY` as a planned anchor but never profiled; PSC allocation never started, and Modal does not provide V100. The choice does not affect the gap-bandwidth findings in [results/rq2.md](results/rq2.md): V100 (900 GB/s) would fall between A10 and A100, not between A100 (1555) and B200 (8000) where XGBoost's interpolation actually fails.

### 1.3 Hardware feature vector (5-dim)

```
h = [fp16_tflops, memory_gb, bandwidth_gbs, l2_cache_mb, sm_count] / normalization
```

All on-chip specs. **No interconnect bandwidth** (PCIe/NVLink): irrelevant during a single-GPU timed forward pass and would inject confounded signal. **No `arch_gen` ordinal**: would behave as a disguised device-ID lookup, undermining the "spec-only zero-shot" claim. Aligns with NeuSight (ASPLOS'25) on-chip-only convention.

Implementation: [src/hetero_cost_model/hardware.py](../src/hetero_cost_model/hardware.py).

### 1.4 Profiling protocol

- **Warmup 50 + timed 100 runs** per config (CUDA events; `time.perf_counter` only as a fallback when CUDA is unavailable).
- Training target = **`p50_ms`** (median); `mean_ms` retained for noise diagnostics.
- **Modal SKU lock** (`H100!` exact-match syntax) prevents auto-upgrade. CSV records `actual_gpu_name` as a runtime double-check.
- **Resume-on-preempt** via Modal Volume + idempotent CSV writer that skips rows already present (keys: `(model, batch, seq, gpu, mode)`).

Implementation: [scripts/run_profiling.py](../scripts/run_profiling.py), [scripts/modal_profiling.py](../scripts/modal_profiling.py).

## 2. Models

### 2.1 GNN v2 architecture

```
Input: graph G (per-node features + edges) + s (batch, seq) + h (5-dim)

  ↓  per-node concat: x' = [x | broadcast(s) | broadcast(h)]   ← Switch 1: node_level_sh
  ↓  3-layer GAT (hidden=64, 4 heads, dropout=0.1)
  ↓  global_add_pool                                            ← Switch 2: sum readout
  ↓  concat: [pool | s | h | g]                                 ← Switch 3: global_skip
  ↓        where g = [log1p(total_flops), log1p(total_memory),
  ↓                   log1p(num_nodes), log1p(num_edges)]
  ↓  2-layer MLP head → ŷ ∈ ℝ
```

**Why these 3 switches**:

1. **`node_level_sh`** — gives each GAT layer access to the inference config and hardware spec, so message passing can learn hardware-dependent op behavior (cf. Akhauri & Abdelfattah 2024).
2. **`sum readout`** — prefill latency is *additive over ops* (kernels execute sequentially), so sum-pool preserves totals; mean-pool would normalize them away and lose the "gpt2-large has 3× more ops than gpt2-small" signal.
3. **`global_skip`** — gives the head a direct view of the Roofline-style aggregates that XGBoost gets, so the GNN only has to learn the *residual* on top of "physics".

The three switches **synergize**: single-on configurations are far worse than full-on (e.g. only-sum readout = 350% MAPE, full v2 = 25.6%). Ablation in [results/rq1.md](results/rq1.md#v2-three-switch-ablation).

Implementation: [src/hetero_cost_model/models/gnn.py](../src/hetero_cost_model/models/gnn.py).

### 2.2 Baselines

| Baseline | Input | Role |
|---|---|---|
| **Roofline** | per-op FLOPs/bytes + hardware peak | Analytical reference (no training). Proves learned models do better than physics. |
| **Per-graph mean** | (model, gpu) lookup | Data-leakage diagnostic. If GNN doesn't beat this, message passing is decoration. |
| **Pooled MLP** | mean-pool(nodes) ‖ s ‖ h | Weakest learned baseline. Tests whether per-node info is needed. |
| **XGBoost** | 12 hand-crafted globals: log1p(total_flops/bytes), num_nodes/edges, batch, seq, h × 5 | **Strong baseline.** Boosting framework is the same Chen Tianqi line that lives inside Ansor's cost model. |
| **Per-kernel MLP** | per-node MLP + scatter-sum (NeuSight-style; no edges) | GNN ablation. Isolates the contribution of message-passing edges. |
| **GNN (GAT v2)** | node features + edges + s + h + global skip | **Main method.** |
| **Router** | metadata + per-sample SHAP-OOD check | System-level integration over (XGBoost, GNN). Details in [results/router.md](results/router.md). |

Implementation: [src/hetero_cost_model/baselines.py](../src/hetero_cost_model/baselines.py), [src/hetero_cost_model/models/](../src/hetero_cost_model/models/), [src/hetero_cost_model/router.py](../src/hetero_cost_model/router.py).

## 3. Training & evaluation pipeline

### 3.1 Splits

| Split | Train | Test | Tests what |
|---|---|---|---|
| `random` | 80% | 20% | In-distribution sanity (used as RQ3 evaluation context) |
| `leave-model=X` | 5 models × all GPUs | model X × all GPUs | RQ1 — workload generalization |
| `leave-gpu=X` | All models × 4 other GPUs | All models × GPU X | RQ2 — hardware generalization |
| `mixed=X,Y` | (4 other models × 80% configs) × 5 GPUs | (X, Y all configs ∪ 4-models × 20% configs) × 5 GPUs | Heterogeneous test set for Router showcase |

Implementation: [scripts/train_and_eval.py](../scripts/train_and_eval.py) `make_split()`.

### 3.2 Hyperparameters (locked across all experiments)

```
hidden_dim = 64
num_layers = 2
heads = 4
dropout = 0.1
ranking_lambda = 0.1
batch_size = 32
lr = 1e-3
seed = 0
v2 switches = all on (node_level_sh + sum + global_skip)
backbone = gat
epochs = 50 (leave-model-out, leave-gpu-out) / 30 (mixed, decode)
```

Modal H100 is used for re-runs after the 5-GPU scope finalization (each `--split leave-X=Y` finishes in ~2 min on H100). Decode leave-model-out runs on CPU (~1 min per split, 600-sample dataset).

### 3.3 Metrics

| Metric | Granularity | Use |
|---|---|---|
| MAPE | Sample-level | Main accuracy headline |
| Spearman ρ | Sample-level | Rank quality |
| `top_k_accuracy` (global) | Test set as one group | Original metric; degenerate on multi-fold splits — kept for backward compatibility |
| **`grouped_top_k_accuracy`** | Per-group (e.g. `(model, batch, seq)` → candidate GPUs) | **RQ3 operationalization.** Within-group "did we pick the actual fastest?" |
| NDCG@k | Per-sample | Ranking quality with positional discount |

Implementation: [src/hetero_cost_model/metrics.py](../src/hetero_cost_model/metrics.py).

### 3.4 Router architecture

Two-tier rule-based router with SHAP-driven Tier 2 feature selection. Predicts XGBoost or GNN per sample at evaluation time:

```python
def route_one(sample, train):
    # Tier 1: architecture identity (cheap metadata check)
    if sample.model_name not in train.model_names:
        return ("gnn", reason="architecture not in training fold")

    # Tier 2: SHAP-driven feature OOD on architecture-distinguishing features
    # Features chosen via SHAP analysis on XGBoost (see results/shap.md):
    #   ARCHITECTURE_FEATURES = ("log1p(total_flops)", "log1p(total_memory_bytes)")
    for feat in ARCHITECTURE_FEATURES:
        if sample[feat] outside train.feature_ranges[feat]:
            return ("gnn", reason=f"{feat} OOD")

    return ("xgboost", reason="in distribution")
```

The "complexity" of this router lives **outside the rule itself** — it's the SHAP analysis on the strong baseline (XGBoost) that selects which features Tier 2 checks. We don't pick features by hand; SHAP picks them by ranking which features XGBoost actually uses to distinguish architectures.

Detailed design + mixed-split results: [results/router.md](results/router.md). SHAP analysis: [results/shap.md](results/shap.md).
