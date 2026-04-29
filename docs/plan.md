# Project Plan & Status

**Project**: Generalized Learned Cost Model for LLM Inference Optimization
**Team**: Xiang Li (xli8) | Xinhao Tan (xinhaota) | Zhikai Hu (zhikaih)
**Course**: CMU 15-442 MLSys, poster final project
**Period**: 2026-04-14 → 2026-04-30 (poster deadline)
**Hardware**: A10 / A100-40GB / L4 / H100 / B200 (5 GPUs on Modal, spanning 4 architecture generations)

## 0. Status (2026-04-29, day before poster)

| Phase | Status | Note |
|---|---|---|
| Day 1-2 Infrastructure | ✅ | pytest 53/53, 6 models fx-tracing PASS, end-to-end smoke 8/8 |
| Day 3-4 Data collection | ✅ | Prefill **1360 samples** (`data/raw/*.csv`); Decode **600 samples** (`data/raw/decode/*.csv`); FP16 eager mode |
| Day 5-7 RQ1 main table (architecture extrapolation) | ✅ | 6/6 prefill + 3/3 decode leave-model-out → GNN beats XGBoost on every split |
| Day 8-9 RQ2 (hardware extrapolation) | ✅ | 5 leave-gpu-out splits; GNN wins 2/5 (a100 + h100), mean MAPE 34.6% vs XGBoost 40.1% |
| Day 10-11 v2 ablation | ✅ | Three switches synergize (single-on far worse than full-on) |
| Day 12 Router + SHAP + Demo | ✅ | Two-tier SHAP-driven router; mixed-split Router strictly beats both single models |
| Day 13 RQ3 (decision effectiveness) | ✅ | Grouped GPU-selection top-K integrated; XGBoost top-1 88.9%, top-2 100% on random split |
| Day 14 Decode exploration | ✅ | `decode-exploration` branch; GNN MAPE 8.9% — 2.5× cleaner than prefill |
| Day 15 Poster build | 🔴 in progress | Hero claim, related work, §Limitations, screenshots |

### Claim status (5-GPU final)

| Claim | Status | Evidence |
|---|---|---|
| **RQ1 Workload generalization** | ✅ | Prefill GNN 28.1% mean / XGBoost 70.9% (6/6 wins). Decode GNN 8.9% / XGBoost 58.2% (3/3 wins, 2.5× better than prefill). |
| **RQ2 Hardware generalization** | ⚠️ partial | GNN wins 2/5 splits (a100, h100); XGBoost wins 3/5 (a10, b200, l4). Mean GNN 34.6%, XGBoost 40.1%. **New finding** — sparse hardware coverage exposes tabular tree-interpolation fragility (h100 between A100 1.55 TB/s and B200 8 TB/s gives XGBoost 99.95% MAPE, GNN 46.4%). |
| **RQ3 Decision effectiveness** | ✅ | `grouped_top_k_accuracy` operationalized in `metrics.py`; Router exposes the same routing logic as a deployable system. |
| **Router (system-level integration)** | ✅ | Two-tier SHAP-driven OOD router; on the heterogeneous mixed split it strictly beats both single models (52.3% vs XGB 62.0%, GNN 59.1%). On RQ2 splits the Tier 1 / Tier 2 rules are insufficient → Tier 3 hardware-OOD is documented future work. |

See [results.md](results.md) — RQ1 (§1), RQ2 (§2), RQ3 (§3), Router (§4), SHAP (§5), Demo (§6).

## 1. Project positioning

**Core function**: learn `f(G, s, h) → T̂` — given a model computation graph G, an inference config s = (batch, seq), and a target GPU's hardware vector h (5-dim), predict end-to-end forward-pass latency.

**Project nature**: CMU 15-442 MLSys course final project. Architecture borrows from NeuSight (ASPLOS'25), Akhauri 2024, HELP. Differentiation is in (a) **architecture extrapolation stability** (leave-model-out wins 9/9 prefill+decode) and (b) the **two-tier SHAP-driven router** as a system-level integration.

### 1.1 Scope decisions vs proposal (must declare in §Intro)

| # | Proposal said | We did | Limitations note |
|---|---|---|---|
| 1 | CPU/GPU operator placement | Pure GPU cross-generation; `s` = (batch, seq) inference config | ✅ required |
| 2 | End-to-end latency | Forward-pass / prefill latency in main results; decode mode added on a separate branch | ✅ required |
| 3 | Transformer + ResNet | Transformer family only (GPT-2 / BERT / T5 — 6 models) | ✅ required |
| 4 | torch.compile? | All HF eager mode + `attn_implementation="eager"` | ✅ required (proposal §4 already stated) |

### 1.2 Three RQs from the proposal — coverage

| RQ | What it tests | Splits run | Status |
|---|---|---|---|
| **RQ1 Workload generalization** | Train on K models, test on the held-out one | `leave-model=*` × 6 (prefill) + × 3 (decode) | ✅ GNN 9/9 wins |
| **RQ2 Hardware generalization** | Train on 4 GPUs, test on the held-out one | `leave-gpu=*` × 5 | ✅ Mixed (GNN 2/5, XGBoost 3/5; GNN wins on mean) |
| **RQ3 Decision effectiveness** | Rank candidate strategies, select near-optimal without exhaustive profiling | grouped GPU-selection top-K + Router mixed split | ✅ Operationalized |

## 2. Hero claim (poster top of fold)

> A graph-aware GNN cost model wins all 9 leave-model-out splits across
> prefill (mean 28.1% MAPE vs XGBoost 70.9%) and decode (mean 8.9% MAPE
> vs XGBoost 58.2%). The framework transfers to decode *more cleanly*
> than prefill — decode latency is dominated by weight-bytes-read, so
> the per-op signature is sharper. On a heterogeneous mixed-architecture
> test set, a two-tier SHAP-driven router strictly beats both XGBoost
> alone (62.0%) and GNN alone (59.1%) at 52.3% MAPE.

## 3. §Limitations (must write these on the poster)

1. **Scope vs proposal**: dropped CPU/GPU operator placement and ResNet to fit the course-project timeline (§1.1).
2. **Forward-pass / prefill** is the main protocol. Decode-mode prediction is validated on a separate branch (5 GPUs × 3 GPT-2 models), but BERT-family decode is not applicable (encoder-only) and T5 enc-dec decode plumbing is left as TODO.
3. **XGBoost feature engineering** — we compare against the standard 12-dim global-feature XGBoost. A richer per-op-type histogram XGBoost was not evaluated; SHAP suggests the dominant feature would still be an aggregate magnitude (`log1p(total_memory_bytes)`) and OOD extrapolation would persist.
4. **Architecture extrapolation range** — within the Transformer family only (GPT-2 / BERT / T5). Cross-family generalization (e.g. to ResNet, Mamba, MoE) is future work.
5. **Hardware coverage** — 5 GPUs span Ampere → Blackwell. Sparse coverage in the bandwidth dimension is a deliberate experimental setting; it is also what production deployments routinely face when a new SKU launches before profiling data accumulates. The Tier-3 hardware-OOD router check that would make `leave-gpu=h100` route to GNN automatically is documented future work.

## 4. Sprint plan (final day)

| Task | File / artifact | Status |
|---|---|---|
| Router backend (2-tier SHAP-driven) | [src/hetero_cost_model/router.py](../src/hetero_cost_model/router.py) | ✅ |
| `predict.py` demo CLI | [scripts/predict.py](../scripts/predict.py) | ✅ ([results.md §6](results.md#6-demo-cli--predictpy)) |
| SHAP diagnosis script | [scripts/shap_xgboost_diagnosis.py](../scripts/shap_xgboost_diagnosis.py) | ✅ ([results.md §5](results.md#5-shap-diagnostic-on-xgboost)) |
| 6 leave-model-out re-run on 5-GPU (Modal H100) | results/leave_model_*_gat_e50.json | ✅ |
| 5 leave-gpu-out (RQ2) on 5-GPU (Modal H100) | results/leave_gpu_*_gat_e50.json | ✅ |
| Mixed split Router showcase | results/mixed_*_gat_e30.json | ✅ |
| Decode profiling on 5 GPUs | data/raw/decode/*.csv | ✅ ([results.md §1.2](results.md#12-decode-leave-model-out-3-splits)) |
| Decode leave-model-out × 3 | runs of train_and_eval on data/raw/decode/* | ✅ |
| RQ3 grouped GPU-selection metric | [src/hetero_cost_model/metrics.py](../src/hetero_cost_model/metrics.py) | ✅ |
| **Poster text** (hero / related work / Limitations / 2 figures) | LaTeX template TBD | 🔴 today |
