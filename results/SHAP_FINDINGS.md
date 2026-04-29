# SHAP Diagnosis — XGBoost Failure Mechanism on Architecture Extrapolation

**Setup**: train fold = 5 models (held-out: gpt2-large), 1568 samples
**XGBoost MAPE on this split**: 39.4% vs GNN 25.6%

## Top features by mean |SHAP|

| Rank | Feature | mean &#124;SHAP&#124; | Train range | Test range | OOD? |
|---|---|---:|---|---|---|
| 1 | `seq_len` | 24.41 | [32, 1024] | [32, 1024] | 0% |
| 2 | `batch_size` | 15.62 | [1, 16] | [1, 16] | 0% |
| 3 | `fp16_tflops` (norm.) | 15.59 | [0.013, 0.9] | [0.013, 0.9] | 0% |
| 4 | **`log1p(total_memory_bytes)`** | **9.53** | **[18, 21]** | **[22, 22]** | **100%** |
| 5 | `bandwidth_gbs` (norm.) | 7.55 | [0.033, 0.89] | [0.033, 0.89] | 0% |

## Poster caption (< 80 words)

> XGBoost relies heavily on graph-aggregate magnitudes such as
> `log1p(total_memory_bytes)` (4th-most-important feature). On the
> held-out gpt2-large split, this aggregate takes values entirely
> **outside the training range** ([22] vs train [18, 21]) — 100% of
> test samples are OOD on this dimension. Tree models cannot
> extrapolate beyond cell boundaries, which explains the 39.4% MAPE
> blow-up. Graph-aware models avoid this via per-op decomposition.

## Files

- `results/shap_importance.png` — feature importance bar chart
- `results/ood_distribution.png` — train/test histogram of dominant OOD feature
