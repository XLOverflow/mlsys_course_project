# Documentation Index

> Single source of truth for the project. All `.md` documentation lives
> under `docs/`; nothing is duplicated outside this tree. Numbers reflect
> the **5-GPU final dataset** (A10 / A100 / B200 / H100 / L4) and the
> experiments completed by 2026-04-29.

| File | Content |
|---|---|
| [plan.md](plan.md) | Project status, scope vs proposal, three-RQ coverage, claim hierarchy, sprint plan |
| [methods.md](methods.md) | Data collection, hardware feature vector, GNN architecture, baselines, training pipeline, router design |
| [results.md](results.md) | All final results — RQ1 / RQ2 / RQ3 / Router / SHAP / Demo |
| [related-work.md](related-work.md) | Research-review of cost-model literature relevant to this project |
| [archive/phase6-7gpu-sweep.md](archive/phase6-7gpu-sweep.md) | Historical 7-GPU Phase-6 16-run sweep, **superseded** by 5-GPU final results in [results.md](results.md). Kept for git-history continuity. |

## Conventions

- **Numbers**: percentages are MAPE unless otherwise noted. "GNN" means the v2 GAT cost model with `node_level_sh + sum readout + global_skip` (see [methods.md §2.1](methods.md#21-gnn-v2-architecture)).
- **Splits**: `leave-model=X` holds out model X (RQ1); `leave-gpu=X` holds out GPU X (RQ2); `mixed=X,Y` holds out both architectures **and** 20% of seen-architecture configs (Router showcase).
- **Mode**: prefill = single forward over the prompt; decode = single-token forward with populated KV cache. Stored in separate CSVs (`data/raw/*.csv` vs `data/raw/decode/*.csv`).
- **Hero claim**: see [plan.md §2](plan.md#2-hero-claim-poster-top-of-fold).
