# Heterogeneous LLM Cost Model

A learned cost model that predicts forward-pass latency for HuggingFace LLMs
across heterogeneous GPU SKUs from a continuous hardware spec vector and the
model's computation graph.

## Layout

| Path | Contents |
| --- | --- |
| `src/hetero_cost_model/` | package: graph extraction, hardware registry, GNN / MLP / XGBoost models, training loop, router, metrics |
| `scripts/` | CLI drivers: profiling, training, evaluation, prediction |
| `tests/` | pytest suite |
| `data/raw/` | profiling CSVs (prefill at the top level, decode under `decode/`) |
| `data/graphs/` | pre-extracted `GraphRepr` pickles, one per HF model |
| `results/` | per-experiment evaluation JSON and figures |

## Setup

```bash
conda env create -f environment.yml
pip install -e .
```

## Quick start

```bash
pytest                                   # unit + smoke tests
python scripts/extract_graphs.py         # regenerate computation graphs
python scripts/train_and_eval.py --help  # train / eval driver
python scripts/predict.py --help         # single-shot latency prediction
```
