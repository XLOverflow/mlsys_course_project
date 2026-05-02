# Pre-extracted computation graphs

Each `.pkl` is a serialized `GraphRepr` (see [src/hetero_cost_model/graph/features.py](../../src/hetero_cost_model/graph/features.py)) holding the nodes, edges, and features produced by tracing one HuggingFace model with `torch.fx`. Training loads these directly, which is roughly 100x faster than re-tracing every epoch.

## Inventory

| File | Nodes | Edges | Total FLOPs | Total memory (FP16) |
| --- | ---: | ---: | ---: | ---: |
| `gpt2-small.pkl`  | 1247 | 1599 | 2.52e+09 |  467.2 MB |
| `gpt2-medium.pkl` | 2471 | 3171 | 3.43e+09 | 1519.7 MB |
| `gpt2-large.pkl`  | 3695 | 4743 | 4.37e+09 | 3390.6 MB |
| `bert-base.pkl`   |  529 |  642 | 7.02e+09 |   43.2 MB |
| `bert-large.pkl`  | 1021 | 1242 | 2.15e+10 |  107.5 MB |
| `t5-small.pkl`    | 1159 | 1404 | 3.91e+09 |   99.4 MB |

Generated on 2026-04-17 with `transformers==4.46.3`, `torch==2.10.0`, on CPU.

## When to regenerate

Re-run `python scripts/extract_graphs.py --force` after any of:

1. Edits to [src/hetero_cost_model/graph/extractor.py](../../src/hetero_cost_model/graph/extractor.py) (node feature fields, op classification, FLOPs estimator).
2. Edits to [features.py](../../src/hetero_cost_model/graph/features.py) or [vocab.py](../../src/hetero_cost_model/graph/vocab.py) (`NODE_FEATURE_DIM`, op vocab).
3. Edits to [flops.py](../../src/hetero_cost_model/graph/flops.py) or [shapes.py](../../src/hetero_cost_model/graph/shapes.py).
4. Adding or removing entries in `MODELS` or changing `load_model` kwargs in [model_zoo.py](../../src/hetero_cost_model/model_zoo.py).
5. A `transformers` version jump within `[4.35, 4.52)` (do not upgrade past 4.52; HFTracer is incompatible with the new `torch.vmap`-based masking).

Changes to modeling configs, training scripts, baselines, the profiler, or the hardware registry do not require regeneration.

## Loading

```python
import pickle
with open("data/graphs/gpt2-small.pkl", "rb") as f:
    graph = pickle.load(f)   # hetero_cost_model.graph.GraphRepr
print(graph.num_nodes(), graph.total_flops())
```

## Regenerating

```bash
python scripts/smoke_test_graphs.py       # smoke test first
python scripts/extract_graphs.py          # idempotent; skips existing pkls
python scripts/extract_graphs.py --force  # force-recompute all
```
