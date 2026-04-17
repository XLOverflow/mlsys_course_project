# Pre-extracted computation graphs

每个 `.pkl` 是一个序列化的 `GraphRepr` 对象（见 [src/hetero_cost_model/graph/features.py](../../src/hetero_cost_model/graph/features.py)），对应一个 HF 模型 fx-trace 后的节点/边/特征。训练时从这里加载，比每 epoch 重新 trace 快约 100×。

## 当前清单

| 文件 | 节点 | 边 | 总 FLOPs | 总内存 (FP16) |
| --- | ---: | ---: | ---: | ---: |
| `gpt2-small.pkl`  | 1247 | 1599 | 2.52e+09 |  467.2 MB |
| `gpt2-medium.pkl` | 2471 | 3171 | 3.43e+09 | 1519.7 MB |
| `gpt2-large.pkl`  | 3695 | 4743 | 4.37e+09 | 3390.6 MB |
| `bert-base.pkl`   |  529 |  642 | 7.02e+09 |   43.2 MB |
| `bert-large.pkl`  | 1021 | 1242 | 2.15e+10 |  107.5 MB |
| `t5-small.pkl`    | 1159 | 1404 | 3.91e+09 |   99.4 MB |

生成于 2026-04-17，环境：`transformers==4.46.3`、`torch==2.10.0`、CPU。

## 何时需要重新生成

**必须重跑 `python scripts/extract_graphs.py --force` 的情况**：

1. [src/hetero_cost_model/graph/extractor.py](../../src/hetero_cost_model/graph/extractor.py) 改动（新增/删除节点特征字段、改 op 分类逻辑、改 FLOPs 估算）
2. [src/hetero_cost_model/graph/features.py](../../src/hetero_cost_model/graph/features.py) 或 [vocab.py](../../src/hetero_cost_model/graph/vocab.py) 改动（`NODE_FEATURE_DIM`、op vocab 等）
3. [src/hetero_cost_model/graph/flops.py](../../src/hetero_cost_model/graph/flops.py) 或 [shapes.py](../../src/hetero_cost_model/graph/shapes.py) 改动
4. [src/hetero_cost_model/model_zoo.py](../../src/hetero_cost_model/model_zoo.py) 的 `MODELS` 列表增删或 `load_model` 的 kwargs 改动
5. `transformers` 版本在 `[4.35, 4.52)` 范围内跳变（默认不升级；若升级必须复查 trace 结果）

**不需要重跑的改动**：modeling 配置、训练脚本、baseline、profiler、硬件注册表。

## 怎么加载

```python
import pickle
with open("data/graphs/gpt2-small.pkl", "rb") as f:
    graph = pickle.load(f)   # hetero_cost_model.graph.GraphRepr
print(graph.num_nodes(), graph.total_flops())
```

## 怎么重新生成

```bash
python scripts/smoke_test_graphs.py       # 先过 smoke test
python scripts/extract_graphs.py          # idempotent，跳过已存在的 pkl
python scripts/extract_graphs.py --force  # 强制重算全部
```

如果 smoke test 挂，参考 [research_review.md §3.1](../../research_review.md) 守护措施。绝对不要升级 `transformers` 到 `>= 4.52`。
