# 数据收集子计划

## 1. 我们到底在收集什么

一条样本 = 一次端到端推理实验的观测结果，对应代价模型的一个训练点：

```
(model_name, gpu, batch_size, seq_len)  →  latency 分布统计
```

代价模型的输入三元组 f(G, s, h) → T̂ 在数据集里的对应关系：

| 代价模型 | 数据集里的表示 |
|---------|-------------|
| G（计算图） | 由 `model_name` 在训练时现场 extract，不存在 CSV 里 |
| s（推理配置） | `batch_size` + `seq_len` 两列 |
| h（硬件向量） | 由 `gpu` 列在训练时从 `HARDWARE_REGISTRY` 查表 |
| T̂（预测延迟） | `mean_ms`（训练目标） |

**关键决策**：图 G 不存在数据文件里。同一个 `model_name`（如 `gpt2-small`）对应的计算图是确定的，训练时现场 extract 即可。CSV 只存可变的测量量。

---

## 2. 配置矩阵（收集什么组合）

### 2.1 模型列表

| model_name | 参数量 | 架构类型 | 角色 |
|-----------|--------|---------|------|
| `gpt2-small` | 124M | Decoder-only | 主训练集 |
| `gpt2-medium` | 355M | Decoder-only | 规模泛化测试 |
| `gpt2-large` | 774M | Decoder-only | 规模泛化测试（显存允许时） |
| `bert-base` | 110M | Encoder-only | 架构泛化测试 |
| `bert-large` | 340M | Encoder-only | 架构泛化测试 |
| `t5-small` | 60M | Enc-Dec | 架构泛化测试 |

### 2.2 GPU 列表

| gpu | 平台 | 训练/测试 |
|-----|------|---------|
| `v100` | PSC | 训练 |
| `a100` | Modal | 训练 |
| `h100` | PSC | 训练 |
| `h200` | Modal | zero-shot 测试 |
| `b200` | Modal | few-shot 测试 |

### 2.3 推理配置网格

```
batch_sizes = [1, 4, 8]        # gpt2-large / bert-large 用 [1, 2, 4]
seq_lens    = [64, 128, 256]   # gpt2 decode-only 建议 64/128；bert encode-only 可到 512
精度        = FP16（全部统一，不测 FP32）
```

**总样本量**：6 模型 × 5 GPU × 9 配置 = **270 条**（最小量）

每条样本对应 50 次实测取统计，不是 50 条数据行。

---

## 3. 数据格式

### 3.1 CSV schema（一行 = 一次实验）

```
model_name, gpu, batch_size, seq_len,
mean_ms, std_ms, p50_ms, p95_ms, n_runs,
peak_memory_mb,
platform, cuda_version, torch_version, timestamp
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `model_name` | str | `gpt2-small` 等 |
| `gpu` | str | 与 `HARDWARE_REGISTRY` 的 key 一致 |
| `batch_size` | int | |
| `seq_len` | int | |
| `mean_ms` | float | **训练目标 T** |
| `std_ms` | float | 用于检测测量噪声 |
| `p50_ms` | float | 中位数 |
| `p95_ms` | float | 长尾延迟 |
| `n_runs` | int | 实测次数（正常应为 50） |
| `peak_memory_mb` | float | 峰值显存占用，OOM 检测用 |
| `platform` | str | `psc` / `modal` |
| `cuda_version` | str | 环境记录，追溯用 |
| `torch_version` | str | 环境记录，追溯用 |
| `timestamp` | str | ISO8601，排查异常用 |

### 3.2 文件组织

```
data/
  raw/
    v100.csv        ← PSC 上 V100 的全部结果
    a100.csv        ← Modal 上 A100 的全部结果
    h100.csv        ← PSC 上 H100 的全部结果
    h200.csv        ← Modal 上 H200 的全部结果
    b200.csv        ← Modal 上 B200 的全部结果
  all_profiling.csv ← 合并后的总表，供训练使用
```

每个 `raw/` 文件由一次 GPU session 产出，互不依赖，可并行收集。

---

## 4. 测量方法（观测体系）

### 4.1 单次实验流程

```
for each (model, batch_size, seq_len) in 配置矩阵:
    1. 构造随机输入 input_ids ∈ [0, vocab_size), shape=(batch_size, seq_len)
    2. 模型已在 GPU 上（float16），不重复加载
    3. 10 次 warmup（不计入统计）
    4. 清空 CUDA memory stats
    5. 50 次测量：
       - cuda.synchronize()
       - t0 = perf_counter()
       - model(input_ids)
       - cuda.synchronize()
       - 记录 (perf_counter() - t0) * 1000
    6. 记录 peak_memory_mb = cuda.max_memory_allocated() / 1e6
    7. 重置 memory stats
    8. 写入 CSV 一行
```

### 4.2 关键测量参数的选择理由

| 参数 | 值 | 理由 |
|------|---|------|
| warmup | 10 次 | 消除 kernel launch、cuDNN 初始化、缓存冷启动的影响 |
| 测量次数 | 50 次 | 平衡精度（std < 2%）与 GPU 时间成本 |
| 精度 | FP16 | Transformer 推理标准精度；V100/H100/B200 均支持；保证跨 GPU 可比性 |
| `torch.no_grad()` | 是 | 推理场景，不需要梯度 |
| `torch.inference_mode()` | 是 | 比 no_grad 更激进的优化，减少测量噪声 |
| `cuda.synchronize()` | 两次（前后） | GPU 是异步执行，不同步会低估延迟 |
| batch 内随机输入 | 每次 warmup/measure 都相同 | 避免不同 token 分布引入方差 |

### 4.3 OOM 处理

```python
try:
    # 跑测量
except torch.cuda.OutOfMemoryError:
    # 写入一行: mean_ms=NaN, peak_memory_mb=OOM, n_runs=0
    # 继续下一个配置，不中断整个 job
```

OOM 的行保留在 CSV 里而不是直接丢弃，训练时过滤掉，但保留 OOM 信息用于分析"该模型在该 GPU 上哪些配置放不下"。

### 4.4 质量检验（每次收集完后执行）

```
1. 检查 std_ms / mean_ms < 0.05（变异系数 < 5%）
   超过的行标记为 noisy=True，训练时可选择性排除

2. 检查 H100 latency < V100 latency（同配置）
   违反的行说明有测量异常

3. 检查 p95_ms < 2 × p50_ms
   超过说明有明显的 outlier（可能被系统进程抢占）

4. 检查每个 (model, gpu) 组合的样本数 = 9
   缺失的说明有配置 OOM 或脚本提前退出
```

---

## 5. 收集流程

### 5.1 PSC 作业脚本结构（V100 / H100）

```bash
#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G

conda activate llmsys_hw6
python scripts/run_profiling.py \
    --gpu v100 \
    --models gpt2-small gpt2-medium bert-base bert-large t5-small \
    --batch-sizes 1 4 8 \
    --seq-lens 64 128 256 \
    --output data/raw/v100.csv
```

一个作业覆盖全部模型+配置，2 小时内可完成。

### 5.2 Modal 作业结构（A100 / H200 / B200）

```python
@app.function(gpu="A100", timeout=3600)
def profile_on_a100():
    run_profiling(gpu="a100", output="a100.csv")
```

三个 GPU 可同时提交，互不依赖。

### 5.3 图的提取（独立步骤，CPU-only）

```
scripts/extract_graphs.py
  → 对每个 model_name，用 torch.fx 提取 GraphRepr
  → 序列化保存到 data/graphs/{model_name}.pkl
  → 打印节点数、FLOPs 总量，做 sanity check
```

这一步在本地 CPU 上跑，不需要 GPU，提前准备好。

---

## 6. 从 CSV 到训练样本的转换

```
for each row in all_profiling.csv:
    graph  = load_pickle(f"data/graphs/{row.model_name}.pkl")
    config = InferenceConfig(row.batch_size, row.seq_len)
    hw     = HARDWARE_REGISTRY[row.gpu]
    target = row.mean_ms
    → Sample(graph, config, hw, target)
    → sample_to_pyg(sample)
    → 加入 LatencyDataset
```

训练/验证/测试集划分方式：
- **硬件维度**：V100+A100+H100 行 → 训练；H200 行 → zero-shot 测试；B200 行 → few-shot 测试
- **工作负载维度**：按 model_name 分层，确保每种架构都在训练集里有代表

---

## 7. 需要对齐的决策点

在开始写 profiling 脚本前，请确认以下几点：

1. **gpt2-large 在 V100 (32GB) 上的 batch=8, seq=256 会 OOM 吗？**
   → 如果会，是否只跑 batch=1,2？还是直接跳过 gpt2-large on V100？

2. **seq_len 对 gpt2（decoder-only）的含义**：
   → 对 decoder-only 模型，seq_len 是输入 prompt 长度，输出只取一个 token（不自回归展开）
   → 还是要测完整的生成（生成 seq_len 个 token）？
   → 建议：只测单次 forward pass（输入 seq_len 个 token，输出 logits），不测自回归生成，否则时间差异太大且与图结构不对应

3. **BERT 的 seq_len=256 是否合理？**
   → BERT 最大支持 512，但 256 已经是常用场景，256 够了

4. **同一个 model_name 在不同 batch_size 下图 G 是否相同？**
   → 是的。`torch.fx` trace 出来的图结构与 batch_size 无关（batch 是动态维度）
   → batch_size 只影响 s 向量，不影响 G，这个设计是对的

5. **要不要测 seq_len=512？**
   → 可以加，但会增加约 1/3 的数据量，GPT-2 large 在 V100 上可能 OOM

确认这些后，profiling 脚本可以一天内写完并在本地 CPU 上调试通过。
