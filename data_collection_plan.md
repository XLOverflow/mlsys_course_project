# 数据收集子计划

> 与 [two_week_execution_plan.md](two_week_execution_plan.md) 和 [research_review.md](research_review.md) 协同。评审中发现的数据采集风险在下文 §3、§4、§5 内嵌修复。
>
> **Scope 对齐（2026-04-17）**：GPU-only cross-generation、测 forward-pass latency、Transformer 家族、HF eager mode。不收集 CPU placement / ResNet / torch.compile 数据。详见 [two_week_execution_plan.md §1](two_week_execution_plan.md)。
>
> **实现状态（2026-04-17）**：profiling 框架 ✅ 已实现（[scripts/run_profiling.py](scripts/run_profiling.py)）；CSV schema 20 列 ✅ 全部填写；OOM/preemption/SKU-upgrade 护栏 ✅ 全部就位；本地 CPU dry-run 已验证通过。等 Day 3 上 GPU 采真数据。

## 1. 我们到底在收集什么

一条样本 = 一次 transformer forward-pass（= LLM prefill 阶段）的观测结果，对应代价模型的一个训练点：

```
(model_name, gpu, batch_size, seq_len)  →  latency 分布统计
```

**测量范围明确**：本项目只测单次 forward pass（prefill），**不测 autoregressive decode**。报告里术语用 "forward-pass latency / prefill latency"，不泛化为 "LLM inference latency"（后者包含 decode 循环 + KV cache 重用，是完全不同的 roofline）。

代价模型的输入三元组 f(G, s, h) → T̂ 在数据集里的对应关系：

| 代价模型 | 数据集里的表示 |
|---------|-------------|
| G（计算图） | 由 `model_name` 在训练时现场 extract，不存在 CSV 里 |
| s（推理配置） | `batch_size` + `seq_len` 两列 |
| h（硬件向量） | 由 **`actual_gpu_name`** 列在训练时从 `HARDWARE_REGISTRY` 查表（不是脚本声明的 `gpu` label，防 Modal 自动升配污染） |
| T̂（预测延迟） | **`p50_ms`**（训练目标，抗长尾异常） |

**关键决策**：

- 图 G 不存在数据文件里：同一个 `model_name`（如 `gpt2-small`）对应的计算图是确定的，训练时现场 extract 即可。CSV 只存可变的测量量
- 训练 target 用 **p50 而非 mean**：shared host 环境下长尾会污染 mean；p50 抗噪，std/mean 超阈值不需要整行丢弃
- 硬件查表用 **`actual_gpu_name`** 而非 `gpu`：Modal 缺货时会把 `gpu="A100"` 悄悄升到 A100-80GB、`gpu="H100"` 升到 H200，若按声明值查表，h 向量就错了

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
| `h200` | Modal | few-shot 测试（同架构不同规格，详见 §6 切分） |
| `b200` | Modal | few-shot 测试 |

### 2.3 推理配置网格

```
batch_sizes = [1, 2, 4, 8, 16]    # gpt2-large / bert-large 用 [1, 2, 4]；gpt2-small / bert-base 扩到 16
seq_lens    = [64, 128, 256, 512] # bert/t5 全部支持；gpt2-large 在 V100 32GB 上 seq=512 × bs=8 可能 OOM（允许 OOM 行）
精度        = FP16 + attn_implementation="eager"（全部统一，不测 FP32/BF16）
随机 hold-out = 每个 GPU 额外 100 个随机 (bs, sl) 组合，用于工作负载泛化评估
```

**总样本量**：

- Hero 矩阵：6 模型 × 5 GPU × (5 batch × 4 seq) = **600 条**（扣除 OOM 约 550 条实际）
- Hold-out 随机点：5 GPU × 100 = **500 条**
- 合计 **~1000+ hero + 500 hold-out ≈ 1500+ 条**

每条样本对应 **100 次实测**（warmup=50 之后）取 p50 / p95 / mean / std 统计，不是 100 条数据行。

**关于精度选择**：

- **FP16**：所有训练/测试 GPU（V100 Volta 到 B200 Blackwell）都支持 FP16 tensor core，可比
- **绝对不测 BF16**：V100 没有 BF16 tensor core，会软件模拟，跨 GPU 不可比
- **`attn_implementation="eager"`**：Flash-Attention-2 不支持 V100，为保证所有 GPU 走同一 attention kernel 路径，强制用 eager。报告里相应声明"eager-attention latency"
- **不加 `torch.compile`**（遵循 proposal §4 声明）。测量范围限于 HF eager mode；compile / vLLM / TensorRT-LLM 在 Limitations 段声明为 out-of-scope

---

## 3. 数据格式

### 3.1 CSV schema（一行 = 一次实验）

```
model_name, gpu, batch_size, seq_len,
mean_ms, std_ms, p50_ms, p95_ms, n_runs, noisy,
peak_memory_mb,
actual_gpu_name, actual_mem_gb, actual_sm_count,
attn_impl, kernel_backend,
platform, cuda_version, driver_version, torch_version, timestamp
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `model_name` | str | `gpt2-small` 等 |
| `gpu` | str | 脚本声明的 GPU label（可能被 Modal 升配） |
| `batch_size` | int | |
| `seq_len` | int | |
| `mean_ms` | float | 参考统计 |
| `std_ms` | float | 用于检测测量噪声 |
| `p50_ms` | float | **训练目标 T**（抗长尾） |
| `p95_ms` | float | 长尾延迟 |
| `n_runs` | int | 实测次数（正常 100） |
| `noisy` | bool | `std/mean > 0.05` 时 True，训练时可选择性过滤 |
| `peak_memory_mb` | float | 峰值显存占用，**仅用于 OOM 检测**（跨平台分配器不同，不能作为 target） |
| `actual_gpu_name` | str | `torch.cuda.get_device_name()`，**训练时用这个查 `HARDWARE_REGISTRY`**，不是 `gpu` 列 |
| `actual_mem_gb` | float | `properties.total_memory / 1e9`，识别 Modal 升配（例如 A100 40GB → 80GB） |
| `actual_sm_count` | int | `properties.multi_processor_count`，再次交叉校验 |
| `attn_impl` | str | 本项目统一用 `"eager"`（遵循 proposal §4，不用 `torch.compile`；Flash-Attention-2 不支持 V100） |
| `kernel_backend` | str | 从 `torch.backends.cudnn` / cuBLAS 查（可选），用于 debug 跨 GPU kernel 选择差异 |
| `platform` | str | `psc` / `modal` |
| `cuda_version` | str | 环境记录，追溯用 |
| `driver_version` | str | 环境记录，Modal 和 PSC 差异来源之一 |
| `torch_version` | str | 环境记录，追溯用 |
| `timestamp` | str | ISO8601，排查异常用 |

### 3.2 schema 关键设计决定的原因

- **`actual_gpu_name` / `actual_mem_gb` / `actual_sm_count` 三件套**：Modal 默认会在缺货时把 `gpu="H100"` 升到 H200、`gpu="A100"` 升到 A100-80GB、`gpu="B200+"` 升到 B300。即使使用 `!` 锁定 SKU 也要在 CSV 里留冗余记录做双重校验。训练集加载时 `HARDWARE_REGISTRY[row.actual_gpu_name]` 而不是 `HARDWARE_REGISTRY[row.gpu]`
- **`p50_ms` 是 target**：shared host 上 CoV 可能到 5-15%（尤其小 kernel），p50 对 outlier 鲁棒
- **`noisy` 标签保留而非过滤**：先不丢数据，训练时用 flag 筛
- **全部用 HF eager mode**：`attn_impl="eager"` 一列固定值，不加 `torch.compile`（遵循 proposal §4）。Limitations 段声明 scope

### 3.3 文件组织

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
for each (model, batch_size, seq_len) in 配置矩阵（随机顺序）:
    1. 构造随机输入 input_ids ∈ [0, vocab_size), shape=(batch_size, seq_len)
    2. 模型已在 GPU 上（float16, attn_implementation="eager"），不重复加载
    3. 50 次 warmup（不计入统计，触发 cuDNN autotune 完成）
    4. 清空 CUDA memory stats
    5. 100 次测量（CUDA events 计时）：
       - start_event = torch.cuda.Event(enable_timing=True)
       - end_event   = torch.cuda.Event(enable_timing=True)
       - start_event.record()
       - with torch.inference_mode(): model(input_ids)
       - end_event.record()
       - torch.cuda.synchronize()
       - 记录 start_event.elapsed_time(end_event)  # 单位 ms
    6. 记录 peak_memory_mb = cuda.max_memory_allocated() / 1e6
    7. 记录 actual_gpu_name / actual_mem_gb / actual_sm_count
    8. 重置 memory stats
    9. 写入 CSV 一行（若文件已存在该 (model, bs, sl, gpu) 则 skip，idempotent）
```

### 4.2 关键测量参数的选择理由

| 参数 | 值 | 理由 |
|------|---|------|
| **计时方式** | `torch.cuda.Event` | 替换 `time.perf_counter()`。CUDA events 基于 GPU 事件时间戳，和 CPU 抖动解耦；shared host 下显著降低测量方差 |
| **warmup** | 50 次 | 10 次对 <5ms 小 batch 不足以让 cuDNN autotune 收敛 |
| **测量次数** | 100 次 | 从 50 提升到 100，配合 p50 target 更稳 |
| **target 统计量** | `p50_ms` | shared host 长尾用中位数抗干扰；std/mean 高不需整行丢弃 |
| **精度** | FP16 + `attn_implementation="eager"` | V100 没 BF16，FA-2 不支持 V100；统一 eager 保证跨 GPU 走同一 kernel 路径 |
| **`torch.inference_mode()`** | 是 | 比 `no_grad` 更激进的优化，减少测量噪声 |
| **配置顺序** | **随机** | 避免时间漂移在某个配置上累积 |
| **CPU 亲和性** | `taskset -c` 绑 4 核 | 避免进程在 CPU 间迁移导致的 cache cold + dispatch 抖动。比申请 `--exclusive` 经济得多 |
| **batch 内随机输入** | 每次 warmup/measure 都相同 | 避免不同 token 分布引入方差 |

### 4.3 为什么不申请 PSC `--exclusive`

PSC 的 `--exclusive` 需要预约整机 8 卡才能生效，代价过高。单卡分配下：

- GPU 本体（SM / HBM / L2 / tensor core）是 SLURM cgroup 独占的，另一块卡的 job 碰不到你的 SM
- 唯一共享的是 host CPU / 主机内存带宽 / PCIe root complex
- 用 CUDA events 基本把 CPU 抖动隔离掉；大 forward pass（> 50ms）CoV < 1-2%，小 forward pass（< 5ms）CoV 5-15% — 后者用 p50 + n=100 就够了

详见 [research_review.md §3.3](research_review.md)。

### 4.4 Modal 锁定 SKU（防自动升配）

Modal 缺货时会**静默升配**按低档价格计费：

- `gpu="H100"` 可能给到 **H200**
- `gpu="A100"` 可能给到 **A100-80GB**（而非 40GB）
- `gpu="B200+"` 可能给到 **B300**

强制用带锁的 SKU 名：

```python
@app.function(gpu="A100-40GB", timeout=7200)   # 锁 40GB，不要 "A100"
def profile_on_a100(): ...

@app.function(gpu="H100!", timeout=7200)       # ! 后缀禁止升到 H200
def profile_on_h100(): ...

@app.function(gpu="B200", timeout=7200)        # 不要 "B200+"
def profile_on_b200(): ...
```

每条样本必须在 CSV 里写入 `actual_gpu_name` / `actual_mem_gb` / `actual_sm_count`，训练集加载时按 `actual_gpu_name` 查 `HARDWARE_REGISTRY`。

### 4.5 OOM 处理

```python
try:
    # 跑测量
except torch.cuda.OutOfMemoryError:
    # 写入一行: p50_ms=NaN, peak_memory_mb=OOM, n_runs=0
    # 继续下一个配置，不中断整个 job
```

OOM 的行保留在 CSV 里而不是直接丢弃，训练时过滤掉，但保留 OOM 信息用于分析"该模型在该 GPU 上哪些配置放不下"。

### 4.6 Modal preemption 的幂等处理

Modal 容器默认 preemptible（non-preemptible 价格 3×）。脚本要做成 idempotent：

```python
if row_exists(csv_path, model, bs, sl, actual_gpu_name):
    continue    # 已采集过的点直接 skip，不覆盖
```

被抢占重启后续跑即可，不需要重置已完成的配置。

### 4.7 质量检验（每次收集完后执行）

```
1. 检查 std_ms / p50_ms < 0.05
   超过的行 noisy=True，训练时可选择性排除

2. 检查延迟 vs 硬件特征的单调性：
   同 (model, bs, sl) 在 compute-bound 区，latency 应随 FP16 TFLOPS 单调递减
   同 (model, bs, sl) 在 memory-bound 区，latency 应随带宽单调递减
   违反的行人工审查（多半是 Modal 升配或 attn_impl 不一致）

3. 检查 p95_ms < 2 × p50_ms
   超过说明有明显 outlier（可能被系统进程抢占或 Modal 容器漂移）

4. 检查每个 (model, actual_gpu_name) 组合的样本数匹配配置矩阵
   缺失说明有配置 OOM 或 job 被抢占

5. 检查 actual_gpu_name 列对所有行一致（每个声明 gpu）
   若发现 A100-40GB 混 A100-80GB、H100 混 H200，说明 Modal 升配发生过
   要么重跑要么按 actual_gpu_name 分组训练
```

---

## 5. 收集流程

### 5.1 PSC 作业脚本结构（V100 / H100）

```bash
#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --time=03:00:00       # warmup 50 + runs 100 + 矩阵扩大，需更长时间
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4     # 配合 taskset 绑核，不用 --exclusive

conda activate mlsys
# 绑到前 4 核，减少 CPU 迁移抖动
taskset -c 0-3 python scripts/run_profiling.py \
    --gpu v100 \
    --models gpt2-small gpt2-medium gpt2-large bert-base bert-large t5-small \
    --batch-sizes 1 2 4 8 16 \
    --seq-lens 64 128 256 512 \
    --random-holdout 100 \
    --warmup 50 --runs 100 \
    --output data/raw/v100.csv
```

一个作业覆盖全部模型+配置，约 3 小时内可完成（因 warmup/runs 翻倍 + 配置矩阵扩大）。

**不要用 `--exclusive`**：PSC 需要预约整机 8 卡，代价过高。单卡分配 + `taskset` + CUDA events 的组合已经足够。

### 5.2 Modal 作业结构（A100 / H200 / B200）

```python
# 锁定 SKU，防自动升配
@app.function(gpu="A100-40GB", timeout=10800)  # 3h
def profile_on_a100():
    run_profiling(gpu="a100", output="data/raw/a100.csv")

@app.function(gpu="H200", timeout=10800)
def profile_on_h200():
    run_profiling(gpu="h200", output="data/raw/h200.csv")

@app.function(gpu="B200", timeout=10800)       # 不要 "B200+"
def profile_on_b200():
    run_profiling(gpu="b200", output="data/raw/b200.csv")
```

三个 GPU 可同时提交，互不依赖；preemptible 被抢占后重跑，脚本 idempotent 会 skip 已完成行。

### 5.3 图的提取（独立步骤，CPU-only）

```
scripts/extract_graphs.py
  → 对每个 model_name，调 extract_graph(model, inputs, hf_input_names=["input_ids"])
    走 transformers.utils.fx.symbolic_trace（需要 transformers<4.52）
  → 序列化保存到 data/graphs/{model_name}.pkl
  → 打印节点数、FLOPs 总量，做 sanity check
```

**Smoke test —— ✅ 2026-04-17 已完成，6/6 PASS**（`scripts/smoke_test_graphs.py`）

```
gpt2-small:  1247 nodes / 1599 edges / 2.52e+09 FLOPs
gpt2-medium: 2471 nodes / 3171 edges / 3.43e+09 FLOPs
gpt2-large:  3695 nodes / 4743 edges / 4.37e+09 FLOPs
bert-base:    529 nodes /  642 edges / 7.02e+09 FLOPs
bert-large:  1021 nodes / 1242 edges / 2.15e+10 FLOPs
t5-small:    1159 nodes / 1404 edges / 3.91e+09 FLOPs
```

**守护措施（每次环境变更必做）**：

- `pip install -r requirements.txt` 或 `conda env create -f environment.yml` 之后
- `transformers` 被任何原因升级之后
- 新成员入组、或迁移到新机器（PSC / Modal 容器）之后

都必须先 `python scripts/smoke_test_graphs.py`，6/6 PASS 才可进入 profiling。

这一步在本地 CPU 上跑，不需要 GPU。

---

## 6. 从 CSV 到训练样本的转换

```
for each row in all_profiling.csv:
    if row.p50_ms is NaN or row.n_runs == 0:
        continue    # OOM 行跳过

    graph  = load_pickle(f"data/graphs/{row.model_name}.pkl")
    config = InferenceConfig(row.batch_size, row.seq_len)
    hw     = HARDWARE_REGISTRY[row.actual_gpu_name]    # ← 用实际卡名，不是脚本声明的 gpu
    target = row.p50_ms                                # ← 用 p50 而非 mean
    → Sample(graph, config, hw, target)
    → sample_to_pyg(sample)
    → 加入 LatencyDataset
```

训练/验证/测试集划分方式（见 [two_week_execution_plan.md §5.2](two_week_execution_plan.md)）：

- **硬件维度**：
  - V100 + T4 + A100 + A10 + L4 + H100 行 → 训练（**6 个 anchor**，覆盖 5 个架构代际）
  - **H200 行 → few-shot 评估**（50/100/200 samples，不再当 zero-shot；理由见 [research_review.md §2.1](research_review.md)）
  - **B200 行 → hero zero-shot + few-shot 对照**
- **工作负载维度**：按 `model_name` 分层，确保每种架构都在训练集里有代表
- **交叉切分**：除随机 hold-out 外，必跑
  - **leave-one-GPU-out CV**（6 个 training anchor 的 6 折）作为主 claim
  - **leave-one-model-out**（验证是否有模型级泄漏）

---

## 7. 需要对齐的决策点

在开始写 profiling 脚本前，请确认以下几点：

1. **gpt2-large 在 V100 (32GB) 上的 OOM 边界**
   → FP16 下 gpt2-large (774M) ≈ 1.5GB 权重 + KV/activation。batch=8 × seq=512 较危险。允许该配置 OOM（CSV 行保留，p50_ms=NaN），不要为此跳过整个模型

2. **seq_len 语义（已敲定，见 §1）**
   → 只测单次 forward pass（输入 seq_len 个 token，输出 logits），**不测 autoregressive decode**
   → 报告里用 "forward-pass latency / prefill latency" 措辞，不用 "LLM inference latency"

3. **BERT 的 seq_len=512**
   → BERT 最大支持 512，可以加入。bert-large × bs=8 × seq=512 在 V100 上可能吃紧，允许 OOM

4. **同一个 model_name 在不同 batch_size 下图 G 是否相同？**
   → 是。`torch.fx` trace 出来的图结构与 batch_size 无关（batch 是动态维度）。batch_size 只影响 s 向量，不影响 G

5. **fx 能否 trace 出 HF 的 GPT-2/BERT/T5** —— ✅ 已解决 2026-04-17
   → 用 `extract_graph(model, inputs, hf_input_names=[...])` 走 `transformers.utils.fx.symbolic_trace`；`transformers` pin 到 `>=4.35,<4.52`（4.46.3 已验证）。T5 用 `["input_ids", "decoder_input_ids"]` + `T5ForConditionalGeneration` + `attn_implementation="eager"` 可跑通

6. **Modal 升配风险是否完全被 `H100!` / `A100-40GB` / `B200` 锁定**
   → 即使锁定，CSV 仍需冗余记录 `actual_gpu_name`，训练时按实际卡型查表

确认这些后，profiling 脚本可以一天内写完并在本地 CPU 上调试通过。

**实际完成情况（2026-04-17）**：以上 6 条决策点均已敲定，[scripts/run_profiling.py](scripts/run_profiling.py) 实现完毕并通过 CPU dry-run（bert-base × bs=1 × seq=32 在 28.2 ms p50，CSV 20 列全部正确写入，idempotent resume 正常）。Day 3 直接上 GPU 即可。

---

## 8. 与其他文档的对应关系

| 本文档章节 | 对应 `two_week_execution_plan.md` 章节 | 对应 `research_review.md` 章节 |
| --- | --- | --- |
| §1 测量范围（只测 prefill） | §1 术语校准 | §2.4 seq_len 语义 |
| §2 配置矩阵（扩大到 1500+） | Day 3-4 数据收集 | §2.2 数据量 + §5.1 总量提升 |
| §3 CSV schema 扩充 | 交付物清单 Profiling 数据集 | §3.4 Modal SKU + §3.6 schema |
| §4.1 CUDA events + p50 | Day 1-2 profiler.py 任务 | §3.3 测量方法 |
| §4.3 不申请 `--exclusive` | Day 3-4 PSC job | §3.3 PSC 校准 |
| §4.4 Modal SKU 锁定 | Day 3-4 / Day 8-9 Modal job | §3.4 Modal SKU |
| §6 训练切分 (leave-one-GPU-out) | §5.2 核心实验对比逻辑 | §4.(B) 评估协议重设 |
