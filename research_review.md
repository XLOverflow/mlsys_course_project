# 研究思路评审：跨异构 GPU 的 LLM 推理代价模型

> 基于 `two_week_execution_plan.md`、`data_collection_plan.md`、`src/hetero_cost_model/` 代码的综合评审
> 初稿日期：2026-04-16；scope 决策对齐日期：2026-04-17
> 评审标准：**CMU 15-442 MLSys 期末 project**（非论文投稿），两周冲刺、3 人团队。借鉴已有论文架构 OK，关注点是"方法自洽 + 数据干净 + 有一小块自己的贡献"，不是"完全原创"。

## 相对 proposal 的 scope 4 项决策（2026-04-17 对齐）

| 维度 | Proposal 原文 | 实际 scope | 处理方式 |
| --- | --- | --- | --- |
| 异构目标 | CPU/GPU heterogeneous (operator placement) | **GPU-only cross-generation** (V100→B200) | Intro 明说收窄，`s` 改为 (bs, sl) inference config |
| 延迟含义 | "end-to-end latency or throughput" | **end-to-end latency of a single forward pass over the full computation graph** (= prefill stage) | 措辞不用"prefill-only"，避免听起来 scope 窄；decode 列入 future work |
| 工作负载 | "Transformer or ResNet" | **Transformer 家族**（GPT-2 / BERT / T5） | 明说收窄，ResNet 不做 |
| 编译对比 | "we do not compare with torch.compile()" | **严格遵守**，eager only | Limitations 段声明 |

---

## 结论速读

1. **相关工作定位**：本项目位于 NeuSight (ASPLOS'25)、Akhauri & Abdelfattah (MLSys'24)、Habitat、HELP、MAPLE 组成的活跃方向里。课程 project 不需要回避这些，相反**要显式借鉴**它们的架构和设计选择，并在报告里点出你们在这个图谱里的位置（"GNN + 连续硬件规格向量 + LLM 图"三者的组合是差异点）。
2. **方法论有 4 处会影响结果可信度的问题**：3-GPU 硬件泛化其实是 3 点回归、270 样本 + 严重数据泄漏、Roofline 是过弱基线、seq_len 语义和 "LLM inference" 题目对不上（实际只测 prefill/forward pass）。这些是"跑出来的数字能不能站得住"的层面，和 novelty 无关。
3. **代码里有一个能直接报废两周的单点故障**：`torch.fx.symbolic_trace` 在 HF 的 GPT-2/BERT/T5 上会炸。Day 1 必须验证。
4. **头两天必须做**：(A) 6 个模型 fx tracing smoke test；(B) 重新设计评估协议（leave-one-GPU-out CV + 换强基线 XGBoost）。

---

## 一、相关工作盘点：Novelty 受挑战，需要重新定位

### 1.1 Tier 1 — 直接竞争（必须引用并明确差异化）

| 工作 | 核心思路 | 与本项目关系 |
|------|---------|-------------|
| **NeuSight** (ASPLOS 2025, [arXiv:2407.13853](https://arxiv.org/abs/2407.13853)) | 把 kernel 分 tile，用 5 个小 MLP 以连续硬件向量（mem/BW/FLOPs/L2）为条件预测 tile 利用率 → 聚合端到端延迟 | **最大威胁**。已在未见 GPU（含 H100）上报 8.8% inference MAPE。没有 GNN、没做 Blackwell — 这是你唯一差异化点。 |
| **Akhauri & Abdelfattah** "On Latency Predictors for NAS" (MLSys 2024) | GAT + DGF（GCN 变体）双 GNN 集成；**可学习 device-ID embedding 表**；HW 在**每 op 节点** concat 后进 GNN；只测 NASBench-201 / FBNet | 是同一个"GNN + HW 融合"大方向，但**四处具体差别都在我们这边**：(1) 单 GAT vs 双 GNN 集成；(2) **连续硬件规格向量 vs 离散 device-ID embedding** — 这是质的不同，他们需要在新硬件先测 latency 做初始化（本质是少样本），我们只靠 spec sheet 就能预测；(3) post-readout 融合 vs 每节点融合（不同的 inductive bias，值得消融对比）；(4) LLM 计算图 vs NAS cell |
| **Habitat** (USENIX ATC 2021) | Wave-scaling + per-op MLP 预测跨 GPU 训练延迟 | 需要 source GPU 实测 trace。你的 **zero-shot-from-spec** 是清晰差异点。 |
| **HELP** (NeurIPS 2021 Spotlight) | Meta-learn 少样本延迟回归器，硬件用"若干参考架构的延迟向量"表达 | 与你 B200 few-shot 设置最直接对应。必须作为 **few-shot baseline** 引用。 |

### 1.2 Tier 2 — 同家族不同范围（要在 related work 里覆盖）

- **PerfSage** (2023, [arXiv:2301.10999](https://arxiv.org/abs/2301.10999))：GNN 延迟预测，但单设备每模型，无跨硬件泛化。作为 GNN encoder 设计的先例引用。
- **TLP / MTL-TLP** (ASPLOS 2023)：TVM 张量 program cost model，schedule 级，不同粒度。
- **TenSet** (NeurIPS 2021 D&B)：52M 跨设备 program-performance 记录 — 数据集，不是模型。可以考虑拿它的 cost-model baseline 来对比。
- **MAPLE / MAPLE-Edge / MAPLE-X** (CVPRW 2022)：performance counter 作为连续硬件描述符 + 3-shot 延迟适配。是"连续硬件向量"的先例。
- **nn-Meter** (MobiSys 2021)：kernel 级 edge 设备延迟，per-device 预测器，无跨硬件。
- **BRP-NAS** (NeurIPS 2020)：GCN on NAS-Bench-201，per-device，无泛化。

### 1.3 Tier 3 — LLM 特定但偏分析式（不是学习模型）

- **Vidur** (MLSys 2024)：LLM 推理**模拟器**，per-operator profiling + 简单回归 + scheduler 模拟，<9% 端到端误差。**互补而非竞争** — 可以把本工作定位为"Vidur-style 模拟器在未见 GPU 上的学习型预测器"。
- **LLMCompass** (ISCA 2024)：分析式 LLM 推理硬件设计空间 simulator，4.1% 误差。
- **GenZ-LLM** ([arXiv:2406.01698](https://arxiv.org/abs/2406.01698))：分析式 LLM 平台需求工具。
- **Helix** (ASPLOS 2025)：异构 GPU LLM **serving** via max-flow。调度，不是预测。可以是你模型的消费者。

### 1.4 Novelty 判决

**已被做过，不要当主打创新**：

- 学习型 DNN 延迟预测（2020-2023 多篇）
- 跨硬件 + 连续硬件特征（NeuSight、MAPLE）
- GNN 做延迟预测（PerfSage、BRP-NAS、Akhauri 2024）—— 但他们**都没有把 GNN + 连续硬件规格向量 + LLM 图**这三个组合起来，你们的组合是新的

**可以理直气壮的差异化**：

1. **真正的零样本硬件泛化**：用连续硬件规格向量预测，不需要在新硬件先采样做初始化。Akhauri 2024 是"device-ID embedding + 相似训练设备初始化"（本质少样本），HELP 需要一组参考架构 latency，Habitat 需要 source-GPU trace —— 只有 NeuSight 和我们是 spec-only zero-shot，而 NeuSight 没有 GNN 且止步 H100
2. LLM 家族计算图（真实 attention + MLP block），不是 NAS cell 或 TVM 随机 program
3. 跨代数据中心 GPU：V100→A100→H100→H200→**B200**。B200/Blackwell 对所有现有工作都 OOD（NeuSight 停在 H100）
4. LLM 推理配置 (batch, seq_len) 作为一等公民的 s 向量
5. 显式 Roofline + XGBoost 对比（大部分学习预测器论文跳过或草率处理）

**建议重写 pitch**：
> *A sample-efficient GNN cost model that extrapolates transformer prefill (forward-pass) latency across GPU generations using only published hardware specs, enabling zero-shot prediction on Blackwell from Hopper/Ampere/Volta training data.*

把 NeuSight、HELP、Akhauri-Abdelfattah 2024 列为你建立其上的三个基础工作，明说借鉴了"GNN + 每节点 vs post-readout 硬件融合"这类设计维度；同时明确指出你们在**硬件表达从离散 embedding 换到连续 spec 向量**是关键差异。

---

## 二、方法论关键风险

### 2.1 "硬件泛化" 其实是 3 点回归 — 最严重的方法论问题

**现象**：[hardware.py:47-53](src/hetero_cost_model/hardware.py#L47-L53) 训练集只有 3 个 h 向量。[gnn.py:89-93](src/hetero_cost_model/models/gnn.py#L89-L93) 把 5 维 h concat 到 256 维图表示后进 MLP head —— MLP 在 5 维空间记住 3 个点 + 线性插值，这不是"学习硬件缩放"。

**后果**：
- **H200 zero-shot 几乎是白送的**：H200 vs H100 只有 `memory_gb` (80→141) 和 `bandwidth_gbs` (3350→4800) 不同，`fp16_tflops/pcie_gbs/sm_count` 完全相等。小 seq_len 下延迟本就 ≈ H100，模型即使完全忽略这两维都能通过评估。
- **B200 在 FP16 TFLOPS 上是 H100 的 2.3 倍**，ReLU MLP 的纯外推没有任何约束，输出是任意的。

**必须做的验证**（否则泛化 claim 站不住）：
1. **Leave-one-GPU-out CV**（V100/A100/H100 三折）作为主 claim
2. **Constant-h 消融**：h 固定成训练集均值，看 MAPE 变化 —— 若几乎不变，h 分支就是摆设
3. **仅硬件-MLP 基线**（去掉图）：若和 GNN 持平，GNN 没在做事
4. **h 合成扰动单调性测试**：compute-bound op 上 TFLOPS↑ 应 ⇒ 预测延迟↓

### 2.2 270 样本 + hidden=128 + 严重数据泄漏

6 模型 × 5 GPU × 9 配置 = 270 条 ([data_collection_plan.md:55](data_collection_plan.md#L55))。

- GAT 有 3 层 × hidden 128 × 4 heads + 256→128→1 head，参数量 100K+
- **关键泄漏**：每个图 G 同时出现在 train 和 test，只有 h 变。GNN 可以把每个图记成指纹，仅把 h 当 GPU ID 查表

**修复**：
- 除 leave-one-GPU-out 外，**一定加 leave-one-model-out**
- 强正则（hidden 32–64，dropout 0.3+，GAT 层数 ≤ 2）
- 加 "per-graph mean + learned GPU offset" 基线 —— 若它和 GNN 持平，message passing 就没起作用
- 目标数据量至少提到 1000+ 条（通过加 seq_len=512、加 batch_size=2/16、加随机 hold-out 配置）

### 2.3 Roofline 是 straw-man baseline

[baselines.py:28-34](src/hetero_cost_model/baselines.py#L28-L34) 的实现有多个硬伤：
- `node.memory_bytes = numel * 4` 硬编码 fp32（[extractor.py:74](src/hetero_cost_model/graph/extractor.py#L74)），但 profiling 跑 fp16，内存项 2× 偏大
- 用 dense tensor-core 理论峰值，小 batch × seq 永远打不到
- 没有 kernel launch overhead、没有 attention 二次项的正确 FLOPs（[flops.py](src/hetero_cost_model/graph/flops.py) 中 attention 只算 `4 * out_n`）

**必须换成强基线**：**per-GPU XGBoost on 全局手工特征**（total_flops、total_bytes、batch、seq_len、5 维 h）。文献里这种通常能到 10–15% MAPE，是真正的 bar。30 分钟能加上，报告可信度显著提升。Roofline 可以保留作为次要参考。

### 2.4 Seq_len 语义与 "LLM inference" 题目对不上

[data_collection_plan.md:240-242](data_collection_plan.md#L240-L242) 明说：decoder-only 只做单次 forward，**不做 autoregressive decode**。

但真实 LLM serving cost 由 O(N) decode 主导，KV cache 驱动。Roofline 完全不同：
- **Prefill**：compute-bound
- **Decode**：memory-bandwidth-bound

你现在预测的是 **transformer forward-pass / prefill latency**，不是 LLM inference latency。

**团队决策（2026-04-17）**：采用**诚实收窄**路线，测量目标统一措辞为 "end-to-end latency of a single forward pass over the full computation graph"。这和 proposal 的 "end-to-end latency" 兼容（此处 end-to-end 指"over computation graph"，不是"over serving pipeline"），同时避免了 decode 带来的 BERT/T5 encoder 不适用、KV cache 工程量大、图结构不一致三重问题。

Related work 里对 Helix/vLLM 的引用相应调整为"同方向不同 scope"（serving pipeline end-to-end 而非 single forward pass），不作为直接对标。decode 写在 Limitations / Future Work。

---

## 三、数据收集风险（代码里已经埋的坑）

### 3.1 `torch.fx.symbolic_trace` 在 HF 模型上大概率炸 —— **最高优先级隐患**

[extractor.py:39](src/hetero_cost_model/graph/extractor.py#L39) 用裸 `fx.symbolic_trace(model)`。这在 GPT-2/BERT/T5 上会因 `if input_ids is not None`、causal mask 构造、tuple 返回而挂掉。

**正确调用**：`transformers.utils.fx.symbolic_trace(model, input_names=["input_ids"])`。即便如此，T5 带 `past_key_values` 时还是会失败。

**Day 1 smoke test**（30 行脚本，CPU 上跑）：对 6 个模型全部调用 `extract_graph`，fail loudly。这是**能直接报废两周的单点故障**。

### 3.2 FP16 跨代不可比

- V100 有 FP16 tensor core 但没 BF16
- **Flash-Attention-2 不支持 V100** —— HF attention 在 V100 上走的 kernel 和 H100 根本不是一个东西
- 你不是在测同一段计算

**修复**：
- 全部显式 `attn_implementation="eager"`，报告里声明是 eager-attention 延迟
- **绝对不要用 BF16**（V100 会软件模拟）
- 在 CSV schema 里加 `attn_impl`、`kernel_backend` 列

### 3.3 测量方法问题

| 问题 | 代码位置 | 修复 |
|------|---------|------|
| 用 `time.perf_counter()` 测 CUDA 异步，混入 Python dispatch overhead | [profiling.py:51](src/hetero_cost_model/profiling.py#L51) | 换成 `torch.cuda.Event(enable_timing=True)` 对 |
| warmup=10 对 <5ms 小 batch 不足 cuDNN autotune | [profiling.py:38](src/hetero_cost_model/profiling.py#L38) | warmup=50, runs=100 |
| PSC 单卡分配下 host 共享（CPU/PCIe/mem BW）抖动 | SLURM 脚本 | 不要 `--exclusive`（需 8 卡预约，不值）。用 `taskset -c` 绑核 + CUDA events 即可 |
| `std/mean < 0.05` 质检标准在 shared host 上常失败 | [data_collection_plan.md:155](data_collection_plan.md#L155) | 保留噪声行但打 `noisy=True` 标签，训练时过滤；用 **p50** 作为训练 target，长尾异常自然被中位数滤掉 |

**关于"shared vs exclusive"的校准**：PSC 单卡分配时 GPU 本体（SM/HBM/L2）是独占的，only host-side（CPU、主机内存带宽、PCIe root complex）是 shared。对 forward pass > 50ms 的中大点，CoV < 1-2%；对 < 5ms 的小点（batch=1/seq=64），CoV 可能到 5-15%，这是真正要警惕的区间。**用 CUDA events 替换 `perf_counter` 基本把 CPU 抖动隔离掉**，比 8 卡独占代价低得多。

### 3.4 Modal SKU 自动升配陷阱 —— 硬件标签会错

Modal **是独占的**（每容器拿整块物理 GPU，无 MIG、无分时），但有一个会污染数据的默认行为：**缺货时 Modal 会悄悄升配**，按低档价格计费：

| 你声明的 | 实际可能拿到 | 后果 |
| --- | --- | --- |
| `gpu="H100"` | H100 或 **H200** | h 向量错、延迟错配 |
| `gpu="A100"` | A100-40GB 或 **A100-80GB** | 内存维度错 |
| `gpu="B200+"` | B200 或 **B300** | 规格错 |

[hardware.py:50](src/hetero_cost_model/hardware.py#L50) 的 `HARDWARE_REGISTRY` 用的是 A100-**40GB** 规格。若 Modal 给你 80GB 的卡，**硬件特征向量错了，训练目标也错了**，整条数据行污染。

**修复**（必做，一行改动）：

- 在代码里强制用带锁的 SKU 名：`gpu="H100!"`、`gpu="A100-40GB"`、`gpu="B200"`（不要 `B200+`）
- Profiling 脚本开头打印并写入 CSV：

  ```python
  props = torch.cuda.get_device_properties(0)
  row["actual_gpu_name"] = torch.cuda.get_device_name()
  row["actual_mem_gb"] = props.total_memory / 1e9
  row["actual_sm_count"] = props.multi_processor_count
  ```

- 加载 CSV 训练时，**按 `actual_gpu_name` 分组**，不按脚本声明的 gpu label 分组

另外两个 Modal quirks 要注意：

- **默认可抢占（preemptible）**：中途被 kill 容易漏写 CSV。脚本做成 idempotent（skip 已存在行），比花 3× 价格买 non-preemptible 值
- **Modal H100 是 SXM 版**（700W、NVLink），PSC H100 大概率也是 SXM。[hardware.py:48](src/hetero_cost_model/hardware.py#L48) 的 `pcie_gbs=64` 对 SXM 变体意义不大（它走 NVLink），两边口径要统一 —— 要么都填 PCIe 带宽，要么都换成 NVLink 互联带宽

### 3.5 Eager 模式是已接受的 scope 限制（非风险）

HF eager 延迟被 Python / dispatch overhead 主导（GPT-2 small 一次 forward 数百个 kernel launch，每个 5–20μs）。在真实部署（vLLM / TensorRT-LLM）下 kernel 选择不同、Python overhead 被消除，**GPU 之间的 ranking 都可能翻转**。

**团队决策（2026-04-17）**：proposal §4 明确声明 "we do not compare with torch.compile() version"，团队决定**严格遵守**，所有 profiling 用 HF eager + `attn_implementation="eager"`，不加 `torch.compile` 作为第二 target。

这不是风险，是**已接受的 scope 限制**。在 report 的 Limitations 段落明说：

> *We measure HF eager-mode latency with `attn_implementation="eager"` to ensure a single, portable kernel path across GPU generations (Flash-Attention-2 does not support V100). Generalization to compile-mode / vLLM / TensorRT-LLM serving stacks is out of scope and left as future work.*

### 3.6 其他细节

- **`peak_memory_mb` 跨平台不可比**：依赖 `PYTORCH_CUDA_ALLOC_CONF`、分配器版本、CUDA driver 版本（PSC vs Modal 不同）。只用来做 OOM 过滤，不能作为 target
- **CSV schema 扩充**：加 `kernel_backend`、`attn_impl`、`cuda_version`、`driver_version`、`actual_gpu_name` / `actual_mem_gb` / `actual_sm_count`。事后切片调试时救命
- **数据量目标**：加 seq_len=512、加 batch_size=2/16、加随机 (bs, sl) hold-out 点，把总量推到 1000+ hero 样本 + 500 交叉验证样本

---

## 四、头两天必须做的两件事（其他都次要）

### (A) Day 1：fx tracing 验证（不上 GPU 之前）

1. `transformers.utils.fx.symbolic_trace` 替换 [extractor.py:39](src/hetero_cost_model/graph/extractor.py#L39)
2. 写 smoke test 跑 gpt2-small/medium/large、bert-base/large、t5-small
3. T5 若挂，**当场决定**：丢掉 T5 还是改 ONNX 导出路径
4. 不先解决这个就调度 GPU 是浪费排队时间

### (B) 重新设计"硬件泛化"的评估协议

- **主 claim**：leave-one-GPU-out CV（V100/A100/H100 内部）—— 有 3 折才是真研究结论
- **H200 改成 few-shot**（zero-shot 在只有 3 个 anchor 时不是研究 claim）
- **B200 作为 hero zero-shot**，但必须在同一张表里报告：
  - Constant-h 基线（h 固定成均值）
  - HW-MLP only 基线（去掉图）
  - Per-graph mean + learned GPU offset 基线
- **主 baseline 换 XGBoost**（全局手工特征），Roofline 留作次要参考

做了这两件，报告里的结果就从"数字看起来好，但原因存疑"变成"方法自洽、数字可解释"——对课程 project 的评分贡献最大。

---

## 五、数据收集建议清单

1. **总量提升**：从 270 → 1000+ hero + 500 hold-out。新增 seq_len=512、batch_size=2/16、随机 (bs, sl) 组合
2. **精度与 kernel 统一**：fp16 + `attn_implementation="eager"`。**不加 `torch.compile`**（遵循 proposal §4 声明），report Limitations 里说明
3. **测量**：CUDA events（替换 `perf_counter`）、warmup=50、runs=100、随机配置顺序、`taskset -c` 绑核。**不要申请 `--exclusive`**（PSC 需要 8 卡整机预约，代价太高；single-GPU + CUDA events 就够了）
4. **SKU 锁定**：Modal 上用 `gpu="H100!"`、`gpu="A100-40GB"`、`gpu="B200"` 避免自动升配；所有 session 写入 `actual_gpu_name` / `actual_mem_gb` / `actual_sm_count` 到 CSV，训练时按实际卡型分组
5. **Schema 扩充**：加 `attn_impl / cuda_version / driver_version / kernel_backend / actual_gpu_name / actual_mem_gb / actual_sm_count`
6. **可比性交叉检查**：同 (model, bs, sl) 在 5 个 GPU 上的 latency 必须**单调于 FP16 TFLOPS**（compute-bound 区）或**单调于带宽**（memory-bound 区）。违反的点人工审查
7. **决策点**：小模型 + 大 batch 可能被 kernel launch 完全支配（Python overhead dominates），profile 时排除或单独分析

---

## 六、如果时间真的来不及的 fallback 定位

把评估从 "zero-shot to B200" 降级到 "**sample-efficient cross-GPU-generation extrapolation**"：

- 主 claim：leave-one-GPU-out + H200 few-shot + XGBoost 对比
- B200 作为 honest stretch goal，外推效果差也能作为 negative finding 写进报告

这样即使 B200 外推不理想，也仍然是一份完整、方法自洽的课程 project 报告。

---

## 附录：引用清单（按与本项目相关度排序）

1. [NeuSight, ASPLOS '25](https://arxiv.org/abs/2407.13853) — must cite and differentiate
2. [On Latency Predictors for NAS, MLSys '24](https://proceedings.mlsys.org/paper_files/paper/2024/file/f03cb785864596fa5901f1359d23fd81-Paper-Conference.pdf) — 架构最近似的先例
3. [HELP, NeurIPS '21](https://arxiv.org/abs/2106.08630) — few-shot baseline 蓝图
4. [Habitat, USENIX ATC '21](https://www.usenix.org/conference/atc21/presentation/yu) — zero-shot GPU transfer 参考
5. [MAPLE, CVPRW '22](https://arxiv.org/abs/2111.15106) — 连续 HW 描述符先例
6. [PerfSage, 2023](https://arxiv.org/pdf/2301.10999) — GNN 延迟预测先例
7. [TLP, ASPLOS '23](https://arxiv.org/abs/2211.03578) — TVM cost model
8. [TenSet, NeurIPS '21 D&B](https://openreview.net/forum?id=aIfp8kLuvc9) — 大规模 cost model 数据集
9. [Vidur, MLSys '24](https://arxiv.org/abs/2405.05465) — LLM 推理 simulator，互补
10. [LLMCompass, ISCA '24](https://parallel.princeton.edu/papers/isca24_llmcompass.pdf) — 分析式 LLM simulator
11. [Helix, ASPLOS '25](https://arxiv.org/abs/2406.01566) — 异构 GPU LLM serving
12. [nn-Meter, MobiSys '21](https://www.microsoft.com/en-us/research/wp-content/uploads/2021/05/nn-Meter-Mobisys21.pdf) — edge kernel 延迟
13. [BRP-NAS, NeurIPS '20](https://proceedings.nips.cc/paper/2020/file/768e78024aa8fdb9b8fe87be86f64745-Paper.pdf) — 早期 GCN 延迟预测
