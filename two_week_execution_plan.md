# 跨异构 GPU 的 LLM 推理延迟泛化代价模型

## 两周冲刺执行计划

**团队成员**：Xiang Li (xli8) | Xinhao Tan (xinhaota) | Zhikai Hu (zhikaih)

**时间跨度**：2026年4月14日 – 4月28日

**硬件资源**：V100 (PSC) | A100-40GB (Modal) | H100 (PSC) | H200 (Modal) | B200 (Modal)

---

## 1. 总体策略

原计划为4周，现压缩至2周。核心目标不变：构建一个图级别(graph-level)的学习代价模型 **f(G, s, h) → T̂**，给定模型计算图 G、推理配置 s（batch size、seq len）、目标 GPU 硬件向量 h，预测端到端推理延迟，并在不同工作负载和未见过的 GPU 上展示泛化能力。

**项目性质定位**：CMU 15-442 MLSys 课程期末 project，不是论文投稿。架构和实现可以借鉴已有工作（NeuSight、Akhauri 2024、HELP 等），重点在方法自洽、数据干净、有一小块自己的差异化贡献（连续硬件规格向量 + LLM 图 + 跨代外推到 Blackwell 的组合）。详细评审见 [research_review.md](research_review.md)。

**相对 proposal 的 scope 4 项决策**（2026-04-17 团队对齐，报告 Intro 必须显式声明）：

1. **异构目标**：原 proposal 是 CPU/GPU operator placement；本项目收窄为 **纯 GPU cross-generation**（V100→A100→H100→H200→B200）。`s` 从 operator placement 改为 (batch_size, seq_len) inference config
2. **延迟含义**：测量 **end-to-end latency of a single forward pass over the full computation graph**（= prefill stage）。Autoregressive decode 不测，列入 Limitations / Future Work
3. **工作负载**：**Transformer 家族**（GPT-2 / BERT / T5）。ResNet 虽在 proposal 中提到，本项目不做
4. **torch.compile**：遵循 proposal §4 声明 "we do not compare with torch.compile() version"，**全部用 HF eager mode + `attn_implementation="eager"`**。Limitations 段声明：测量范围限于 HF eager-mode latency；vLLM / TensorRT-LLM 等 serving 栈不在 scope 内

**关键调整思路**：

- 三人全程并行，分别负责数据管线、模型训练、实验评估三条主线
- GPU 排队时间宝贵，所有 profiling 脚本必须在本地完全调试通过后再提交
- 将 profiling 任务设计为"一次排队、批量收集"模式，一次 GPU 预约覆盖一个模型族的所有配置
- 报告写作从第10天就开始，不留到最后

---

## 2. 当前研究现状（关键参考文献）

以下最新工作直接影响我们的技术方案设计和贡献定位：

### 2.1 CALO-GNN (KDD 2025 Workshop)

- **核心思路**：首个基于图神经网络的 TVM 代价模型，提供校准的认识不确定性估计（evidential GNN）；采用两阶段迁移方法，利用400万历史调度记录，仅需2000次测量即可适配新设备
- **关键指标**：跨设备自动调优时间减少30%以上
- **与我们的区别**：CALO-GNN 在算子(kernel)级别运行，为 TVM MetaSchedule 的调度搜索服务。我们在计算图(graph)级别操作，预测端到端延迟，并将硬件建模为连续特征向量

### 2.2 Felix (ASPLOS 2024)

- **核心思路**：基于梯度下降的张量程序优化框架，构建可微分的延迟预测函数，替代传统的组合搜索
- **关键指标**：达到90%峰值性能的搜索时间比 Ansor 快5.8倍
- **与我们的区别**：Felix 聚焦于单设备上的 schedule 调优。我们关注跨 GPU 硬件的端到端延迟预测，是更高层次的硬件选择和配置决策问题

### 2.3 Helix (ASPLOS 2025)

- **核心思路**：将异构 GPU 集群上的 LLM 推理建模为最大流问题，使用混合整数线性规划(MILP)联合优化模型放置和请求调度
- **关键指标**：吞吐提升最高3.3倍，prompt延迟降低最高66%
- **与我们的区别**：Helix 使用分析/MILP方法，我们使用学习模型来预测代价，能泛化到未见过的硬件

### 2.4 Q-Infer (ACM TACO 2025)

- **核心思路**：混合 GPU-CPU 异构并行推理，结合稀疏性感知的自适应动态调度
- **与我们的区别**：Q-Infer 针对 CPU+GPU 混合执行设计手工调度规则；我们聚焦于跨 GPU 硬件代际的端到端延迟泛化预测，问题设定不同

### 2.5 Ansor / MetaSchedule (OSDI'20 / NeurIPS'22)

- **地位**：TVM 中学习代价模型的基础工作，算子级别的代价模型驱动调度搜索
- **与我们的区别**：逐算子、逐任务在线学习；我们追求跨工作负载和跨硬件的图级别泛化

---

## 3. 硬件资源与特征向量设计

五种 GPU 横跨四代架构（Volta → Ampere → Hopper → Blackwell），构成训练集（3个）和泛化测试集（2个）：

**6 个 training anchor + 2 个 test target**（全部 Modal 覆盖；PSC 可作冗余备份）：

| 特征 | V100 | T4 | A100-40GB | A10 | L4 | H100 | H200 | B200 |
|------|------|------|-------|------|------|------|------|------|
| **角色** | 训练 | 训练 | 训练 | 训练 | 训练 | 训练 | few-shot 测试 | **hero zero-shot** + few-shot 对照 |
| **架构 (sm)** | Volta (70) | Turing (75) | Ampere (80) | Ampere (86) | Ada (89) | Hopper (90) | Hopper+ (90) | Blackwell (100) |
| FP16 TFLOPS | 125 | 65 | 312 | 125 | 121 | 1,979 | 1,979 | ~4,500 |
| HBM / 内存 | 32 GB | 16 GB | 40 GB | 24 GB | 24 GB | 80 GB | 141 GB | 180 GB |
| 显存带宽 | 900 | 320 | 1,555 | 600 | 300 | 3,350 | 4,800 | 8,000 GB/s |
| L2 cache | 6 MB | 4 MB | 40 MB | 6 MB | 48 MB | 50 MB | 50 MB | 96 MB |
| SM 数量 | 80 | 40 | 108 | 72 | 58 | 132 | 132 | 160 |
| **Modal SKU lock** | — | `T4` | `A100-40GB` | `A10` | `L4` | `H100!` | `H200` | `B200` |

**为什么 6 个 training anchor**：原计划只有 3 个（V100 / A100 / H100），5 维连续硬件向量在 3 个点上欠定（3 个点定义不了 5D 函数）。加到 6 个后覆盖 **5 个架构代际**（Volta, Turing, Ampere, Ada, Hopper），且 Ampere 内部有 2 个点（A100 sm_80 + A10 sm_86），测试"同 arch 不同 spec"的学习能力。论文里的"spec-only zero-shot 外推到 Blackwell"claim 站得住的前提。

**硬件特征向量 h（5 维）**：峰值 FP16 TFLOPS、HBM 容量、HBM 带宽、**L2 cache size**、SM 数量。所有数值做归一化处理（除以各维理论上限），使模型能通过连续物理特征感知硬件差异。**全部为片上 spec**，没有互联带宽、没有代际 ordinal——跟 NeuSight (ASPLOS'25) 的 on-chip-only 惯例一致。

**为什么用 L2 cache 而不是 PCIe/NVLink 带宽**：PCIe / NVLink 这类互联带宽在单 GPU 的 forward-pass 测量窗口内不会被使用（输入和权重早在 GPU 上），作为预测特征是 confound；L2 cache 是片上特征，影响带宽放大和小 kernel 延迟，是真实有物理信号的维度。

**为什么不加"架构代际" ordinal**：初版曾包含 `arch_gen` ∈ {0.00, 0.33, 0.67, 1.00} 四档，但 arch_gen 本质是"披着连续向量外衣的离散 device-ID"——MLP 可以把它当成 lookup 直接查表（H200 的 arch_gen 和 H100 完全一样，所以泛化被"白送"；B200 的 1.00 在训练从未出现，MLP 外推不受约束）。移除后模型只能依赖真实物理量（TFLOPS / 带宽 / L2 / SM 数），spec-only 零样本泛化的 claim 干净许多。即使 B200 外推效果不理想，这也是**有价值的 negative finding**（说明需要更多代际区分物理量），比伪装的 ordinal 诚实。

H200 与 H100 同架构但规格不同（HBM 容量 80→141、带宽 3350→4800；fp16_tflops/L2/SM 完全相同），是"同族内插值"泛化；B200 是全新架构，是"跨架构外推"泛化——两种难度的泛化测试使实验更有层次。

---

## 4. 技术方案（适配两周周期）

### 4.1 计算图表示

通过 ONNX 或 `torch.fx` 导出模型计算图，将每个算子节点转化为特征向量：

- 算子类型（one-hot 编码：Conv, MatMul, LayerNorm, Attention, ...）
- 输入/输出张量形状
- 估算 FLOPs
- 估算内存占用
- 数据类型

边表示数据依赖关系，整体构成一个有向无环图(DAG)，可直接输入 GNN。

### 4.2 推理配置编码

推理配置 s 定义为一次推理的服务参数，与模型结构无关：

- **batch size**：请求批大小（1 / 4 / 8 / 16）
- **sequence length**：输入序列长度（64 / 128 / 256）

s 编码为归一化的 2 维全局向量，在图级别 readout 之后拼接到 MLP head 的输入，而不广播到每个节点。这样节点特征只编码算子结构信息，s 和 h 在全局层面影响延迟预测，反映它们的物理意义。

**排序意义**：对同一模型 G，不同 (s, h) 组合的延迟排序即为"在哪块 GPU 上、用什么 batch 跑最快"的决策依据。

### 4.3 代价模型架构

我们采用 **GNN 为主架构、Graph Transformer 为对比变体** 的方案。

#### 主架构：GAT（Graph Attention Network）

```
输入: 标注计算图 G (节点特征 + 边)  +  配置 s=(bs,sl)  +  硬件向量 h
  │
  ├── 节点嵌入: 算子特征（op_type, shapes, FLOPs, memory）
  │
  ├── 3层 GAT, hidden_dim=128, 4 heads
  │
  ├── 图级别 Readout: mean pooling + max pooling 拼接 → 256维
  │
  ├── 拼接全局特征: [图表示 | s(2维) | h(5维)] → 263维
  │
  ├── 2层 MLP Head (263 → 128 → 1)
  │
  └── 输出: 预测延迟 T̂
```

**选择 GNN 的理由**：

- 计算图是天然的 DAG 结构，GNN 的消息传递沿数据依赖边进行，归纳偏置最匹配
- 不同模型的图拓扑差异极大（GPT-2 decoder-only vs BERT encoder-only vs T5 encoder-decoder），GNN 天然处理变长、不规则结构，无需 padding
- 数据量有限（~1500 条），GNN 的图结构先验更强，小样本下更不容易过拟合
- CALO-GNN 等最新工作已验证 GNN 在代价模型任务上的有效性

#### 对比变体：Graph Transformer

```
输入: 同上
  │
  ├── 节点嵌入 + Laplacian 位置编码（编码图拓扑结构）
  │
  ├── 3层 Transformer Encoder (d_model=128, 4 heads, 带图结构偏置)
  │
  ├── [CLS] token 或 mean pooling 做图级别表示
  │
  ├── 2层 MLP Head
  │
  └── 输出: 预测延迟 T̂
```

**Graph Transformer 的潜在优势**：全局自注意力可一步捕获长距离算子依赖（GNN 需要多层堆叠，存在过度平滑风险）。我们将其作为消融实验的一个变体，对比 GNN 和 Graph Transformer 在预测精度、训练效率和泛化能力上的差异。如果数据量不足以支撑 Transformer 的参数量，可通过减小 d_model 或层数来控制。

**损失函数**：MSE Loss + λ × Pairwise Ranking Loss（确保策略排序正确性）

**实现框架**：PyTorch Geometric (PyG)，Graph Transformer 可基于 PyG 的 `TransformerConv` 或独立实现

### 4.4 训练策略

- **Phase 1**：在 V100 + T4 + A100 + A10 + L4 + H100 数据上训练，学习跨 GPU 代际的预测能力
- **Phase 2**：在 H200 上做 **few-shot** 评估（50/100/200 samples，同架构不同规格，测试插值泛化）；在 B200 上做 **hero zero-shot** 评估（全新架构，测试外推泛化）；然后用少量 B200 样本做 few-shot 微调，测量改善幅度
  > H200 改用 few-shot 而非 zero-shot 的原因：H200 在 h 向量中与 H100 只有 `memory_gb` 和 `bandwidth_gbs` 不同，compute-bound 区延迟本就 ≈ H100，zero-shot 时几乎是白送（即使现在有 6 个 training anchor，H200 仍和 H100 的 3 维 h 完全相同，同族插值缺乏挑战性）

---

## 5. Baseline 设计

本项目涉及两类 baseline 对比：**推理性能 baseline**（代价模型选出的策略要跟谁比延迟）和**代价模型 baseline**（GNN 预测器要跟什么预测方法比精度）。

项目的核心贡献在于代价模型的**预测精度**和**泛化能力**，因此 baseline 聚焦在代价模型层面。选择 baseline 时要避免 straw-man（例如只放一个弱公式），至少要有一个有竞争力的非图学习方法作为主对比。

| Baseline | 做法 | 实现位置 | 对比意义 | 状态 |
| --- | --- | --- | --- | --- |
| **Roofline 分析模型** | 延迟 = Σᵢ max(FLOPsᵢ / 峰值TFLOPS, bytesᵢ / 带宽) | [baselines.py:roofline_latency](src/hetero_cost_model/baselines.py) | 次要参考，证明"学习模型胜过手写公式"；`memory_bytes` 已按 FP16 修正（2 bytes/element） | ✅ |
| **Per-graph mean + learned GPU offset** | ŷ(m, g) = graph_mean[m] + gpu_offset[g]；不访问图结构 / op 特征 / (bs, sl) | [baselines.py:PerGraphMeanBaseline](src/hetero_cost_model/baselines.py) | **数据泄漏诊断**。GNN 必须显著优于它，否则 message passing 和硬件特征都是装饰 | ✅ |
| **XGBoost on global features** | 12 维手工特征（log1p(total_flops)、log1p(total_bytes)、num_nodes、num_edges、batch、seq_len、5 维 h）→ GBDT | [baselines.py:XGBoostBaseline](src/hetero_cost_model/baselines.py) | **强 baseline**。文献里通常 10-15% MAPE，GNN 需要至少 3-5 MAPE 点击败 | ✅ |

**统一驱动脚本**：`python scripts/train_and_eval.py --csv <csv> --split <spec>` 一次性跑 Roofline / Per-graph mean / XGBoost / GNN，输出 MAPE / Spearman ρ / Top-1 并列表。`--split` 支持 `random` / `leave-gpu=<key>` / `leave-model=<name>` / `zero-shot=<key>`。

### 5.2 研究 claim 与实验表的映射

论文要 defend 三条 claim，每条 claim 对应**唯一一张表**，避免实验发散：

| # | Claim | 支撑表 | 如果失败的 negative finding |
|---|---|---|---|
| **C1** | 学习型代价模型在已见硬件上优于分析式和朴素基线 | Table 1 | "Roofline + 手工特征就够了" |
| **C2** | 连续硬件规格向量 + GNN 能从 V100/A100/H100 零样本外推到 Blackwell | Table 2 | "spec-only zero-shot 在跨代架构上失效" |
| **C3** | 图结构（message passing）对 prefill 延迟预测有边际贡献 | Table 3 | "per-op 特征就够，图结构是装饰" |

**原则**：每个数字都必须直接回答某个 claim 的 "yes / no"；不产出"看起来有趣但不支撑 claim"的数字。

#### Table 1 — 预测精度（主结果，支撑 C1）

5 method × 5 split = 25 格。每格报 MAPE / Spearman ρ / Top-1 三个指标。

| Method \ Split | Random 80/20 | Leave-one-GPU-out (V100) | Leave-one-GPU-out (A100) | Leave-one-GPU-out (H100) | Leave-one-model-out (avg 6 folds) |
|---|:---:|:---:|:---:|:---:|:---:|
| Roofline | | | | | |
| Per-graph mean | | | | | |
| Graph-level pooled MLP | | | | | |
| XGBoost | | | | | |
| **GNN (GAT)** | | | | | |

**解读规则**：

- GNN 必须在所有 5 列击败 Roofline、Per-graph mean、pooled MLP → C1 成立
- GNN 和 XGBoost 的 gap ≥ 3-5 MAPE 点是 graph-level learning 的增益
- leave-one-GPU-out 三折必须分开报告，不报 avg 以揭示 outlier

**复现**：

```bash
python scripts/train_and_eval.py --csv data/raw/all.csv --split random
python scripts/train_and_eval.py --csv data/raw/all.csv --split leave-gpu=v100
python scripts/train_and_eval.py --csv data/raw/all.csv --split leave-gpu=a100
python scripts/train_and_eval.py --csv data/raw/all.csv --split leave-gpu=h100
for m in gpt2-small gpt2-medium gpt2-large bert-base bert-large t5-small; do
  python scripts/train_and_eval.py --csv data/raw/all.csv --split leave-model=$m
done
```

#### Table 2 — 硬件泛化（Hero，支撑 C2）

训练集固定 = V100 + T4 + A100 + A10 + L4 + H100，改变 target。

| Method \ Target | H100 hold-out<br/>(in-distribution) | H200 zero-shot<br/>(same arch interp.) | H200 few-shot (100) | **B200 zero-shot**<br/>(hero, cross-arch) | B200 few-shot (50/100/200) |
|---|:---:|:---:|:---:|:---:|:---:|
| Roofline | | | | | |
| Per-graph mean | | | | | |
| XGBoost | | | | | |
| **GNN** | | | | | |

**补充图表**：B200 few-shot 学习曲线，x 轴 N ∈ {0, 50, 100, 200}（log scale），y 轴 MAPE。

**解读规则**：

- B200 zero-shot MAPE < 30% → C2 成立
- B200 zero-shot > H200 zero-shot → "跨架构外推比同架构插值难"，符合假设
- B200 few-shot N=50 能消除 > 50% 的 zero-shot 误差 → 迁移学习有效

**复现**：

```bash
python scripts/train_and_eval.py --csv data/raw/all.csv --split zero-shot=h200
python scripts/train_and_eval.py --csv data/raw/all.csv --split zero-shot=b200
python scripts/train_and_eval.py --csv data/raw/all.csv --split zero-shot=h200 --few-shot-samples 100
python scripts/train_and_eval.py --csv data/raw/all.csv --split zero-shot=b200 --few-shot-samples 50
python scripts/train_and_eval.py --csv data/raw/all.csv --split zero-shot=b200 --few-shot-samples 100
python scripts/train_and_eval.py --csv data/raw/all.csv --split zero-shot=b200 --few-shot-samples 200
```

#### Table 3 — 消融（支撑 C3 + 辩护 C2）

只做 **2 行消融**，每行从 GNN 只去掉一个组件，跑在最重要的 2 个 split 上。

| Variant | 从 GNN 去掉 | 保留 | 支撑 |
|---|---|---|:---:|
| GNN w/o edges（= Per-kernel sum MLP） | 图结构（边 / message passing） | 节点特征 + s + h | **C3** 图结构边际贡献 |
| GNN w/o `h`（constant-h） | 硬件向量（替换为训练均值） | 图 + 节点 + s | **C2 诊断** h 向量是否真被学到 |
| **GNN (main)** | — | 全部 | 上限 |

表格形态：

| Variant | leave-one-GPU-out avg MAPE | B200 zero-shot MAPE |
|---|:---:|:---:|
| GNN w/o edges | | |
| GNN w/o `h` | | |
| **GNN (full)** | | |

**解读规则**（两组关键对比）：

- `GNN − (w/o edges)` 的 gap ⇒ 图结构的边际贡献。Gap 显著 → C3 成立；gap 为零 → **诚实报告**："图结构在本项目 scope 下无边际贡献，主要信号来自节点级特征"（这是可发表的 negative finding，比掩盖更有价值）
- `GNN − (w/o h)` 的 gap，尤其在 B200 zero-shot 上 ⇒ h 向量是否真承载硬件信息。Gap 为零 → C2 的 "spec-based 泛化" claim 站不住

**正文 sanity check（不占表格行）**：

- 人为把 h 中 TFLOPS 维度调高 2×，compute-bound op 的预测延迟应单调下降——通过率 > 80% 才算 h 分支物理意义合理
- 把 `s = (batch, seq)` 清零，MAPE 应显著恶化（验证 s 分支被使用，属于功能性检查）

**复现**：

```bash
# 需要先实现：scripts/train_and_eval.py 的 --model-variant={gnn,per_kernel_mlp} + --constant-h 开关
python scripts/train_and_eval.py --csv data/raw/all.csv --split leave-gpu=v100 --model-variant per_kernel_mlp
python scripts/train_and_eval.py --csv data/raw/all.csv --split leave-gpu=a100 --model-variant per_kernel_mlp
python scripts/train_and_eval.py --csv data/raw/all.csv --split leave-gpu=h100 --model-variant per_kernel_mlp
python scripts/train_and_eval.py --csv data/raw/all.csv --split zero-shot=b200 --model-variant per_kernel_mlp
python scripts/train_and_eval.py --csv data/raw/all.csv --split leave-gpu=v100 --constant-h
# ... 依此类推
```

### 5.3 明确不做的实验（scope 控制）

为避免 scope 发散，以下在 Week 1 初稿里显式排除，只在时间充裕时加回：

| 候选实验 | 为什么不做 |
|---|---|
| Graph Transformer vs GAT 主表对比 | 降级为正文一句话；代码已支持 `--backbone=transformer`，附录给数字即可，不占主表 |
| Per-node vs post-readout HW fusion（Akhauri 2024 风格） | 同上；作为"有想法没时间做"写进 Future Work |
| 移除推理配置 s（zero-s） | **不做**——不支撑任何 claim，属于 sanity check 范畴 |
| `arch_gen` ordinal 消融 | 已在 2026-04-17 从硬件向量移除，无此字段 |
| torch.compile / vLLM 对比 | Proposal §4 明确排除；写进 Limitations |
| ResNet 跨家族泛化 | Scope 收窄到 Transformer，写进 Limitations |

### 5.4 Report Section 5 Experiments 骨架（直接可抄）

```
5. Experiments
  5.1 Setup
      - Data: 1500+ samples across V100/A100/H100/H200/B200
      - Splits: random / leave-one-GPU-out / leave-one-model-out /
                zero-shot / few-shot
      - Metrics: MAPE / Spearman ρ / Top-1
      - Baselines: Roofline / Per-graph mean / Pooled MLP / XGBoost
  5.2 Main Results — supports C1 (Table 1)
  5.3 Cross-GPU-Generation Extrapolation — supports C2 (Table 2 + Figure 1)
  5.4 Ablations — supports C3 and diagnoses C2 (Table 3)
      - GNN vs Per-kernel MLP: graph structure contribution
      - GNN vs constant-h GNN: hardware vector contribution
  5.5 Limitations
      - HF eager mode only (no torch.compile / vLLM)
      - Transformer family only (no CNN / other archs)
      - Forward-pass / prefill only (no decode / KV cache)
```

---

## 6. 两周详细时间线

### Week 1：基础设施 + 数据收集 + Baseline（4月14日–20日）

#### Day 1–2（周一–周二）：基础设施搭建 + 硬阻塞项先验证

**Day 1 硬阻塞项 —— ✅ 已完成 2026-04-17**：

- ✅ [extractor.py](src/hetero_cost_model/graph/extractor.py) 新增 `hf_input_names` 参数，走 `transformers.utils.fx.symbolic_trace`
- ✅ [scripts/smoke_test_graphs.py](scripts/smoke_test_graphs.py) 对 6 个模型全量 CPU 调用 `extract_graph`，**6/6 PASS**（gpt2-small: 1247 nodes, gpt2-medium: 2471, gpt2-large: 3695, bert-base: 529, bert-large: 1021, t5-small: 1159）
- ✅ T5 通过（需要 `["input_ids", "decoder_input_ids"]` + `T5ForConditionalGeneration` + `attn_implementation="eager"`）
- ✅ `transformers` 版本 pin 到 `>=4.35,<4.52`（4.46.3 已验证；v5 删 `transformers.utils.fx`，4.52+ 的 `masking_utils.py` 用 vmap 和 HFProxy 不兼容）

**守护措施（必须保留）**：不要升级 `transformers` 到 ≥ 4.52；每次环境重装 / 新成员入组后先跑 smoke test；加新模型时先在 `MODELS` 列表里加一条验证。详见 [research_review.md §3.1](research_review.md)。

| 负责人 | 任务 | 交付物 | 需要GPU | 状态 |
|--------|------|--------|---------|------|
| **Xiang Li** | (a) fx smoke test；(b) 用 HF fx 修复 [extractor.py](src/hetero_cost_model/graph/extractor.py) 节点特征提取；(c) 增加 FP16 精度下 `memory_bytes` 的正确估算；(d) `scripts/extract_graphs.py` 图序列化 + 6 个 pkl committed；(e) `HARDWARE_REGISTRY` 校对（pcie_gbs→l2_cache_mb，移除 arch_gen）；(f) 本地端到端冒烟 | `graph_extractor.py` + smoke test + `extract_graphs.py` + `end_to_end_smoke.py` + `hardware.py` | 否 | ✅ 完成 2026-04-17 |
| **Xinhao Tan** | 构建 profiling 框架：(a) CUDA events 计时 ✅；(b) warmup=50、runs=100 ✅；(c) 训练 target 改用 **p50** ✅；(d) 全部模型 `attn_implementation="eager"` ✅（在 `model_zoo.py` 里统一）；(e) `actual_gpu_name` / `actual_mem_gb` / `actual_sm_count` 写入 CSV ✅（`runtime_info.py`）；(f) 驱动脚本 `scripts/run_profiling.py`（idempotent + OOM-safe + 20 列 schema）| `profiling.py` + `runtime_info.py` + `scripts/run_profiling.py` | 否（本地 CPU 调试） | ✅ 完成 2026-04-17 |
| **Zhikai Hu** | (a) GNN 代价模型（GAT + Graph Transformer 变体）已存于 [models/gnn.py](src/hetero_cost_model/models/gnn.py)；(b) 硬件特征 schema 已敲定（5 维 L2-cache 版）；(c) 训练循环 MSE + ranking loss 已存于 [training/loop.py](src/hetero_cost_model/training/loop.py)；(d) **XGBoost baseline** 新增 [baselines.py:XGBoostBaseline](src/hetero_cost_model/baselines.py)；(e) **Per-graph mean 诊断 baseline** 新增；(f) CSV→Sample 管线 [data.py:load_samples_from_csv](src/hetero_cost_model/data.py)；(g) 一键评估驱动 `scripts/train_and_eval.py`（4 种切分） | `cost_model.py` + `train.py` + `baselines.py` + `data.py` + `train_and_eval.py` | 否 | ✅ 完成 2026-04-17 |

**Day 2 检查点**：三人在本地完成各自模块的单元测试，确认接口可以对接。**全部 绿灯**：pytest 22/22 + smoke test 6/6 + end-to-end 8/8 + train_and_eval dry-run 4 baseline 并列输出均正常。下游代码 pipeline 完备，等 Day 3 真数据。

#### Day 3–4（周三–周四）：数据收集冲刺

| 负责人 | 任务 | 交付物 | 需要GPU |
|--------|------|--------|---------|
| **Xiang Li** | 【V100, PSC】Profile：6 个模型 × (batch ∈ {1,2,4,8,16}) × (seq ∈ {64,128,256,512}) + 随机 hold-out 点。单卡分配 + `taskset -c` 绑核（**不要 `--exclusive`**，PSC 需要 8 卡整机预约）。目标：≥ 300 样本 | `v100.csv` | V100 |
| **Xinhao Tan** | 【H100, PSC】同配置矩阵；Modal 端用 **锁定 SKU**：`gpu="A100-40GB"` 避免被升到 80GB。目标：≥ 600 样本（H100+A100 合计） | `h100.csv` + `a100.csv` | H100 (PSC) + A100 (Modal) |
| **Zhikai Hu** | 数据管线：合并 CSV、特征归一化、构造 PyG Data 对象；**按 `actual_gpu_name` 分组而不是脚本声明的 gpu label**（防 Modal 升配污染）；准备 train/val/test 切分 | `dataset.py` + `data/` | 否 |

**关键提示**：

- Profiling 配置矩阵统一，所有 GPU 跑完全相同的 (模型, batch_size, seq_len) 组合，结果 CSV 只有 gpu 列不同，便于直接对比
- Modal job 即提交即运行，A100 数据与 H100 数据可并行收集
- **Modal 必须用锁定 SKU**（`H100!` / `A100-40GB` / `B200`，不要 `H100` / `A100` / `B200+`）——缺货时 Modal 会悄悄升配，`HARDWARE_REGISTRY` 里的规格就对不上了
- 配置矩阵按 [data_collection_plan.md](data_collection_plan.md) 执行，总量目标从原 270 提升到 **1000+ hero + 500 hold-out**

#### Day 5–7（周五–周日）：Baseline 训练与验证 → 产出 Table 1（主结果）

产出目标是 §5.2 **Table 1** 的完整 5 method × 5 split 格子（见 §5.2）。三种 baseline（Roofline / Per-graph mean / XGBoost / Pooled MLP）已在 Day 1-2 由 Zhikai 实现，这里只是用真数据跑一遍。

| 负责人 | 任务 | 交付物 | 需要GPU |
|--------|------|--------|---------|
| **三人协作** | 跑 Table 1 全部 25 格：`train_and_eval.py` 依次过 5 个 split（random / leave-gpu × 3 / leave-model avg 6 folds）；每个 split 自动输出 4 method（GNN / XGBoost / Per-graph mean / Roofline）并列对比；pooled MLP 加 `--model-variant pooled_mlp` 走一遍 | Table 1 完整 5×5×3 数字（MAPE / Spearman / Top-1） | 训练很轻，CPU 可以 |
| **Xiang Li** | 整合所有数字生成报告用 LaTeX 表格模板；若发现 outlier fold（某折 MAPE 异常大）做诊断 | `table1_main_results.tex` | 否 |

> **Week 1 里程碑（C1 判据）**：GNN 在所有 5 split 上击败 Roofline / Per-graph mean / Pooled MLP；对 XGBoost 至少 3-5 MAPE 点优势。leave-one-GPU-out CV 必须报三折具体数字而非 avg。若 C1 不成立 → **不强行刷数字**，在 Day 7 复盘日决定是 fallback 到更小 scope 还是调整模型。

---

### Week 2：泛化实验 + 消融研究 + 报告撰写（4月21日–28日）

#### Day 8–9（周一–周二）：B200 数据 & 硬件泛化

| 负责人 | 任务 | 交付物 | 需要GPU |
|--------|------|--------|---------|
| **Xiang Li** | 【Modal】同时提交 H200 + B200 profiling job（锁定 SKU：`gpu="B200"`，不是 `B200+`）。数据拆分：**H200 改为 few-shot 设置**（50/100/200 samples）；**B200 作为 hero zero-shot**，另留 20% 做 few-shot 对照 | `h200.csv` + `b200.csv` | H200 + B200 (Modal) |
| **Xinhao Tan** | Zero-shot 评估：V100+T4+A100+A10+L4+H100 训练好的模型直接在 B200 上跑。**同表必须报告对照基线**：(a) Constant-h 基线、(b) HW-MLP only 基线、(c) Per-graph mean + GPU offset。这是诊断 h 分支是否真的在学硬件缩放的关键 | `zero_shot_results.csv` | 否 |
| **Zhikai Hu** | 实现 few-shot 微调：取训练好的模型，分别用 50/100/200 个 H200/B200 样本微调。对比从头在目标 GPU 上训练的效果。输出误差改善曲线 | `few_shot_results.csv` | 否（CPU 训练即可） |

**"H200 改成 few-shot"的原因**：评审发现 H200 vs H100 在 h 向量中只有 `memory_gb` 和 `bandwidth_gbs` 不同（其余 3 维 `fp16_tflops / l2_cache_mb / sm_count` 完全相等），compute-bound 区 latency 本就 ≈ H100——同架构插值对 C2 缺乏挑战性（即使 6 个 training anchor 也无济于事）。few-shot 更诚实。详见 [research_review.md §2.1](research_review.md)。

#### Day 10–11（周三–周四）：消融研究 & 跨模型评估

严格锁定 §5.2 Table 3，只做 2 行消融（GNN w/o edges = Per-kernel sum MLP；GNN w/o h = constant-h）。其他候选实验全部 push 到 Future Work（见 §5.3）。

| 负责人 | 任务 | 交付物 | 需要GPU |
|--------|------|--------|---------|
| **Xiang Li** | 实现 `models/per_kernel_mlp.py`（~40 行，NeuSight 风格 per-node sum MLP）；加 `--model-variant per_kernel_mlp` + `--constant-h` 开关到 `train_and_eval.py`；跑 Table 3 的 2 消融 × 2 split = 4 次训练 | `ablation_table.csv`（3 method × 2 split = 6 数字） | 否 |
| **Xinhao Tan** | 跨模型泛化（Table 1 的 leave-one-model-out 6 折已覆盖，此处只补**规模泛化曲线**图）：GPT-2 Small 训练 → Medium / Large 测试 MAPE | `cross_model_results.csv` + Figure | 否 |
| **Zhikai Hu** | 两个 sanity check（正文文字引用，不占表格）：(1) 把 `s` 清零看 MAPE 恶化幅度；(2) h 单调性扰动——人为 TFLOPS +2×，compute-bound op 的预测延迟应单调降；通过率 > 80% 算合格 | `sanity_checks.md`（写进 report §5.4） | 否 |

**Day 10 同步开始报告写作**：在 Overleaf 上创建共享 LaTeX 项目，搭建报告骨架。

#### Day 12–14（周五–周日）：报告撰写 & 最终验证

| 负责人 | 任务 | 交付物 |
|--------|------|--------|
| **Xiang Li** | 撰写 Sections 1-3（Introduction、Problem Definition、Related Work）；生成所有图表（架构图、延迟对比图）；代码整理 + README | 报告 Sec 1-3 + 图表 |
| **Xinhao Tan** | 撰写 Section 4-5（Method、Experiments）；制作所有结果表格；确保可复现性（编写端到端实验复现脚本） | 报告 Sec 4-5 + 表格 |
| **Zhikai Hu** | 撰写 Section 6-7（Ablation/Analysis、Conclusion）；合并最终报告；通读校对；准备代码提交包 | 最终报告 PDF + 代码 zip |

> **Week 2 里程碑**：完整报告，包含 B200 硬件泛化结果（zero-shot + few-shot）、跨模型评估、消融研究，以及整洁的代码库。

---

## 7. GPU 排队优化策略

### Profiling Job 设计

每个 profiling job 是一个自包含脚本：

```
加载模型 → 遍历 (batch_size, seq_len) 配置网格 → 每个配置运行 (50 次 warmup + 100 次测量，CUDA events 计时) → 保存结果到 CSV
```

**关键改动 vs 初稿**（详见 [data_collection_plan.md](data_collection_plan.md)）：

- warmup 10 → 50，runs 50 → 100，训练 target 用 **p50** 而非 mean
- `torch.cuda.Event(enable_timing=True)` 替换 `time.perf_counter()`
- `taskset -c` 绑核，不申请 `--exclusive`（PSC 要 8 卡整机预约）
- Modal 锁定 SKU（`H100!` / `A100-40GB` / `B200`），CSV 记录 `actual_gpu_name`

单次 GPU session 约 1.5-2 小时（因 warmup/runs 翻倍 + 配置矩阵扩大）即可覆盖全部 6 个模型 × 所有配置。

### 排队/提交计划

| 时间 | 任务 | 平台 | 谁负责 |
|------|------|------|--------|
| Day 3-4 | V100 profiling | PSC（排队） | Xiang |
| Day 3-4 | H100 profiling | PSC（排队） | Xinhao |
| Day 4 | A100 profiling | Modal（即时） | Xinhao |
| Day 8 | H200 + B200 profiling | Modal（即时） | Xiang |

Modal job 无需排队，提交后几分钟内开始运行，是 PSC 排队的天然缓冲。

### 备选方案

如果 PSC V100/H100 排队超过 2 天：

1. 先用 Modal A100 数据开始训练（A100 即时可得）
2. PSC 数据到位后继续补充训练集
3. H200 和 B200 不受影响（均在 Modal）

---

## 8. 评估指标体系

每个指标都绑定到 §5.2 的某张表，不产出"游离指标"。

| 指标 | 定义 | 出现在 | 目标阈值 |
|---|---|---|---|
| **MAPE** | 预测延迟 vs 实际延迟的平均绝对百分比误差 | Table 1 / 2 / 3 所有格 | 主 claim 判据，见下方各表阈值 |
| **Spearman ρ** | 预测排序 vs 实际排序的秩相关 | Table 1 / 2 | 补充指标，对 outlier 鲁棒 |
| **Top-1 准确率** | 对每个 (model, gpu) 组正确识别最优 (batch, seq) 配置的比例 | Table 1 / 2 | > 70% |

### Claim 判据（论文能 defend 的量化门槛）

| Claim | 指标 | 判据 |
|---|---|---|
| **C1** — 学习模型击败分析式 / 诊断基线 | Table 1 MAPE | GNN 在所有 5 split 上击败 Roofline、Per-graph mean、Pooled MLP；且对 XGBoost 至少 3-5 MAPE 点优势 |
| **C2** — spec-only 零样本外推到 Blackwell | Table 2 MAPE | B200 zero-shot MAPE < 30%；H200 zero-shot < 20%；B200 few-shot N=50 消除 ≥ 50% zero-shot 误差 |
| **C3** — 图结构有边际贡献 | Table 3 MAPE | GNN 相对 Per-kernel sum MLP 在 leave-one-GPU-out avg 或 B200 zero-shot 上 gap ≥ 3 MAPE 点 |
| **C2 诊断** — h 向量真被学到 | Table 3 MAPE | GNN 相对 constant-h GNN gap 显著（尤其在 B200 zero-shot） |

**如果判据不达标**：不强行宣布胜利，在 report §5 正文诚实写 "we find that ..." 并在 §5.5 Limitations / Future Work 讨论成因。负面结论对课程项目评分无负面影响，**但掩盖或选择性汇报有严重影响**。

### 两个正文 sanity check（不占表格）

- **h 物理单调性**：人为 +2× TFLOPS，compute-bound op 预测延迟应单调降；通过率 > 80% 才说 h 分支物理意义合理。若失败 → C2 核心假设不成立，需在 Limitations 中讨论
- **s 功能性**：把 `s` 清零，MAPE 应显著恶化（简单验证 config 分支不被忽略）

---

## 9. 工作负载模型列表

聚焦 Transformer 架构，与项目标题 "LLM Inference Optimization" 保持一致。通过不同规模、不同类型（encoder-only / decoder-only / encoder-decoder）来测试泛化能力：

| 模型 | 参数量 | 类型 | 角色 | 测试 Batch Size |
|------|--------|------|------|----------------|
| GPT-2 Small | 124M | Decoder-only | 主要训练工作负载 | 1, 4, 8 |
| GPT-2 Medium | 355M | Decoder-only | 规模泛化测试（训练在 Small → 测试在 Medium） | 1, 4 |
| GPT-2 Large | 774M | Decoder-only | 进一步规模泛化（如 GPU 内存允许） | 1, 2 |
| BERT-base | 110M | Encoder-only | 架构类型泛化（decoder → encoder） | 1, 4, 8, 16 |
| BERT-large | 340M | Encoder-only | encoder 内部规模泛化 | 1, 4, 8 |
| T5-small | 60M | Encoder-Decoder | 架构类型泛化（第三种 Transformer 变体） | 1, 4, 8 |

---

## 10. 风险评估与应对

| 风险 | 影响 | 应对策略 |
|------|------|----------|
| ~~**`torch.fx.symbolic_trace` 挂在 HF 模型上**~~ **✅ 已解决 2026-04-17** | ~~Day 1 单点故障~~ | 已改走 `transformers.utils.fx.symbolic_trace` + pin `transformers<4.52` + smoke test（6/6 PASS）。**守护措施**：不要升级 transformers 到 ≥ 4.52；环境重装后先跑 smoke test |
| **Modal 自动 SKU 升配污染数据** | h 向量与实际硬件对不上，整行污染 | 一律用锁定 SKU（`H100!` / `A100-40GB` / `B200`）；CSV 强制写入 `actual_gpu_name`；加载训练集时按实际卡型分组 |
| **3-GPU 训练 → h 分支在查表而非学缩放** | 硬件泛化 claim 站不住 | 必须跑 Constant-h、HW-MLP-only、Per-graph-mean 三个对照；做 h 合成扰动单调性测试 |
| **GPU 排队等待过长** | profiling 数据不足；B200 结果缺失 | 脚本全部预先调试好批量运行；Modal 天然无排队可作缓冲；必要时减少模型数量 |
| **代价模型精度不够 / 被 XGBoost 追平** | 无法证明 GNN 结构的价值 | 回退到排序准确性（比绝对预测更容易）；若 GNN 和 XGBoost 持平，**诚实报告**，重点放在连续 h 向量 + LLM 图的组合与泛化上 |
| **B200 驱动/CUDA 兼容性问题** | 无法在 Blackwell 上 profiling | 使用 V100→H100 趋势外推的合成 B200 特征向量；在报告中作为 future work 讨论 |
| **跨模型泛化失败** | 模型只在同架构族内有效 | 诚实报告负面结果（这对课程项目也是合格结论）；增加架构类型条件特征 |
| **报告写作时间紧张** | 报告不完整或仓促 | 第 10 天就开始写报告骨架；三人并行各写各的章节；用共享 Overleaf 协作 |

**兜底 fallback 定位**（见 [research_review.md §6](research_review.md)）：如果 B200 外推失败，把评估从 "zero-shot to B200" 降级为 "**sample-efficient cross-GPU-generation extrapolation**"。主 claim 换成 leave-one-GPU-out + H200 few-shot + XGBoost 对比，B200 作为 honest stretch goal 写成 negative finding。

---

## 11. 最终交付物清单

1. **项目报告**（8-10 页，LaTeX）：Introduction（含 4 项 scope 决策 vs proposal 声明）、Problem Definition、Related Work（至少覆盖 NeuSight、Akhauri 2024、HELP、Habitat、MAPLE、PerfSage、Vidur）、Method、Experiments（5+ 表格/图表，含三种切分的 leave-one-out CV + 硬件泛化 + 消融）、Limitations（eager mode only, no torch.compile）、Conclusion

2. **代码库** — 已在 repo 中（GitHub 可见）：

   | 路径 | 角色 | 状态 |
   | --- | --- | --- |
   | [src/hetero_cost_model/graph/](src/hetero_cost_model/graph/) | HF fx tracing + 节点特征 + FLOPs 估算 | ✅ |
   | [src/hetero_cost_model/hardware.py](src/hetero_cost_model/hardware.py) | 5 维硬件特征注册表 | ✅ |
   | [src/hetero_cost_model/strategies.py](src/hetero_cost_model/strategies.py) | `InferenceConfig` + `config_grid` | ✅ |
   | [src/hetero_cost_model/profiling.py](src/hetero_cost_model/profiling.py) | CUDA events 计时 | ✅ |
   | [src/hetero_cost_model/runtime_info.py](src/hetero_cost_model/runtime_info.py) | `actual_gpu_name` 查询 + SKU 锁定 | ✅ |
   | [src/hetero_cost_model/data.py](src/hetero_cost_model/data.py) | Sample / LatencyDataset / `load_samples_from_csv` | ✅ |
   | [src/hetero_cost_model/models/gnn.py](src/hetero_cost_model/models/gnn.py) | GAT + Graph Transformer 变体 | ✅ |
   | [src/hetero_cost_model/baselines.py](src/hetero_cost_model/baselines.py) | Roofline / Per-graph mean / XGBoost | ✅ |
   | [src/hetero_cost_model/training/](src/hetero_cost_model/training/) | 训练循环 + MSE + ranking loss | ✅ |
   | [src/hetero_cost_model/model_zoo.py](src/hetero_cost_model/model_zoo.py) | 6 模型注册 + eager 加载 + fx input_names | ✅ |
   | [src/hetero_cost_model/metrics.py](src/hetero_cost_model/metrics.py) | MAPE / Spearman / NDCG / Top-K | ✅ |
   | [scripts/smoke_test_graphs.py](scripts/smoke_test_graphs.py) | Day 1 fx smoke test（6/6 PASS） | ✅ |
   | [scripts/extract_graphs.py](scripts/extract_graphs.py) | 图序列化到 `data/graphs/*.pkl` | ✅ |
   | [scripts/end_to_end_smoke.py](scripts/end_to_end_smoke.py) | 8 阶段本地冒烟 | ✅ |
   | [scripts/run_profiling.py](scripts/run_profiling.py) | profiling 驱动（idempotent + OOM-safe） | ✅ |
   | [scripts/train_and_eval.py](scripts/train_and_eval.py) | 4 baseline 统一评估 | ✅ |
   | README（含 reproducibility 指南） | | ⏳ Day 12-14 |

3. **Profiling 数据集**：CSV 文件，V100(PSC) / A100(Modal) / H100(PSC) / H200(Modal) / B200(Modal) 合计 **~1000+ hero 样本 + 500+ hold-out ≈ 1500+ 条**；含 `actual_gpu_name` / `attn_impl` / `kernel_backend` 等 schema 扩展列（见 [data_collection_plan.md §3.1](data_collection_plan.md)）

4. **训练模型**：最优代价模型 + 各消融变体（包括 per-node vs post-readout HW 融合）的保存权重

5. **Pre-extracted graphs**：[data/graphs/*.pkl](data/graphs/) 已 committed，6 个模型的 `GraphRepr` pickle（~1 MB 总共）；同学 clone 即用，无需重跑 fx tracing

6. **实验脚本**：一键复现（见 `scripts/train_and_eval.py` 命令行）

**Day 1-2 完成度**（2026-04-17 盘点）：三人 6/6 任务交付 ✅；pytest 22/22、smoke 6/6、end-to-end 8/8、`run_profiling.py` CPU dry-run、`train_and_eval.py` 四路对比，全部本地绿。等 Day 3 上 GPU 采真数据。
