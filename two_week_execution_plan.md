# CPU/GPU 异构 LLM 推理优化 — 泛化学习代价模型

## 两周冲刺执行计划

**团队成员**：Xiang Li (xli8) | Xinhao Tan (xinhaota) | Zhikai Hu (zhikaih)

**时间跨度**：2026年4月14日 – 4月28日

**硬件资源**：V100 (1-2卡) | H100 (1-2卡) | B200 (1-2卡) — 共享排队

---

## 1. 总体策略

原计划为4周，现压缩至2周。核心目标不变：构建一个图级别(graph-level)的学习代价模型 **f(G, s, h) → T̂**，能够在异构 CPU/GPU 环境中预测端到端推理延迟，并在不同工作负载和硬件上展示泛化能力。

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
- **与我们的区别**：Felix 聚焦于单设备上的 schedule 调优。我们关注异构 CPU/GPU 放置决策，是更高层次的调度问题

### 2.3 Helix (ASPLOS 2025)

- **核心思路**：将异构 GPU 集群上的 LLM 推理建模为最大流问题，使用混合整数线性规划(MILP)联合优化模型放置和请求调度
- **关键指标**：吞吐提升最高3.3倍，prompt延迟降低最高66%
- **与我们的区别**：Helix 使用分析/MILP方法，我们使用学习模型来预测代价，能泛化到未见过的硬件

### 2.4 Q-Infer (ACM TACO 2025)

- **核心思路**：混合 GPU-CPU 异构并行推理，结合稀疏性感知的自适应动态调度
- **与我们的区别**：Q-Infer 使用手工设计的调度规则，我们学习预测策略的代价

### 2.5 Ansor / MetaSchedule (OSDI'20 / NeurIPS'22)

- **地位**：TVM 中学习代价模型的基础工作，算子级别的代价模型驱动调度搜索
- **与我们的区别**：逐算子、逐任务在线学习；我们追求跨工作负载和跨硬件的图级别泛化

---

## 3. 硬件资源与特征向量设计

三种 GPU 横跨三代架构，是硬件泛化的理想测试平台：

| 特征 | V100 (Volta) | H100 (Hopper) | B200 (Blackwell) |
|------|-------------|---------------|-----------------|
| FP16 Tensor TFLOPS | 125 | 1,979 | ~4,500 (FP8: 9,000) |
| HBM 容量 | 32 GB (HBM2) | 80 GB (HBM3) | 180 GB (HBM3e) |
| 显存带宽 | 900 GB/s | 3,350 GB/s | 8,000 GB/s |
| 互连 | PCIe 3.0 / NVLink 2.0 | PCIe 5.0 / NVLink 4.0 | PCIe 5.0 / NVLink 5.0 |
| 特征向量 h 示例 | [125, 32, 900, ...] | [1979, 80, 3350, ...] | [4500, 180, 8000, ...] |

**硬件特征向量 h 包含**：峰值 TFLOPS（按精度）、HBM 容量、显存带宽、PCIe 带宽、CPU 规格（核心数、频率、缓存大小）。所有数值做归一化处理，使模型能通过连续特征感知硬件差异，而非依赖架构特定的离散编码。

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

### 4.2 执行策略编码

执行策略 s 定义为图上每个算子的设备放置决策：

- 每个算子分配到 CPU 或 GPU
- 批大小 (batch size)
- 序列长度配置

两周范围内，我们聚焦于**算子级别的 CPU/GPU 放置**作为核心策略维度。

### 4.3 代价模型架构

我们采用 **GNN 为主架构、Graph Transformer 为对比变体** 的方案。

#### 主架构：GAT（Graph Attention Network）

```
输入: 标注计算图 G (节点特征 + 边)  +  策略 s  +  硬件向量 h
  │
  ├── 节点嵌入: 算子特征 ⊕ 硬件向量 h ⊕ 放置标记（来自 s）
  │
  ├── 3层 GAT, hidden_dim=128, 4 heads
  │
  ├── 图级别 Readout: mean pooling + max pooling 拼接
  │
  ├── 2层 MLP Head (256 → 128 → 1)
  │
  └── 输出: 预测延迟 T̂
```

**选择 GNN 的理由**：

- 计算图是天然的 DAG 结构，GNN 的消息传递沿数据依赖边进行，归纳偏置最匹配
- 不同模型的图拓扑差异极大（GPT-2 vs ResNet），GNN 天然处理变长、不规则结构，无需 padding
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

- **Phase 1**：在 V100 + H100 数据上训练，学习跨硬件的预测能力
- **Phase 2**：在 B200 上做 zero-shot 评估（不用 B200 训练数据），然后用少量 B200 样本做 few-shot 微调，测量改善幅度

---

## 5. Baseline 设计

本项目涉及两类 baseline 对比：**推理性能 baseline**（代价模型选出的策略要跟谁比延迟）和**代价模型 baseline**（GNN 预测器要跟什么预测方法比精度）。

项目的核心贡献在于代价模型的**预测精度**和**泛化能力**，因此 baseline 聚焦在代价模型层面，选用简单、好复现且不会过强的 rule-based 方法：

| Baseline | 做法 | 实现成本 | 对比意义 |
|----------|------|----------|----------|
| **Roofline 分析模型** | 延迟 = max(FLOPs / 峰值TFLOPS, 数据量 / 带宽) + CPU-GPU 传输开销估算 | 约50行代码，纯公式 | 证明学习方法比手写公式更准 |
| **随机选择** | 从候选策略中 `random.choice()` | 一行代码 | 策略排序能力的下界 |

更复杂的模型变体对比（MLP vs GNN、Graph Transformer vs GAT 等）归入消融实验，属于模型设计层面的分析，不作为主 baseline。

### 5.2 核心实验对比逻辑

```
实验1 — 预测精度:
  GNN Cost Model  vs  Roofline  vs  Random  →  MAPE / Spearman ρ

实验2 — 硬件泛化:
  V100+H100 上训练 → B200 zero-shot  vs  Few-shot (50/100/200 samples)

实验3 — 工作负载泛化:
  规模泛化: GPT-2 Small 上训练 → GPT-2 Medium/Large 上测试
  架构类型泛化: Decoder-only (GPT-2) 上训练 → Encoder-only (BERT) / Enc-Dec (T5) 上测试

消融实验（模型设计层面）:
  (1) 移除硬件特征 h  (2) MLP 替换 GNN  (3) 移除策略特征 s  (4) Graph Transformer 替换 GAT
```

---

## 6. 两周详细时间线

### Week 1：基础设施 + 数据收集 + Baseline（4月14日–20日）

#### Day 1–2（周一–周二）：基础设施搭建

| 负责人 | 任务 | 交付物 | 需要GPU |
|--------|------|--------|---------|
| **Xiang Li** | 构建 ONNX/torch.fx 图提取管线；实现节点特征提取（算子类型、形状、FLOPs、内存）；在 GPT-2、BERT、T5 上测试 | `graph_extractor.py` | 否 |
| **Xinhao Tan** | 构建 profiling 测试框架：给定模型 + 放置策略 + 硬件，运行 N 次推理并记录延迟；实现策略枚举（关键子图的所有 CPU/GPU 放置组合） | `profiler.py` + `strategy_gen.py` | 否（本地测试） |
| **Zhikai Hu** | 用 PyG 实现 GNN 代价模型（GAT/GraphSAGE）；定义硬件特征向量 schema；搭建训练循环（MSE + ranking loss） | `cost_model.py` + `train.py` | 否 |

**Day 2 检查点**：三人在本地完成各自模块的单元测试，确认接口可以对接。

#### Day 3–4（周三–周四）：数据收集冲刺

| 负责人 | 任务 | 交付物 | 需要GPU |
|--------|------|--------|---------|
| **Xiang Li** | 【V100】Profile：GPT-2 (small/medium)、BERT (base/large)、T5-small。每个模型 × 10-20种放置策略 × 3种batch size。目标：500+ 样本 | `v100_data.csv` | V100 × 1-2 |
| **Xinhao Tan** | 【H100】相同的模型/策略矩阵在 H100 上运行。同时收集所有模型的纯 CPU baseline。目标：500+ 样本 | `h100_data.csv` | H100 × 1-2 |
| **Zhikai Hu** | 构建数据管线：合并 CSV、特征归一化、构造 PyG Data 对象。准备训练/验证/测试集（按模型类型分层抽样） | `dataset.py` + `data/` | 否 |

**关键提示**：每个 profiling job 设计为自包含脚本，一次 GPU 预约运行3-4小时，覆盖一个模型族的全部配置。在排队前彻底本地调试，确保 GPU 时间100%用于 profiling、0%用于 debug。

#### Day 5–7（周五–周日）：Baseline 训练与验证

| 负责人 | 任务 | 交付物 | 需要GPU |
|--------|------|--------|---------|
| **三人协作** | 在 V100+H100 合并数据上训练代价模型。评估：(1) hold-out集预测 MAPE，(2) 策略排序的 Spearman 秩相关系数，(3) 与 Roofline baseline 对比。迭代优化模型架构 | 训练好的模型 + baseline 指标 | 任意 GPU（训练很轻量） |
| **Xiang Li** | 实现 Roofline 分析 baseline；在相同测试集上跑 Roofline 和随机选择，生成对比数据 | `baseline_results.csv` | 否 |

> **Week 1 里程碑**：工作的代价模型在 V100+H100 上 MAPE < 15%；在至少一个模型上展示对比 naive 放置的延迟改善。

---

### Week 2：泛化实验 + 消融研究 + 报告撰写（4月21日–28日）

#### Day 8–9（周一–周二）：B200 数据 & 硬件泛化

| 负责人 | 任务 | 交付物 | 需要GPU |
|--------|------|--------|---------|
| **Xiang Li** | 【B200】收集 profiling 数据（相同模型/策略矩阵）。数据拆分：80% 用于 zero-shot 评估，20% 用于 few-shot 微调 | `b200_data.csv` | B200 × 1-2 |
| **Xinhao Tan** | Zero-shot 评估：将 V100+H100 训练好的模型直接在 B200 测试集上运行，测量 MAPE 和秩相关系数。**这是核心泛化实验** | `zero_shot_b200_results` | 否 |
| **Zhikai Hu** | 实现 few-shot 微调：取训练好的模型，分别用 50/100/200 个 B200 样本微调。对比从头在 B200 上训练 | `few_shot_transfer_results` | 否（CPU训练即可） |

#### Day 10–11（周三–周四）：消融研究 & 跨模型评估

| 负责人 | 任务 | 交付物 | 需要GPU |
|--------|------|--------|---------|
| **Xiang Li** | 消融实验：(1) 移除硬件特征 h；(2) 用 MLP 替换 GNN（去掉图结构）；(3) 移除策略特征 s；(4) 用 Graph Transformer 替换 GAT（对比架构选择） | `ablation_table.csv` | 否 |
| **Xinhao Tan** | 跨模型泛化：(1) 规模泛化——GPT-2 Small 上训练 → GPT-2 Medium/Large 上测试；(2) 架构类型泛化——decoder-only 上训练 → encoder-only (BERT) 和 encoder-decoder (T5) 上测试 | `cross_model_results` | V100或H100（如需新profile） |
| **Zhikai Hu** | 决策有效性：对每个测试模型，按预测代价排序所有策略，对比 top-K 预测 vs 实际 top-K（NDCG / top-1准确率） | `ranking_eval_results` | 否 |

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
加载模型 → 遍历放置策略列表 → 每个配置运行 (100次warmup + 500次测量) → 保存结果到CSV
```

一次 GPU 预约可以收集一个模型族所有配置的数据，无需重新排队。

### 排队计划

- **Week 1 Day 3-4**：Xiang 排 V100，Xinhao 排 H100，**同时**提交，互不依赖
- **Week 2 Day 8**：Xiang 排 B200 任务
- 尽量利用工作日晚间和周末时段，排队竞争较小
- 所有脚本在排队前已完全本地调试通过

### 备选方案

如果 B200 排队时间过长：

1. 仅在 B200 上 profile 最小子集（GPT-2 small + BERT-base）
2. 使用 V100→H100 趋势外推的合成 B200 数据点
3. 在报告中诚实讨论局限性

---

## 8. 评估指标体系

| 指标 | 定义 | 目标 |
|------|------|------|
| **预测 MAPE** | 预测延迟 vs 实际延迟的平均绝对百分比误差 | 已见硬件 < 15%；B200 zero-shot < 25% |
| **Spearman 秩相关** | 预测排序 vs 实际策略排序的秩相关系数 | 已见硬件 > 0.85；B200 > 0.70 |
| **Top-1 准确率** | 正确识别最优策略的比例 | > 60%（选中实际最优或误差在5%以内） |
| **优化加速比** | 模型选出的策略 vs naive 全GPU baseline 的延迟比 | 至少2个模型族上 > 1.1x |
| **Few-Shot 迁移效果** | 用 N 个 B200 样本微调后 MAPE 改善幅度 | 50个样本应消除 > 50% 的 zero-shot 误差 |

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
| **GPU排队等待过长** | profiling数据不足；B200结果缺失 | 脚本全部预先调试好批量运行；优先利用晚间/周末时段；必要时减少模型数量 |
| **代价模型精度不够** | 无法展示有意义的优化效果 | 回退到更简单的 MLP 模型；聚焦排序准确性（比绝对预测更容易）；增加分析特征（roofline 估算） |
| **B200 驱动/CUDA兼容性问题** | 无法在 Blackwell 上 profiling | 使用 V100→H100 趋势外推的合成 B200 特征向量；在报告中作为 future work 讨论 |
| **跨模型泛化失败** | 模型只在同架构族内有效 | 诚实报告负面结果（这仍是有价值的研究发现）；增加架构类型条件特征 |
| **报告写作时间紧张** | 报告不完整或仓促 | 第10天就开始写报告骨架；三人并行各写各的章节；用共享 Overleaf 协作 |

---

## 11. 最终交付物清单

1. **项目报告**（8-10页，LaTeX）：Introduction、Problem Definition、Related Work、Method、Experiments（5+表格/图表）、Ablation Studies、Conclusion
2. **代码库**（GitHub）：`graph_extractor.py`、`profiler.py`、`cost_model.py`、`train.py`、`evaluate.py`、README（含复现指南）
3. **Profiling 数据集**：CSV 文件，V100 / H100 / B200 上共 1500+ 条延迟测量记录
4. **训练模型**：最优代价模型 + 各消融变体的保存权重
5. **实验脚本**：一键复现所有报告中数据的脚本
