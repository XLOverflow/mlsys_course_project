# Experiments Log

> 截至 2026-04-17 晚的所有训练实验、目的、结论。
> 原始 JSON 文件在本目录同级。

---

## 1. 实验系统概况

### 数据集（2026-04-17 完成）

| 维度 | 值 |
|---|---|
| 模型 | 6 个 Transformer 家族：gpt2-{small,medium,large}、bert-{base,large}、t5-small |
| 硬件 | 7 块 GPU（全 Modal 采集）：T4、A100、A10(G)、L4、H100、H200、B200 |
| 配置网格 | `batch ∈ {1,2,3,4,6,8,12,16}` × `seq ∈ {32,64,128,256,512,1024}` = 48 configs |
| 精度 | FP16 eager mode |
| 总样本 | 1903 条有效 latency（B 类 BERT×seq=1024 已过滤；T4×gpt2-large 47/48） |
| 硬件向量 h | 5 维：FP16 TFLOPS / HBM GB / HBM BW / L2 MB / SM count |
| 推理配置 s | 2 维：batch_size、seq_len（归一化到 [0,1]） |

### 评估的 6 种方法

| 方法 | 类别 | 输入 |
|---|---|---|
| Roofline | 分析式（0 训练） | per-op FLOPs / bytes + hardware peak |
| Per-graph mean | 诊断基线 | `(model_name, gpu)` → 历史均值查表 |
| Pooled MLP | 弱学习 | mean-pool(nodes) ‖ s ‖ h |
| XGBoost | 强学习（基线） | 12 维手工全局特征 |
| Per-kernel MLP | GNN ablation | per-node MLP + scatter-sum |
| **GNN (GAT)** | **主方法** | 3 层 GAT + readout + MLP head |

### GNN 两代架构

| | v1（初版） | **v2（2026-04-17 重设计）** |
|---|---|---|
| s/h 注入层级 | 只在 readout 之后拼 | **每节点 broadcast，GAT 之前就拼到 x** |
| Readout | mean + max pool | **sum pool**（物理 latency 加性） |
| Head 输入 | `pool ‖ s ‖ h` | **`pool ‖ s ‖ h ‖ g`**（新增 global skip） |
| `data.g` | 无 | `[log1p(total_flops), log1p(total_bytes), log1p(num_nodes), log1p(num_edges)]` |

v2 的三个改动各自独立可控（`--gnn-node-level-sh / --gnn-readout / --gnn-global-skip`）。

### 划分语义

| Split | 训练集 见过…… | 测试集 见过…… | 考验什么 |
|---|---|---|---|
| `random` | 所有图 + 所有 GPU | 80% leakage | in-distribution sanity |
| `leave-gpu=X` | 所有图 + 其他 GPU | 所有图 + 未见 GPU `X` | 跨硬件（同分布）泛化 |
| `zero-shot=B200` | 所有图 + 5 个训练 GPU | 所有图 + 未见 Blackwell | 跨代际硬件外推（🏆 hero） |
| `leave-model=X` | 5 个图 + 所有 GPU | 未见图 `X` + 所有 GPU | **未见图结构**泛化 |

---

## 2. 实验时间线

### Phase 1 — v1 首轮结果（Day 5 起）

**目的**：确认 pipeline 在真数据上跑通，产出 Table 1/2 的第一批数字。

**实验**：

| Split | 命令 | GNN (v1) | XGBoost | 结论 |
|---|---|---:|---:|---|
| random | `--split random --epochs 50` | 72.8% | 8.8% | XGBoost 大胜 |
| leave-gpu=h100 | `--split leave-gpu=h100 --epochs 50` | 46.5% | 14.0% | XGBoost 大胜 |
| zero-shot=b200 🏆 | `--split zero-shot=b200 --epochs 50` | 51.0% | 12.7% | XGBoost 大胜 |

**Phase 1 关键发现**：GNN v1 被 XGBoost 碾压 30-40 MAPE 点。

### Phase 2 — 诊断：为什么 v1 这么差

**假设 A**：容量不够 → 试 hidden=128、layers=3、epochs=100

| Split | v1 (h64/L2/e50) | v1 (h128/L3/e100) | Δ |
|---|---:|---:|---:|
| random | 72.8% | 46.8% | **-26.0** |
| leave-gpu=h100 | 46.5% | 53.6% | **+7.2** ⚠️ |

**Random 上大改善，但 leave-gpu-out 反而更差** → 不是单纯容量问题，而是架构/数据交互问题。

**假设 B**：ranking_lambda 权重不对 → 扫 {0.0, 0.1, 1.0}

| rank_λ | GNN MAPE (random) |
|---|---:|
| 0.0 | 68.2% |
| 0.1 (默认) | 72.8% |
| 1.0 | 68.8% |

**基本无差别** → ranking loss 不是瓶颈。

**Phase 2 关键发现**：问题在架构，不在超参。

### Phase 3 — Table 2 Few-shot (v1)

| Split | Few-shot N | GNN | XGBoost |
|---|---:|---:|---:|
| zero-shot=b200 | 0 | 51.0% | 12.7% |
| zero-shot=b200 | 50 | 44.5% | 8.3% |
| zero-shot=b200 | 100 | **139%** 💥 | 8.7% |
| zero-shot=b200 | 200 | 37.8% | 5.6% |
| zero-shot=h200 | 0 | 45.2% | 12.0% |
| zero-shot=h200 | 100 | **127%** 💥 | 4.4% |

**Phase 3 关键发现**：
- XGBoost few-shot 响应良好（12.7% → 5.6% 在 N=200）
- GNN 训练**不稳定**：fs100 两次都爆炸超过 100%，fs50 / fs200 反而正常
- 怀疑 ranking loss 在少量 OOD 样本混入训练时产生不良梯度

### Phase 4 — 外部代码审查暴露 v1 架构缺陷

外部审查指出 v1 GNN 的 3 个结构问题：

1. **GAT 层对 `s/h` 完全盲**：`data.s / data.h` 只在 readout 之后拼，message passing 不知道当前是在预测哪块 GPU
2. **mean/max pool 丢掉加性信号**：latency 物理上是 `Σ op_latency`，但 mean pool 归一化了
3. **Head 没有 Roofline 级输入**：XGBoost 直接吃 log1p(total_flops) 等全局量，GNN 要从 per-op 特征重建

### Phase 5 — v2 架构重设计 + 验证 sweep

**设计改动**（[commit 866bbee](../commits/866bbee)）：

1. `node_level_sh=True`：把 s 和 h broadcast 到每个节点，GAT 之前就拼到 `x`
2. `readout="sum"`：`global_add_pool` 替代 mean+max，对齐 latency 加性结构
3. `global_skip=True`：在 sample_to_pyg 新增 `data.g = [log1p(total_flops), log1p(total_bytes), log1p(num_nodes), log1p(num_edges)]`，直接塞给 head

Pooled MLP 和 Per-kernel MLP 也同步获得 `global_skip` 开关以保持 ablation 公平。

**v1 → v2 验证 sweep（50 epochs，默认超参）**：

| Split | v1 GNN | **v2 GNN** | XGBoost | Δ vs v1 | Δ vs XGB |
|---|---:|---:|---:|---:|---:|
| random | 72.8% | **34.7%** | 8.8% | **-38.1** 🔥 | +25.9 |
| leave-gpu=h100 | 46.5% | 67.8% | 14.0% | **+21.3** 💀 | +53.8 |
| zero-shot=b200 🏆 | 51.0% | **27.9%** | 12.7% | **-23.1** 🔥 | +15.1 |
| **leave-model=gpt2-large** ⭐ | — | **25.6%** | 39.4% | — | **-13.7** 🏆 |
| **leave-model=bert-large** ⭐ | — | **31.0%** | 50.8% | — | **-19.8** 🏆 |

**Phase 5 关键发现**：

✅ **v2 在 `leave-model-out` 下 GNN 打过 XGBoost 了**——这是 C3 claim（graph 结构有价值）成立的**决定性证据**。

✅ **B200 zero-shot MAPE 从 51% 降到 28%**，达到 plan §8 定的 "<30%" 阈值。

⚠️ **v2 在 leave-gpu=h100 反而退步**（47% → 68%）：
- 同图不同 GPU 的 setting 下，global_skip 的 4 个特征**对所有 h100 样本完全相同**（同 model、同 config），没帮助 discriminate GPU
- 推测：node-level s/h 让 GNN 过拟合"见过的 5 个 GPU 组合"

---

## 3. 结论（能站住的几条）

### ✅ 3.1 GNN 的价值体现在"未见图结构"

**证据**：`leave-model-out` 两次实验，GNN v2 都显著优于 XGBoost（-14 / -20 MAPE 点）。

**解释**：XGBoost 的 12 维特征里，`total_flops` / `total_bytes` 对未见模型是 OOD——GBM 遇到没见过的取值区间外推失效。GNN 通过 per-op 节点特征 + message passing，不依赖"见过这种 total flops"，所以 generalize。

### ✅ 3.2 B200 Blackwell zero-shot 成功（C2 claim 成立）

**证据**：v2 GNN 在 B200 上 27.9% MAPE，< plan §8 定的 <30% 阈值。

**对比**：XGBoost 12.7%（仍更好，但差距仅 15 pts vs v1 时代 38 pts）。对论文来说，"spec-only zero-shot 到 Blackwell MAPE < 30%" 是可宣告的结果。

### ✅ 3.3 XGBoost 在"见过图 + 分布内"场景下极强

**证据**：random 8.8%、leave-gpu=* 14%、zero-shot=b200 12.7%——全部 < 15%。

**说明**：Roofline 输入（total_flops、total_bytes、硬件 5 维）加上 GBM 的非线性能力，**本身就足以描述大部分 latency 规律**。这不是 bug，是数据分布决定的。

### ⚠️ 3.4 v2 在 leave-gpu-out 退步——unresolved

**现象**：leave-gpu=h100 MAPE 从 46.5% → 67.8%。

**假设**：
- global_skip 的 4 维在此 split 下是 redundant（所有 h100 样本 g 相同）
- node-level s/h 过拟合见过的 (graph × gpu) 组合
- 还没做控制实验区分

**行动项**：跑 v2 三个开关的 ablation，看哪个是元凶。

### ⚠️ 3.5 Per-kernel MLP 训练不稳定

**证据**：
- v1 leave-gpu=h100: 85% → v2: 559% 💥
- B200 fs100 变种上 GNN v1 也爆到 139%

**假设**：per-node 全连接 + ranking loss + 少样本混入 → 优化景观陡峭。

**行动项**：为 Per-kernel MLP 单独调 LR / 梯度裁剪 / warm-up。

---

## 4. 实验产物（文件索引）

| 文件 | 内容 |
|---|---|
| `random_gat_e5.json` | 冒烟 5-epoch 验证 Modal H100 dispatch |
| `random_gat_e50.json` | **v2** random 50 ep |
| `random_gat_e100__h128_L3.json` | v1 扩容实验（h128 L3 e100） |
| `random_gat_e50__rank0.0.json` | v1 MSE only |
| `random_gat_e50__rank1.0.json` | v1 ranking loss 重权 |
| `leave_gpu_h100_gat_e50.json` | **v2** leave-gpu=h100 50 ep |
| `leave_gpu_h100_gat_e100__h128_L3.json` | v1 扩容 |
| `zero_shot_b200_gat_e50.json` | **v2** B200 hero |
| `zero_shot_b200_gat_e50__fs50.json` | v1 B200 few-shot N=50 |
| `zero_shot_b200_gat_e50__fs100.json` | v1 B200 few-shot N=100（不稳定） |
| `zero_shot_b200_gat_e50__fs200.json` | v1 B200 few-shot N=200 |
| `zero_shot_h200_gat_e50.json` | v1 H200 zero-shot |
| `zero_shot_h200_gat_e50__fs100.json` | v1 H200 few-shot N=100（不稳定） |
| `leave_model_gpt2_large_gat_e50.json` | **v2** leave-model gpt2-large（GNN 胜 XGB） |
| `leave_model_bert_large_gat_e50.json` | **v2** leave-model bert-large（GNN 胜 XGB） |

**注意**：v1 和 v2 的 random / leave-gpu=h100 / zero-shot=b200 共用文件名，v2 已覆盖 v1。v1 数字已记录在本文件 Phase 1 表格中，不会丢失。

---

## 5. 开放问题 / 下一步候选

### P0（关键 validation）

- [ ] 跑剩 4 个 `leave-model=*`（gpt2-small、gpt2-medium、bert-base、t5-small），验证 "GNN 在未见图下胜 XGBoost" 是否稳定（不是只对 large 模型成立）

### P1（v2 内部组件拆解）

- [ ] v2 三开关的 ablation（在 leave-model=gpt2-large 或 zero-shot=b200 上跑）：
  - 仅 `node_level_sh`
  - 仅 `readout=sum`
  - 仅 `global_skip`
  - 全关（= v1 等价）
- [ ] 找出谁是主要贡献者

### P2（诊断 leave-gpu-out 退步）

- [ ] 在 leave-gpu=h100 上单独关掉 `node_level_sh` 或 `global_skip`，看哪个导致退步
- [ ] 如果 global_skip 是罪魁 → 在 leave-gpu-out 下它确实无信号（同模型同配置），该改成 split-aware 的选择

### P3（Per-kernel MLP 稳定性）

- [ ] 降 LR / 加 gradient clipping / warm-up 重跑 Per-kernel MLP
- [ ] 确认"Per-kernel MLP 打过 GNN"或"GNN 打过 Per-kernel MLP"哪个是真的 C3 结论

### P4（Table 2 few-shot 重跑）

- [ ] v1 的 B200 fs100 / H200 fs100 爆炸数字，在 v2 下是否修复
- [ ] 新 B200 few-shot 曲线：N ∈ {0, 50, 100, 200} 下 v2 GNN 的表现

### P5（写 report）

- [ ] §5.2 Table 1 / 2 / 3 的骨架写作，填已有数字
- [ ] §5.5 Limitations 诚实写 "v2 leave-gpu 退步"、"Per-kernel MLP 不稳定" 这些 negative findings
