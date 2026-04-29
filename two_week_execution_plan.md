# 跨异构 GPU 的 LLM 推理延迟泛化代价模型 — 项目计划

**团队**：Xiang Li (xli8) | Xinhao Tan (xinhaota) | Zhikai Hu (zhikaih)
**周期**：2026-04-14 ~ 2026-04-30（poster 截止）
**硬件**：T4 / A10 / A100-40GB / L4 / H100 / H200 / B200（全部 Modal）

---

## 0. 当前状态（2026-04-29，poster 截止前 1 天）

| 阶段 | 状态 | 备注 |
|---|---|---|
| Day 1-2 基础设施 | ✅ 完成 | pytest 22/22、smoke 6/6、end-to-end 8/8 |
| Day 3-4 数据收集 | ✅ 完成 | 7 GPU × 6 模型 = **1903 样本**（超额） |
| Day 5-7 Table 1（架构外推） | ✅ 完成 | 6/6 leave-model-out，GNN 胜 XGBoost 13.7-104.7 MAPE 点 |
| Day 8-9 Table 2（硬件外推） | ✅ 完成 | B200 zero-shot 27.9% 达标，但 XGBoost 在硬件外推上更强 |
| Day 10-11 Table 3（消融） | ✅ 完成 | v2 三开关协同；leave-gpu trade-off 诊断 |
| **Day 14-15 Poster 冲刺** | 🔴 **进行中** | Router + SHAP + Demo + 文案重写 |

**Claim 状态**：

- ✅ **C1**（学习模型 > baselines）：6/6 leave-model-out 全部成立
- ⚠️ **C2**（spec-only zero-shot to Blackwell）：B200 27.9% 达标，但 XGBoost 在 zero-shot 硬件 split 上更强 → 重排为 trade-off 叙事
- ✅ **C3**（graph 结构 load-bearing）：Per-kernel MLP 在 6/6 leave-model split 上 44-1093% 不稳定，GNN 稳定 20-33%

**详细结果**：[results/EXPERIMENTS.md](results/EXPERIMENTS.md)（Phase 6，16 轮 sweep）

---

## 1. 项目定位与 Scope 决策

**核心目标**：构建图级别的学习代价模型 **f(G, s, h) → T̂**，给定模型计算图 G、推理配置 s（batch、seq）、目标 GPU 硬件向量 h，预测端到端 forward-pass latency，**展示跨工作负载和跨硬件的外推能力**。

**项目性质**：CMU 15-442 MLSys 课程期末 project，poster 终点（不是论文投稿）。架构借鉴 NeuSight (ASPLOS'25)、Akhauri 2024、HELP；差异化贡献在**架构外推（leave-model-out）的稳定性**。

### 1.1 相对 proposal 的 4 项 scope 决策（Intro 必须显式声明）

| # | Proposal 写的 | 实际做的 | 写入 §Limitations |
|---|---|---|---|
| 1 | CPU/GPU operator placement | **纯 GPU cross-generation**；s = (batch, seq) inference config | ✅ 必须 |
| 2 | end-to-end latency | **forward-pass / prefill 单次延迟**；autoregressive decode 不测 | ✅ 必须 |
| 3 | Transformer + ResNet | **仅 Transformer 家族**（GPT-2 / BERT / T5 共 6 个） | ✅ 必须 |
| 4 | torch.compile? | 全部 HF eager mode + `attn_implementation="eager"` | ✅ 必须（proposal §4 已声明） |

### 1.2 Proposal 三个 RQ 的覆盖

| RQ | 对应实验 split | 实验数 | 谁赢 | 状态 |
|---|---|---:|---|---|
| **RQ1 Workload 泛化**（架构外推） | `leave-model=*` | 6 | **GNN 全胜** | ✅ 主 hero |
| **RQ2 Hardware 泛化**（硬件外推） | `leave-gpu=h100` + `zero-shot=b200` + `zero-shot=h200` | 3 | **XGBoost 全胜** | ⚠️ trade-off 叙事 |
| **RQ3 Decision effectiveness** | （新增）Router demo | 1 | n/a | 🔴 sprint 中操作化 |

---

## 2. 技术方案（已实现，~~此处压缩历史~~）

### 2.1 硬件特征向量（5 维）

| 特征 | 物理意义 |
|---|---|
| FP16 TFLOPS | 峰值算力 |
| HBM GB | 显存容量 |
| HBM 带宽 | 显存带宽 |
| L2 cache MB | 片上缓存（影响小 kernel + 带宽放大） |
| SM 数量 | 并行单元数 |

**为什么这 5 维**：全部为片上 spec，没有互联带宽，没有代际 ordinal——跟 NeuSight 的 on-chip-only 惯例一致。详见 [research_review.md](research_review.md) §2.2。

### 2.2 GNN v2 架构（commit 866bbee）

3 个改动**协同生效**（不是可加的，详见 EXPERIMENTS §6.3 ablation）：

1. **node_level_sh=True**：s 和 h broadcast 到每节点，GAT 之前就拼到 `x`
2. **readout="sum"**：`global_add_pool` 替代 mean+max，对齐 latency 加性结构
3. **global_skip=True**：拼 `[log1p(total_flops), log1p(total_bytes), log1p(num_nodes), log1p(num_edges)]` 给 head

实现：[src/hetero_cost_model/models/gnn.py](src/hetero_cost_model/models/gnn.py)

### 2.3 Baselines

| Baseline | 输入 | 角色 |
|---|---|---|
| **Roofline** | per-op FLOPs/bytes + hardware peak | 分析式参考（0 训练） |
| **Per-graph mean** | (model, gpu) lookup | 数据泄漏诊断 |
| **Pooled MLP** | mean-pool(nodes) ‖ s ‖ h | 弱学习对照 |
| **XGBoost** | 12 维全局特征（log1p(total_flops/bytes), num_nodes/edges, batch, seq, h×5） | **强 baseline**（陈天奇的工具） |
| **Per-kernel MLP** | per-node MLP + scatter-sum（NeuSight 风格） | GNN 去边消融 |
| **GNN (GAT v2)** | 3 层 GAT + sum readout + global skip | 主方法 |

**统一驱动**：`scripts/train_and_eval.py`，支持 `--split={random,leave-gpu=X,leave-model=X,zero-shot=X}` + few-shot。

---

## 3. 实验结果（重组为 RQ1/RQ2 视角）

详细 16 轮 sweep 数字见 [results/EXPERIMENTS.md](results/EXPERIMENTS.md)。下表是**外推主表**（删除 random 的 in-distribution 行）。

### 3.1 主表 Extrapolation Results（poster Table 1，MAPE %）

| Split | Roofline | XGBoost | GNN (v2) | **Router** |
|---|---:|---:|---:|---:|
| **── RQ2 硬件外推 ──** (Router 100% → XGB) | | | | |
| leave-gpu=h100 | 97.0 | **14.0** | 67.8 | **14.0** ← XGB |
| zero-shot=h200 | 98.0 | **12.0** | 20.9 | **12.0** ← XGB |
| zero-shot=b200 | 98.7 | **12.7** | 27.9 | **12.7** ← XGB |
| **── RQ1 架构外推 ──** (Router 100% → GNN) | | | | |
| leave-model=gpt2-small | 94.1 | 96.1 | **20.0** | **20.0** ← GNN |
| leave-model=gpt2-medium | 91.5 | 53.4 | **22.0** | **22.0** ← GNN |
| leave-model=gpt2-large | 88.7 | 39.4 | **25.6** | **25.6** ← GNN |
| leave-model=bert-base | 97.2 | 125.9 | **21.2** | **21.2** ← GNN |
| leave-model=bert-large | 96.0 | 50.8 | **31.0** | **31.0** ← GNN |
| leave-model=t5-small | 98.8 | 59.7 | **33.0** | **33.0** ← GNN |
| **── 混合 (per-sample 路由展示) ──** | | | | |
| mixed=gpt2-small,gpt2-large | 92.2 | 63.2 | 46.0 | **39.9** 🏆 |

**Hero**（poster 主结论）：
> **Tabular methods (XGBoost) handle hardware extrapolation well (12-14% MAPE); graph-aware methods (GNN) handle architecture extrapolation well (20-33% MAPE). A two-tier SHAP-driven router achieves the best of both — and on a heterogeneous test set with mixed extrapolation regimes, the router (39.9%) strictly beats both XGBoost alone (63.2%) and the GNN alone (46.0%).**

**Mixed split 细节**（[results/mixed_split_router.json](results/mixed_split_router.json)）：

- 训练：4 个模型（bert-base/bert-large/gpt2-medium/t5-small）× 80% configs × 7 GPUs = 987 样本
- 测试：留出的 2 个模型（gpt2-small/gpt2-large）全量 + 4 个训练模型剩 20% configs = 916 样本
- Router tier breakdown：73.3% 走 Tier 1（架构身份）→ GNN，26.7% 走 default → XGBoost，0% 走 Tier 2（SHAP feature OOD，因为 graph 聚合特征是图级常量，本数据集中不会触发——defense-in-depth 设计，写进 §Limitations）

**Per-tier MAPE 拆解**（验证 Router 每一层的决策都是局部最优）：

| Subset | n | XGBoost | GNN | Router 选 | 是否选对 |
|---|---:|---:|---:|---|---|
| Tier 1（未见架构） | 671 | 82.42% | **50.50%** | GNN | ✅ GNN 胜 32 pts |
| Default（分布内） | 245 | **10.70%** | 33.71% | XGBoost | ✅ XGBoost 胜 23 pts |

> **这两行数据是 Router 设计的硬证据**：在每个子集上 Router 都选了**实际更好**的那个模型。Router 在 default 子集上把 GNN 33.71% 换成 XGB 10.70%，省下 23 个 MAPE 点 × 26.7% 占比 ≈ 6.1 点——正好等于整体 Router 39.85% vs GNN-alone 46.01% 的 6 点差距。

### 3.2 Few-shot 曲线（B200，可选附录）

| N | Roofline | XGBoost | **GNN (v2)** | v1 (历史) |
|---|---:|---:|---:|---:|
| 0 | 98.7 | 12.7 | 27.9 | 51.0 |
| 50 | 98.8 | 8.3 | 22.9 | 44.5 |
| 100 | 98.8 | 8.7 | 25.2 | **139** 💥 |
| 200 | 98.7 | 5.6 | **9.7** | 37.8 |

**关键**：v1 fs100 爆炸（139%）在 v2 修复（25.2%）；fs200 是首个 < 10% 的 v2 GNN 数字。

### 3.3 V2 三开关 ablation（leave-model=gpt2-large）

| 配置 | MAPE | 说明 |
|---|---:|---|
| **v2 三开关全开** | **25.6%** 🏆 | baseline |
| 仅 node_level_sh | 82.8% | |
| 仅 sum readout | 350.6% | 单开崩溃 |
| 仅 global_skip | 40.4% | ≈ XGBoost |
| 三开关全关（≈ v1） | 68.3% | |

**结论**：**三个改动作为整体生效，单开任何一项都不够**。Table 3 caption 必须强调 synergy。

---

## 4. Poster 冲刺计划（最后 1 天）⭐

### 4.1 战略选择回顾（2026-04-29 决定）

| 路径 | 决策 |
|---|---|
| 残差混合（XGB+GNN delta） | ❌ 否决（作弊嫌疑） |
| XGB 喂 GNN 当特征 | ❌ 否决（仍有叠加味） |
| **Router + SHAP 诊断 + Demo** | ✅ **采纳** |
| 重新跑训练 | ❌ 沿用现有 JSON 数据 |
| 改 GNN 架构 | ❌ v2 已冻结 |

### 4.2 任务拆解

#### A. Router（backend，✅ 完成）

**新文件**：[src/hetero_cost_model/router.py](src/hetero_cost_model/router.py) (~190 行)

**两层 SHAP 驱动 OOD 路由**：

```python
# Tier 1: 架构身份 (metadata-cheap，部署时立即可查)
if sample.model_name not in train.model_names:
    return ("gnn", reason="unseen architecture")

# Tier 2: SHAP 驱动的 feature OOD
# 检查的特征不是手挑的，而是 SHAP 分析 XGBoost 后选出的：
# log1p(total_flops), log1p(total_memory_bytes)
# 这两个是 XGBoost 真正依赖的"架构区分特征"，OOD 时 XGB 必失败
for feat in ARCHITECTURE_FEATURES:
    if feat OOD vs train range:
        return ("gnn", reason=f"{feat} OOD")

# Default: 分布内 → 强 baseline 接管
return ("xgboost", reason="in-distribution")
```

**为什么不只是 if-else**：

- Tier 1 是 metadata 级别的快速通道
- Tier 2 是 per-sample 特征级 OOD 检测，特征**由独立的 SHAP 分析决定**（不是凭空选）
- 路由设计的"复杂度"在 **SHAP 反向 inform Router** 这个 evidence 链，不在 router 内部代码
- 每个决策都有可解释的 `reason`（为 demo 和 poster 服务）

**修改**：[scripts/train_and_eval.py](scripts/train_and_eval.py) 加 `run_router()`，复用已 fit 好的 xgb 和 trained gnn 的预测，并打印 tier breakdown 表。新增 `mixed=<m1,m2>` split。

**验收（已通过）**：

- 9 个外推 split 上 Router 数学上等于 GNN 或 XGBoost 列（确定性 by routing rule）
- mixed split 上 Router 39.9% **严格优于** GNN 46.0% 和 XGBoost 63.2%（per-sample 路由真有价值）

#### B. Demo / Extension（frontend，✅ 完成）

**新文件**：[scripts/predict.py](scripts/predict.py) (~80 行 CLI)

输入 / 输出契约：

```bash
# Case 1: 见过的模型 + 见过的 GPU
$ python scripts/predict.py --model gpt2-large --batch 4 --seq 256 --gpu h100
[Router] gpt2-large ∈ training set → use XGBoost (in-distribution)
[Predicted latency] 23.4 ms
[Reference (GNN, for comparison)] 28.9 ms

# Case 2: 见过的模型 + 未见的 GPU（zero-shot）
$ python scripts/predict.py --model bert-base --batch 8 --seq 128 --gpu b200
[Router] b200 ∉ training GPUs → still use XGBoost (hardware extrapolation regime)
[Predicted latency] 4.1 ms
[Reference (GNN)] 6.3 ms

# Case 3: 模拟未见架构（强制 GNN）
$ python scripts/predict.py --model gpt2-large --batch 4 --seq 256 --gpu h100 \
    --simulate-unseen-architecture
[Router] (simulated) architecture treated as unseen → use GNN
[Predicted latency] 28.9 ms (graph-aware, OOD-safe regime)
[Caveat] In production this regime applies when deploying a new model architecture.
```

**Poster screenshot 用 case 1 + case 3 对比**——同样输入，Router 行为不同，输出不同。一眼看懂"路由器在干啥"。

**实现要点**：

- 加载现有 `data/graphs/*.pkl` 作为 model graph 字典
- 加载 train fold 的 fitted XGBoost 和 trained GNN（可直接 pickle 缓存现有 train_and_eval 中间结果）
- 调用 router → 调用对应模型 → 打印格式化输出

**验收**：3 种 case 的命令都能跑出正确格式的输出，可截图。

#### C. SHAP 诊断（✅ 完成）

**新文件**：[scripts/shap_xgboost_diagnosis.py](scripts/shap_xgboost_diagnosis.py)（~165 行）
**结果**：[results/SHAP_FINDINGS.md](results/SHAP_FINDINGS.md) + 2 张 PNG

**实测发现**（held-out=gpt2-large，XGB MAPE 39.4%）：

| Rank | Feature | mean &#124;SHAP&#124; | OOD on test? |
|---|---|---:|---|
| 1 | `seq_len` | 24.41 | 0% |
| 2 | `batch_size` | 15.62 | 0% |
| 3 | `fp16_tflops` | 15.59 | 0% |
| **4** | **`log1p(total_memory_bytes)`** | **9.53** | **100%** ⚠️ |
| 5 | `bandwidth_gbs` | 7.55 | 0% |

**关键发现**（细化原假设）：

- XGBoost 的 top 3 特征是 config + hardware（`seq_len` / `batch_size` / `fp16_tflops`），它们覆盖均匀，**不 OOD**
- 第 4 名 `log1p(total_memory_bytes)` 是**唯一**架构区分特征——gpt2-large 在它上面 100% OOD（test [22, 22] vs train [18, 21]）
- 这就是 XGBoost 在架构外推上失败的精确机制：架构识别全靠 graph 聚合量，而聚合量对未见架构必然 OOD

**Poster §Diagnostic 框**（< 80 字 caption）：
> XGBoost's decisions are dominated by config + hardware features (top 3 SHAP) that are well-covered in training. The only architecture-distinguishing feature in its top-5 — `log1p(total_memory_bytes)` — is **100% OOD** on the held-out gpt2-large set (test [22] vs train [18, 21]). Tree models cannot extrapolate beyond cell boundaries; graph-aware models avoid this via per-op decomposition.

**SHAP 顺便驱动了 Router 设计**：Router Tier 2 检查的就是 SHAP 找出的这 2 个架构区分特征（`log1p(total_flops)` + `log1p(total_memory_bytes)`），不是手挑。

#### D. Poster 文案重写（~2h）

**Hero claim 替换**：

- ❌ 旧："27.9% MAPE on B200 zero-shot"
- ✅ 新："Two extrapolation axes, two regimes: tabular for hardware, graph-aware for architecture. Router achieves the best of both."

**§Related Work 加一段**：
> Recent graph-level cost models for LLM inference (NeuSight, ASPLOS'25; Habitat, MLSys'21) focus on cross-GPU generalization. Our work complements these by evaluating cross-architecture generalization within the Transformer family, characterizing when tabular feature representations break down.

**§Limitations 必写 5 条**：

1. **Scope 主动收缩**：CPU/GPU placement 和 ResNet 在 proposal 提及但本课程项目未实现，列入 future work
2. **HF eager mode only**：未对比 torch.compile / vLLM / TensorRT-LLM 等 serving 栈
3. **Forward-pass / prefill only**：autoregressive decode + KV cache 未覆盖
4. **XGBoost feature representation**：我们对比的 XGBoost 用 4 维 graph feature；richer per-op-type histogram 未评估，但 SHAP 显示 dominant feature 仍是 aggregate magnitude，OOD 外推问题可能保留
5. **架构外推范围**：仅 Transformer 家族内部（GPT2/BERT/T5 互留一作测试），不包括 ResNet / Mamba / MoE

### 4.3 时间预算

| 任务 | 估时 | 阻塞 |
|---|---:|---|
| A. Router 实现 | 2.0h | 无 |
| B. Demo / predict.py CLI | 1.5h | A 完成后 |
| C. SHAP 脚本 + 2 张图 | 1.5h | 无（与 A 并行） |
| D. Poster 文案 + 主表 + screenshot | 2.0h | A/B/C 全部完成后 |
| 整合 / 校对 | 1.0h | D 之后 |
| **总计** | **8.0h** | buffer 4h |

### 4.4 验收 checklist（poster ready）

- [ ] Router 9 行 MAPE 数字齐全，主表挂上 poster
- [ ] `predict.py` 至少 3 个 case 跑通，screenshot 截图入 poster
- [ ] SHAP 2 张图生成 + < 80 字英文 caption
- [ ] Hero claim 改为 "two extrapolation axes" 版本
- [ ] §Related Work 加 NeuSight + Habitat 引用
- [ ] §Limitations 写齐 5 条
- [ ] poster 不出现 "GNN beats XGBoost" 字样
- [ ] poster 出现 "complementary" / "regime" / "extrapolation axes" framing 词

---

## 5. 风险 & Fallback

| 风险 | 概率 | 应对 |
|---|---|---|
| Router 跟 min(XGB, GNN) 没区别 → "这不就是 if-else" 质疑 | 高 | poster 上**主动承认** "rule is intentionally simple"；价值在系统化界定 regime，不在路由复杂度 |
| `predict.py` 加载已训 GNN 时遇坑（state_dict 路径） | 中 | fallback：从 `train_and_eval.py` 跑一次 leave-model split，用 in-memory GNN 对象直接做 demo |
| SHAP 在 XGBoost 上跑不动 | 极低 | 用 `xgboost.plot_importance` 或 sklearn `permutation_importance` 替代 |
| 时间超预算 | 中 | 砍 demo case 3（保留 case 1 + 2）；砍 SHAP 第 2 张图（保 1 张 + 文字） |
| 陈天奇追问 "你们 XGB feature 太弱" | 高 | §Limitations 第 4 条 + SHAP 诊断已主动覆盖；回答 "agreed, this is a representation limitation we explicitly characterize" |
| 陈天奇追问 "你们没测 ResNet/Mamba" | 中 | "scope 限制，写在 §Limitations，future work 第 1 项" |

---

## 6. 最终交付物

### 6.1 Poster（PDF）

包含：

- Title + 三人作者
- Hero claim（"two extrapolation axes" 版本）
- 主表（Extrapolation Results，9 行 × 4 method）
- Demo screenshot（`predict.py` 3 个 case）
- SHAP 诊断（1-2 张图 + caption）
- Related Work（NeuSight + Habitat + Ansor）
- Limitations（5 条）

### 6.2 代码库（GitHub）

| 路径 | 角色 | 状态 |
|---|---|---|
| [src/hetero_cost_model/](src/hetero_cost_model/) | 全部已有 | ✅ |
| [src/hetero_cost_model/router.py](src/hetero_cost_model/router.py) | Router | 🔴 sprint 新增 |
| [scripts/train_and_eval.py](scripts/train_and_eval.py) | 已有 + 加 router | ⚠️ 修改 |
| [scripts/predict.py](scripts/predict.py) | Demo CLI | 🔴 sprint 新增 |
| [scripts/shap_xgboost_diagnosis.py](scripts/shap_xgboost_diagnosis.py) | SHAP | 🔴 sprint 新增 |
| README（含一键复现 + demo 用法） | | 🔴 sprint 完善 |

### 6.3 数据集 + 训练产物（已 committed）

- `data/raw/*.csv` — 7 GPU × 6 模型 × 48 配置 = 1903 样本
- `data/graphs/*.pkl` — 6 模型 GraphRepr pickle
- `results/*.json` — 16 轮 Phase 6 sweep 全部 prediction + metrics

---

## 7. 历史时间线（参考）

详细 day-by-day 任务、Modal SKU lock、踩坑记录等见 git history (commits before 2026-04-18) 与 [results/EXPERIMENTS.md](results/EXPERIMENTS.md) Phase 1-6。

**关键里程碑**：

- 2026-04-17：Day 1-2 全绿，6 模型图提取通过（HF fx + transformers<4.52）
- 2026-04-17：7 GPU 数据采集完成（1903 样本）
- 2026-04-17：v1 → v2 GNN 重设计（per-node s/h + sum readout + global skip）
- 2026-04-18：16 轮 Phase 6 sweep 跑完，三张表数字齐备
- 2026-04-29：poster 冲刺（Router + SHAP + Demo）启动
- 2026-04-30：poster 截止
