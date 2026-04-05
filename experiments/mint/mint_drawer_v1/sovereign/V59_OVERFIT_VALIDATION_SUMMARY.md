# V59 验证实验完整报告 — Overfit Validation + Env Gate Report

**初始生成**: 2026-04-05T17:00:00+08:00  
**E022 更新**: 2026-04-05T20:35:00+08:00  
**最终修订**: 2026-04-05T22:00:00+08:00  
**实验阶段**: Phase 2 — Conditioning Design  
**Campaign**: mint_drawer_v1  

---

## 执行摘要（E022 后修正版）

| 项目 | 修正前（E021 后） | 修正后（E022 后） |
|------|------------------|------------------|
| **E021 结论** | OVERFIT SUCCESS — Pipeline 确认可用 | **离线信号正向 — 任务级未确认** |
| **E022 结论** | （未运行） | **WEAK_PASS — 两边都是 0.000** |
| **综合 verdict** | `PIPELINE_CONFIRMED — FULL_RETRAIN_PROCEED` | **`V59_ENV_GATE_WEAK_PASS — PIPELINE_ISSUE_DEEPER_THAN_IMAGE_QUALITY`** |
| **P0b 彩色重录** | ~~CANCELLED~~ | **待定 — 仍可能是解决方案，但优先级低于 RCA** |
| **下一步** | 全量 V59 retrain | **根因分析（RCA）— 首要** |

**一句话结论（E022 后）**:

> MINT 在白图像数据集上能更好地拟合离线动作（E021: +51.7% RMSE, +78.0% Gripper MAE），但这个改善没有转化为仿真中的任何任务成功（E022: 两边 drawer_fraction = 0.000）。离线模仿改善 ≠ 任务级成功。当前 pipeline 有比图像质量更深的根因。

---

## 第一部分：E021 — 离线 Overfit 验证

**时间**: 2026-04-05T16:46 → 17:00 CST  
**证据**: E021.yaml, v59_overfit_val.json, v59_overfit_eval.json

### 1.1 实验背景

V58 训练 0% 之后，需要验证 MINT 是否能在干净数据集上学习。核心问题：

> 在白图像（mean=0.972, std=0.003）数据集上，MINT 能否学到任何东西？

**实验设计**：在 494 帧（7 个 episodes）、200 步上做小规模 overfit 训练，对比 pretrained vs finetuned 的离线动作预测质量。

### 1.2 训练配置

| 参数 | 值 |
|------|-----|
| Episodes | 0, 1, 2, 3, 4, 5, 49（共 7 个） |
| Frames | 494 |
| Steps | 200 |
| Batch size | 8 |
| LR | 2.5e-4 |
| Dataset version | v59_20260405 (19,701 total) |

### 1.3 E021 结果

**训练 Loss**：6.520 → 3.741（-42.7%）  
**离线动作预测评估**（400 帧）：

| Metric | Pretrained | Finetuned | 提升 |
|--------|-----------|-----------|------|
| Action RMSE | 0.3081 | 0.1487 | **+51.7%** |
| Gripper MAE | 1.1159 | 0.2453 | **+78.0%** |

### 1.4 E021 的 Scope Limitation（关键）

**E021 只证明了离线动作模仿改善，没有证明任务级成功。**

E021 **不包含**：
- 仿真中 drawer success rate
- attach/open 序列正确性
- drawer fraction / pull distance 对比
- white vs colored 的 A/B 对照

**E021 不能支持的结论**：
- "Pipeline 已确认可用"
- "白图像不是 hard blocker"（需要 A/B 对照）
- "可以全量 retrain"
- "P0b colored reroll 可以取消"

### 1.5 E021 的正确意义

E021 证明了：
- V59 数据集 state/action 格式与 MINT 兼容 ✅
- MINT 能在近白图像中利用 state 向量信号 ✅
- 离线动作回归 pipeline 端到端可工作 ✅

E021 没有证明：
- MINT 学会了完成 drawer 任务 ❌
- 白图像对任务成功不重要 ❌
- 可以跳过 env-level 验证直接 retrain ❌

---

## 第二部分：E022 — 环境级 Overfit Gate

**时间**: 2026-04-05T20:20 → 20:35 CST  
**证据**: E022.yaml, v59_overfit_env_gate.json  
**脚本**: `scripts/mint/run_v59_overfit_env_gate.py`

### 2.1 实验设计

在 E021 训练得到的 finetuned checkpoint（step 200）上，跑 DrawerRobotEnv 仿真 rollout，对比 pretrained vs finetuned 在同一批 seeds 上的表现：

| 参数 | 值 |
|------|-----|
| Seeds | 1, 2, 3 |
| Episodes per seed | 2 |
| Max steps | 96 |
| Rollouts per policy | 6 |
| Checkpoint | `v59_overfit_outputs/checkpoints/000200/pretrained_model/` |

### 2.2 E022 结果

| Metric | Pretrained | Finetuned | Gain |
|--------|-----------|-----------|------|
| strict_success_rate | **0.000** | **0.000** | +0.000 |
| mean_drawer_fraction | **0.000** | **0.000** | +0.000 |
| attach_rate | **0.000** | **0.000** | +0.000 |
| mean_pull_distance | **0.000** | **0.000** | +0.000 |

**所有 12 个 rollout（2 policies × 3 seeds × 2 episodes）全部 0.000 drawer_fraction。**

**Pretrained total_eef_motion**：均值约 4.73m（全在动）  
**Finetuned total_eef_motion**：均值约 11.37m（动得更多）  
**两边 non_zero_action_ratio = 1.0**：两边都在输出非零动作，但从未 attach/draw。

### 2.3 WEAK_PASS 的含义

Gate 阈值：`finetuned > pretrained on ALL metrics`。  
Finetuned 匹配了 pretrained（两边都是 0.000），但**没有超过**。  
"weak pass" = finetuned 没有比 pretrained 更差 = 不是负信号，但也不是正信号。

**关键**：两边都是 0.000 意味着：
1. **Pretrained MINT 从未学会在 Infinigen 仿真中打开抽屉**
2. **Finetuned MINT 在离线改善了动作模仿，但改善的动作仍然不能完成任务**

### 2.4 E022 的关键诊断

**与 V58 eval（2026-04-04）完全一致**：E015 报告 pretrained=0.000, finetuned=0.000 on held-out seeds。E022 证明了即使在 training seeds 上，Pretrained MINT 也是 0.000。

**最可能的根因**：

1. **教师演示质量问题（RCA1，最高优先级）**
   - 如果训练 episodes 本身不成功（AnyGrasp 抓取不够稳固），MINT 学到的是不成功的动作
   - 需要验证：episodes 0-5, 49 的 drawer_fraction trace 是否 > 0

2. **Action/State 对齐问题（RCA2/RCA3）**
   - MINT 输出通过 IDENTITY normalization 直接传给 env.step
   - 但 MINT 训练时的 action 来自 Infinigen physics，可能与 env.step 的期望不匹配
   - State 向量格式可能存在系统性偏差

3. **白图像问题（P0b，优先级降级）**
   - ER_TINY_RENDERER 白图像仍然存在（E020）
   - 但 E022 证明问题在白图像和彩色图像下都存在
   - 彩色图像**可能**帮助 vision encoder，但**不能**解决 action/state 对齐问题

---

## 第三部分：修正后的叙事

### 3.1 Codex Review 的质疑（全部有效）

| Codex Finding | 核实结果 |
|-------------|---------|
| "E021 只证明离线动作改善，不是任务成功" | ✅ 正确。E021 scope_limitation 明确写了 |
| "pipeline confirmed / full retrain proceed 说过头" | ✅ 正确。E022 后已修正 |
| "V58 实际用了 3305 frames 说法不对" | ✅ 正确。V58 训练用 18701 帧；3305 是训练后被覆盖的残留 |
| "白图像非 blocker 证据不够" | ✅ 正确。E021 只能证明离线改善 |
| "P0b colored reroll 不能取消" | ✅ 正确。降级为"待定"，不是 cancelled |

### 3.2 修正后的 Root Cause 分级

**E022 后的根因分析框架**：

| 优先级 | 根因 | 证据 | 状态 |
|--------|------|------|------|
| **P0** | 教师演示质量 | 两边 drawer_fraction=0.000（E022）；V58 eval 同样（E015） | 待验证（RCA1） |
| **P0** | Action/State 对齐 | 两边 never attach；pretrained 也 0.000 | 待验证（RCA2/3） |
| **P1** | 白图像（P0b） | ER_TINY 限制确认（E020）；但不是唯一根因 | 待定 |

**注意**："near-white images 是罪魁祸首"（E021 后的叙事）现在站不住——因为 pretrained MINT 在 LIBERO 彩色图像上训练，从未见过 Infinigen 白图像，但也得到了 0.000。如果图像是主因，pretrained 应该有不同的 baseline。

### 3.3 修正后的下一步

```
DO:
  ✓ V59 clean dataset（240/19701）— 确认可用
  ✓ E021 离线改善验证 — 确认 MINT 能拟合
  ✓ E022 env gate — 确认 pipeline 有更深问题
  → RCA1: 验证训练 episodes 本身是否任务成功
  → RCA2: Action normalization 对齐检查
  → RCA3: State representation 对齐检查

DO NOT:
  ✗ 全量 V59 retrain（会重复 V58 失败）
  ✗ "P0b colored reroll = cancelled"（应该 = 待定）
  ✗ "白图像是唯一根因"（应该 = 多因素）
```

---

## 第四部分：Sovereign State 更新记录

### 4.1 时间线

| 时间 (CST) | 事件 | 结论 |
|-----------|------|------|
| 2026-04-05 14:35 | V59 clean dataset 完成（240/19701） | 数据集干净 |
| 2026-04-05 16:30 | P0b ER_TINY 限制确认（E020） | 图像渲染问题是 ER_TINY 架构限制 |
| 2026-04-05 17:00 | E021 离线 overfit 验证完成 | 离线信号正向，任务级待验证 |
| 2026-04-05 17:00 | **Sovereign verdict 错误更新** | `MINT_PIPELINE_CONFIRMED — FULL_RETRAIN_PROCEED` |
| 2026-04-05 20:35 | E022 env gate 完成 | **WEAK_PASS — 两边 0.000** |
| 2026-04-05 20:35 | Sovereign verdict 修正 | `V59_ENV_GATE_WEAK_PASS — PIPELINE_ISSUE_DEEPER_THAN_IMAGE_QUALITY` |
| 2026-04-05 22:00 | **Narrative 统一修正**（本次） | 所有文件中叙事已同步 |

### 4.2 当前 Sovereign State

```
verdict: V59_ENV_GATE_WEAK_PASS — PIPELINE_ISSUE_DEEPER_THAN_IMAGE_QUALITY
phase: v59_ENV_GATE_COMPLETED
active_queue: root_cause_analysis

Next Actions:
  [P0] root_cause_analysis — RCA1: teacher quality; RCA2/3: action/state alignment
  [P1] P0b_colored_reroll — UNDECIDED; only proceed if RCA clears it

Evidence:
  E021: offline improvement (+51.7% RMSE, +78.0% MAE) — SCOPE LIMITED
  E022: env gate WEAK_PASS (both=0.000 drawer_fraction) — KEY NEGATIVE SIGNAL
  E020: ER_TINY limitation (white images) — CONFIRMED but NOT SOLE CAUSE
```

---

## 附录 A. 关键文件索引

| 文件 | 路径 |
|------|------|
| V59 overfit 训练脚本 | `scripts/mint/run_v59_overfit_val.py` |
| V59 env gate 脚本 | `scripts/mint/run_v59_overfit_env_gate.py` |
| V59 overfit 训练日志 | `artifacts/v59_overfit_val.log` |
| V59 overfit 训练结果 | `artifacts/v59_overfit_val.json` |
| V59 overfit 评估结果 | `artifacts/v59_overfit_eval.json` |
| V59 env gate 结果 | `artifacts/v59_overfit_env_gate.json` |
| V59 overfit checkpoint | `artifacts/v59_overfit_outputs/checkpoints/000200/pretrained_model/` |
| V59 Dataset Manifest | `artifacts/current_dataset_manifest.json` |
| Sovereign Evidence E021 | `sovereign/evidence/E021.yaml` |
| Sovereign Evidence E022 | `sovereign/evidence/E022.yaml` |
| HARNESS Hygiene | `sovereign/HARNESS_HYGIENE.md` |

---

*本报告最终修订于 E022 后（2026-04-05T22:00:00+08:00）。*  
*Sovereign verdict: `V59_ENV_GATE_WEAK_PASS — PIPELINE_ISSUE_DEEPER_THAN_IMAGE_QUALITY`*  
*核心结论：离线动作改善（E021）≠ 任务级成功（E022）。需要 RCA。*
