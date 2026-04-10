# Infinigen 仿真数据对齐 MINT 训练——详细方案与执行计划

> **本文档版本**：v1.1（2026-04-10 修正）
> **生成日期**：2026-04-10
> **Sovereign 入口**：E045 / E046 / E047
> **⚠️ 重要修正**：本文档 v1.0 中关于"E048 MINT 模型加载损坏"的描述是错误的。实测 external/MINT@4eab579 的 `MINTPolicy.from_pretrained()` 输出 `Missing keys: 1`（仅 embed_tokens.weight），模型加载本身没有问题。
> **目标**：让 MINT 在 Infinigen + AnyGrasp + MuJoCo 数据上学会开抽屉，达到 held-out seeds > 0/5 成功率

---

## 1. 问题诊断总结

### 当前状态
| 项目 | 状态 | 说明 |
|------|------|------|
| 官方 MINT baseline | ✅ 正常 | p1c11 = 3/3（LIBERO drawer） |
| MINT 训练 | ✅ 完成 | 8000 步，loss → 0.000 |
| Infinigen 评估 | ❌ 失败 | pretrained 0/5 + finetuned 0/5 |

### ⚠️ 重要修正
**之前的 E048 推断（模型加载损坏）是错误升级**。实测结果：
- `Missing keys: 1`（仅 `embed_tokens.weight`）
- `paligemma.model` 和 `paligemma.model.get_image_features()` 均存在
- 模型加载本身没有问题

### 失败根因（基于修正后的判断）

#### 🔴 P0 — 尚未建立

**Exact-Control Alignment 未建立**
- p1g 中 `state_gap={}` 和 `action_gap={}`，没有硬统计证据
- 缺少官方 LIBERO drawer 成功轨迹的 state/action traces
- 无法确定 Infinigen 与 LIBERO 在 state/action 维度上的真实差距

#### 🟠 P1 — 已确认但优先级待定

**E050：视觉信息量不足**
- Infinigen 图像 std_rgb ≈ 0.06，LIBERO ≈ 0.20（**3.3x 差距**）
- edge_density：0.024 vs 0.063（**2.6x 差距**）
- entropy：< 1.0 vs 6.77（**12x+ 差距**）
- 根因：`_calibrate_image()` 只做亮度增益，没有纹理合成

**Action Rotation 恒为 0**
- `action[3:6]`（rotation delta）在所有 500 个训练帧中恒为 0.0
- 根因：`_script_action()` 只计算平移 + gripper，rotation 未初始化
- **⚠️ 需要 Phase 0.2 的 control action trace 来确认 LIBERO 中的 rotation 范围**

**State Proxy**
- `state[3:7]` 是合成的 motor proxy（末端到把手相对位移），不是真实关节角
- **⚠️ 需要 Phase 0.2 的 control state trace 来确认 LIBERO 中的关节角分布**

**数据规模不足**
- 5 seeds × 10 重复 = 500 帧（LIBERO 有 273,465 帧，**1/547**）

**推理配置不一致**
- `n_action_steps=1`（官方 baseline 用 4）

---

## 2. 科学验证节点（Gate）

| Gate | 验收条件 | 对应阶段 |
|------|---------|---------|
| G1 | Control state trace 捕获完成 | Phase 0 |
| G2 | Control action trace 捕获完成 | Phase 0 |
| G3 | state[3:7] Wasserstein 距离量化 | Phase 1 |
| G4 | action[0:6] Wasserstein 距离量化 | Phase 1 |
| G5 | edge_density > 0.04, entropy > 4.0, std_rgb > 0.12 | Phase 4 |
| G6 | finetuned > pretrained，且至少 1 个 seed drawer_fraction > 0.1 | Phase 7 |

---

## 3. 分阶段执行计划

### Phase 0：精确 Control Surface 重现

**目标**：建立严格的一对一 control baseline，作为后续所有对齐的参考基准。

**操作**：
1. 重新干净运行 p1c11（解决 returncode bookkeeping 冲突）
2. 捕获官方 LIBERO drawer 成功轨迹的 **state trace**（8D × 每帧 × 10 episodes）
3. 捕获官方 LIBERO drawer 成功轨迹的 **action trace**（7D × 每帧 × 10 episodes）
4. 计算 control 分布的均值和方差

**产出**：`control_state_trace.json`、`control_action_trace.json`、`control_video_metrics.json`

**验收**：可以与 target 数据做逐帧 Wasserstein 距离比较

---

### Phase 1：Exact Control State/Action 对齐验证

**目标**：用 Phase 0 捕获的 traces 与 Infinigen target 数据做严格统计对比。

**操作**：
- State：计算 state[3:7] 的 Wasserstein 距离（Infinigen target vs LIBERO control）
- Action：计算 action[0:6] 的 Wasserstein 距离（Infinigen target vs LIBERO control）
- Visual：建立像素级统计 baseline（std_rgb, edge_density, entropy）

**验收**：Gap 量化完成，确定修复优先级

---

### Phase 2：Rotation 动作修复（条件触发）

**目标**：如果 Phase 1 确认 rotation 差距显著，在 `_script_action()` 中引入姿态旋转。

**触发条件**：LIBERO control action[3:6] 有显著非零值

**操作**：
- 文件：`scripts/mint/drawer_robot_env_mujoco.py`（第 394-399 行）
- 在 `_script_action()` 中，基于 `target_pos - current_pos` 向量计算到达把手所需的末端执行器旋转

**验收**：`action[3:6] std > 0.01`

---

### Phase 3：State 语义修复（条件触发）

**目标**：如果 Phase 1 确认 state[3:7] 差距显著，用真实的机械臂关节角替代合成的 motor proxy。

**触发条件**：LIBERO control state[3:7] 与 Infinigen state[3:7] Wasserstein 距离显著

**操作**：
- 文件：`scripts/mint/drawer_robot_env_mujoco.py`（第 242-258 行）
- 从 MuJoCo URDF 解析机械臂关节顺序
- 用 `self.data.qpos[arm_joint_indices]` 替代 `_synthetic_motor_state()`

**验收**：state[3:7] Wasserstein 距离 < 阈值

---

### Phase 4：视觉校准增强

**目标**：提升 Infinigen 渲染图像的纹理和边缘信息量。

**短期方案**（快速修复）：
- 在 `_calibrate_image()` 中加入边缘感知增强

**长期方案**（根本修复）：
- 修改 Infinigen 的 URDF 生成 pipeline，为 mesh 添加 PBR 材质

**验收**：**G5 通过**（edge_density > 0.04, entropy > 4.0, std_rgb > 0.12）

---

### Phase 5：数据规模扩展

**目标**：从 500 帧扩展到 3000+ 独立帧。

**操作**：
- MIN_ANYGRASP_SEEDS：4 → 10
- SUCCESS_REPEAT：10 → 2
- MAX_STEPS：96 → 200

**产出**：新 dataset = 15+ seeds × 2 重复 × 200 帧 = 6000 帧

**验收**：episode_count > 30, unique seeds > 10, frame_count > 3000

---

### Phase 6：推理路径对齐

**目标**：确保 eval 时的推理配置与官方 baseline 一致。

**操作**：
- 在 `_load_policy_bundle()` 中显式设置 `policy.config.n_action_steps = 4`
- 对齐 pre/post-processor 配置与 LeRobot eval.py

---

### Phase 7：完整训练与评估

**操作**：
1. MINT 训练（16000 步）
2. 在 held-out seeds（16-20）上评估 pretrained 和 finetuned

**验收**：**G6 通过**
- finetuned max_drawer_fraction > pretrained
- 至少 1 个 seed 的 drawer_fraction > 0.1
- 帧间像素 diff > 2.0

---

## 4. 执行顺序与依赖关系

```
Phase 0（独立，可立即执行）
  ↓
Phase 1（State/Action 对齐验证）
  ↓ Phase 1 结果
Phase 2 或 Phase 3（条件触发）
  ↓
Phase 4（视觉校准，P1）
Phase 6（推理路径，P1）
Phase 5（数据扩展，P1，可与 Phase 4/6 并行）
  ↓
Phase 7（Gate 验证）
```

---

## 5. 资源估算

| 阶段 | GPU 时间 | 预计运行时长 |
|------|---------|------------|
| Phase 0（Control traces 捕获） | 0 GPU | ~4 小时 |
| Phase 1（对齐验证） | 0 GPU | ~2 小时 |
| Phase 2（Rotation 修复） | 0 GPU | ~1 小时 |
| Phase 3（State 修复） | 0 GPU | ~2 小时 |
| Phase 4（视觉校准） | 0 GPU | ~4 小时 |
| Phase 5（数据扩展） | ~1 小时 AnyGrasp | ~6 小时 |
| Phase 6（推理路径） | 0 GPU | ~1 小时 |
| Phase 7（训练 + 评估） | ~8 小时 MINT train | ~10 小时 |
| **总计** | **~9 小时 GPU** | **~30 小时** |

---

## 6. 关键文件索引

| 文件 | 说明 |
|------|------|
| `external/MINT/lerobot_policy_mint/src/lerobot_policy_mint/modeling_mint.py` | MINT 模型实现 |
| `scripts/mint/drawer_robot_env_mujoco.py` | MuJoCo 环境，含 `_script_action()` 和 `_calibrate_image()` |
| `scripts/mint/dataset_builder.py` | LeRobot dataset 打包 |
| `experiments/mint/mint_drawer_v1/artifacts/p1c11_*.json` | 官方 baseline（3/3 成功） |
| `experiments/mint/mint_drawer_v1/artifacts/m7_*.json` | 当前 Infinigen 评估（0/5 失败） |
| `experiments/mint/mint_drawer_v1/outputs/p1g_*.md` | exact-control 对齐审计 |

---

## 7. Sovereign 证据链

| Evidence ID | 内容 | 置信度 |
|------------|------|--------|
| E036 | p1c11 官方 baseline（3/3） | 高 |
| E045 | p1f 运行报告 | 高 |
| E046 | p1g exact-control alignment audit | 中（state_gap={}, action_gap={} 待补） |
| E047 | p1h 工作集清理 | 高 |
| E033 | Key remap 修复记录 | 高 |
| E034 | Forward pass 诊断 | 中 |

---

## 8. 经验教训

### E048 错误升级的错误

**发生了什么**：
- 在没有 A800 现场实测的情况下，基于代码审查推断 MINT 模型加载有 604 vision + 603 LM key 的不匹配
- 错误地将这个推断写入了 sovereign verdict 和 Phase 8 升级
- 创建了 E048/E049/E050 evidence 文件，并错误地将 action rotation 恒为 0 升级为 P0 根因

**为什么会错**：
1. 没有在目标环境（A800）上运行实测验证
2. 将从旧版代码分析中看到的 warning 推断为实际缺失的 key 数量
3. 将"假设"和"已确认"明确区分不足

**正确的做法**：
- 在声称"P0 根因"之前，必须在目标环境上运行实测验证
- 将"假设"和"已确认"明确区分
- 不将未验证假设写入 sovereign

---

*本文档作为 task_plan.md 的补充，提供可读的方案说明。核心计划以 task_plan.md 为准。*
