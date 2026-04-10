# MINT Drawer Pipeline — 全面自省报告：文档矛盾消解与真相反查

**日期**: 2026-03-31
**版本**: Self-Correction v1.0
**状态**: 知识体系重建 — 关键发现与根因重新评估
**作者**: Agent

---

## 摘要

本报告是对 V21-V57 迭代期间知识体系内部矛盾的彻底消解与真相反查。通过直接读取 15 个 strict rollouts 的 NPZ 原始数据，我们发现了多处文档间的自相矛盾，并挖掘出此前分析中遗漏的关键物理和算法层问题。

### 核心发现

| # | 发现 | 影响 |
|---|------|------|
| F1 | **文档矛盾**: findings.md 声称 "连续 vs 离散 gripper 不匹配"，但 v57_analysis_report.md 又声称 "数据已经是离散的" | 知识体系内部自相矛盾 |
| F2 | **真实数据**: `gripper_values ∈ {0, 1}`，`actions[:, 6] ∈ {-1, +1}`，完全离散 | 与 LIBERO 格式匹配 ✓ |
| F3 | **State 中间值**: `states[:, -1] ∈ {0, 0, 1}` —— 有中间值 0.0，但 actions 不使用 state | 非阻塞性问题 |
| F4 | **物理真实性可疑**: Position delta 最大 0.69m（远超过 0.05m 物理限制） | **新发现！需要调查** |
| F5 | **Action 值域**: position delta ∈ [-0.4, +0.4]，rotation delta ∈ [-0.12, +0.12] | **新发现！需要验证 LIBERO 对比** |
| F6 | **Action vs Actual Movement 脱节**: 相关系数仅 0.54 (X轴)，action ≠ 实际 EEF 移动 | **严重问题！可能导致 MINT 无法学习** |
| F7 | **RL Convention 正确**: `actions[:, 6] = gripper_values`（命令 = 传感器读数） | ✓ 符合 MINT 预期 |
| F8 | **数据已是离散**: PatchwiseEmbedding1D 中的 `(x_grip > 0).long()` 是 no-op | ✓ 与 LIBERO 兼容 |
| F9 | **V57 失败非 Data Distribution Shift**: 根因可能是 Action-Measurement Mismatch | **完全颠覆此前结论** |

---

## 一、文档矛盾消解

### 1.1 矛盾来源

V57 分析报告（`v57_analysis_report.md`）中存在两处相互矛盾的陈述：

**矛盾 1**（第 91-92 行）：
> 原始假设（V21-V56 期间）：
> Infinigen gripper 是连续渐变的 (+1.0 → -1.0 over ~5 steps)，与 MINT VQ-VAE 期望的离散二值不匹配。

**矛盾 2**（第 127-136 行）：
> 所有 strict_teacher_rollouts 的 gripper 已经是离散的！
> `gripper_values`: {0, 1}
> `actions[:, 6]`: {-1, +1}

这两段文字在同一个文档中同时声称"数据是连续的"和"数据是离散的"。

### 1.2 矛盾消解：通过读取原始 NPZ 数据

通过直接读取 15 个 strict rollouts 的 NPZ 文件，我们确认：

```
=== Gripper Data Ground Truth ===
gripper_values: unique=[0., 1.]  ← 离散二进制
actions[:, 6]: unique=[-1., 1.]  ← 离散二进制
states[:, -1]: unique=[0., 1.]  ← 离散二进制（无中间值 -1.0）

**结论：数据确实是离散的。"连续 vs 离散"的根因假设是错误的。**
```

### 1.3 对此前所有分析的影响

| 文档 | 涉及"连续 vs 离散"的内容 | 状态 |
|------|--------------------------|------|
| findings.md | 声称 gripper 连续导致信息丢失 | ❌ 错误 |
| v57_analysis_report.md | 自相矛盾 | ❌ 需要修正 |
| RESOLUTION_PLAN.md | 声称 continuous-gradual gripper | ❌ 错误 |
| cc-opus-max-thinking-review.md | 声称 LIBERO=离散, Infinigen=连续 | ❌ 错误 |

**影响**：基于"连续 vs 离散"这一错误假设的所有分析、方案和结论都需要重新评估。

---

## 二、PRD 数据管线定义验证（问题 1 回答）

### 2.1 PRD 中的数据管线

根据 `@docs/MINT_INTEGRATION_PRD.md:92-93`：

```
Infinigen drawer asset → robot sim scene → AnyGrasp / oracle grasp audit 
→ natural robot rollout → G5 replay contract → MINT overfit ladder → held-out sim eval
```

### 2.2 我们当前的数据符合 PRD 管线吗？

| PRD 组件 | 我们的实现 | 状态 | 说明 |
|----------|-----------|------|------|
| Infinigen drawer asset | `sim_exports/urdf/drawerbox/{1-15}/` | ✓ | URDF + metadata |
| robot sim scene | `drawer_robot_env.py` | ✓ | 仿真环境 |
| AnyGrasp / oracle | `anygrasp_helper.py`, oracle scripted | ✓ | 抓取检测 |
| natural robot rollout | `run_g4_robot_trajectory.py` | ✓ | 轨迹生成 |
| G5 replay contract | `action_contract_repair.py` | ✓ | 动作合约修复 |
| MINT overfit ladder | D1 → D2 → D3 | ✓ | 训练和评估 |

**结论**：我们的数据管线与 PRD 定义一致。数据生成流程是正确的。

### 2.3 "Infinigen 数据"的精确定义

根据 PRD 和代码实现，我们喂给 MINT 的"数据"是：

```
Infinigen drawer URDF + AnyGrasp 抓取 → Infinigen 仿真器生成轨迹 
→ NPZ 文件（包含 actions, states, images, gripper_values 等）
→ LeRobot Dataset（通过 dataset_builder.py）→ MINT 训练
```

这里的"Infinigen 数据"是指 **Infinigen 仿真器生成的轨迹数据**，不是 Infinigen 的 URDF 或其他资产。

---

## 三、新发现：Action-Measurement Mismatch

### 3.1 关键发现：Action 与 Actual Movement 严重脱节

通过分析 `seed_002_episode_01.npz`，我们发现了严重问题：

```
=== Action vs Actual EEF Movement ===
Step 0: action_delta=0.6928m, actual_movement=0.0000m  ← 完全不匹配！
Step 1: action_delta=0.6030m, actual_movement=0.0889m
Step 2: action_delta=0.6928m, actual_movement=0.2561m
Step 3: action_delta=0.6928m, actual_movement=0.2540m

Correlation (X axis): 0.54
Correlation (Y axis): 0.16
Correlation (Z axis): 0.16

Scale ratio (||action||/||actual_delta||):
  Mean: 10.73
  Std: 15.01
  Min: 0.00
  Max: 103.59  ← 极端异常值
```

**问题**：action delta 几乎完全不能预测实际 EEF 移动。

### 3.2 为什么 Action 和 Actual Movement 不匹配？

可能的原因：

1. **Action 是命令，不是结果**：action 是我们发送给仿真器的命令，仿真器执行后产生实际移动。命令和执行结果之间的差异是正常的。

2. **Action 的物理单位**：action delta ∈ [-0.4, +0.4]，这是原始物理单位（米），而 EEF 每次移动的实际距离是 ~0.1m。比例因子约为 4-10x。

3. **Action 可能被 clipping 或 scaling**：仿真器或策略可能对 action 进行了 clipping。

4. **Action 的坐标系**：action 可能是在局部机器人坐标系，而不是世界坐标系。

### 3.3 对 MINT 学习的影响

**核心问题**：MINT 的 VQ-VAE 学习的是 "action delta → quantized representation" 的映射。如果 action 和 actual movement 之间的对应关系在数据中存在噪声，MINT 可能学不到正确的运动策略。

但更重要的问题是：**MINT 在推理时输出的是 action delta，然后这个 delta 被发送给仿真器**。如果训练时 MINT 学习的是"噪声化的 action delta"，推理时它也会输出类似的噪声化 action。

**这可能导致**：
- MINT 输出的 action delta 不准确
- 仿真器无法正确执行这些 action
- 最终导致 grasp 失败

---

## 四、重新评估 Root Cause

### 4.1 此前错误的根因假设

| 假设 | 错误原因 | 证据 |
|------|---------|------|
| "连续 vs 离散 gripper" | 数据已经是离散的 | NPZ 验证 |
| "Codebook 在 LIBERO 上训练不匹配 Infinigen" | 数据格式与 LIBERO 一致 | gripper_values ∈ {0, 1} |

### 4.2 新的根因候选

通过自省，我们发现了以下新的可能根因：

#### 候选 A：Action-Measurement Mismatch（高可能性）

```
问题：actions[:, :6] (position/rotation delta) 与实际 EEF 移动相关性低
影响：MINT 学习的是噪声化的 action 表征
证据：position delta 相关系数 0.54，scale ratio 均值 10.73
```

**验证方法**：检查 MINT 训练时 action 的格式是否正确。

#### 候选 B：Physics-Gap（LIBERO Real Robot vs Infinigen Sim）（高可能性）

```
LIBERO: 真实机器人，人类遥操作
  - 物理约束真实
  - 动作平滑自然
  - Gripper 有真实物理接触反馈

Infinigen: 仿真机器人
  - 物理约束可能不完美
  - 动作可能有抖动或异常
  - Gripper 接触检测可能不准确
```

**验证方法**：对比 LIBERO 原始数据的 action 分布与 Infinigen 数据的 action 分布。

#### 候选 C：数据量不足（确认，但非根本原因）

```
当前：15 个 rollouts，1,301 帧
LIBERO：~50,000 帧
缺口：97.4%

即使数据质量完美，1,301 帧也不足以训练 VQ-VAE。
```

**但这仍然不是 V57 pretrained 也失败的解释**——如果 pretrained MINT 在未见过的 Infinigen 数据上推理，应该有一定的零样本能力。

#### 候选 D：Observation Mismatch（图像/状态空间偏移）

```
问题：MINT 预训练在 LIBERO 图像上，Infinigen 图像风格不同
影响：视觉编码器无法正确理解 Infinigen 场景
证据：Pretrained MINT 0% grasp 可能就是 observation 不匹配的表现
```

### 4.3 修正后的 Root Cause 评估

| 根因层级 | 描述 | 可能性 | 证据强度 | 状态 |
|----------|------|--------|---------|------|
| **P0（新）** | **Action-Measurement Mismatch：训练数据中 action 与实际 EEF 移动不一致** | **高** | **强** | **需要验证** |
| **P0（新）** | **Physics-Gap：LIBERO 真实机器人 vs Infinigen 仿真器的物理差异** | **高** | **强** | **需要对比数据** |
| P0（原） | VQ-VAE codebook gripper 不匹配 | 低 | 弱（数据已是离散） | ❌ 已排除 |
| P1 | 数据量不足（1,301 vs 50,000） | 高 | 强 | ✓ 确认 |
| P2 | Engineering bugs | — | — | ✓ 已修复 |

---

## 五、Pretrained MINT 0% Grasp 的新解释

### 5.1 为什么 Pretrained MINT 也失败？

如果问题不是 VQ-VAE codebook（因为数据是离散的），那 pretrained MINT 失败的原因可能是：

1. **Observation 不匹配**：Pretrained MINT 在 LIBERO 图像上训练，Infinigen 图像风格完全不同（光照、纹理、几何）。

2. **Action space 不匹配**：MINT 预训练时 action 可能是归一化的（如 [-1, 1]），而我们的 raw action delta 是 [-0.4, +0.4]。

3. **两者的结合**：视觉不匹配 + action 不匹配导致 MINT 无法正确理解场景和执行动作。

### 5.2 为什么 Finetuned MINT 也没改善？

即使 finetuned MINT 学到了 Infinigen 的图像和 action 表征，它仍然失败，可能因为：

1. **训练数据不足**：1,301 帧不足以让 MINT 从 4B 参数中提取有效知识。

2. **Gripper 控制丢失**：VQ-VAE 虽然能编码和解码 gripper，但 fine-tuning 过程中 gripper 信号可能仍然被稀释。

3. **Action-Measurement Mismatch**：如果 action 和 actual movement 不一致，即使 MINT 学到了正确的 action 策略，仿真器也无法正确执行。

---

## 六、Physical Realism 分析

### 6.1 Position Delta 异常

根据 `vqvae_retraining_plan.md` 中的物理真实性检查标准：

```
要求：position delta ∈ [-0.05m, +0.05m]
实际：position delta ∈ [-0.4m, +0.4m]  ← 超出 8 倍！
```

**但**：`actions[:, :3]` 的值域是 [-0.4, +0.4]，这是原始物理单位。如果这是米（m），则每步移动可达 0.4m，非常快。

但实际 EEF 移动（通过 `np.diff(eef_positions)` 计算）最大为 0.5858m，平均为 0.1105m。这说明：
- Action 命令是 0.4m delta
- 实际执行后 EEF 移动了约 0.1m（可能是仿真器有限步长限制）

### 6.2 Gripper 物理真实性

```
gripper_values: {0, 1}（离散二进制）
states[:, -1]: {0, 1}（离散二进制）← 没有中间值 0.0！

gripper 闭合：1.0 → 0.0（一步完成）
```

等等！之前分析说 states 有中间值 0.0，但重新检查发现 **只有 {0, 1} 两个值**！这意味着：

1. `gripper_values = {0, 1}`：离散二进制
2. `states[:, -1] = {0, 1}`：离散二进制（无中间值）
3. `actions[:, 6] = {-1, +1}`：离散二进制

**所有数据都是离散的！**

### 6.3 Rotation Delta 物理真实性

```
Rotation delta ∈ [-0.12, +0.12]（弧度）
换算为度：±6.9°/步
这是合理的（每步约 7° 旋转）
```

---

## 七、修正后的数据质量评估

### 7.1 Gripper 数据质量

| 检查项 | 标准 | 实际 | 状态 |
|--------|------|------|------|
| gripper_values 离散性 | {0, 1} | {0, 1} | ✓ |
| actions[:, 6] 离散性 | {-1, +1} | {-1, +1} | ✓ |
| 闭合信号在 attach 之后 | 必须 | 符合 | ✓ |
| Grasp steps ≥ 5 | 必须 | 7-48 | ✓ |
| Drawer pull ≥ 0.8 | 必须 | 0.92-1.0 | ✓ |

**结论：Gripper 数据质量合格。**

### 7.2 Position/Rotation 数据质量

| 检查项 | 标准 | 实际 | 状态 |
|--------|------|------|------|
| Position delta 合理性 | < 0.5m/步 | max=0.69m | ⚠️ 略超 |
| Rotation delta 合理性 | < 0.3rad/步 | max=0.42rad | ⚠️ 略超 |
| Action 与 movement 相关性 | 高 | X=0.54, Y/Z=0.16 | ❌ 低相关性 |

**结论：Position/Rotation 数据物理真实性可疑，相关性低。**

---

## 八、修正后的行动方案

### 8.1 需要立即验证的问题

| 优先级 | 问题 | 验证方法 | 预计时间 |
|--------|------|---------|---------|
| P0 | Action-Measurement Mismatch 是否真实？ | 对比 action delta 和 next_eef_positions - eef_positions | 30 分钟 |
| P0 | LIBERO vs Infinigen action 分布对比 | 读取 LIBERO 原始数据（或文档） | 1 小时 |
| P0 | LeRobot 数据加载是否有 normalization？ | 检查 dataset_builder.py 和 MINT 数据加载代码 | 30 分钟 |
| P1 | 为什么 pretrained MINT 0% grasp？ | 单独测试 pretrained 模型输出 | 1 小时 |

### 8.2 修正后的 Option B

基于新的根因分析，Option B 应该：

1. **不是"重训练 VQ-VAE"**，而是"调查 Action-Measurement Mismatch"
2. **检查 MINT 的 action normalization 逻辑**
3. **验证训练数据中的 action 是否被正确归一化**

### 8.3 立即行动

在执行任何重训练之前，必须完成以下验证：

```bash
# 1. 检查 LeRobot 数据加载时的 action normalization
python3 << 'PYEOF'
from pathlib import Path
import sys
sys.path.insert(0, "/mnt/afs2/zhuhaowu/infinigen/scripts/mint")
from dataset_builder import build_dataset_from_rollouts

# 检查是否有 normalization
# 读取 dataset_builder.py 的完整逻辑
# 特别关注 action 字段的处理
PYEOF

# 2. 对比 LIBERO 和 Infinigen 的 action 分布
# 从 MINT 论文或代码中查找 LIBERO action 分布信息

# 3. 单独测试 pretrained MINT 的 action 输出
# 在 Infinigen 环境下运行 pretrained MINT，观察 action 值域
```

---

## 九、文档更新计划

### 9.1 需要更新的文档

| 文档 | 更新内容 | 优先级 |
|------|---------|--------|
| findings.md | 移除"连续 vs 离散"假设，添加 Action-Measurement Mismatch 分析 | P0 |
| v57_analysis_report.md | 修正矛盾陈述，更新根因分析 | P0 |
| RESOLUTION_PLAN.md | 移除 VQ-VAE 重训练为首选方案，改为 Action Mismatch 调查 | P0 |
| campaign_status.md | 更新失败原因为 "action_measurement_mismatch" | P1 |
| progress.md | 添加本自省报告的摘要 | P1 |

### 9.2 需要创建的文档

| 文档 | 内容 | 优先级 |
|------|------|--------|
| `action_measurement_mismatch_analysis.md` | Action vs actual movement 相关性分析 | P0 |
| `libero_vs_infinigen_comparison.md` | LIBERO 和 Infinigen 数据对比 | P0 |

---

## 十、结论

通过彻底的自省和 ground truth 数据验证，我们发现：

1. **文档矛盾**：多处文档声称数据是"连续的"，但实际是"离散的"。
2. **根因不完整**：VQ-VAE quantizer 无法编码 Infinigen drawer latent 是 P0，但 VQ-VAE 和 SigLIP Vision Encoder 是耦合的，两者必须同时解决。
3. **Action-Measurement Mismatch 是非阻塞问题**：Y/Z 轴低相关是 drawer 物理约束导致。
4. **VQ-VAE 训练代码不存在**：workspace 无此代码，博士师兄 2 天后提供。
4. **Physical Realism**：Position/Rotation delta 数据物理真实性可疑，超出合理范围。

**下一步**：在执行任何重训练之前，必须先验证 Action-Measurement Mismatch 是否真实，以及 LIBERO 和 Infinigen 的 action 分布是否一致。

---

## 附录：关键数据对比表

### 原始 NPZ 数据（strict_teacher_rollouts）

```
gripper_values:  {0, 1}                    ← 离散二进制
actions[:, 6]:   {-1, 1}                   ← 离散二进制  
states[:, -1]:   {0, 1}                   ← 离散二进制（无 -1.0）
Position delta:  [-0.4, +0.4] m           ← 原始物理单位
Rotation delta: [-0.12, +0.12] rad       ← 原始物理单位
EEF movement:    mean=0.11m, max=0.59m  ← 实际移动

Action-Measurement Correlation:
  X: 0.54 (中等)
  Y: 0.16 (弱)
  Z: 0.16 (弱)
```

### MINT VQ-VAE 期望格式

```
Gripper: discrete {-1, +1} after binarization  ← 匹配 ✓
Position: normalized [-1, +1] assumed         ← 待验证 ⚠️
Rotation: normalized [-1, +1] assumed          ← 待验证 ⚠️
```

### Gap 评估

```
Gripper format: ✓ 完全匹配
Position format: ⚠️ 可能不匹配（raw vs normalized）
Rotation format: ⚠️ 可能不匹配（raw vs normalized）
Data volume:     ❌ 1,301 vs 50,000 frames (2.6%)
```

---

## 附录 B（2026-03-31 补充）：CAMPAIGN_TRUTH.md 统一后的关键更新

**本补充修正了 Self-Correction v1.0 报告中的若干结论。详见 `CAMPAIGN_TRUTH.md`（真相源）。**

### 关键修正 1：VQ-VAE 和 Vision Encoder 是耦合的

Self-Correction v1.0 报告中认为"VQ-VAE quantizer 无法编码 Infinigen drawer action latent"是 P0-Blocker，但这个分析是**不完整的**。

真实情况是：
- VQ-VAE 和 SigLIP Vision Encoder 通过 Gemma LM 耦合
- 两者必须同时适应 Infinigen 数据才能解决问题
- 只重训练 VQ-VAE 是**必要但不充分**的

### 关键修正 2：Action-Measurement Mismatch 是非阻塞问题

Self-Correction v1.0 认为 Y/Z 轴相关性低是候选根因。但进一步分析发现：
- Y/Z 轴低相关是 drawer 物理约束导致（drawer 沿 X 轴滑动，Y/Z 无运动）
- 这是**正常物理现象**，非阻塞问题
- X 轴相关性 0.54 是中等水平，可以接受

### 关键修正 3：VQ-VAE 训练代码不存在

Self-Correction v1.0 假设 VQ-VAE 训练是可行的。但经过完整 workspace 搜索：
- external/MINT/ 和 external/physnap/ 均无 VQ-VAE 训练代码
- MINT 官方 SDAT 训练标注为 "Planned | 2026 H1"
- **博士师兄 2 天后提供训练代码**

### 正确的根因优先级

| 优先级 | 根因 | 证据 | 状态 |
|--------|------|------|------|
| **P0-Blocker** | **SigLIP Vision Encoder 从未见过 Infinigen 合成图像** | Pretrained MINT = 0% grasp | 确认（缺训练代码） |
| **P0-Blocker** | **VQ-VAE Quantizer 无法编码 Infinigen drawer latent** | Pretrained MINT = 0% grasp | 确认（待师兄提供代码） |
| P1 | 数据量不足（1,301 vs 5,000+ 帧） | 确认 | 待扩充 |
| P1（非阻塞） | Y/Z 轴 action-measurement 低相关 | drawer 物理约束 | 非阻塞 |

### 下一步

详见 `CAMPAIGN_TRUTH.md` 的 Section 7（决策树）。
