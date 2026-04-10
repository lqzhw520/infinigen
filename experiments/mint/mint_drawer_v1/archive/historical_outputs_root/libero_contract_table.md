# LIBERO vs Infinigen 数据合同对照表

**生成时间**: 2026-04-03
**来源**: 从 `external/LIBERO/datasets/libero/meta/` + 100 episodes parquet 样本实测
**指纹文件**: [libero_fingerprint.json](libero_fingerprint.json)
**用途**: Phase A — 参考指纹提取；为所有后续 P1a/P1b/P0 工程提供 ground-truth 对齐基准

---

## 1. Feature Schema 对照

| 维度 | LIBERO (ground truth) | Infinigen (当前实现) | 一致性 | 修复优先级 |
|------|----------------------|---------------------|--------|-----------|
| **主图像 key** | `image` (扁平 key) | `observation.images.image` (LeRobot 嵌套) | ⚠️ **不一致** | **P1a 修复** |
| **腕部图像 key** | `wrist_image` (扁平) | `observation.images.image2` (嵌套) | ⚠️ **不一致** | **P1a 修复** |
| **State key** | `state` (扁平) | `observation.state` (LeRobot 嵌套) | ⚠️ **不一致** | **P1a 修复** |
| **Action key** | `actions` (扁平, 注意复数) | `action` (单数) | ⚠️ **不一致** | **P1a 修复** |
| **State dtype** | `float32` | `float32` | ✅ 一致 | — |
| **Action dtype** | `float32` | `float32` | ✅ 一致 | — |
| **图像 dtype** | `image` (RGB uint8 stored as numpy array) | `uint8` | ✅ 一致 | — |

> **关键发现**：LeRobotDataset 的 **key 约定是扁平的** (`image`/`wrist_image`/`state`/`actions`)，但 MINT 的 `processor_mint.py` 中的 `NormalizerProcessorStep` 期望 **LeRobot 嵌套格式** (`observation.images.*`)。当前 `dataset_builder.py` 使用嵌套格式写入。需验证 LeRobotDataset 加载时的 rename_map 是否做了扁平→嵌套转换，否则 MINT 训练会报 key 不存在。

---

## 2. State 向量语义（8 维）

### LIBERO ground truth（实测）

```
state[0]: eef_pos_x (m)       — EEF 世界位置，单位米
state[1]: eef_pos_y (m)
state[2]: eef_pos_z (m)
state[3]: motor_joint[0] (m) — Panda motor position，原值 ≈ [1.1, 3.7]
state[4]: motor_joint[1] (m) — 原值范围 [-3.6, 3.6]
state[5]: motor_joint[2] (m) — 原值范围 [-1.7, 1.0]
state[6]: motor_joint[3] (m) — 原值范围 [-0.001, 0.041] ← ⚠️ 与 state[7] 范围重叠！
state[7]: gripper_joint_position (m) — 连续浮点，**负值=闭合，正值=张开**
```

> **重要修正（实测）**：`state[3:7]` 的行范数为 **3.27**（不是 1.0），说明 LIBERO 存储的是 **Panda 电机位置（motor joint values）**，不是 EEF 四元数。LeRobot 的 `NormalizerProcessorStep` 使用 **quantiles 模式**，将所有 state 维度的 q01/q99 映射到 [-1, 1]，因此电机位置和 EEF 位置被统一归一化。**Infinigen 当前写入的是 eef_quat（PyBullet link state），与 LIBERO motor positions 是完全不同的物理量**，在 quantile normalization 后可能数值上对齐，但物理意义不同。

> **图像存储**：LIBERO parquet 中 `image` 列存储的是 PNG bytes（`{bytes, path}`），由 LeRobotDataset 在加载时解码为 numpy 数组。

### 关键数值

| 维度 | LIBERO min | LIBERO max | LIBERO mean | Infinigen 当前 | 一致性 |
|------|-----------|-----------|------------|--------------|--------|
| **state[0]** x | -0.464 | +0.210 | -0.047 | ~[-0.55, +0.55] EEF 范围 | ⚠️ 坐标系不同 |
| **state[1]** y | -0.326 | +0.391 | +0.034 | ~[-0.55, +0.55] | ⚠️ 坐标系不同 |
| **state[2]** z | +0.446 | +1.326 | +0.765 | ~[0.05, 0.70] | ⚠️ 桌面高度不同 |
| **state[7]** gripper | **-0.042** | **+0.001** | -0.029 | **0.0 或 1.0 (二进制！)** | ❌ **P0 critical** |

### ❌ P0 Critical: state[7] 必须修复

**LIBERO 真实数据**：
```
state[7] ∈ [-0.042, +0.001]    ← 连续浮点
state[7] unique count: 410     ← 连续分布，不是 {0, 1}
state[7] mean: -0.029           ← 整体偏负（多数时间夹爪闭合）
state[7] convention: NEGATIVE = CLOSED, POSITIVE = OPEN
```

**额外关键发现 — state[3:7] 不是四元数**：
```
state[3:7] 行范数 ≈ 3.27（不是 1.0）
state[3]: ≈ [1.1, 3.7]  range
state[4]: ≈ [-3.6, 3.6] range
state[5]: ≈ [-1.7, 1.0] range
state[6]: ≈ [-0.001, 0.041] ← 与 state[7] 范围重叠！

结论：state[3:7] = Panda 电机位置（motor_joint_values），不是 EEF 四元数。
LIBERO 的 LeRobot normalizer 用 quantiles 模式将所有维度（包括电机值）
统一归一化到 [-1, 1]，因此电机值和 EEF 位置被混合归一化。

Infinigen 当前用 eef_quat（PyBullet link state，单位四元数）写入 state[3:7]。
修复后有两条路：
  方案 A：改为连续电机位置（与 LIBERO 语义一致，但需要从 PyBullet joint 获取）
  方案 B：保持 eef_quat，但需在 meta/stats.json 中单独提供 stats，
          让 LeRobot normalizer 单独对 quat 归一化（不与电机值混用）
```

**Infinigen 当前实现**（[drawer_robot_env.py:385](scripts/mint/drawer_robot_env.py)）：
```python
# WRONG: binary {0, 1} for state[7]
state = np.concatenate([eef_pos, eef_quat, np.array([self.gripper_open])])
# gripper_open ∈ {0.0, 1.0}  ← 错误！
```

**正确实现**（来自 [handoff.md:53-57](experiments/mint/mint_drawer_v1/sovereign/handoff.md)）：
```python
# CORRECT: continuous joint position (meters), NEGATIVE=closed
finger_pos = np.mean([
    self.p.getJointState(self.robot_id, 9)[0],   # panda_finger_joint1
    self.p.getJointState(self.robot_id, 10)[0],  # panda_finger_joint2
])
state = np.concatenate([eef_pos, eef_quat, np.array([-finger_pos])])  # NEGATE
# 结果: closed ≈ -0.038, open ≈ +0.038
```

**修复验证指标**：
- [ ] `state[7]` unique count > 100（连续，非二进制）
- [ ] `state[7]` range 包含负值（closed）和接近零的正值（open）
- [ ] `state[7]` 与 gripper joint 实际位置线性相关

---

## 3. Action 向量语义（7 维）

### LIBERO ground truth

```
action[0-2]: delta_xyz — 归一化到 [-1, 1]，实际物理位移 = action[:3] * translation_scale
action[3-5]: delta_rxyz — rotvec 旋转增量，归一化到 [-1, 1]
action[6]:   gripper_command — {-1.0 = CLOSE, +1.0 = OPEN} 离散二进制
```

### 关键数值

| 维度 | LIBERO min | LIBERO max | 实际 cap 范围 | Infinigen 当前 | 一致性 |
|------|-----------|-----------|-------------|--------------|--------|
| **action[0-2]** 平移 | -0.938 | +0.938 | **0.938** (不是 1.0！) | [-1.0, +1.0] clip | ⚠️ 人 teleop 从不用满 |
| **action[3-5]** 旋转 | -0.375 | +0.375 | **0.375** (不是 1.0！) | [-1.0, +1.0] clip | ⚠️ 人 teleop 从不用满 |
| **action[6]** 夹爪 | **-1.0** 或 **+1.0** | 二进制！ | {-1, +1} 仅两值 | **✓ 正确**（已有） | ✅ 一致 |

> **关键发现**：LIBERO 人类 teleop 数据中，translation action 从不超过 0.938，rotation action 从不超过 0.375。这说明 **Infinigen 允许 clip 到 [-1, 1] 是合理的宽松上界**，但实际 teacher 运动比这个范围平滑得多。`action_delta_l2` p95 = 0.161 意味着每步的 |Δa|_2 ≈ 0.16，远小于 clip 值 1.0。

### Action 约定验证

**LIBERO action → state 时序关系（实测）**：

| 维度 | Best lag | Pearson r | 含义 |
|------|---------|----------|------|
| action[0] vs delta_state_x | **lag=1** | **0.972** | action[t] 预测 delta_state[t+1] ✅ |
| action[1] vs delta_state_y | **lag=1** | **0.995** | ✅ |
| action[2] vs delta_state_z | lag=0 | 0.992 | Z 方向同步生效 ✅ |
| action[3] vs delta_state_rx | lag=1 | 0.950 | ✅ |
| action[6] vs gripper | lag=0 | -0.410 | 夹爪即时切换 ✅ |

> **Infinigen 当前**：`_action_from_transition()` 计算 `next_obs.eef_pos - current_obs.eef_pos`，即 `action[t]` 确实表示 `Δ_pos[t→t+1]`。**时序约定一致**。

---

## 4. 图像规格

| 属性 | LIBERO | Infinigen 当前 | 一致性 | 修复 |
|------|--------|--------------|--------|------|
| **分辨率** | **256×256** | 224×224 | ⚠️ 不一致 | **P1b** |
| **通道** | RGB 3 | RGB 3 | ✅ | — |
| **dtype** | uint8 | uint8 | ✅ | — |
| **主相机 key** | `image` | `observation.images.image` | ⚠️ LeRobot key 不匹配 | **P1a** |
| **腕部相机 key** | `wrist_image` | `observation.images.image2` | ⚠️ LeRobot key 不匹配 | **P1a** |
| **渲染引擎** | robosuite/MuJoCo | PyBullet `ER_TINY_RENDERER` | ❌ 渲染风格差异 | **P1c** |

### P1b: 分辨率和相机参数

LIBERO 使用 `camera_heights=128, camera_widths=128`，但 `meta/info.json` 中 `shape: [256, 256, 3]`。可能存在 resize。需验证：

```bash
# 检查 LIBERO 原始 parquet 中 image 的实际 shape
python3 -c "
import pandas as pd, numpy as np
df = pd.read_parquet('external/LIBERO/datasets/libero/data/chunk-000/episode_000000.parquet')
img = df['image'].iloc[0]
print(f'Type: {type(img)}')
print(f'Shape: {np.array(img).shape}')
"
```

### P1c: 渲染风格（根本性差异）

| 维度 | LIBERO | Infinigen PyBullet |
|------|--------|-------------------|
| 光照 | robosuite/MuJoCo 光照模型 | PyBullet 简化光照 |
| 材质 | 真实感 PBR | 基础 Lambertian |
| 阴影 | 有 | 无 |
| 纹理 | 照片级 | 程序化简化 |
| **图像风格** | **接近真实感** | **简化色块** |

> 这是 SigLIP P0 blocker 的核心原因之一。**换渲染引擎（MuJoCo EGL）可缓解但不能消除**。

---

## 5. 相机配置（LIBERO 参考值）

### LIBERO 相机定义

从 `external/LIBERO/libero/libero/envs/env_wrapper.py` 读取的标准配置：

```python
camera_names = ["agentview", "robot0_eye_in_hand"]
camera_heights = 128   # 原始分辨率
camera_widths = 128
camera_depths = False
```

### AgentView 相机参数（world-fixed overhead）

```python
# 位置（world frame）
camera_pos = np.array([0.5886131746834771, 0.0, 1.4903500240372423])
# 四元数 (w, x, y, z)
camera_quat = np.array([0.6380177736282349, 0.3048497438430786,
                         0.30484986305236816, 0.6380177736282349])
```

### Eye-in-Hand 相机

- 安装在 Panda 末端夹爪上
- 随 EEF 运动而运动
- 物理标定的内外参

### Infinigen 当前相机

```python
# Primary camera (world-fixed, angled side-front)
CAMERA_PRIMARY = {
    "target": np.array([0.18, 0.0, 0.40]),  # drawer center
    "distance": 1.0,
    "yaw": 165.0,       # ← LIBERO agentview 是 overhead，无 yaw 强调
    "pitch": -32.0,      # ← LIBERO 是纯 overhead，pitch ≈ -90
    "roll": 0.0,
}

# Wrist camera (EEF-parented, PyBullet computeViewMatrix)
CAMERA_SECONDARY_OFFSET = np.array([0.08, 0.0, 0.02])  # EEF frame offset
```

**P1b 相机修复方向**：
1. Primary camera → 改为 overhead 配置（类似 LIBERO agentview）
2. 或保留 angled view 但明确这是 sim-specific 的，不强求与 LIBERO 一致
3. 腕部相机标定方式需要更物理精确（建议直接用 LIBERO robosuite 的 Panda gripper camera model）

---

## 6. 数据集规格

| 属性 | LIBERO | Infinigen 当前 | 一致性 |
|------|--------|--------------|--------|
| **fps** | 10 | 10 (max_steps=48 → ~5fps，实际取决于 env.step()) | ⚠️ 不确定 |
| **Episode 长度** | mean=277, std=58 | 90-102 帧（当前 rollouts） | ⚠️ 短于 LIBERO |
| **总 episodes** | 1693 | 15 (c2_replay_valid_rollouts) | ❌ 数据量不足 |
| **总帧数** | 273,465 | ~1,301 | ❌ 需扩充到 5,000+ |
| **任务数** | 40 | 1 ("open the drawer") | ⚠️ 单一任务 |

---

## 7. 修复优先级总结

| 优先级 | 问题 | 修复目标 | 文件 |
|--------|------|---------|------|
| **P0-Critical** | state[7] = 二进制 {0,1} | 连续 joint position [-0.042, +0.001], negative=closed | `drawer_robot_env.py` `_state_vector()` |
| **P1a-High** | feature key 不匹配 | LeRobot 格式统一（`image`/`wrist_image`/`state`/`actions` 或确认 rename_map 覆盖） | `dataset_builder.py` |
| **P1a-High** | action clip 与 LIBERO 实际分布差异 | action scale 已合理（p95=0.16 << 1.0）；无需改 clip 值，但需确认 translation_scale=0.03m 是否与 LIBERO 0.03m/unit 一致 | `mint_common.py` |
| **P1b-Medium** | 分辨率 224×224 vs 256×256 | 改为 256×256 或在 LeRobot 写入前 resize | `drawer_robot_env.py` + `dataset_builder.py` |
| **P1b-Medium** | 相机视角 overhead vs angled | 评估是否需要与 LIBERO 一致的 overhead view | `drawer_robot_env.py` |
| **P1c-Long-term** | 渲染风格 PyBullet vs robosuite | 迁移到 robosuite + MuJoCo + EGL | mint 侧新 env |
| **P2** | 数据量不足 | 扩充到 5,000+ frames | 数据生成管线 |

---

## 8. 立即可执行的验证脚本

```bash
# 验证 state[7] 是否为连续浮点（非二进制）
source /root/anaconda3/etc/profile.d/conda.sh && conda activate infinigen
python3 - << 'EOF'
import numpy as np, json
from scripts.mint.drawer_robot_env import DrawerRobotEnv

env = DrawerRobotEnv(seed=2)
obs = env.reset()
frames_state7 = []
for _ in range(100):
    action = np.zeros(7); action[6] = -1.0
    obs, _, _, _ = env.step(action)
    frames_state7.append(obs.state[7])
    if obs.state[7] < 0.02:  # 已闭合
        action[6] = 1.0  # 张开
        obs, _, _, _ = env.step(action)
        frames_state7.append(obs.state[7])

vals = np.array(frames_state7)
print(f"state[7] unique count: {len(np.unique(np.round(vals, 4)))}")
print(f"state[7] range: [{vals.min():.4f}, {vals.max():.4f}]")
print(f"Is binary {0,1}? {set(np.unique(np.round(vals))) == {0.0, 1.0}}")
print(f"Expected: unique > 100, range includes negative values")
EOF
```

---

## 附录：LIBERO LeRobotDataset 加载验证

```bash
source /root/anaconda3/etc/profile.d/conda.sh && conda activate mint
python3 - << 'EOF'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

libero = LeRobotDataset(
    repo_id="dummy",  # 使用本地 root
    root="/mnt/afs2/zhuhaowu/infinigen/external/LIBERO/datasets/libero",
    revision="main"
)
print(f"Features: {list(libero.features.keys())}")
print(f"Episode 0 length: {libero.episode_lengths[0]}")
sample = libero[0]
print(f"Sample keys: {list(sample.keys())}")
EOF
```
