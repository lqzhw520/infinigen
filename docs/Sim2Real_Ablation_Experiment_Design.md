# Sim2Real Ablation Experiment Design（Phase 3.1）

本文档定义 **Sim2Real 消融实验** 的可复现方案：从数据引擎（Phase-1）到感知模型（Phase-2），再到规划/控制（Phase-3.2）的端到端评估，重点回答：

- **数据量**、**物理对齐强度**、**域随机化（DR）强度** 对性能与稳定性的影响有多大？
- 哪些因素对 **Sim→Real transfer gap** 最关键？
- 实验如何做到 **统计显著** 与 **可复现**？

---

## 1. 总体实验闭环（Data → Model → Planner → Metrics）

**训练数据（Sim）**

- 由 `scripts/export_mailerbox_simple_phase1_data_engine.py` 生成：
  - RGB / Depth
  - link/instance segmentation
  - joint keypoints（轴线端点）
  - camera intrinsics/extrinsics
  - URDF GT（拓扑/关节/limit）
  - metadata（material physics + manifest）

**模型（Perception）**

- Topo-Box-Net（见 `docs/Topo_Box_Net_Architecture.md`）输出：
  - box_type、joint_axis/origin/state 等

**规划（Planning）**

- Phase 3.2：VAMP 规划器（或兼容接口的 planner）基于：
  - URDF（GT 或 Perception reconstructed）
  - 初始 joint state（来自观测或估计）
  - goal spec（例如“打开盖子到 90°”）

**指标（Metrics）**

- Perception：box type accuracy、joint axis/origin/state error
- Planning：成功率、碰撞率、路径长度、耗时
- Sim2Real：真实数据集上的同类指标（或 domain holdout）

---

## 2. 消融维度（因素设计）

我们将因素分为三组：**数据量**、**物理对齐**、**DR 强度**。建议优先做 **2×2×K** 的 factorial 设计，再扩展更细网格。

### 2.1 数据量（Data Quantity）

建议使用“seed 数量 × joint states × views”的等效样本数定义：

\[
N_{samples} = N_{seeds}\cdot N_{jointstates}\cdot N_{views}
\]

推荐水平：

- **Small**：\(N_{samples}\approx 1\text{k}\)
- **Medium**：\(10\text{k}\)
- **Large**：\(100\text{k}\)（若算力允许）

控制变量：

- box_type 固定（单盒型）→ 再做多盒型混合
- image resolution 固定（例如 256×256 或 320×240）

### 2.2 物理对齐强度（Physical Fidelity / Alignment）

核心思想：将“物理一致性”拆成可控开关，并观察对 Sim2Real 的贡献。

建议至少包含：

- **Inertia fidelity**
  - **Aligned**：使用 thin-shell inertia + 平行轴定理合并（当前实现）
  - **Degraded**：人为扰动密度或惯性（例如 density×(0.5, 2.0) 或加噪），用于评估物理对齐的必要性

- **Collision fidelity**
  - collision mesh：开启（visual_only=False 导出） vs 关闭（visual_only=True）
  - self-collision：planner/simulator 侧开启 vs 关闭（PyBullet flags）

- **Joint constraint fidelity**
  - 正确 joint limits（URDF 中 lower/upper）
  - 放宽 limits（upper-lower 放大）用于测评“约束误差”对 planner 稳定性的影响

> 注意：对 perception-only 指标，某些物理项影响较弱；但对 planner 成功率/碰撞率与 Sim2Real gap 通常更敏感。

### 2.3 DR 强度（Domain Randomization Intensity）

建议把 DR 分为三个“可控子空间”：

- **Material DR**：颜色/roughness/specular/normal/bump；以及物理密度（但密度属于物理对齐，需要单独控制）
- **Lighting DR**：HDRI/三点光/曝光/阴影强度（若管线支持）
- **Camera DR**：距离/俯仰/方位/roll、焦距扰动、裁剪平面

推荐水平（示例）：

- **DR-0**：固定材质 + 固定相机分布
- **DR-1（mild）**：颜色/粗糙度小范围扰动，相机小扰动
- **DR-2（strong）**：多材质类型 + 大范围扰动 + 多曝光/多角度

---

## 3. 指标定义（Metrics）

### 3.1 Perception（Topo-Box-Net）

- **BoxType Accuracy**：Top-1 / Top-k
- **Joint axis error**：
  - 方向误差：\(\arccos(\langle \hat a, a\rangle)\)
- **Joint origin error**：
  - \(\|\hat o - o\|_2\)
- **Joint state error**：
  - hinge：\(|\hat\theta-\theta|\)
  - prismatic：\(|\hat d - d|\)
- **Keypoint reprojection error（可选）**
  - 以 `keypoints.json` 为 GT，对预测轴端点投影后的像素误差

### 3.2 Planner / Control

- **Success rate**（达到目标且无碰撞）
- **Collision rate**（自碰撞/与环境碰撞）
- **Path length**（关节空间 \(\sum\|\Delta q\|\) 或时间积分）
- **Planning time**（wall-clock）
- **Energy / smoothness（可选）**
  - 例如 \(\sum\|\Delta^2 q\|\)

### 3.3 Sim2Real Gap

- 对每个 metric 定义：
  - gap = metric_real − metric_sim（或 ratio）
- 对分类指标可用：
  - gap = acc_sim − acc_real

---

## 4. 统计检验与实验规范（严谨性）

### 4.1 重复次数（seeds）

至少三重随机源：

- 数据生成 seed（asset seed / view seed / DR seed）
- 模型训练 seed（初始化、shuffle）
- 评估 seed（planner 随机采样等）

建议：

- 每个配置 **≥ 3 个训练 seed**
- 对成功率/碰撞率用 **bootstrap 置信区间** 或 Wilson interval

### 4.2 显著性检验

- **2×2×K factorial**：
  - 两因素（physical × DR）可用 two-way ANOVA（连续指标）
  - success rate（比例）可用 logistic regression 或 Fisher exact test（配对/非配对）
- 多重比较：Holm–Bonferroni 或 Benjamini–Hochberg

### 4.3 复现实验记录（必须输出）

每个 run 输出 `run.json`，至少包含：

- dataset manifest path + seeds list
- DR/physics 配置（参数范围或 config 文件）
- model config（网络/损失权重/训练超参）
- eval config（任务定义、planner 参数）
- 所有随机 seed

---

## 5. 可复现脚本结构（推荐）

建议用“单一入口 + JSON 配置”来组织实验（不强制依赖 hydra）。

- `scripts/run_sim2real_ablation_matrix.py`
  - 生成消融配置网格（JSON）
  - 为每个 run 生成可执行命令（数据生成 / 训练 / 评估）

每个 run 建议落盘到：

```
sim_exports/experiments/sim2real/<run_id>/
  run.json
  dataset/         # 可选：生成数据在此
  train_logs/
  eval_logs/
  metrics.json
```

---

## 6. 推荐的最小消融矩阵（可直接开始跑）

以 MAILER + DRAWER 为例：

- 数据量：Small / Medium
- 物理对齐：Aligned / Degraded（密度×2 + inertia noise）
- DR：DR-0 / DR-2

共 \(2\times2\times2=8\) 组，每组 3 个训练 seed → 24 次训练（可分批并行）。

