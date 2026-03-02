# Infinigen-AnyBox: Phase Architecture Overview

**Date**: 2026-02-26  
**Version**: v1.0

---

## Architecture Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DATA ENGINE                                │
│                                                                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │ Phase1.1 │──▶│ Phase1.2 │──▶│  Phase 1.3   │──▶│   Phase 1.4    │  │
│  │ 盒型拓扑  │   │ 惯性修正  │   │ 域随机化材质  │   │  自动标注管线   │  │
│  │ 扩展     │   │ 推广      │   │   系统       │   │               │  │
│  └──────────┘   └──────────┘   └──────────────┘   └───────┬────────┘  │
│       │              │               │                    │           │
│       ▼              ▼               ▼                    ▼           │
│  ┌─────────┐   ┌──────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │ URDF    │   │ Verified │   │  Material    │   │   Dataset/     │  │
│  │ per box │   │ Inertia  │   │  Configs     │   │   train/       │  │
│  │ type    │   │ + PyBullet│   │  (metadata)  │   │   000001/      │  │
│  └─────────┘   └──────────┘   └──────────────┘   │   ├─ rgb.png   │  │
│                                                   │   ├─ depth.npy │  │
│                                                   │   ├─ seg.*     │  │
│                                                   │   ├─ ...       │  │
│                                                   │   └─ urdf_gt   │  │
│                                                   └────────────────┘  │
└───────────────────────────────────────┬─────────────────────────────────┘
                                        │ dataset/
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: PERCEPTION MODEL                           │
│                                                                        │
│  ┌────────────────────┐         ┌──────────────────────┐               │
│  │    Phase 2.1       │         │     Phase 2.2        │               │
│  │  Topo-Box-Net      │────────▶│ Kinematic Consistency │               │
│  │  架构设计 + 训练    │         │     Loss 数学形式化   │               │
│  └────────┬───────────┘         └──────────────────────┘               │
│           │                                                            │
│           ▼                                                            │
│  ┌──────────────────┐                                                  │
│  │  Trained Model   │                                                  │
│  │  (box type,      │                                                  │
│  │   joint state,   │                                                  │
│  │   joint params)  │                                                  │
│  └────────┬─────────┘                                                  │
└───────────┼────────────────────────────────────────────────────────────┘
            │ predictions
            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 3: SIM2REAL VALIDATION                        │
│                                                                        │
│  ┌────────────────────┐         ┌──────────────────────┐               │
│  │    Phase 3.1       │         │     Phase 3.2        │               │
│  │  Sim2Real 消融      │────────▶│ Motion Planner 集成   │               │
│  │  实验设计 + 执行    │         │  (VAMP / baseline)   │               │
│  └────────────────────┘         └──────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Engine（数据引擎）

### Phase 1.1 — 盒型拓扑扩展

| 项目 | 内容 |
|------|------|
| **目标** | 为每种包装盒拓扑实现 Blender 几何生成 + URDF 导出 |
| **输入** | 盒型参数（width/depth/height/thickness + 关节配置） |
| **输出** | `sim_exports/urdf/<box_type>/<seed>/<box_type>.urdf` + `assets/*.obj` |
| **已完成盒型** | MAILER (2-DOF hinge), TUCK_END (4-DOF hinge), DRAWER (1-DOF prismatic), SLIP_LID (1-DOF prismatic) |
| **验证方式** | PyBullet `loadURDF` with `URDF_USE_INERTIA_FROM_FILE` + `URDF_USE_SELF_COLLISION` |
| **关键代码** | `infinigen/assets/sim_objects/modular_box_factory.py` |

### Phase 1.2 — 惯性修正推广

| 项目 | 内容 |
|------|------|
| **目标** | 对所有盒型的 URDF，确保每个 link 只有一个 `<inertial>`，且由平行轴定理正确合并 |
| **输入** | Phase 1.1 输出的 URDF |
| **输出** | 惯性正确的 URDF（同路径，覆盖） |
| **数学** | $M=\sum m_i$，$C=\frac{1}{M}\sum m_i c_i$，$I=\sum[I_i + m_i((r_i \cdot r_i)\mathbf{E} - r_i \otimes r_i)]$ |
| **验证方式** | `python scripts/verify_urdf_inertia_fix.py --urdf-dir <path> --quiet` |
| **关键代码** | `infinigen/core/sim/physics/thin_shell_inertia.py`, `infinigen/core/sim/exporters/urdf_exporter.py` |

### Phase 1.3 — 域随机化材质系统

| 项目 | 内容 |
|------|------|
| **目标** | 确定性采样视觉 + 物理材质参数，耦合到 URDF 质量/惯性 |
| **输入** | `seed` + `allowed_families` 配置 |
| **输出** | `BoxMaterialConfig`（density/friction/restitution/color/roughness/bump 等）→ 写入 Blender 对象 `physics_density` + URDF |
| **材质族** | corrugated (A/B/C/E)、kraft、glossy (白卡覆膜)、plastic (PP/PET/HDPE) |
| **确定性** | 给定 `(seed, material_seed)` 完全可复现（本地 RNG，不用全局 `np.random`） |
| **验证方式** | 检查 `metadata.json` 中 `material.density` 在物理合理范围内 |
| **关键代码** | `infinigen/assets/sim_objects/box_material_domain_randomization.py` |

### Phase 1.4 — 自动标注管线

| 项目 | 内容 |
|------|------|
| **目标** | 一键批量生成可训练的标注数据集 |
| **输入** | `--box_type` + `--seeds` + `--n_views` + `--joint_states` + `--width/height` |
| **输出（per sample folder）** | 见下表 |
| **验证方式** | `python scripts/verify_phase1_sample_folder.py <sample_dir>` |
| **关键代码** | `scripts/export_mailerbox_simple_phase1_data_engine.py` |

**Phase 1.4 输出 schema（每个 sample folder）：**

```
dataset/train/000001/
├── rgb.png                          # RGB 渲染 (W×H)
├── depth.npy                        # float32 深度图 (H×W), 单位 meters
├── segmentation.npy                 # int64 link-level 语义分割 (H×W)
├── segmentation.png                 # 可视化 (伪彩色)
├── segmentation_label_map.json      # label_id → link name/role/color 映射
├── instance.npy                     # int64 实例分割 (H×W)
├── instance.png                     # 可视化
├── keypoints.json                   # 关节轴 keypoints (world + camera + 2D)
├── joint_state.json                 # 当前关节角度/位移
├── camera_intrinsics.json           # 相机内参 (K, H, W)
├── urdf_gt.urdf                     # Ground Truth URDF 副本
├── metadata.json                    # 盒型/材质/尺寸/seed 等元信息
└── camview_*.npz                    # 完整 camera extrinsics + intrinsics
```

---

## Phase 2: Perception Model（感知模型）

### Phase 2.1 — Topo-Box-Net 架构设计与训练

| 项目 | 内容 |
|------|------|
| **目标** | 从单张 RGB-D 推断盒型拓扑、关节状态、关节参数 |
| **输入** | Phase 1.4 的 dataset（RGB-D + segmentation + URDF GT） |
| **输出** | 训练好的模型 checkpoint |
| **架构** | CNN encoder → multi-head: box_type (16-class), joint_state, joint_type, joint_axis, joint_origin |
| **当前状态** | 代码骨架完成（schema/parser/loader/model/losses），**无训练脚本** |
| **关键代码** | `infinigen/perception/topo_box_net/` |
| **文档** | `docs/Topo_Box_Net_Architecture.md` |

### Phase 2.2 — Kinematic Consistency Loss

| 项目 | 内容 |
|------|------|
| **目标** | 利用 URDF 父子关系约束预测的 link poses 满足 hinge/prismatic 约束 |
| **输入** | 模型预测的 link poses + joint parameters |
| **输出** | 可微损失函数 $\mathcal{L}_{kinematic}$ |
| **数学** | Rodrigues 旋转 + hinge/prismatic 约束残差（详见 `docs/Kinematic_Consistency_Loss.md`） |
| **当前状态** | NumPy + PyTorch 参考实现完成，6 个单元测试通过 |
| **关键代码** | `infinigen/perception/topo_box_net/kinematic_consistency.py` |

---

## Phase 3: Sim2Real Validation（仿真→真机验证）

### Phase 3.1 — Sim2Real 消融实验设计

| 项目 | 内容 |
|------|------|
| **目标** | 量化 Physics-Aligned 数据对感知/操作的影响 |
| **输入** | Phase 2.1 训练好的模型 + Phase 1 的不同配置数据集 |
| **输出** | 消融实验报告（含统计显著性） |
| **消融维度** | 数据量 (1k/10k/100k)、物理对齐程度 (with/without inertia)、DR 强度 (low/medium/high) |
| **指标** | mIoU (seg), ADD-S (pose), Topology Accuracy, Open/Close Success Rate |
| **当前状态** | 设计文档完成 + run matrix 脚本骨架，**无实际实验** |
| **关键代码** | `scripts/run_sim2real_ablation_matrix.py` |
| **文档** | `docs/Sim2Real_Ablation_Experiment_Design.md` |

### Phase 3.2 — Motion Planner 集成

| 项目 | 内容 |
|------|------|
| **目标** | 接入 VAMP 等规划器，实现感知→规划→操作闭环 |
| **输入** | Phase 2 推断的 URDF + joint state → PlanRequest |
| **输出** | PlanResult（轨迹 + collision flag） |
| **Baseline** | 线性 joint-space 插值 + PyBullet 自碰撞检查 |
| **当前状态** | API 定义 + baseline planner 可运行 + VAMP adapter stub |
| **关键代码** | `infinigen/planning/` |
| **文档** | `docs/VAMP_Motion_Planner_Integration.md` |

---

## Phase 间数据流

```
Phase 1.1 (URDF) ──┐
Phase 1.2 (惯性)  ──┤
Phase 1.3 (材质)  ──┤──▶ Phase 1.4 (标注管线) ──▶ dataset/train/
                    │                                    │
                    │                                    ▼
                    │                            Phase 2.1 (训练)
                    │                                    │
                    │                                    ▼
                    │                            Phase 2.2 (约束 loss)
                    │                                    │
                    │                                    ▼
                    │                          trained model checkpoint
                    │                                    │
                    ▼                                    ▼
              sim_exports/urdf/          Phase 3.1 (消融实验)
                    │                    Phase 3.2 (Planner 集成)
                    │                           │
                    └───────────────────────────┘
                          sim_exports/urdf/ 同时被
                          Planner (Phase 3.2) 直接使用
```

---

## 当前完成度总览

| Phase | Sub-phase | 状态 | 正式数据集输出 |
|-------|-----------|------|--------------|
| 1 | 1.1 盒型拓扑 | ✅ 代码完成 + URDF 已导出 | `sim_exports/urdf/` (4 盒型) |
| 1 | 1.2 惯性修正 | ✅ 完成 + 11 seeds 验证 | 内嵌于 URDF |
| 1 | 1.3 域随机化材质 | ✅ 代码完成 | 内嵌于管线 |
| 1 | 1.4 自动标注管线 | ✅ 代码完成 | **仅 smoke test（1-4 样本），无正式批量数据集** |
| 2 | 2.1 Topo-Box-Net | ⚠️ 骨架 only | 无 |
| 2 | 2.2 Kinematic Loss | ✅ 实现 + 6 单测 | 无 |
| 3 | 3.1 消融实验 | ⚠️ 设计 only | 无 |
| 3 | 3.2 Planner 集成 | ⚠️ baseline only | demo JSON |
