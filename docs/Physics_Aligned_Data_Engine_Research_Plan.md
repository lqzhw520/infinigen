# Physics-Aligned Procedural Data Engine for Generalizable Articulated Object Manipulation

**项目代号**: Infinigen-AnyBox  
**版本**: v0.1 (Research Proposal)  
**日期**: 2026-01-27  
**目标会议**: CVPR 2027 / ICRA 2027 / RSS 2027

---

## 1. Executive Summary

### 1.1 研究动机

具身智能（Embodied AI）的核心瓶颈之一是**高质量铰接物体数据的稀缺**。现有数据集（如 PartNet-Mobility）存在以下问题：

| 问题 | 影响 |
|------|------|
| 规模有限（~2000 个模型） | 无法支持大规模预训练 |
| 人工标注误差 | 关节轴向、范围不准确 |
| 缺乏物理属性 | 惯性、摩擦等参数缺失 |
| 拓扑单一 | 主要是门、抽屉等简单结构 |

### 1.2 核心创新

我们提出 **Physics-Aligned Procedural Data Engine**，其创新点为：

1. **Infinite Scale**: 基于程序化规则生成无限数量的铰接物体
2. **Physics-Aligned**: 每个资产的 URDF 包含精确的惯性、碰撞、动力学参数
3. **Topology-Diverse**: 覆盖 16+ 种盒子拓扑结构，每种有无限参数变体
4. **Sim-Ready**: 生成的资产可直接用于 PyBullet/Isaac Sim 训练

### 1.3 已完成的技术基础

基于 `MailerBox_Simple_Technical_Analysis_Report.md`，我们已解决：

| 技术挑战 | 解决方案 | 验证状态 |
|---------|---------|---------|
| 薄壳惯性数值不稳定 | `thin_shell_inertia.py` 稳健计算 | ✅ 10 seeds 验证通过 |
| 多资产惯性合并 | 平行轴定理 `combine_multiple_inertias()` | ✅ 数学正确性验证 |
| URDF 规范合规 | 每个 link 最多一个 `<inertial>` | ✅ PyBullet 加载测试 |
| Self-collision | CoACD 凸分解生成 collision mesh | ✅ 碰撞检测正常 |

---

## 2. Research Questions

### RQ1: Data-Centric
> 程序化生成的 Physics-Aligned 数据能否显著提升感知模型对未见铰接物体的泛化能力？

### RQ2: Perception
> 如何设计网络架构，使其能从单张 RGB-D 推断出完整的 URDF 拓扑和关节状态？

### RQ3: Sim2Real
> Physics-Aligned 数据能否有效缩小 Sim-to-Real Gap，实现零样本操作策略迁移？

---

## 3. Technical Roadmap

### Phase 1: Data Engine (当前阶段)

#### 3.1.1 盒型拓扑扩展

基于 `Box_URDF_Generation_Plan.md`，需实现的盒型：

| 优先级 | 盒型 | DOF | 特殊挑战 |
|--------|------|-----|---------|
| P0 | MailerBox-Simple | 2 | ✅ 已完成 |
| P0 | TuckEndBox (双插盒) | 4 | 多关节联动（✅ 已有 URDF 导出链路；后续做批量/标注推广） |
| P1 | SlipLidBox (天地盒) | 1 | ✅ 已完成最小 prismatic 版本（`SlipLidBoxFactory` + URDF 导出/验证） |
| P1 | DrawerBox (抽屉盒) | 1 | ✅ 已完成最小 prismatic 版本（`DrawerBoxFactory` + URDF 导出/验证） |
| P2 | RSCBox (平口箱) | 8 | 多面板联动 |
| P2 | LockBottomBox (锁底盒) | 4 | 自锁约束 |

#### 3.1.2 惯性修正推广

将 `thin_shell_inertia.py` 的修正算法推广到所有盒型：

```python
# 通用惯性计算流程
def compute_link_inertia(link_assets: List[Mesh]) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    对 link 内的所有资产应用平行轴定理合并惯性。
    
    数学原理:
        M = Σ m_i
        C = (1/M) × Σ (m_i × c_i)
        I = Σ [I_i + m_i × ((r·r)E - r⊗r)]
    
    其中 r_i = c_i - C
    """
    masses, coms, tensors = [], [], []
    
    for asset in link_assets:
        m, I, c = thinshell.calculate_robust_inertia(
            vertices=asset.vertices,
            faces=asset.faces,
            density=asset.density,
        )
        masses.append(m)
        coms.append(c)
        tensors.append(I)
    
    return thinshell.combine_multiple_inertias(masses, coms, tensors)
```

#### 3.1.3 域随机化材质系统

| 材质类型 | 视觉参数 | 物理参数 |
|---------|---------|---------|
| Corrugated (瓦楞纸) | 纹理、颜色 | ρ=150-250 kg/m³, μ=0.4-0.6 |
| Kraft (牛皮纸) | 纹理、磨损 | ρ=200-350 kg/m³, μ=0.3-0.5 |
| Glossy (覆膜) | 反光、彩印 | ρ=300-500 kg/m³, μ=0.2-0.4 |
| Plastic (塑料) | 透明度、颜色 | ρ=900-1200 kg/m³, μ=0.3-0.5 |

#### 3.1.4 自动标注管线

输出格式设计：

```
infinigen_anybox_dataset/
├── train/
│   ├── 000001/
│   │   ├── rgb.png                 # 1024×768 RGB
│   │   ├── depth.npy               # 对应深度图 (float32, meters)
│   │   ├── segmentation.png        # Link-level semantic mask
│   │   ├── segmentation_label_map.json # label-id -> URDF link 名称/语义（可读映射）
│   │   ├── instance.png            # Instance segmentation
│   │   ├── keypoints.json          # 关节轴线起止点 + 可见性
│   │   ├── urdf_gt.urdf            # Ground Truth URDF
│   │   ├── joint_state.json        # 当前关节角度
│   │   ├── camera_intrinsics.json  # 相机内参
│   │   └── metadata.json           # 盒型、材质、尺寸等
│   └── ...
├── val/
└── test/
```

---

### Phase 2: Perception Model

#### 3.2.1 Topo-Box-Net 架构设计

```
                    RGB-D Input
                         │
                         ▼
              ┌──────────────────┐
              │  DINOv2 + SAM2   │  (Frozen Backbone)
              │  Visual Encoder  │
              └────────┬─────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐
   │ Topology │  │   Part   │  │ Physics  │
   │   Head   │  │Seg Head  │  │   Head   │
   └────┬─────┘  └────┬─────┘  └────┬─────┘
        │             │             │
        ▼             ▼             ▼
   Box Type +    Per-Link       Joint Axis +
   #Joints     6D Pose + Mask    Dynamics
```

**输出**：
- Topology Head: 盒型分类 (16 classes) + 关节数量
- Part Seg Head: 每个 link 的 mask + 6D pose (NPCS inspired)
- Physics Head: 关节轴向、范围、动力学参数

#### 3.2.2 Kinematic Consistency Loss

**核心思想**：利用 URDF 的父子关系，约束预测的 link poses 必须满足 hinge/prismatic 约束。

**数学形式化**：

对于一个 hinge joint 连接 parent link $P$ 和 child link $C$：

$$
\mathcal{L}_{kinematic} = \sum_{joints} \left\| \mathbf{R}_C^T (\mathbf{t}_P - \mathbf{t}_C) - \mathbf{a} \cdot \theta \right\|^2 + \lambda \left\| \mathbf{a}^T \mathbf{a} - 1 \right\|^2
$$

其中：
- $\mathbf{R}_C, \mathbf{t}_C$: child link 的旋转矩阵和位置
- $\mathbf{a}$: 预测的关节轴
- $\theta$: 预测的关节角度
- $\lambda$: 轴向单位化惩罚权重

---

### Phase 3: Sim2Real Validation

#### 3.3.1 实验设计

**消融变量**：
1. 数据量 (1k, 10k, 100k samples)
2. 物理对齐程度 (with/without inertia correction)
3. 域随机化强度 (low/medium/high)

**评估指标**：
- **Perception**: mIoU (segmentation), ADD-S (pose), Topology Accuracy
- **Manipulation**: Open Success Rate, Close Success Rate, Generalization to Unseen Types

#### 3.3.2 真机验证方案

| 设备 | 用途 |
|------|------|
| RealSense D435 | RGB-D 输入 |
| Franka Emika Panda | 操作执行 |
| 3D 打印盒子 | 测试物体（覆盖训练中未见的变体） |

---

## 4. Timeline

| 阶段 | 任务 | 目标 |
|------|------|------|
| Phase 1.1 | 完成 TuckEndBox, SlipLidBox URDF 导出 | 验证架构泛化性 |
| Phase 1.2 | 实现自动标注管线 | 生成 10k 训练样本 |
| Phase 1.3 | 域随机化材质 | 增强视觉多样性 |
| Phase 2.1 | Topo-Box-Net v1 | 基线感知模型 |
| Phase 2.2 | Kinematic Loss | 结构一致性约束 |
| Phase 3.1 | Sim 验证 | 仿真中操作成功率 |
| Phase 3.2 | Real 验证 | 真机操作成功率 |

---

## 5. Expected Contributions

1. **Dataset**: Infinigen-AnyBox — 首个大规模 Physics-Aligned 铰接容器数据集
2. **Method**: Topo-Box-Net — 拓扑感知的铰接物体感知网络
3. **Insight**: 证明 Physics-Aligned 数据对 Sim2Real 的关键作用

---

## 6. Related Work

### 6.1 铰接物体数据集

| 数据集 | 规模 | 物理属性 | 拓扑多样性 |
|--------|------|---------|-----------|
| PartNet-Mobility | ~2000 | ❌ | 低 |
| HOI4D | ~4000 | ❌ | 中 |
| **Infinigen-AnyBox (Ours)** | **∞** | **✅** | **高** |

### 6.2 铰接物体感知

| 方法 | 输入 | 输出 | 局限 |
|------|------|------|------|
| CAP-Net | RGB-D | Part Pose | 需要 NPCS 标注 |
| PhysNAP | Point Cloud | URDF | 推理慢 (Diffusion) |
| **Topo-Box-Net (Ours)** | **RGB-D** | **URDF + State** | **实时 + 结构感知** |

---

## 7. Risk Analysis

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|---------|
| 感知网络泛化差 | 中 | 高 | 增加数据多样性、引入对比学习 |
| Sim2Real Gap 过大 | 中 | 高 | 强化域随机化、真机微调 |
| 计算资源不足 | 低 | 中 | 使用高效 backbone (MobileSAM) |

---

## 8. Next Immediate Actions

1. **[P0]** 开始 TuckEndBox (双插盒) 的 URDF 生成开发
2. **[P0]** 设计数据集目录结构和标注格式规范
3. **[P1]** 阅读 CAP-Net 论文，理解 NPCS 表示的数学原理

---

*本文档将随研究进展持续更新。*
