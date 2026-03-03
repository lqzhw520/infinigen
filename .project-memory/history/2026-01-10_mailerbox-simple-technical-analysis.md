# MailerBox-Simple URDF 技术分析报告

**日期**: 2026-02-03  
**版本**: v1.1  
**作者**: 技术分析团队  
**目的**: 解答下游仿真团队关于 self-collision、joint limit 和 inertial 计算的技术疑问

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [Self-Collision 实现分析](#2-self-collision-实现分析)
   - 2.1 文档中的做法评估
   - 2.2 生成 Self-Collision 的方式和逻辑
   - 2.3 代码详细注释
   - 2.4 流程图
3. [Joint Limit 与 Self-Collision 的关系](#3-joint-limit-与-self-collision-的关系)
   - 3.1 Joint Limit 是否等于 Self-Collision？
   - 3.2 PyBullet 中 Joint Limit 的约束机制
   - 3.3 当前实现的严格性分析
4. [Inertial 标签的严重问题](#4-inertial-标签的严重问题)
   - 4.1 问题发现
   - 4.2 URDF 规范分析
   - 4.3 当前代码的逻辑缺陷
   - 4.4 物理正确性评估
   - 4.5 修复方案
5. [结论与建议](#5-结论与建议)
6. [附录](#6-附录)
7. [URDF 结构规范详解](#7-urdf-结构规范详解-annotated-urdf-reference)
   - 7.1 URDF 文件整体结构
   - 7.2 标签完整释义
   - 7.3 MailerBox-Simple 实例标注
   - 7.4 PyBullet 加载配置说明
8. [P0 修复完成记录](#8-p0-修复完成记录-2026-01-27)
   - 8.1 修复内容
   - 8.2 验证结果
   - 8.3 修复前后对比
   - 8.4 关键数学验证

---

## 1. 执行摘要

本报告针对下游仿真团队提出的三个关键技术问题进行了深入分析：

| 问题 | 分析结论 | 严重程度 |
|------|---------|---------|
| Self-collision 实现方式 | 使用 CoACD 凸分解生成 collision meshes，**做法正确** | ✅ 正确 |
| Joint limit ≠ Self-collision | 两者是完全不同的物理约束机制，**需要澄清概念** | ⚠️ 需澄清 |
| Inertial 标签多个（单 link 多 inertial） | ✅ 已修复：对每个 link 合并多个子 mesh 惯性为 **单个 `<inertial>`**（平行轴定理）并通过 PyBullet 验证 | ✅ 已修复 |

**关键结论（更新）**：
1. **URDF 规范约束仍然成立**：一个 `<link>` **至多一个** `<inertial>`。
2. **当前实现已满足规范**：`infinigen/core/sim/exporters/urdf_exporter.py` 会收集同一 link 内多个几何体的 \((m_i,c_i,I_i)\)，并使用平行轴定理合并为 **单个 `<inertial>`**。
3. **严格验证已通过**：`scripts/verify_urdf_inertia_fix.py` 已对 `mailerbox_simple`（10 个 seeds）以及 `drawerbox/sliplidbox/tuckendbox`（seed=42）完成结构与 PyBullet 载入验证。

---

## 2. Self-Collision 实现分析

### 2.1 文档中的做法评估

**文档描述**（来自 `Box_URDF_Generation_Plan.md:87`）：
> 本项目标准做法：导出时 `visual_only=False`，并使用 `coacd` 对网格做凸分解，生成专用 collision meshes。

**评估结论**：✅ **做法正确且合理**

原因分析：
1. **凸分解的必要性**：物理仿真器（PyBullet/MuJoCo/Isaac）的碰撞检测对凸形状效率最高
2. **CoACD 算法**：Collision-Aware Approximate Convex Decomposition 是当前业界领先的凸分解算法
3. **视觉与碰撞分离**：`<visual>` 保留高精度渲染网格，`<collision>` 使用简化凸形状，符合标准实践

### 2.2 生成 Self-Collision 的方式和逻辑

Self-collision 在本项目中的实现分为两个层次：

#### 层次 1：Collision Mesh 生成（几何层面）

```
原始网格 (visual mesh)
    ↓
export_sim_ready() 函数
    ↓
Trimesh 加载并预处理
    ↓
CoACD 凸分解
    ↓
输出 *_col*.obj 文件
    ↓
URDF 中写入 <collision> 标签
```

#### 层次 2：运行时碰撞检测（仿真层面）

```
PyBullet/MuJoCo 加载 URDF
    ↓
解析每个 link 的 <collision> geometry
    ↓
构建碰撞对 (collision pairs)
    ↓
仿真步进时检测接触
    ↓
施加接触力/约束
```

### 2.3 代码详细注释

以下是 `infinigen/tools/export.py` 中 `export_sim_ready()` 函数的关键代码，附详细注释：

```python
def export_sim_ready(
    obj: bpy.types.Object,
    output_folder: Path,
    image_res: int = 1024,
    translation: Tuple = (0, 0, 0),
    name: Optional[str] = None,
    visual_only: bool = False,       # 关键参数：是否只导出视觉网格
    collision_only: bool = False,
    separate_asset_dirs: bool = True,
    zaxis: np.array = np.array([0, 0, 1]),
) -> Dict[str, List[Path]]:
    """
    导出 visual 和 collision 两类资产。
    
    核心逻辑：
    1. 先导出 visual mesh (原始几何)
    2. 如果 visual_only=False，使用 CoACD 对网格做凸分解
    3. 每个凸分解结果保存为独立的 *_col*.obj 文件
    """
    
    # 检查 CoACD 依赖
    if not visual_only:
        assert coacd is not None, "coacd is required to export simulation assets."

    # ... 目录创建等 ...

    # 导出 visual mesh (用于渲染)
    with butil.SelectObjects(obj, active=1):
        bpy.ops.wm.obj_export(
            filepath=str(visual_export_file),
            up_axis="Z",
            forward_axis="Y",
            export_selected_objects=True,
            export_triangulated_mesh=True,  # CoACD 要求三角化网格
        )
    
    if visual_only:
        return asset_exports  # 提前返回，不生成碰撞网格

    # 以下是碰撞网格生成的核心逻辑
    clone = butil.deep_clone_obj(obj)
    parts = butil.split_object(clone)  # 按材质/松散几何拆分

    collision_count = 0
    for part in parts:
        # 导出临时 OBJ 文件
        with butil.SelectObjects(part, active=1):
            bpy.ops.wm.obj_export(
                filepath=str(part_export_obj_file),
                export_triangulated_mesh=True,
            )

        # 使用 Trimesh 加载并预处理
        mesh_tri = trimesh.load(str(part_export_obj_file), force="mesh")
        
        # 坐标系对齐
        T = trimesh.geometry.align_vectors(zaxis, np.array([0, 0, 1]))
        mesh_tri.apply_transform(T)
        trimesh.repair.fix_inversion(mesh_tri)
        
        # 检查网格是否为有效体积
        preprocess_mode = "off"
        if not mesh_tri.is_volume:
            preprocess_mode = "on"  # 需要预处理修复非流形网格

        # ===== 关键：CoACD 凸分解 =====
        mesh = coacd.Mesh(mesh_tri.vertices, mesh_tri.faces)
        subparts = coacd.run_coacd(
            mesh=mesh,
            threshold=0.05,        # 凸度阈值：越小分解越精细
            max_convex_hull=-1,    # 不限制凸包数量
            preprocess_mode=preprocess_mode,
            mcts_max_depth=3,      # 搜索深度
        )
        
        # 保存每个凸分解结果
        for vs, fs in subparts:
            collision_export_file = (
                collision_export_folder / f"{export_name}_col{collision_count}.obj"
            )
            subpart_mesh = trimesh.Trimesh(vs, fs)
            subpart_mesh.export(str(collision_export_file))
            asset_exports["collision"].append(collision_export_file)
            collision_count += 1

    return asset_exports
```

### 2.4 流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Self-Collision 生成流程                               │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌───────────────────────┐
                    │  Blender 几何节点树   │
                    │  (MailerBoxFactory)   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  运动学编译器         │
                    │  kinematic_compiler   │
                    │  - 构建关节树         │
                    │  - 分配 link IDs      │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  URDF 导出器          │
                    │  urdf_exporter.py     │
                    └───────────┬───────────┘
                                │
           ┌────────────────────┼────────────────────┐
           │                    │                    │
           ▼                    ▼                    ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   link_0    │     │   link_1    │     │   link_2    │
    │  (box_base) │     │    (lid)    │     │(front_flap) │
    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
           │                   │                   │
           ▼                   ▼                   ▼
    ┌─────────────────────────────────────────────────────┐
    │              export_sim_ready() 函数                │
    │                                                     │
    │  ┌─────────────────────────────────────────────┐   │
    │  │ Step 1: 导出 Visual Mesh                    │   │
    │  │         geom_*.obj (高精度渲染用)           │   │
    │  └─────────────────────────────────────────────┘   │
    │                         │                          │
    │                         ▼                          │
    │  ┌─────────────────────────────────────────────┐   │
    │  │ Step 2: Trimesh 加载 + 预处理               │   │
    │  │         - 三角化                            │   │
    │  │         - 坐标系对齐                        │   │
    │  │         - 非流形修复                        │   │
    │  └─────────────────────────────────────────────┘   │
    │                         │                          │
    │                         ▼                          │
    │  ┌─────────────────────────────────────────────┐   │
    │  │ Step 3: CoACD 凸分解                        │   │
    │  │         coacd.run_coacd(threshold=0.05)     │   │
    │  │         - 输入: 任意三角网格                │   │
    │  │         - 输出: 多个凸包                    │   │
    │  └─────────────────────────────────────────────┘   │
    │                         │                          │
    │                         ▼                          │
    │  ┌─────────────────────────────────────────────┐   │
    │  │ Step 4: 保存 Collision Meshes               │   │
    │  │         geom_*_col0.obj, geom_*_col1.obj... │   │
    │  └─────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  URDF 文件生成        │
                    │                       │
                    │  <link name="link_0"> │
                    │    <visual>           │
                    │      <mesh>geom_0.obj │
                    │    </visual>          │
                    │    <collision>        │
                    │      <mesh>geom_0_col0│
                    │    </collision>       │
                    │  </link>              │
                    └───────────────────────┘
```

---

## 3. Joint Limit 与 Self-Collision 的关系

### 3.1 Joint Limit 是否等于 Self-Collision？

**答案：否！这是两个完全不同的物理约束机制。**

| 特性 | Joint Limit | Self-Collision |
|------|-------------|----------------|
| **定义** | 关节运动范围的边界约束 | 刚体之间的几何穿透检测 |
| **作用层级** | 关节层 (Joint) | 几何层 (Geometry) |
| **约束类型** | 运动学约束 | 碰撞约束 |
| **URDF 位置** | `<joint><limit lower="..." upper="..."/>` | `<link><collision>...</collision>` |
| **物理含义** | "这个铰链最多能转到哪里" | "这两个部件不能互相穿透" |

**学术定义**：

- **Joint Limit（关节限位）**：
  - 是一种**单边约束（unilateral constraint）**
  - 施加于广义坐标空间 \(q\)：\(q_{\min} \leq q \leq q_{\max}\)
  - 与几何形状无关，纯粹是运动学范围限制

- **Self-Collision（自碰撞）**：
  - 是一种**接触约束（contact constraint）**
  - 施加于笛卡尔空间：\(\phi(x_A, x_B) \geq 0\)（符号距离函数）
  - 依赖于 collision geometry 的具体形状
  - 两个 link 之间的碰撞通常称为 self-collision（同一机器人/物体内部）

**图示对比**：

```
Joint Limit:                          Self-Collision:
                                      
   ●══════════●                          ▓▓▓▓▓
   │          │                          ▓    ▓
   │  q_min   │   q_max                  ▓    ▓◄── link_1 collision mesh
   │    ↓     │     ↓                    ▓▓▓▓▓
   └────●─────┴─────●                        │
        │     ^     │                        │ 碰撞检测
        │  当前角度  │                        ↓
        └───────────┘                    ▓▓▓▓▓▓▓
                                         ▓      ▓
  "关节只能在这个范围内转动"              ▓      ▓◄── link_0 collision mesh
                                         ▓▓▓▓▓▓▓
                                      
                                      "这两个几何体不能穿透"
```

### 3.2 PyBullet 中 Joint Limit 的约束机制

您同事的观察是正确的：**PyBullet 中的 joint limit 是软约束，在大力矩下可能被违反。**

**技术原理**：

1. **PyBullet 使用 Bullet 物理引擎的约束求解器**
   - 约束通过拉格朗日乘子法求解
   - 采用 Sequential Impulse 方法迭代求解
   - 迭代次数有限（默认 50 次），无法保证精确满足所有约束

2. **Joint Limit 的实现**
   ```python
   # PyBullet 内部伪代码
   def solve_joint_limit(joint, external_torque):
       # 计算约束违反量
       if joint.position < joint.lower_limit:
           violation = joint.position - joint.lower_limit
       elif joint.position > joint.upper_limit:
           violation = joint.position - joint.upper_limit
       else:
           return  # 未违反
       
       # 计算恢复冲量（受最大冲量限制）
       corrective_impulse = clamp(
           -violation * stiffness,
           -max_impulse, max_impulse
       )
       
       # 如果外部力矩过大，恢复冲量可能不足以阻止违反
       joint.apply_impulse(corrective_impulse)
   ```

3. **为什么鼠标拖拽会超过 limit？**
   - 鼠标拖拽施加的是位置约束（虚拟弹簧）
   - 弹簧力可能非常大（取决于 PyBullet 的拖拽参数）
   - 当拖拽力 > joint limit 恢复力时，limit 被突破

#### 3.2.1 影响“软/硬程度”的关键参数（数值求解视角）

无论是 joint limit 还是 contact（self-collision），在 Bullet/PyBullet 中其“严格性”本质上受 **离散时间步进 + 迭代求解器**控制，关键参数包括：

- `fixedTimeStep` / `numSubSteps`
  - 时间步越大，单步误差越大；substep 越多，约束更接近满足
- `numSolverIterations`
  - 迭代次数越多，约束违反量（penetration / limit violation）通常越小，但更耗时
- `contactERP` / `frictionERP` / `globalCFM`
  - ERP/CFM 控制约束“弹簧-阻尼”特性，影响允许的穿透量与回弹速度

因此，“joint limit 可被突破”与“self-collision 可能出现小穿透”在数值上是同一类现象：**迭代求解有限步近似**，并非概念混淆。

### 3.3 当前实现的严格性分析

**问题**：我们的 self-collision 是"严格"还是"宽松"的？

**分析**：

| 维度 | 当前实现 | 评估 |
|------|---------|------|
| **Collision Mesh 精度** | CoACD 凸分解 (threshold=0.05) | 中等精度 |
| **碰撞检测算法** | PyBullet 默认 GJK/EPA | 精确凸-凸检测 |
| **接触响应** | 基于约束的接触力 | 软约束（可穿透） |
| **Joint Limit 限制** | viewer_safe 版本有约束 | 仅约束 lid 角度 |

**结论**：当前实现属于**中等严格**的 self-collision：

1. **几何层面**：CoACD 生成的碰撞网格是原始网格的凸近似，存在一定误差
2. **物理层面**：PyBullet 的接触约束是软约束，在大力/高速下可能穿透
3. **实用性**：对于正常仿真场景（重力作用、机器人操作力）足够使用

**对于您的任务场景的适用性评估**：

```
适用场景：
├── ✅ 重力作用下的自然运动（盖子落下）
├── ✅ 小力矩操作（正常机器人夹取）
├── ✅ 规划验证（碰撞检测用于运动规划）
└── ⚠️ 需要注意：高速碰撞、大力拖拽时可能穿透

不适用场景：
├── ❌ 需要绝对刚性接触的场景
└── ❌ 需要精确到毫米级的间隙仿真
```

---

## 4. Inertial 标签的严重问题

### 4.1 问题发现

**历史问题（修复前）**：曾经在旧版本导出的 `mailerbox_simple.urdf` 中观察到 `link_0` 内部包含 **多个 `<inertial>` 元素**（例如 5 个），这**违反 URDF 规范**（每个 link 最多 1 个 inertial）。

> ✅ **当前状态**：该问题已修复。最新导出满足：每个物理 link（`link_0/link_1/link_2`）各自 **最多一个** `<inertial>`，并已通过 `scripts/verify_urdf_inertia_fix.py` 的结构 + PyBullet 校验（见 Section 8）。

```xml
<!-- 修复前示例（历史版本摘录，用于说明“单 link 多 inertial”的问题形态） -->
<link name="link_0">
    <visual>...</visual>
    <inertial>  <!-- #1: 背板 -->
      <mass value="0.0883799956691"/>
      <inertia ixx="..." .../>
      <origin xyz="0.0 -0.0848066806793 0.0"/>
    </inertial>
    <collision>...</collision>
    
    <visual>...</visual>
    <inertial>  <!-- #2: 后板 -->
      <mass value="0.0959830257748"/>
      ...
    </inertial>
    <collision>...</collision>
    
    <visual>...</visual>
    <inertial>  <!-- #3: 左侧板 -->
      <mass value="0.0664200004506"/>
      ...
    </inertial>
    <collision>...</collision>
    
    <visual>...</visual>
    <inertial>  <!-- #4: 右侧板 -->
      <mass value="0.0664200004506"/>
      ...
    </inertial>
    <collision>...</collision>
    
    <visual>...</visual>
    <inertial>  <!-- #5: 底板 -->
      <mass value="0.201297989126"/>
      ...
    </inertial>
    <collision>...</collision>
</link>
```

### 4.2 URDF 规范分析

根据 **URDF 官方规范**（http://wiki.ros.org/urdf/XML/link）：

> **Link Element Structure:**
> ```xml
> <link name="my_link">
>   <inertial>        <!-- 最多一个 -->
>     <origin .../>
>     <mass .../>
>     <inertia .../>
>   </inertial>
>   <visual>          <!-- 可以多个 -->
>     ...
>   </visual>
>   <collision>       <!-- 可以多个 -->
>     ...
>   </collision>
> </link>
> ```

| 元素 | 允许数量 | 当前实现 | 符合规范？ |
|------|---------|---------|-----------|
| `<inertial>` | **0 或 1** | 5 | ❌ **违反** |
| `<visual>` | 0 或多个 | 5 | ✅ 符合 |
| `<collision>` | 0 或多个 | 5 | ✅ 符合 |

### 4.3 当前代码的逻辑缺陷

问题出在 `urdf_exporter.py` 的 `_populate_links()` 函数中：

```python
# urdf_exporter.py 第 142-221 行 (简化)
for asset in root.assets:
    # ... 创建 visual 和 collision 元素 ...
    
    # 问题：每个 asset 都创建一个独立的 inertial！
    robust_mass, I_tensor, com = thinshell.calculate_robust_inertia(...)
    
    inertial = create_element("inertial")
    inertial.append(create_element("mass", value=str(robust_mass)))
    inertial.append(create_element("inertia", ...))
    inertial.append(create_element("origin", xyz=...))
    
    link.append(inertial)  # ❌ 每次循环都添加一个 inertial！
```

**根本原因**：代码将每个几何 asset 的惯性单独写入，而不是先合并再写入。

### 4.4 物理正确性评估

**下游博士同事的观察是正确的**：合并多个几何体的惯性需要使用**平行轴定理**。

**正确的数学过程**：

设有 \(n\) 个几何体，各自的质量、质心、惯性张量为 \((m_i, \vec{c}_i, \mathbf{I}_i)\)：

1. **合并质量**：
   $$M = \sum_{i=1}^{n} m_i$$

2. **合并质心**：
   $$\vec{C} = \frac{1}{M} \sum_{i=1}^{n} m_i \vec{c}_i$$

3. **合并惯性张量（平行轴定理）**：
   $$\mathbf{I}_{total} = \sum_{i=1}^{n} \left[ \mathbf{I}_i + m_i \left( (\vec{c}_i - \vec{C})^T (\vec{c}_i - \vec{C}) \mathbf{E} - (\vec{c}_i - \vec{C})(\vec{c}_i - \vec{C})^T \right) \right]$$

   其中 \(\mathbf{E}\) 是 3×3 单位矩阵，第二项是平行轴偏移的贡献。

**当前实现的问题**：
- PyBullet 解析器行为不确定：可能只读取第一个 inertial
- 即使读取所有 inertial，也没有正确合并
- 导致 `link_0` 的物理属性**完全错误**

**量化影响**（以 seed 101 为例）：

| 场景 | 质量 | 质心位置 |
|------|------|----------|
| 只读第一个 inertial | 0.088 kg | (0, -0.085, 0) |
| 正确合并所有 | ~0.52 kg | 需重新计算 |

**误差**：质量相差约 **6 倍**！

### 4.5 修复方案

需要修改 `urdf_exporter.py`，在写入 URDF 之前合并所有 asset 的惯性：

```python
# 建议的修复代码（伪代码）
def _populate_links(self, root, ...):
    link = create_element("link", name=link_name)
    
    # 收集所有 asset 的惯性数据
    masses = []
    coms = []
    inertias = []
    
    for asset in root.assets:
        # 计算单个 asset 的惯性
        mass, I_tensor, com = thinshell.calculate_robust_inertia(...)
        masses.append(mass)
        coms.append(com)
        inertias.append(I_tensor)
        
        # visual 和 collision 照常添加
        link.append(visual)
        link.append(collision)
    
    # 合并惯性（使用平行轴定理）
    total_mass, combined_com, combined_inertia = combine_inertias(
        masses, coms, inertias
    )
    
    # 只创建一个 inertial 元素
    inertial = create_element("inertial")
    inertial.append(create_element("mass", value=str(total_mass)))
    inertial.append(create_element("inertia", 
        ixx=..., ixy=..., ixz=..., iyy=..., iyz=..., izz=...))
    inertial.append(create_element("origin", xyz=array_to_string(combined_com)))
    link.append(inertial)  # 只添加一次！


def combine_inertias(masses, coms, inertias):
    """
    使用平行轴定理合并多个刚体的惯性。
    
    Returns:
        total_mass: 总质量
        combined_com: 合并后的质心
        combined_inertia: 合并后的惯性张量 (3x3)
    """
    import numpy as np
    
    masses = np.array(masses)
    coms = np.array(coms)  # (n, 3)
    
    # 总质量
    total_mass = np.sum(masses)
    
    # 合并质心
    combined_com = np.sum(masses[:, None] * coms, axis=0) / total_mass
    
    # 合并惯性张量（平行轴定理）
    combined_inertia = np.zeros((3, 3))
    for m, c, I in zip(masses, coms, inertias):
        r = c - combined_com  # 质心偏移向量
        # 平行轴定理：I_new = I_old + m * (r·r * E - r ⊗ r)
        r_squared = np.dot(r, r)
        r_outer = np.outer(r, r)
        parallel_axis_contribution = m * (r_squared * np.eye(3) - r_outer)
        combined_inertia += I + parallel_axis_contribution
    
    return total_mass, combined_com, combined_inertia
```

---

## 5. 结论与建议

### 5.1 问题总结

| 问题 | 严重程度 | 状态 | 建议优先级 |
|------|---------|------|-----------|
| Self-collision 实现 | - | ✅ 正确 | - |
| Joint limit ≠ Self-collision 概念混淆 | 低 | ✅ 已在本文 Section 7.2.3 澄清 | 低 |
| Inertial 标签重复（单 link 多 inertial） | **高** | ✅ 已修复（`urdf_exporter.py` 合并惯性） | **P0 - 已完成** |

### 5.2 修复优先级

**P0（阻塞性问题）**：✅ 已完成（本仓库环境已验证）
1. ✅ 修改 `infinigen/core/sim/exporters/urdf_exporter.py`：对每个 `<link>` 收集各子 mesh 的 `(m_i, c_i, I_i)`，并使用 `thin_shell_inertia.combine_multiple_inertias(...)` 基于平行轴定理合并为 **单个 `<inertial>`**。
2. ✅ 重新导出所有 10 个 seed 的 URDF：`sim_exports/urdf/mailerbox_simple/{101..105,201..205}/mailerbox_simple.urdf`
3. ✅ 验证通过：
   - `sim_exports/urdf/mailerbox_simple/inertia_verification_report.json`（10/10 seeds passed）

**P1（重要但不阻塞）**：
1. ✅ 添加 URDF 规范/惯性校验脚本：`scripts/verify_urdf_inertia_fix.py`（输出 `sim_exports/urdf/mailerbox_simple/inertia_verification_report.json`）
2. ✅ 添加惯性合并单元测试：`tests/sim/test_combine_multiple_inertias.py`
3. ✅ 文档更新（本文件 Section 7 + Section 8）

### 5.3 对下游仿真的影响

| 仿真场景 | 当前状态下的问题 | 修复后 |
|---------|------------------|--------|
| 静态展示 | 无影响 | 无变化 |
| 重力仿真 | 质量错误导致下落行为异常 | 正确 |
| 碰撞响应 | 惯性错误导致碰撞后反弹不自然 | 正确 |
| 机器人操作 | 抓取力矩计算错误 | 正确 |

### 5.4 建议的验证步骤

修复后，建议进行以下验证（**务必开启 `URDF_USE_INERTIA_FROM_FILE`**，确保读取 URDF 内的惯性而非引擎近似）：

```python
import pybullet as p
import pybullet_data

# 加载修复后的 URDF
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
robot_id = p.loadURDF("mailerbox_simple.urdf", useFixedBase=True, flags=flags)

# 验证 link 惯性（base link 使用 linkIndex=-1，其余 linkIndex==jointIndex）
base_mass, _, base_I_diag, base_com, _ = p.getDynamicsInfo(robot_id, -1)[:5]
print(f"Base(link_0): mass={base_mass:.6f} kg, I_diag={base_I_diag}, com={base_com}")

for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    link_name = info[12].decode("utf-8")
    mass, _, I_diag, com, _ = p.getDynamicsInfo(robot_id, j)[:5]
    print(f"{link_name}: mass={mass:.6f} kg, I_diag={I_diag}, com={com}")

# 期望输出（修复后）：
# Link 0: mass=0.52xx kg, I_diag=(合理的合并值)
# Link 1: mass=0.19xx kg, ...
# Link 2: mass=0.09xx kg, ...
```

---

## 6. 附录

### 6.1 URDF 规范参考

- 官方规范: http://wiki.ros.org/urdf/XML
- Link 元素: http://wiki.ros.org/urdf/XML/link
- Inertial 元素: http://wiki.ros.org/urdf/XML/link#Elements

### 6.2 平行轴定理公式推导

对于质量为 \(m\)、质心在 \(\vec{c}\) 处、惯性张量为 \(\mathbf{I}_c\) 的刚体，当参考点从质心 \(\vec{c}\) 移动到新点 \(\vec{p}\) 时，新的惯性张量为：

$$\mathbf{I}_p = \mathbf{I}_c + m \left[ (\vec{r}^T \vec{r}) \mathbf{E} - \vec{r} \vec{r}^T \right]$$

其中 \(\vec{r} = \vec{c} - \vec{p}\)。

展开为分量形式：

$$I_{xx}^{(p)} = I_{xx}^{(c)} + m(r_y^2 + r_z^2)$$
$$I_{yy}^{(p)} = I_{yy}^{(c)} + m(r_x^2 + r_z^2)$$
$$I_{zz}^{(p)} = I_{zz}^{(c)} + m(r_x^2 + r_y^2)$$
$$I_{xy}^{(p)} = I_{xy}^{(c)} - m r_x r_y$$
$$I_{xz}^{(p)} = I_{xz}^{(c)} - m r_x r_z$$
$$I_{yz}^{(p)} = I_{yz}^{(c)} - m r_y r_z$$

### 6.3 相关代码文件

| 文件 | 作用 |
|------|------|
| `infinigen/core/sim/exporters/urdf_exporter.py` | URDF 导出器（需修复） |
| `infinigen/core/sim/physics/thin_shell_inertia.py` | 薄壳惯性计算 |
| `infinigen/tools/export.py` | 网格导出（含 CoACD 凸分解） |
| `scripts/export_mailerbox_simple_variants.py` | 批量导出脚本 |

### 6.4 修复代码完整实现

以下是建议添加到 `thin_shell_inertia.py` 的合并函数：

```python
def combine_multiple_inertias(
    masses: List[float],
    centers_of_mass: List[np.ndarray],
    inertia_tensors: List[np.ndarray],
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    使用平行轴定理合并多个刚体的惯性属性。
    
    这是解决 URDF 规范中“每个 link 最多只能有一个 <inertial>”的标准方法。
    注意：一个 URDF 文件通常包含多个 link，因此**会出现多个 <inertial>**（每个物理 link 一个）；
    本次修复针对的是 `link_0` 由于含多个面板资产而被错误写出多个 <inertial> 的问题。
    
    Args:
        masses: 各刚体的质量列表
        centers_of_mass: 各刚体的质心位置列表 (每个为 3D 向量)
        inertia_tensors: 各刚体的惯性张量列表 (每个为 3x3 矩阵)
        
    Returns:
        total_mass: 合并后的总质量
        combined_com: 合并后的质心位置 (3D 向量)
        combined_inertia: 合并后的惯性张量 (3x3 矩阵)
        
    物理原理:
        平行轴定理 (Parallel Axis Theorem):
        当惯性张量的参考点从质心 c 移动到新点 p 时:
        I_p = I_c + m * [(r·r)*E - r⊗r]
        其中 r = c - p, E 是单位矩阵, ⊗ 表示外积
    """
    import numpy as np
    
    n = len(masses)
    assert len(centers_of_mass) == n and len(inertia_tensors) == n
    
    masses = np.array(masses)
    coms = np.array(centers_of_mass)  # (n, 3)
    
    # 1. 计算总质量
    total_mass = np.sum(masses)
    
    if total_mass <= 0:
        # 边界情况：无质量
        return 0.001, np.zeros(3), np.eye(3) * 1e-9
    
    # 2. 计算合并后的质心 (质量加权平均)
    combined_com = np.sum(masses[:, None] * coms, axis=0) / total_mass
    
    # 3. 使用平行轴定理合并惯性张量
    combined_inertia = np.zeros((3, 3))
    
    for i in range(n):
        m = masses[i]
        I = inertia_tensors[i]
        r = coms[i] - combined_com  # 质心偏移向量
        
        # 平行轴偏移贡献: m * [(r·r)*E - r⊗r]
        r_dot_r = np.dot(r, r)
        r_outer_r = np.outer(r, r)
        parallel_axis_term = m * (r_dot_r * np.eye(3) - r_outer_r)
        
        # 累加
        combined_inertia += I + parallel_axis_term
    
    # 4. 确保正定性
    combined_inertia = _ensure_positive_definite(combined_inertia)
    
    return total_mass, combined_com, combined_inertia
```

---

## 7. URDF 结构规范详解 (Annotated URDF Reference)

本节为下游仿真团队提供 URDF 格式的**完整标签释义**，确保交付对齐和学术严谨性。

### 7.1 URDF 文件整体结构

```
┌─────────────────────────────────────────────────────────────────────┐
│  <?xml version="1.0" ?>                                             │
│  <robot name="object">                                              │
│    ├── <link name="world"/>            ← 虚拟世界锚点 (无质量)       │
│    ├── <joint name="world_joint"/>     ← 固定到世界                 │
│    ├── <joint name="mailer_lid_0"/>    ← lid 旋转关节               │
│    ├── <joint name="mailer_front_flap_0"/> ← front_flap 旋转关节    │
│    ├── <link name="link_0">            ← box_base (5 个面板合并)    │
│    │     ├── <visual> × 5                                           │
│    │     ├── <collision> × 5                                        │
│    │     └── <inertial> × 1  ← 平行轴定理合并后的惯性               │
│    ├── <link name="link_1">            ← lid                        │
│    │     ├── <visual> × 1                                           │
│    │     ├── <collision> × 1                                        │
│    │     └── <inertial> × 1                                         │
│    └── <link name="link_2">            ← front_flap                 │
│          ├── <visual> × 1                                           │
│          ├── <collision> × 1                                        │
│          └── <inertial> × 1                                         │
│  </robot>                                                           │
└─────────────────────────────────────────────────────────────────────┘
```

**关键计数**：
- 一个 URDF 文件有 **3 个物理 link**（`link_0/link_1/link_2`）
- 因此文件中有 **3 个 `<inertial>` block**（每个 link 一个）
- 搜索 `inertial` 字符串会命中 **6 次**（3 个开标签 + 3 个闭标签）

### 7.2 标签完整释义

#### 7.2.1 `<robot>` — 根节点

```xml
<robot name="object">
  ...
</robot>
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `name` | string | 机器人/物体的标识符，对物理仿真无影响 |

#### 7.2.2 `<link>` — 刚体定义

```xml
<link name="link_0">
  <visual>...</visual>      <!-- 可有多个 -->
  <collision>...</collision> <!-- 可有多个 -->
  <inertial>...</inertial>  <!-- 最多一个！URDF 规范硬约束 -->
</link>
```

| 子元素 | 数量限制 | 说明 |
|--------|---------|------|
| `<visual>` | 0..∞ | 渲染用几何体，不参与碰撞检测 |
| `<collision>` | 0..∞ | 碰撞检测用几何体（凸分解后） |
| `<inertial>` | **0..1** | 动力学属性：质量、质心、惯性张量 |

**URDF 规范引用**（[ROS Wiki](http://wiki.ros.org/urdf/XML/link)）：
> The `<inertial>` element describes the inertial properties of the link. **Only one `<inertial>` element is allowed per link.**

#### 7.2.3 `<joint>` — 关节定义

```xml
<joint name="mailer_lid_0" type="revolute">
  <origin xyz="0.0 0.097 0.045"/>       <!-- 关节在父 link 中的位姿 -->
  <parent link="link_0"/>               <!-- 父 link -->
  <child link="link_1"/>                <!-- 子 link -->
  <axis xyz="1.0 0.0 0.0"/>             <!-- 旋转轴方向 -->
  <limit lower="-3.14" upper="3.14"/>   <!-- 关节角度范围 [rad] -->
  <dynamics damping="0.45" friction="0.12"/>
</joint>
```

| 属性/子元素 | 说明 |
|------------|------|
| `type` | 关节类型：`fixed`（刚性连接）、`revolute`（旋转）、`prismatic`（滑动）等 |
| `<origin>` | 关节坐标系相对于父 link 坐标系的变换 |
| `<parent>` / `<child>` | 定义运动链拓扑结构 |
| `<axis>` | 可动关节的运动轴方向（在关节坐标系中表达） |
| `<limit>` | 关节位置范围（`lower`/`upper`），**这是运动学约束，非 self-collision** |
| `<dynamics>` | 关节阻尼和摩擦（物理引擎支持程度不同） |

**重要辨析**：
- `<limit>` 是**运动学约束**（kinematic constraint），定义关节可达范围
- **Self-collision** 是通过 `<collision>` 几何体在运行时检测的**动力学约束**
- 两者是完全不同的机制，不能互相替代

#### 7.2.4 `<visual>` — 渲染几何体

```xml
<visual>
  <geometry>
    <mesh filename="assets/geom_0.obj"/>  <!-- 高精度网格 -->
  </geometry>
  <origin xyz="0.0 -0.085 0.0"/>          <!-- 几何体在 link 中的位姿 -->
</visual>
```

| 子元素 | 说明 |
|--------|------|
| `<geometry>` | 几何形状（`mesh`/`box`/`sphere`/`cylinder`） |
| `<origin>` | 几何体相对 link 坐标系原点的摆放位姿 |
| `<material>` | 渲染材质（可选） |

#### 7.2.5 `<collision>` — 碰撞检测几何体

```xml
<collision>
  <geometry>
    <mesh filename="assets/geom_0_col0.obj"/>  <!-- 凸分解后的简化网格 -->
  </geometry>
  <origin xyz="0.0 -0.085 0.0"/>
</collision>
```

**本项目实践**：
- 使用 **CoACD（Collision-Aware Approximate Convex Decomposition）** 算法对原始网格进行凸分解
- 凸形状对物理引擎的碰撞检测效率最高
- `*_col0.obj`、`*_col1.obj` 等文件是凸分解后的各个凸包

#### 7.2.6 `<inertial>` — 动力学属性（核心）

```xml
<inertial>
  <mass value="0.5185"/>                                      <!-- 质量 [kg] -->
  <inertia ixx="0.00292" ixy="0" ixz="0"                     <!-- 惯性张量 [kg·m²] -->
           iyy="0.00448" iyz="0"
           izz="0.00649"/>
  <origin xyz="3e-19 0.0045 -0.0172"/>                        <!-- 质心位置 -->
</inertial>
```

| 子元素 | 物理意义 | 数学表达 |
|--------|---------|---------|
| `<mass>` | 刚体总质量 | $m$ [kg] |
| `<origin>` | 质心（COM）相对 link 原点的位置 | $\vec{c}$ [m] |
| `<inertia>` | 关于质心的惯性张量（6 个独立分量） | 见下方 |

**惯性张量数学定义**：

对于质量分布 $\rho(\vec{r})$，关于质心的惯性张量为：

$$
\mathbf{I} = \begin{bmatrix}
I_{xx} & I_{xy} & I_{xz} \\
I_{xy} & I_{yy} & I_{yz} \\
I_{xz} & I_{yz} & I_{zz}
\end{bmatrix}
$$

其中：
$$
I_{xx} = \int (y^2 + z^2) \, dm, \quad
I_{xy} = -\int xy \, dm, \quad \ldots
$$

**URDF 中的约定**：
- `<inertia>` 的 6 个属性 (`ixx`, `ixy`, `ixz`, `iyy`, `iyz`, `izz`) 是**相对于 `<origin>` 定义的惯性坐标系**表达的
- 惯性张量必须是**正定矩阵**（所有特征值 > 0）
- 对于薄壳物体，本项目使用 `thin_shell_inertia.py` 的稳健计算方法

#### 7.2.7 `<origin>` — 位姿变换（上下文敏感）

`<origin>` 在不同上下文中含义不同：

| 上下文 | 物理意义 |
|--------|---------|
| 在 `<joint>` 中 | 关节坐标系相对父 link 的变换（定义关节锚点） |
| 在 `<visual>`/`<collision>` 中 | 几何体相对 link 原点的摆放位姿 |
| 在 `<inertial>` 中 | **质心位置**相对 link 原点的偏移 |

```xml
<origin xyz="x y z" rpy="roll pitch yaw"/>
```

- `xyz`: 平移 [m]
- `rpy`: 欧拉角（Roll-Pitch-Yaw）[rad]，默认 "0 0 0"

#### 7.2.8 `<parent>` / `<child>` — 运动链拓扑（link 连接关系）

```xml
<parent link="link_0"/>
<child link="link_1"/>
```

- **作用**：在 `<joint>` 中定义“父 link → 子 link”的拓扑边。
- **属性**
  - `link`：link 名称（必须与 URDF 中 `<link name="...">` 一致）
- **物理含义**：该 joint 的广义坐标 \(q\) 作用于 parent 与 child 之间的相对变换。

#### 7.2.9 `<axis>` — 运动轴（在 joint frame 表达）

```xml
<axis xyz="1 0 0"/>
```

- revolute/continuous：绕该轴旋转（右手定则）
- prismatic：沿该轴平移
- 工程注意：
  - URDF 允许 axis 非单位向量，但多数引擎会在内部归一化；为了避免歧义，建议始终输出单位向量
  - axis 的坐标系是 **joint frame**（由 `<joint><origin .../>` 决定），不是 world frame

#### 7.2.10 `<limit>` — 关节范围（单位随 joint type 变化）

```xml
<limit lower="..." upper="..."/>
```

- revolute：`lower/upper` 单位为 **rad**
- prismatic：`lower/upper` 单位为 **m**
- continuous：通常不需要 `<limit>`（可视需求加入软限制）
- 重要：`<limit>` 是**运动学可达范围**，不是 self-collision；self-collision 由 `<collision>` 几何 + 仿真器接触求解决定。

#### 7.2.11 `<geometry>` / `<mesh>` — 几何资源与路径解析

```xml
<geometry>
  <mesh filename="assets/geom_0.obj"/>
</geometry>
```

- `<geometry>`：几何描述容器（mesh/box/sphere/cylinder）
- `<mesh filename="...">`：
  - `filename` 通常是 **相对 URDF 文件目录** 的相对路径（例如 `assets/geom_0.obj`）
  - 不同引擎对相对路径的搜索策略不同：
    - PyBullet 常需要设置 search path（例如 `p.setAdditionalSearchPath(urdf_dir)`）或使用绝对路径
  - 本项目导出目录结构为：`<urdf_dir>/assets/*.obj`，与 `filename="assets/..."` 对齐

#### 7.2.12 `<dynamics>` — 阻尼/摩擦（引擎依赖）

```xml
<dynamics damping="0.45" friction="0.12"/>
```

- **用途**：为关节提供阻尼/摩擦参数，提升仿真稳定性或拟合材料阻尼行为
- **注意**：不同仿真器对该标签支持程度不同（可能忽略或采用不同解释），因此不应把它当作“硬约束”。

### 7.3 MailerBox-Simple 实例标注

以 `seed 101` 的 `mailerbox_simple.urdf` 为例，完整标注：

```xml
<?xml version="1.0" ?>
<robot name="object">
  <!-- ============================================================
       世界锚点（虚拟 link，无物理属性）
       ============================================================ -->
  <link name="world"/>
  
  <!-- ============================================================
       world_joint: 将 box_base (link_0) 固定到世界坐标系
       type="fixed" 表示刚性连接，无自由度
       ============================================================ -->
  <joint name="world_joint" type="fixed">
    <origin xyz="0.0 0.0 0.0"/>
    <parent link="world"/>
    <child link="link_0"/>
    <dynamics damping="0.0" friction="0.0"/>
  </joint>
  
  <!-- ============================================================
       mailer_lid_0: lid 的旋转关节
       type="revolute" 表示绕轴旋转
       axis="1 0 0" 表示绕 X 轴旋转
       limit: 关节角度范围 [-π, +π] rad
       ============================================================ -->
  <joint name="mailer_lid_0" type="revolute">
    <origin xyz="0.0 0.096 0.045"/>   <!-- 关节位置：后边缘中点 -->
    <parent link="link_0"/>
    <child link="link_1"/>
    <dynamics damping="0.45" friction="0.12"/>
    <axis xyz="1.0 0.0 0.0"/>
    <limit lower="-3.141593" upper="3.141593"/>
  </joint>
  
  <!-- ============================================================
       mailer_front_flap_0: front_flap 的旋转关节
       连接在 lid (link_1) 的前边缘
       ============================================================ -->
  <joint name="mailer_front_flap_0" type="revolute">
    <origin xyz="0.0 0.0015 0.18"/>   <!-- 相对于 lid 的关节位置 -->
    <parent link="link_1"/>
    <child link="link_2"/>
    <dynamics damping="0.3" friction="0.1"/>
    <axis xyz="1.0 0.0 0.0"/>
    <limit lower="-3.1415927" upper="3.1415927"/>
  </joint>
  
  <!-- ============================================================
       link_2: front_flap
       只有 1 个资产，因此 visual/collision/inertial 各 1 个
       ============================================================ -->
  <link name="link_2">
    <visual>
      <geometry><mesh filename="assets/geom_6.obj"/></geometry>
      <origin xyz="0.0 0.00075 0.045"/>
    </visual>
    <collision>
      <geometry><mesh filename="assets/geom_6_col0.obj"/></geometry>
      <origin xyz="0.0 0.00075 0.045"/>
    </collision>
    <inertial>
      <mass value="0.096"/>
      <inertia ixx="6.48e-05" ixy="0" ixz="0" iyy="6.09e-04" iyz="0" izz="5.44e-04"/>
      <origin xyz="0.0 0.098 0.27"/>  <!-- front_flap 的质心 -->
    </inertial>
  </link>
  
  <!-- ============================================================
       link_1: lid
       只有 1 个资产
       ============================================================ -->
  <link name="link_1">
    <visual>
      <geometry><mesh filename="assets/geom_5.obj"/></geometry>
      <origin xyz="0.0 0.00075 0.09"/>
    </visual>
    <collision>
      <geometry><mesh filename="assets/geom_5_col0.obj"/></geometry>
      <origin xyz="0.0 0.00075 0.09"/>
    </collision>
    <inertial>
      <mass value="0.190"/>
      <inertia ixx="5.14e-04" ixy="0" ixz="0" iyy="1.59e-03" iyz="0" izz="1.08e-03"/>
      <origin xyz="0.0 0.097 0.135"/>  <!-- lid 的质心 -->
    </inertial>
  </link>
  
  <!-- ============================================================
       link_0: box_base
       包含 5 个面板资产（底板 + 前/后/左/右侧板）
       - visual × 5: 每个面板一个渲染网格
       - collision × 5: 每个面板一个碰撞网格
       - inertial × 1: 【关键】使用平行轴定理合并 5 个面板的惯性
       ============================================================ -->
  <link name="link_0">
    <!-- 后侧板 -->
    <visual>
      <geometry><mesh filename="assets/geom_0.obj"/></geometry>
      <origin xyz="0.0 -0.085 0.0"/>
    </visual>
    <collision>
      <geometry><mesh filename="assets/geom_0_col0.obj"/></geometry>
      <origin xyz="0.0 -0.085 0.0"/>
    </collision>
    
    <!-- 前侧板 -->
    <visual>
      <geometry><mesh filename="assets/geom_1.obj"/></geometry>
      <origin xyz="0.0 0.095 0.0"/>
    </visual>
    <collision>
      <geometry><mesh filename="assets/geom_1_col0.obj"/></geometry>
      <origin xyz="0.0 0.095 0.0"/>
    </collision>
    
    <!-- 左侧板 -->
    <visual>
      <geometry><mesh filename="assets/geom_2.obj"/></geometry>
      <origin xyz="-0.12 0.005 0.0"/>
    </visual>
    <collision>
      <geometry><mesh filename="assets/geom_2_col0.obj"/></geometry>
      <origin xyz="-0.12 0.005 0.0"/>
    </collision>
    
    <!-- 右侧板 -->
    <visual>
      <geometry><mesh filename="assets/geom_3.obj"/></geometry>
      <origin xyz="0.12 0.005 0.0"/>
    </visual>
    <collision>
      <geometry><mesh filename="assets/geom_3_col0.obj"/></geometry>
      <origin xyz="0.12 0.005 0.0"/>
    </collision>
    
    <!-- 底板 -->
    <visual>
      <geometry><mesh filename="assets/geom_4.obj"/></geometry>
      <origin xyz="0.0 0.0 -0.044"/>
    </visual>
    <collision>
      <geometry><mesh filename="assets/geom_4_col0.obj"/></geometry>
      <origin xyz="0.0 0.0 -0.044"/>
    </collision>
    
    <!-- ======================================================
         合并后的惯性（5 个面板通过平行轴定理计算）
         
         数学原理：
         M = Σ m_i                        (总质量)
         C = (1/M) × Σ (m_i × c_i)        (合并质心)
         I = Σ [I_i + m_i×((r·r)E - r⊗r)] (平行轴定理)
         
         其中 r_i = c_i - C
         ====================================================== -->
    <inertial>
      <mass value="0.5185"/>   <!-- 5 个面板质量之和 -->
      <inertia ixx="0.00292" ixy="0" ixz="0"
               iyy="0.00448" iyz="0"
               izz="0.00649"/>
      <origin xyz="0.0 0.0045 -0.0172"/>  <!-- 合并后的质心 -->
    </inertial>
  </link>
</robot>
```

### 7.4 PyBullet 加载配置说明

为确保物理引擎**严格使用 URDF 中的惯性参数**，加载时需设置：

```python
import pybullet as p

# 关键 flags:
# - URDF_USE_SELF_COLLISION: 启用 link 间的自碰撞检测
# - URDF_USE_INERTIA_FROM_FILE: 使用 URDF 文件中的惯性，而非重新估计
flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE

body_id = p.loadURDF("mailerbox_simple.urdf", useFixedBase=True, flags=flags)
```

**注意**：若不设置 `URDF_USE_INERTIA_FROM_FILE`，PyBullet 可能根据 collision geometry 重新估计惯性，导致 `getDynamicsInfo()` 返回的值与 URDF 中不一致。

---

## 8. P0 修复完成记录 (2026-01-27)

### 8.1 修复内容

| 修复项 | 文件 | 状态 |
|--------|------|------|
| 添加平行轴定理惯性合并函数 | `infinigen/core/sim/physics/thin_shell_inertia.py` | ✅ 完成 |
| 修改 URDF 导出器使用惯性合并 | `infinigen/core/sim/exporters/urdf_exporter.py` | ✅ 完成 |
| 创建验证脚本 | `scripts/verify_urdf_inertia_fix.py` | ✅ 完成 |
| 重新导出 10 个 seed | `sim_exports/urdf/mailerbox_simple/` | ✅ 完成 |
| 验证所有 URDF | 所有 10 个 seed | ✅ 通过 |

### 8.2 验证结果

```
######################################################################
# 最终验证结果
######################################################################
  总计: 10
  通过: 10
  失败: 0

✓ 所有 URDF 验证通过！惯性合并修复成功。
```

**PyBullet 关键验证配置（非常重要）**：
- 为确保 PyBullet **严格使用 URDF 文件中写出的惯性参数**，验证脚本加载 URDF 时使用：
  `flags = URDF_USE_SELF_COLLISION | URDF_USE_INERTIA_FROM_FILE`
- 若不设置 `URDF_USE_INERTIA_FROM_FILE`，PyBullet 可能会根据 collision geometry 重新估计惯性，从而导致
  `getDynamicsInfo()` 返回的惯性与 URDF 文件中的惯性不一致（这不是 URDF 写出错误，而是加载策略不同）。

### 8.3 修复前后对比

**修复前 (错误)**:
```xml
<link name="link_0">
    <visual>...</visual>
    <inertial>...</inertial>  <!-- 第1个 inertial：违反 URDF 规范 -->
    <collision>...</collision>
    
    <visual>...</visual>
    <inertial>...</inertial>  <!-- 第2个 inertial：违反 URDF 规范 -->
    ...
    <!-- 共 5 个 inertial 元素！ -->
</link>
```

**修复后 (正确)**:
```xml
<link name="link_0">
    <visual>...</visual>
    <collision>...</collision>
    <visual>...</visual>
    <collision>...</collision>
    <visual>...</visual>
    <collision>...</collision>
    <visual>...</visual>
    <collision>...</collision>
    <visual>...</visual>
    <collision>...</collision>
    <!-- link_0 内只有一个 inertial（由多个面板惯性合并得到） -->
    <inertial>
      <mass value="0.5185010114709838"/>
      <inertia ixx="0.00292..." ixy="0.0" ixz="..." iyy="0.00447..." iyz="0.0" izz="0.00648..."/>
      <origin xyz="3.0e-19 0.00449... -0.01717..."/>
    </inertial>
</link>
```

**补充说明（避免误解）**：
- `mailerbox_simple.urdf` 中有 3 个物理 link（`link_0/link_1/link_2`），因此一个文件里看到 **3 个 `<inertial>` block 是完全正确的**。
- 若在编辑器里直接搜索字符串 `inertial`，通常会命中 **6 次**：3 个 `<inertial>` + 3 个 `</inertial>`（开闭标签都会被匹配）。
- 本次修复的目标是：把 **`link_0` 内部的“多个 `<inertial>`”合并为 1 个**；`link_1/link_2` 本来就各只有 1 个资产，因此各自天然只有 1 个 `<inertial>`。

### 8.4 关键数学验证

以 seed 101 的 `link_0` 为例：

| 属性 | 修复前（只读第一个） | 修复后（正确合并） | 差异 |
|------|---------------------|-------------------|------|
| 质量 | ~0.088 kg | **0.518 kg** | 约 6 倍 |
| 质心 Y | -0.085 m | **0.0045 m** | 位置完全不同 |

修复后的值是使用平行轴定理正确合并 5 个面板的惯性后得到的。

---

**报告结束**

*修复日期: 2026-01-27*
*如有疑问，请联系技术分析团队。*
