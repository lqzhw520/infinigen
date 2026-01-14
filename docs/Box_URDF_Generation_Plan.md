# 程序化生成动态铰链3D盒子资产方案

## 文档目的

本文档详细描述如何基于 Infinigen 框架程序化生成符合 `docs/boxes-style.jpg` 中 16 种样式的**动态铰链3D盒子资产**，输出为 **URDF 格式**，用于机器人仿真系统。

---

## 当前实现状态（2026-01-12，已在本仓库环境验证）

> [!IMPORTANT]
> 本文档的关键结论已与当前代码对齐，并以 `seed=42` 的 **TuckEndBox（#1 双插盒）** 在在线 URDF 查看器中验证通过。

### 已跑通的盒型：#1 双插盒（TuckEndBox）

- **输出路径**: `sim_exports/urdf/tuckendbox/<seed>/`
- **关节规模**: 8 个可动铰链（6 个折页 + 2 个插舌二段铰链）+ 1 个 `world_joint`
- **已验证能力**:
  - 折页不悬浮（坐标系/导出偏移已修复）
  - 6 个折页均可向内折
  - 插舌为独立铰链，可在“合口后”下插（更贴近 `boxes-style.jpg` #1）
  - “推荐折叠顺序 + 最终闭合态”无穿插（允许接触级别贴合）

### 新增需求（当前任务）：#7 飞机盒简化版（MailerBox-simple，2-DOF）

> 目标样式参考：`docs/images/MailerBox-simple.jpg` / `docs/images/MailerBox-simple-1.jpg`
>
> 与 `boxes-style.jpg` 中“飞机盒”概念一致（属于 **1-2 DOF** 的低复杂度盒型），但本需求明确要求：
> - **只做 2 个折页铰链自由度** 的可动盒子（不扩展到其它 16 种盒型）
> - 结构极简：**一个大折页（盖） + 一个与盖相连的小折页（前片）**

**几何/拓扑定义（按本需求口径）**

- **盒体（固定 link）**：
  - 底板 + 四周墙（薄壁）
- **大折页（lid，可动 link）**：
  - 与盒体背面上沿铰接
  - 折下后应覆盖盒体开口（基本覆盖与盒体主体同等的开口尺寸）
- **小折页（front flap，可动 link）**：
  - 与大折页前沿铰接
  - 折下后覆盖盒体高度，底边与盒体底部对齐（即 flap 长度 ≈ Height）

**关节定义（只允许 2 个 hinge）**

- `mailer_lid`：`body -> lid`
  - hinge 位置：背面上沿折痕处（实现中会随 `EdgeExtension` 对盒体墙整体做 \(+E/2\) 的 Y 平移；并对厚度做一个小的校准偏移以消除“盖子与盒体之间的可见缝隙”）
  - axis：\((1,0,0)\)
  - 运动范围：\(-\pi \rightarrow +\pi\)（支持向内/向外折叠）
- `mailer_front_flap`：`lid -> front_flap`
  - hinge 位置：位于 lid 前沿（在“展开态” lid 竖直上翻时，前沿位于 \((0,\ +D/2,\ +H/2 + D)\)）
  - axis：\((1,0,0)\)
  - 运动范围：\(-\pi \rightarrow +\pi\)（支持向内/向外折叠）

**新增可随机参数：边缘凸出（EdgeExtension，同步随机）**

为满足 `MailerBox-simple-1` 样式中“底部三侧凸出 + 折页侧边凸出便于抓取”的需求，引入一个同步随机参数：

- `EdgeExtension = E`：凸出长度（米）
  - 采样：\(E \sim \mathcal{U}(0.05, 0.15) \cdot \min(W, D)\)
  - 同步作用于：
    - 盒体底板（bottom plate）：左右 +E（总宽 \(W+2E\)），前侧形成 +E 的“相对凸出”
      - 实现方式：底板深度取 \(D+E\) 且保持中心对齐；四周墙整体向 \(+Y\) 平移 \(E/2\)，使得凸出只发生在 \(-Y\)（前侧）方向，同时保持背面边缘与折痕/盖子对齐
    - 盒体背墙（与 lid 铰接的固定页）：左右 +E（总宽 \(W+2E\)），保证与 lid/底板在侧边外轮廓连续（避免“盖子变宽但背墙不变宽”造成的视觉缝隙）
    - 第一折页 lid：左右 +E（总宽 \(W+2E\)）
    - 第二折页 front flap：左右 +E（与 lid 宽度完全对齐）

**新增可随机参数：第二折页长度（FrontFlapLen，仅在需求 5 启用）**

- `FrontFlapLen = Lf`：第二折页（front flap）长度（沿 Z）
  - 默认：\(Lf = H\)（与盒体高度对齐，折下到达底部）
  - 随机（需求 5）：\(Lf = s \cdot H,\ s \sim \mathcal{U}(0.6, 1.0)\)（允许“不到底”）

对应实现位置：
- `infinigen/assets/sim_objects/modular_box_factory.py::MailerBoxFactory.sample_parameters()` 负责采样 `EdgeExtension/FrontFlapLen`
- `MailerBoxFactory.create_geometry_nodegroup()` 使用上述输入生成凸出几何，并保持原有 2-DOF 铰链结构不变

**对比导出（10 个 URDF，用于在线查看）**

- 脚本：`scripts/export_mailerbox_simple_variants.py`
- 设计目的：避免把“盒体尺寸随机”混入对比，脚本会 **固定 W/D/H/T**，然后：
  - 前 5 个：仅改变 `EdgeExtension`（`FrontFlapLen = H`）
  - 后 5 个：改变 `EdgeExtension` + 改变 `FrontFlapLen`（第二折页可不到底）

> 备注：为了让在线查看对比更直观，脚本当前把 `EdgeExtension` 设置为一组“固定但差异很大”的数值（例如 1cm~15cm），避免“随机值太接近看不出来”。

**推荐闭合态（用于在线查看器验证）**

- `mailer_lid = π/2`（大折页折下）
- `mailer_front_flap = π/2`（小折页折下覆盖前壁）

**实现文件（本需求只新增这些，不触碰其它盒型）**

- `infinigen/assets/sim_objects/modular_box_factory.py`
  - 新增 `MailerBoxFactory`（注册为 `BoxType.MAILER`）
  - 采用与 `TuckEndBoxFactory` 相同的 `nodegroup_hinge_joint` 元数据注入方式，确保 URDF 导出稳定
- `scripts/export_mailerbox_simple_urdf.py`
  - 一键导出用于在线 URDF viewer 的资产（默认 seed=42）
  - 输出路径：`sim_exports/urdf/mailerbox_simple/<seed>/`

### 关键实现文件（已实装）

- `infinigen/assets/sim_objects/modular_box_factory.py`: `TuckEndBoxFactory` 几何 + 关节（含 `top_tuck_tab`/`bottom_tuck_tab` 二段铰链、尺寸/层叠策略）
- `infinigen/core/sim/exporters/urdf_exporter.py`: 多关节导出（R6 已修复）、visual origin 计算、可选 `INFINIGEN_URDF_DEBUG`
- `infinigen/core/sim/physics/thin_shell_inertia.py`: 薄壳惯性修正（R-Deep-1 已修复）
- `scripts/export_tuckendbox_urdf.py`: 在 Blender 环境导出并用于在线查看/验证

### 重要工程经验（踩坑总结，后续盒型必须遵守）

- **Hinge-local 建模规则**: `nodegroup_hinge_joint` 会以 `Position` 作为枢轴装配 child，因此 child 几何必须在 **joint-local frame** 下建模；否则 URDF 导出时会出现 joint origin 与 visual origin 的重复平移（折页“悬浮”/错位）。
- **无碰撞约束的 viewer 限制**: 在线 URDF 查看器通常不做物理碰撞求解，因此“任强调节关节角度”会出现几何穿插。正确做法是用程序化碰撞检测校验关键姿态，并在 UI 侧遵循折叠顺序。

---

## 目录

1. [核心需求分析](#1-核心需求分析)
   - 1.1 目标盒型分类
   - 1.2 物理约束要求
2. [Infinigen 现有代码分析](#2-infinigen-现有代码分析) ✅ v5.0 深度审查
   - 2.1 现有 Box 资产分析
   - 2.2 关节系统分析
   - 2.3 URDF 导出器分析
   - 2.4 运动学编译器分析
   - 2.5 材质物理系统分析 (新增)
   - 2.6 关节动力学系统分析 (新增)
   - 2.7 代码分析总结
3. [16种盒型的技术实现方案](#3-16种盒型的技术实现方案)
   - 3.1 方案概览与核心架构
   - 3.2 各盒型详细实现方案
4. [实现路径与验证方法](#4-实现路径与验证方法)
   - 4.0 开发环境准备 (新增)
   - 4.1 实现路径（Phase 0-2）
   - 4.2 程序化验证方法
   - 4.3 关键技术点
5. [URDF 输出规范](#5-urdf-输出规范)
   - 5.1 文件结构
   - 5.2 URDF 示例
   - 5.3 元数据规范
6. [技术风险详细解决方案](#6-技术风险详细解决方案)
   - 6.1 风险总览与解决方案索引
   - 6.2 R1: 几何穿透
   - 6.3 R2: 关节位置不精确
   - 6.4 R3: 多关节联动复杂
   - 6.5 R4: 物理参数不准确
   - 6.6-6.11: R5-R8, R-Deep-1/2 扩展风险
7. [材质多样化系统](#7-材质多样化系统)
   - 7.1 材质系统架构
8. [批量生成系统](#8-批量生成系统)
   - 8.1 批量生成架构
9. [后续扩展方向](#9-后续扩展方向)
10. [验证检查点清单](#10-验证检查点清单)
    - 10.1 单个资产检查
    - 10.2 批量生成检查
    - 10.3 验证报告示例
11. [参考资源](#11-参考资源)
12. [详细开发工作计划（PM 可追溯版）](#12-详细开发工作计划pm-可追溯版) 🆕
    - 12.1 开发里程碑总览
    - 12.2 Phase 0: 稳定性基础
    - 12.3 Phase 1: 核心框架
    - 12.4 Phase 2.1: 基础盒型实现
    - 12.5 可追溯性与回滚策略
    - 12.6 PM 每日/每周检查表

---

## 1. 核心需求分析

### 1.1 目标盒型分类（来自 boxes-style.jpg）

根据图片，16种盒型可分为以下几类：

| 编号 | 中文名称 | 英文标识 | 关节类型 | 关节数量 | 复杂度 |
|-----|---------|---------|---------|---------|--------|
| 1 | 双插盒 | TuckEndBox | 铰链 (Hinge) | 6-8 (含插舌二段) | ⭐⭐ |
| 2 | 锁底盒 | LockBottomBox | 铰链 | 4-6 | ⭐⭐⭐ |
| 3 | 带保险双插盒 | TuckEndWithSafety | 铰链 | 3-5 | ⭐⭐⭐ |
| 4 | 带保险锁底盒 | LockBottomWithSafety | 铰链 | 5-7 | ⭐⭐⭐⭐ |
| 5 | 带挂钩锁底盒 | HookLockBottomBox | 铰链 | 5-7 | ⭐⭐⭐⭐ |
| 6 | 带挂钩双插盒 | HookTuckEndBox | 铰链 | 3-5 | ⭐⭐⭐ |
| 7 | 飞机盒 | MailerBox | 铰链 | 1-2 | ⭐⭐ |
| 8 | 抽屉盒 | DrawerBox | 滑轨 (Slide) | 1-2 | ⭐⭐ |
| 9 | 天地盒 | SlipLidBox | 无铰链/滑轨 | 0-1 | ⭐ |
| 10 | 自带手提礼盒 | GiftBoxWithHandle | 铰链 | 1-2 | ⭐⭐ |
| 11 | 对插手提礼盒 | TuckHandleBox | 铰链 | 2-3 | ⭐⭐⭐ |
| 12 | 屋脊手提礼盒 | GableTopBox | 铰链 | 2-4 | ⭐⭐⭐ |
| 13 | 双盖手提礼盒 | DoubleLidGiftBox | 铰链 | 2 | ⭐⭐ |
| 14 | 塑料手提礼盒 | PlasticHandleBox | 铰链 | 1-2 | ⭐⭐ |
| 15 | 平口箱 | RSCBox (Regular Slotted Container) | 铰链 | 4-8 | ⭐⭐⭐⭐ |
| 16 | 自定义盒型 | CustomBox | 可配置 | 可配置 | ⭐⭐⭐⭐⭐ |

### 1.2 物理约束要求

为确保生成的资产符合物理规律，必须满足：

1. **几何约束**：
   - 盒壁不能互相穿透
   - 关节旋转/滑动范围不能导致几何碰撞
   - 折叠部件厚度需要正确计算

2. **运动学约束**：
   - 铰链轴位置必须精确定位在折痕处
   - 旋转角度范围需考虑材料物理限制
   - 多关节联动需要正确的父子关系

3. **材料约束**：
   - 卡纸/瓦楞纸厚度限制折叠角度
   - 质量和惯性矩需要正确计算

---

## 2. Infinigen 现有代码分析

> [!NOTE]
> **代码审查版本**: v5.0 (2026-01-10)
> **审查结论**: 完整覆盖 Section 4 实现路径和 Section 6 技术风险所需的代码分析

### 2.1 现有 Box 资产 (`box.py`) 分析

**文件位置**: `infinigen/assets/sim_objects/box.py` (6162 行)

**架构特点**：
- 采用**单一巨型几何节点树**设计 (`nodegroup_geometry_nodes`)
- 通过 `version` 参数 + `Switch` 节点控制不同盒型
- 包含约 30+ 子节点组函数，高度耦合

```python
# 核心类结构 (简化)
@gin.configurable
class BoxFactory(AssetFactory):
    def sample_parameters(self):
        version = randint(0, 7)  # 7种变体
        w = uniform(0.2, 1)      # 宽度范围 (米)
        d = uniform(0.2, 1)      # 深度范围 (米)
        h = uniform(0.2, 1)      # 高度范围 (米)
        thickness = uniform(0.01, 0.04) * min([w, d, h])  # 壁厚 = 相对比例
```

**现有关节系统**：
- 使用 `nodegroup_hinge_joint` 实现铰链
- 支持多层级关节嵌套
- 已实现的关节标签：
  - `F1_top_lid` / `F1_bottom_lid` 顶部/底部盖子
  - `F1234_folding_edge` 四面折叠边缘
  - `F12_folding_edge` / `F34_folding_edge` 对侧面间折叠

**现有 Version 映射** (代码验证):
| Version | 类型 | 关节数 | 描述 |
|---------|------|--------|------|
| 0 | 基础双插盒 | 2 | 上下翻盖 |
| 1 | 单盖盒 | 1 | 单侧翻盖 |
| 2 | 锁底变体 | 2-4 | 底部锁定机构 |
| 3 | 带侧翼盒 | 4 | 侧翼加强结构 |
| 4 | 带插舌盒 | 2-3 | 插舌锁定 |
| 5 | 飞机盒变体 | 1-2 | 翻盖式 |
| 6 | 复杂折叠盒 | 4-6 | 多层折叠 |

> [!WARNING]
> **架构风险 (R-Deep-2)**: 现有单一节点树设计无法扩展到 16 种盒型。
> 需要重构为 `ModularBoxFactory` 动态加载架构。

### 2.2 关节系统分析 (`joints.py`)

**文件位置**: `infinigen/assets/utils/joints.py` (1927 行)

**铰链关节 (Hinge Joint)** - 代码确认的完整输入参数:
```python
@node_utils.to_nodegroup("nodegroup_hinge_joint", singleton=False, type="GeometryNodeTree")
def nodegroup_hinge_joint(nw: NodeWrangler):
    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketString", "Joint ID (do not set)", ""),   # 系统自动设置
            ("NodeSocketString", "Joint Label", ""),             # 用户可设置的标签
            ("NodeSocketGeometry", "Parent", None),              # 父几何体
            ("NodeSocketGeometry", "Child", None),               # 子几何体
            ("NodeSocketVector", "Position", (0, 0, 0)),         # 铰链位置 (相对于父体)
            ("NodeSocketVector", "Axis", (0, 0, 1)),             # 旋转轴 (默认 Z 轴)
            ("NodeSocketFloat", "Value", 0.0),                   # 当前角度 (弧度)
            ("NodeSocketFloat", "Min", 0.0),                     # 最小角度 (弧度)
            ("NodeSocketFloat", "Max", 0.0),                     # 最大角度 (弧度)
            ("NodeSocketBool", "Show Joint", False),             # 调试用可视化
        ],
    )
```

**滑轨关节 (Sliding Joint)** - 输入参数与铰链关节一致:
- `Position`: 滑动起点位置
- `Axis`: 滑动方向向量
- `Value`: 当前滑动距离 (米)
- `Min/Max`: 滑动范围 (米)

**关节输出处理**:
- 自动存储 `posparent_{joint_id}`, `poschild_{joint_id}` 位置属性
- 自动存储 `axis_{joint_id}`, `min_{joint_id}`, `max_{joint_id}` 关节属性
- 通过 `part_id` 属性区分不同刚体部件

### 2.3 URDF 导出器分析 (`urdf_exporter.py`)

**文件位置**: `infinigen/core/sim/exporters/urdf_exporter.py` (442 行)

**导出流程**:
```
Blender Object + Kinematic Blueprint
         ↓
    URDFBuilder.build()
         ↓
    _construct_rigid_body_skeleton()  ← 从运动学树构建刚体骨架
         ↓
    _populate_links()  ← 遍历骨架填充 URDF links 和 joints
         ↓
    _create_joint()  ← 创建单个 URDF 关节元素
         ↓
    export_sim_ready()  ← 导出 OBJ 网格文件
         ↓
    save()  ← 保存 URDF XML 文件
```

**关节类型映射** (urdf_exporter.py:302-309):
```python
if joint_type == JointType.HINGE:
    jt = "revolute"      # URDF 旋转关节
elif joint_type == JointType.SLIDING:
    jt = "prismatic"     # URDF 移动关节
elif joint_type == JointType.WELD or joint_type == JointType.NONE:
    jt = "fixed"         # 固定关节
```

**惯性计算逻辑**（✅ 已修复 R-Deep-1，当前实现为稳健薄壳惯性）:

```python
# urdf_exporter.py: 使用薄壳稳健惯性（避免 vol≈0 导致 mass≈0）
# （示意：与当前实现一致）
robust_mass, I_tensor, com = thinshell.calculate_robust_inertia(
    vertices=vertices,
    faces=faces,
    density=mat_physics["density"],
    volume=vol,
)
```

> [!NOTE]
> **实现要点（与当前代码一致）**：
> - URDF 导出器对 mesh 做三角化与体积估计后，会调用 `thin_shell_inertia.calculate_robust_inertia(...)`
> - 对薄壳（卡纸面板等）使用“表面积×有效厚度”的等效质量与惯性估算，并强制惯性张量正定、质量最小值

> [!NOTE]
> **R6 多关节限制已修复（已实装）**：`URDFBuilder._populate_links()` 现支持 `len(joint_nodes) > 1`，通过创建 “intermediate link” 串联多个 joint（保证 URDF 有效并兼容多关节联动盒型）。

> [!NOTE]
> **R-Deep-1 薄壳惯性已修复（已实装）**：实现位于 `infinigen/core/sim/physics/thin_shell_inertia.py`，并在 URDF 导出时调用，避免 `mass≈0` 与非正定惯性导致的仿真爆炸/加载失败。

> [!IMPORTANT]
> **visual origin 语义（本次踩坑根因之一）**：`export_sim_ready()` 导出 OBJ 时会对网格做 `translation=-geometry_center`（将 OBJ 顶点平移到自身 AABB center），导出器随后用 `geom_center - pos_offset` 写入 `<visual><origin>`。这要求几何节点生成时 child 处于 **joint-local frame**，否则会出现 joint origin 与 visual origin 的重复平移（折页“悬浮”/错位）。

### 2.4 运动学编译器分析 (`kinematic_compiler.py`)

**文件位置**: `infinigen/core/sim/kinematic_compiler.py` (317 行)

**编译流程**:
```python
def compile(obj: bpy.types.Object) -> Dict:
    # 1. 获取几何节点修改器
    mods = [mod for mod in obj.modifiers if mod.type == "NODES"]
    
    # 2. 构建几何连接图 (仅追踪 GEOMETRY socket)
    geo_graph = get_geometry_graph(mods)
    
    # 3. 深度优先遍历构建运动学树
    def build_kinematic_graph(blend_node) -> KinematicNode:
        # 识别关节类型 (hinge/sliding/weld)
        # 设置 joint_id
        # 递归处理子节点
        # 返回 KinematicNode
    
    root = build_kinematic_graph(output_node)
    
    # 4. 返回运动学信息
    return {
        "graph": root.get_graph(),
        "metadata": {...},  # 包含关节标签
        "labels": labels    # 部件语义标签
    }
```

**关节识别逻辑** (kinematic_compiler.py:191-213):
```python
if utils.is_hinge(node):
    kinematic_type = KinematicType.JOINT
    joint_type = JointType.HINGE
elif utils.is_sliding(node):
    kinematic_type = KinematicType.JOINT
    joint_type = JointType.SLIDING
```

### 2.5 材质物理系统分析 (`material_physics.py` & `material_definitions.py`)

**文件位置**:
- `infinigen/core/sim/physics/material_physics.py`
- `infinigen/core/sim/physics/material_definitions.py`

> [!NOTE]
> **状态更新（v6.0）**：`material_definitions.py` 已包含 `cardboard` / `corrugated` 及其变体（例如 `cardboard_kraft`, `corrugated_b` 等），可直接用于密度/摩擦采样。

**材质属性获取流程**:
```python
def get_material_properties(obj: bpy.types.Object) -> Dict:
    # 从 Blender 对象获取材质名称
    material_name = obj.data.materials[...].name
    # 查找材质注册表
    return sample_mat_physics(material_name)  # 返回 {friction, density}
```

### 2.6 关节动力学系统分析 (`joint_dynamics.py`)

**文件位置**: `infinigen/core/sim/physics/joint_dynamics.py`

**导出器如何使用（与当前代码一致）**：

- URDF 导出阶段会调用 `jointdyna.get_joint_properties(joint_name, joint_params)` 获取 `damping/friction`
- `joint_params` 由导出入口的 `sample_joint_params_fn` 提供（例如 `TuckEndBoxFactory.sample_joint_parameters`）

```python
# urdf_exporter.py（示意）
joint_properties = jointdyna.get_joint_properties(joint_name, joint_params)
...
<dynamics damping="..." friction="..."/>
```

**状态更新（v6.0）**：

- 模块已扩展并包含材质/盒型预设（例如 `MATERIAL_JOINT_PRESETS`, `BOX_TYPE_JOINT_PRESETS`）
- 当前导出器仍保持旧接口兼容（`get_joint_properties` 优先使用显式传入的 `joint_params`）
- 后续可选增强：当未传 `joint_params` 时，改用 `get_box_type_joint_dynamics(...)` 做默认值回退（避免全 0 动力学）

### 2.7 代码分析总结 - 对 Section 4/6 的覆盖

| 需求 | 相关代码 | 分析状态 | 影响 Section |
|------|---------|----------|-------------|
| 多关节导出 | urdf_exporter.py (`URDFBuilder._populate_links`) | ✅ 已修复并回归 | 4.1 Phase 0 |
| 薄壳惯性 | thin_shell_inertia.py + urdf_exporter.py 集成 | ✅ 已修复并回归 | 4.1 Phase 0, 6.10 |
| 关节参数传递 | joints.py | ✅ 参数完整 | 3.2, 4.2 |
| 材质系统 | material_definitions.py | ✅ Cardboard/Corrugated 已具备 | 7.1, Phase 1.4 |
| 动力学参数 | joint_dynamics.py | ✅ 已扩展（材质/盒型预设） | 6.9 |
| 节点树架构 | box.py | ✅ 确认需重构 | 6.11 |
| 运动学编译 | kinematic_compiler.py | ✅ 流程清晰 | 4.1 |

---

## 3. 16种盒型的技术实现方案

### 3.1 方案概览与核心架构

采用**模块化节点树加载 + 关节注入 + 材质系统**的方式实现：

> [!IMPORTANT]
> **架构决策 (来自 R-Deep-2 风险评估)**：  
> 不采用单一巨型节点树设计，而是为每种盒型族创建独立的节点生成函数，通过 `ModularBoxFactory` 动态加载。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ModularBoxFactory                               │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                     动态节点树加载器                                  │ │
│  │  BOX_TYPE_GENERATORS = {                                            │ │
│  │    "TuckEndBox": geometry_nodes_tuck_box,                           │ │
│  │    "MailerBox": geometry_nodes_mailer_box,                          │ │
│  │    "DrawerBox": geometry_nodes_drawer_box,                          │ │
│  │    "RSCBox": geometry_nodes_rsc_box, ...                            │ │
│  │  }                                                                   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                  │                                       │
│                                  ▼                                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
│  │ 底部模块   │  │ 侧面模块   │  │ 顶部模块   │  │  材质系统          │ │
│  │ BottomModule│ │ SideModule │  │ TopModule  │  │  MaterialSystem    │ │
│  │            │  │            │  │            │  │  - CardboardMaterial│ │
│  │            │  │            │  │            │  │  - CorrugatedMaterial│
│  │            │  │            │  │            │  │  - PlasticMaterial  │ │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────────┬──────────┘ │
│        │               │               │                   │             │
│        └───────────────┴───────────────┴───────────────────┘             │
│                                  │                                       │
│                                  ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                 关节注入系统 (Joint Injector)                        │ │
│  │  - hinge_joint: 铰链关节 (统一轴向，通过 Min/Max 控制方向)           │ │
│  │  - sliding_joint: 滑轨关节                                          │ │
│  │  - ✅ 材质相关关节动力学预设（joint_dynamics.py / R8）              │ │
│  └──────────────────────┬─────────────────────────────────────────────┘ │
│                         │                                               │
│                         ▼                                               │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                  运动学编译器 (Kinematic Compiler)                   │ │
│  │  - ✅ 薄壳稳健惯性（thin_shell_inertia.py / R-Deep-1）              │ │
│  │  - 运动学树构建 + 循环依赖检测                                       │ │
│  └──────────────────────┬─────────────────────────────────────────────┘ │
│                         │                                               │
│                         ▼                                               │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    URDF 导出器 (URDF Exporter)                       │ │
│  │  - ✅ 多关节导出支持（R6 已修复：intermediate links）               │ │
│  │  - 碰撞网格简化 / Box Primitive 输出 (R7)                            │ │
│  │  - 关节软限制和动力学参数输出                                        │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

**关键技术点 (已集成风险解决方案)**：

| 模块 | 集成的风险解决方案 | 详见章节 |
|------|-------------------|---------|
| ModularBoxFactory | R-Deep-2 动态节点树加载 | 6.11 |
| MaterialSystem | 材料物理属性数据库 | 7.1 |
| Joint Injector | R8 关节动力学预设, 统一轴向规范 | 6.9 |
| Kinematic Compiler | R-Deep-1 薄壳惯性修正 | 6.10 |
| URDF Exporter | R6 多关节扩展, R7 碰撞网格优化 | 6.7, 6.8 |

### 3.2 各盒型详细实现方案

#### 盒型 1: 双插盒 (TuckEndBox)

> [!IMPORTANT]
> **状态（v6.0 / 2026-01-12）**：该盒型已在本仓库环境中导出 URDF，并在在线 URDF 查看器中验证：
> - 6 个折页均可向内折叠且不悬浮
> - 插舌为独立铰链，可在“合口后”下插
> - 按推荐折叠顺序达到最终闭合态无穿插（允许接触）

**对齐目标**：`docs/boxes-style.jpg` #1（双插盒，含左右 dust flaps；顶/底插舌可“二段折叠”后插入盒口内侧）。

**结构分解（刚体/关节）**：

- **盒体（root）**：`link_0`（四面薄壁直筒，无上下盖）
- **顶部机构（4 个关节）**：
  - `top_dust_left` / `top_dust_right`：左右 dust flap（轴约为 ±Y）
  - `top_back_flap`：顶部主翻盖（轴约为 X）
  - `top_tuck_tab`：顶部插舌（二段铰链；父为 `top_back_flap`，用于“合口后下插”）
- **底部机构（4 个关节）**：
  - `bottom_dust_left` / `bottom_dust_right`：左右 dust flap（注意底部轴向与顶部相反才能向内折入）
  - `bottom_front_flap`：底部主翻盖（轴约为 X）
  - `bottom_tuck_tab`：底部插舌（二段铰链；父为 `bottom_front_flap`）

**关节数量说明（文档统一口径）**：

- **2**：仅主翻盖（不含 dust flaps / 不含插舌二段）
- **6**：主翻盖 + dust flaps（插舌刚性并入主翻盖）
- **8（当前实现）**：主翻盖 + dust flaps + 插舌二段铰链（更贴近真实双插盒“插入盒口”动作）

**关键几何参数（当前实现的有效默认值）**：

- `major_flap_len = Depth - Thickness`（确保主翻盖能覆盖盒口）
- `dust_flap_len = 0.35 * Width`（避免挡住插舌下插）
- `tab_len = 0.20 * Depth`
- `tab_width = 0.28 * Width`
- **层叠策略**：主翻盖在最终闭合态相对 dust flaps 做 `±Thickness/2` 的微小 Y 偏移，使“厚度堆叠”而非“厚度互穿”

**折叠顺序（在无碰撞约束的 viewer 中必须遵守）**：

1. 合左右 dust flaps
2. 合主翻盖（形成盒口矩形开口）
3. 下插 tuck tab（插舌二段折叠）

**实现要点（避免折页悬浮/错位）**：

- `nodegroup_hinge_joint` 会以 `Position` 作为枢轴装配 child，因此 child 几何必须在 **joint-local frame** 下建模，否则 URDF 会出现 joint origin 与 visual origin 的重复平移（折页悬浮）。

**实现位置**：

- 几何与关节：`infinigen/assets/sim_objects/modular_box_factory.py`（`TuckEndBoxFactory.create_geometry_nodegroup` / `get_default_joints`）
- 导出脚本：`scripts/export_tuckendbox_urdf.py`

---

#### 盒型 2: 锁底盒 (LockBottomBox)

**结构分解**:
```
  顶部: 同双插盒
  
  底部锁定机构:
      ┌──────────────────┐
      │    底面板 base   │ (固定)
      └──────────────────┘
           ┌────┐
    ┌──────┤锁舌├──────┐
    │      └────┘      │
    │  底侧翼A   底侧翼B │
    │  (hinge)  (hinge) │
    └──────────────────┘
```

**关节配置**:
```python
joints = [
    {"name": "top_lid", "type": "hinge", ...},
    {"name": "bottom_flap_A", "type": "hinge", "axis": (0, 1, 0),
     "range": (0°, 90°)},
    {"name": "bottom_flap_B", "type": "hinge", "axis": (0, -1, 0),
     "range": (0°, 90°)},
    {"name": "lock_tab", "type": "hinge", "axis": (1, 0, 0),
     "range": (0°, 90°)},
]
```

---

#### 盒型 7: 飞机盒 (MailerBox)

**结构分解**:
```
       ┌─────────────────────┐
       │      翻盖主体       │ ← hinge_joint (main_lid)
       ├─────────────────────┤
  ┌────┤                     ├────┐
  │翼A │     盒体底部        │翼B │
  └────┤                     ├────┘
       └─────────────────────┘
```

**关节配置**:
```python
joints = [
    {"name": "main_lid", "type": "hinge", "axis": (0, 1, 0),
     "range": (0°, 180°), "position": "back_edge"},
]
```

**物理特点**:
- 单铰链设计
- 翻盖可完全打开
- 侧翼与翻盖联动

---

#### 盒型 8: 抽屉盒 (DrawerBox)

**结构分解**:
```
┌────────────────────────────┐
│         外壳 (固定)        │
│  ┌──────────────────────┐  │
│  │                      │  │
│  │    抽屉 (滑动)       │← sliding_joint
│  │                      │  │
│  └──────────────────────┘  │
└────────────────────────────┘
```

**关节配置**:
```python
joints = [
    {"name": "drawer", "type": "slide", "axis": (0, 1, 0),
     "range": (0, drawer_depth * 0.9), "position": "center"},
]
```

---

#### 盒型 9: 天地盒 (SlipLidBox)

**结构分解**:
```
  ┌─────────────────┐
  │                 │
  │    天盖 (上)    │ ← 无铰链，滑动套合
  │                 │
  ├─────────────────┤
  │                 │
  │    地盒 (下)    │ (固定)
  │                 │
  └─────────────────┘
```

**关节配置** (可选滑轨):
```python
# 方案A: 纯滑动（无关节）
joints = []

# 方案B: 带滑轨关节
joints = [
    {"name": "lid", "type": "slide", "axis": (0, 0, 1),
     "range": (0, box_height), "position": "center"},
]
```

---

#### 盒型 15: 平口箱 / RSC箱 (Regular Slotted Container)

**结构分解**:
```
       ┌────────顶部小翼B────────┐
       │                        │
  ┌────┤     顶部大翼A (盖)     ├────┐
  │    └────────────────────────┘    │
  │  ↑                            ↑  │
  │ hinge                      hinge │
  │                                  │
  ├──────────────────────────────────┤
  │           侧面F1                 │
  ├──────────────────────────────────┤
  │           侧面F2                 │
  ├──────────────────────────────────┤
  │           侧面F3                 │
  ├──────────────────────────────────┤
  │           侧面F4                 │
  ├──────────────────────────────────┤
  │                                  │
  │ hinge                      hinge │
  │  ↓                            ↓  │
  │    ┌────────────────────────┐    │
  └────┤     底部大翼A          ├────┘
       │                        │
       └────────底部小翼B────────┘
```

**关节配置**:
```python
joints = [
    # 顶部四个翻盖 (注意：轴向统一，通过范围控制方向)
    {"name": "top_flap_1", "type": "hinge", "axis": (1, 0, 0), "range": (-180°, 0°)},
    {"name": "top_flap_2", "type": "hinge", "axis": (1, 0, 0), "range": (0°, 180°)},  # 相同轴，相反范围
    {"name": "top_flap_3", "type": "hinge", "axis": (0, 1, 0), "range": (-180°, 0°)},
    {"name": "top_flap_4", "type": "hinge", "axis": (0, 1, 0), "range": (0°, 180°)},  # 相同轴，相反范围
    # 底部四个翻盖
    {"name": "bottom_flap_1", "type": "hinge", "axis": (1, 0, 0), "range": (0°, 180°)},
    {"name": "bottom_flap_2", "type": "hinge", "axis": (1, 0, 0), "range": (-180°, 0°)},
    {"name": "bottom_flap_3", "type": "hinge", "axis": (0, 1, 0), "range": (0°, 180°)},
    {"name": "bottom_flap_4", "type": "hinge", "axis": (0, 1, 0), "range": (-180°, 0°)},
]
```

---

## 4. 实现路径与验证方法

### 4.0 开发环境准备（PM 前置条件）

> [!IMPORTANT]
> **开发环境必须在开始 Phase 0 之前配置完成**

#### 4.0.0 当前开发环境配置（2026-01-10 验证）

> **实际使用的容器环境配置**

| 配置项 | 实际值 | 状态 |
|--------|--------|------|
| **操作系统** | Ubuntu 20.04.6 LTS (Focal Fossa) | ✅ |
| **容器类型** | CCI 实例 | ✅ |
| **GPU** | NVIDIA A800-SXM4-80GB × 1 | ✅ |
| **CUDA 版本** | 12.8 | ✅ |
| **显存** | 80GB | ✅ |
| **vCPU** | 12 核 | ✅ |
| **内存** | 120 GiB | ✅ |
| **项目路径** | `/mnt/afs2/zhuhaowu/infinigen` | ✅ |
| **Conda 环境** | `infinigen` (已激活) | ✅ |
| **PyBullet** | 3.2.7 | ✅ 已安装 |
| **NumPy** | 1.26.4 | ✅ 已安装 |
| **trimesh** | 3.22.5 | ✅ 已安装 |
| **Blender (Linux)** | 4.2.0 | ✅ 已安装 (`blender/blender`) |
| **Infinigen** | 可导入 | ✅ |
| **URDF 导出器** | ✅ 支持多关节导出（R6 已修复）+ 薄壳惯性修正（R-Deep-1 已修复） | ✅ |

**环境自检（推荐命令）**：
- `./blender/blender --version`
- `python -c "import numpy, trimesh, pybullet; print('✅ deps ok')"`

#### 4.0.1 Python 环境

```bash
# 使用项目现有的 conda 环境
conda activate infinigen

# 确认已安装的依赖 (已验证通过)
pip list | grep -E "pybullet|trimesh|numpy"
# 预期输出:
# numpy          1.26.4
# pybullet       3.2.7
# trimesh        3.22.5
```

#### 4.0.2 Blender 环境 (R5 验证前置) ⚠️ 需要配置

项目目录中已有 macOS 版 Blender 4.2.0 (`Blender.app/`)，但在 **Ubuntu 容器** 中需要安装 **Linux 版本**：

```bash
# 在容器中执行 - 下载并安装 Blender 4.2.0 Linux 版
cd /mnt/afs2/zhuhaowu/infinigen

# 下载 (约 400MB)
wget https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz

# 解压到 blender/ 目录
tar -xf blender-4.2.0-linux-x64.tar.xz
mv blender-4.2.0-linux-x64 blender

# 验证安装
./blender/blender --version
# 预期输出: Blender 4.2

# 清理安装包 (可选)
rm blender-4.2.0-linux-x64.tar.xz

# 验证几何节点功能
./blender/blender --background --python-expr "
import bpy
print('Blender version:', bpy.app.version)
assert bpy.app.version >= (4, 0, 0), 'Need Blender 4.0+'
print('✅ Blender version OK')
"
```

> [!TIP]
> Infinigen 的 `launch_blender.py` 会自动检测以下路径：
> - `./blender/blender` (Linux)
> - `./Blender.app/Contents/MacOS/Blender` (macOS)

##### 4.0.2.1 关键问题：Blender Python 与 Conda 环境隔离 ⚠️

**问题说明**：Blender 使用内置的 Python 解释器（`./blender/4.2/python/bin/python3.11`），
与 `conda activate infinigen` 的 Python 环境**完全隔离**。这意味着：

| 运行方式 | 使用的 Python | 能访问 conda 包？ |
|---------|--------------|-----------------|
| `python script.py` | conda Python | ✅ 可以 |
| `./blender/blender --python script.py` | Blender Python | ❌ 不能 |

**解决方案**：使用修改后的 `launch_blender.py`，通过环境变量传递 conda 的 site-packages：

```bash
# 正确的 Blender 脚本运行命令
cd /mnt/afs2/zhuhaowu/infinigen

# 设置环境变量并启动 Blender
INFINIGEN_PYTHONPATH=$(python -c "import site; print(':'.join(site.getsitepackages()))") \
python -m infinigen.launch_blender -s tests/sim/test_tuckendbox_blender.py

# 或者先导出环境变量
export INFINIGEN_PYTHONPATH=$(python -c "import site; print(':'.join(site.getsitepackages()))")
python -m infinigen.launch_blender -s tests/sim/test_tuckendbox_blender.py
```

**验证 Blender 能访问 conda 包**：
```bash
INFINIGEN_PYTHONPATH=$(python -c "import site; print(':'.join(site.getsitepackages()))") \
python -m infinigen.launch_blender -s - <<< "
import trimesh
import numpy
import pybullet
print('✅ Blender can access conda packages')
"
```

> [!IMPORTANT]
> **所有需要在 Blender 中运行的测试脚本，必须使用以上命令格式！**
> 直接使用 `./blender/blender --python` 会报 `ModuleNotFoundError`。

#### 4.0.3 PyBullet 验证环境 (R-Deep-1 验证前置)

```bash
# 安装 PyBullet
pip install pybullet

# 验证 PyBullet 可以加载 URDF
python -c "
import pybullet as p
p.connect(p.DIRECT)
print('✅ PyBullet ready')
p.disconnect()
"
```

**PyBullet 薄壳惯性验证（推荐 pytest 回归）**：

```bash
pytest -q tests/sim/test_thin_shell_inertia.py
pytest -q tests/sim/test_tuckendbox_physics.py
```

**（可选）独立脚本模板**：如果需要不依赖 pytest 的单文件脚本，可参考以下模板（可放到 `scripts/validate_thin_shell_inertia.py`）：
```python
#!/usr/bin/env python3
"""验证薄壳惯性修正是否防止模拟爆炸"""
import pybullet as p
import time

def validate_thin_shell_urdf(urdf_path: str, max_steps: int = 1000) -> bool:
    """
    加载 URDF 并模拟若干步，检测是否发生位置爆炸
    
    Returns:
        True = 稳定, False = 爆炸
    """
    physics_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    
    try:
        robot_id = p.loadURDF(urdf_path, [0, 0, 0.5])
    except Exception as e:
        print(f"❌ URDF 加载失败: {e}")
        return False
    
    explosion_threshold = 100.0  # 如果位置超过 100m 视为爆炸
    
    for step in range(max_steps):
        p.stepSimulation()
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        
        if max(abs(p) for p in pos) > explosion_threshold:
            print(f"❌ 模拟爆炸 @ step {step}, position: {pos}")
            p.disconnect()
            return False
    
    p.disconnect()
    print(f"✅ 模拟稳定 ({max_steps} steps)")
    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python validate_thin_shell_inertia.py <urdf_path>")
        sys.exit(1)
    success = validate_thin_shell_urdf(sys.argv[1])
    sys.exit(0 if success else 1)
```

#### 4.0.4 验证通过标准

| 检查项 | 命令 | 预期输出 |
|--------|------|----------|
| Blender 版本 | `blender --version` | >= 4.0.0 |
| PyBullet 安装 | `python -c "import pybullet"` | 无错误 |
| trimesh 安装 | `python -c "import trimesh"` | 无错误 |
| numpy 版本 | `python -c "import numpy; print(numpy.__version__)"` | >= 1.20 |

---

### 4.1 实现路径（修订版 - 集成风险解决方案）

> [!NOTE]
> **Phase 0 已完成（v6.0）**：R6（URDF 多关节导出）与 R-Deep-1（薄壳惯性）已在当前代码中修复，并用于 `TuckEndBox` 导出与在线验证。
> 后续盒型开发默认依赖这些修复，建议保留/补齐回归脚本，避免回退。

```
Phase 0: 稳定性基础 (✅ 已完成 / 需回归)
│
├── 0.1 修复 URDF 导出器多关节限制 (R6)
│   ├── 扩展 urdf_exporter.py _populate_links() 方法
│   ├── 支持 len(joint_nodes) > 1 的情况
│   └── 回归建议: 导出含 multi-joint edge 的 URDF，确保可解析/可加载
│
├── 0.2 实现薄壳惯性修正 (R-Deep-1)
│   ├── 新建 calculate_robust_inertia() 函数
│   ├── 集成到 urdf_exporter.py 惯性计算流程
│   └── 回归建议: PyBullet DIRECT 模拟 1000 steps 稳定（薄板厚度极限用例）
│
├── 0.3 Blender 版本兼容性检查 (R5)
│   └── ✅ 验收标准: 启动时自动检查版本
│
└── 0.4 碰撞网格基础设施 (R7)
    ├── simplify_collision_mesh() 函数
    └── generate_box_primitive_collider() 函数
    └── ✅ 验收标准: 碰撞网格面数 < 500

Phase 1: 核心框架 + 材质系统 (1-2周)
│
├── 1.1 创建 ModularBoxFactory 基类 (集成 R-Deep-2)
│   ├── 继承 AssetFactory
│   ├── 实现 BOX_TYPE_GENERATORS 动态加载机制
│   ├── 实现通用参数采样接口
│   └── 集成材质系统接口
│
├── 1.2 实现基础几何模块
│   ├── create_base_panel()     # 底面板
│   ├── create_side_panel()     # 侧面板
│   ├── create_lid_panel()      # 盖板
│   └── create_flap()           # 翻盖
│
├── 1.3 实现关节注入系统 (集成 R8)
│   ├── add_hinge_at_edge()     # 边缘添加铰链
│   ├── add_slide_joint()       # 添加滑轨
│   ├── MATERIAL_JOINT_PRESETS / BOX_TYPE_JOINT_PRESETS  # 材质/盒型动力学预设
│   └── validate_joint_range()  # 验证关节范围
│
├── 1.4 材质多样化系统 (详见第7章)
│   ├── MaterialSystem          # 材质管理系统
│   ├── BaseMaterial            # 材质基类
│   │   ├── physical_properties # 物理属性 (密度/摩擦/恢复系数)
│   │   ├── visual_properties   # 视觉属性 (颜色/粗糙度)
│   │   └── thickness_range     # 厚度范围
│   ├── CardboardMaterial       # 卡纸材质 (4种颜色变体)
│   ├── CorrugatedMaterial      # 瓦楞纸 (A/B/C/E型)
│   ├── PlasticMaterial         # 塑料 (PP/PET/HDPE)
│   └── WoodMaterial            # 木质 (MDF/胶合板)
│
└── 1.5 批量生成系统 (详见第8章)
    ├── BatchBoxGenerator       # 批量生成器
    ├── ParameterSpaceSampler   # 参数空间采样器 (均匀/LHS/网格)
    ├── ValidationPipeline      # 验证管线
    └── ExportManager           # 导出管理器

Phase 2: 盒型实现 + 程序化验证 (2-4周)
│
├── 2.1 实现基础盒型 (优先级高)
│   ├── TuckEndBoxFactory       # 双插盒
│   ├── MailerBoxFactory        # 飞机盒
│   ├── DrawerBoxFactory        # 抽屉盒
│   └── RSCBoxFactory           # 平口箱
│
├── 2.2 实现锁定机构盒型
│   ├── LockBottomBoxFactory    # 锁底盒
│   └── SafetyTuckBoxFactory    # 带保险双插盒
│
├── 2.3 实现礼盒类型
│   ├── GableTopBoxFactory      # 屋脊手提盒
│   ├── HandleGiftBoxFactory    # 手提礼盒
│   └── SlipLidBoxFactory       # 天地盒
│
└── 2.4 程序化验证系统 (每个盒型必须通过)
    ├── GeometryValidator       # 几何验证器 (12项检查)
    ├── KinematicsValidator     # 运动学验证器
    ├── CollisionChecker        # 碰撞检测器
    └── PhysicalParameterCalculator  # 物理参数验证
```

### 4.2 程序化验证方法（无需外部仿真器）

#### 4.2.1 几何验证（纯代码实现）

```python
import numpy as np
from typing import Dict, List, Tuple
import trimesh

class GeometryValidator:
    """纯程序化几何验证器，不依赖任何外部仿真环境"""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.errors = []
        self.warnings = []
    
    def validate(self, urdf_path: str) -> Dict:
        """
        主验证入口
        
        Returns:
            {
                "passed": bool,
                "errors": List[str],
                "warnings": List[str],
                "metrics": Dict
            }
        """
        self.errors = []
        self.warnings = []
        
        # 1. 解析 URDF
        links, joints = self._parse_urdf(urdf_path)
        
        # 2. 加载所有网格
        meshes = self._load_meshes(links, urdf_path)
        
        # 3. 执行验证检查
        results = {
            # 基础网格检查
            "mesh_watertight": self._check_watertight(meshes),
            "mesh_volume_positive": self._check_positive_volume(meshes),
            "thickness_consistency": self._check_thickness(meshes),
            "joint_position_valid": self._check_joint_positions(joints, meshes),
            "no_initial_collision": self._check_initial_collision(meshes),
            
            # 新增检查项（根据 Review 补充）
            "urdf_syntax_valid": self._check_urdf_syntax(urdf_path),
            "joint_names_unique": self._check_joint_name_uniqueness(joints),
            "link_connectivity": self._check_link_connectivity(links, joints),
            "mesh_files_exist": self._check_mesh_files_exist(links, urdf_path),
            "units_consistent": self._check_unit_consistency(meshes),
            "mesh_manifold": self._check_manifold(meshes),
            "no_degenerate_faces": self._check_no_degenerate_faces(meshes),
        }
        
        return {
            "passed": all(results.values()) and len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": results
        }
    
    def _check_watertight(self, meshes: Dict[str, trimesh.Trimesh]) -> bool:
        """检查所有网格是否为闭合流形"""
        all_watertight = True
        for name, mesh in meshes.items():
            if not mesh.is_watertight:
                self.errors.append(f"网格 {name} 不是闭合流形")
                all_watertight = False
        return all_watertight
    
    def _check_positive_volume(self, meshes: Dict) -> bool:
        """检查所有网格是否有正体积（法线方向正确）"""
        all_positive = True
        for name, mesh in meshes.items():
            if mesh.volume <= 0:
                self.errors.append(f"网格 {name} 体积为负或零，法线方向可能错误")
                all_positive = False
        return all_positive
    
    def _check_thickness(self, meshes: Dict, expected_thickness: float = None) -> bool:
        """检查面板厚度一致性"""
        thicknesses = []
        for name, mesh in meshes.items():
            # 使用边界框最小维度估算厚度
            dims = mesh.bounding_box.extents
            min_dim = min(dims)
            thicknesses.append((name, min_dim))
        
        if expected_thickness:
            for name, t in thicknesses:
                if abs(t - expected_thickness) > self.tolerance:
                    self.warnings.append(f"网格 {name} 厚度 {t:.4f} 与预期 {expected_thickness:.4f} 不一致")
        
        return True  # 厚度检查仅产生警告，不阻断
    
    def _check_joint_positions(self, joints: List, meshes: Dict) -> bool:
        """检查关节位置是否在父部件边界上"""
        all_valid = True
        for joint in joints:
            parent_name = joint["parent"]
            origin = np.array(joint["origin"])
            
            if parent_name not in meshes:
                continue
            
            parent_mesh = meshes[parent_name]
            # 检查关节位置是否在父部件边界框附近
            bbox = parent_mesh.bounding_box
            distance = self._point_to_bbox_distance(origin, bbox)
            
            if distance > 0.01:  # 1cm 容差
                self.warnings.append(
                    f"关节 {joint['name']} 位置距离父部件 {parent_name} 边界 {distance:.4f}m"
                )
        
        return all_valid
    
    def _check_initial_collision(self, meshes: Dict) -> bool:
        """检查初始状态下是否有部件碰撞"""
        mesh_list = list(meshes.items())
        no_collision = True
        
        for i in range(len(mesh_list)):
            for j in range(i + 1, len(mesh_list)):
                name_a, mesh_a = mesh_list[i]
                name_b, mesh_b = mesh_list[j]
                
                # 使用 trimesh 碰撞管理器
                collision_mgr = trimesh.collision.CollisionManager()
                collision_mgr.add_object(name_a, mesh_a)
                is_collision, contacts = collision_mgr.in_collision_single(
                    mesh_b, return_data=True
                )
                
                if is_collision:
                    self.errors.append(f"初始状态下 {name_a} 与 {name_b} 发生碰撞")
                    no_collision = False
        
        return no_collision
    
    def _point_to_bbox_distance(self, point: np.ndarray, bbox) -> float:
        """计算点到边界框的距离"""
        mins = bbox.bounds[0]
        maxs = bbox.bounds[1]
        
        clamped = np.maximum(mins, np.minimum(point, maxs))
        return np.linalg.norm(point - clamped)
    
    # ========== 新增验证方法（根据 Review 补充）==========
    
    def _check_urdf_syntax(self, urdf_path: str) -> bool:
        """验证 URDF XML 语法合规性"""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            if root.tag != 'robot':
                self.errors.append("URDF 根元素必须是 <robot>")
                return False
            return True
        except ET.ParseError as e:
            self.errors.append(f"URDF XML 语法错误: {e}")
            return False
    
    def _check_joint_name_uniqueness(self, joints: List) -> bool:
        """验证关节名称唯一性（URDF规范要求）"""
        names = [j["name"] for j in joints]
        duplicates = [n for n in names if names.count(n) > 1]
        if duplicates:
            self.errors.append(f"关节名称重复: {set(duplicates)}")
            return False
        return True
    
    def _check_link_connectivity(self, links: List, joints: List) -> bool:
        """验证所有 link 互相连通（无孤立节点）"""
        link_names = set(l["name"] for l in links)
        connected = {"base_link"}  # 从根节点开始
        
        # 通过关节找到所有连通的 link
        changed = True
        while changed:
            changed = False
            for j in joints:
                if j["parent"] in connected and j["child"] not in connected:
                    connected.add(j["child"])
                    changed = True
        
        orphans = link_names - connected
        if orphans:
            self.errors.append(f"存在孤立的 link: {orphans}")
            return False
        return True
    
    def _check_mesh_files_exist(self, links: List, urdf_path: str) -> bool:
        """验证 URDF 引用的所有 mesh 文件实际存在"""
        urdf_dir = os.path.dirname(urdf_path)
        all_exist = True
        for link in links:
            for mesh_path in link.get("meshes", []):
                full_path = os.path.join(urdf_dir, mesh_path)
                if not os.path.exists(full_path):
                    self.errors.append(f"Mesh 文件不存在: {mesh_path}")
                    all_exist = False
        return all_exist
    
    def _check_unit_consistency(self, meshes: Dict) -> bool:
        """验证单位一致性（URDF 标准使用米）"""
        for name, mesh in meshes.items():
            # 检查网格尺寸是否在合理范围内（盒子应该在0.01m-2m之间）
            max_extent = max(mesh.bounding_box.extents)
            if max_extent > 10:  # 可能使用了毫米
                self.warnings.append(f"{name} 尺寸 {max_extent:.2f}m 过大，是否使用了错误单位？")
            elif max_extent < 0.001:  # 小于1mm
                self.warnings.append(f"{name} 尺寸 {max_extent:.4f}m 过小，是否使用了错误单位？")
        return True  # 仅产生警告
    
    def _check_manifold(self, meshes: Dict) -> bool:
        """检查网格是否为流形（无非流形边/顶点）"""
        all_manifold = True
        for name, mesh in meshes.items():
            # trimesh 没有直接的 is_manifold，用边检查代替
            edges = mesh.edges_unique
            edge_face_count = np.bincount(mesh.edges_sorted.view('i8,i8').view('i8'), minlength=len(edges))
            non_manifold_edges = np.sum(edge_face_count > 2)
            if non_manifold_edges > 0:
                self.warnings.append(f"网格 {name} 存在 {non_manifold_edges} 条非流形边")
        return all_manifold

    def _check_no_degenerate_faces(self, meshes: Dict) -> bool:
        """检查是否存在退化面（面积为0的三角形）"""
        for name, mesh in meshes.items():
            areas = mesh.area_faces
            degenerate_count = np.sum(areas < 1e-10)
            if degenerate_count > 0:
                self.warnings.append(f"网格 {name} 存在 {degenerate_count} 个退化面")
        return True  # 仅产生警告
```

#### 4.2.2 运动学验证（纯几何计算）

```python
class KinematicsValidator:
    """纯程序化运动学验证器，基于几何变换，不需要物理仿真器"""
    
    def __init__(self, sample_points: int = 20):
        self.sample_points = sample_points
        self.collision_checker = CollisionChecker()
    
    def validate_all_joints(self, urdf_path: str) -> Dict:
        """
        验证所有关节在整个运动范围内的碰撞情况
        
        算法：
        1. 解析 URDF 获取关节信息
        2. 对每个关节的运动范围进行采样
        3. 在每个采样点应用几何变换
        4. 使用 GJK/EPA 算法检测碰撞
        
        Returns:
            {
                "joint_name": {
                    "passed": bool,
                    "collision_angles": List[float],  # 发生碰撞的角度
                    "safe_range": Tuple[float, float]  # 安全范围
                }
            }
        """
        links, joints = self._parse_urdf(urdf_path)
        meshes = self._load_meshes(links, urdf_path)
        
        results = {}
        for joint in joints:
            if joint["type"] == "fixed":
                continue
            
            result = self._validate_single_joint(joint, meshes, joints)
            results[joint["name"]] = result
        
        return results
    
    def _validate_single_joint(self, joint: Dict, meshes: Dict, all_joints: List) -> Dict:
        """验证单个关节的运动范围"""
        
        lower = joint.get("lower", -np.pi)
        upper = joint.get("upper", np.pi)
        axis = np.array(joint["axis"])
        origin = np.array(joint["origin"])
        
        collision_angles = []
        
        for angle in np.linspace(lower, upper, self.sample_points):
            # 应用关节变换
            transformed_meshes = self._apply_joint_transform(
                meshes, joint, angle, all_joints
            )
            
            # 检测碰撞
            collisions = self.collision_checker.check_all_pairs(transformed_meshes)
            
            if len(collisions) > 0:
                collision_angles.append(angle)
        
        # 计算安全范围
        safe_range = self._compute_safe_range(lower, upper, collision_angles)
        
        return {
            "passed": len(collision_angles) == 0,
            "collision_angles": collision_angles,
            "safe_range": safe_range,
            "suggested_limits": safe_range if len(collision_angles) > 0 else (lower, upper)
        }
    
    def _apply_joint_transform(self, meshes: Dict, joint: Dict, 
                               angle: float, all_joints: List) -> Dict:
        """
        应用关节变换到受影响的网格
        
        使用正向运动学计算：
        T_child = T_parent * R_joint(angle)
        """
        origin = np.array(joint["origin"])
        axis = np.array(joint["axis"])
        child_link = joint["child"]
        
        # 创建旋转矩阵
        rotation_matrix = self._axis_angle_to_matrix(axis, angle)
        
        # 变换子部件
        result = {}
        for name, mesh in meshes.items():
            if name == child_link or self._is_descendant(name, child_link, all_joints):
                # 应用变换
                transformed = mesh.copy()
                # 1. 平移到关节原点
                transformed.apply_translation(-origin)
                # 2. 旋转
                transformed.apply_transform(rotation_matrix)
                # 3. 平移回去
                transformed.apply_translation(origin)
                result[name] = transformed
            else:
                result[name] = mesh
        
        return result
    
    def _axis_angle_to_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """轴角表示转换为4x4变换矩阵（Rodrigues公式）"""
        axis = axis / np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        T = np.eye(4)
        T[:3, :3] = R
        return T
    
    def _compute_safe_range(self, lower: float, upper: float, 
                           collision_angles: List[float]) -> Tuple[float, float]:
        """计算安全运动范围"""
        if len(collision_angles) == 0:
            return (lower, upper)
        
        # 找到第一个碰撞角度和最后一个碰撞角度
        collision_angles = sorted(collision_angles)
        
        # 从两端收缩到安全范围
        safe_lower = lower
        safe_upper = upper
        
        for angle in collision_angles:
            if angle <= (lower + upper) / 2:
                safe_lower = max(safe_lower, angle + 0.05)  # 5度安全余量
            else:
                safe_upper = min(safe_upper, angle - 0.05)
        
        return (safe_lower, safe_upper)
```

#### 4.2.3 碰撞检测（GJK/EPA 算法实现）

```python
class CollisionChecker:
    """
    基于 GJK (Gilbert-Johnson-Keerthi) 算法的碰撞检测器
    纯 Python 实现，不依赖外部物理引擎
    """
    
    def __init__(self, margin: float = 0.001):
        """
        Args:
            margin: 碰撞检测余量（米），用于防止数值误差
        """
        self.margin = margin
    
    def check_all_pairs(self, meshes: Dict[str, trimesh.Trimesh]) -> List[Tuple[str, str]]:
        """
        检测所有网格对之间的碰撞
        
        优化：使用 AABB（轴对齐边界框）进行粗筛选
        """
        collisions = []
        mesh_list = list(meshes.items())
        
        for i in range(len(mesh_list)):
            for j in range(i + 1, len(mesh_list)):
                name_a, mesh_a = mesh_list[i]
                name_b, mesh_b = mesh_list[j]
                
                # 1. AABB 粗筛选
                if not self._aabb_overlap(mesh_a, mesh_b):
                    continue
                
                # 2. 精确碰撞检测
                if self._gjk_collision(mesh_a, mesh_b):
                    collisions.append((name_a, name_b))
        
        return collisions
    
    def _aabb_overlap(self, mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> bool:
        """检查两个网格的 AABB 是否重叠"""
        bounds_a = mesh_a.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        bounds_b = mesh_b.bounds
        
        # 对每个轴检查重叠
        for axis in range(3):
            if bounds_a[1, axis] < bounds_b[0, axis] - self.margin:
                return False
            if bounds_b[1, axis] < bounds_a[0, axis] - self.margin:
                return False
        
        return True
    
    def _gjk_collision(self, mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> bool:
        """
        GJK 算法判断两个凸多面体是否碰撞
        
        核心思想：
        如果两个凸多面体 A 和 B 相交，则它们的闵可夫斯基差 A - B 包含原点
        """
        # 使用 trimesh 内置的碰撞检测
        collision_mgr = trimesh.collision.CollisionManager()
        collision_mgr.add_object("a", mesh_a)
        return collision_mgr.in_collision_single(mesh_b)
    
    def compute_penetration_depth(self, mesh_a: trimesh.Trimesh, 
                                  mesh_b: trimesh.Trimesh) -> float:
        """
        使用 EPA (Expanding Polytope Algorithm) 计算穿透深度
        
        Returns:
            穿透深度（米），0 表示无穿透
        """
        collision_mgr = trimesh.collision.CollisionManager()
        collision_mgr.add_object("a", mesh_a)
        is_collision, data = collision_mgr.in_collision_single(mesh_b, return_data=True)
        
        if not is_collision:
            return 0.0
        
        # 返回最大穿透深度
        if len(data) > 0:
            return max(d.depth for d in data)
        return 0.0
```

#### 4.2.4 集成验证管线

```python
class BoxValidationPipeline:
    """
    完整的盒子资产验证管线
    自动化验证 + 报告生成 + 错误定位
    """
    
    def __init__(self, output_dir: str = "validation_reports"):
        self.geometry_validator = GeometryValidator()
        self.kinematics_validator = KinematicsValidator()
        self.collision_checker = CollisionChecker()
        self.output_dir = output_dir
    
    def validate(self, urdf_path: str, verbose: bool = True) -> Dict:
        """
        执行完整验证流程
        
        Returns:
            {
                "overall_passed": bool,
                "geometry": {...},
                "kinematics": {...},
                "collision": {...},
                "report_path": str
            }
        """
        results = {
            "urdf_path": urdf_path,
            "timestamp": datetime.now().isoformat(),
        }
        
        # 1. 几何验证
        if verbose:
            print("🔍 [1/3] 几何验证...")
        geo_result = self.geometry_validator.validate(urdf_path)
        results["geometry"] = geo_result
        self._print_result("几何验证", geo_result["passed"], verbose)
        
        # 2. 运动学验证
        if verbose:
            print("🔍 [2/3] 运动学验证...")
        kin_result = self.kinematics_validator.validate_all_joints(urdf_path)
        all_joints_passed = all(j["passed"] for j in kin_result.values())
        results["kinematics"] = {"joints": kin_result, "passed": all_joints_passed}
        self._print_result("运动学验证", all_joints_passed, verbose)
        
        # 3. 汇总结果
        results["overall_passed"] = geo_result["passed"] and all_joints_passed
        
        # 4. 生成报告
        report_path = self._generate_report(results)
        results["report_path"] = report_path
        
        if verbose:
            status = "✅ 通过" if results["overall_passed"] else "❌ 失败"
            print(f"\n{'='*50}")
            print(f"验证结果: {status}")
            print(f"详细报告: {report_path}")
            print(f"{'='*50}")
        
        return results
    
    def _print_result(self, name: str, passed: bool, verbose: bool):
        if verbose:
            status = "✅" if passed else "❌"
            print(f"   {status} {name}")
    
    def _generate_report(self, results: Dict) -> str:
        """生成 JSON 格式的详细报告"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        filename = f"validation_{Path(results['urdf_path']).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(self.output_dir, filename)
        
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        return report_path
    
    def batch_validate(self, urdf_paths: List[str]) -> Dict:
        """批量验证多个 URDF 文件"""
        all_results = []
        passed_count = 0
        
        for i, path in enumerate(urdf_paths):
            print(f"\n[{i+1}/{len(urdf_paths)}] 验证: {path}")
            result = self.validate(path, verbose=False)
            all_results.append(result)
            if result["overall_passed"]:
                passed_count += 1
                print(f"   ✅ 通过")
            else:
                print(f"   ❌ 失败")
                for err in result["geometry"]["errors"][:3]:
                    print(f"      - {err}")
        
        print(f"\n总计: {passed_count}/{len(urdf_paths)} 通过")
        return {
            "total": len(urdf_paths),
            "passed": passed_count,
            "failed": len(urdf_paths) - passed_count,
            "results": all_results
        }
```

### 4.3 关键技术点

#### 4.3.1 铰链位置精确计算

```python
def calculate_hinge_position(parent_panel, child_panel, edge_type):
    """
    计算铰链精确位置
    
    Args:
        parent_panel: 父面板几何
        child_panel: 子面板几何
        edge_type: 边缘类型 ("top", "bottom", "left", "right")
    
    Returns:
        position: 铰链中心位置 (x, y, z)
        axis: 旋转轴方向 (ax, ay, az)
    """
    # 获取共享边
    shared_edge = get_shared_edge(parent_panel, child_panel)
    
    # 铰链位置在边缘中心
    position = (shared_edge.start + shared_edge.end) / 2
    
    # 旋转轴平行于边缘
    axis = normalize(shared_edge.end - shared_edge.start)
    
    return position, axis
```

#### 4.3.2 关节范围约束计算

```python
def calculate_joint_limits(parent, child, joint_pos, axis, thickness):
    """
    计算关节范围以避免碰撞
    
    Args:
        parent: 父部件几何
        child: 子部件几何
        joint_pos: 关节位置
        axis: 旋转轴
        thickness: 材料厚度
    
    Returns:
        (min_angle, max_angle): 安全旋转范围
    """
    # 二分搜索最大安全角度
    max_angle = np.pi
    while max_angle > 0:
        rotated_child = rotate_around_axis(child, joint_pos, axis, max_angle)
        if not check_collision(parent, rotated_child):
            break
        max_angle -= 0.01
    
    # 添加安全余量
    max_angle -= thickness / 2
    
    return (0, max_angle)
```

#### 4.3.3 防止几何穿透

```python
def add_thickness_offset(panel, thickness, direction):
    """
    为面板添加厚度偏移，防止多层折叠时穿透
    
    Args:
        panel: 面板几何
        thickness: 材料厚度
        direction: 偏移方向
    """
    # 计算累积偏移
    offset = thickness * get_layer_index(panel) * direction
    
    # 应用偏移
    panel.location += offset
```

---

## 5. URDF 输出规范

### 5.1 文件结构

导出目录：`<export_dir>/<asset_name>/<seed>/`（默认 `export_dir=sim_exports/urdf`）

示例（TuckEndBox, `seed=42`，与当前脚本输出一致）：

```
sim_exports/urdf/tuckendbox/42/
├── tuckendbox.urdf
├── metadata.json
└── assets/
    ├── geom_0.obj
    ├── geom_0.mtl
    ├── geom_1.obj
    ├── geom_1.mtl
    ├── ...
    └── geom_N.obj
```

> [!NOTE]
> - 当 `visual_only=True`（在线查看/快速验证常用）时，仅导出 `<visual>` mesh，不生成 `<collision>`。
> - 当 `visual_only=False` 时，会在同一 `assets/` 目录下额外生成碰撞网格（例如 `*_col0.obj`），并在 URDF 中生成 `<collision>` 元素。

### 5.2 URDF 示例

```xml
<?xml version="1.0" ?>
<robot name="object">
  <!-- URDFBuilder 会创建一个 world link，并用 world_joint 固定连接到 link_0 -->
  <link name="world"/>
  <joint name="world_joint" type="fixed">
    <origin xyz="0.0 0.0 0.0"/>
    <parent link="world"/>
    <child link="link_0"/>
    <dynamics damping="0.0" friction="0.0"/>
  </joint>

  <!-- 示例：一个折页关节（名称来自 joint label + index） -->
  <joint name="top_back_flap_0" type="revolute">
    <origin xyz="0.0 0.0 0.0"/>
    <parent link="link_0"/>
    <child link="link_1"/>
    <dynamics damping="0.4" friction="0.1"/>
    <axis xyz="1.0 0.0 0.0"/>
    <limit lower="0.0" upper="1.5708"/>
  </joint>

  <!-- 每个 link 可能包含多个 visual（多个 mesh 资产） -->
  <link name="link_1">
    <visual>
      <geometry>
        <mesh filename="assets/geom_4.obj"/>
      </geometry>
      <origin xyz="0.0 0.0 0.0"/>
    </visual>
    <inertial>
      <mass value="0.001"/>
      <inertia ixx="1e-8" ixy="0.0" ixz="0.0" iyy="1e-8" iyz="0.0" izz="1e-8"/>
      <origin xyz="0.0 0.0 0.0"/>
    </inertial>
  </link>
</robot>
```

### 5.3 元数据规范

当前导出器会写出一个最小可用的 `metadata.json`（用于标注 joint label 与资产 bounding box）。

示例（与 `tuckendbox/42/metadata.json` 同构）：

```json
{
  "joint5": { "joint label": "top_back_flap" },
  "joint4": { "joint label": "top_tuck_tab" },
  "joint3": { "joint label": "top_dust_left" },
  "joint2": { "joint label": "top_dust_right" },
  "joint1": { "joint label": "bottom_front_flap" },
  "joint0": { "joint label": "bottom_tuck_tab" },
  "part_labels": [],
  "bounding_box": {
    "min": [-0.1, -0.1, -0.1],
    "max": [0.1, 0.1, 0.1]
  }
}
```

> [!NOTE]
> 后续若需要更“产品化”的元数据（尺寸、材料、关节层级与推荐折叠顺序），建议在 `metadata.json` 中扩展字段，
> 但应保持现有字段兼容（尤其是 `joint*` 与 `bounding_box`）。

---

## 6. 技术风险详细解决方案

### 6.1 风险总览与解决方案索引

| 风险ID | 风险描述 | 可能性 | 影响 | 解决方案章节 | 验证方式 |
|--------|---------|-------|------|-------------|---------|
| R1 | 几何穿透 | 高 | 严重 | 6.2 | CollisionChecker |
| R2 | 关节位置不精确 | 中 | 中 | 6.3 | EdgeAligner |
| R3 | 多关节联动复杂 | 高 | 中 | 6.4 | KinematicTreeValidator |
| R4 | 物理参数不准确 | 中 | 低 | 6.5 | PhysicsValidator |
| **R5** | **Blender 版本兼容性** | 中 | 中 | 6.6 | VersionChecker |
| **R6** | **URDF 导出器多关节导出**（已修复，需回归） | 低 | ✅ 已修复 | 6.7 | URDFExporterTest |
| **R7** | **碰撞网格质量** | 中 | 中 | 6.8 | MeshSimplifier |
| **R8** | **关节动力学参数校准** | 中 | 低 | 6.9 | DynamicsPresets |
| **R-Deep-1** | **薄壳惯性张量不稳定**（已修复，需回归） | 低 | ✅ 已修复 | 6.10 | ThinShellInertiaFix |
| **R-Deep-2** | **几何节点单一化架构** | 中 | 高 | 6.11 | ModularNodeTree |

> [!NOTE]
> **状态更新（v6.0）**：R6（多关节导出）与 R-Deep-1（薄壳惯性）曾是阻断性风险，但已在当前代码中修复并验证。
> 后续盒型开发应保留对应回归测试，避免回退（见 6.7 / 6.10）。

### 6.2 风险 R1: 几何穿透 - 详细解决方案

#### 6.2.1 问题分析

几何穿透发生的三种场景：
1. **初始状态穿透**: 部件几何在设计时就有重叠
2. **运动过程穿透**: 关节旋转/滑动过程中发生碰撞
3. **多层折叠穿透**: 多个面板折叠堆叠时厚度累积

#### 6.2.2 解决算法

```python
class AntiPenetrationSystem:
    """
    防穿透系统 - 从设计阶段消除穿透可能性
    
    核心策略:
    1. 设计时预留间隙 (Design-time Gap)
    2. 运动范围预计算 (Motion Range Pre-computation)
    3. 多层偏移补偿 (Multi-layer Offset Compensation)
    """
    
    def __init__(self, base_gap: float = 0.0005):  # 0.5mm 基础间隙
        self.base_gap = base_gap
    
    # ========== 策略 1: 设计时预留间隙 ==========
    
    def compute_panel_gap(self, thickness: float, fold_angle: float) -> float:
        """
        计算面板间的最小间隙
        
        公式推导:
        当两个厚度为 t 的面板以角度 θ 折叠时，
        接触点的最小间隙 = t * (1 - cos(θ/2)) + base_gap
        
        Args:
            thickness: 面板厚度 (m)
            fold_angle: 最大折叠角度 (rad)
        
        Returns:
            所需间隙 (m)
        """
        geometric_gap = thickness * (1 - np.cos(fold_angle / 2))
        return geometric_gap + self.base_gap
    
    def adjust_panel_dimensions(self, panel_params: Dict) -> Dict:
        """
        调整面板尺寸以预留间隙
        
        Returns:
            调整后的面板参数
        """
        adjusted = panel_params.copy()
        thickness = panel_params["thickness"]
        
        # 宽度和深度各减去一个厚度
        adjusted["width"] -= 2 * thickness
        adjusted["depth"] -= 2 * thickness
        
        return adjusted
    
    # ========== 策略 2: 运动范围预计算 ==========
    
    def compute_safe_joint_range(self, 
                                  parent_mesh: trimesh.Trimesh,
                                  child_mesh: trimesh.Trimesh,
                                  joint_axis: np.ndarray,
                                  joint_origin: np.ndarray,
                                  initial_range: Tuple[float, float] = (-np.pi, np.pi)
                                  ) -> Tuple[float, float]:
        """
        预计算关节的安全运动范围
        
        算法: 二分搜索 + 碰撞检测
        时间复杂度: O(log(precision) * collision_check)
        
        Args:
            parent_mesh: 父部件网格
            child_mesh: 子部件网格  
            joint_axis: 关节旋转轴
            joint_origin: 关节原点
            initial_range: 初始范围
            
        Returns:
            (safe_min, safe_max): 安全角度范围
        """
        precision = 0.01  # 约 0.57 度
        
        # 二分搜索下界
        safe_min = self._binary_search_limit(
            parent_mesh, child_mesh, joint_axis, joint_origin,
            start=0, end=initial_range[0], direction=-1
        )
        
        # 二分搜索上界
        safe_max = self._binary_search_limit(
            parent_mesh, child_mesh, joint_axis, joint_origin,
            start=0, end=initial_range[1], direction=1
        )
        
        # 添加安全余量
        margin = 0.02  # ~1.15 度
        return (safe_min + margin, safe_max - margin)
    
    def _binary_search_limit(self, parent, child, axis, origin, 
                             start, end, direction) -> float:
        """二分搜索碰撞边界"""
        precision = 0.01
        
        while abs(end - start) > precision:
            mid = (start + end) / 2
            rotated = self._rotate_mesh(child, axis, origin, mid)
            
            if self._check_collision(parent, rotated):
                end = mid  # 碰撞了，收缩范围
            else:
                start = mid  # 没碰撞，扩展范围
        
        return start if direction > 0 else end
    
    # ========== 策略 3: 多层偏移补偿 ==========
    
    def compute_layer_offset(self, layer_index: int, thickness: float, 
                             fold_direction: np.ndarray) -> np.ndarray:
        """
        计算多层折叠时的偏移量
        
        当多个面板层叠折叠时，每层需要向外偏移以避免穿透
        
        公式:
        offset = layer_index * thickness * fold_direction
        
        Args:
            layer_index: 层索引 (0, 1, 2, ...)
            thickness: 面板厚度
            fold_direction: 折叠方向单位向量
            
        Returns:
            偏移向量
        """
        return layer_index * thickness * fold_direction
    
    def apply_layer_offsets(self, panels: List[Dict], 
                           fold_hierarchy: List[List[int]]) -> List[Dict]:
        """
        应用多层偏移到所有面板
        
        Args:
            panels: 面板列表
            fold_hierarchy: 折叠层级关系 [[0], [1,2], [3,4,5], ...]
            
        Returns:
            调整偏移后的面板列表
        """
        adjusted_panels = []
        
        for level, panel_indices in enumerate(fold_hierarchy):
            for idx in panel_indices:
                panel = panels[idx].copy()
                offset = self.compute_layer_offset(
                    level, 
                    panel["thickness"],
                    panel["fold_direction"]
                )
                panel["position"] = np.array(panel["position"]) + offset
                adjusted_panels.append(panel)
        
        return adjusted_panels

    # ========== 验证接口 ==========
    
    def verify_no_penetration(self, box_asset: Dict) -> Dict:
        """
        验证盒子资产无穿透
        
        Returns:
            {
                "passed": bool,
                "initial_penetrations": List,
                "motion_penetrations": List,
                "recommendations": List[str]
            }
        """
        result = {
            "passed": True,
            "initial_penetrations": [],
            "motion_penetrations": [],
            "recommendations": []
        }
        
        # 检查初始状态
        meshes = self._load_meshes(box_asset)
        for (name_a, mesh_a), (name_b, mesh_b) in itertools.combinations(meshes.items(), 2):
            depth = self._compute_penetration_depth(mesh_a, mesh_b)
            if depth > 0:
                result["initial_penetrations"].append({
                    "parts": [name_a, name_b],
                    "depth": depth
                })
                result["passed"] = False
                result["recommendations"].append(
                    f"增加 {name_a} 和 {name_b} 之间的间隙至少 {depth + 0.001:.4f}m"
                )
        
        # 检查运动过程
        for joint in box_asset["joints"]:
            motion_result = self._check_motion_penetration(box_asset, joint)
            if not motion_result["safe"]:
                result["motion_penetrations"].extend(motion_result["collisions"])
                result["passed"] = False
                result["recommendations"].append(
                    f"将关节 {joint['name']} 范围调整为 {motion_result['safe_range']}"
                )
        
        return result
```

---

### 6.3 风险 R2: 关节位置不精确 - 详细解决方案

#### 6.3.1 问题分析

关节位置不精确导致的问题：
- 旋转时产生不自然的间隙或碰撞
- 多关节系统产生累积误差
- 与实际纸盒折痕不符

#### 6.3.2 解决算法

```python
class PrecisionJointPlacer:
    """
    精确关节定位器
    
    核心算法: 边缘匹配 + 顶点对齐
    """
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
    
    def find_shared_edge(self, mesh_a: trimesh.Trimesh, 
                         mesh_b: trimesh.Trimesh) -> Dict:
        """
        寻找两个网格的共享边缘
        
        算法: 
        1. 提取两个网格的所有边
        2. 使用 KD-Tree 进行最近邻搜索
        3. 识别共线的边对
        4. 合并为共享边缘
        
        Returns:
            {
                "start": np.ndarray,  # 边缘起点
                "end": np.ndarray,    # 边缘终点
                "center": np.ndarray, # 边缘中心
                "direction": np.ndarray,  # 边缘方向(单位向量)
                "length": float,
                "confidence": float  # 匹配置信度
            }
        """
        edges_a = self._extract_boundary_edges(mesh_a)
        edges_b = self._extract_boundary_edges(mesh_b)
        
        best_match = None
        best_score = 0
        
        for edge_a in edges_a:
            for edge_b in edges_b:
                score = self._edge_similarity(edge_a, edge_b)
                if score > best_score:
                    best_score = score
                    best_match = self._merge_edges(edge_a, edge_b)
        
        if best_match:
            best_match["confidence"] = best_score
        
        return best_match
    
    def compute_hinge_transform(self, shared_edge: Dict, 
                                parent_normal: np.ndarray,
                                child_normal: np.ndarray) -> Dict:
        """
        计算铰链的精确变换参数
        
        Args:
            shared_edge: 共享边缘信息
            parent_normal: 父面板法线
            child_normal: 子面板法线
            
        Returns:
            {
                "origin": np.ndarray,   # 铰链原点
                "axis": np.ndarray,     # 旋转轴
                "initial_angle": float, # 初始角度
            }
        """
        origin = shared_edge["center"]
        
        # 旋转轴 = 边缘方向
        axis = shared_edge["direction"]
        
        # 计算初始角度 (父法线和子法线的夹角)
        cos_angle = np.dot(parent_normal, child_normal)
        initial_angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        # 确定旋转方向
        cross = np.cross(parent_normal, child_normal)
        if np.dot(cross, axis) < 0:
            axis = -axis
        
        return {
            "origin": origin,
            "axis": axis,
            "initial_angle": initial_angle
        }
    
    def align_vertices_to_edge(self, mesh: trimesh.Trimesh, 
                               edge: Dict) -> trimesh.Trimesh:
        """
        将网格顶点对齐到边缘
        
        目的: 确保关节位置的精确性
        
        算法:
        1. 找到网格上最接近边缘的顶点
        2. 将这些顶点精确对齐到边缘线上
        3. 保持其他顶点相对关系
        """
        aligned = mesh.copy()
        edge_start = edge["start"]
        edge_direction = edge["direction"]
        
        # 找到需要对齐的顶点 (距离边缘 < tolerance)
        vertices = aligned.vertices
        for i, v in enumerate(vertices):
            # 计算点到边缘的投影
            t = np.dot(v - edge_start, edge_direction)
            projection = edge_start + t * edge_direction
            distance = np.linalg.norm(v - projection)
            
            if distance < self.tolerance * 10:  # 10x tolerance 范围内对齐
                # 对齐到边缘线上
                vertices[i] = projection
        
        aligned.vertices = vertices
        return aligned
    
    # ========== 验证接口 ==========
    
    def verify_joint_precision(self, joint: Dict, meshes: Dict) -> Dict:
        """
        验证关节位置精度
        
        Returns:
            {
                "passed": bool,
                "position_error": float,  # 位置误差(m)
                "axis_error": float,      # 轴向误差(度)
                "alignment_score": float  # 对齐分数 0-1
            }
        """
        parent_mesh = meshes[joint["parent"]]
        child_mesh = meshes[joint["child"]]
        
        # 找到实际共享边
        actual_edge = self.find_shared_edge(parent_mesh, child_mesh)
        
        if actual_edge is None:
            return {
                "passed": False,
                "position_error": float('inf'),
                "axis_error": float('inf'),
                "alignment_score": 0,
                "error": "无法找到共享边缘"
            }
        
        # 计算位置误差
        declared_origin = np.array(joint["origin"])
        position_error = np.linalg.norm(declared_origin - actual_edge["center"])
        
        # 计算轴向误差
        declared_axis = np.array(joint["axis"])
        axis_dot = abs(np.dot(declared_axis, actual_edge["direction"]))
        axis_error = np.degrees(np.arccos(np.clip(axis_dot, 0, 1)))
        
        passed = position_error < 0.001 and axis_error < 1.0  # 1mm, 1度
        
        return {
            "passed": passed,
            "position_error": position_error,
            "axis_error": axis_error,
            "alignment_score": actual_edge["confidence"],
            "suggested_origin": actual_edge["center"].tolist(),
            "suggested_axis": actual_edge["direction"].tolist()
        }
```

---

### 6.4 风险 R3: 多关节联动复杂 - 详细解决方案

#### 6.4.1 问题分析

多关节系统的复杂性：
- 父子关系链需要正确传递变换
- 多个关节同时运动时可能产生干涉
- 关节间的约束关系需要显式定义

#### 6.4.2 解决算法

```python
class KinematicTreeBuilder:
    """
    运动学树构建器
    
    职责:
    1. 构建正确的关节层级关系
    2. 验证运动学链的完整性
    3. 检测关节间的潜在冲突
    """
    
    def __init__(self):
        self.tree = {}
        self.joint_order = []
    
    def build_tree(self, joints: List[Dict]) -> Dict:
        """
        从关节列表构建运动学树
        
        Returns:
            {
                "root": "base_link",
                "nodes": {
                    "link_name": {
                        "parent": str,
                        "children": List[str],
                        "joint": Dict or None,
                        "depth": int
                    }
                },
                "joint_order": List[str]  # 拓扑排序后的关节顺序
            }
        """
        nodes = {"base_link": {"parent": None, "children": [], "joint": None, "depth": 0}}
        
        for joint in joints:
            parent = joint["parent"]
            child = joint["child"]
            
            if child not in nodes:
                nodes[child] = {"parent": parent, "children": [], "joint": joint, "depth": 0}
            else:
                nodes[child]["parent"] = parent
                nodes[child]["joint"] = joint
            
            if parent in nodes:
                nodes[parent]["children"].append(child)
        
        # 计算深度
        self._compute_depths(nodes, "base_link", 0)
        
        # 拓扑排序
        joint_order = self._topological_sort(nodes, joints)
        
        return {
            "root": "base_link",
            "nodes": nodes,
            "joint_order": joint_order
        }
    
    def _compute_depths(self, nodes: Dict, current: str, depth: int):
        nodes[current]["depth"] = depth
        for child in nodes[current]["children"]:
            self._compute_depths(nodes, child, depth + 1)
    
    def _topological_sort(self, nodes: Dict, joints: List) -> List[str]:
        """拓扑排序: 确保父关节在子关节之前处理"""
        joint_names = [j["name"] for j in joints]
        joint_depths = {j["name"]: nodes[j["child"]]["depth"] for j in joints}
        return sorted(joint_names, key=lambda x: joint_depths[x])
    
    def detect_joint_conflicts(self, tree: Dict, joints: List) -> List[Dict]:
        """
        检测关节间的潜在冲突
        
        冲突类型:
        1. 共享轴冲突: 两个关节共享相同的旋转轴
        2. 嵌套碰撞: 子关节运动导致与父部件碰撞
        3. 循环依赖: 关节形成环路(不应存在)
        
        Returns:
            冲突列表
        """
        conflicts = []
        
        # 检查共享轴冲突
        for i, j1 in enumerate(joints):
            for j2 in joints[i+1:]:
                if self._axes_parallel(j1["axis"], j2["axis"]):
                    if np.linalg.norm(np.array(j1["origin"]) - np.array(j2["origin"])) < 0.01:
                        conflicts.append({
                            "type": "shared_axis",
                            "joints": [j1["name"], j2["name"]],
                            "description": "两个关节共享近似相同的旋转轴，可能导致运动干涉"
                        })
        
        # 检查循环依赖
        if self._has_cycle(tree):
            conflicts.append({
                "type": "cycle",
                "joints": "all",
                "description": "关节层级中存在循环依赖"
            })
        
        return conflicts
    
    def _axes_parallel(self, axis1, axis2) -> bool:
        dot = abs(np.dot(axis1, axis2))
        return dot > 0.99  # 约 8 度以内
    
    def _has_cycle(self, tree: Dict) -> bool:
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            
            for child in tree["nodes"].get(node, {}).get("children", []):
                if child not in visited:
                    if dfs(child):
                        return True
                elif child in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        return dfs(tree["root"])
    
    # ========== 验证接口 ==========
    
    def verify_kinematic_tree(self, urdf_path: str) -> Dict:
        """
        验证运动学树的正确性
        
        Returns:
            {
                "passed": bool,
                "tree_valid": bool,
                "conflicts": List,
                "warnings": List,
                "tree_visualization": str  # ASCII 树形图
            }
        """
        links, joints = self._parse_urdf(urdf_path)
        tree = self.build_tree(joints)
        conflicts = self.detect_joint_conflicts(tree, joints)
        
        visualization = self._visualize_tree(tree)
        
        return {
            "passed": len(conflicts) == 0,
            "tree_valid": not self._has_cycle(tree),
            "conflicts": conflicts,
            "warnings": [],
            "tree_visualization": visualization
        }
    
    def _visualize_tree(self, tree: Dict) -> str:
        """生成 ASCII 树形可视化"""
        lines = []
        
        def print_node(node, prefix="", is_last=True):
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{node}")
            
            children = tree["nodes"].get(node, {}).get("children", [])
            for i, child in enumerate(children):
                extension = "    " if is_last else "│   "
                print_node(child, prefix + extension, i == len(children) - 1)
        
        print_node(tree["root"], "", True)
        return "\n".join(lines)
```

---

### 6.5 风险 R4: 物理参数不准确 - 详细解决方案

#### 6.5.1 问题分析

物理参数影响仿真真实性：
- 质量/密度影响动力学行为
- 摩擦系数影响接触行为
- 惯性矩影响旋转动力学

#### 6.5.2 解决算法

```python
class PhysicalParameterCalculator:
    """
    物理参数精确计算器
    
    基于材料数据库和几何分析计算物理属性
    """
    
    # 材料属性数据库 (基于真实数据)
    MATERIAL_DATABASE = {
        "cardboard": {
            "density": 700,        # kg/m³
            "friction": 0.5,       # 摩擦系数
            "restitution": 0.1,    # 恢复系数
            "thickness_range": (0.0003, 0.002),  # 0.3mm - 2mm
        },
        "corrugated": {
            "density": 150,        # kg/m³ (含空气)
            "friction": 0.6,
            "restitution": 0.2,
            "thickness_range": (0.003, 0.01),  # 3mm - 10mm
        },
        "plastic_pp": {
            "density": 900,
            "friction": 0.3,
            "restitution": 0.4,
            "thickness_range": (0.0005, 0.003),
        },
        "wood": {
            "density": 600,
            "friction": 0.4,
            "restitution": 0.3,
            "thickness_range": (0.003, 0.02),
        }
    }
    
    def __init__(self, material_type: str = "cardboard"):
        self.material = self.MATERIAL_DATABASE.get(material_type, self.MATERIAL_DATABASE["cardboard"])
    
    def calculate_mass(self, mesh: trimesh.Trimesh) -> float:
        """
        计算网格质量
        
        公式: mass = volume * density
        """
        volume = abs(mesh.volume)
        return volume * self.material["density"]
    
    def calculate_inertia_tensor(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        计算惯性张量
        
        使用 trimesh 的质量属性计算
        返回相对于质心的惯性张量
        """
        # 设置密度
        mesh.density = self.material["density"]
        
        # 获取惯性张量
        inertia = mesh.moment_inertia
        
        # 确保正定性
        inertia = np.clip(inertia, 1e-10, None)
        
        return inertia
    
    def calculate_center_of_mass(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """计算质心"""
        return mesh.center_mass
    
    def validate_physical_params(self, urdf_path: str) -> Dict:
        """
        验证 URDF 中的物理参数合理性
        
        检查项:
        1. 质量是否在合理范围
        2. 惯性张量是否正定
        3. 惯性张量主轴是否合理
        
        Returns:
            {
                "passed": bool,
                "link_validations": Dict[str, Dict],
                "recommendations": List[str]
            }
        """
        links, _ = self._parse_urdf(urdf_path)
        meshes = self._load_meshes(links, urdf_path)
        
        result = {
            "passed": True,
            "link_validations": {},
            "recommendations": []
        }
        
        for link_name, mesh in meshes.items():
            link_result = self._validate_link_physics(link_name, mesh)
            result["link_validations"][link_name] = link_result
            
            if not link_result["passed"]:
                result["passed"] = False
                result["recommendations"].extend(link_result["recommendations"])
        
        return result
    
    def _validate_link_physics(self, name: str, mesh: trimesh.Trimesh) -> Dict:
        """验证单个链接的物理参数"""
        mass = self.calculate_mass(mesh)
        inertia = self.calculate_inertia_tensor(mesh)
        
        issues = []
        recommendations = []
        
        # 检查质量范围 (盒子应该在 0.001kg - 10kg 之间)
        if mass < 0.001:
            issues.append("质量过小")
            recommendations.append(f"{name}: 质量 {mass:.6f}kg 可能过小，检查几何体积")
        elif mass > 10:
            issues.append("质量过大")
            recommendations.append(f"{name}: 质量 {mass:.2f}kg 可能过大，检查密度设置")
        
        # 检查惯性张量正定性
        eigenvalues = np.linalg.eigvalsh(inertia)
        if np.any(eigenvalues <= 0):
            issues.append("惯性张量非正定")
            recommendations.append(f"{name}: 惯性张量存在非正特征值")
        
        return {
            "passed": len(issues) == 0,
            "mass": mass,
            "inertia": inertia.tolist(),
            "center_of_mass": self.calculate_center_of_mass(mesh).tolist(),
            "issues": issues,
            "recommendations": recommendations
        }
```

---

### 6.6 风险 R5: Blender 版本兼容性

**问题描述**: Infinigen 使用 Blender 几何节点系统，不同版本可能导致节点行为差异。

**解决方案**:
```python
def verify_blender_version():
    """验证 Blender 版本兼容性"""
    import bpy
    version = bpy.app.version
    if version < (4, 0, 0):
        raise RuntimeError("需要 Blender 4.0+，低版本存在几何节点兼容性问题")
    return True
```

---

### 6.7 风险 R6: URDF 多关节导出（✅ 已修复，需回归）

**背景**：早期 `urdf_exporter.py` 对 `len(joint_nodes) > 1` 会抛出 `NotImplementedError`，导致多关节盒型无法导出。

**当前状态（v6.0）**：已在 `infinigen/core/sim/exporters/urdf_exporter.py` 实装多关节导出支持：

- 对同一 `parent_link -> child_link` 之间的 **多关节连接**，创建一个或多个 “intermediate link”（最小惯性/质量）来串联关节
- 保持 URDF 语义正确，并兼容后处理阶段的碰撞处理（通过 `exclude_links` 避免错误合并）

**实现示意（与当前代码一致）**：

```python
# urdf_exporter.py（示意）
if len(joint_nodes) > 0:
    current_parent_link = parent_link
    current_pos_offset = pos_offset

    for joint_idx, joint_node in enumerate(joint_nodes):
        is_last_joint = (joint_idx == len(joint_nodes) - 1)

        if is_last_joint:
            current_child_link = link_name
        else:
            # intermediate link with minimal inertial
            current_child_link = intermediate_link_name

        self._create_joint(
            parent_link=current_parent_link,
            child_link=current_child_link,
            origin=abs_joint_pos - current_pos_offset,
            axis=coord_frame @ axis,
            min_range=range_min,
            max_range=range_max,
        )

        current_parent_link = current_child_link
        current_pos_offset = abs_joint_pos
```

**回归测试建议**：

- 导出包含 multi-joint edge 的盒型（例如 RSC / 合成多关节链路用例），并确保：
  - URDF XML 合法
  - PyBullet/MuJoCo 可加载
  - 运动学链路拓扑正确（无孤立 link）

---

### 6.8 风险 R7: 碰撞网格质量

**问题描述**: 自动生成的碰撞网格可能过于复杂导致仿真性能下降。

**解决方案**:
```python
def simplify_collision_mesh(mesh: trimesh.Trimesh, 
                            target_faces: int = 500) -> trimesh.Trimesh:
    """简化碰撞网格以提高仿真性能"""
    if len(mesh.faces) > target_faces:
        simplified = mesh.simplify_quadratic_decimation(target_faces)
        return simplified
    return mesh

def generate_box_primitive_collider(panel_bounds: np.ndarray) -> Dict:
    """直接生成 Box Primitive 作为碰撞体（最稳健）"""
    center = panel_bounds.mean(axis=0)
    size = panel_bounds[1] - panel_bounds[0]
    return {"type": "box", "center": center.tolist(), "size": size.tolist()}
```

---

### 6.9 风险 R8: 关节动力学参数校准

**问题描述**: 阻尼、刚度等参数难以在没有实际仿真的情况下准确设定。

**当前实现（v6.0，已在代码中具备）**：

- 预设位置：`infinigen/core/sim/physics/joint_dynamics.py`
- 预设形式：`JointDynamicsParams` + `MATERIAL_JOINT_PRESETS`（材质默认）+ `BOX_TYPE_JOINT_PRESETS`（盒型/关节标签默认）

```python
from infinigen.core.sim.physics import joint_dynamics as jd

# 1) 材质默认（粗粒度）
d = jd.get_material_joint_dynamics(jd.MaterialCategory.CARDBOARD, jd.JointDynamicsType.HINGE)

# 2) 盒型 + joint label（细粒度，优先）
d = jd.get_box_type_joint_dynamics("TuckEndBox", "top_back_flap", material=jd.MaterialCategory.CARDBOARD)

# 3) URDF 导出需要的字段
urdf_dyn = d.to_urdf_dict()  # {"damping": ..., "friction": ...}
```

> [!NOTE]
> 当前 URDF 导出器使用 `sample_joint_params_fn`（例如 `TuckEndBoxFactory.sample_joint_parameters`）提供的 `joint_params`，
> 并通过 `get_joint_properties()` 输出 `<dynamics damping friction>`；后续可选将“预设回退”直接接入导出器，避免未配置时全 0。

**（可选增强）URDF 输出增强**：软限制 + 动力学参数（并非当前必需）
```xml
<joint name="lid_joint" type="revolute">
  <safety_controller k_position="0.5" k_velocity="0.1"/>
  <dynamics damping="0.2" friction="0.5"/>
</joint>
```

---

### 6.10 风险 R-Deep-1: 薄壳惯性张量不稳定（✅ 已修复，需回归）

**背景**：薄壁面板（卡纸盖子、dust flap 等）体积接近 0，若直接用体积法计算质量/惯性，会出现：

- `mass ≈ 0` → 物理引擎产生极端加速度 → 仿真不稳定/“爆炸”
- 惯性张量非正定/过小 → 引擎拒绝加载或数值抖动

**当前状态（v6.0）**：已在 `infinigen/core/sim/physics/thin_shell_inertia.py` 实装薄壳稳健惯性，并在 `urdf_exporter.py` 中集成调用。

**实现要点（与当前代码一致）**：

- 通过 `volume / surface_area` 比值判断薄壳（对应“等效厚度”）
- 对薄壳使用 `surface_area × MIN_EFFECTIVE_THICKNESS` 估算质量
- 使用边界框尺寸计算等效薄板惯性，并设置 `MIN_INERTIA_VALUE` 保底
- 强制惯性张量正定；并设置最小质量（默认最小 1g）

**回归测试建议**：

- 使用 PyBullet DIRECT 加载导出的 URDF，并运行若干步仿真，检查位置/速度不发散
- 对极薄厚度参数（例如 0.5mm 级别）做专门回归，确保仍稳定

---

### 6.11 风险 R-Deep-2: 几何节点单一化架构

**问题描述**: 
现有 `box.py` 是**单一整块**设计 (`nodegroup_geometry_nodes`)，通过 `Switch` 节点隐藏/显示部分。
- 内存与性能：加载包含所有 16 种变体逻辑的超大节点树非常低效
- 维护噩梦：在一个节点树里维护 16 种拓扑几乎不可能

**解决方案**: 重构为**动态节点树加载模式**
```python
# 为每种盒型族创建独立的节点生成函数
BOX_TYPE_GENERATORS = {
    "TuckEndBox": geometry_nodes_tuck_box,
    "LockBottomBox": geometry_nodes_lock_bottom_box,
    "MailerBox": geometry_nodes_mailer_box,
    "DrawerBox": geometry_nodes_drawer_box,
    "SlipLidBox": geometry_nodes_slip_lid_box,
    "RSCBox": geometry_nodes_rsc_box,
    "GableTopBox": geometry_nodes_gable_top_box,
    "HandleGiftBox": geometry_nodes_handle_gift_box,
}

class ModularBoxFactory(AssetFactory):
    """模块化盒子工厂：按需加载节点树"""
    
    def create_asset(self, params: Dict):
        box_type = params.get("box_type", "TuckEndBox")
        
        # 动态获取对应的节点生成函数
        generator_fn = BOX_TYPE_GENERATORS.get(box_type)
        if generator_fn is None:
            raise ValueError(f"Unknown box type: {box_type}")
        
        # 创建并应用节点组
        obj = butil.spawn_vert()
        butil.modify_mesh(
            obj,
            "NODES",
            apply=False,
            node_group=generator_fn(),  # 动态调用
            ng_inputs=params,
        )
        return obj
```

---

## 7. 材质多样化系统

### 7.1 材质系统架构

```python
class MaterialSystem:
    """
    材质管理系统
    
    功能:
    1. 材质注册与管理
    2. 物理属性映射
    3. 视觉材质生成
    4. 参数随机采样
    """
    
    def __init__(self):
        self.materials = {}
        self._register_default_materials()
    
    def _register_default_materials(self):
        """注册默认材质"""
        
        # 1. 卡纸材质
        self.register("cardboard", CardboardMaterial())
        
        # 2. 瓦楞纸材质  
        self.register("corrugated", CorrugatedMaterial())
        
        # 3. 塑料材质
        self.register("plastic_pp", PlasticMaterial("PP"))
        self.register("plastic_pet", PlasticMaterial("PET"))
        
        # 4. 木质材质
        self.register("wood_mdf", WoodMaterial("MDF"))
        self.register("wood_plywood", WoodMaterial("Plywood"))
    
    def register(self, name: str, material: "BaseMaterial"):
        self.materials[name] = material
    
    def get(self, name: str) -> "BaseMaterial":
        return self.materials.get(name)
    
    def sample_random(self) -> "BaseMaterial":
        """随机采样一种材质"""
        name = random.choice(list(self.materials.keys()))
        return self.materials[name]


class BaseMaterial:
    """材质基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    @property
    def physical_properties(self) -> Dict:
        """物理属性"""
        raise NotImplementedError
    
    @property
    def visual_properties(self) -> Dict:
        """视觉属性 (用于渲染)"""
        raise NotImplementedError
    
    @property
    def thickness_range(self) -> Tuple[float, float]:
        """厚度范围 (m)"""
        raise NotImplementedError
    
    def sample_thickness(self) -> float:
        """采样厚度"""
        return random.uniform(*self.thickness_range)
    
    def create_blender_material(self) -> bpy.types.Material:
        """创建 Blender 材质节点"""
        raise NotImplementedError


class CardboardMaterial(BaseMaterial):
    """卡纸材质"""
    
    # 预定义颜色变体
    COLOR_VARIANTS = [
        (0.76, 0.60, 0.42),  # 牛皮纸色
        (0.95, 0.95, 0.90),  # 白卡
        (0.85, 0.75, 0.65),  # 灰卡
        (0.70, 0.55, 0.40),  # 深牛皮
    ]
    
    def __init__(self, color_index: int = None):
        super().__init__("cardboard")
        self.color = self.COLOR_VARIANTS[color_index] if color_index else random.choice(self.COLOR_VARIANTS)
    
    @property
    def physical_properties(self) -> Dict:
        return {
            "density": 700,        # kg/m³
            "friction": 0.5,
            "restitution": 0.1,
            "stiffness": 1e6,      # Pa
        }
    
    @property
    def visual_properties(self) -> Dict:
        return {
            "base_color": self.color,
            "roughness": 0.8,
            "metallic": 0.0,
            "normal_strength": 0.3,  # 纸纹理
        }
    
    @property
    def thickness_range(self) -> Tuple[float, float]:
        return (0.0003, 0.002)  # 0.3mm - 2mm


class CorrugatedMaterial(BaseMaterial):
    """瓦楞纸材质"""
    
    # 瓦楞类型
    FLUTE_TYPES = {
        "A": {"thickness": 0.0047, "density": 130},
        "B": {"thickness": 0.0025, "density": 150},
        "C": {"thickness": 0.0036, "density": 140},
        "E": {"thickness": 0.0015, "density": 180},
    }
    
    def __init__(self, flute_type: str = "B"):
        super().__init__("corrugated")
        self.flute = self.FLUTE_TYPES.get(flute_type, self.FLUTE_TYPES["B"])
    
    @property
    def physical_properties(self) -> Dict:
        return {
            "density": self.flute["density"],
            "friction": 0.6,
            "restitution": 0.2,
        }
    
    @property
    def visual_properties(self) -> Dict:
        return {
            "base_color": (0.72, 0.56, 0.40),  # 牛皮纸色
            "roughness": 0.9,
            "metallic": 0.0,
        }
    
    @property
    def thickness_range(self) -> Tuple[float, float]:
        t = self.flute["thickness"]
        return (t * 0.9, t * 1.1)


class PlasticMaterial(BaseMaterial):
    """塑料材质"""
    
    PLASTIC_TYPES = {
        "PP": {"density": 900, "friction": 0.3, "color": (0.95, 0.95, 0.95)},
        "PET": {"density": 1380, "friction": 0.35, "color": (0.9, 0.92, 0.95)},
        "HDPE": {"density": 950, "friction": 0.25, "color": (0.85, 0.85, 0.85)},
    }
    
    def __init__(self, plastic_type: str = "PP"):
        super().__init__(f"plastic_{plastic_type}")
        self.props = self.PLASTIC_TYPES.get(plastic_type, self.PLASTIC_TYPES["PP"])
    
    @property
    def physical_properties(self) -> Dict:
        return {
            "density": self.props["density"],
            "friction": self.props["friction"],
            "restitution": 0.4,
        }
    
    @property  
    def visual_properties(self) -> Dict:
        return {
            "base_color": self.props["color"],
            "roughness": 0.2,
            "metallic": 0.0,
            "transmission": 0.1,  # 轻微透明
        }
    
    @property
    def thickness_range(self) -> Tuple[float, float]:
        return (0.0005, 0.003)
```

---

## 8. 批量生成系统

### 8.1 批量生成架构

```python
class BatchBoxGenerator:
    """
    批量盒子生成器
    
    功能:
    1. 参数空间采样
    2. 并行生成
    3. 自动验证
    4. 失败重试
    5. 生成报告
    """
    
    def __init__(self, 
                 box_types: List[str] = None,
                 material_system: MaterialSystem = None,
                 validation_pipeline: BoxValidationPipeline = None,
                 output_dir: str = "generated_boxes"):
        
        self.box_types = box_types or ["TuckEndBox", "MailerBox", "DrawerBox", "RSCBox"]
        self.material_system = material_system or MaterialSystem()
        self.validation_pipeline = validation_pipeline or BoxValidationPipeline()
        self.output_dir = output_dir
        
        # 统计信息
        self.stats = {
            "total_attempts": 0,
            "successful": 0,
            "failed_validation": 0,
            "failed_generation": 0
        }
    
    def generate_batch(self, 
                       count: int,
                       box_type: str = None,
                       material: str = None,
                       dimension_range: Dict = None,
                       max_retries: int = 3,
                       parallel: bool = True) -> List[str]:
        """
        批量生成盒子资产
        
        Args:
            count: 生成数量
            box_type: 盒子类型 (None = 随机)
            material: 材质类型 (None = 随机)
            dimension_range: 尺寸范围 {"width": (min, max), ...}
            max_retries: 最大重试次数
            parallel: 是否并行生成
            
        Returns:
            成功生成的 URDF 文件路径列表
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        generated_paths = []
        
        for i in range(count):
            success = False
            retries = 0
            
            while not success and retries < max_retries:
                self.stats["total_attempts"] += 1
                
                try:
                    # 1. 采样参数
                    params = self._sample_parameters(
                        box_type, material, dimension_range
                    )
                    
                    # 2. 生成盒子
                    urdf_path = self._generate_single(params, i)
                    
                    # 3. 验证
                    validation_result = self.validation_pipeline.validate(urdf_path, verbose=False)
                    
                    if validation_result["overall_passed"]:
                        generated_paths.append(urdf_path)
                        self.stats["successful"] += 1
                        success = True
                        print(f"✅ [{i+1}/{count}] 生成成功: {urdf_path}")
                    else:
                        self.stats["failed_validation"] += 1
                        retries += 1
                        print(f"⚠️  [{i+1}/{count}] 验证失败, 重试 {retries}/{max_retries}")
                        # 删除失败的文件
                        self._cleanup(urdf_path)
                        
                except Exception as e:
                    self.stats["failed_generation"] += 1
                    retries += 1
                    print(f"❌ [{i+1}/{count}] 生成错误: {e}, 重试 {retries}/{max_retries}")
            
            if not success:
                print(f"❌ [{i+1}/{count}] 达到最大重试次数, 跳过")
        
        # 生成报告
        self._generate_batch_report(generated_paths)
        
        return generated_paths
    
    def _sample_parameters(self, box_type: str, material: str, 
                           dimension_range: Dict) -> Dict:
        """采样生成参数"""
        
        # 盒子类型
        if box_type is None:
            box_type = random.choice(self.box_types)
        
        # 材质
        if material is None:
            mat = self.material_system.sample_random()
        else:
            mat = self.material_system.get(material)
        
        # 尺寸
        dim_range = dimension_range or {
            "width": (0.1, 0.5),
            "depth": (0.1, 0.5),
            "height": (0.1, 0.5),
        }
        
        dimensions = {
            "width": random.uniform(*dim_range["width"]),
            "depth": random.uniform(*dim_range["depth"]),
            "height": random.uniform(*dim_range["height"]),
        }
        
        # 厚度 (基于材质)
        thickness = mat.sample_thickness()
        
        return {
            "box_type": box_type,
            "material": mat,
            "dimensions": dimensions,
            "thickness": thickness,
            "seed": random.randint(0, 2**31),
        }
    
    def _generate_single(self, params: Dict, index: int) -> str:
        """生成单个盒子"""
        
        # 调用对应的 BoxFactory
        factory_class = self._get_factory_class(params["box_type"])
        factory = factory_class(factory_seed=params["seed"])
        
        # 注入参数
        factory.set_dimensions(params["dimensions"])
        factory.set_thickness(params["thickness"])
        factory.set_material(params["material"])
        
        # 生成资产
        asset = factory.create_asset()
        
        # 导出 URDF
        urdf_dir = os.path.join(
            self.output_dir, 
            params["box_type"],
            f"seed_{params['seed']}"
        )
        os.makedirs(urdf_dir, exist_ok=True)
        
        urdf_path = export_urdf(asset, urdf_dir)
        
        return urdf_path
    
    def _generate_batch_report(self, generated_paths: List[str]):
        """生成批量生成报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.stats,
            "success_rate": self.stats["successful"] / max(self.stats["total_attempts"], 1),
            "generated_files": generated_paths,
            "by_type": self._count_by_type(generated_paths),
        }
        
        report_path = os.path.join(self.output_dir, "batch_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 批量生成完成!")
        print(f"   成功: {self.stats['successful']}")
        print(f"   验证失败: {self.stats['failed_validation']}")
        print(f"   生成错误: {self.stats['failed_generation']}")
        print(f"   成功率: {report['success_rate']:.1%}")
        print(f"   报告: {report_path}")


class ParameterSpaceSampler:
    """
    参数空间采样器
    
    支持多种采样策略:
    - 均匀采样
    - 拉丁超立方采样
    - 网格采样
    """
    
    def __init__(self, parameter_ranges: Dict):
        """
        Args:
            parameter_ranges: {
                "width": (0.1, 0.5),
                "depth": (0.1, 0.5),
                "height": (0.1, 0.5),
                "thickness": (0.001, 0.005),
            }
        """
        self.ranges = parameter_ranges
    
    def uniform_sample(self, n: int) -> List[Dict]:
        """均匀随机采样"""
        samples = []
        for _ in range(n):
            sample = {k: random.uniform(*v) for k, v in self.ranges.items()}
            samples.append(sample)
        return samples
    
    def latin_hypercube_sample(self, n: int) -> List[Dict]:
        """
        拉丁超立方采样
        
        确保参数空间的均匀覆盖
        """
        dimensions = len(self.ranges)
        keys = list(self.ranges.keys())
        
        # 生成 LHS 样本
        lhs_samples = self._generate_lhs(n, dimensions)
        
        # 映射到实际范围
        samples = []
        for lhs_sample in lhs_samples:
            sample = {}
            for i, key in enumerate(keys):
                low, high = self.ranges[key]
                sample[key] = low + lhs_sample[i] * (high - low)
            samples.append(sample)
        
        return samples
    
    def _generate_lhs(self, n: int, d: int) -> np.ndarray:
        """生成拉丁超立方样本"""
        samples = np.zeros((n, d))
        for i in range(d):
            perm = np.random.permutation(n)
            samples[:, i] = (perm + np.random.rand(n)) / n
        return samples
    
    def grid_sample(self, points_per_dim: int) -> List[Dict]:
        """网格采样"""
        keys = list(self.ranges.keys())
        grids = []
        
        for key in keys:
            low, high = self.ranges[key]
            grids.append(np.linspace(low, high, points_per_dim))
        
        # 生成网格点
        mesh = np.meshgrid(*grids)
        samples = []
        
        for idx in np.ndindex(*[points_per_dim] * len(keys)):
            sample = {keys[i]: mesh[i][idx] for i in range(len(keys))}
            samples.append(sample)
        
        return samples
```

---

## 9. 后续扩展方向

1. **装饰元素**: 添加把手、锁扣、标签等装饰部件
2. **参数化尺寸 API**: 支持用户自定义尺寸规格接口
3. **仿真集成**: 与 MuJoCo/PyBullet/Isaac Sim 深度集成
4. **数据增强**: 添加纹理变换、光照变化等增强

---

## 10. 验证检查点清单

### 10.1 每个盒子资产必须通过的检查

| 检查ID | 检查项 | 验证器 | 阈值 | 可检测 | 可反馈 |
|--------|-------|--------|------|--------|--------|
| C1 | 网格闭合性 | GeometryValidator | 100% watertight | ✅ | ✅ |
| C2 | 网格体积正值 | GeometryValidator | volume > 0 | ✅ | ✅ |
| C3 | 初始无碰撞 | CollisionChecker | 0 collisions | ✅ | ✅ |
| C4 | 关节位置精度 | PrecisionJointPlacer | error < 1mm | ✅ | ✅ |
| C5 | 运动范围安全 | KinematicsValidator | 0 collisions in range | ✅ | ✅ |
| C6 | 运动学树有效 | KinematicTreeBuilder | no cycles | ✅ | ✅ |
| C7 | 质量合理 | PhysicalParameterCalculator | 0.001-10 kg | ✅ | ✅ |
| C8 | 惯性张量正定 | PhysicalParameterCalculator | all eigenvalues > 0 | ✅ | ✅ |

### 10.2 批量生成必须通过的检查

| 检查ID | 检查项 | 验证器 | 阈值 |
|--------|-------|--------|------|
| B1 | 单个资产验证通过率 | BatchBoxGenerator | > 95% |
| B2 | 盒型分布均匀性 | ParameterSpaceSampler | χ² < 0.05 |
| B3 | 尺寸覆盖度 | ParameterSpaceSampler | LHS coverage |
| B4 | 材质多样性 | MaterialSystem | all materials used |

### 10.3 验证报告示例

```json
{
  "asset_id": "TuckEndBox_seed_12345",
  "timestamp": "2026-01-08T10:30:00",
  "overall_passed": true,
  "checks": {
    "C1_mesh_watertight": {"passed": true, "details": "all 5 meshes watertight"},
    "C2_positive_volume": {"passed": true, "details": "all volumes positive"},
    "C3_no_initial_collision": {"passed": true, "details": "0 collision pairs"},
    "C4_joint_precision": {"passed": true, "details": "max error 0.3mm"},
    "C5_motion_safe": {"passed": true, "details": "all 2 joints safe"},
    "C6_kinematic_tree": {"passed": true, "details": "tree valid, depth 3"},
    "C7_mass_valid": {"passed": true, "details": "total mass 0.045kg"},
    "C8_inertia_valid": {"passed": true, "details": "all positive definite"}
  },
  "generation_params": {
    "box_type": "TuckEndBox",
    "material": "cardboard",
    "dimensions": {"width": 0.2, "depth": 0.15, "height": 0.1},
    "thickness": 0.002
  }
}
```

---

## 11. 参考资源

- Infinigen 现有资产: `infinigen/assets/sim_objects/box.py`
- 关节系统: `infinigen/assets/utils/joints.py`
- URDF导出器: `infinigen/core/sim/exporters/urdf_exporter.py`
- 运动学编译器: `infinigen/core/sim/kinematic_compiler.py`

---

---

## 12. 详细开发工作计划（PM 可追溯版）

> [!NOTE]
> 本章节为 PM 提供可追溯、可回滚的详细开发计划。
> 每个 Task 包含：交付物、验收标准、依赖关系、预估时间。

### 12.1 开发里程碑总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         开发里程碑 (6-8 周)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ M0: 环境就绪    │ M1: 稳定性基础  │ M2: 核心框架   │ M3: 基础盒型完成      │
│ (0.5 天)        │ (1 周)          │ (2 周)         │ (2-3 周)               │
│                 │                 │                │                        │
│ ✓ Blender 4.0+  │ ✓ R6 修复       │ ✓ 模块化工厂   │ ✓ 7 种基础盒型         │
│ ✓ PyBullet      │ ✓ R-Deep-1 修复 │ ✓ 材质系统     │ ✓ 程序化验证通过       │
│ ✓ trimesh       │ ✓ 碰撞网格优化  │ ✓ 关节注入系统 │ ✓ 批量生成测试         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 12.2 Phase 0: 稳定性基础（✅ 已完成 / 需回归）

**目标**：作为所有可动盒子 URDF 的基座能力（多关节导出 + 薄壳惯性），已完成但必须持续回归测试，避免回退。

| Task ID | 任务 | 交付物 | 验收标准 | 依赖 | 预估 |
|---------|------|--------|----------|------|------|
| **P0-T1** | 修复 URDF 多关节限制 (R6) | `urdf_exporter.py` 修改版 | `pytest -q tests/sim/test_urdf_multi_joint.py` 通过 | - | 2天 |
| **P0-T2** | 实现薄壳惯性修正 (R-Deep-1) | `thin_shell_inertia.py` 新模块 | `pytest -q tests/sim/test_thin_shell_inertia.py` 通过（PyBullet 稳定） | - | 2天 |
| **P0-T3** | Blender 版本检查 (R5) | 启动时检查逻辑 | 低版本时抛出明确错误 | - | 0.5天 |
| **P0-T4** | 碰撞网格简化 (R7) | `collision_utils.py` 新模块 | 简化后面数 < 500，保持形状一致 | - | 1天 |
| **P0-T5** | Phase 0 集成测试 | `tests/sim/test_phase0_integration.py` | 所有 T1-T4 验收标准通过 | T1-T4 | 0.5天 |

**P0-T1 详细实现步骤**:
```
1. 定位 `URDFBuilder._populate_links()`
2. 将“不支持多关节”的逻辑替换为“多关节串联导出”（intermediate links）
3. 保证每个 joint 的 `origin = abs_joint_pos - current_pos_offset`，并在创建后更新 `current_pos_offset = abs_joint_pos`
4. 回归建议：导出包含 multi-joint edge 的 URDF，并用 PyBullet/MuJoCo 加载验证（XML 合法 + 链路拓扑正确）
```

**P0-T2 详细实现步骤**:
```
1. 创建 infinigen/core/sim/physics/thin_shell_inertia.py
2. 实现 `calculate_robust_inertia(vertices, faces, density, volume=None)` + 正定/最小值修正
3. 在 urdf_exporter.py 中替换现有惯性计算
4. 回归建议：PyBullet DIRECT 1000 steps 稳定（含薄壳极限厚度用例）
```

**PM 检查点 (Phase 0 结束 / v6.0 现状)**:
- [x] urdf_exporter.py 多关节支持代码已合并
- [x] thin_shell_inertia.py 模块已创建并接入 exporter
- [x] 自动化回归测试已覆盖（例如 `tests/sim/test_urdf_multi_joint.py`, `tests/sim/test_thin_shell_inertia.py`, `tests/sim/test_phase0_integration.py`）
- [ ] Git commit / tag（由项目维护者按流程补齐）

---

### 12.3 Phase 1: 核心框架 + 材质系统

**目标**: 建立可扩展的模块化盒子生成架构

| Task ID | 任务 | 交付物 | 验收标准 | 依赖 | 预估 |
|---------|------|--------|----------|------|------|
| **P1-T1** | ModularBoxFactory 基类 | `modular_box_factory.py` | 动态加载节点生成函数成功 | P0 | 2天 |
| **P1-T2** | 基础几何模块 | `box_geometry_modules.py` | 底/侧/顶面板生成正确 | P1-T1 | 1天 |
| **P1-T3** | 关节注入系统 | `joint_injector.py` | 铰链/滑轨正确附加到面板边缘 | P1-T2 | 2天 |
| **P1-T4** | 材质系统 | 扩展 `material_definitions.py` | Cardboard/Corrugated 材质可用 | P0 | 1天 |
| **P1-T5** | 关节动力学预设 | 扩展 `joint_dynamics.py` | 材料相关 damping/friction 输出 | P1-T4 | 0.5天 |
| **P1-T6** | 批量生成器框架 | `batch_generator.py` | 支持参数采样和验证管线 | P1-T1~T5 | 1.5天 |
| **P1-T7** | Phase 1 集成测试 | `test_phase1.py` | 生成简单盒子并导出 URDF 成功 | P1-T1~T6 | 1天 |

**P1-T1 ModularBoxFactory 骨架**:
```python
# infinigen/assets/sim_objects/modular_box_factory.py
from infinigen.core.placement.factory import AssetFactory

BOX_TYPE_GENERATORS = {}  # 将在后续 Phase 2 填充

class ModularBoxFactory(AssetFactory):
    def __init__(self, factory_seed, box_type="TuckEndBox", coarse=False):
        super().__init__(factory_seed, coarse)
        self.box_type = box_type
        self.material_system = MaterialSystem()
    
    def sample_parameters(self):
        generator_fn = BOX_TYPE_GENERATORS.get(self.box_type)
        if generator_fn is None:
            raise ValueError(f"Unknown box type: {self.box_type}")
        return self._sample_dimensions()
    
    def create_asset(self, **params):
        generator_fn = BOX_TYPE_GENERATORS[self.box_type]
        obj = butil.spawn_vert()
        butil.modify_mesh(obj, "NODES", node_group=generator_fn(**params))
        return obj
```

**P1-T4 材质系统扩展** (添加到 `material_definitions.py`):
```python
@dataclass
class Cardboard(BaseMaterial):
    min_friction: float = 0.4
    max_friction: float = 0.6
    min_density: float = 600   # kg/m³
    max_density: float = 800   # kg/m³

@dataclass
class Corrugated(BaseMaterial):
    min_friction: float = 0.5
    max_friction: float = 0.7
    min_density: float = 100   # kg/m³ (含空气)
    max_density: float = 200   # kg/m³

# 添加到 MATERIALS 注册表
MATERIALS["cardboard"] = Cardboard
MATERIALS["corrugated"] = Corrugated
```

**PM 检查点 (Phase 1 结束)**:
- [ ] ModularBoxFactory 类已创建并可实例化
- [ ] Cardboard/Corrugated 材质已添加到注册表
- [ ] test_phase1.py 生成简单盒子成功
- [ ] Git commit: `feat(assets): Phase 1 modular box framework`

---

### 12.4 Phase 2.1: 基础盒型实现 (7 种 1-2 星难度)

**目标**: 完成 7 种低复杂度盒型，验证整个管线

**优先级顺序** (从简到难):
1. **SlipLidBox** (天地盒) - 0-1 关节，最简单
2. **MailerBox** (飞机盒) - 1-2 关节
3. **DrawerBox** (抽屉盒) - 1 滑轨关节
4. **TuckEndBox** (双插盒) - 6-8 关节（含插舌二段）
5. **GiftBoxWithHandle** (自带手提礼盒) - 1-2 关节
6. **DoubleLidGiftBox** (双盖手提礼盒) - 2 关节
7. **PlasticHandleBox** (塑料手提礼盒) - 1-2 关节

| Task ID | 盒型 | 交付物 | 验收标准 | 依赖 | 预估 |
|---------|------|--------|----------|------|------|
| **P2.1-T1** | SlipLidBox | `geometry_nodes_slip_lid_box.py` | URDF 导出 + 验证通过 | P1 | 1天 |
| **P2.1-T2** | MailerBox | `geometry_nodes_mailer_box.py` | URDF 导出 + 关节运动范围正确 | P1 | 1.5天 |
| **P2.1-T3** | DrawerBox | `geometry_nodes_drawer_box.py` | 滑轨关节导出 + 滑动范围正确 | P1 | 1天 |
| **P2.1-T4** | TuckEndBox | `geometry_nodes_tuck_box.py` | 6-8 关节导出（含插舌二段） + 无初始碰撞 | P2.1-T1~T3 | 2天 |
| **P2.1-T5** | GiftBoxWithHandle | `geometry_nodes_gift_box.py` | 关节 + 手柄几何正确 | P2.1-T4 | 1.5天 |
| **P2.1-T6** | DoubleLidGiftBox | `geometry_nodes_double_lid.py` | 双铰链同步正确 | P2.1-T5 | 1天 |
| **P2.1-T7** | PlasticHandleBox | `geometry_nodes_plastic_handle.py` | 塑料材质属性正确 | P2.1-T6 | 1天 |
| **P2.1-T8** | Phase 2.1 批量验证 | 验证报告 JSON | 7 种盒型各 10 个变体全部通过 | P2.1-T1~T7 | 1天 |

**每个盒型的验收检查清单**:
```
□ 几何验证 (GeometryValidator):
  □ 网格闭合 (watertight)
  □ 体积正值
  □ 无初始碰撞
  
□ 运动学验证 (KinematicsValidator):
  □ 关节位置在边缘上
  □ 运动范围内无碰撞
  □ 运动学树无环
  
□ URDF 验证:
  □ XML 语法正确
  □ 关节名称唯一
  □ Mesh 文件全部存在
  
□ 物理验证:
  □ 质量范围合理 (0.001-10 kg)
  □ 惯性张量正定
```

**PM 检查点 (Phase 2.1 结束)**:
- [ ] 7 种基础盒型全部实现
- [ ] 每种盒型 10 个变体程序化验证通过
- [ ] 批量验证报告已生成
- [ ] Git commit: `feat(assets): Phase 2.1 - 7 basic box types complete`

---

### 12.5 可追溯性与回滚策略

**Git 分支策略**:
```
main
  └── feature/box-urdf-generation
        ├── phase0-stability
        ├── phase1-framework
        └── phase2.1-basic-boxes
```

**每个 Task 的 Commit 规范**:
```
<type>(<scope>): <description>

type: feat | fix | refactor | test | docs
scope: sim | assets | physics | validation

Example:
feat(sim): fix URDF multi-joint export - P0-T1
fix(physics): thin shell inertia correction - P0-T2
feat(assets): add TuckEndBox generator - P2.1-T4
```

**回滚脚本** (`scripts/rollback_to_checkpoint.sh`):
```bash
#!/bin/bash
# Usage: ./rollback_to_checkpoint.sh <phase_tag>
# Example: ./rollback_to_checkpoint.sh phase0-complete

TAG=$1
if [ -z "$TAG" ]; then
    echo "Usage: ./rollback_to_checkpoint.sh <phase_tag>"
    exit 1
fi

git checkout $TAG
echo "✅ Rolled back to $TAG"
git log -1 --oneline
```

**Phase 标签**:
```bash
git tag phase0-complete  # Phase 0 稳定性修复完成
git tag phase1-complete  # Phase 1 核心框架完成
git tag phase2.1-complete  # Phase 2.1 基础盒型完成
```

---

### 12.6 PM 每日/每周检查表

**每日站会检查点**:
- [ ] 当前 Task 进度 (% 完成)
- [ ] 是否有阻塞问题
- [ ] 是否需要设计决策

**每周里程碑检查点**:
- [ ] 本周计划 Tasks 完成情况
- [ ] 验证测试通过率
- [ ] 代码已提交并 push
- [ ] 文档更新同步

**最终验收标准** (Phase 2.1 结束):
- [ ] 7 种基础盒型 URDF 导出成功
- [ ] 每种盒型 10 个变体程序化验证 100% 通过
- [ ] PyBullet 加载所有 URDF 无错误
- [ ] 批量生成报告 success_rate > 95%
- [ ] 所有代码有对应测试
- [ ] 文档与代码同步

---

**文档版本**: v6.0
**创建日期**: 2026-01-08
**更新日期**: 2026-01-12
**状态**: ✅ 已与当前实现对齐（TuckEndBox 已导出并在线验证），可作为后续盒型开发的可执行指南

### 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| v1.0 | 2026-01-08 | 初始方案 |
| v2.0 | 2026-01-08 | 1) 新增程序化验证系统(无需外部仿真器)<br>2) 详细展开4个技术风险的算法解决方案<br>3) 新增材质多样化系统(当前阶段)<br>4) 新增批量生成系统(当前阶段)<br>5) 新增验证检查点清单 |
| v3.0 | 2026-01-08 | **根据专家评审完善**:<br>1) 修正 RSC 盒子关节轴向配置 (统一轴向，相反范围)<br>2) 新增风险 R5-R8: Blender版本兼容性、URDF导出器多关节限制(阻断性)、碰撞网格质量、关节动力学校准<br>3) 新增致命风险 R-Deep-1: 薄壳惯性张量不稳定 + 修正算法<br>4) 新增架构风险 R-Deep-2: 几何节点单一化 + 模块化重构方案<br>5) 增强 GeometryValidator: 新增7项检查<br>6) 新增关节回弹(Hysteresis)与软限制建模<br>7) 新增碰撞网格凸分解方案 |
| v4.0 | 2026-01-08 | **文档重构为开发指南**:<br>1) **重构 Section 3.1**: 集成 ModularBoxFactory 架构 (R-Deep-2)、MaterialSystem、风险解决方案交叉引用<br>2) **重构 Section 4.1**: 新增 Phase 0 稳定性基础阶段, 将 R6/R-Deep-1 设为开发第一优先级<br>3) 集成材质系统详情到 Phase 1.4, 批量生成到 Phase 1.5<br>4) 验证所有章节编号 (7-8-9-10-11) 正确连续<br>5) 文档现可作为 Cursor 开发的全面详细开发文档 |
| **v5.0** | **2026-01-10** | **深度代码审查 + PM 工作计划**:<br>1) **重构 Section 2**: 深入代码审查，验证分析覆盖 Section 4/6 需求<br>2) **发现关键缺失**: material_definitions.py 缺少 Cardboard/Corrugated 材质<br>3) **新增 Section 4.0**: 开发环境准备 (PyBullet/Blender 配置)<br>4) **新增 Section 12**: PM 可追溯详细开发工作计划<br>5) 为每个 Task 添加交付物、验收标准、预估时间<br>6) 添加 Git 分支策略和回滚机制 |
| **v6.0** | **2026-01-12** | **对齐实装结果（TuckEndBox 已验证）**:<br>1) 更新 TuckEndBox 方案：关节规模统一为 6-8（当前实现 8），补充插舌二段铰链与推荐折叠顺序<br>2) 修正文档中 R6/R-Deep-1 状态：已修复并给出实现位置与 pytest 回归路径<br>3) 补充 hinge-local 建模规范与 visual origin 语义（解决折页悬浮/错位）<br>4) 更新 URDF 输出规范与目录结构为当前 exporter 实际输出<br>5) 更新材质系统现状：Cardboard/Corrugated 已在 material_definitions 中具备并可用 |