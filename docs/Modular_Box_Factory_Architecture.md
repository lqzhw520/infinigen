# Modular Box Factory Architecture（盒型工厂可扩展架构）

本文件给出 **`ModularBoxFactory` 盒型工厂体系**的工程化结构说明：如何在 Infinigen 中以统一方式扩展盒型几何、关节、物理材质，并与 URDF 导出与 Phase-1 数据引擎（自动标注）对齐。

> 目标：在保证 **URDF 物理一致性**（单 `<inertial>` / link、正确的 joint limit 与轴、可加载 self-collision）与 **数据标注一致性**（link-id ↔ URDF link name 可读映射）的前提下，支持扩展到约 16 种盒型（见 `BoxType`）。

---

## 1. 关键文件与数据流（从几何到 URDF/数据集）

- **盒型工厂与参数定义**
  - `infinigen/assets/sim_objects/modular_box_factory.py`
    - `BoxType`：盒型枚举（预留 16 类）
    - `BoxDimensions` / `BoxMaterialConfig` / `BoxJointConfig` / `BoxParameters`：统一参数数据结构
    - `ModularBoxFactory`：抽象基类 + registry（`register()` / `get_factory()`）
    - 已实现示例：`TuckEndBoxFactory` / `MailerBoxFactory` / `DrawerBoxFactory` / `SlipLidBoxFactory`

- **关节注入（几何节点 -> 运动学图/URDF）**
  - `infinigen/assets/utils/joints.py`
    - `nodegroup_hinge_joint(...)`：铰链（revolute）注入
    - `nodegroup_sliding_joint(...)`：滑动（prismatic）注入

- **URDF 导出（物理严谨性）**
  - `infinigen/core/sim/exporters/urdf_exporter.py`
    - 将编译得到的运动学图（`kinematic_compiler.compile(...)`）导出为 URDF
    - 重要约束：**每个 `<link>` 至多一个 `<inertial>`**，并使用平行轴定理合并多个几何体惯性

- **Phase-1 数据引擎（自动标注）**
  - `scripts/export_mailerbox_simple_phase1_data_engine.py`
    - 入口：`--box_type {MAILER,DRAWER,SLIP_LID,TUCK_END,...}`
    - 每个 sample 输出：RGB、Depth、link-seg、instance、keypoints、camera intrinsics、URDF GT、metadata、`segmentation_label_map.json`

---

## 2. 统一参数模型（为什么必须统一）

`modular_box_factory.py` 定义了跨盒型通用的参数结构：

- **`BoxDimensions`**：\(W,D,H,T\)（宽/深/高/厚），并提供合理性校验。
- **`BoxMaterialConfig`**：将**视觉材质**与**物理材质**统一绑定（尤其是 `density/friction/restitution/material_type`）。
  - 核心原因：URDF inertia 的质量来自几何体体积 × **密度**，若不统一，会出现“视觉材质≠物理材质”导致的物理偏差。
- **`BoxJointConfig`**：盒型关节统一配置（hinge/slide、axis、limit、damping/friction）。
- **`BoxParameters`**：将以上统一封装，并允许 `extra_params` 承载盒型特化参数（如插舌长度、扣锁结构等）。

工程实践建议：

- **只在 `extra_params` 中放“结构特化参数”**，避免把通用参数分散到多处。
- `sample_parameters()` 产出应当是**完全可复现**的（由 `factory_seed` 控制）。

---

## 3. 工厂扩展模式（添加新盒型的最小步骤）

以 `DrawerBoxFactory` / `SlipLidBoxFactory` 为模板，新增盒型建议遵循以下步骤：

1. **在 `BoxType` 中选择/添加枚举值**
2. **实现工厂子类并注册**

```python
@ModularBoxFactory.register(BoxType.YOUR_TYPE)
class YourBoxFactory(ModularBoxFactory):
    def get_box_type(self) -> BoxType:
        return BoxType.YOUR_TYPE

    def get_default_joints(self, params: BoxParameters) -> List[BoxJointConfig]:
        # 返回该盒型默认 DOF 与 limit/dynamics
        ...

    def create_geometry_nodegroup(self, nw: NodeWrangler, params: BoxParameters):
        # 用几何节点构建薄壳面板，并在可动部件上注入 hinge/slide joint nodegroup
        ...
```

3. **保持“薄壳建模”一致性**
   - 不要用整块实心 `MeshCube` 直接当盒体（会导致体积/质量严重偏大）。
   - 推荐用多片薄板（bottom/back/left/right/front/top）构建盒体，并用 `thickness` 控制板厚。

4. **关节注入规则**
   - hinge：用 `nodegroup_hinge_joint` 注入 revolute joint（axis 与 origin 必须严格定义）
   - slide：用 `nodegroup_sliding_joint` 注入 prismatic joint（axis 与 travel 必须严格定义）

5. **验证**
   - URDF（含碰撞）导出验证：运行对应 `scripts/export_<box>_urdf.py` 或新增脚本后，使用
     - `python scripts/verify_urdf_inertia_fix.py --urdf-dir sim_exports/urdf/<asset>`
   - Phase-1 标注样本验证：
     - `python scripts/verify_phase1_sample_folder.py <sample_dir>`

---

## 4. 关节与坐标系规范（避免“看起来对、物理错”）

### 4.1 Hinge（revolute）

必须确定：

- **joint origin**：铰链轴线上一点（在 parent link frame 内）
- **axis**：单位向量（在 joint frame 内；导出器会根据注入信息构造）
- **limit**：\([lower, upper]\)（注意单位弧度）
- **dynamics**：damping/friction 用于仿真稳定性（但不是“避免自碰撞”的手段）

### 4.2 Prismatic（slide）

必须确定：

- **axis**：滑动方向单位向量
- **limit**：\([lower, upper]\) 对应 travel（米）
- 注意：对于抽屉/天地盒，推荐在几何上预留 **clearance**（微小间隙），避免初始状态碰撞穿插。

---

## 5. Self-collision 与 collision mesh（工程约束）

URDF exporter 支持 `visual_only`：

- `visual_only=True`：Phase-1 数据集导出使用（只需要视觉 mesh + inertia），不生成 collision mesh
- `visual_only=False`：仿真验证/训练用 URDF（需要 collision mesh + self-collision）

工程建议：

- **开发盒型时先用 `visual_only=False` 验证 URDF 能在 PyBullet 中稳定加载**
- 若出现初始自碰撞：
  - 优先在几何上引入微小 clearance（真实制造也存在）
  - 再考虑缩放/简化 collision mesh（例如 convex decomposition 参数）

---

## 6. Phase-1 标注与“可读映射”（segmentation_label_map）

Phase-1 exporter 输出 `segmentation.npy`（整数 id mask）+ `segmentation_label_map.json`：

- `link_name_to_id`: `{ "link_0": 1, ... }`（稳定 ID 分配）
- `labels`: 每个 id 的 `name/color_rgb_uint8/present_in_frame`，并对少数盒型提供 `role`（语义标签）

语义 `role` 的工程策略：

- **只对语义稳定且 DOF 少的盒型提供 role**（例如 MAILER/DRAWER/SLIP_LID）
- 对复杂盒型（如 TUCK_END）默认不附加 role，避免“猜测式语义”造成下游误用

---

## 7. 16 盒型扩展路线（建议实现顺序）

结合 `BoxType` 预留枚举与 `docs/Box_URDF_Generation_Plan.md` 的难度评估，推荐按“关节类型复杂度”递进：

1. **单 DOF prismatic**：`DRAWER`, `SLIP_LID`
2. **2 DOF hinge**：`MAILER`
3. **多 DOF hinge（联动/插舌二段）**：`TUCK_END`
4. **更复杂结构**：`LOCK_BOTTOM`, `*_SAFETY`, `HOOK_*`, `GABLE_TOP`, `*_GIFT`, `RSC`

每新增一个盒型，建议同时新增：

- `scripts/export_<box>_urdf.py`（用于 `visual_only=False` 仿真验证）
- 最小 smoke test（Phase-1 exporter）+ `verify_phase1_sample_folder.py` 验证

---

## 8. 快速命令索引

- 导出 URDF（含碰撞）：
  - `python -m infinigen.launch_blender -s scripts/export_drawerbox_urdf.py`
  - `python -m infinigen.launch_blender -s scripts/export_sliplidbox_urdf.py`
  - `python -m infinigen.launch_blender -s scripts/export_mailerbox_simple_urdf.py`
  - `python -m infinigen.launch_blender -s scripts/export_tuckendbox_urdf.py`

- URDF 结构 + PyBullet 物理一致性验证：
  - `python scripts/verify_urdf_inertia_fix.py --urdf-dir sim_exports/urdf/<asset>`

- Phase-1 数据引擎（自动标注）：
  - `python -m infinigen.launch_blender -s scripts/export_mailerbox_simple_phase1_data_engine.py -- --box_type MAILER --seeds 42 --n_views 1 --joint_states "0.0,0.0"`
  - `python -m infinigen.launch_blender -s scripts/export_mailerbox_simple_phase1_data_engine.py -- --box_type TUCK_END --seeds 42 --n_views 1 --joint_states ""`

- Phase-1 sample folder 验证：
  - `python scripts/verify_phase1_sample_folder.py <sample_dir>`

