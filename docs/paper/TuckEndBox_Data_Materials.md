# TuckEndBox (Infinigen) — Paper Data Materials

> 本文档用于论文作业的数据素材整理（供 Deep Research / LLM 直接使用）。
>
> **重要约束**：
> - 不修改、不影响现有 `TuckEndBox` 已完成的实现与导出结果
> - 不更新 `docs/Box_URDF_Generation_Plan.md`
> - 批量统计/消融实验产生的中间数据会尽量清理；为保证论文可复核与可复现，**保留小体量结果文件**：
>   - `docs/paper/TuckEndBox_Data_Materials.md`（本文档）
>   - `docs/paper/TuckEndBox_Data_Materials.audit_summary.json`（1000 样本统计表 + scaling 数据）
>   - `docs/paper/TuckEndBox_Data_Materials.audit_raw_results.json`（1000 样本逐 seed 的布尔审计结果）
>   - `docs/paper/audit_chunks_20260114_113036_v5/`（并行审计的 worker 结果与日志，便于追溯；可选保留）

---

## Data Material 1 — Topology & Diversity（参数空间与“几何流形”说明）

### 1.1 参数空间（关键参数）

TuckEndBox 的关键参数（论文使用口径）：

- **Width** \(W\): 盒体 X 向宽度（m）
- **Depth** \(D\): 盒体 Y 向深度（m）
- **Height** \(H\): 盒体 Z 向高度（m）
- **Thickness** \(T\): 面板厚度（m）
- **tab_len**: 插舌长度（m）

### 1.2 代码来源与采样范围（现有实现）

- 尺寸采样：`infinigen/assets/sim_objects/modular_box_factory.py` 的 `TuckEndBoxFactory.sample_dimensions()`
  - \(W \sim \mathcal{U}(0.06, 0.18)\)
  - \(D \sim \mathcal{U}(0.05, 0.14)\)
  - \(H \sim \mathcal{U}(0.18, 0.45)\)
  - \(T \sim \mathcal{U}(0.0008, 0.002)\)
- 插舌长度（几何节点中派生）：`TuckEndBoxFactory.create_geometry_nodegroup()` 内部
  - \(\text{tab\_len} = 0.20 \cdot D\)

> 说明：当前实现中 `tab_len` 是 **Depth 的确定性函数**，并非独立采样维度；但在论文叙述中可作为一个显式参数（等价于把 \(D\) 映射到 \(\text{tab\_len}\) 坐标）。

### 1.3 学术说明：参数 → 高维几何流形；采样 = 在流形上的随机游走

将一次资产生成视为一个映射：

\[
f:\ (W, D, H, T)\ \mapsto\ \mathcal{G}
\]

其中 \(\mathcal{G}\) 是“带铰链拓扑 + 网格几何 + 物理属性”的复合对象（URDF 链路树、mesh、惯性、动力学参数等）。

由于 `tab_len = 0.2D`，我们可以把论文要求的 5 维参数向量写为嵌入：

\[
\Phi(W,D,H,T) = (W, D, H, T, 0.2D) \in \mathbb{R}^5
\]

因此有效参数集合在 \(\mathbb{R}^5\) 中形成一个 4 维子流形（函数图像 / graph manifold）。程序化生成的“随机种子”驱动 \((W,D,H,T)\) 的随机采样，相当于在该流形上进行离散随机过程（可被视为随机游走或蒙特卡洛采样），从而产生拓扑一致但几何尺度不同的一族资产。

---

## Data Material 2 — Audit Metrics（1000 样本物理合法性审计）

> 本节将由脚本实际生成数据并回填表格。

### 2.1 审计定义（Validity 分解指标）

对每个随机样本（一个 seed）统计以下布尔指标：

- **Initial Collision**（初始自碰撞/穿透）：PyBullet 启用 self-collision 后，初始姿态 `getContactPoints(body, body)` 中是否存在**深度穿透接触**（默认阈值示例：`contactDistance < -5e-3`；用于规避 mesh collision margin 的影响），并且**排除直接关节相连的 parent-child 链接对**（折痕附近可能出现数值层面的微小重叠）
- **Inertia PD**（惯性正定）：URDF 中所有非 world link 的惯性张量是否正定（特征值全为正，且质量 > 0）
- **PyBullet Explosion**（加载爆炸/不稳定）：PyBullet 能否加载 + 在重力与地面接触下模拟若干步不发生发散（例如位移 > 100m 或出现 NaN）

> 工程实现备注（论文可选披露）：为支持 \(N=1000\) 的大规模审计，PyBullet 中的碰撞几何使用**每个 link 的 mesh AABB 代理 box**（由导出 OBJ 的顶点快速统计得到），避免加载/构建复杂 mesh collision 造成的数量级开销；惯性张量与关节参数仍来自真实导出 URDF。

总体 **Validity** 定义为三项同时通过：

\[
\text{Valid} = \neg\text{InitialCollision} \land \text{InertiaPD} \land \neg\text{Explosion}
\]

### 2.2 审计统计（将由脚本生成）

| 指标 | 计数 (out of 1000) | Rate |
|---|---:|---:|
<!-- AUDIT_TABLE_START -->
| Validity Rate | 1000 | 100.000% |
| Initial Collision Rate | 0 | 0.000% |
| Inertia PD Rate | 1000 | 100.000% |
| PyBullet Explosion Rate | 0 | 0.000% |
<!-- AUDIT_TABLE_END -->

> 备注：若个别样本发生“导出失败/URDF 不可加载”，会计入 Explosion/Invalid，并在附录记录失败原因摘要。

---

## Data Material 3 — Scaling Law（Validity Gap vs Sample Size）

### 3.1 曲线数据（将由脚本生成）

定义不通过率（Validity Gap）：

\[
\text{Gap}(N) = 1 - \text{ValidityRate}(N)
\]

其中 \(\text{ValidityRate}(N)\) 是前 \(N\) 个样本的通过率。

建议取样点（log-spaced 近似）：

- \(N \in \{10, 20, 50, 100, 200, 500, 1000\}\)

输出数据（TBD）：

| N | Validity Gap |
|---:|---:|
<!-- SCALING_TABLE_START -->
| 10 | 0.000% |
| 20 | 0.000% |
| 50 | 0.000% |
| 100 | 0.000% |
| 200 | 0.000% |
| 500 | 0.000% |
| 1000 | 0.000% |
<!-- SCALING_TABLE_END -->

### 3.2 拟合说明（可选）

可在 log-log 空间拟合幂律：

\[
\text{Gap}(N) \approx a \cdot N^{-b}
\]

若所有样本均通过（Gap 恒为 0），则该 scaling law 退化为零曲线（表示“在该样本规模内未观测到失效事件”）。

---

## Data Material 4 — Ablation Study（thin_shell_inertia 修复有效性证据）

### 4.1 实验设计

- **实验组（Enabled）**：保持现有实现（`urdf_exporter` 调用 `thin_shell_inertia.calculate_robust_inertia`）
- **对照组（Disabled / Naive）**：运行时 monkey-patch，使导出器回退到“体积法”并允许出现极小质量/惯性（模拟修复前的薄壳不稳定风险）

### 4.2 证据（将由脚本生成）

#### 控制组：PyBullet 报错/爆炸日志快照（TBD）

```text
<!-- ABLATION_CONTROL_START -->
[CONTROL_NAIVE] dims={'width': 0.12, 'depth': 0.08, 'height': 0.25, 'thickness': 0.0001} density=300.0
[CONTROL_NAIVE] inertia_pd=False min_mass=5.3760916929361405e-05 min_eig=0.0
[CONTROL_NAIVE] pybullet_load=OK joints=8 exploded=True info={'steps': 500, 'threshold': 5.0, 'check_every': 5, 'vel_threshold': 200.0, 'max_link_pos_norm': 5.06612916546877, 'max_abs_joint_vel': 45.28196743994206, 'reason': 'link_pos_norm>5.0@step35,link0'}
<!-- ABLATION_CONTROL_END -->
```

#### 实验组：稳定运行数据（TBD）

```text
<!-- ABLATION_ENABLED_START -->
[ENABLED] dims={'width': 0.12, 'depth': 0.08, 'height': 0.25, 'thickness': 0.0001} density=300.0
[ENABLED] inertia_pd=True min_mass=0.00217024061222 min_eig=4.63002901669e-08
[ENABLED] pybullet_load=OK joints=8 exploded=False info={'steps': 500, 'threshold': 5.0, 'check_every': 5, 'vel_threshold': 200.0, 'max_link_pos_norm': 2.423070456948434, 'max_abs_joint_vel': 45.28358800604294}
<!-- ABLATION_ENABLED_END -->
```

### 4.3 结论（论文表述模板）

在薄壁结构（极小厚度）下，体积法会产生接近 0 的质量与惯性张量，引发仿真数值不稳定；加入 thin-shell robust inertia 相当于向物理参数引入“先验偏置”（physical bias），为极端薄壳提供最小有效厚度与正定惯性下界，使得资产在大规模生成/仿真中保持可加载与稳定。

---

## Data Material 65 — URDF Semantic Structure（语义树与 intermediate link）

### 5.1 URDF 拓扑片段（intermediate link 示例）

> 说明：URDF 标准本质上要求关节-链接形成树结构。若同一刚体边上需要表达多个关节（multi-joint edge），导出器会插入 **intermediate link** 将其线性化为链结构。

片段来源：`tests/sim/test_urdf_multi_joint.py::test_urdf_xml_structure`（用于验证 `urdf_exporter.py` 的 intermediate-link 策略；命名与惯性参数与导出器保持一致）。

```xml
<link name="world"/>

<link name="link_0"> ... </link>

<link name="link_1_intermediate_0">
  <inertial>
    <mass value="0.001"/>
    <origin xyz="0 0 0"/>
    <inertia ixx="1e-9" ixy="0" ixz="0" iyy="1e-9" iyz="0" izz="1e-9"/>
  </inertial>
</link>

<link name="link_N"> ... </link>

<joint name="hinge_0" type="revolute">
  <parent link="link_0"/>
  <child link="link_1_intermediate_0"/>
</joint>

<joint name="hinge_1" type="revolute">
  <parent link="link_1_intermediate_0"/>
  <child link="link_N"/>
</joint>
```

### 5.2 学术说明

插入 intermediate link 的作用是把“一个 parent-child 对应多个关节”的不可表示结构，转换为 URDF 可表示的串联链路；这避免了导出器在多关节联动结构上出现“不可导出/闭环”的问题，并使动力学求解器能够按树结构进行正向动力学传播与约束求解。

---

## Reproducibility（复现说明）

本 md 的统计部分由一次性脚本生成并回填（运行于 Blender 环境）：

```bash
cd /mnt/afs2/zhuhaowu/infinigen
export INFINIGEN_PYTHONPATH=$(conda run -n infinigen python -c "import site; print(':'.join(site.getsitepackages()))")
conda run -n infinigen python -m infinigen.launch_blender -s scripts/paper/tuckendbox_paper_audit.py
```

结果文件位置（已生成，可直接用于论文）：

- `docs/paper/TuckEndBox_Data_Materials.audit_summary.json`
- `docs/paper/TuckEndBox_Data_Materials.audit_raw_results.json`


###论文作业最终交付物（请直接用这些写论文）
主文档（四类素材都在里面）：docs/paper/TuckEndBox_Data_Materials.md
审计统计与 scaling 数据（机器可读）：docs/paper/TuckEndBox_Data_Materials.audit_summary.json
1000 seed 逐样本审计原始结果（机器可读，可复核）：docs/paper/TuckEndBox_Data_Materials.audit_raw_results.json
并行审计 worker 结果与追溯材料（可选保留）：docs/paper/audit_chunks_20260114_113036_v5/
关键结果确认
Audit Metrics（1000 seeds）：Validity=1000/1000；InitialCollision=0；InertiaPD=1000/1000；Explosion=0
Scaling Law 数据点：N ∈ {10,20,50,100,200,500,1000} 的 Validity Gap 均为 0
Ablation：对照组（naive inertia）出现 inertia_pd=False 且 exploded=True；实验组（robust inertia）稳定且 inertia_pd=True（日志已在 md 中）

