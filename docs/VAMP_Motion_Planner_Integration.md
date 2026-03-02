# VAMP Motion Planner Integration（Phase 3.2）

本文档定义将 **VAMP Motion Planner** 接入 Infinigen “articulated box” 数据闭环的最小工程接口（interfaces / data formats / evaluation protocol），并给出一个 **可运行的基线 planner**（线性关节插值 + PyBullet 自碰撞检查）作为集成骨架。

> 说明：本仓库不内置 VAMP；因此提供 adapter stub + baseline planner，保证接口闭环、可运行、可扩展。

---

## 1. 接入目标

输入（来自 Phase-1/Phase-2）：

- URDF（GT 或 Perception 重建）
- 初始 joint state（来自观测/估计）
- goal spec（例如：打开盖子到 90°；抽屉拉出 8cm）

输出：

- 一条可执行的 joint-space trajectory（时间序列的 joint positions）
- 规划质量指标（成功、碰撞、路径长度、耗时）

---

## 2. 数据格式（Planner-agnostic）

统一使用 joint-space 目标（对盒子任务足够且易复现）：

- URDF：`urdf_path`
- `start_positions`: `{joint_name: q0}`
- `goal_positions`: `{joint_name: qT}`
- `joint_limits`: `{joint_name: (lower, upper)}`（可选，但强烈建议）

对应代码结构：

- `infinigen/planning/motion_planner_api.py`
  - `PlanRequest` / `PlanResult` / `MotionPlanner`

---

## 3. 最小可运行基线（用于 CI / sanity-check）

基线 planner：线性插值（JointSpaceLinearPlanner）

- `infinigen/planning/simple_jointspace_planner.py`
  - `JointSpaceLinearPlanner.plan(req)`
  - 支持：
    - joint limits 预检查（可选）
    - PyBullet self-collision 检查（可选）

基线用途：

- 在 VAMP 尚未接入前，先保证 “URDF → 轨迹 → collision check → metrics” 的闭环可跑通
- 用作后续 VAMP 的对照组（同一任务、同一指标）

---

## 4. VAMP Adapter（外部依赖对接点）

- `infinigen/planning/vamp_adapter.py`
  - `VampMotionPlanner.plan(req)`
  - 负责将 `PlanRequest` 翻译为 VAMP 的 problem spec，并返回 `PlanResult`

建议 VAMP adapter 输出的最小字段：

- `trajectory`: List[Dict[joint_name -> position]]
- `metrics`:
  - planning_time_s
  - path_length_l1（或 l2）
  - collided（bool）
  - min_contact_distance（若可获得）

---

## 5. Evaluation Protocol（统一评估协议）

对每个 task instance（一个 URDF + start + goal）记录：

- **Success**：达到 goal 且无碰撞（自碰撞 / 环境碰撞）
- **Collision**：沿轨迹任何时刻出现 penetration（contactDistance < -eps）
- **Path length**：\(\sum_t \|q_{t+1} - q_t\|_1\)
- **Time**：wall-clock（秒）

建议每个配置重复 ≥3 次（不同 planner seed / sampling），并报告均值与置信区间。

---

## 6. Demo 运行方式（推荐）

建议使用 Phase-1 sample folder 作为输入（包含 `urdf_gt.urdf` + `joint_state.json`）：

- 见脚本：`scripts/planning/run_motion_planner_demo.py`

该脚本会：

- 从 sample folder 读取 start joint state
- 从 URDF 读取 joint limits 并构造一个 “打开/拉出” 的目标状态
- 用 baseline planner 生成轨迹并做 PyBullet self-collision 检查
- 输出 `PlanResult` JSON

