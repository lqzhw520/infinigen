## Retrospective — Modular Box Roadmap Completion (2026-02-03)

这份文档是为了解决你当前的困惑：**断连前后两次 request 混在一起**导致“我到底做了哪些改动、每一步解决了什么、验证了什么、现在到底完成到哪”不清晰。

说明：

- ~~**本工作区顶层不是一个 git repo**~~ **勘误（2026-02-26）**：实际上本工作区 **是** git repo（`git rev-parse --show-toplevel` → `/mnt/afs2/zhuhaowu/infinigen`，branch `feature/3d-assets`）。之前 shell 环境未正确配置 PATH 导致 `git` 命令未找到，误判为无 git。下面的改动清单仍有效，可与 `git status` 交叉对照。
- 你要求忽略 `mcp-feedback-enhanced`，这里完全不涉及它。

---

### 0) 与“最开始第一件事”的对齐说明（为何你会觉得不一致）

你最开始的第一件事（你重新粘贴的那段原话）是：

- **优先把单个盒型 `MailerBox_Simple` 的 Phase‑1 补齐到科学无差**：也就是补完 `@docs/Physics_Aligned_Data_Engine_Research_Plan.md` 里的 **3.1.3 域随机化材质系统** + **3.1.4 自动标注管线**；
- **采用 `planning-with-files` 做 plan/实现闭环**，并在实时开发中用 `ralph-loop` 迭代（≤5 次）把这两块做成一个可复用的“Data Engine 示例样本”，然后再用这个 skill 扩展到其他盒型；
- 最后 **完善 `@.cursor/skills/infinigen-scientific-dev`**，补齐它缺乏的“域随机化材质系统 + 自动标注管线”两项能力。

这件“第一优先事项”的闭环证据链其实是存在的（并且是文件落盘的闭环）：

- **Plan/闭环文件**（`planning-with-files`）：`task_plan.md` / `findings.md` / `progress.md`
  - `task_plan.md` 的标题就是 “MailerBox_Simple Phase 1 completion (3.1.3 + 3.1.4)” 且状态为 completed
- **迭代次数约束**（`ralph-loop`）：`scripts/ralph/log.md`
  - 只用了 **3 次迭代（Iteration 1/5~3/5）**，分别完成：
    - Iteration 1/5：Phase 1.3 材质 DR（视觉 + 物理耦合）
    - Iteration 2/5：Phase 1.4 自动标注导出（RGB‑D/seg/keypoints/URDF GT/metadata）
    - Iteration 3/5：升级 `infinigen-scientific-dev` skill（把 1.3/1.4 写进 protocol + templates）
- **验收条目**（user stories）：`docs/user-stories/mailerbox-simple-phase1.json`（三条 story 全部 `passes=true`）

那你为什么会觉得与本文件不一致？核心原因是：**你在完成“第一件事”之后，又追加了更强的硬要求：把拆分的“8 个 to-dos（Phase 1.1 → Phase 3.2）全部完成”**。因此，本文件的 scope 变成了一个**超集总结**：

- 原始“第一件事”主要对应本文件最后的四条总结里的前两条：
  - **Phase‑1（样本文件夹 + 可读 label_map）**：即 Phase‑1 Data Engine 示例样本闭环（原始 3.1.3 + 3.1.4 的落地形态）
  - **URDF（惯性合并 + PyBullet 验证）**：属于 Phase‑1 的“Physics‑Aligned 基础设施”与验证
- 而四条总结里的后两条（Phase‑2/Phase‑3）是你后续追加的“8 to-dos 全做完”目标导致的扩展，并非最初那句“先把单盒型 Phase‑1 走完再扩展”的必要条件：
  - **Phase‑2（感知模块骨架 + Phase‑2.2 数学/单测闭环）**
  - **Phase‑3（消融实验设计 + run matrix + 规划接口闭环）**

这也解释了你在 `Retrospective:269-272` 处产生的疑惑：那 4 条并不是“最开始第一件事”的等价陈述，而是**在完成第一件事后，被新加目标扩展出来的终局总结**。

---

### 0) 你最终要求的“4 条 verify commands”执行结果（全部达标）

你让我先把我自己给出的验证命令跑一遍，确认是否真的完成预期目标。结果如下（全部 **PASS/成功**）：

#### 0.1 URDF inertia/spec + PyBullet load（mailerbox_simple）

命令：

```bash
python scripts/verify_urdf_inertia_fix.py --urdf-dir sim_exports/urdf/mailerbox_simple --quiet
```

结果（摘要）：

- Seeds: `101..105, 201..205, 42`
- `总计 11 / 通过 11 / 失败 0`
- ✅ 结构验证 + PyBullet 验证均通过（每 link 最多 1 个 `<inertial>`；并启用 `URDF_USE_INERTIA_FROM_FILE | URDF_USE_SELF_COLLISION`）

#### 0.2 Phase-1 sample folder sanity（单个样本）

命令：

```bash
python scripts/verify_phase1_sample_folder.py sim_exports/data_engine/_smoke_test15_labelmap_complete_mailer/dataset/train/000001
```

结果（摘要）：

- ✅ required files 齐全（含 `segmentation_label_map.json`）
- ✅ depth/seg/instance shape 一致
- ✅ depth 无 NaN/Inf
- ✅ keypoints 可见点都在图像内
- ✅ label_map ids 完整 & `present_in_frame` 与 `segmentation.npy` 一致

#### 0.3 Kinematic consistency unit tests（纯 Python 单测）

命令：

```bash
python -m pytest -q tests/perception/test_kinematic_consistency.py
```

结果（摘要）：

- ✅ `6 passed`（1 个 pytest “unknown config option: timeout” 警告，不影响测试结果）

#### 0.4 Planner demo（drawer sample）

命令：

```bash
python scripts/planning/run_motion_planner_demo.py \
  --sample-dir sim_exports/data_engine/_smoke_test16_prismatic_fk_drawer/dataset/train/000001 \
  --out sim_exports/planning_demo_drawer_collision.json
```

结果（摘要）：

- ✅ `success: true`
- ✅ `message: planned (collision-free)`
- ✅ self-collision 打开、PyBullet 载入成功、轨迹长度 60

> 这 4 条 verify commands 覆盖了你当时要的“达标信号”：URDF 惯性正确 + Phase-1 数据样本一致性 + Phase-2.2 数学/代码正确性 + Phase-3.2 最小规划闭环可跑通。

---

### 1) 总体目标（你最关心的 8 个 to-dos）与最终交付状态

你要求“不要打断问我、把分解的 8 个 to-dos 全部做完达标再停”。对应完成情况如下：

- ✅ **Phase 1.1（数据引擎）盒型拓扑扩展**：支持并验证 `MAILER / TUCK_END / DRAWER / SLIP_LID` 的 URDF 导出与仿真载入（含 self-collision/collision meshes）。
- ✅ **Phase 1.4（数据引擎）自动标注管线**：统一脚本导出 RGB-D + seg/instance + keypoints + camera params + URDF GT + metadata + `segmentation_label_map.json`（可读映射）。
- ✅ **Phase 2.1（感知模型）Topo-Box-Net 架构设计**：文档 + 代码骨架（数据 schema / URDF parser / loader / model heads / losses）。
- ✅ **Phase 2.2（感知模型）Kinematic Consistency Loss 数学形式化**：hinge/prismatic 的严格残差定义 + NumPy/PyTorch 参考实现 + 单元测试。
- ✅ **Phase 3.1（验证）Sim2Real 消融实验设计**：消融维度、指标、统计检验、可复现目录结构与脚本骨架。
- ✅ **Phase 3.2（验证）集成 VAMP Motion Planner**：提供 planner-agnostic API + baseline joint-space planner（可运行）+ VAMP adapter stub + demo runner + 文档。
- ✅ **Self-collision Explanation**：在 `docs/MailerBox_Simple_Technical_Analysis_Report.md` 中补齐并澄清 joint limit vs self-collision，补强“严格性/软约束”数值分析。
- ✅ **URDF Tag Annotation**：扩展“标签释义”覆盖 `<parent>/<child>/<axis>/<limit>/<dynamics>/<geometry>/<mesh>` 等，并修正 inertial 章节为“历史问题已修复”。

---

### 2) 关键问题复盘：你关心的 P0（URDF 惯性 / 多 inertial）到底怎么解决的？

#### 2.1 问题本质

URDF 规范要求：**每个 `<link>` 最多一个 `<inertial>`**。但一个 link 往往由多个 mesh 面板组成（例如箱体 5 块薄板），因此必须把多个子几何的惯性合成到一个 `(M, C, I)`。

#### 2.2 修复点（代码层）

在 `infinigen/core/sim/exporters/urdf_exporter.py` 中：

- 对同一 link 内的每个几何体计算 `robust_mass, com, I_tensor`
- 使用 `thin_shell_inertia.combine_multiple_inertias(...)` 做平行轴定理合并
- 最终只写出 **一个** `<inertial>` block 到该 `<link>`

这保证：

- 质量守恒：\(M=\sum m_i\)
- COM：\(C=\frac{1}{M}\sum m_i c_i\)
- 惯性：\(I=\sum (I_i + m_i[(r_i\cdot r_i)I - r_i r_i^T])\)，其中 \(r_i=c_i-C\)

#### 2.3 严格验证点（工具层）

`scripts/verify_urdf_inertia_fix.py` 做两件事：

1) 结构验证：每个 `<link>` 的 `<inertial>` 个数 ≤ 1  
2) PyBullet 验证：使用 `URDF_USE_INERTIA_FROM_FILE` 确保读到 URDF 的惯性，而不是让 PyBullet 重新估计

最终你要求的“10 个 mailerbox seeds”也已全部 PASS（本次 verify 输出里甚至多了 seed=42）。

---

### 3) Phase-1 数据引擎与“可读映射 label_map”复盘

你要求输出 segmentation 的“更可读的额外映射方式”。最终落地为：

- `segmentation.npy`：像素级 label id（int）
- `segmentation_label_map.json`：**id ↔ link name（+可选 role）** + 稳定颜色 + `present_in_frame`

关键修复/增强：

1) **label_map 不再只列出“本帧出现的 ids”**  
   - 改为：列出 `link_name_to_id` 中所有 link ids（即使这帧没出现，也保留并 `present_in_frame=false`），避免下游 batch 对齐困难。
2) **颜色映射稳定**  
   - 从“按出现的 uniq 列表顺序采样颜色”改为“按 label id 单独 RNG”，保证不同帧不会因为缺少某个 id 而导致颜色整体漂移。
3) **role 语义只对语义稳定的盒型提供**  
   - `MAILER/DRAWER/SLIP_LID` 提供 role（base/lid/drawer 等）
   - `TUCK_END` 默认不猜语义（避免下游误用）
4) **prismatic FK 修复（关键）**  
   - 之前 `compute_link_poses_world()` 对所有非 fixed joint 都当成旋转，导致 prismatic joint 的位姿错误。
   - 现在对 prismatic 用 `T = Translation(axis * q)`。

验证方式：

- `scripts/verify_phase1_sample_folder.py` 对样本 folder 做一致性校验（required files、shape、label_map schema、present_in_frame 与 seg 对齐等）。

---

### 4) Phase-2 感知模块（Topo-Box-Net）做了什么？

交付目标不是“训练出 SOTA 模型”，而是把 Phase-2 设计变成 **可运行、对齐 Phase-1 数据字段** 的骨架：

- 文档：`docs/Topo_Box_Net_Architecture.md`
- 代码：`infinigen/perception/topo_box_net/`
  - `urdf.py`：最小 URDF parser（link/joint/origin/axis/limit）
  - `dataset.py`：Phase-1 sample loader（严格 shape/dtype）
  - `model.py`：CNN encoder + 多头输出骨架（需要 torch 才能运行）
  - `losses.py`：masked multi-task loss（box_type/joint_state/joint_axis/joint_origin）

---

### 5) Phase-2.2 运动学一致性损失（数学+实现+单测）

交付：

- 文档：`docs/Kinematic_Consistency_Loss.md`
- 实现：`infinigen/perception/topo_box_net/kinematic_consistency.py`
  - hinge / prismatic 的残差定义（NumPy + PyTorch）
- 单测：`tests/perception/test_kinematic_consistency.py`

并为了解决“非 Blender 环境跑不了 pytest”的问题：

- 更新 `tests/conftest.py`：在没有 `bpy` 时跳过 Blender 清理逻辑，使纯 Python 测试可运行（你现在看到的 verify 已通过）。

---

### 6) Phase-3.1 / Phase-3.2 做了什么？

#### 6.1 Phase-3.1 Sim2Real 消融实验设计

- 文档：`docs/Sim2Real_Ablation_Experiment_Design.md`
  - 消融维度（数据量 / 物理对齐 / DR 强度）
  - 指标（perception + planner）
  - 统计显著性建议（bootstrap/ANOVA/logistic 等）
  - 可复现目录结构
- 脚本：`scripts/run_sim2real_ablation_matrix.py`
  - 生成 run matrix JSON（不直接训练，先把实验计划固化下来）

#### 6.2 Phase-3.2 VAMP Motion Planner 集成骨架

因为 VAMP 外部依赖不在仓库中，我做的是“**接口闭环 + baseline 可运行**”：

- API：`infinigen/planning/motion_planner_api.py`
  - `PlanRequest/PlanResult`（URDF + start/goal joint states + collision/self-collision flags）
- baseline planner：`infinigen/planning/simple_jointspace_planner.py`
  - 线性 joint 插值 + PyBullet 自碰撞检查（验证闭环）
  - 自动 `p.setAdditionalSearchPath(urdf_dir)` 支持 `assets/*.obj` 相对路径
- VAMP adapter stub：`infinigen/planning/vamp_adapter.py`
- 文档：`docs/VAMP_Motion_Planner_Integration.md`
- demo runner：`scripts/planning/run_motion_planner_demo.py`
  - 从 Phase-1 sample 推断一个能加载 meshes 的 URDF（优先 `sim_exports/urdf/...`）

> 额外信息：MailerBox 在 collision URDF 上做“从 0 插到上限”的 naive 线性轨迹会触发自碰撞（这是合理现象：说明自碰撞检测在工作；也说明需要更强 planner，例如 VAMP）。

---

### 7) 文档同步：修正“报告里说 inertial 很严重但实际上已修复”的矛盾

你提到“前后两次断连有点错乱”。一个典型错乱点是：

- `docs/MailerBox_Simple_Technical_Analysis_Report.md` 早期写成“当前 link_0 有多个 inertial 是严重错误”
- 但后来代码已修复、验证也通过

本次我把报告升级到 v1.1（2026-02-03）并做了关键修正：

- 执行摘要里把 inertial 从“严重错误”改为“✅ 已修复 + 已验证”
- 将“多 inertial”示例明确标注为“修复前历史版本摘录”
- 扩展标签释义（`<parent>/<child>/<axis>/<limit>/<dynamics>/<mesh>` 等）
- 在 joint limit vs self-collision 章节补充“求解器软/硬程度”的数值因素

---

### 8) 改动清单（按文件归档：新增/修改 + 目的）

#### 8.1 新增文件（Added）

- `docs/Modular_Box_Factory_Architecture.md`：盒型工厂扩展架构说明（扩展到 16 types 的工程约束）
- `docs/Topo_Box_Net_Architecture.md`：Phase 2.1 架构与数据字段对齐
- `docs/Kinematic_Consistency_Loss.md`：Phase 2.2 数学形式化
- `docs/Sim2Real_Ablation_Experiment_Design.md`：Phase 3.1 消融实验设计
- `docs/VAMP_Motion_Planner_Integration.md`：Phase 3.2 规划接口与评估协议
- `docs/Retrospective_Roadmap_Completion_2026-02-03.md`：本文件（复盘）

- `scripts/verify_phase1_sample_folder.py`：Phase-1 sample folder 一致性验证脚本
- `scripts/run_sim2real_ablation_matrix.py`：生成消融实验 run matrix JSON
- `scripts/planning/run_motion_planner_demo.py`：规划 demo runner（从 sample folder 读取 URDF/状态）

- `infinigen/perception/topo_box_net/*`：Topo-Box-Net skeleton（schema/urdf/dataset/model/loss/kinematic consistency）
- `infinigen/planning/*`：planner API + baseline planner + VAMP adapter stub

- `tests/perception/test_kinematic_consistency.py`：Phase 2.2 单元测试

#### 8.2 修改文件（Updated）

- `scripts/export_mailerbox_simple_phase1_data_engine.py`
  - 增加/完善 `segmentation_label_map.json`
  - 修复 prismatic FK（Drawer/SlipLid 位姿正确）
  - role_map 语义映射（MAILER/DRAWER/SLIP_LID）
- `docs/Physics_Aligned_Data_Engine_Research_Plan.md`
  - 补充 label_map 文件
  - 更新盒型状态（Drawer/SlipLid 完成）
- `docs/MailerBox_Simple_Technical_Analysis_Report.md`
  - v1.1：修正 inertial 章节为“历史问题已修复”
  - 扩展标签释义与 solver 软约束分析
- `.cursor/skills/infinigen-scientific-dev/verification-templates.md`
  - 强化 `segmentation_label_map.json` 校验逻辑（expected ids、record schema、present_in_frame 对齐）
- `tests/conftest.py`
  - 在无 `bpy` 环境下允许运行纯 Python 测试
- `infinigen/planning/simple_jointspace_planner.py`
  - PyBullet 设置 search path 以解析 URDF 相对 mesh 路径

（以及更早阶段已在仓库内对齐的：`modular_box_factory.py` 新增 Drawer/SlipLid 工厂、`tests/sim/test_combine_multiple_inertias.py`、`scripts/export_drawerbox_urdf.py`、`scripts/export_sliplidbox_urdf.py` 等。）

---

### 9) 当前最终结果（你现在可以依赖什么）

- **Phase-1**：可生成、可复现、可验证的样本文件夹结构（含可读 label_map）  
- **URDF**：惯性合并规范正确，PyBullet 载入验证通过（且验证脚本可直接复用）  
- **Phase-2**：感知模块有文档 + 可运行骨架；Phase-2.2 数学与单测闭环  
- **Phase-3**：有消融实验设计与 run matrix 脚手架；规划接口闭环可跑通，且 collision/self-collision 验证已接入  

如果你接下来要推进“创新且数据真实的物理生成引擎”，我们可以在这个基础上把：

- DR/物理对齐的配置体系做成可组合的 config（并固化到 manifest）
- Planner 从 baseline 升级到真正的 VAMP（或其他）并加入环境碰撞
- Perception 训练入口脚本与指标汇总（真正跑 Phase-3.1 的矩阵）

