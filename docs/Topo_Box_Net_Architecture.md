# Topo-Box-Net（Phase 2.1）— 架构设计与数据对齐（可执行骨架）

本文档定义 **Topo-Box-Net**：从 Phase-1 生成的数据（RGB-D + 标注 + URDF GT）中学习 **盒型拓扑（links/joints graph）** 与 **关节状态（joint positions）** 的感知模型。目标是提供：

- 明确的 **输入/输出定义**
- 可实现的 **模块结构图**
- 训练目标（loss）与 **Phase-1 字段对齐**
- 对应的 **代码骨架**（位于 `infinigen/perception/topo_box_net/`）

---

## 1. 任务定义（What / Why）

给定单视角或多视角的观测（本阶段先从单视角开始），Topo-Box-Net 预测：

1. **box type**（例如 MAILER / DRAWER / SLIP_LID / TUCK_END）
2. **可动关节集合**（关节数量/类型/顺序与 URDF 一致）
3. **关节几何参数**（axis、origin）
4. **关节状态**（角度/位移）

其核心价值是让下游模块（控制/规划/仿真）能够从视觉观测直接恢复可操控的 **URDF 兼容参数化模型**。

---

## 2. Phase-1 数据字段对齐（输入/监督信号）

每个 sample folder（由 `scripts/export_mailerbox_simple_phase1_data_engine.py` 生成）包含：

- **观测输入**
  - `rgb.png`：RGB
  - `depth.npy`：深度（米）
  - （可选输入）`segmentation.npy` / `instance.npy`：如果训练时允许使用 GT segmentation，可作为辅助通道；推理阶段可替换为自训练分割器输出

- **拓扑/运动学监督**
  - `urdf_gt.urdf`：link/joint graph + (origin, axis, limits)
  - `joint_state.json`：各 joint 名称与当前位置（与 URDF joint name 对齐）
  - `keypoints.json`：每个 joint 的轴线端点（2D uv / 可见性），用于“几何一致性”监督

- **几何映射（解决 seg-id 的可读性）**
  - `segmentation_label_map.json`：`link_name_to_id` + `labels[id]->name/role/present`

- **相机参数**
  - `camera_intrinsics.json`：K、HW、T_cam_to_world

---

## 3. 模块结构图（建议实现）

下面给出一个从“可跑通骨架”到“可扩展研究模型”的结构图：

```
RGB + Depth (+ Seg)  ──►  Encoder (CNN/ViT)  ──►  Global Feature (B,D)
                                      │
                                      ├──► BoxType Head (CE)
                                      │
                                      ├──► Joint Heads (masked, max_joints)
                                      │      - joint_state (L1/Huber)
                                      │      - joint_type (CE)
                                      │      - joint_axis (L1 + unit-norm)
                                      │      - joint_origin (L1)
                                      │
                                      └──► (Phase 2.2) Kinematic Consistency Loss
                                             - project axis keypoints to image
                                             - hinge/prismatic constraints
```

**阶段性策略（避免一开始就“可变长度图预测”过难）：**

- Phase 2.1：采用 `max_joints` + `joint_mask` 的方式处理变长 joint 列表（TuckEnd 多 DOF、Mailer 2 DOF、Drawer/SlipLid 1 DOF）
- Phase 2.2：加入 kinematic consistency loss（见下一阶段任务）

---

## 4. 输出定义（与 URDF 对齐）

Topo-Box-Net 输出统一为：

- `box_type_logits`: \((B, N_{types})\)
- `joint_state`: \((B, M)\)
- `joint_type_logits`: \((B, M, 3)\)，例如 \{revolute, prismatic, fixed/pad\}
- `joint_axis`: \((B, M, 3)\)（单位向量）
- `joint_origin`: \((B, M, 3)\)（在 base frame）

其中 \(M = \text{max\_joints}\)，对不足 \(M\) 的样本用 `joint_mask` 屏蔽 padding 部分 loss。

---

## 5. 训练目标（loss 组合）

建议的 supervised loss（Phase 2.1）：

- **BoxType 分类**：\( \mathcal{L}_{type} = CE(\text{logits}, y) \)
- **Joint state 回归**（masked）：\( \mathcal{L}_{q} = \| \hat{q} - q \|_1 \)
- **Joint type 分类**（masked）：\( \mathcal{L}_{jt} = CE(\text{joint\_type\_logits}, y_{jt}) \)
- **Joint axis / origin 回归**（masked）：\( \mathcal{L}_{axis}, \mathcal{L}_{origin} \)

总损失：

\[
\mathcal{L}=\lambda_{type}\mathcal{L}_{type}+\lambda_q\mathcal{L}_{q}+\lambda_{jt}\mathcal{L}_{jt}+\lambda_{axis}\mathcal{L}_{axis}+\lambda_{origin}\mathcal{L}_{origin}
\]

Phase 2.2 将加入：

- **Kinematic Consistency Loss**（约束 hinge/prismatic 的运动学合法性；并与 keypoints/segmentation 一致）

---

## 6. 代码骨架位置（已添加）

已提供可执行骨架（不侵入核心 Infinigen 生成器）：

- `infinigen/perception/topo_box_net/`
  - `urdf.py`: 最小 URDF parser（links/joints/origin/axis/limits）
  - `dataset.py`: Phase-1 sample folder loader
  - `model.py`: `TopoBoxNet`（CNN encoder + heads）
  - `losses.py`: 多任务 supervised losses（masked）
  - `schema.py`: dataclass schema（URDFJoint/URDFModel/Phase1Sample）

> 注：`model.py/losses.py` 需要 PyTorch；未安装时仅在实例化/调用时提示，不会影响 Infinigen 核心生成流程。

---

## 7. 下一步（Phase 2.2 对接点）

Phase 2.2 将在 `infinigen/perception/topo_box_net/` 中补充：

- hinge/prismatic 的严格数学约束 loss（可直接运行的 NumPy/PyTorch 参考实现）
- 单元测试（确保约束与梯度方向正确）

