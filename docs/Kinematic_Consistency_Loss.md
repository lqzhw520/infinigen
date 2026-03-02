# Kinematic Consistency Loss（Phase 2.2）— 严格数学形式化（hinge + prismatic）

本文档给出 **铰链（revolute）** 与 **滑动（prismatic）** 两类关节的“运动学一致性约束”数学形式化，并提供与代码实现一一对应的残差定义。

对应参考实现：`infinigen/perception/topo_box_net/kinematic_consistency.py`（NumPy + PyTorch 版本）。

---

## 1. 记号与约定

我们只讨论 **parent link frame** 下的相对变换（不依赖 world/camera）：

- 旋转 \(R \in SO(3)\)
- 平移 \(t \in \mathbb{R}^3\)
- 关节轴单位向量 \(a \in \mathbb{R}^3, \|a\|=1\)
- 关节轴上一点（关节原点）\(o \in \mathbb{R}^3\)

采用右乘/左乘都可以，只要一致。这里采用向量变换：

\[
x_c = R x_p + t
\]

其中 \(x_p\) 是 parent frame 下点坐标，\(x_c\) 是 child frame（或“经过关节变换后”）的点坐标。

---

## 2. Revolute（hinge）关节：纯绕轴旋转（无螺旋位移）

### 2.1 旋转矩阵（Rodrigues）

给定轴 \(a\) 与角度 \(\theta\)：

\[
R(a,\theta) = I + \sin\theta [a]_\times + (1-\cos\theta)[a]_\times^2
\]

其中 \([a]_\times\) 为反对称矩阵（叉乘矩阵）。

### 2.2 绕“过点 o 的轴线”旋转的平移项

绕过点 \(o\) 的轴线旋转，满足：

\[
x' = R(x-o) + o = Rx + (o - Ro)
\]

因此 hinge 的相对变换是：

\[
R = R(a,\theta), \quad t = o - Ro
\]

并立即得到两个关键性质：

- **轴不变性**：\(Ra=a\)
- **平移无轴向分量**：\(a^\top t = 0\)

---

## 3. Prismatic 关节：纯沿轴平移（无相对旋转）

给定轴 \(a\) 与位移 \(d\)：

\[
R = I,\quad t = d a
\]

因此 prismatic 的约束是：

- **无相对旋转**：\(R=I\)
- **平移方向必须平行轴**：\((I-aa^\top)t = 0\)

---

## 4. 运动学一致性残差（可作为 loss 项）

### 4.1 Hinge 约束残差

给定观测到的相对变换 \((R,t)\) 与关节参数 \((a,o)\)，定义 residual：

1) 轴不变性残差：\(\|Ra - a\|_2^2\)  
2) 平移轴向残差：\((a^\top t)^2\)  
3) 绕轴线旋转残差：\(\|(I-R)o - t\|_2^2\)

合并：

\[
\mathcal{R}_{hinge}(R,t;a,o)=\|Ra-a\|^2 + (a^\top t)^2 + \|(I-R)o - t\|^2
\]

对“理想 hinge 变换”应满足 \(\mathcal{R}_{hinge}=0\)（数值误差范围内）。

### 4.2 Prismatic 约束残差

给定 \((R,t)\) 与轴 \(a\)，定义 residual：

1) 旋转残差：\(\|R-I\|_F^2\)  
2) 平移垂直分量残差：\(\|(I-aa^\top)t\|_2^2\)

合并：

\[
\mathcal{R}_{pris}(R,t;a)=\|R-I\|_F^2 + \|(I-aa^\top)t\|^2
\]

同样，理想 prismatic 变换应满足 \(\mathcal{R}_{pris}=0\)。

---

## 5. 与学习任务的对接方式（推荐）

上述残差的用途：

- **当模型预测的是 link pose / keypoints 的 SE(3) 变换**时：
  - 用 \((R,t)\) 作为观测（来自预测 pose 或由关键点拟合的刚体变换）
  - 用预测的 \((a,o)\) 作为约束参数
  - 将 \(\mathcal{R}_{hinge}/\mathcal{R}_{pris}\) 加入总 loss，作为“物理一致性正则项”

- **当模型直接预测 joint 参数（axis/origin/state）**时：
  - 监督 loss（L1/CE）是主项
  - 仍可用该 residual 作为“软约束”，防止数值漂移（例如 axis 未归一、平移出现垂直分量）

---

## 6. 参考实现与单测

- 实现：`infinigen/perception/topo_box_net/kinematic_consistency.py`
  - NumPy：`rodrigues_np / hinge_transform_np / prismatic_transform_np`
  - 残差：`hinge_constraint_residual_np / prismatic_constraint_residual_np`
  - PyTorch 对应版本：`*_torch`

- 单元测试：`tests/perception/test_kinematic_consistency.py`

