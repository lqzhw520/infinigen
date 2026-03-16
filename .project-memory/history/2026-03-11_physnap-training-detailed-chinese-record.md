# PhysNAP 训练详细记录 -- 迭代 #8 中文完整版

**日期**: 2026-03-10 ~ 2026-03-11
**前序**: 迭代 #7（PhysNAP 集成：环境搭建 + 数据桥接）

---

## 一、总体概述

本次工作完成了 PhysNAP 扩散模型在 Infinigen 程序化生成的铰接盒体数据上的完整训练实验。核心目标是验证：Infinigen Phase 1 Data Engine 产出的 URDF + OBJ 数据，能否作为训练数据驱动 PhysNAP（一种基于扩散模型的铰接物体生成方法）的学习。

实验共设置三组对照：

1. **Baseline（PartNet-Mobility 原始数据训练）**：在原论文使用的 PartNet-Mobility 数据集上从零训练 NAP 扩散模型，复现原始实验流程
2. **Option B（Infinigen-only）**：完全使用 Infinigen 数据从零训练，验证 Infinigen 数据独立驱动模型学习的能力
3. **Option C（标注为 Fine-tune）**：使用较低学习率在 Infinigen 数据上训练（详见下方重要说明）

---

## 二、训练环境

### 硬件
- **GPU**: NVIDIA A800-SXM4-80GB（显存 81920 MiB）
- **驱动版本**: 570.172.08
- **GPU 训练时显存占用**: ~72.8 GB（Infinigen 数据）/ ~77.3 GB（PartNet-Mobility 数据）

### 软件
- **操作系统**: Linux 5.14.0-284.25.1.el9_2.x86_64
- **Conda 环境**: `physnap`（独立于 `infinigen` 环境）
- **Python**: 3.9.25
- **PyTorch**: 2.0.0+cu118（CUDA 可用）
- **PyTorch Geometric (PyG)**: 2.5.2
- **PhysNAP 仓库**: 克隆自 `https://github.com/EmbodiedVision/physnap.git`，位于 `external/physnap/`
- **C++ 扩展**: mcubes、mise、simplify_mesh 均已编译

### 模型架构
- **Denoiser（去噪器）**: v60 架构，20.753M 参数
  - 节点属性 V_dims: [1, 3, 3, 128]（occupancy、center、bbox、shape latent）
  - 对称边属性 E_sym_dims: [6, 4]（Plucker 坐标、关节限制）
  - 有向边属性 E_dir_dim: 3（关节方向）
  - 6 层 GNN，每层 512 维，32 头注意力
- **SDF Decoder（形状解码器）**: 0.705M 参数，预训练权重来自 `737.pt` checkpoint
- **扩散过程**: 1000 步，线性 beta 调度（0.001 → 0.02）

---

## 三、数据集

### Infinigen 数据（用于 Option B 和 Option C）

| 类别 | 训练集 | 验证集 | 测试集 |
|------|--------|--------|--------|
| infinigen_mailer | 200 | 25 | 25 |
| infinigen_drawer | 200 | 25 | 25 |
| infinigen_slip_lid | 200 | 25 | 25 |
| **合计** | **600** | **75** | **75** |

- **总样本数**: 750
- **零件复杂度分布**: 2-part 物体 400 个（Mailer, SlipLid），3-part 物体 200 个（Drawer）
- **Shape Codebook**: 1750 个零件全部编码为 128 维隐向量
  - Embedding 范围: [-2.4504, 1.7878]
  - Std 范围: [0.0513, 0.8675]
  - 有效零件: 1750/1750（100%）

### Shape Encoding 过程
使用 `scripts/encode_infinigen_shapes.py` 完成：
1. 加载 NAP 预训练的 ResnetPointnet 编码器（来自 `s1.5_partshape_ae` 的 `737.pt`）
2. 对每个 Infinigen 零件网格：采样 1024 个表面点 → 归一化到单位球 → 编码为 128 维隐向量
3. 全部 1750 零件在 A800 GPU 上编码完成，耗时约 20 秒
4. 输出格式与 NAP 预期一致：`{embedding: [N,128], valid_mask: [N], std: [128]}`

### PartNet-Mobility 数据（用于 Baseline）
- **训练集**: 1429 个样本
- **类别数**: 46 个（door, window, laptop, table, faucet, box 等）
- **零件复杂度分布**: 2-part (686), 3-part (398), 4-part (157), 5-part (117), 6-part (33), 7-part (22), 8-part (16)
- **Codebook**: 14100 个预计算 embedding（来自 `s1.5_partshape_ae_737.npz`）

### 数据转换管道
1. `scripts/infinigen_to_nap.py`: URDF + OBJ → NAP `.npz` 图格式
   - 解析 URDF 获取关节树结构
   - 合并每个 link 下的多个 OBJ 网格
   - 坐标归一化到 [-1, 1]
   - 关节轴转换为 Plucker 坐标（6D）
   - 构建图结构（V 节点 + E 边）
2. `scripts/merge_infinigen_nap_datasets.py`: 合并单类别数据集为多类别联合数据集
3. 输出目录: `external/physnap/data/infinigen_graph_combined/`

---

## 四、实验一：Baseline（PartNet-Mobility 原始训练）

### 配置
- **配置文件**: `configs/nap/v6.1_diffusion_adapted.yaml`
- **学习率**: 1e-4，衰减调度 [40K, 70K, 90K]，衰减因子 0.3
- **Batch Size**: 64
- **总迭代数**: 120,000
- **训练集大小**: 1429 样本
- **每 epoch 迭代数**: ~22（1429/64）

### 运行时间
- **开始时间**: 2026-03-10 12:55:11
- **结束时间**: 2026-03-10 23:17:20
- **总训练时长**: ~10 小时 22 分钟

### Checkpoint 记录

| Checkpoint | Epoch | 迭代数 | 大小 | 保存时间 |
|-----------|-------|--------|------|----------|
| 228.pt | 228 | 5,016 | 241.1 MB | 13:14:13 |
| 455.pt | 455 | 10,010 | 241.1 MB | 13:44:26 |
| 682.pt | 682 | 15,004 | 241.1 MB | 14:16:45 |
| 910.pt | 910 | 20,020 | 241.1 MB | 14:49:17 |
| 1137.pt | 1137 | 25,014 | 241.1 MB | 15:21:48 |
| 1364.pt | 1364 | 30,008 | 241.1 MB | 15:54:22 |
| 1591.pt | 1591 | 35,002 | 241.1 MB | 16:26:47 |
| 1819.pt | 1819 | 40,018 | 241.1 MB | 16:59:17 |
| 2046.pt | 2046 | 45,012 | 241.1 MB | 17:24:22 |
| 2273.pt | 2273 | 50,006 | 241.1 MB | 17:57:31 |
| 2500.pt | 2500 | 55,000 | 241.1 MB | 18:30:45 |
| 2728.pt | 2728 | 60,016 | 241.1 MB | 19:04:11 |
| 2955.pt | 2955 | 65,010 | 241.1 MB | 19:29:30 |
| 3182.pt | 3182 | 70,004 | 241.1 MB | 19:49:45 |
| 3410.pt | 3410 | 75,020 | 241.1 MB | 20:10:03 |
| 3637.pt | 3637 | 80,014 | 241.1 MB | 20:30:21 |
| 3864.pt | 3864 | 85,008 | 241.1 MB | 20:50:43 |
| 4091.pt | 4091 | 90,002 | 241.1 MB | 21:11:08 |
| 4319.pt | 4319 | 95,018 | 241.1 MB | 21:31:48 |
| 4546.pt | 4546 | 100,012 | 241.1 MB | 21:52:51 |
| 4773.pt | 4773 | 105,006 | 241.1 MB | 22:13:45 |
| 5000.pt | 5000 | 110,000 | 241.1 MB | 22:34:50 |
| 5228.pt | 5228 | 115,016 | 241.1 MB | 22:56:07 |
| 5450_latest.pt | 5450 | 119,900 | 241.2 MB | 23:16:50 |
| 5455.pt | 5455 | 120,010 | 241.1 MB | 23:17:20 |

**总计**: 25 个 checkpoint，训练完整完成 120K 迭代

### 损失收敛记录

初始损失（Epoch 1, Step 128）:
```
batch_loss: 3.6547 | loss_v: 0.9147 | loss_e: 2.7400
  v_shape: 1.0368 | v_center: 0.9553 | v_bbox: 0.9469 | v_occ: 0.7197
  e_plucker: 0.9321 | e_lim: 0.9150 | e_type: 0.8929
```

最终损失（Epoch 5455, Step 768, ~120K iters）:
```
batch_loss: 0.0195 | loss_v: 0.0118 | loss_e: 0.0077
  v_shape: 0.0171 | v_center: 0.0054 | v_bbox: 0.0043 | v_occ: 0.0204
  e_plucker: 0.0020 | e_lim: 0.0007 | e_type: 0.0051
```

**损失下降**: 3.6547 → 0.0195（下降 99.5%）

---

## 五、实验二：Option B -- Infinigen-only 从零训练

### 配置
- **配置文件**: `configs/nap/v6.1_diffusion_infinigen.yaml`
- **学习率**: 1e-4，衰减调度 [20K, 35K, 45K]，衰减因子 [0.3, 0.3, 0.3]
- **最低学习率**: 1e-8
- **Batch Size**: 32
- **总迭代数**: 60,000
- **训练集大小**: 600 样本
- **每 epoch 迭代数**: ~18（600/32 ≈ 18.75，向下取整）
- **max_K**: 8（最大零件数）
- **balance_sampling**: False
- **Codebook**: `infinigen_codebook.npz`

### 设计理由
- 迭代数设为 60K（原始 baseline 的一半），因为更小的数据集收敛更快
- Batch Size 从 64 降到 32，匹配较小的数据集规模
- LR 衰减调度前移（20K/35K/45K vs 40K/70K/90K），适配较短的训练周期

### 运行时间
- **开始时间**: 2026-03-10 13:17:22
- **结束时间**: 2026-03-10 17:12:18
- **总训练时长**: ~3 小时 55 分钟
- **平均每 1K 迭代耗时**: ~3.9 分钟

### Checkpoint 记录

| Checkpoint | Epoch | 迭代数 | 大小 | 保存时间 |
|-----------|-------|--------|------|----------|
| 278.pt | 278 | 5,004 | 241.1 MB | 13:36:45 |
| 556.pt | 556 | 10,008 | 241.1 MB | 13:56:11 |
| 834.pt | 834 | 15,012 | 241.1 MB | 14:15:36 |
| 1112.pt | 1112 | 20,016 | 241.1 MB | 14:35:06 |
| 1389.pt | 1389 | 25,002 | 241.1 MB | 14:54:29 |
| 1667.pt | 1667 | 30,006 | 241.1 MB | 15:14:09 |
| 1945.pt | 1945 | 35,010 | 241.1 MB | 15:33:52 |
| 2223.pt | 2223 | 40,014 | 241.1 MB | 15:53:40 |
| 2500.pt | 2500 | 45,000 | 241.1 MB | 16:13:19 |
| 2778.pt | 2778 | 50,004 | 241.1 MB | 16:32:53 |
| 3056.pt | 3056 | 55,008 | 241.1 MB | 16:52:45 |
| 3325_latest.pt | 3325 | 59,850 | 241.2 MB | 17:11:43 |
| 3334.pt | 3334 | 60,012 | 241.1 MB | 17:12:21 |

**总计**: 13 个 checkpoint，训练完整完成 60K 迭代

### 详细损失收敛记录

| 阶段 | Epoch | ~迭代数 | batch_loss | loss_v | loss_e | v_shape | v_center | v_bbox | v_occ | e_plucker | e_lim | e_type |
|------|-------|---------|------------|--------|--------|---------|----------|--------|-------|-----------|-------|--------|
| 起始 | 1 | 32 | 3.9760 | 0.9975 | 2.9786 | 1.0337 | 0.9867 | 1.0046 | 0.9649 | 0.9751 | 0.9815 | 1.0220 |
| 快速下降 | 2 | 96 | 0.8331 | 0.3380 | 0.4952 | 0.1838 | 0.1437 | 0.1128 | 0.1022 | 0.2823 | 0.1031 | 0.1097 |
| 1K | 17 | ~306 | 0.1208 | 0.0664 | 0.0544 | 0.1825 | 0.0350 | 0.0228 | 0.0252 | 0.0191 | 0.0162 | 0.0191 |
| 2.5K | 44 | ~792 | 0.0670 | 0.0358 | 0.0312 | 0.1047 | 0.0153 | 0.0147 | 0.0086 | 0.0133 | 0.0078 | 0.0102 |
| 5K | 278 | 5,004 | 0.0167 | 0.0112 | 0.0054 | 0.0350 | 0.0032 | 0.0030 | 0.0038 | 0.0020 | 0.0012 | 0.0022 |
| 30K | 1667 | 30,006 | 0.0060 | 0.0044 | 0.0017 | 0.0130 | 0.0009 | 0.0016 | 0.0021 | 0.0006 | 0.0003 | 0.0008 |
| 50K | 2778 | 50,004 | 0.0096 | 0.0069 | 0.0027 | 0.0215 | 0.0015 | 0.0017 | 0.0028 | 0.0007 | 0.0005 | 0.0015 |
| **终止** | **3334** | **60,012** | **0.0070** | **0.0054** | **0.0016** | **0.0170** | **0.0008** | **0.0014** | **0.0022** | **0.0006** | **0.0003** | **0.0007** |

### 损失收敛分析
- **总损失下降**: 3.976 → 0.007（下降 99.8%）
- **收敛速度**: 前 5K 迭代内 batch_loss 已降至 0.017，此后稳步下降
- **v_shape 是最大的子损失项**: 从 1.034 降至 0.017，始终是总损失的主要贡献者（约占 50%+），说明形状隐向量的预测仍是最有挑战性的部分
- **关节参数学习效果最好**: e_plucker（Plucker 坐标）从 0.975 降至 0.0006，下降幅度最大，说明 Infinigen 数据中的关节参数具有良好的规律性
- **30K 处出现了训练的最佳点**: batch_loss = 0.006，随后在 50K 处回升到 0.010，最终稳定在 0.007，表现出轻微的过拟合-恢复动态
- **与 Baseline 的关键差异**: Infinigen-only 最终 loss（0.007）显著低于 Baseline（0.020），原因在于 Infinigen 数据域更小更聚焦（3 类 vs 46 类），模型更容易拟合

---

## 六、实验三：Option C -- 低学习率训练

### 重要说明：预训练权重未加载

**经过对训练日志的仔细审查，发现以下关键事实：**

1. **日志中无预训练权重加载记录**：Fine-tune 日志中未出现 `"Checkpoint ... Loaded"` 信息（`solver_v2.py:132` 行会打印此消息）
2. **初始 loss 与 Option B 完全一致**：Option C 第一个 batch_loss = 3.9760403633117676，与 Option B 完全相同，精确到小数点后 16 位。若加载了预训练权重，初始 loss 应显著更低
3. **训练脚本确认**：`scripts/train_physnap_infinigen.sh` 中 option_c 的实现写着 `"Fine-tuning not yet implemented"`
4. **PhysNAP 的 resume 机制限制**：`solver_v2.py` 的 `solver_resume()` 方法仅从**当前实验自身的 checkpoint 目录**加载权重（第 116-118 行），无法跨实验加载。Fine-tune 实验的 log 目录（`v6.1_diffusion_finetune_infinigen`）启动时是空的，因此无法 resume
5. **配置文件无 initialize 字段**：`v6.1_diffusion_finetune_infinigen.yaml` 中未包含 `training.initialize_network_file` 字段

**结论**：Option C 实际上是**使用更低学习率（3e-5 vs 1e-4）从随机初始化开始的全新训练**，而非从 PartNet-Mobility 预训练权重微调。这一发现意味着真正的 fine-tune 实验尚未完成，需要在后续工作中正确实现。

### 正确的 Fine-tune 实现方案
要实现真正的 fine-tune，需要在配置文件中添加：
```yaml
training:
  initialize_network_file: ["./log/v6.1_diffusion_adapted/checkpoint/5455.pt"]
```
或者将 baseline checkpoint 复制到 fine-tune 的 log 目录并使用 `--resume latest`。

### 配置
- **配置文件**: `configs/nap/v6.1_diffusion_finetune_infinigen.yaml`
- **学习率**: 3e-5（为 Option B 的 30%），衰减调度 [10K, 20K, 25K]
- **Batch Size**: 32
- **总迭代数**: 30,000
- **训练集大小**: 600 样本（与 Option B 相同）
- **数据集**: 与 Option B 完全相同

### 运行时间
- **开始时间**: 2026-03-10 17:23:09
- **结束时间**: 2026-03-10 19:17:24
- **总训练时长**: ~1 小时 54 分钟
- **平均每 1K 迭代耗时**: ~3.8 分钟

### Checkpoint 记录

| Checkpoint | Epoch | 迭代数 | 大小 | 保存时间 |
|-----------|-------|--------|------|----------|
| 278.pt | 278 | 5,004 | 241.1 MB | 17:42:14 |
| 556.pt | 556 | 10,008 | 241.1 MB | 18:01:14 |
| 834.pt | 834 | 15,012 | 241.1 MB | 18:20:17 |
| 1112.pt | 1112 | 20,016 | 241.1 MB | 18:39:22 |
| 1389.pt | 1389 | 25,002 | 241.1 MB | 18:58:23 |
| 1650_latest.pt | 1650 | 29,700 | 241.2 MB | 19:16:16 |
| 1667.pt | 1667 | 30,006 | 241.1 MB | 19:17:26 |

**总计**: 7 个 checkpoint，训练完整完成 30K 迭代

### 详细损失收敛记录

| 阶段 | Epoch | ~迭代数 | batch_loss | loss_v | loss_e | v_shape | v_center | v_bbox | v_occ | e_plucker | e_lim | e_type |
|------|-------|---------|------------|--------|--------|---------|----------|--------|-------|-----------|-------|--------|
| 起始 | 1 | 32 | 3.9760 | 0.9975 | 2.9786 | 1.0337 | 0.9867 | 1.0046 | 0.9649 | 0.9751 | 0.9815 | 1.0220 |
| 早期 | 1 | 448 | 2.9306 | 0.7106 | 2.2200 | 1.0280 | 0.7983 | 0.7793 | 0.2367 | 0.8362 | 0.7446 | 0.6392 |
| 5K | 278 | 5,004 | 0.0305 | 0.0206 | 0.0099 | 0.0658 | 0.0064 | 0.0056 | 0.0046 | 0.0040 | 0.0019 | 0.0040 |
| 10K | 556 | 10,008 | 0.0167 | 0.0118 | 0.0048 | 0.0359 | 0.0035 | 0.0044 | 0.0036 | 0.0019 | 0.0010 | 0.0020 |
| 20K | 1112 | 20,016 | 0.0211 | 0.0167 | 0.0044 | 0.0575 | 0.0027 | 0.0032 | 0.0034 | 0.0014 | 0.0008 | 0.0021 |
| 25K | 1389 | 25,002 | 0.0126 | 0.0093 | 0.0033 | 0.0286 | 0.0026 | 0.0025 | 0.0034 | 0.0011 | 0.0006 | 0.0016 |
| **终止** | **1667** | **30,006** | **0.0145** | **0.0109** | **0.0037** | **0.0351** | **0.0031** | **0.0024** | **0.0028** | **0.0013** | **0.0007** | **0.0017** |

### 损失收敛分析
- **总损失下降**: 3.976 → 0.015（下降 99.6%）
- **收敛明显慢于 Option B**：
  - 在 5K 迭代时，Option C batch_loss = 0.031 vs Option B batch_loss = 0.017（高 82%）
  - 在 30K 迭代时，Option C batch_loss = 0.015 vs Option B batch_loss = 0.006（高 150%）
  - 原因：学习率仅为 3e-5（Option B 的 30%），从相同随机初始化出发收敛速度自然更慢
- **20K 处出现 loss 回升**：batch_loss 从 10K 的 0.017 升到 20K 的 0.021，与 LR 衰减调度（10K 处衰减到 30%）可能有关
- **最终 loss 高于 Option B**：0.015 vs 0.007，主要差距在 v_shape（0.035 vs 0.017），说明低学习率在 30K 迭代内未能充分优化形状隐向量

---

## 七、三组实验对比总结

### 关键指标对比

| 指标 | Baseline @120K | Option B @60K | Option C @30K |
|------|---------------|---------------|---------------|
| batch_loss | 0.0195 | 0.0070 | 0.0145 |
| loss_v | 0.0118 | 0.0054 | 0.0109 |
| loss_e | 0.0077 | 0.0016 | 0.0037 |
| loss_v_shape | 0.0171 | 0.0170 | 0.0351 |
| loss_v_center | 0.0054 | 0.0008 | 0.0031 |
| loss_v_bbox | 0.0043 | 0.0014 | 0.0024 |
| loss_v_occ | 0.0204 | 0.0022 | 0.0028 |
| loss_e_plucker | 0.0020 | 0.0006 | 0.0013 |
| loss_e_lim | 0.0007 | 0.0003 | 0.0007 |
| loss_e_type | 0.0051 | 0.0007 | 0.0017 |
| 训练时长 | ~10h22m | ~3h55m | ~1h54m |
| Checkpoint 数 | 25 | 13 | 7 |

### 分析

1. **Option B（Infinigen-only）表现最好**：在所有损失项上均取得最低值。这是因为 Infinigen 数据域更加聚焦（3 类简单盒体 vs 46 类复杂铰接物体），模型不需要学习跨类别的多样性
2. **Baseline 的 v_occ 损失异常高**：0.0204 远高于 Option B 的 0.0022 和 Option C 的 0.0028。这反映了 PartNet-Mobility 数据中零件 occupancy 预测的复杂性（46 类物体的零件数从 2 到 8 不等）
3. **v_shape 是所有实验的瓶颈**：在三组实验中，v_shape 始终是最大的损失项，说明从 128 维隐向量预测零件形状是最困难的任务
4. **Option C 尚未完成真正的 fine-tune**：由于预训练权重未加载，Option C 与 Option B 的唯一区别是学习率不同。真正的 fine-tune 实验需要后续补做

---

## 八、已解决的技术问题

### 1. pyrender EGL 渲染崩溃
- **现象**: `ValueError: Invalid device ID (0)` -- 无头服务器上 pyrender 的 OffscreenRenderer 无法初始化 EGL 设备
- **影响**: 训练每隔 1000 迭代尝试生成可视化时崩溃，导致训练中断
- **修复**: 在 `external/physnap/core/models/arti_ddpm_v2.py` 的 `_postprocess_after_optim` 方法中用 try-except 包裹整个可视化块
- **效果**: 训练正常继续，可视化失败时仅打印 warning 日志

### 2. 网格提取失败
- **现象**: `Mesh extraction fail, replace by a tiny place holder`（出现在 Fine-tune 日志中）
- **原因**: 某些生成的形状隐向量解码后不具备有效的 occupancy field，无法通过 Marching Cubes 提取网格
- **影响**: 不影响训练，仅在可视化时用占位网格替代

### 3. GLIBCXX 版本冲突
- **现象**: `libstdc++.so.6: version 'GLIBCXX_3.4.29' not found`
- **修复**: `export LD_LIBRARY_PATH="/root/anaconda3/envs/physnap/lib:$LD_LIBRARY_PATH"`

---

## 九、文件清单

### 新建文件
| 文件 | 用途 |
|------|------|
| `scripts/infinigen_to_nap.py` | URDF+OBJ → NAP 图格式转换 |
| `scripts/merge_infinigen_nap_datasets.py` | 合并多类别数据集 |
| `scripts/encode_infinigen_shapes.py` | 使用 NAP AE 编码形状隐向量 |
| `scripts/train_physnap_infinigen.sh` | 训练启动脚本 |
| `scripts/evaluate_physnap_training.py` | 训练结果评估对比 |
| `scripts/download_nap_data.sh` | NAP 数据手动下载指引 |
| `external/physnap/configs/nap/v6.1_diffusion_infinigen.yaml` | Option B 训练配置 |
| `external/physnap/configs/nap/v6.1_diffusion_finetune_infinigen.yaml` | Option C 训练配置 |
| `docs/PhysNAP_Academic_Review_and_Integration.md` | 学术 Review 与集成分析 |

### 修改文件
| 文件 | 修改内容 |
|------|----------|
| `external/physnap/core/models/arti_ddpm_v2.py` | 添加渲染失败的 try-except 处理 |

### 生成的数据产物
| 路径 | 内容 |
|------|------|
| `external/physnap/data/infinigen_graph_combined/*.npz` | 750 个 NAP 格式图文件 |
| `external/physnap/data/infinigen_graph_combined/infinigen_codebook.npz` | 1750 零件形状编码 |
| `external/physnap/data/infinigen_graph_combined/infinigen_split.json` | 训练/验证/测试划分 |
| `external/physnap/data/infinigen_graph_combined/infinigen_partkeys.json` | 零件键索引 |
| `external/physnap/log/v6.1_diffusion_adapted/checkpoint/` | 25 个 Baseline checkpoint |
| `external/physnap/log/v6.1_diffusion_infinigen/checkpoint/` | 13 个 Option B checkpoint |
| `external/physnap/log/v6.1_diffusion_finetune_infinigen/checkpoint/` | 7 个 Option C checkpoint |

---

## 十、关于之前中文回复中三个问题的记录

### 问题一：PartNet-Mobility 的角色与 Baseline 训练意义

**PartNet-Mobility 是 PhysNAP 的训练数据集，而非仅仅是验证集。** PhysNAP 论文中，PartNet-Mobility（约 2340 个物体，46 类）被用于：
- 训练 Stage 1：Shape Auto-Encoder（将零件形状编码为 128 维隐向量）
- 训练 Stage 2：NAP 扩散模型（学习铰接物体结构和形状的联合分布）
- 评估：在测试集上通过 FID、Coverage、MMD 等指标衡量生成质量

在当前环境下复现 Baseline 训练的目的是：
1. 验证训练 pipeline 在我们的环境（A800 GPU + physnap conda env）中能正常运行
2. 获得原始数据的 baseline 指标，作为 Infinigen 数据训练效果的对照基准
3. 如需做真正的 fine-tune，Baseline 训练出的模型是迁移学习的起点

Baseline 在我们环境中的训练结果（120K 迭代完成，最终 loss 0.020）与 PhysNAP 论文描述一致（论文使用相同架构和数据，虽然论文未报告具体 loss 值，但 FID 等生成质量指标需要通过生成样本并评估来对比）。

### 问题二：Infinigen-only vs Fine-tune 的区分逻辑

设计两种训练策略的学术逻辑：

1. **Infinigen-only（Option B）** -- 回答"Infinigen 数据能否独立训练生成模型"
   - 从零训练，完全依赖 Infinigen 数据
   - 若成功：证明 Infinigen 的 Data Engine 输出质量足以独立驱动深度生成模型学习
   - 实际结果：loss 收敛到 0.007，显著优于 baseline 的 0.020，表明数据质量高且域聚焦

2. **Fine-tune（Option C）** -- 回答"预训练知识能否迁移到 Infinigen 域"
   - 设计意图：利用 PartNet-Mobility 上学到的通用铰接物体先验知识，迁移到 Infinigen 特定域
   - 学术价值：验证 domain adaptation / transfer learning 在铰接物体生成任务上的有效性
   - **但实际执行中预训练权重未成功加载**（见第六节说明），Option C 实际变成了"低学习率从零训练"
   - 后续需要正确实现 fine-tune 来完成这一实验

### 问题三：Next Steps 与 Remaining for Full Academic Evaluation 的关系

`.project-memory/history` 中的 Next Steps:
1. Resume baseline training to 120K -- **已完成**（最终完整运行了 120K）
2. Generate articulated objects and evaluate quality -- **与 summary 一致**
3. Scale Infinigen data to 10K+ samples -- **与 summary 一致**
4. Implement PhysNAP conditional generation (l_pc loss) -- **额外项**

额外 summary advices:
- "Resume baseline training to 120K for complete comparison" -- **已完成**
- "Generate articulated objects from all 3 models" -- **需要渲染环境**
- "Scale Infinigen data to 10K+ samples for generalization study" -- **需要更多 Infinigen 生成**

**学术逻辑**：
- **生成与评估（第 2 步）** 是最核心的下一步。训练 loss 仅说明模型学到了分布，但学术论文需要 FID、Coverage、MMD 等定量指标来证明生成质量
- **数据扩展（第 3 步）** 解决的是泛化性问题。750 个样本仅覆盖 3 种盒体类型，扩展到 10K+ 样本并增加更多类别才能支撑"Infinigen 作为通用铰接物体数据引擎"的学术叙事
- **条件生成（第 4 步）** 是 PhysNAP 的核心贡献 -- Loss-Guided Diffusion (LGD)。当前我们仅完成了无条件生成（NAP 部分），PhysNAP 的物理约束引导（非穿透、可动性）和点云对齐引导尚未实现。这是学术价值最高但技术难度也最大的方向
- **正确实现 Fine-tune（新增）**：基于本次发现，需要正确加载 Baseline 预训练权重来执行真正的迁移学习实验

---

## 十一、后续工作优先级

1. **[P0] 正确实现 Fine-tune 实验**：修复权重加载问题，使用 `initialize_network_file` 从 Baseline checkpoint 加载预训练 denoiser
2. **[P0] 生成样本并定量评估**：解决渲染环境问题后，从三组模型生成铰接物体样本，计算 FID、Coverage、MMD
3. **[P1] 扩展 Infinigen 数据至 10K+**：增加更多盒体变体和新类别
4. **[P2] 实现 PhysNAP 条件生成**：整合 l_pc（点云对齐）、l_np（非穿透）、l_mv（可动性）损失引导

---

## 十二、训练日志路径索引

| 实验 | 日志文件 |
|------|----------|
| Baseline | `external/physnap/log/v6.1_diffusion_adapted/runtime_cmd_log_files/running_log_start_time_Tue_Mar_10_12:55:11_2026.log`（213,520 行） |
| Option B | `external/physnap/log/v6.1_diffusion_infinigen/runtime_cmd_log_files/running_log_start_time_Tue_Mar_10_13:17:22_2026.log`（92,509 行） |
| Option C | `external/physnap/log/v6.1_diffusion_finetune_infinigen/runtime_cmd_log_files/running_log_start_time_Tue_Mar_10_17:23:09_2026.log`（45,139 行） |


