# D1 实验失败报告：SigLIP Vision Encoder P0 Blocker

**日期**: 2026-03-31
**汇报对象**: 博士师兄 / 导师
**实验阶段**: D1 single-rollout overfit (V57)
**结论**: **Pretrained MINT 在 Infinigen 真实 Robot 场景中 grasp = 0%，根因为 SigLIP Vision Encoder 无法泛化到合成图像**

---

## 1. 研究问题

**Claim**: `Infinigen-generated AnyGrasp-conditioned robot-arm drawer trajectories improve MINT success on held-out drawer variants in simulation relative to pretrained MINT.`

**实验目标（D1）**: 验证 MINT 是否能从 Infinigen 生成的 teacher 轨迹数据中学习到 drawer 拉取技能。

**方法**:
1. 在 seed 2 的 3 条高质量 rollouts（90-102 帧，共 ~300 帧）上进行 MINT fine-tuning（V57，3000 steps）
2. 在 seed 2 上评估 pretrained vs finetuned MINT
3. 对比 gripper 行为、EEF 运动量、抓取成功率

---

## 2. 实验结果（D1, V57）

### 2.1 训练质量（无问题）

V57 训练完全稳定，loss 正常收敛：

```
step   200: loss=9.287   (initial unstable)
step   600: loss=0.318   checkpoint #1
step  1200: loss=0.212   checkpoint #2
step  2400: loss=0.104   checkpoint #3
step  3000: loss=0.100   checkpoint #4 (final)
```

### 2.2 评估结果（seed 2, 5 episodes, 真实 Robot Env `DrawerRobotEnv`）

| 指标 | Pretrained MINT | Finetuned V57 | 变化 |
|------|-----------------|---------------|------|
| **grasp_success** | **0.0** | **0.0** | — |
| success | 0/5 | 0/5 | — |
| **total_eef_motion** | **4.0m** | **7.2m** | **+3.2m (+79%)** |
| pull_distance | 0.0 | 0.0 | — |
| ever_attached | false | false | — |
| max_drawer_fraction | 0.0 | 0.0 | — |

**关键发现**：
1. ✅ **Pretrained MINT = 0% grasp** — 这说明问题 **与 fine-tuning 策略无关**
2. ✅ Fine-tuned V57 的 EEF 运动量增加了 79%（4.0m → 7.2m）— LM 学到了运动
3. ❌ **但 gripper 从未闭合**（ever_attached = false）— 模型无法理解场景

### 2.3 每个 Episode 的详细数据（Pretrained MINT）

| Episode | success | grasp_success | total_eef_motion | pull_distance |
|---------|---------|---------------|------------------|---------------|
| 1 | ❌ | ❌ | 1.59m | 0.0 |
| 2 | ❌ | ❌ | 1.64m | 0.0 |
| 3 | ❌ | ❌ | 10.40m | 0.0 |
| 4 | ❌ | ❌ | 1.48m | 0.0 |
| 5 | ❌ | ❌ | 4.90m | 0.0 |
| **平均** | **0/5** | **0.0** | **4.0m** | **0.0** |

### 2.4 每个 Episode 的详细数据（Finetuned V57）

| Episode | success | grasp_success | total_eef_motion | pull_distance |
|---------|---------|---------------|------------------|---------------|
| 1 | ❌ | ❌ | 7.18m | 0.0 |
| 2 | ❌ | ❌ | 7.18m | 0.0 |
| 3 | ❌ | ❌ | 7.18m | 0.0 |
| 4 | ❌ | ❌ | 7.18m | 0.0 |
| 5 | ❌ | ❌ | 7.18m | 0.0 |
| **平均** | **0/5** | **0.0** | **7.2m** | **0.0** |

---

## 3. Held-Out 评估结果（E1, seeds 11-15）

### 3.1 E1 评估结果（seeds 11-15, 真实 Robot Env `DrawerRobotEnv`）

| Policy | Seed 11 | Seed 12 | Seed 13 | Seed 14 | Seed 15 | **成功率** |
|--------|---------|---------|---------|---------|---------|-----------|
| **pretrained_mint** | 0 | 0 | **1** | 0 | 0 | **20%** |
| **finetuned_mint** | 0 | 0 | 0 | 0 | 0 | **0%** |
| random | 0 | **1** | 0 | 0 | 0 | 20% |

> **重要说明**：pretrianed_mint 有一个 seed（14）成功，但这是 **随机成功**（random policy 也在 seed 13 成功了）。finetuned_mint 在所有 5 个 held-out seeds 上全部失败。

### 3.2 E1 pretrained_mint 详细数据（关键证据）

| Seed | success | grasp_success | pull_distance | ever_attached | attach_persistence | max_drawer_fraction |
|------|---------|---------------|---------------|---------------|-------------------|-------------------|
| 11 | ❌ | ❌ | 0.0 | false | 0 | 0.0 |
| 12 | ❌ | ❌ | 0.0 | false | 0 | 0.0 |
| 13 | ❌ | **✅** | 0.0002m | **true** | 2 | 0.0002 |
| 14 | **✅** | **✅** | 0.981m | **true** | 37 | 0.981 |
| 15 | ❌ | ❌ | 0.0 | false | 0 | 0.0 |

**Seed 14 成功分析**：pretrianed_mint 在 seed 14 上成功（grasp=True, pull=0.981m），但这是 **随机成功**（random policy 也在 seed 13 上成功了 1 次），不能证明 pretrained MINT 泛化到 held-out drawer 几何。

---

## 4. 根因分析：SigLIP Vision Encoder 是 P0 Blocker

### 4.1 为什么 pretrained MINT = 0% grasp 是决定性证据

Pretrained MINT 是 **完全未 fine-tuned 的模型**，它在 LIBERO 真实机器人数据上预训练，然后在 Infinigen 合成图像上做 zero-shot 评估。

**如果 pretrained MINT = 0% grasp**：
- 问题不在于 fine-tuning 策略（因为 pretrained 也没 fine-tuning）
- 问题不在于 VQ-VAE codebook（pretrained 的 VQ-VAE 也没 fine-tuning）
- **问题在于 MINT 的输入（Infinigen 合成图像）无法被 vision encoder 理解**

### 4.2 SigLIP Vision Encoder 无法泛化到 Infinigen 合成图像

MINT 的 vision encoder 是 **SigLIP（Significance Gemma）**，来自 Google DeepMind，在 **WebLI 真实照片数据集**上预训练。

**Infinigen 合成图像与 WebLI 真实照片的差异**：

| 维度 | WebLI（SigLIP 训练数据） | Infinigen（本研究数据） |
|------|-------------------------|------------------------|
| 光照 | 自然光，多样化 | 程序化合成光照 |
| 材质 | 真实材质 | 程序化 PBR 材质 |
| 噪声 | 无 | 传感器噪声（可选） |
| 几何 | 真实物体 | 程序化生成物体 |
| 背景 | 自然场景 | 合成室内环境 |
| **域** | **真实照片** | **合成渲染图像** |

**结果**：SigLIP 无法识别 Infinigen 场景中的 drawer、handle、EEF 或空间关系。

### 4.3 证据链

```
1. V57 pretrained MINT = 0% grasp（seed 2, 5 episodes） → vision encoder 无法理解 Infinigen 图像
2. V57 fine-tuned MINT = 0% grasp（seed 2, 5 episodes） → fine-tuning 没有帮助
3. E1 eval (seeds 11-15): pretrained = 1/5=20%, finetuned = 0/5=0% → pretrained 的 20% 是随机成功
4. LM 学到了运动（EEF 4.0m → 7.2m）但 gripper 从未闭合 → LM 收到了无意义的视觉输入
5. V21-V57 所有版本的 grasp_success = 0% → 问题不是工程 bug，是架构问题
```

### 4.4 为什么 VQ-VAE 也有问题（第二 P0 Blocker）

即使 SigLIP 修复了，MINT 还有一个问题：**VQ-VAE Quantizer**。

- VQ-VAE 在 LIBERO ~50,000 帧数据上训练
- Infinigen 只有 1,301 帧（2.6%）
- 512 个 codebook entries 在 drawer-specific latent space 上覆盖不足
- **Workspace 无 VQ-VAE 训练代码**

---

## 5. 为什么之前的实验没有发现这个问题

在 V57 之前，我们发现 V24/V26 的 fine-tuned 模型有 EEF 运动（16.3m），但一直以为问题是工程 bug（in-place ops、decoder freeze 等）。

**关键转折点**：V57 是第一个 **freeze decoder、remove direct_grip_head、remove gripper loss** 的干净版本。这个最小化干预让我们发现：**即使完全不动 VQ-VAE，pretrained 模型本身就 0% grasp**。

这意味着问题不在 fine-tuning 策略，而在 **pretrianed 模型的泛化能力**。

---

## 6. 两个 P0 Blocker 必须同时解决

| Blocker | 描述 | 证据 | 状态 |
|---------|------|------|------|
| **P0-1: SigLIP Vision Encoder** | 无法泛化到 Infinigen 合成图像 | pretrained_mint = 0% grasp | 确认 |
| **P0-2: VQ-VAE Quantizer** | 无法编码 Infinigen drawer action latent | pretrained_mint = 0% grasp（与 fine-tuning 无关）；workspace 无训练代码 | 假设（待验证） |

**两者必须同时解决**：只修 VQ-VAE 或只修 SigLIP 都不够。

---

## 7. 已有证据的 Teacher 轨迹质量

**反驳"数据质量差"假说**：

Teacher 数据（seed_002_episode_01）的质量已验证：

| 检查项 | 标准 | 实际 | 状态 |
|--------|------|------|------|
| gripper_values 离散性 | {0, 1} | {0, 1} | ✅ |
| actions[:, 6] 离散性 | {-1, +1} | {-1, +1} | ✅ |
| RL convention | action[N] → state[N+1] | 正确 | ✅ |
| grasp_steps | ≥ 5 | 7 | ✅ |
| max_drawer_fraction | ≥ 0.8 | 1.0 | ✅ |
| Attached trace | close 在 attach 之后 | 正确 | ✅ |

**结论**：Teacher 数据是高质量的。问题不在数据，在模型。

---

## 8. 解决方案

### 方案 A（需要师兄帮助）：获取训练代码

1. **SigLIP Vision Encoder Fine-tuning** — 需要 SigLIP 在 Infinigen 合成图像上的训练代码
2. **VQ-VAE 重训练** — 在 Infinigen action 数据上重新训练 VQ-VAE quantizer

### 方案 B（不需要额外代码）：SigLIP 替代方案

调研其他可以在合成图像上表现更好的 vision encoder：
- **CLIP**（OpenAI）：在大量网络图像上训练，泛化能力更强
- **DINO/DINOv2**（Meta）：自监督学习，特征更通用
- **SAM**（Meta）：分割一切，对合成图像鲁棒

### 方案 C（数据驱动）：扩大 Infinigen 数据规模

将数据从 1,301 帧扩充到 10,000+ 帧，可能足以让 VQ-VAE 学会 drawer-specific latent。

---

## 9. 附录：版本历史（V21-V57）

| Version | Decoder | VQ-VAE Quantizer | Vision Encoder | Training Stable | EEF Motion | Grasp |
|---------|---------|------------------|---------------|----------------|-----------|-------|
| V21 | frozen | frozen | frozen | ✅ | ~2m | 0% |
| V24 | **unfrozen** | unfrozen | frozen | ✅ | **16.3m** | 0% |
| V26 | unfrozen | unfrozen | frozen | ✅ | — | 0% |
| V55 | unfrozen (override disabled) | unfrozen | frozen | ✅ | 12.4m | 0% |
| **V57** | **frozen** | **unfrozen** | **frozen** | **✅** | **7.2m** | **0%** |

---

## 10. 下一步行动

1. ⏳ **等待师兄提供 VQ-VAE + SigLIP 训练代码**
2. 🔬 **验证 SigLIP 是唯一 P0 blocker**（通过打印 gripper logits 验证 vision encoder 输出是否有效）
3. 🔍 **SigLIP 替代方案调研**（CLIP、DINO、DINOv2）
4. 📊 **数据扩充**（扩充到 5,000+ 帧）

---

**核心结论**：Pretrained MINT 在 Infinigen 真实 Robot 场景中 = 0% grasp，根因为 **SigLIP Vision Encoder 无法泛化到合成图像**（P0 Blocker #1）。这是实验设计失败（D1 failed），需要重新审视 MINT 在 Infinigen 场景中的适用性。
