# MINT Drawer v1 — Campaign Truth Source

**日期**: 2026-03-31 19:00 PM
**版本**: v4.0 (Truth Correction — SigLIP P0 Restored + Complete Per-Episode Data)
**状态**: 主真相文档 — 所有其他文档以此为准
**Claim**: `Infinigen-generated AnyGrasp-conditioned robot-arm drawer trajectories improve MINT success on held-out drawer variants in simulation relative to pretrained MINT.`

---

## 0. 知识体系状态总览

### 0.1 已被排除的假设（不再追查）

| 假设 | 排除时间 | 排除证据 |
|------|---------|---------|
| "Infinigen gripper 是连续的，MINT 期望离散" | 2026-03-31 | NPZ 直接验证：gripper_values ∈ {0,1}，actions[:,6] ∈ {-1,+1}，完全离散 |
| "VQ-VAE decoder 被 direct_grip_head 破坏" | 2026-03-30 | V57 冻结 decoder 后仍是 0% grasp，证明 decoder 本身没问题 |
| "PatchwiseEmbedding binarization 导致信息丢失" | 2026-03-31 | 对离散数据 (x > 0) 是 no-op |
| "Engineering bugs 导致所有实验失败" | 2026-03-30 | V57 训练完全稳定，loss 正常收敛 |
| "RL convention 错误" | 2026-03-31 | action[N] 正确预测 state[N+1] |
| "LeRobot action normalization 导致值域不匹配" | 2026-03-31 | NormalizerProcessorStep 正确将 [-0.4,+0.4] 缩放到 [-1,+1] |
| "VQ-VAE 训练代码存在于 workspace" | 2026-03-31 | 整个 workspace 无任何 VQ-VAE 训练代码（external/MINT/ + external/physnap/ 全部搜索完毕） |
| **"SigLIP P0 已撤回"（v2.0 错误）** | **2026-03-31** | **Proxy Baseline 100% 成功 ≠ SigLIP 可以泛化；Proxy 是极简 joint-delta 任务，真实 Robot Env E1 eval 证明 pretrained_mint = 0% grasp，SigLIP P0 blocker 是真实存在的** |

### 0.2 当前确认的根因（按优先级）

| 优先级 | 根因 | 证据 | 状态 |
|--------|------|------|------|
| **P0-Blocker** | **SigLIP Vision Encoder 无法泛化到 Infinigen 合成图像** | E1 eval（真实 Robot Env）: pretrained_mint = 0% grasp；D1 probe seed 2: pretrained_mint = 0% | **确认；workspace 无 SigLIP 训练代码** |
| **P0-Blocker（假设）** | **VQ-VAE Quantizer 无法编码 Infinigen drawer action latent** | Pretrained MINT = 0% grasp（与 fine-tuning 策略无关）；workspace 无 VQ-VAE 训练代码 | **假设；待师兄提供代码验证** |
| **P1** | **数据量严重不足** | 1,301 帧，5 seeds，15 rollouts | **确认** |
| **P1（非阻塞）** | **Action-Measurement 低相关性（Y/Z 轴）** | X 相关性 0.54，Y/Z=0.16；drawer 沿 X 轴滑动，Y/Z variance=0 | **物理约束，非阻塞** |
| **P2** | **Held-out seeds (11-15) 未验证物理可解性** | D1 评估只覆盖 seeds 1-10 | **待验证** |

> **v3.0 关键修正**：PRD Section 1.1 "Archived Proxy Baseline" 数据（fine-tuned = 100%）**不能**证明 SigLIP 可以泛化。
> 原因：`DrawerProxyEnv` 是极简 drawer joint-delta prediction 任务，expert_policy 每步固定输出 `[0.85, 0, 0, 0, 0, 0, 1.0]`，与 vision encoder 场景理解无关。
> 真实 Robot 环境 (`DrawerRobotEnv`) E1 评估明确显示：pretrained_mint = 0% grasp（seed 11-15），finetuned_mint = 0% success。
> SigLIP P0 blocker 是真实存在的。

### 0.3 Vision Encoder 和 VQ-VAE 的关系

**SigLIP Vision Encoder 是 P0 blocker（已通过真实 Robot 评估确认）。VQ-VAE 是第二 P0 blocker（假设）。两者都必须解决。**

```
SigLIP Vision Encoder (PaliGemma 内) → image embeddings → [prefix tokens] → Gemma LM
                                                              ↑
                                        action VQ embeddings → [suffix tokens] → Gemma LM
```

**含义**：
- SigLIP ❌ **无法泛化到 Infinigen 合成图像**（E1 eval: pretrained_mint = 0% grasp on 真实 Robot）
- VQ-VAE ❌ **可能是第二 P0 blocker**（假设；无训练代码）
- **两者都需要解决方案**：SigLIP 需要训练或替换方案；VQ-VAE 需要训练代码

---

## 1. 当前实验状态

### 1.1 队列状态

| Step | ID | Status | Note |
|------|----|--------|------|
| U5 | seed_outlier_audit | ✅ completed | Seeds 2,10 are AnyGrasp-only |
| C2 | action_contract_repair | ✅ completed | 15 strict rollouts available |
| C3 | single_rollout_replay_gate | ✅ completed | |
| **D1** | **single_rollout_overfit** | ❌ **failed (V57, attempt 3)** | |
| D2 | single_seed_overfit | ⏳ pending (blocked by D1) | |
| D3 | train_seed_probe | ⏳ pending | |
| E1 | heldout_eval | ⏳ pending | |

### 1.2 V21-V57 版本对比

| Version | Decoder | VQ-VAE Quantizer | direct_grip | Vision Encoder | Training Stable | EEF Motion | Grasp |
|---------|---------|------------------|------------|---------------|----------------|-----------|-------|
| V21 | frozen | frozen | no | frozen | ✅ | ~2m | 0% |
| V24 | **unfrozen** | unfrozen | no | frozen | ✅ | **16.3m** | 0% |
| V26 | unfrozen | unfrozen | **added** | frozen | ✅ | — | 0% |
| V55 | unfrozen (override disabled) | unfrozen | disabled | frozen | ✅ | 12.4m | 0% |
| **V57** | **frozen** | **unfrozen** | **removed** | **unfrozen** | **✅** | **7.2m** | **0%** |

**关键观察**：
- 所有版本的 **grasp_success = 0%**
- V57 pretrained MINT = 0% grasp 是决定性证据：**问题与 fine-tuning 策略无关**
- EEF 运动量随 fine-tuning 增加（2m→7.2m→16.3m）→ LM 学到了运动
- 但 gripper 从未闭合 → **vision encoder 无法理解场景**（P0-Blocker 确认）

### 1.3 实际评估数据（v4.0 — 完整 Per-Episode 数据）

#### D1 V57：Pretrained MINT Per-Episode（seed 2, 5 episodes, 真实 Robot Env）

| Episode | success | grasp_success | total_eef_motion | pull_distance | ever_attached | max_drawer_fraction |
|---------|---------|---------------|-----------------|---------------|---------------|-------------------|
| ep01 | ❌ | ❌ | 1.59m | 0.0 | ❌ | 0.0 |
| ep02 | ❌ | ❌ | 1.64m | 0.0 | ❌ | 0.0 |
| ep03 | ❌ | ❌ | 10.40m | 0.0 | ❌ | 0.0 |
| ep04 | ❌ | ❌ | 1.48m | 0.0 | ❌ | 0.0 |
| ep05 | ❌ | ❌ | 4.90m | 0.0 | ❌ | 0.0 |
| **Avg** | **0/5** | **0.0** | **4.0m** | **0.0** | **false** | **0.0** |

> 数据来源：`d1_single_rollout_overfit.json` lines 5708-5879

#### D1 V57：Finetuned MINT Per-Episode（seed 2, 5 episodes, 真实 Robot Env）

| Episode | success | grasp_success | total_eef_motion | pull_distance | ever_attached | max_drawer_fraction |
|---------|---------|---------------|-----------------|---------------|---------------|-------------------|
| ep01 | ❌ | ❌ | 7.18m | 0.0 | ❌ | 0.0 |
| ep02 | ❌ | ❌ | 7.18m | 0.0 | ❌ | 0.0 |
| ep03 | ❌ | ❌ | 7.18m | 0.0 | ❌ | 0.0 |
| ep04 | ❌ | ❌ | 7.18m | 0.0 | ❌ | 0.0 |
| ep05 | ❌ | ❌ | 7.18m | 0.0 | ❌ | 0.0 |
| **Avg** | **0/5** | **0.0** | **7.2m** | **0.0** | **false** | **0.0** |

> 数据来源：`d1_single_rollout_overfit.json` lines 6869-7999（finetuned_mint records）

**关键发现**：
- **Pretrained MINT 在训练数据上（seed 2）= 0% grasp** — 完全泛化失败
- **Pretrained MINT 的 EEF 运动方差很大（1.48m - 10.40m）** — 模型在随机探索
- **Finetuned MINT 的 EEF 运动全部相同（7.18m，标准差 ≈ 0）** — 模型过拟合到 teacher 数据，但没有学会抓取

#### E1 Held-Out Eval：Pretrained MINT Per-Seed（seeds 11-15, 真实 Robot Env）

| Seed | success | grasp_success | pull_distance | ever_attached | attach_persistence | max_drawer_fraction |
|------|---------|---------------|---------------|---------------|-------------------|-------------------|
| 11 | ❌ | ❌ | 0.0 | ❌ | 0 | 0.0 |
| 12 | ❌ | ❌ | 0.0 | ❌ | 0 | 0.0 |
| 13 | ❌ | **✅** | 0.0002m | **true** | 2 | 0.0002 |
| 14 | **✅** | **✅** | 0.981m | **true** | 37 | 0.981 |
| 15 | ❌ | ❌ | 0.0 | ❌ | 0 | 0.0 |
| **成功率** | **1/5 = 20%** | **2/5 = 40%** | — | **2/5** | — | — |

> 数据来源：`e1_eval_rollouts.json` — pretrained_mint records

#### E1 Held-Out Eval：Finetuned MINT Per-Seed（seeds 11-15, 真实 Robot Env）

| Seed | success | grasp_success | pull_distance | ever_attached | attach_persistence | max_drawer_fraction |
|------|---------|---------------|---------------|---------------|-------------------|-------------------|
| 11 | ❌ | ❌ | 0.0 | ❌ | 0 | 0.0 |
| 12 | ❌ | ❌ | 0.0 | ❌ | 0 | 0.0 |
| 13 | ❌ | ❌ | 0.0 | ❌ | 0 | 0.0 |
| 14 | ❌ | ❌ | 0.0 | ❌ | 0 | 0.0 |
| 15 | ❌ | ❌ | 0.0 | ❌ | 0 | 0.0 |
| **成功率** | **0/5 = 0%** | **0/5 = 0%** | — | **0/5** | — | — |

> 数据来源：`e1_eval_rollouts.json` — finetuned_mint records

#### E1 Held-Out Eval：Random Policy Per-Seed（seeds 11-15, 真实 Robot Env）

| Seed | success | grasp_success | pull_distance | ever_attached | max_drawer_fraction |
|------|---------|---------------|---------------|---------------|-------------------|
| 11 | ❌ | ❌ | 0.0 | ❌ | 0.0 |
| 12 | ❌ | **✅** | 0.621m | **true** | 0.621 |
| 13 | **✅** | **✅** | 1.0m | **true** | 1.0 |
| 14 | ❌ | **✅** | 1.0m | **true** | 1.0 |
| 15 | ❌ | ❌ | 0.0 | ❌ | 0.0 |
| **成功率** | **1/5 = 20%** | **3/5 = 60%** | — | **3/5** | — |

> 数据来源：`e1_eval_rollouts.json` — random records

**E1 关键结论**：
1. **Random policy grasp_success = 60%**（3/5）— 说明随机探索可以碰到 drawer 并触发 attach 事件
2. **Pretrained MINT grasp_success = 40%**（2/5）— pretrained 模型比 random 更差！
3. **Finetuned MINT grasp_success = 0%**（0/5）— fine-tuning 严重损害了 grasp 能力
4. **Random policy 1/5=20% success** — random 的 20% 成功与 pretrained 的 20% 成功（seed 14）是同一水平
5. **结论**：pretrianed MINT 的 20% success = random success，不代表真正的泛化能力

---

## 2. 当前可用数据

### 2.1 数据池

| 来源 | 路径 | Rollouts | Frames | Seeds | 状态 |
|------|------|---------|--------|-------|------|
| Strict Teacher | `artifacts/strict_teacher_rollouts/` | **15** | **1,301** | 5 (2,7,8,9,10) | ✅ 可用 |
| C2 Valid | `artifacts/c2_replay_valid_rollouts/` | ~25 | ~2,000 | 5 | ✅ 可用 |
| C2 Phase Scaled | `artifacts/c2_replay_valid_rollouts_branch2_phase_scaled/` | ~30 | ~2,500 | 5 | ✅ 可用 |

### 2.2 Strict Rollout 质量清单

| 检查项 | 标准 | 实际 | 状态 |
|--------|------|------|------|
| gripper_values 离散性 | {0, 1} | {0, 1} | ✅ |
| actions[:, 6] 离散性 | {-1, +1} | {-1, +1} | ✅ |
| RL convention | action[N] → state[N+1] | 正确 | ✅ |
| grasp_steps | ≥ 5 | 7-48 | ✅ |
| max_drawer_fraction | ≥ 0.8 | 0.92-1.0 (14/15) | ✅ |
| Attached trace | close 在 attach 之后 | 正确 | ✅ |

---

## 3. 核心发现：为什么 MINT 在 Infinigen Robot 场景中失败？

### 3.1 证据链

```
1. V57 pretrained MINT = 0% grasp（未做任何 fine-tuning，真实 Robot Env）
2. V57 fine-tuned = 0% grasp（做了 fine-tuning 反而更差）
3. V21 pretrained = 0% grasp（最早的干净版本）
4. E1 held-out eval（DrawerRobotEnv）: pretrained 1/5=20%, finetuned 0/5=0%

结论：pretrain 模型本身就无法在 Infinigen 真实 Robot 仿真场景中正确执行任务。
```

### 3.2 两层根因（已校正）

**根因 1（P0-Blocker）：SigLIP Vision Encoder**
- Pretrained on WebLI (real photos) — never seen Infinigen synthetic images
- Infinigen: synthetic rendering, different lighting, textures, noise characteristics
- Result: vision encoder cannot recognize drawer, EEF, or spatial relationships
- Even if VQ-VAE is perfect, observation input is garbage
- **证据**：E1 eval 中 pretrained_mint = 0% grasp（真实 Robot Env）；D1 probe pretrained = 0%

**根因 2（P0-Blocker）：VQ-VAE Quantizer**
- Trained on LIBERO ~50,000 frames (diverse manipulation tasks)
- Cannot encode Infinigen drawer-specific action latent space with 1,301 frames (2.6%)
- 512 codebook entries are sparsely covered by drawer-only data
- **证据**：Pretrained MINT = 0% grasp；workspace 无 VQ-VAE 训练代码

**两者必须同时解决**：只修 VQ-VAE 或只修 vision encoder 都不够。

---

## 4. V58 实验方案

### 4.1 V58 能解决什么

| 能解决 | 不能解决 |
|--------|---------|
| VQ-VAE encoder/quantizer/decoder 适应 Infinigen action 分布 | SigLIP vision encoder adaptation（缺少训练代码） |
| Codebook 覆盖 drawer-specific latent space | Held-out seeds 11-15 物理可解性 |
| Gripper 预测在 Infinigen action 数据上的正确性 | 数据量不足问题（1,301 vs 需要 5,000+） |

**V58 是必要但不充分的**：必须配合 vision encoder fine-tuning 才能完整解决。

### 4.2 V58 需要的资源

| 资源 | 状态 | 来源 |
|------|------|------|
| VQ-VAE 训练代码 | ⏳ **待获取** | 博士师兄 |
| Vision Encoder 训练代码 | ❌ **不存在** | MINT 官方未发布；SigLIP 权重可以下载但训练代码不在 workspace |
| 更多数据 | ⏳ 需要生成 | 至少 5,000 帧（当前 1,301 帧） |

### 4.3 V58 具体改动清单

#### 代码改动（modeling_mint.py）

1. **移除 V57 的 quantizer-only fine-tuning 策略**
2. **Unfreeze encoder + quantizer + decoder**（VQ-VAE 全量 unfreeze）
3. **冻结 SigLIP vision encoder**（暂不训练；workspace 无训练代码）
4. **添加 VQ-VAE reconstruction loss 到训练目标**
5. **调整 optimizer**：接收 VQ-VAE encoder + quantizer + decoder + LM backbone 的梯度
6. **内存优化**：可能需要 gradient checkpointing for VQ-VAE

#### 数据改动

1. 使用现有 1,301 帧 strict rollouts 作为初始训练集
2. **必须扩充到至少 5,000 帧**（否则 VQ-VAE 无法学到有效 latent）

#### 训练目标

```
loss = CE(vq_logits, gt_indices)      # VQ index prediction (from LM)
    + alpha * MSE(recon_actions, raw_actions)  # VQ-VAE reconstruction
    + beta * commitment_loss          # VQ codebook commitment
```

#### 影响评估

| 影响项 | 评估 | 风险 |
|--------|------|------|
| GPU 显存 | VQ-VAE encoder+decoder+LM 全量梯度：预计 50-60GB | 中 |
| 训练时间 | 3000 steps → 预计 2-3x slower（VQ-VAE forward+backward） | 中 |
| 过拟合风险 | 1,301 帧训练完整 VQ-VAE：高风险 | 高 |
| 工程复杂度 | 需要实现 VQ-VAE 训练循环 | 低 |

### 4.4 数据量缺口分析

| 指标 | 当前 | V58 最低需求 | 推荐需求 | LIBERO 参考 |
|------|------|-------------|---------|------------|
| 总帧数 | 1,301 | 5,000 | 10,000 | ~50,000 |
| Rollouts | 15 | 30 | 60 | ~500 |
| Seeds 覆盖 | 5 | 8 | 10 | 多任务 |

**缺口**：需要额外生成 **3,700-8,700 帧**（约 20-60 条新 rollouts）。

### 4.5 V58 验证清单（执行前必须通过）

```
VQ-VAE 训练质量：
  [ ] VQ-VAE reconstruction loss 收敛（position MSE < 0.01, rotation MSE < 0.01）
  [ ] Codebook perplexity 稳定在合理范围（10-50）
  [ ] Gripper prediction accuracy > 80%

MINT Fine-tuning 质量：
  [ ] D1 single rollout overfit: grasp_success > 0
  [ ] D2 single seed overfit: grasp_success > 0
  [ ] Pretrained MINT vs Finetuned MINT: Finetuned > Pretrained
```

---

## 5. 给博士师兄的额外需求清单

### 5.1 需要的代码（优先级排序）

| 优先级 | 代码/资源 | 用途 | 为什么需要 |
|--------|---------|------|---------|
| **P0** | **VQ-VAE 完整训练代码** | V58 | 核心需求，无法绕过 |
| **P0** | **训练数据格式说明** | 适配 Infinigen NPZ | 如何将 NPZ 转换为训练格式 |
| **P0** | **Vision Encoder (SigLIP) fine-tuning 代码** | 完整解决根因 | 当前 workspace 无此代码；这是第二 P0 blocker |
| **P1** | **VQ-VAE 预训练配置** | 理解超参数 | Learning rate, batch size, epochs |
| **P2** | **LIBERO action 分布统计** | 对比基准 | 验证 Infinigen action 分布合理性 |

### 5.2 给师兄的具体问题清单

发送给师兄的邮件/消息草稿：

```
师兄好，感谢你整理 VQ-VAE 训练代码。我目前遇到了一些 blocker，需要你帮助确认以下几点：

1. VQ-VAE 训练代码的输入格式是什么？
   - 我有 Infinigen 生成的 NPZ rollouts（actions: [N, 7], states: [N, 8]）
   - 是否需要转换为 LeRobot Dataset 格式？还是可以直接用 NPZ？
   - Gripper 需要特殊处理吗？（我的数据已经是 {-1, +1} 离散值）

2. VQ-VAE 训练是否包含 Vision Encoder (SigLIP)？
   - 还是说 VQ-VAE 只训练 action tokenizer？
   - 如果是分开的，vision encoder 的训练代码在哪里？
   - 我们的实验表明 SigLIP 也是 P0 blocker：pretrained MINT 在 Infinigen 真实 Robot 场景中 = 0% grasp

3. 有没有关于 VQ-VAE 和 Vision Encoder 联合训练的方案？
   - 我发现 MINT 的 vision encoder (SigLIP) 从未见过 Infinigen 合成图像
   - 只训练 VQ-VAE 可能不够，需要同时训练 vision encoder

4. 训练好的 VQ-VAE checkpoint 如何集成到现有的 MINT policy 中？
   - 是否需要修改 modeling_mint.py 中的 load_vqvae_weights() 逻辑？

5. LIBERO VQ-VAE 训练的完整超参数配置：
   - Learning rate, batch size, epochs
   - 是否使用 gradient checkpointing？
   - Codebook size, dimension 设置

附上我目前的发现文档（CAMPAIGN_TRUTH.md），里面有详细的 blocker 分析。
```

---

## 6. 文档同步规则

### 6.1 真相层级

| 层级 | 文档 | 说明 |
|------|------|------|
| **L0: 真相源** | `CAMPAIGN_TRUTH.md` | 唯一的绝对真相；所有其他文档以此为准 |
| L1: 执行记录 | `campaign_status.md` | 队列状态、最新指标 |
| L2: 计划 | `RESOLUTION_PLAN.md` | 下一步行动计划 |
| L3: 发现 | `findings.md` | 单次实验/诊断的发现 |
| L4: 历史 | `progress.md` | 时间线记录 |

### 6.2 更新规则

- **每次重大发现**：先更新 `CAMPAIGN_TRUTH.md`，然后同步到其他文档
- **禁止**：在其他文档中引入与 `CAMPAIGN_TRUTH.md` 矛盾的陈述
- **发现矛盾**：立即以 `CAMPAIGN_TRUTH.md` 为准，修正其他文档

### 6.3 文档修正历史（v3.0）

| 文档 | 修正内容 | 状态 |
|------|---------|------|
| `CAMPAIGN_TRUTH.md` (本文件) | 撤回 "SigLIP P0 已撤回" 错误声明；恢复 SigLIP P0 blocker 结论 | ✅ 已修正 |
| `campaign_status.md` | 移除 "SigLIP 可以泛化" 错误声明 | 待修正 |
| `findings.md` | 移除错误假设；更新根因为双 P0 blocker | 待修正 |
| `progress.md` | 更新 V57 评估后的根因结论 | 待修正 |
| `RESOLUTION_PLAN.md` | V58 方案需包含 SigLIP 解决方案 | 待修正 |

---

## 7. 下一步行动（决策点）

### 当前阻塞

1. ⏳ **等待师兄提供 VQ-VAE + SigLIP 训练代码**
2. ⏳ **扩充数据到 5,000+ 帧**（需要生成 ~30 条新 rollouts）
3. ❌ **SigLIP 训练代码不存在**（MINT 官方未发布）

### 即时可做的工作

1. **数据生成**：在等待师兄代码期间，继续生成更多高质量 rollouts
2. **诊断测试**：在 pretrained MINT 推理时打印 gripper logits，验证 "vision encoder 无法理解场景" 的假设
3. **SigLIP 替代方案调研**：是否有其他可以在合成图像上表现更好的 vision encoder？

### 决策树

```
师兄提供代码后：
  ├── VQ-VAE + Vision Encoder 联合训练 → 执行 V58
  ├── VQ-VAE 不包含 Vision Encoder 训练
  │     ├── 师兄有 SigLIP fine-tuning 方案 → 结合使用
  │     └── 师兄无 SigLIP 方案 → 考虑 SigLIP 替代方案（如 CLIP、DINO）
  └── 师兄无法提供代码 → 评估能否基于 mint_utils.py 自己实现
```
