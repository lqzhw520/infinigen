# Progress

- 2026-03-31T13:00:00+08:00: [MINT] **知识体系统一完成（CAMPAIGN_TRUTH.md 创建）**。所有发现已整合为单一真相源。
  - **创建** `experiments/mint/mint_drawer_v1/CAMPAIGN_TRUTH.md` 作为唯一的真相源文档
  - **关键修正**：VQ-VAE 和 SigLIP Vision Encoder 是耦合的，不是独立的。只重训练 VQ-VAE 不够；必须同时重训练 Vision Encoder + VQ-VAE
  - **VQ-VAE 训练代码不存在**：workspace 搜索完毕，external/MINT/ 和 external/physnap/ 均无训练代码；博士师兄 2 天后提供
  - **根因更新**：SigLIP Vision Encoder 从未见过 Infinigen 合成图像（P0-Blocker）+ VQ-VAE 无法编码 Infinigen drawer latent（P0-Blocker）
  - **Action-Measurement Mismatch 降级**：Y/Z 轴低相关是 drawer 物理约束导致，非阻塞问题
  - **文档已同步**：findings.md, CAMPAIGN_TRUTH.md; RESOLUTION_PLAN.md, campaign_status.md, progress.md 待完成
  - **下一步**：等待师兄 VQ-VAE 代码 + 扩充数据到 5,000+ 帧 + V58 实验

- 2026-03-31T11:00:00+08:00: [MINT] **全面自省完成（Self-Correction v1.0）**。通过直接读取 15 个 strict rollouts NPZ 原始数据，发现并消解了文档内部矛盾。
  - **文档矛盾消解**：多处文档声称数据是"连续的"，但 NPZ 验证 gripper_values ∈ {0, 1}，actions[:, 6] ∈ {-1, +1}，**完全是离散二进制**
  - **重大修正**：连续 vs 离散假设是错误的；VQ-VAE codebook 不匹配不再是 P0
  - **新发现**：Action-Measurement Mismatch — action delta 与 actual EEF movement 相关性 X=0.54, Y/Z=0.16，scale ratio 均值 10.73
  - **PRD 数据管线验证**：数据生成流程符合 PRD 完整定义（Infinigen URDF → robot sim → AnyGrasp → rollout → MINT）
  - **新根因候选**：Physics-Gap（LIBERO real robot vs Infinigen sim）+ 数据量不足（1,301 vs 50,000 帧）+ Observation Mismatch（图像风格差异）
  - **文档已更新**：findings.md, self_correction_report.md, RESOLUTION_PLAN.md（待更新）
  - **下一步**：在执行任何 VQ-VAE 重训练前，必须先验证 Action-Measurement Mismatch 和 LIBERO vs Infinigen 数据对比

- 2026-03-30T21:00:00+08:00: [MINT V57] **执行完成，结果失败**。V57 核心思路转变：从"修复 decoder"改为"保护 pretrained decoder"。冻结 VQ-VAE decoder，删除 direct_grip_head 和 gripper loss。训练正常（loss 9.287→0.100，3000 步），但 grasp_success=0/5。EEF motion=7.2m（+79% vs pretrained 4.0m）证明 LM 学到了运动。**但"连续 vs 离散 gripper"假设已被证实为错误**，详见 2026-03-31 自省报告。

- 2026-03-29T12:14:00+08:00: [MINT V57] 开始执行。Commit c43099bd（Decoder full training）被回滚，删除 direct_grip_head，恢复 frozen decoder。MINT submodule 已更新。

- 2026-03-20T21:00:00+08:00: Synced final Phase 2 terminal verdict into `.project-memory/history/` with a dedicated `claim_not_supported` snapshot.
- 2026-03-20T21:00:00+08:00: Wrote `experiments/physnap/box_conditioning_v2/root_cause_tree.md` to record the strict four-layer negative-result analysis.
- 2026-03-20T21:00:00+08:00: Repaired the project-memory hook contract so post-commit validation no longer rewrites tracked files and re-dirties the worktree.
- 2026-03-25T19:30:00+08:00: [MINT] Fixed Layer 0 bug — `attachment_local` not set in `rollout_policy()`. Built per-seed grasp cache from C2 AnyGrasp NPZ data.
- 2026-03-25T19:45:00+08:00: [MINT] D1 re-run PASSED — seed_009_ep02::extended_overfit promoted (success_gain=0.2, grasp_gain=0.4, pull_gain=0.368).
- 2026-03-25T19:50:00+08:00: [MINT] Generated oracle grasp cache for all 15 seeds (train + held-out) for D3/E1.
- 2026-03-25T19:58:00+08:00: [MINT] D2 launched — base_overfit(600), extended_overfit(1200), phase_balanced_overfit(1200).
- 2026-03-25T21:34:00+08:00: [MINT] D2 FAILED — all 3 variants below gate. Best: extended_overfit 20% success (1/5), success_gain=0.2 but only 1 source rollout.
- 2026-03-25T21:46:00+08:00: [MINT] D3 launched with soft D2 dependency (positive transfer from extended_overfit).
- 2026-03-25T22:27:00+08:00: [MINT] D3 FAILED — finetuned 0% vs pretrained 33% on train seeds. Catastrophic forgetting confirmed.
- 2026-03-25T22:29:00+08:00: [MINT] E1 launched with soft D3 dependency (training completed).
- 2026-03-25T22:43:00+08:00: [MINT] E1 COMPLETED — verdict=`scientific_not_supported`. Finetuned 0% vs pretrained 20% on held-out seeds. Random policy (20%) ties pretrained.
