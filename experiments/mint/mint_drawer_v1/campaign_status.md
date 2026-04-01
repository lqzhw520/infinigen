<!-- LEGACY DOCUMENT — DO NOT USE AS SOURCE OF TRUTH -->
<!-- Canonical sources: sovereign/ -->
<!-- Generated truth: sovereign/CAMPAIGN_TRUTH.generated.md -->
<!-- Archived location: archive/20260331_legacy/ -->

# MINT Drawer Campaign Dashboard
**真相源**: `experiments/mint/mint_drawer_v1/CAMPAIGN_TRUTH.md` — 所有内容以此为准。

**Updated**: 2026-03-31T19:00:00+08:00
**Campaign**: `mint_drawer_v1`
**Phase**: `mint_robot_trajectory_claim_push`
**Gate**: `clean_c2_available_ready_for_d1`
**Claim**: `Infinigen-generated AnyGrasp-conditioned robot-arm drawer trajectories improve MINT success on held-out drawer variants in simulation relative to pretrained MINT.`
**Revision**: `robot_revision_v3_claim_push`

## Queue

- `u1_asset_geometry_audit`: completed
- `u2_joint_semantics_audit`: completed
- `u3_handle_region_audit`: completed
- `u4_export_consistency_audit`: completed
- `u5_seed_outlier_audit`: completed
- `a1_asset_scene_audit`: completed
- `a2_frame_transform_audit`: completed
- `b1_oracle_scripted_baseline`: completed
- `b2_anygrasp_scripted_baseline`: completed
- `c1_teacher_native_rollout_rebuild`: completed
- `c2_action_contract_repair`: completed
- `c3_single_rollout_replay_gate`: completed
- `d1_single_rollout_overfit`: **failed (attempt 3 = V57)**
- `d2_single_seed_overfit`: pending
- `d3_train_seed_probe`: pending
- `e1_heldout_eval`: pending
- `write_claim_memo`: pending

## Runtime

- Controller ID: `d1_v57`
- Run ID: `d1_v57_main`
- Active Step: `d1_single_rollout_overfit`
- Active Lane: `learnability`
- Active Branch: `None`
- Worker PID: `None`
- Worker Kind: `manual_v57_frozen_decoder`
- Attempt: `3`
- Launched At: `2026-03-30T20:30:00+08:00`
- Heartbeat: `2026-03-30T23:25:51+08:00`
- Latest Artifact: `artifacts/d1_single_rollout_overfit.json`
- Last Error: `u3_regime_b_required`

## V57 Execution Summary

**Version**: V57
**Date**: 2026-03-30
**Root cause approach**: Freeze VQ-VAE decoder (protect LIBERO pretrained gripper), remove direct_grip_head, remove gripper loss
**Status**: FAILED — but a clean failure that confirms the real blocker

### V57 Engineering Verification

| Fix | Versions | Status |
|-----|----------|--------|
| mint_utils.py in-place ops (F.silu, f_hat.add_) | V21 | VERIFIED FIXED |
| VQ-VAE decoder unfreeze path | V24 | VERIFIED FIXED (V57: decoder frozen intentionally) |
| direct_grip_head removal | V57 | VERIFIED REMOVED |
| gripper reconstruction loss removal | V57 | VERIFIED REMOVED |
| Training subprocess (OOM, blocking) | N/A | RESOLVED |

### V57 Key Metrics

| Metric | Pretrained MINT | Finetuned V57 | Gain |
|--------|-----------------|---------------|------|
| success | 0/5 | 0/5 | +0 |
| grasp_success | 0.000 | 0.000 | +0 |
| total_eef_motion | 4.003m | 7.178m | **+3.175m (+79%)** |
| pull_distance | 0.000 | 0.000 | +0 |
| ever_attached | false | false | — |
| max_drawer_fraction | 0.0 | 0.0 | — |

### V57 Training Curve

```
step   200: loss=9.287   (initial unstable)
step   400: loss=1.228
step   600: loss=0.318   (checkpoint #1)
step   800: loss=0.263
step  1000: loss=0.212
step  1200: loss=0.212   (checkpoint #2)
step  1600: loss=0.169
step  2000: loss=0.136
step  2400: loss=0.104   (checkpoint #3)
step  2800: loss=0.120
step  3000: loss=0.100   (checkpoint #4 = final)
```

### V57 Core Finding（已同步至 CAMPAIGN_TRUTH.md v3.0 — 2026-03-31）

**根本原因（P0-Blocker）**：
1. **SigLIP Vision Encoder 无法泛化到 Infinigen 合成图像** — E1 eval: pretrained_mint = 0% grasp（真实 Robot Env）
2. **VQ-VAE Quantizer 无法编码 Infinigen drawer action latent** — Pretrained MINT = 0% grasp（与 fine-tuning 无关）

> **v3.0 关键修正**：PRD Section 1.1 "Archived Proxy Baseline" 数据（fine-tuned = 100%）**不能**证明 SigLIP 可以泛化。
> 原因：`DrawerProxyEnv` 是极简 drawer joint-delta prediction，expert_policy 每步固定 `[0.85, 0, 0, 0, 0, 0, 1.0]`，与 vision encoder 场景理解无关。
> 真实 Robot 环境 (`DrawerRobotEnv`) E1 评估：pretrained_mint = 0% grasp，finetuned_mint = 0% success。
> **SigLIP P0 blocker 是真实存在的；两者都需要解决方案。**

**已排除的假设**：
- ❌ "连续 vs 离散 gripper" — 数据完全是离散二进制
- ❌ "decoder 被 direct_grip_head 破坏" — V57 冻结 decoder 仍是 0% grasp
- ❌ "SigLIP 可以泛化（Proxy Baseline 证明）" — Proxy 是极简任务，不能泛化推论到真实 Robot

**VQ-VAE + Vision Encoder 训练代码不存在**：workspace 无训练代码；博士师兄待提供

### Gripper Statistics (seed_002_episode_01) — 修正版：离散二进制

| Property | Value | Note |
|----------|-------|------|
| Episode length | 90 frames | — |
| Open frames | 67 (74.4%) | — |
| Closed frames | 23 (25.6%) | — |
| Transitions | 2 (open→closed at step 66) | — |
| First close step | 66 | — |
| Gripper pattern | **Discrete binary: {-1.0, 1.0}** | ✅ 修正 |
| gripper_values unique | {0.0, 1.0} | ✅ 离散 |
| states[:, -1] unique | {0.0, 1.0} | ✅ 离散 |

## Teacher Rollout Data (c2_replay_valid_rollouts)

| Seed | Accepted Rollouts | Strong | Coherent |
|------|-----------------|--------|----------|
| 2 | 3 (ep01, ep02, ep04) | 3 | **Yes** (max_phase_dist=0.282) |
| 7 | 1 (ep00) | 1 | No |
| 8 | 4 (ep00-03) | 3 | No |
| 9 | 1 (ep02) | 1 | No |
| 10 | 5 (ep00-04) | 1 | No |
| **Total** | **15** | **9** | **1 (seed 2 only)** |

### Gripper Statistics (seed_002_episode_01)

| Property | Value |
|----------|-------|
| Episode length | 90 frames |
| Open frames | 67 (74.4%) |
| Closed frames | 23 (25.6%) |
| Transitions | 2 (open→closed at step 66) |
| First close step | 66 |
| Gripper pattern | **Discrete binary: {-1.0, 1.0}** | ✅ Verified from NPZ |

## Verdict & Decision

- **Verdict**: `root_cause_confirmed_dual_blocker`
- **Decision**: `await_phd_brother_vqvae_and_siglip_code_and_data_expansion`
- **Next Phase**: `v58_end_to_end_finetune_with_siglip_solution`
- **Note**: `SigLIP P0 blocker 是真实存在的（E1 eval 确认）；Proxy Baseline 100% 成功不能证明 SigLIP 泛化；两者都需要解决方案：VQ-VAE 训练代码 + SigLIP 训练或替换方案。`

## Lane Status

- **Strict Replay Lane**: `completed` — 9 strong rollouts, seed 2 coherent
- **Learnability Lane**: `blocked_by_dual_p0` — V57 confirms SigLIP vision encoder + VQ-VAE quantizer dual blocker (fine-tuning irrelevant; pretrained MINT = 0% grasp)
- **Data Quality Lane**: `pending` — new high-quality rollouts required (5,000+ frames target)
- **D2-Feasible Seeds**: `[2, 8, 10]`

## Recent History

- 2026-03-31T10:00:00+08:00: plan_created | V57 review + Option B VQ-VAE retraining plan
- 2026-03-30T23:25:51+08:00: step_reconciled | d1_single_rollout_overfit => failed (V57, attempt 3)
- 2026-03-30T21:41:54+08:00: step_reconciled | d1_single_rollout_overfit => failed (V57 pre-cleanup)
- 2026-03-30T20:30:00+08:00: v57_launched | V57: frozen decoder, removed direct_grip_head, MINT submodule c43099bd
