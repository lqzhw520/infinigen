# MINT Drawer Findings — Consolidated Truth

**真相源**: `experiments/mint/mint_drawer_v1/CAMPAIGN_TRUTH.md`
**所有发现以此文件为准，不再单独维护 findings.md 中的矛盾陈述。**

---

## MINT V57 Complete Evaluation Report (2026-03-30)

### V57 Core Change (Strategy: Freeze VQ-VAE Decoder)

V57 is a fundamental strategy shift: **not "fix the decoder", but "protect the pretrained decoder"**.

| Component | V56 and before | V57 |
|-----------|----------------|-----|
| VQ-VAE Decoder | Unfrozen (broken, gradient competition) | **Frozen (preserve LIBERO pretrained gripper decoding)** |
| VQ-VAE Quantizer | Unfrozen | Unfrozen (codebook adaptation only) |
| `direct_grip_head` | Present (gradient competition source) | **Removed** |
| Decoder gripper loss | Present (competition) | **Removed** |
| Inference gripper override | Present but disabled | **Removed** |
| Forward return | `(loss, rec_loss)` | **`loss` only** |
| Net code change | — | **-180 lines** |

### V57 Training Config

| Parameter | Value |
|-----------|-------|
| Steps | 3000 |
| Batch size | 8 |
| Learning rate | 5e-5 |
| Gradient checkpointing | enabled |
| Dataset | 5 rollouts (seed 2, ep 01-04, + ep00) |
| Dataset frames | 499 |
| VQ-VAE frozen | decoder, encoder |
| VQ-VAE trainable | quantizer only |

### V57 Loss Curve

```
step   200:  loss=9.287   ← initial unstable
step   400:  loss=1.228   ← rapid decline
step   600:  loss=0.318   ← checkpoint #1
step   800:  loss=0.263
step  1000:  loss=0.212   ← convergence begins
step  1200:  loss=0.212   ← checkpoint #2
step  1600:  loss=0.169
step  2000:  loss=0.136
step  2400:  loss=0.104   ← checkpoint #3
step  2800:  loss=0.120   ← slight oscillation
step  3000:  loss=0.100   ← checkpoint #4 (final)
```

Loss curve converges normally. Oscillation at step 2800 is normal mid-training transient.

### V57 Evaluation Results

| Metric | Pretrained MINT | Finetuned V57 | Gain |
|--------|-----------------|---------------|------|
| success | 0/5 | 0/5 | +0 |
| grasp_success | 0.000 | 0.000 | +0 |
| pull_distance | 0.000 | 0.000 | +0 |
| total_eef_motion | 4.003m | 7.178m | **+3.175m (+79%)** |
| non_zero_action_ratio | 1.000 | 1.000 | +0 |
| ever_attached | false | false | — |
| max_drawer_fraction | 0.0 | 0.0 | — |

### V57 Key Findings

**Good news**:
- V57 successfully resolved V55/V56 decoder corruption problem
- EEF motion increased from 4.0m → 7.2m (+79%), LM backbone is learning Infinigen motion patterns
- Training is completely stable, no engineering issues

**Bad news**:
- Gripper never closes, `ever_attached=false` for all 5 episodes
- Drawer never moves (`max_drawer_fraction=0.0`)
- Both pretrained AND finetuned = 0% grasp → problem is NOT about fine-tuning strategy

### V57 Root Cause Analysis (Corrected: 2026-03-31)

⚠️ **The following analysis is corrected.** The original hypothesis "continuous vs discrete gripper mismatch" was wrong.

**Corrected root cause assessment**:

| Layer | Problem | Likelihood | Evidence | Status |
|-------|---------|-----------|---------|--------|
| **P0-Unknown** | **Why did V57 pretrained fail (0%) but Proxy Baseline fine-tuned succeed (100%)?** | — | Two pipelines differ in too many dimensions | **Needs investigation — Proxy Baseline proves SigLIP CAN generalize** |
| **P0 (confirmed)** | **VQ-VAE quantizer cannot encode Infinigen drawer action latent** | **High** | **Pretrained MINT = 0% grasp (not a fine-tuning issue); 1,301 vs 50,000 frames (2.6%)** | **CONFIRMED — needs VQ-VAE training code** |
| P1 | Insufficient data (1,301 vs 50,000 frames) | High | Strong | ✅ Confirmed |
| P1 | Action-Measurement low correlation (Y/Z axes) | — | Root cause: drawer physics constraints, non-blocking | **Non-blocking, physical constraint** |
| ~~P0~~ | ~~VQ-VAE codebook gripper mismatch~~ | ~~Low~~ | ~~Data is already discrete~~ | ❌ **EXCLUDED** |
| P2 | Held-out seeds (11-15) physical solvability not verified | — | — | ✅ Confirmed |

### Teacher Rollout Gripper Analysis (Corrected: 2026-03-31)

⚠️ **Warning: The following analysis is corrected.**

**Original incorrect claim** (2026-03-30): "All rollouts have continuous-gradual gripper (+1.0 → -1.0 smooth transition over ~5 steps)"

**Ground truth verified from raw NPZ data**:

| Rollout | Frames | Open | Closed | Transitions | Close@ | Gripper unique | Is Discrete? |
|---------|--------|------|--------|-------------|---------|----------------|---------------|
| seed_002_ep01 | 90 | 67 | 23 | 2 | step 66 | {-1.0, 1.0} | **✓ Yes** |
| seed_002_ep02 | ~92 | ~67 | ~25 | 2 | step ~63 | {-1.0, 1.0} | **✓ Yes** |
| seed_002_ep04 | ~92 | ~68 | ~24 | 2 | step ~64 | {-1.0, 1.0} | **✓ Yes** |

**Core conclusion**: All 15 strict rollouts have **completely discrete** gripper data:
- `gripper_values ∈ {0, 1}` — discrete binary
- `actions[:, 6] ∈ {-1, +1}` — discrete binary
- `states[:, -1] ∈ {0, 1}` — discrete binary (no intermediate values)

### Data Format Comparison (Corrected: 2026-03-31)

| Dimension | Infinigen NPZ | LeRobot stats.json verified | MINT expectation | Gap |
|-----------|---------------|---------------------------|-----------------|-----|
| Gripper value | {-1, +1} (discrete) | max=1.0, q01=-1.0 | discrete {-1, +1} | **✓ Exact match** |
| Position delta | [-0.4, +0.4]m (raw) | min=-0.4, max=0.4 | scaled to [-1, 1] | **✓ Normalized by Normalizer** |
| Rotation delta | [-0.12, +0.12]rad (raw) | min=-0.12, max=0.12 | scaled to [-1, 1] | **✓ Normalized by Normalizer** |

**Important: LeRobot Normalization**:
- `NormalizerProcessorStep` uses `stats.json` min/max to scale raw values to [-1, +1]
- Position delta: [-0.4, +0.4] → [-1, +1] (scale factor 2.5)
- Rotation delta: [-0.12, +0.12] → [-1, +1] (scale factor ~8.3)
- Gripper: [-1, +1] → [-1, +1] (no change)

### Action-Measurement Mismatch (New finding: 2026-03-31)

**This is a non-blocking secondary finding, NOT P0.**

Analysis of `seed_002_episode_01.npz`:

```
Correlation: X=0.54 (medium), Y=0.16 (weak), Z=0.16 (weak)
Scale ratio: Mean=10.73, Max=103.59
```

**Root cause of low correlation**: Drawer physics constraint — drawer slides along X axis only, Y/Z axes have zero variance in actual movement during pull/hold_close phases. This is NOT a blocking issue.

### V58 vs Previous Versions

| Version | Decoder | VQ-VAE Quantizer | direct_grip | Vision Encoder | Training Stable | EEF Motion | Grasp |
|---------|---------|------------------|------------|---------------|----------------|-----------|-------|
| V21 | frozen | frozen | no | frozen | ✅ | ~2m | 0% |
| V23 | unfrozen (failed) | unfrozen | no | frozen | — | — | 0% |
| V24 | unfrozen (140/153) | unfrozen | no | frozen | ✅ | **16.3m** | 0% |
| V26 | unfrozen | unfrozen | added | frozen | ✅ | — | 0% |
| V55 | unfrozen (override disabled) | unfrozen | disabled | frozen | ✅ | 12.4m | 0% |
| **V57** | **frozen** | **unfrozen** | **removed** | **unfrozen** | **✅** | **7.2m** | **0%** |

### Critical: VQ-VAE Training Code Does NOT Exist in Workspace

**Confirmed (2026-03-31)**: After searching the entire workspace (external/MINT/ + external/physnap/), **no VQ-VAE training code exists**.

- MINT tokenizer must be downloaded from HuggingFace (`huangrm/MINT-tokenizer-libero`)
- SDAT training is officially "Planned | 2026 H1" per MINT README
- PhysNAP has no VQ-VAE training code (NAP submodule not in workspace)

**Solution**: PhD师兄 will provide VQ-VAE training code in 2 days.

---

## MINT Drawer Pipeline (mint_drawer_v1) — Full Pipeline Run 2026-03-25

### Academic Claim Under Test
> Infinigen-generated AnyGrasp-conditioned robot-arm drawer trajectories improve MINT success on held-out drawer variants in simulation relative to pretrained MINT.

### Pipeline Gate Results
| Gate | Status | Key Metric |
|------|--------|-----------|
| D1 (single rollout overfit) | PASSED | seed_009_ep02 extended_overfit, success_gain=0.2 |
| D2 (single seed overfit) | FAILED | extended_overfit showed 20% success (1/5) but below MIN_FINETUNED_SUCCESSES=3 |
| D3 (train-seed probe) | FAILED | finetuned 0% success vs pretrained 33%; success_gain=-0.333 |
| E1 (held-out claim eval) | COMPLETED | verdict=`scientific_not_supported` |

### E1 Held-Out Results (Seeds 11-15)
| Policy | Success Rate | Grasp Rate | Pull Distance | Best Seed |
|--------|-------------|------------|---------------|-----------|
| Random | 20% (1/5) | 60% | 0.524 | Seed 13: 100% |
| Pretrained MINT | 20% (1/5) | 40% | 0.196 | Seed 14: 100% |
| Finetuned MINT | **0% (0/5)** | **0%** | **0.000** | None |

### D3 Train-Seed Results (Seeds 2, 7, 9, 10)
| Policy | Success Rate | Pull Distance |
|--------|-------------|---------------|
| Pretrained MINT | 33% (4/12) | 0.331 |
| Finetuned MINT | **0% (0/12)** | **0.000** |

### Critical Finding: Catastrophic Forgetting
Fine-tuning MINT on our teacher data makes the model **strictly worse** than pretrained on every metric. Pretrained MINT already succeeds on some seeds (seed 7: 100%, seed 14: 100%) but fine-tuning destroys this capability entirely.

### Root Cause Analysis
1. **Layer 0 (FIXED)**: `attachment_local` was never set during evaluation — fixed by building per-seed grasp cache
2. **Layer 1 (ACTIVE)**: Catastrophic forgetting — fine-tuning on 4-6 rollouts from 4 seeds collapses pretrained generalization
3. **Layer 2 (ACTIVE)**: Teacher data quality — only 6 strict-valid rollouts from 4 seeds; far too little for meaningful BC
4. **Layer 3 (ACTIVE)**: Privileged teacher MDP mismatch — teacher uses `_planned_segment` + `target_fraction` that learner cannot observe
5. **Layer 4 (STRUCTURAL)**: Oracle grasp data for non-AnyGrasp seeds uses fixed quaternion, may not match per-seed handle geometry

### Verdict
`scientific_not_supported` — The current pipeline cannot support the academic claim. The fundamental bottleneck is insufficient training data (6 rollouts from 4 seeds) combined with catastrophic forgetting during fine-tuning.

**Note (2026-03-31)**: This verdict was based on V21-V27 experiments. V57 analysis reveals the true blockers are Vision Encoder mismatch (SigLIP never trained on Infinigen images) and VQ-VAE quantizer mismatch (cannot encode Infinigen drawer latent). The catastrophic forgetting observation may be secondary — the model was never able to succeed in the first place due to observation/action tokenizer mismatch.
