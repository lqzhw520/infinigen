<!-- GENERATED FILE — DO NOT EDIT -->
<!-- Source: V58 training logs + eval logs + transcript analysis -->
<!-- Generated at: 2026-04-05T14:00:00+08:00 -->
<!-- HARNESS: sovereign_cli.py go was NOT updated because sovereign files were never written -->
<!-- Issue: The analysis was presented but NOT written to sovereign files -->
<!-- Fix: Write all findings to sovereign, then sovereign_cli.py go will reflect them -->

# CAMPAIGN_TRUTH — V58 Critical Analysis

**Campaign**: `mint_drawer_v1`
**Phase**: V58 End-to-End Failure Root Cause Analysis
**Verdict**: `v58_data_pipeline_quality_confirmed_fail`
**Status**: `await_V58_data_pipeline_fixes`

---

## Executive Summary

V58 MINT fine-tuning + evaluation produced **0% success rate for both pretrained and finetuned models**. This is NOT evidence that MINT cannot learn drawer manipulation. It IS evidence of **catastrophic data quality failure** at the observation interface (L2). The failure is upstream of any model learning capacity.

**Core finding**: Both pretrained=0% and finetuned=0% because the input data is broken at L2 — gripper_joint is frozen constant, images are near-white. Fix the data, then retrain.

---

## Critical Finding 1: Gripper Joint State is **Completely Frozen**

V58 state[7] (gripper_joint):
```
min:  -0.042
max:  -0.042
mean: -0.042
std:   9.6e-9  ← ZERO variation — ALL episodes have the SAME value
```

LIBERO state[7] (gripper_joint):
```
min:  -0.041
max:   0.001
mean: -0.029
std:   0.013  ← Meaningful variation
```

**The gripper_joint value is constant (-0.042) across all 18,701 V58 frames.**

Meanwhile, the gripper **action** in the dataset does vary:
```
V58 action[6] (gripper_command):
  min:  -1.0
  max:  +1.0
  mean: +0.211
  std:   0.977  ← Meaningful variation
```

**Mechanism**: Oracle rollouts record varying gripper actions, but the corresponding gripper joint state never updates. This is a data generation bug — `drawer_robot_env_lerobot.py`'s `_state_vector()` returns a frozen gripper_joint because the continuous gripper conversion fix was applied to `drawer_robot_env.py` (used during evaluation) but NOT to the rollout recording code path.

**Downstream effect**: All 18,701 frames produce the SAME state token for gripper_joint → MINT's state tokenizer receives zero gripper information → language model has no gripper state signal → must rely entirely on vision encoder.

---

## Critical Finding 2: Near-White Images in V58 Dataset

V58 rendered images:
```
mean: [0.976, 0.976, 0.976]  ← ~98% white
std:  [0.003, 0.003, 0.003]  ← essentially uniform
```

LIBERO images:
```
mean: [0.414, 0.371, 0.329]  ← realistic scene colors
std:  [0.252, 0.247, 0.241]  ← rich texture
```

**V58 produces near-white images** — PyBullet ERendered output is essentially a blank image. This provides almost zero useful visual information for a vision encoder pretrained on LIBERO's realistic-looking scenes.

---

## Critical Finding 3: Root Cause Chain — 4-Layer Analysis

| Layer | Target | V58 Status | Evidence | Severity |
|-------|--------|-----------|----------|----------|
| **L1: Task Semantics** | instruction / init_state / success | ✅ Aligned | drawer task defined, success=drawer_fraction≥0.9 | OK |
| **L2: Observation Interface** | image / state fields | ❌ **CATASTROPHIC** | gripper_joint frozen + images near-white | **BLOCKING** |
| **L3: Action Interface** | 7D delta action format | ✅ Aligned | format matches, normalization correct | OK |
| **L4: Distribution Alignment** | visual/dynamic/task distribution | ❌ **CATASTROPHIC** | gripper_joint frozen + images near-white | **BLOCKING** |

**Root cause chain**:
```
Layer 2 (Observation) bug: gripper_joint = -0.042 for ALL frames
    ↓
State quantization: ALL frames produce the SAME state token (bin for -0.042)
    ↓
MINT's language model receives: "State: 000000000000000000000000000 ... Action: ..."
    (256 identical tokens for gripper_joint — zero information)
    ↓
Language model has NO gripper state signal from state input
    ↓
Must rely on vision encoder to infer gripper state
    ↓
Vision encoder sees: near-white images (mean=0.976, std≈0)
    ↓
Vision encoder CANNOT infer gripper state from white noise
    ↓
Model outputs random/biased gripper actions
    ↓
Gripper never closes → never attaches → drawer never opens → 0% success
```

---

## Critical Finding 4: Data Volume Correction

The claim "1,301 frames" was a **stale snapshot** from pre-V58 recovery era:

| Source | Frames | Status |
|--------|--------|--------|
| `sovereign/next_actions.json` | 1,301 | **Stale** — pre-V58 recovery |
| `review.json` | 1,301 | **Stale** — same source |
| V58 training **actually used** | **18,701** | Verified from `v58_train.log` |
| V58 recovery output | **18,906** | 185 rollouts from recovery |
| P4 gate (physics-legal) | **795** | Seeds 1-6 |

**The 1,301 figure was never accurate for V58.** V58 used 18,701 frames from 229 episodes. However, the gripper_joint bug means these 18,701 frames are severely corrupted.

**More data with this bug would not help. Fix the bug first, then generate data.**

---

## Why Pretrained = Finetuned = 0%

Both get 0% because the **input data is broken** at L2:

- **Pretrained MINT**: trained on LIBERO where `state[7]` varies (std=0.013). Its state tokenizer learned embeddings for the full range [-0.041, +0.001]. When it sees constant `-0.042`, it produces degenerate state embeddings → language model confused.
- **Finetuned MINT**: trained on V58's corrupted data where `state[7]` is constant. It learned to ignore state[7] entirely. During finetuning, the model had the same broken vision input (near-white images), so it learned nothing useful about gripper control.

The `finetuned_eef_motion = 4.13m` confirms the model does learn **some** motion (presumably moving toward the drawer based on visual features), but the gripper is the blocker.

---

## Action Normalization — Confirmed Correct

The action normalization is **correct**:

```
MINT output: raw delta action (IDENTITY normalization, no de-normalization)
    ↓ env.step():
clip action[:3] to [-1, 1] → multiply by 0.03m/step
clip action[3:6] to [-1, 1] → multiply by 0.25rad/step
gripper: ≥0 → open, <0 → close
```

V58 action distributions (translation max=0.55, rotation max=0.14) are well within the clipping range. **Action normalization is NOT the root cause.**

---

## GRADE Evidence Quality Assessment

**VERY LOW** (⊕○○○): Multiple highly severe confounding variables prevent causal inference that "MINT cannot learn drawer manipulation."

| Confounder | Severity | Direction |
|------------|----------|-----------|
| Gripper state frozen | Critical | Makes gripper learning impossible |
| Near-white images | Critical | Makes vision learning impossible |
| Action physics violations | High | Makes action space unreliable |
| State format mismatch | Medium | MINT state tokenizer receives degenerate input |

**Conclusion**: V58 0% is NOT a valid test of MINT's learning capacity.

---

## Required Fixes (Priority Order)

| Priority | Action | Impact | Feasibility |
|----------|--------|--------|-------------|
| **P0a** | Fix gripper_joint in recording pipeline: `drawer_robot_env_lerobot.py` → `_state_vector()` — apply same continuous gripper fix as `drawer_robot_env.py` | Unblocks gripper state signal | High |
| **P0b** | Fix image rendering: verify ERendered is actually capturing scene, not outputting blank frames | Unblocks vision encoder signal | High |
| **P1** | Regenerate V58 dataset with fixes (target: 5000+ frames, all physics-legal) | Provides clean training data | High |
| **P2** | Retrain MINT on corrected dataset | Trains model on valid data | High |
| **P3** | Evaluate on held-out seeds (16-20) | Validates learned behavior | Medium |

---

## V58 Training Summary

| Metric | Value |
|--------|-------|
| Frames used | 18,701 (NOT 1,301) |
| Episodes | 229 |
| Seeds | 1-15 |
| Steps | 1,000 |
| Loss trajectory | 6.52 → 5.18 → 4.83 → 4.58 → 4.43 |
| Checkpoint | `artifacts/v58_training_outputs/checkpoints/001000/pretrained_model` |
| Training time | 1,430.8 seconds |

V58 eval results:
- Stage 1 (train seeds 1-10): finetuned=0.000, pretrained=0.000, gain=+0.000
- Stage 2 (held-out seeds 11-15): finetuned=0.000, pretrained=0.000, random=0.0
- Verdict: `scientific_not_supported`

---

## Why Harness Did Not Update

The `sovereign_cli.py go` command reads from `sovereign/next_actions.json`, `sovereign/state.json`, and `sovereign/claims.yaml`. The analysis above was presented in a transcript but **never written to these files**.

**Harness engineering gap**: There's no automated pipeline to translate analysis output into sovereign file updates. The agent must explicitly call `sovereign_cli.py` subcommands (or write directly) to update sovereign state.

**Recommended fix**: After any significant analysis, the agent must:
1. Write analysis to `sovereign/CAMPAIGN_TRUTH.*.generated.md`
2. Update `sovereign/next_actions.json` with new action priorities
3. Update `sovereign/state.json` with new findings
4. Update `sovereign/claims.yaml` with new or revised claims

This is now being done.

---

## Files Updated in This Session

| File | Change |
|------|--------|
| `sovereign/CAMPAIGN_TRUTH.v58_critical_analysis.generated.md` | Created — this file |
| `sovereign/next_actions.json` | Updated — new P0 priorities (P0a_gripper_joint_fix, P0b_image_render_fix) |
| `sovereign/state.json` | Updated — V58 critical findings + new verdict |
| `sovereign/claims.yaml` | New claims: C_GRIPPER_JOINT_FROZEN, C_IMAGE_NEAR_WHITE, C_V58_DATA_CORRUPTED, C_DATA_VOLUME_CORRECTION |

---
<!-- END OF ANALYSIS — run sovereign_cli.py render-truth to update CAMPAIGN_TRUTH.generated.md -->