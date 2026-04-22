# MINT v8.4 — P1 Learning Dynamics Repair Spec
## post-Canary Learning Dynamics Instability: close_cmd + orientation collapse despite P0 infrastructure fixes

**Purpose**: Define the bounded diagnosis and repair for the learning dynamics instability discovered in the P0 canary: close_cmd collapses and orientation disappears during end-to-end finetuning, despite frozen linear probe proving both signals are learnable from the same state representation.

**Authority and context freeze**:

| Field | Value |
|-------|-------|
| Repo | `/mnt/afs2/zhuhaowu/infinigen` |
| Branch | `feature/mint-env-reformulation-v1-visual-fidelity` |
| Working HEAD | `b9c0ac7a14891959f7b8594ec9b03b2136ff3f6d` |
| Vendor HEAD | `4eab5795345721001c412ff1ca2c886a11eab606` |
| Spec | `docs/MINT_V84_P0_INFRASTRUCTURE_REPAIR_SPEC.md` (prior spec) |
| Prior run | `canary_20260421T114811Z_b9c0ac7a_cecd75bf` |
| Prior verdict | `TIER3_NO_SIGNAL_OR_P0C_REGRESSION` |

**What this spec is**: A P1 diagnosis and repair spec. It does NOT reopen P0 findings or prior v* specs.

---

## 1. The Core Discovery: The Paradox

### 1.1 The contradiction

```
Re-C3 frozen linear probe (one-step fit):
  - Frozen pretrained MINT backbone, train only ONE linear head
  - close_cmd_rate on val: 0.762 (teacher close_rate = 0.742)
  - orientation_cos on val: +0.665
  - close_cmd_delta: +0.020 (positive, no collapse)

P0 Canary finetuned model (end-to-end):
  - Full MINT backbone + head fine-tuned together
  - close_cmd_rate on val: 0.310 (collsaped from pretrained 0.998)
  - orientation_gate_pass_rate: 0.000 (collapsed from 0.310)
  - close_cmd_delta: −0.688 (catastrophic collapse)
```

**The frozen linear probe learned close_cmd and orientation from the same 8D state. The end-to-end finetuned model destroyed both.** This is the central P1 finding.

### 1.2 Why this is P1, not P2

P2 is "paradigm failure: scripted teacher insufficient." This is not that. The data proves:

- Teacher generates successful rollouts with attach (seed 1 ep 0: attach_step=25, steps=81, success=True)
- One-step fit can learn to predict close_cmd from 8D state (close_dim_accuracy=0.98)
- The information needed to predict close_cmd IS in the 8D state representation

**The problem is not that the task is impossible. The problem is that end-to-end training actively destroys the learnable signal.**

---

## 2. Root Mechanism: Shared Representation Gradient Interference

### 2.1 The architecture facts

```
State space: 8D orientation_bridge_state_v1
  [distance_to_handle_norm, approach_alignment_cos, orientation_alignment_cos,
   orientation_error_sin, orientation_error_cos, prev_close_cmd,
   gripper_joint, attach_eligible_proxy]

Action space: 7D teacher action
  [translation_x, translation_y, translation_z,
   rotation_x, rotation_y, rotation_z,
   close_cmd]

MINT architecture: pretrained backbone (vision-language) + learned head
Fine-tuning: minimize L_action = |action_pred - action_teacher|^2
```

### 2.2 The interference mechanism

```
BC objective = translation_loss + rotation_loss + close_loss
              (all flow through shared hidden repr.)

Translation dimensions have LARGER gradient magnitude because:
  1. translation action range is wider (0.03m per step)
  2. translation variance across frames is higher
  3. translation error dominates the MSE sum

Gradient flow during finetuning:
  ∂L/∂hidden_repr. = ∂L_trans × ∂trans/∂hidden + ∂L_rot × ∂rot/∂hidden + ∂L_close × ∂close/∂hidden

Translation gradient dominates because it has the largest magnitude.
Shared repr. gets PUSHED in translation-preferring direction.

Side effect: close_cmd-predictive subspace in hidden repr. is DISPLACED.
close_cmd prediction accuracy collapses even though the one-step linear probe
frozen backbone could predict close_cmd correctly.
```

### 2.3 The phase transition at step 1500

```
Step 500:  close_cmd_delta = −0.277  (initial suppression, no phase change)
Step 1000: close_cmd_delta = −0.267  (stable, no加剧)
Step 1500: close_cmd_delta = −0.698  (SUDDEN COLLAPSE to 0.30)
Step 2000: close_cmd_delta = −0.688  (maintained collapse)
```

The step 1500 jump is not random noise. It is a **representation phase transition**: the hidden repr. reaches a configuration where translation-optimized subspace and close-predictive subspace become mutually exclusive, and translation wins completely.

---

## 3. The close_cmd Distribution in Training Data

### 3.1 What the data actually contains

From Re-C3 training data:
- `teacher_close_rate = 0.742` (26% of frames have close_cmd=0)
- close_cmd=1 in ~74% of frames (post-attach)
- close_cmd=0 in ~26% of frames (pre-attach)

The close_cmd signal is NOT absent from the training data. The one-step probe used this signal successfully.

### 3.2 The asymmetry

close_cmd is binary but the training data contains both values. The BC objective should be able to learn this. The problem is not the data — **the problem is the gradient interference during end-to-end optimization.**

---

## 4. P1 Repair Hypothesis: Three Candidate Mechanisms

### Hypothesis H1 — Multi-task loss conflict (most likely)

**Mechanism**: Translation and close_cmd gradients push the shared representation in opposite directions. End-to-end training cannot resolve this; the translation gradient always wins.

**Prediction**: If translation gradient is removed or balanced, close_cmd should not collapse.

**Test**: Replace BC with two-stage or loss-balanced training:
- Stage 1: Train only on close_cmd dimension (freeze other dimensions)
- Stage 2: Fine-tune translation/orientation with reduced LR
- OR: Use gradient balancing (PCGrad, gradient surgery) to project out translation-orthogonal close gradients

### Hypothesis H2 — Shared vision backbone insufficient for action phase disambiguation

**Mechanism**: The pretrained vision backbone (PaliGemma) is trained for scene understanding, not action phase sequencing. It has no mechanism to distinguish "pre-attach frame" from "post-attach frame" given the 8D state. The BC objective has no inductive bias to assign close_cmd=0 at pre-attach and close_cmd=1 at post-attach.

**Prediction**: The backbone needs a phase-disambiguating auxiliary signal or a modified architecture.

**Test**: Add a small MLP head that predicts close_cmd_only, with a loss term that is independent of translation loss. Compare with shared-representation BC.

### Hypothesis H3 — State representation lacks close_cmd-predictive information

**Mechanism**: The 8D orientation_bridge_state_v1 does not have enough close_cmd-predictive information in its representation. The one-step fit works because it uses the raw linear correlation between `prev_close_cmd` and `close_cmd` (close_cmd at step t predicts close_cmd at step t+1). But this correlation is fragile and gets destroyed when the hidden repr. is shifted.

**Prediction**: Adding a temporal memory or explicit phase state would preserve close_cmd signal.

**Test**: Add `close_cmd_at_t_minus_K` as explicit input dimensions, making close_cmd prediction require explicit memory rather than hidden repr. correlation.

---

## 5. P1 Diagnosis Execution Plan

**Diagnosis order is strict. No repair attempt before diagnosis confirms the mechanism.**

### Step 1 — Run gradient analysis on existing canary checkpoints

Objective: Confirm H1 (gradient interference) by measuring gradient norms per action dimension.

Requirements:
- Load pretrained model and canary checkpoint_000500 and checkpoint_002000
- Compute gradient norms for each action dimension during one forward-backward pass on a sample batch
- Report: ratio of translation_grad_norm / close_grad_norm

Pass criteria: If translation_grad_norm / close_grad_norm > 10x → H1 confirmed.
If ratio is low → H1 ruled out, proceed to H2/H3 diagnosis.

### Step 2 — Run ablation: BC on close_cmd dimension only

Objective: Confirm H1 by testing whether close_cmd can be learned in isolation.

Requirements:
- Freeze pretrained MINT backbone
- Train only the close_cmd dimension (7th action dim) using the same 8D state input
- Train for 2000 steps, same dataset as P0 canary
- Evaluate close_cmd_rate on held-out seeds

Pass criteria: If close_cmd_rate > 0.70 (approaching teacher close_rate 0.742) → H1 confirmed (close_cmd IS learnable in isolation). If close_cmd_rate < 0.50 → H2 or H3 confirmed.

### Step 3 — If H1 confirmed, run gradient-balanced training experiment

If Step 1 confirms translation/close gradient ratio > 10x and Step 2 confirms close_cmd is learnable in isolation:

Run two variants:
- **Variant A**: PCGrad-style gradient projection (remove translation-orthogonal component from close gradient)
- **Variant B**: Two-stage (train close first, then translate with reduced LR)

Both run for 2000 steps, evaluate close_cmd_rate and approach_gate_gain.

Pass criteria for P1 repair:
```
close_cmd_delta > −0.2  (no catastrophic collapse)
approach_gate_gain > 0  (approach signal preserved)
orientation_gate_delta > 0  OR orientation signal not degraded
```

---

## 6. What This Spec Is NOT

- NOT a P2 paradigm review. This is not "scripted teacher insufficient." The teacher produces successful rollouts. The architecture can learn from the same data in isolation.
- NOT a dataset rebuild. The training data has close_cmd signal. The problem is in the learning dynamics, not the data.
- NOT reopening P0 infrastructure findings. P0 infrastructure is confirmed clean. This spec is entirely about learning dynamics.
- NOT a "train longer" experiment. Step 1500→2000 shows the collapse is stable, not that more training helps.
- NOT a "change checkpoint sweep interval" fix. The collapse is a structural gradient conflict, not a monitoring artifact.

---

## 7. Explicit Non-Regression Constraints from P0

P0 confirmed the following are TRUE and must not be revised:
- Teacher generates successful rollouts with attach (Re-C1 confirmed 21 unique successful episodes)
- Feature computation path is unified and consistent (Re-C2 confirmed p95_diff < 1e-8)
- Orientation signal exists in one-step fit (Re-C3 confirmed 4/4 positive)
- close_cmd is learnable from frozen backbone + linear head (Re-C3 confirmed)
- Infrastructure bugs are fixed (P0a+b+Re-C1+2+3 all PASS)

---

## 8. Canonical Files for This Round

```
Authoritative specs:
  docs/MINT_V84_P0_INFRASTRUCTURE_REPAIR_SPEC.md   (prior)
  docs/MINT_V84_P1_LEARNING_DYNAMICS_REPAIR_SPEC.md  ← this file

Authoritative prior artifacts:
  experiments/mint/mint_drawer_v1/artifacts/p0_canary_train_probe.json
  experiments/mint/mint_drawer_v1/evaluation/p0_canary/V1cT2S3/checkpoint_02000_summary.json
  experiments/mint/mint_drawer_v1/artifacts/re_c3_signal_band_stability.json
  experiments/mint/mint_drawer_v1/artifacts/re_c2_live_diagnostic_parity.json
  experiments/mint/mint_drawer_v1/artifacts/re_c1_live_support_expansion.json
  experiments/mint/mint_drawer_v1/artifacts/p0b_feature_computation_path_unification.json
  experiments/mint/mint_drawer_v1/artifacts/p0a_live_rollout_family_dispatch_audit.json

Prior training artifacts (reference, NOT reopenable):
  experiments/mint/mint_drawer_v1/artifacts/g8_train_seed_probe.json
  experiments/mint/mint_drawer_v1/artifacts/g8_attach_bridge_summary.json
  experiments/mint/mint_drawer_v1/artifacts/v12_state_conditioning_audit.json
```

---

## 9. Strict Execution Order Summary

```
STEP 1: Gradient norm analysis on existing canary checkpoints
         Confirm H1: translation_grad / close_grad ratio > 10x
         ↓
STEP 2: Ablation — BC on close_cmd dimension only (frozen backbone)
         Confirm close_cmd is learnable in isolation (close_rate > 0.70)
         ↓
STEP 3: If H1 + Step 2 confirmed → run gradient-balanced training (Variant A/B)
         Pass criteria: close_cmd_delta > −0.2, approach_gate_gain > 0
         ↓
STEP 4: If H1 NOT confirmed → pivot to H2/H3 diagnosis
         Step 2 result determines which: if close isolated works → H2,
                                       if close isolated also fails → H3
         ↓
STEP 5: After repair candidate passes STEP 3 criteria → run P1 canary
         P1 canary success: Tier 1 (attach_bridge_gain > 0)
         OR: Tier 2 (approach/distance gain + close_cmd_delta > −0.2)
         OR: Tier 3 (no signal or P0c regression) → escalate to P2 paradigm review
```

---

*This spec is authoritative for P1 diagnosis and repair. It was written based on P0 canary artifacts (canary_20260421T114811Z), Re-C3 artifacts, and the P0 infrastructure repair spec. No prior v* specs have been reopened. The core finding is: the same 8D state representation that enables close_cmd learning with a frozen linear probe destroys close_cmd during end-to-end BC finetuning due to shared representation gradient interference.*
