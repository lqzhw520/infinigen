# Box Prior v1 Positioning

## Current Gate Question

Can Infinigen articulated box assets be converted into a faithful PhysNAP-compatible representation that preserves kinematics and yields healthy GT physics metrics?

This gate must pass before the campaign re-enters the higher-level prior question. Until then, model comparisons are substrate checks, not claim evidence.

## Main Claim

Procedurally generated, physics-valid, multi-state articulated box assets can improve the robustness of articulated generative models beyond zero-articulation-only training.

## Phase Ordering

1. Representation fidelity / benchmark validity
2. Minimal repaired hypothesis test: baseline vs scratch vs true fine-tune
3. Only then mixed replay / partial fine-tune diagnostics

## Secondary Claim

True fine-tuning from the PartNet baseline checkpoint is a necessary control. Scratch training with a lower learning rate is not evidence of transfer.

## Rejected Claims

- Training loss alone proves the paper claim.
- The workflow is already a fully autonomous ARIS-style system.
- PartNet sanity metrics are the headline result for this campaign.

## Required Experiment Matrix

1. Baseline `v6.1_diffusion_adapted` checkpoint `5455.pt`
2. Scratch `v6.1_diffusion_infinigen_k10`
3. True fine-tune `v6.1_diffusion_finetune_infinigen_k10`

## Acceptance Rules

- Every experiment must produce box-domain zero-state and multi-state `MMD / COV / 1NN-acc`
- Every experiment must report `E_pen` and `E_mob`
- Every experiment must have a sample gallery and saved `stats.json`
- The final report must include a PartNet sanity table, but that table is not the main claim
