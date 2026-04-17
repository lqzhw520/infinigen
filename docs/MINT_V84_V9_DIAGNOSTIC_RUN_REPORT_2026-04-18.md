# MINT v8.4 v9 Diagnostic Tiny-Retrain Run Report (2026-04-18)

## Purpose
This document is the current handoff report for Science Agent review.
It consolidates the actual remote-sovereign `v9` execution result, the evidence chain, the negative findings, and the remaining root-cause questions.

## 1. Authority and freeze

### 1.1 Remote sovereign used for execution
- Host: `ssh -p 30017 root@10.210.0.88`
- Repo: `/mnt/afs2/zhuhaowu/infinigen`
- Branch: `feature/mint-env-reformulation-v1-visual-fidelity`
- Execution run head: `08a1235bc229a6f31e7c8f248d26df1af8fbc06d`
- Frozen vendor: `external/MINT@4eab5795345721001c412ff1ca2c886a11eab606`
- Run instance id: `v9_20260417T113557Z_08a1235b_df929974`

### 1.2 Frozen objects during `v9`
- `Line A`: operationally frozen
- `Line B`: frozen
- `external/MINT`: frozen vendor
- No vendor edits were used to obtain the final `v9` result

### 1.3 Authoritative baseline identity used in the current system
- authoritative launcher: `run_p1c10_release_runtime_matched_ab.py --variant-run`
- authoritative python: `/root/anaconda3/envs/mint/bin/python`
- authoritative baseline contract remained external to vendor modifications

## 2. What was executed

The first `v9` execution order was completed end-to-end:
1. `prepare`
2. `train`
3. `probe`
4. `eval`
5. `finalize`

Observed gate sequence:
- `G0 PASS`
- `G1 PASS`
- `G2 PASS`
- `G3 PASS`
- `G4 DIAGNOSTIC_PASS`
- `G5 PASS`
- `G6 STOP`
- `G7 DIAGNOSTIC_PASS`
- `G8 DIAGNOSTIC_PASS`

Final verdict:
- `diagnostic_no_learning_signal_readiness_likely_causal`

## 3. Current synchronized corpus state

Current dataset build after synchronization to the solved teacher path:
- `dataset_valid = true`
- `used_rollout_count = 96`
- `effective_frame_count = 5196`
- `used_successful_seed_count = 8`
- `accepted_seed_coverage = [1, 2, 3, 4, 5, 6, 7, 8]`
- `accepted_unique_teacher_family_count = 8`
- `strict_unique_teacher_family_count = 8`
- `near_strict_unique_teacher_family_count = 0`
- `family_collapse_suspected = true`

Current readiness result:
- `teacher_readiness_passed = false`
- remaining failed clauses are exactly:
  - `accepted_unique_teacher_families_ge_18`
  - `near_strict_unique_teacher_families_ge_6`

Interpretation:
- the stale-artifact explanation is gone
- the current corpus is synchronized and valid
- what remains unresolved is diversity / near-strict readiness, not raw episode count

## 4. What the run positively proved

The negative result does **not** mean the project is still blocked on environment or train launch.
The run positively proved all of the following:
- authoritative baseline governance is functioning
- dataset build is valid under the current synchronized state
- diagnostic train launches successfully
- train completed to `10000` steps
- checkpoints were written through `010000`
- authoritative probe execution under the historical `mint` runtime path works
- diagnostic-only gating correctly prevented held-out overclaim

In short:

> The environment is now open enough to run the experiment honestly.
> The negative result came after execution, not before execution.

## 5. Why the result is negative

### 5.1 High-level explanation
The run failed **behaviorally**, not infrastructurally.

That is why the right summary is:
- not: "environment still not working"
- but: "environment works, training runs, but the model still does not learn the attach/open behavior"

### 5.2 Probe outcome
Current train-seed probe summary:
- `passed = false`
- `trend_passed = false`
- `attach_bridge_pass = false`
- `selected_checkpoint_step = 2000`

Important note:
- the selected step `2000` is a tie-breaking artifact in checkpoint selection
- it is **not** the reason for the verdict
- authoritative per-checkpoint summaries through `2000/4000/6000/8000/10000` remained zero-signal

### 5.3 Attach-bridge evidence
Current attach-bridge summary:

Pretrained:
- `ever_attached_rate = 0.0`
- `stable_attach_rate = 0.0`
- `phase_locked_rate = 0.0`
- `effective_pull_progress_peak_mean = 0.0`
- `max_drawer_fraction_mean = 0.0`
- `strict_success_rate = 0.0`

Finetuned:
- `ever_attached_rate = 0.0`
- `stable_attach_rate = 0.0`
- `phase_locked_rate = 0.0`
- `effective_pull_progress_peak_mean = 0.0`
- `max_drawer_fraction_mean = 0.0`
- `strict_success_rate = 0.0`

Interpretation:
- finetuning did not even rebuild the attach bridge on the train seeds
- the policy failed before the problem could even become a held-out generalization question

### 5.4 Per-checkpoint pattern
Across `2000/4000/6000/8000/10000`:
- pretrained success stayed `0.0`
- finetuned success stayed `0.0`
- pretrained and finetuned attach/phase-lock stayed `0.0`
- dominant failure mode stayed `never_reach_attach_distance`

This pattern is the most important empirical reason the verdict became:
- `diagnostic_no_learning_signal_readiness_likely_causal`

## 6. Why this specifically points back to readiness

The strongest current reasoning chain is:
1. Line A/B/C were already synchronized downstream.
2. The corpus is valid and train is executable.
3. The run is diagnostic-only because readiness is still false.
4. The only remaining readiness failures are diversity / near-strict failures.
5. The trained policy still shows zero attach-learning signal.

So the current best explanation is:

> The remaining diversity collapse and missing near-strict support are still plausible causal blockers for trainability.

This is stronger than before because the run was completed under current synchronized conditions.

## 7. What this does **not** prove

This result does **not** prove a formal impossibility theorem.
It does **not** prove that MINT can never learn from Infinigen data.
It does **not** prove that Line A/B/C were irrelevant.

What it does prove is narrower and more operationally useful:
- under the current synchronized corpus,
- under the current frozen vendor/runtime contract,
- and under a full 10k diagnostic tiny-retrain,
- the policy still did not learn the attach bridge.

That sharply elevates the importance of the remaining readiness failures.

## 8. Current unresolved root-cause questions

These are the questions I want Science Agent to pressure-test.

### Q1. Is diversity the true causal blocker, or is diversity only a proxy for something deeper?
Current evidence points at diversity / near-strict readiness because those are the last failed clauses.
But the deeper mechanism still needs interpretation:
- Is the corpus too collapsed because there is effectively one strict family per seed?
- Is near-strict absence preventing the policy from learning approach/attach curricula?
- Is the model overfitting to narrow strict windows without learning the attach geometry?

### Q2. Why is the learned policy failing specifically at `never_reach_attach_distance`?
This failure mode is earlier than drawer opening.
That raises a deeper question:
- Is the diagnostic run failing because the policy cannot reconstruct the approach-to-handle behavior from the current teacher windows?
- Or is there a representation/conditioning mismatch between the dataset and the policy’s deployed control state?

### Q3. Is the current teacher truth window too narrow for learning approach/attach behavior?
The corpus is truthful and valid under current contracts.
But it may still be too narrow as a learning signal.
Possible concern:
- the dataset may preserve the successful opening window,
- while underrepresenting the lead-in distribution needed to learn reaching/attachment.

### Q4. Is the model actually learning something latent that the current probe is not crediting?
Current evidence argues against this because attach metrics stay zero.
Still, Science Agent should inspect whether:
- low-level action outputs changed in a meaningful but insufficient way,
- or whether the policy remains effectively pretrained-like across all train seeds.

### Q5. Should near-strict support be treated as a curriculum requirement rather than just a readiness bookkeeping clause?
The current result increases the plausibility that `near_strict_unique_teacher_families_ge_6` is not just paperwork.
It may be part of the actual trainability condition.

## 9. My current interpretation

My current best interpretation is:

> We are past the point where environment/runtime confusion is the main story.
> The negative result now looks more like a real learning failure induced by the structure of the current corpus.

More concretely:
- the corpus is large enough to train
- but not rich enough in family diversity / near-strict support to teach attach-reaching behavior
- so the policy never gets off the floor behaviorally

That is the most honest explanation I can defend from the evidence I now have.

## 10. Most important evidence files

Core gate artifacts:
- `experiments/mint/mint_drawer_v1/autopilot/gates/G4_readiness_interpretation.json`
- `experiments/mint/mint_drawer_v1/autopilot/gates/G5_train_launch.json`
- `experiments/mint/mint_drawer_v1/autopilot/gates/G6_probe_analysis.json`
- `experiments/mint/mint_drawer_v1/autopilot/gates/G7_heldout_eligibility.json`
- `experiments/mint/mint_drawer_v1/autopilot/gates/G8_publication_gate.json`

Core run summaries:
- `experiments/mint/mint_drawer_v1/evaluation/tiny_retrain_confirmation_summary.json`
- `experiments/mint/mint_drawer_v1/artifacts/g8_train_summary.json`
- `experiments/mint/mint_drawer_v1/artifacts/g8_train_seed_probe.json`
- `experiments/mint/mint_drawer_v1/artifacts/g8_family_conditioned_probe.json`
- `experiments/mint/mint_drawer_v1/artifacts/g8_attach_bridge_summary.json`
- `experiments/mint/mint_drawer_v1/artifacts/g8_canonical_dataset_build.json`
- `experiments/mint/mint_drawer_v1/artifacts/g6_teacher_readiness_contract.json`

Authoritative per-checkpoint probe summaries:
- `experiments/mint/mint_drawer_v1/evaluation/tiny_retrain/V1cT2S0/s0/honest/authoritative_train_probe_002000_summary.json`
- `...004000_summary.json`
- `...006000_summary.json`
- `...008000_summary.json`
- `...010000_summary.json`
