# MINT v8.4 v11 Science Agent Review Handoff (2026-04-19)

## Purpose
This document is the current review handoff for Science Agent after the full remote `v11` execution.
It is meant to do four things:

1. Freeze the authoritative execution context.
2. State the final `v11` result without infra/runtime drift.
3. Explain why the current best interpretation is now: support-family collapse was real and was partially mitigated, but it was not the only root cause.
4. Give Science Agent the exact questions and battle surface needed to return the next root-cause spec.

## 1. Authority and non-drift context

### 1.1 Only authoritative objects
This handoff is grounded on exactly these objects:

- `125ba7b`: the point where current Line A/B/C teacher results were truly synchronized into downstream materialization.
- `08a1235`: the `v9` execution run head and emitted artifacts.
- `64d8bc0`: the `v9` publication head.
- `dfa0ef0f0acd9443dc010d019d25ad7eb78b40d6`: the `v11` execution working head recorded in artifacts and gates.
- `44df1ca31521b202848f3d91ca501b29987e44f2`: the remote publication head that now contains the landed `v11` code, gate artifacts, evaluation outputs, and this review handoff path.
- frozen vendor: `external/MINT@4eab5795345721001c412ff1ca2c886a11eab606`

### 1.2 Authoritative remote environment
- host: `ssh -p 30017 root@10.210.0.88`
- repo: `/mnt/afs2/zhuhaowu/infinigen`
- branch: `feature/mint-env-reformulation-v1-visual-fidelity`
- run instance: `v11_20260418T101620Z_dfa0ef0f_dc2d4581`

### 1.3 What not to use
Do not use any older explanation as an active blocker if it says one of the following:
- teacher sync was still stale
- wrapper/runtime was still the main blocker
- old G6 teacher path was still what the dataset consumed
- the current result was still blocked on infra rather than post-train behavior

Those were earlier stages. They are not the active interpretation anymore.

## 2. What v11 actually executed

The full remote `v11` execution order was completed on the authoritative A800 environment:

1. `G1` forensic audit
2. `G2B` bounded pre-attach support-family diversification
3. `G3` rebuild and readiness refresh
4. `G4` diagnostic train
5. `G5` attach-first probe
6. `G6` final interpretation

Key paths:
- spec: `docs/MINT_V84_SUPPORT_FAMILY_COLLAPSE_REPAIR_SPEC.md`
- forensic audit: `experiments/mint/mint_drawer_v1/artifacts/v11_learning_support_family_forensic_audit.json`
- trainability contract: `experiments/mint/mint_drawer_v1/artifacts/g6_trainability_support_contract.json`
- dataset summary: `experiments/mint/mint_drawer_v1/artifacts/g8_canonical_dataset_build.json`
- train probe: `experiments/mint/mint_drawer_v1/artifacts/g8_train_seed_probe.json`
- family-conditioned probe: `experiments/mint/mint_drawer_v1/artifacts/g8_family_conditioned_probe.json`
- attach-bridge summary: `experiments/mint/mint_drawer_v1/artifacts/g8_attach_bridge_summary.json`
- G6 gate: `experiments/mint/mint_drawer_v1/autopilot/gates_v11/G6_interpretation.json`
- final summary: `experiments/mint/mint_drawer_v1/evaluation/tiny_retrain_confirmation_summary.json`
- eval summary: `experiments/mint/mint_drawer_v1/evaluation/comparison_summary.json`

Final verdict:
- `diagnostic_learning_support_no_signal_after_support_family_fix`

This is the current authoritative `v11` outcome.

## 3. Executive summary

### 3.1 What v11 proved
`v11` proved all of the following:

- the current problem is no longer infra/runtime/wrapper drift;
- support-family collapse in the learning-support corpus was a real dataset-level blocker;
- bounded pre-attach teacher-family diversification repaired that blocker at the dataset/readiness level;
- training then ran honestly under the same frozen vendor/runtime assumptions;
- despite that repair, the learned policy still failed to establish the attach bridge.

### 3.2 The shortest correct conclusion
The shortest correct statement is:

> `v11` repaired the support-family collapse blocker, but did not produce attach-first learning signal. Therefore support-family collapse was a real causal blocker, but not the only root cause. The current main blocker has moved upward toward state/conditioning, orientation-stage support, or architecture-objective mismatch.

## 4. Why this is not infra/runtime anymore

The current result is post-infra, not pre-infra.

Evidence:
- dataset is valid: `dataset_valid = true`
- learning-support trainability gate passes: `trainability_support_passed = true`
- claim readiness still remains false, as expected for a diagnostic-only run
- training completed through the full tiny-retrain path
- probe completed under the authoritative current harness path
- held-out eval was correctly skipped only because the run remained diagnostic-only, not because execution failed

Concretely:
- `experiments/mint/mint_drawer_v1/artifacts/g8_canonical_dataset_build.json`
- `experiments/mint/mint_drawer_v1/autopilot/gates_v11/G4_diagnostic_training.json`
- `experiments/mint/mint_drawer_v1/autopilot/gates_v11/G5_attach_first_probe.json`
- `experiments/mint/mint_drawer_v1/evaluation/comparison_summary.json`

This matters because it eliminates an entire class of earlier excuses.
The negative result now came **after** a real data-repair attempt and a real train/probe cycle.

## 5. Why I judge support-family collapse was a real causal blocker and was partially mitigated

This is the first question I want Science Agent to review hard, so I am spelling out the reasoning chain exactly.

### 5.1 G1 showed the collapse was real, not just a fingerprint artifact
Before diversification, `G1` forensic audit reported:
- `recorded_learning_support_unique_family_count = 8`
- `effective_support_unique_family_count = 8`
- `seed2_effective_support_families = 1`
- `seed4_effective_support_families = 1`
- `fingerprint_underexpression_flag = false`
- `true_generator_collapse_flag = true`

This matters because it rules out the easy interpretation that the current corpus already had hidden support diversity and that only the fingerprint was under-expressive.

In other words:

> Before `G2B`, the current generator was truly producing one effective learning-support family per seed.

### 5.2 G2B/G3 materially repaired the dataset-level blocker
After bounded pre-attach diversification, `G3` reported:
- `dataset_valid = true`
- `claim_readiness_passed = false`
- `trainability_support_passed = true`
- `used_rollout_count = 93`
- `effective_frame_count = 6696`
- `learning_support_unique_teacher_family_count = 31`
- `learning_support_family_count_by_seed = {'1': 4, '2': 4, '3': 4, '4': 3, '5': 4, '6': 4, '7': 4, '8': 4}`
- `attach_eligible_seed_coverage = [1, 2, 3, 4, 5, 6, 7, 8]`
- `learning_support_prebridge_frame_p50 = 22`

Relative to the pre-v11 state, the decisive changes are:
- total learning-support families: `8 -> 31`
- seed 2 families: `1 -> 4`
- seed 4 families: `1 -> 3`
- `trainability_support_passed: false -> true`

So `v11` did not merely create cosmetic variant labels.
It changed the effective trainability contract enough to cross the diagnostic support gate.

### 5.3 G4/G5 showed behavior moved, but not far enough
The learned policy still did not reach attach-first success, but it no longer looked identical to the pre-v11 failure pattern.

Selected probe result at `checkpoint_step = 10000`:
- pretrained dominant failure: `never_reach_attach_distance`
- finetuned dominant failure: `orientation_gate_miss`
- `distance_pass_rate_mean` gain: `+0.08897569511706631`
- `approach_gate_pass_rate_mean` gain: `+0.28472222023022675`
- `orientation_gate_pass_rate_mean` gain: `-0.13411458608849597`
- `ever_attach_eligible_fraction_gain = 0.0`
- `ever_attached_rate_gain = 0.0`
- `stable_attach_gain = 0.0`
- `phase_locked_gain = 0.0`

There is a second important piece of evidence from per-checkpoint summaries:

At `002000`:
- pretrained failure: `never_reach_attach_distance`
- finetuned failure: `orientation_gate_miss`
- pretrained `min_dist_to_handle_mean = 0.2875684220343828`
- finetuned `min_dist_to_handle_mean = 0.09949085271606843`
- pretrained `distance_pass_rate_mean = 0.0`
- finetuned `distance_pass_rate_mean = 0.08159722217048208`
- pretrained `approach_gate_pass_rate_mean = 0.07291666658905645`
- finetuned `approach_gate_pass_rate_mean = 0.3059895808498065`

At `010000`:
- pretrained failure: `never_reach_attach_distance`
- finetuned failure: `orientation_gate_miss`
- pretrained `min_dist_to_handle_mean = 0.2875684220343828`
- finetuned `min_dist_to_handle_mean = 0.047450469030688204`
- pretrained `distance_pass_rate_mean = 0.0`
- finetuned `distance_pass_rate_mean = 0.08897569511706631`
- pretrained `approach_gate_pass_rate_mean = 0.07291666658905645`
- finetuned `approach_gate_pass_rate_mean = 0.3576388868192832`

What this means:
- the repair changed behavior enough to move the failure later in the pipeline;
- the model is getting closer to the handle and passing more of the reach/approach checks;
- but it is still not converting those improvements into attach-eligible or attach-stable behavior.

So the strongest honest interpretation is:

> support-family collapse was a real causal blocker because repairing it changed both the dataset gate outcome and the observed behavior; but the remaining failure shows that support-family collapse was not the whole root cause.

## 6. Why I judge it is not the only root cause

If support-family collapse had been the only blocker, then after `G3 PASS` we would expect at least one of the attach-first indicators to rise:
- `ever_attach_eligible_fraction`
- `ever_attached_rate`
- `stable_attach_rate`
- `phase_locked_rate`

But after repair, all of them remain zero.

This is the critical negative evidence.

The repaired dataset makes the policy better at reaching and approaching, but still not good enough at the next stage that converts approach into attach-eligible alignment.
That means the residual blocker is now more likely in the layer that governs:
- state representation near attach eligibility,
- orientation-stage support,
- or the policy/objective’s ability to exploit the repaired support distribution.

So the root-cause ladder now looks like this:

1. `support-family collapse`
   - real blocker
   - mitigated by `v11`
2. `orientation-stage / state-conditioning / objective mismatch`
   - now the best remaining blocker class
3. `architecture incapacity`
   - still possible, but not yet the best explanation

## 7. Why I recommend the next round focus on orientation-stage support, state-conditioning, and objective mismatch

This is the second question I want Science Agent to review critically.

### 7.1 Why orientation-stage support is now first in line
The most important shift after `v11` is not success; it is the failure mode transition:
- before: `never_reach_attach_distance`
- after: `orientation_gate_miss`

That means the policy is no longer mostly failing to get near the handle.
It is now more often failing after the reach/approach improvement, at the stage where relative pose/orientation needs to become attach-eligible.

So the next diagnostic target should not be “more family count” in the abstract.
It should be:

> what support does the learner have for the orientation-stage transition between approach and attach-eligible?

### 7.2 Why state-conditioning is now a strong candidate
The current active state mode is still the same training/deployment state mode used in this diagnostic path.
If the repaired dataset moves the behavior later but attach-eligible remains zero, one plausible interpretation is that the model does not have the right state support for the pre-attach to attach-eligible transition.

That can happen in several ways:
- the state representation may not encode enough relative pose/orientation information for this stage;
- the information may exist but may be poorly aligned with the decision boundary the policy needs;
- the current conditioning may be sufficient for opening once attached, but not sufficient for the geometry needed to become attach-eligible.

### 7.3 Why objective mismatch is now on the table
`v11` changed the support family distribution without changing vendor/model/hyperparameter regime.
That produced real reach/approach movement, but not attach-bridge establishment.

That pattern is consistent with a policy/objective mismatch story such as:
- the current BC-style training objective rewards the easier reach-like portions of the repaired dataset first,
- but does not create enough learning pressure on the narrow orientation-to-attach transition,
- or the policy class struggles to exploit the repaired support without an additional objective, auxiliary signal, or stage-structured curriculum.

This is why I do **not** think the next spec should be just “v11 but with even more support families.”
That would risk continuing to optimize the old proxy after it has already been repaired enough to reveal the next blocker.

## 8. Concrete questions and battle points for Science Agent

This is the third thing I need from review: not just agreement or disagreement, but pressure-tested alternatives.

### Q1. Do you agree that support-family collapse was a real causal blocker, not just an audit artifact?
Evidence to challenge:
- `G1`: real collapse, not fingerprint under-expression
- `G3`: family count increased `8 -> 31`, hard seeds repaired, trainability support passed
- `G5`: behavior moved later in the pipeline

If you disagree, what stronger alternative explanation fits **all three** observations at once?

### Q2. Do you agree that the current main blocker has moved to orientation-stage / state-conditioning / objective mismatch?
Evidence to challenge:
- failure mode transition from `never_reach_attach_distance` to `orientation_gate_miss`
- positive gains in `min_dist_to_handle_mean`, `distance_pass_rate_mean`, and `approach_gate_pass_rate_mean`
- zero gains in attach-eligible / attach / phase-lock

If you disagree, what blocker class better explains “later-stage movement with zero attach-bridge signal”?

### Q3. Is `orientation_gate_miss` the correct next scientific object, or is it still too downstream/noisy to anchor the next spec?
I need Science Agent to decide whether the next spec should explicitly target:
- orientation-stage support,
- attach-eligible support,
- or an even earlier hidden representation issue that only manifests as orientation miss.

### Q4. Is the current state mode under-specifying the pre-attach geometry?
Please inspect whether the current `m0_proxy`-based representation and training conditioning plausibly underrepresent one or more of:
- relative handle pose,
- local orientation error,
- contact-relevant alignment state,
- or transition structure between approach and attach-eligible.

I need a concrete yes/no judgment here, not just “possible.”

### Q5. Do we need a new diagnostic support object for orientation-stage, analogous to what v10/v11 did for pre-attach support family?
Candidate next objects include:
- orientation-support families,
- attach-eligible transition windows,
- explicit pre-attach-to-attach labels,
- or curriculum partitions around orientation correction.

I need Science Agent to say whether the next spec should stay data-centric, or whether it must now become representation/objective-centric.

### Q6. Is the current policy objective the wrong tool for the residual gap?
Please examine whether the observed pattern is consistent with:
- “data was the problem and now is fixed enough,” versus
- “data repair helped, but the remaining transition is not learnable enough under the current objective.”

If the latter, what is the smallest next-step change that is still scientifically interpretable?

### Q7. What next falsifiable spec should we run?
I need Science Agent to return a next-round spec that is explicit about:
- what the root-cause hypothesis is,
- what changes are allowed,
- what the gate sequence is,
- what counts as support for the hypothesis,
- and what result would falsify it.

## 9. My current working hypothesis

My current best hypothesis is:

> `v11` fixed a real support-family diversity bottleneck, which is why the learner stopped failing purely at reach distance and began failing later at orientation-stage. But because attach-eligible and attach remain zero, the dominant blocker has now moved to the transition between approach and attach-eligible. That transition is more likely limited by state-conditioning and/or objective mismatch than by raw support-family count alone.

This is a hypothesis, not a theorem.
But it is the current hypothesis most tightly supported by the actual `v11` data.

## 10. What I am asking Science Agent to return

I want Science Agent to return a review that includes all of the following:

1. An explicit adjudication of whether it agrees or disagrees with the current root-cause ladder:
   - support-family collapse was real and partially mitigated;
   - current main blocker is now orientation-stage / state-conditioning / objective mismatch.
2. A direct challenge to the weakest parts of that argument.
3. A better alternative explanation if it has one.
4. A next-step root-cause spec with falsifiable gates.
5. A recommendation on whether the next spec should primarily modify:
   - data support at the orientation stage,
   - state representation / conditioning,
   - or the training objective / policy learning setup.

## 11. One important integrity note

During `v11`, there was a late harness-side write-path issue where the `probe` sidecar artifacts were not fully refreshed to the same run instance during the first pass.
That was repaired and replayed **without retraining**, and the final aligned artifacts are now:
- same `run_instance_id = v11_20260418T101620Z_dfa0ef0f_dc2d4581`
- same final verdict
- same diagnostic conclusion

This repair changed artifact consistency, not the scientific outcome.

## 12. Bottom line

The bottom line I want reviewed is:

> `v11` is a real negative, but not the same negative as `v9`.
> `v9` said the support-family-collapsed corpus could not produce learning.
> `v11` says that even after repairing support-family collapse, the learner still does not establish the attach bridge.
> Therefore support-family collapse was real, but the remaining blocker is now higher-level than raw support-family diversity.

That is the current state I need Science Agent to challenge and refine into the next root-cause spec.
