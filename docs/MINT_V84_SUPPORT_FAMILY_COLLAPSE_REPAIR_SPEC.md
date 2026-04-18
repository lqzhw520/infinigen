# Codex MINT v8.4 / v11 Support-Family Collapse Repair Execution Spec — verified against `dfa0ef0`

## 0. Authority and non-drift context

This spec supersedes the earlier unverified v11 draft. It is grounded on the public GitHub commit and gate artifacts currently reviewable at:

- repo branch: `feature/mint-env-reformulation-v1-visual-fidelity`
- v10 publication commit under review: `dfa0ef0f0acd9443dc010d019d25ad7eb78b40d6`
- v10 execution instance: `v10_20260418T074813Z_64d8bc0b_09bd0ca5`
- v10 execution working head recorded in gates: `64d8bc0b2f27ab4d04880e665e680b2da085d535`
- vendor: `external/MINT@4eab5795345721001c412ff1ca2c886a11eab606`
- active v10 result: `G4 STOP`, `G8 STOP`, final verdict `prepare_stop`

Important interpretation:

- `dfa0ef0...` is the publication commit that lands the v10 code/spec/gate state.
- The gate artifacts correctly record `working_head_commit = 64d8bc0...` because the execution run was started from the pre-publication implementation head and then published into `dfa0ef0`.
- This is not a sovereignty mismatch by itself.
- Do not use older v9/v8 explanations as active blockers unless explicitly referenced by v10 artifacts.

## 1. Final diagnosis

v10 correctly eliminated these earlier blockers:

- environment/path/wrapper drift
- missing learning-support metadata
- missing learning-support truth-summary fields
- dataset invalidation caused by validator field mismatch
- prebridge window absence
- attach-eligible seed coverage gap

v10 did **not** reach diagnostic training because `G4` correctly stopped. The remaining active blocker is:

> **support-family collapse in the learning-support corpus.**

Authoritative v10 numbers:

- `dataset_valid = true`
- `used_rollout_count = 96`
- `effective_frame_count = 7392`
- `learning_support_seed_coverage_all_train_seeds = true`
- `attach_eligible_seed_coverage_ge_6 = true`
- `learning_support_prebridge_frame_p50_ge_16 = true`
- `trainability_support_passed = false`
- failed clauses:
  - `seed2_learning_support_families_ge_2`
  - `seed4_learning_support_families_ge_2`
  - `learning_support_unique_teacher_families_ge_12`
- family statistics:
  - `learning_support_unique_teacher_family_count = 8`
  - `learning_support_family_count_by_seed = 1` for every train seed
  - `learning_support_teacher_class_counts.strict_teacher = 96`

This means:

> The learning-support window fix is connected, but the current teacher generator still produces only one effective learning-support family per seed.

## 2. Why v10 STOP is scientifically correct

v10 had a clean stop rule:

- `AUTHORITATIVE_PASS` only if claim readiness passes.
- `DIAGNOSTIC_PASS` only if dataset is valid and trainability-support passes.
- `STOP` otherwise.

Because trainability-support failed exactly on family-count clauses, continuing to train would have repeated the v9 mistake: testing trainability on a support-collapsed corpus.

So `prepare_stop` is not a failure of the harness. It is the correct result of a stricter and better-defined gate.

## 3. What v11 must decide

v11 must answer one question before generating new training data:

> Is `learning_support_unique_teacher_family_count = 8` a real behavior collapse or an under-expressive fingerprint?

These two cases require different fixes.

### Case A — fingerprint under-expression

Raw pre-attach trajectories are meaningfully different, but `learning_support_fingerprint` collapses them into one family per seed.

Fix:

- do not change teacher generation;
- improve the fingerprint representation;
- rerun `prepare`.

### Case B — true generator collapse

Raw pre-attach trajectories are also one-family-per-seed.

Fix:

- introduce bounded, preattach-only teacher-family variants under the same `interaction_frame_hybrid` controller;
- ensure variants create real effective support trajectories;
- then rerun `prepare`.

Do not skip this distinction.

## 4. Files to change

Only these files may be changed for v11:

1. `scripts/mint/run_v84_learning_support_family_audit.py`
   - new file
2. `scripts/mint/tiny_retrain_mainline.py`
3. `scripts/mint/drawer_robot_env_mujoco.py`
4. `scripts/mint/dataset_builder.py`
5. `scripts/mint/run_tiny_retrain_confirmation.py`
6. `scripts/mint/tiny_retrain_gate_utils.py`
7. `docs/MINT_V84_SUPPORT_FAMILY_COLLAPSE_REPAIR_SPEC.md`
   - new doc for this v11 execution

Do not modify:

- `external/MINT`
- Line A controller verdict documents
- Line B truth contract fields
- claim-bearing truth contract predicates
- authoritative P1C/P1C10 baseline docs
- acceptance thresholds

## 5. Gate structure

### G0 — Sovereign sync

Inputs:

- current branch head
- vendor head
- v10 `G2/G3/G4/G8` gate artifacts
- v10 spec hash

Pass only if:

- branch is `feature/mint-env-reformulation-v1-visual-fidelity`
- vendor is `4eab5795345721001c412ff1ca2c886a11eab606`
- v10 `G3_dataset_integrity.status = PASS`
- v10 `G4_readiness_interpretation.status = STOP`
- v10 `G4.trainability_support_failed_clauses` equals exactly:
  - `seed2_learning_support_families_ge_2`
  - `seed4_learning_support_families_ge_2`
  - `learning_support_unique_teacher_families_ge_12`

Output:

- `experiments/mint/mint_drawer_v1/autopilot/gates_v11/G0_sovereign_sync.json`

### G1 — Family-collapse forensic audit

New script:

```bash
/root/anaconda3/envs/infinigen/bin/python scripts/mint/run_v84_learning_support_family_audit.py \
  --rollout-dir experiments/mint/mint_drawer_v1/artifacts/g6_learning_support_train_rollouts \
  --output experiments/mint/mint_drawer_v1/artifacts/v11_learning_support_family_forensic_audit.json
```

The script must compute two separate family counts:

1. `recorded_learning_support_families`
   - direct unique count from `learning_support_fingerprint`
2. `effective_support_families`
   - recomputed from raw trace signatures, independent of the current fingerprint

For each rollout, extract:

- `seed`
- `learning_support_fingerprint`
- `teacher_episode_class`
- `learning_support_teacher_class`
- `learning_support_window_start`
- `learning_support_window_end`
- `learning_support_prebridge_frame_count`
- `first_attach_eligible_step`
- `first_attach_step`
- `phase_labels`
- EEF positions over the learning-support window if present
- action trace over the learning-support window if present
- close-command trace if present
- distance/attach-eligible trace if present
- orientation/approach alignment trace if present

Define `effective_support_signature_v1` with these components:

- `seed`
- `first_attach_eligible_step_bucket`
- `first_attach_step_bucket`
- `close_onset_step_bucket`
- `prebridge_len_bucket`
- `phase_prebridge_hash`
- `eef_path_signature_hash`
- `action_signature_hash`
- `distance_curve_signature_hash`
- `orientation_curve_signature_hash`

Bucket rules:

- frame buckets: floor to nearest 8 frames, or `"none"`
- EEF path: sample 5 equally spaced prebridge points relative to first point; round to 2 decimal places; hash
- action signature: mean and std over prebridge actions; round to 2 decimal places; hash
- distance curve: min/mean/final distance over prebridge; round to 2 decimal places; hash
- orientation/approach curves: mean/min/final; round to 2 decimal places; hash
- if a trace is missing, use `"missing"` and report missingness explicitly

Audit outputs:

- `recorded_learning_support_unique_family_count`
- `effective_support_unique_family_count`
- `recorded_family_count_by_seed`
- `effective_family_count_by_seed`
- `seed2_effective_support_families`
- `seed4_effective_support_families`
- `fingerprint_underexpression_flag`
- `true_generator_collapse_flag`
- `trace_missingness_summary`
- `recommended_next_action`

G1 pass conditions:

- audit file exists
- no fatal parse errors
- all 96 v10 learning-support rollouts are analyzed or missing rollouts are explicitly reported
- recommendation is one of:
  - `fix_fingerprint_only`
  - `run_bounded_teacher_family_grid`
  - `cannot_decide_trace_missing`

Stop rule:

- If `cannot_decide_trace_missing`, do not inject variants. First fix trace logging.

### G2A — Fingerprint repair branch

Run this branch only if:

- `effective_support_unique_family_count >= 12`
- `seed2_effective_support_families >= 2`
- `seed4_effective_support_families >= 2`
- but recorded family count remains 8 or hard seeds remain 1.

Implementation:

- modify `learning_support_fingerprint` in `tiny_retrain_mainline.py`
- include additional trace-derived fields from `effective_support_signature_v1`
- do not generate new teacher rollouts
- rerun `prepare`
- expect `trainability_support_passed = true`

### G2B — Bounded teacher-family grid branch

Run this branch only if true generator collapse is confirmed.

Add in `tiny_retrain_mainline.py`:

```python
TEACHER_FAMILY_GRID_V11 = [
    {
        "teacher_family_variant": "base",
        "handle_tangent_offset_m": 0.0,
        "handle_vertical_offset_m": 0.0,
        "approach_speed_scale": 1.0,
        "close_distance_offset_m": 0.0,
        "pregrasp_hold_steps": 0,
    },
    {
        "teacher_family_variant": "early_close_slow",
        "handle_tangent_offset_m": 0.0,
        "handle_vertical_offset_m": 0.0,
        "approach_speed_scale": 0.75,
        "close_distance_offset_m": 0.015,
        "pregrasp_hold_steps": 2,
    },
    {
        "teacher_family_variant": "tangent_plus",
        "handle_tangent_offset_m": 0.012,
        "handle_vertical_offset_m": 0.0,
        "approach_speed_scale": 0.9,
        "close_distance_offset_m": 0.005,
        "pregrasp_hold_steps": 1,
    },
    {
        "teacher_family_variant": "vertical_plus",
        "handle_tangent_offset_m": 0.0,
        "handle_vertical_offset_m": 0.010,
        "approach_speed_scale": 0.9,
        "close_distance_offset_m": 0.005,
        "pregrasp_hold_steps": 1,
    },
]
```

These are not new Line A controllers. They are preattach-only entry variants.

In `drawer_robot_env_mujoco.py`, implement a preattach-only hook:

- read `teacher_family_variant` from the rollout/materialization spec
- apply offsets only before attach / stable-attach / phase-lock
- once `stable_attach` or `phase_locked` is true, the normal `interaction_frame_hybrid` opening law must be unchanged
- record all variant parameters in trace/info

Required trace fields:

- `teacher_family_variant`
- `handle_tangent_offset_m`
- `handle_vertical_offset_m`
- `approach_speed_scale`
- `close_distance_offset_m`
- `pregrasp_hold_steps`
- `variant_applied_phase`
- `variant_deactivated_on_attach`

Fake-family prevention:

- nominal `teacher_family_variant` must not be counted as a family by itself
- family count must come from `effective_support_signature_v1`
- if variant labels differ but effective signatures do not differ, G2B fails

Materialization:

- train seeds: `1..8`
- variants: all `TEACHER_FAMILY_GRID_V11`
- target at least 2 effective families for seeds `2` and `4`
- target at least 12 total effective support families
- stop if variants fail to create effective family diversity

### G3 — Dataset rebuild

Run:

```bash
/root/anaconda3/envs/infinigen/bin/python scripts/mint/run_tiny_retrain_confirmation.py --phase prepare --dataset-selection-mode diagnostic_learning_support
```

Expected outputs:

- `g6_learning_support_rollout_materialization.json`
- `g8_canonical_dataset_build.json`
- `g6_trainability_support_contract.json`
- gate files under `autopilot/gates` or `autopilot/gates_v11`

G3 pass conditions:

- `dataset_valid = true`
- `trainability_support_passed = true`
- `learning_support_unique_teacher_families_ge_12 = true`
- `seed2_learning_support_families_ge_2 = true`
- `seed4_learning_support_families_ge_2 = true`
- `learning_support_prebridge_frame_p50_ge_16 = true`
- `attach_eligible_seed_coverage_ge_6 = true`

If G3 fails:

- do not train
- publish `prepare_stop`
- include whether failure is fingerprint, generator, or trace-missing

### G4 — Diagnostic training

Only run if G3 passes.

Run:

```bash
/root/anaconda3/envs/infinigen/bin/python scripts/mint/run_tiny_retrain_confirmation.py --phase train --dataset-selection-mode diagnostic_learning_support
```

Requirements:

- train on the learning-support dataset, not the claim-only canonical corpus
- preserve `diagnostic_only = true`
- checkpoint at the same attach-first probe cadence as v10/v9 infrastructure supports

### G5 — Attach-first probe

Run:

```bash
/root/anaconda3/envs/infinigen/bin/python scripts/mint/run_tiny_retrain_confirmation.py --phase probe --dataset-selection-mode diagnostic_learning_support
```

Checkpoint ranking order:

1. `attach_bridge_pass`
2. highest `ever_attach_eligible_fraction`
3. highest `ever_attached_rate_gain`
4. highest `stable_attach_gain`
5. highest `phase_locked_gain`
6. highest `strict_success_rate_gain`
7. latest checkpoint

Pass signal:

- strict success is not required
- positive diagnostic signal exists if any of:
  - `ever_attach_eligible_fraction_gain > 0.05`
  - `ever_attached_rate_gain > 0.05`
  - `stable_attach_gain > 0.03`
  - `phase_locked_gain > 0.03`
  - dominant failure mode moves later than `never_reach_attach_distance`

If no signal after G3 passed, then support-family collapse is not sufficient as the explanation.

### G6 — Interpretation

Possible final states:

1. `fingerprint_underexpression_fixed_prepare_pass`
   - fingerprint was the blocker, G3 now passes
2. `teacher_family_grid_prepare_pass`
   - generator collapse was fixed, G3 now passes
3. `teacher_family_grid_fake_variants`
   - nominal variants did not create effective support diversity
4. `diagnostic_learning_support_signal_detected`
   - training/probe shows attach-learning signal
5. `diagnostic_learning_support_no_signal_after_support_family_fix`
   - support-family fixed but model still does not learn
6. `trace_logging_insufficient`
   - cannot decide family-collapse source due trace missingness

Next-step mapping:

- If state 4: consider recalibrating readiness / proceed to a larger diagnostic or claim-bearing corpus once claim readiness is addressed.
- If state 5: escalate to state/conditioning/action-label mismatch.
- If state 3 after one repair attempt: escalate to grasp-family diversification frontier assay.
- If state 6: fix trace logging, not teacher generation.

## 6. Why this is the right next solution

v10 proved:

- prebridge learning-support windows exist;
- learning-support dataset is valid;
- seed and attach-eligible coverage are not the blocker;
- family count is the blocker.

Therefore the next experiment must not be more training and must not be more identical rollouts. It must determine whether the apparent family collapse is representational or real. If real, it must generate genuine preattach support variation while keeping the solved opening law fixed.

This is the minimal change that can unblock trainability-support without reopening Line A/B or corrupting claim-bearing truth.

## 7. What not to do

- Do not lower `learning_support_unique_teacher_families_ge_12`.
- Do not lower seed2/seed4 family thresholds.
- Do not train when G3/G4 says STOP.
- Do not count nominal variant IDs as families.
- Do not change claim-bearing truth.
- Do not reopen Line A.
- Do not edit vendor MINT.
- Do not interpret `prepare_stop` as model incapacity.
- Do not interpret a future `no_signal` after G3 pass as teacher-readiness failure; it would be a state/conditioning/action-interface failure.
