# v10 Learning-Support Corpus Fix Spec

## Summary
This spec is grounded on exactly three authoritative objects and nothing older:
- `125ba7b`: current Line A/B/C teacher path truly synchronized into downstream materialization
- `08a1235`: the actual `v9` execution run head and its emitted artifacts
- `64d8bc0`: the publication head that froze the `v9` results, docs, and gate artifacts into repo state

### Adjudication
- Accepted: the `v9` negative is a real post-infra learning failure, not a wrapper/runtime excuse.
- Accepted: training-support mismatch is the strongest current root-cause hypothesis.
- Accepted: current `diversity / near-strict` readiness failures are better treated as proxy symptoms than as the deepest mechanism.
- Qualified: support mismatch is the strongest hypothesis, not a proof; state/conditioning mismatch remains the main secondary hypothesis.
- Rejected: `08a1235` is not the full implementation baseline; it is the execution head, while `64d8bc0` is the implementation/docs publication head.

## Core Diagnosis
Current training data is cut around `bridge_start -> bridge_end` and only keeps `strict_teacher / near_strict_teacher`, while the probe fails at `never_reach_attach_distance`. The next fix must therefore target pre-attach learning support, not only additional post-bridge family diversity.

## Implementation Changes

### 1. Parallel Learning-Support Window
Primary file: `scripts/mint/tiny_retrain_mainline.py`

Keep claim-bearing `canonical_training_truth` unchanged. Add a diagnostic-only parallel object:
- `learning_support_window_start`
- `learning_support_window_end`
- `learning_support_window_len`
- `learning_support_contains_prebridge`
- `learning_support_contains_bridge`
- `learning_support_prebridge_frame_count`
- `learning_support_whole_window_truthful_ratio`
- `learning_support_bridge_truthful_ratio`
- `learning_support_bridge_longest_interior_gap`
- `learning_support_bridge_tail_truthful_count_last6`
- `learning_support_interval_mask_empty`
- `learning_support_interval_anchor_invalid`
- `first_attach_eligible_step`
- `learning_support_anchor_step`
- `measurement_truthful_for_learning_support`
- `learning_support_truth_adjudication = "learning_support_window_v1"`

Algorithm:
1. Reuse `bridge_start / bridge_end` from `_bridge_interval(...)`.
2. Read `attach_step` from `rollout["attach_step"]`, else `strict_metrics["first_attach_step"]`.
3. Read `first_attach_eligible_step` from `attach_eligible_trace`.
4. Define `support_anchor_step = min(non-null of {bridge_start, attach_step, first_attach_eligible_step})`.
5. Use `PREBRIDGE_FRAMES = 24`.
6. Set `learning_support_window_start = max(0, support_anchor_step - PREBRIDGE_FRAMES)`.
7. Set `learning_support_window_end = bridge_end`.
8. Set `learning_support_prebridge_frame_count = max(0, bridge_start - learning_support_window_start)`.
9. Compute whole-window truth metrics on the support window and bridge-specific metrics on `[bridge_start:bridge_end]`.
10. `measurement_truthful_for_learning_support = true` only if:
   - `bridge_start / bridge_end` exist
   - window length `>= 24`
   - prebridge frame count `>= 8`
   - contains both prebridge and bridge
   - whole-window truthful ratio `>= 0.60`
   - bridge truthful ratio `>= 0.80`
   - bridge longest interior gap `<= 2`
   - bridge tail truthful count `>= 5`
   - no bridge interval mask-empty region
   - no bridge interval anchor-invalid region

### 2. Parallel Teacher Classification
Primary file: `scripts/mint/tiny_retrain_mainline.py`

Keep `teacher_episode_class` unchanged:
- `strict_teacher`
- `near_strict_teacher`
- `rejected_teacher`

Add:
- `learning_support_teacher_class`

Rules:
- `strict_teacher` if current strict rule passes
- `near_strict_teacher` if current near-strict rule passes
- `learning_support_teacher` if:
  - `measurement_truthful_for_learning_support = true`
  - contains prebridge and bridge
  - and at least one of:
    - `ever_attached = true`
    - `first_attach_eligible_step is not None`
    - `attach_step is not None`
- otherwise `rejected_teacher`

### 3. Learning-Support Fingerprint
Primary file: `scripts/mint/tiny_retrain_mainline.py`

Keep `teacher_fingerprint` unchanged for claim/readiness.

Add:
- `learning_support_fingerprint`
- `learning_support_fingerprint_version = "v10_learning_support_fingerprint_v1"`

Fingerprint payload:
- `seed`
- `first_attach_step_bucket`
- `first_attach_eligible_step_bucket`
- `learning_support_prebridge_frame_count_bucket`
- `learning_support_window_len_bucket`
- `learning_support_prebridge_truthful_ratio_bucket`
- `attach_persistence_bucket`
- `max_drawer_fraction_bucket`
- `approach_phase_schedule_hash`

### 4. Dual Corpora
Primary file: `scripts/mint/tiny_retrain_mainline.py`

Keep claim corpus output unchanged:
- `g6_canonical_train_rollouts`
- `g6_canonical_rollout_materialization.json`

Add diagnostic corpus:
- `artifacts/g6_learning_support_train_rollouts`
- `artifacts/g6_learning_support_rollout_materialization.json`

Selection:
- canonical corpus keeps only `teacher_episode_class in {strict_teacher, near_strict_teacher}`
- learning-support corpus keeps `learning_support_teacher_class in {strict_teacher, near_strict_teacher, learning_support_teacher}`

Slicing:
- canonical corpus uses `truthful_window_start/end`
- learning-support corpus uses `learning_support_window_start/end`

### 5. Split Dataset Modes
Primary file: `scripts/mint/dataset_builder.py`

Keep canonical validator unchanged and add:
- `validate_rollout_for_learning_support_training(meta, plan)`

Plan field:
- `dataset_selection_mode = "claim_canonical" | "diagnostic_learning_support"`

Diagnostic learning-support validation requires:
- canonical provenance consistency
- `measurement_truthful_for_learning_support = true`
- accepted `learning_support_teacher_class`
- support window len `>= 24`
- prebridge frame count `>= 8`
- contains both prebridge and bridge
- bridge interval free of mask-empty and anchor-invalid segments
- `learning_support_fingerprint` present

### 6. Split Readiness Contracts
Primary file: `scripts/mint/run_tiny_retrain_confirmation.py`

Keep existing `g6_teacher_readiness_contract.json` as claim readiness.

Add:
- `g6_trainability_support_contract.json`

Top-level booleans:
- `claim_readiness_passed`
- `trainability_support_passed`

Trainability-support clauses:
- `learning_support_seed_coverage_all_train_seeds`
- `seed2_learning_support_families_ge_2`
- `seed4_learning_support_families_ge_2`
- `learning_support_unique_teacher_families_ge_12`
- `learning_support_prebridge_frame_p50_ge_16`
- `attach_eligible_seed_coverage_ge_6`

Gate semantics:
- `AUTHORITATIVE_PASS` at `G4` only if `claim_readiness_passed = true`
- `DIAGNOSTIC_PASS` at `G4` only if:
  - dataset is valid
  - `trainability_support_passed = true`
  - claim readiness may still be false
- otherwise `STOP`

### 7. Attach-First Probe Interpretation
Primary files:
- `scripts/mint/run_g8_train_seed_probe.py`
- `scripts/mint/run_tiny_retrain_confirmation.py`

Checkpoint ranking order:
1. `attach_bridge_pass = true`
2. highest `ever_attach_eligible_fraction` for finetuned policy
3. highest `ever_attached_rate_gain`
4. highest `stable_attach_gain`
5. highest `phase_locked_gain`
6. highest `strict_success_rate_gain`
7. latest checkpoint step

Add:
- `ever_attach_eligible_fraction_gain`
- `selected_checkpoint_reason`
- `attach_first_rank_vector`

Diagnostic outcomes:
- `diagnostic_learning_support_signal_detected`
- `diagnostic_learning_support_no_signal`

## Execution Order

### Phase 0
- Base all work on `64d8bc0`
- Treat `08a1235` as frozen execution artifact head only
- Keep vendor frozen at `external/MINT@4eab579...`
- Do not reopen Line A or Line B
- Do not edit claim-bearing truth contract fields or vendor code

### Phase 1
1. Add `learning_support_window`
2. Add `learning_support_teacher_class`
3. Add `learning_support_fingerprint`
4. Add second rollout materialization target

### Phase 2
1. Add `diagnostic_learning_support` dataset mode
2. Add `g6_trainability_support_contract.json`
3. Thread `claim_readiness_passed / trainability_support_passed`
4. Update `G4` logic

### Phase 3
1. Implement attach-first probe ranking
2. Preserve held-out skip for diagnostic mode
3. Keep narrative non-claim-bearing unless `AUTHORITATIVE_PASS`

### Phase 4
1. Run `py_compile`
2. Run CodeRabbit on the committed implementation diff
3. Fix critical or major findings before runtime

### Phase 5
1. Run `prepare` in `diagnostic_learning_support` mode
2. Expect:
   - `claim_readiness_passed = false`
   - `trainability_support_passed = true`
   - `G4 = DIAGNOSTIC_PASS`
3. Run full diagnostic train
4. Run attach-first probe
5. Skip held-out eval under diagnostic scope
6. Finalize to one of:
   - `diagnostic_learning_support_signal_detected`
   - `diagnostic_learning_support_no_signal`
   - `prepare_stop`

## Acceptance
The fix is successful only if the next diagnostic run can cleanly separate:
- claim readiness still false, but learning-support corpus produces attach-learning signal
from
- even after learning-support correction, the model still shows no attach-learning signal

## Executed Local Result (2026-04-18)
The first local `v10` execution order in this worktree reached:
- `G0 PASS`
- `G1 PASS`
- `G2 STOP`
- `G3 STOP`
- `G4 STOP`
- `G8 STOP`

Observed final verdict:
- `prepare_stop`

Observed blocker:
- `materialization_refresh_failed:FileNotFoundError`
- missing local asset root:
  - `sim_exports/urdf/drawer`

Interpretation:
- the `v10` code path itself is implemented, syntax-clean, and its learning-support scenario checks pass locally
- but this local desktop worktree does not contain the authoritative raw drawer asset / rollout source needed to rebuild the corpus
- so the first honest local execution result is a structured `prepare_stop`, not a train/probe verdict

Code review checkpoint:
- CodeRabbit CLI was not available in this environment, so the spec’s external safety-review step could not be executed without installing new tooling
- no replacement CodeRabbit invocation path was exposed by the current desktop tool surface
