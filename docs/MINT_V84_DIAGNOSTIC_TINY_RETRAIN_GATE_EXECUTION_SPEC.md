# v9 Gate-Controlled Diagnostic Tiny-Retrain Plan

## Summary
We treat the current remote sovereign as the only source of truth and execute `v9`
as an implementation plus first-run program, not as a loose training attempt.

Current authoritative starting state:
- top-level head: `125ba7bceb42d9f396bdf643e9e41e09433036e7`
- frozen vendor: `external/MINT@4eab5795345721001c412ff1ca2c886a11eab606`
- downstream corpus is synchronized to the current solved teacher path
- `dataset_valid = true`
- `teacher_readiness_passed = false`
- current failed readiness clauses are exactly:
  - `accepted_unique_teacher_families_ge_18`
  - `near_strict_unique_teacher_families_ge_6`

Current expected first-run outcome:
- `G4 = DIAGNOSTIC_PASS`
- then one diagnostic-only tiny-retrain run to determine whether the remaining
  diversity and near-strict readiness failures are truly causal blockers for trainability

## Freeze Rules
- `Line A` remains operationally frozen
- `Line B` remains frozen
- `external/MINT` remains frozen vendor
- do not edit:
  - `docs/MINT_V84_UNIFIED_EXECUTION_SPEC.md`
  - `docs/MINT_V84_SYSTEM_AUDIT_2026-04-16.md`
  before `G8`

## Gate Artifacts
The runner must emit `G0-G8` under:
- `experiments/mint/mint_drawer_v1/autopilot/gates/`

Each gate artifact must include:
- `gate_id`
- `gate_name`
- `run_instance_id`
- `working_head_commit`
- `vendor_head_commit`
- `branch`
- `truth_contract_hash`
- `acceptance_contract_hash`
- `spec_doc_path`
- `spec_doc_hash`
- `status`
- `blocking_reasons`
- `allowed_next_phases`
- `timestamp_utc`

Allowed statuses:
- `PASS`
- `DIAGNOSTIC_PASS`
- `AUTHORITATIVE_PASS`
- `STOP`

## Runtime Order
1. `prepare`
   - emits `G0-G4`
2. `train`
   - emits `G5`
3. `probe`
   - emits `G6`
4. `eval`
   - emits `G7`
5. `finalize`
   - emits `G8`

## G4 Interpretation
- `AUTHORITATIVE_PASS` only if `teacher_readiness_passed = true`
- `DIAGNOSTIC_PASS` only if:
  - `dataset_valid = true`
  - all non-readiness provenance checks pass
  - `teacher_readiness_passed = false`
  - failed readiness clauses are a subset of:
    - `accepted_unique_teacher_families_ge_18`
    - `near_strict_unique_teacher_families_ge_6`
  - all other readiness clauses pass
- otherwise `STOP`

## Scope Rules
- `DIAGNOSTIC_PASS` at `G4` allows train and probe, but only as diagnostic-only
- held-out eval is blocked in diagnostic mode
- diagnostic outputs must be non-authoritative in scope and narration
- no automatic reopening of `Line A`
- no automatic reopening of `Line B`
- no vendor edits
- no authoritative claim if `G4 != AUTHORITATIVE_PASS`

## Expected First-Run Verdicts
- `diagnostic_learning_signal_readiness_not_yet_authoritative`
- `diagnostic_no_learning_signal_readiness_likely_causal`
- `prepare_stop`
- `claim_supported`
- `invalid_publication_state`

## Publication Gate
At `G8`, enforce:
- frozen docs unchanged since `G0`
- current repo head matches sovereign snapshot
- current vendor head matches sovereign snapshot
- `docs_update_intent.json` exists and matches final scope
- diagnostic runs cannot be narrated as claim-bearing
