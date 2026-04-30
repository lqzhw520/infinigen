# V11_G4_TASK_REWRITE_TO_GOC_V3 Execution Contract

Task: V11_G4_TASK_REWRITE_TO_GOC_V3
Type: HARNESS_MAINTENANCE_TASK_SPEC_REWRITE

This stage rewrites the active V11-G4 harness task authority from GOC-v2 count authority to GOC-v3 exact geom-id authority. It does not execute Phase 1H.

## Allowed
- Review exact GOC-v3 IDs and write review artifacts.
- Update `sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml` to require GOC-v3 exact IDs.
- Update `scripts/harness/validators/validate_task_authority.py` to validate GOC-v3 exact sets while preserving v2 compatibility.
- Regenerate production lock and post-push attestation after the task spec and validator change.
- Write run evidence under this run_dir.

## Forbidden
- No rollout, replay, render, teacher generation, training, fine-tuning, or evaluation rollout.
- No runtime/controller patch.
- No direct edit to sovereign/current_truth.json or sovereign/next_actions.json.
- No branch creation and no force push.

## Acceptance
- Current harness preflight passes before changes.
- GOC-v3 ID review explains legal IDs [63, 81, 90] and handle IDs [0..8].
- `validate_task_authority.py` rejects count-only/body-based 31/27 for a v3 task.
- `validate_task_authority.py` accepts exact-set match for GOC-v3.
- Production lock/attestation are regenerated and post-push origin-visible.
- Final preflight passes with no bypass flags.

## Success Closeout
V11_G4_TASK_REWRITE_TO_GOC_V3_READY_FOR_CONTACT_REPORT_SMOKE
