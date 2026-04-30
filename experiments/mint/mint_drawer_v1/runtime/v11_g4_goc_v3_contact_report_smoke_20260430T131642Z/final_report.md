# V11-G4 GOC-v3 Contact Report Smoke Closeout

- closeout_classification: `V11_G4_GOC_V3_CONTACT_REPORT_SMOKE_PASSED`
- generated_at_utc: `2026-04-30T13:25:36.480870+00:00`
- task_id: `V11_G4_GOC_V3_CONTACT_REPORT_SMOKE`
- harness_preflight_passed: `True`
- contact_report_smoke_generated: `True`
- runtime_exact_geom_ids_emitted: `True`
- goc_v3_contact_report_validator_passed: `True`
- body_based_31_27_rejected: `True`
- legacy_name_only_report_rejected: `True`
- observed_target_contact_pairs: `[[0, 63], [2, 63], [4, 63], [7, 63], [0, 81], [2, 81], [7, 81]]`
- observed_forbidden_contact_pairs: `[[0, 47], [2, 47], [3, 47], [7, 47], [0, 49], [2, 49], [4, 49], [7, 49], [0, 54], [2, 54], [4, 54], [7, 54]]`
- legacy_body_based_fields_observed_not_used: `True`
- harness_status: `production_ready`
- all_required_regressions_passed: `True`
- attestation_origin_verified: `True`
- attestation_file_blobs_verified: `True`
- current_truth_modified: `False`
- next_actions_modified: `False`
- runtime_code_modified: `False`
- forbidden_dirty: `[]`
- rollout_render_train_run: `False`
- phase1h_rollout_run: `False`
- phase1h_success_claim: `False`
- committed: `pending_final_commit`
- pushed_to_origin: `pending_final_push`
- remote_commit_hash: `recorded_after_push_in_post_push_verification_and_final_answer`
- next_gate: `PHASE1H_BOUNDED_EXECUTION_REVIEW_NOT_AUTORUN`

Notes: this phase used reset-only/no-action contact report inspection. It did not run rollout, render, teacher generation, replay, training, or Phase 1H execution.
The raw runtime report still exposes legacy body-based gripper fields; the canonical smoke report marks them as legacy and the validator rejects using them as authority.
