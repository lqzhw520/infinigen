# V11-G4 GOC-v3 Composite Phase1H Bounded Autonomous Run V2

closeout_classification: MODEL_BUILDER_REPAIR_REQUIRED

The composite task spec was created, production lock was rebound to it, and harness preflight passed. Reset-only contact report emitted exact GOC-v3 geom IDs and the authority validator accepted the positive report while rejecting body-based 31/27 and legacy name-only fixtures.

Bounded Phase1H execution was not entered. Reset physics is invalid under the composite spec: forbidden robot link contact with handle/cabinet exists at reset, penetration exceeds 0.02 m, and contact force exceeds 1e6 N.

Qpos-only reset repair probes did not remove the reset forbidden contacts. The next gate is model assembly / clearance repair under GOC-v3, followed by GOC/provenance review if model geometry changes.

Final fields:
```json
{
  "bounded_rollout_attempted": false,
  "closeout_classification": "MODEL_BUILDER_REPAIR_REQUIRED",
  "committed": false,
  "composite_task_spec_lock_bound": true,
  "composite_task_spec_written": true,
  "current_truth_modified": false,
  "exact_goc_v3_ids_emitted": true,
  "forbidden_contacts_at_reset": 12,
  "harness_preflight_passed": true,
  "local_visual_artifact_created": false,
  "max_reset_contact_force_n": 2.0387027460025715e+18,
  "max_reset_penetration_m": 0.42044898910674633,
  "next_actions_modified": false,
  "next_gate": "RESET_CONTACT_OR_MODEL_BUILDER_REPAIR",
  "preaction_contact_smoke_passed": false,
  "pushed_to_origin": false,
  "remote_commit_hash": null,
  "reset_physical_plausibility_passed": false,
  "rollout_attempts_used": 0,
  "runtime_patch_applied": false,
  "runtime_patch_files": [],
  "strict_candidate_drawer_fraction": null,
  "strict_candidate_found": false,
  "strict_candidate_has_forbidden_contact": null,
  "strict_candidate_has_goc_v3_target_contact": false
}
```

## Post-Push Verification

```json
{
  "all_origin_visible": true,
  "generated_at_utc": "2026-04-30T14:18:44.579085Z",
  "local_head": "f49e0c7c59b282099dbaac7912bede9d2bcdbf8e",
  "origin_visible_paths": {
    "experiments/mint/mint_drawer_v1/autopilot/agent_execution_harness_attestation.json": true,
    "experiments/mint/mint_drawer_v1/autopilot/agent_execution_harness_lock.json": true,
    "experiments/mint/mint_drawer_v1/runtime/v11_g4_goc_v3_composite_phase1h_bounded_20260430T140419Z/bounded_attempts.jsonl": true,
    "experiments/mint/mint_drawer_v1/runtime/v11_g4_goc_v3_composite_phase1h_bounded_20260430T140419Z/candidate_selection_report.json": true,
    "experiments/mint/mint_drawer_v1/runtime/v11_g4_goc_v3_composite_phase1h_bounded_20260430T140419Z/contact_report_physical_plausibility.json": true,
    "experiments/mint/mint_drawer_v1/runtime/v11_g4_goc_v3_composite_phase1h_bounded_20260430T140419Z/final_report.md": true,
    "experiments/mint/mint_drawer_v1/runtime/v11_g4_goc_v3_composite_phase1h_bounded_20260430T140419Z/minimal_repair_decision.json": true,
    "experiments/mint/mint_drawer_v1/runtime/v11_g4_goc_v3_composite_phase1h_bounded_20260430T140419Z/preaction_contact_smoke.json": true,
    "experiments/mint/mint_drawer_v1/sovereign/experiment_specs/v11_g4_goc_v3_composite_phase1h_bounded_autonomous.yaml": true,
    "experiments/mint/mint_drawer_v1/sovereign/proposed_current_truth_delta_phase1h_goc_v3_composite.json": true,
    "experiments/mint/mint_drawer_v1/sovereign/proposed_next_actions_phase1h_goc_v3_composite.json": true
  },
  "pushed_to_origin": true,
  "remote_head": "f49e0c7c59b282099dbaac7912bede9d2bcdbf8e",
  "remote_ref": "my-origin/feature/mint-env-reformulation-v1-visual-fidelity",
  "stage": "stage10_post_push_verification"
}
```

Updated final fields after push:

```json
{
  "bounded_rollout_attempted": false,
  "closeout_classification": "MODEL_BUILDER_REPAIR_REQUIRED",
  "committed": true,
  "composite_task_spec_lock_bound": true,
  "composite_task_spec_written": true,
  "current_truth_modified": false,
  "exact_goc_v3_ids_emitted": true,
  "forbidden_contacts_at_reset": 12,
  "harness_preflight_passed": true,
  "local_visual_artifact_created": false,
  "max_reset_contact_force_n": 2.0387027460025715e+18,
  "max_reset_penetration_m": 0.42044898910674633,
  "next_actions_modified": false,
  "next_gate": "RESET_CONTACT_OR_MODEL_BUILDER_REPAIR",
  "preaction_contact_smoke_passed": false,
  "pushed_to_origin": true,
  "remote_commit_hash": "f49e0c7c59b282099dbaac7912bede9d2bcdbf8e",
  "reset_physical_plausibility_passed": false,
  "rollout_attempts_used": 0,
  "runtime_patch_applied": false,
  "runtime_patch_files": [],
  "strict_candidate_drawer_fraction": null,
  "strict_candidate_found": false,
  "strict_candidate_has_forbidden_contact": null,
  "strict_candidate_has_goc_v3_target_contact": false
}
```
