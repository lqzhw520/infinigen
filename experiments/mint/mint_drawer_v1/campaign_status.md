# MINT Drawer Campaign Dashboard

**Updated**: 2026-03-30T23:37:09+08:00
**Campaign**: `mint_drawer_v1`
**Phase**: `mint_robot_trajectory_claim_push`
**Gate**: `clean_c2_available_ready_for_d1`
**Claim**: `Infinigen-generated AnyGrasp-conditioned robot-arm drawer trajectories improve MINT success on held-out drawer variants in simulation relative to pretrained MINT.`
**Revision**: `robot_revision_v3_claim_push`

## Queue

- `u1_asset_geometry_audit`: completed
- `u2_joint_semantics_audit`: completed
- `u3_handle_region_audit`: completed
- `u4_export_consistency_audit`: completed
- `u5_seed_outlier_audit`: completed
- `a1_asset_scene_audit`: completed
- `a2_frame_transform_audit`: completed
- `b1_oracle_scripted_baseline`: completed
- `b2_anygrasp_scripted_baseline`: completed
- `c1_teacher_native_rollout_rebuild`: completed
- `c2_action_contract_repair`: completed
- `c3_single_rollout_replay_gate`: completed
- `d1_single_rollout_overfit`: failed
- `d2_single_seed_overfit`: pending
- `d3_train_seed_probe`: pending
- `e1_heldout_eval`: pending
- `write_claim_memo`: pending

## Runtime

- Controller ID: `d1_v57`
- Run ID: `d1_v57_main`
- Active Step: `d1_single_rollout_overfit`
- Active Lane: `learnability`
- Active Branch: `None`
- Worker PID: `None`
- Worker Kind: `manual_reconcile`
- Attempt: `2`
- Launched At: `2026-03-30T23:37:09+08:00`
- Heartbeat: `2026-03-30T23:37:09+08:00`
- Latest Artifact: `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/d1_single_rollout_overfit.json`
- Last Error: `u3_regime_b_required`

## Review

- Success Semantics Version: `v3_attached_open_contract`
- Dataset Fingerprint: `{'builder_version': 'd1_v6_gripper_binarize', 'variant_id': 'multi_episode_overfit', 'contract_mode': 'teacher_success_fallback', 'source_branch_id': 'branch4_densified_attach_window', 'rollout_multiplier': 5, 'balancing_mode': 'multi_episode_seed2', 'selected_seed_ids': [2], 'rollouts': [{'path': '/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/c2_replay_valid_rollouts/seed_002_episode_00.npz', 'size': 2407687, 'content_sha256': 'e0d58597833ae302d614dd432e8f991e9e729b66f33b36d5ed909b61cdef4181', 'seed': 2, 'episode_index': 0, 'success': True, 'ever_attached': True, 'max_drawer_fraction': 1.0, 'branch_id': 'branch4_densified_attach_window', 'contract_mode': None, 'teacher_mode': 'closed_loop_native', 'uses_planned_segment': False, 'uses_target_fraction': False, 'attach_step': 66}, {'path': '/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/c2_replay_valid_rollouts/seed_002_episode_01.npz', 'size': 2126951, 'content_sha256': '748501433c89c611544cb31987021d5945953845ce627ba39f01d9d60ffdac04', 'seed': 2, 'episode_index': 1, 'success': True, 'ever_attached': True, 'max_drawer_fraction': 1.0, 'branch_id': 'branch4_densified_attach_window', 'contract_mode': None, 'teacher_mode': 'closed_loop_native', 'uses_planned_segment': False, 'uses_target_fraction': False, 'attach_step': 66}, {'path': '/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/c2_replay_valid_rollouts/seed_002_episode_02.npz', 'size': 2324339, 'content_sha256': 'c95bc3fcc3d6fbdb5bb2e03d1e4c16dc07fb6f4bac280b47fe93cf21e089b391', 'seed': 2, 'episode_index': 2, 'success': True, 'ever_attached': True, 'max_drawer_fraction': 0.9467273798838806, 'branch_id': 'branch4_densified_attach_window', 'contract_mode': None, 'teacher_mode': 'closed_loop_native', 'uses_planned_segment': False, 'uses_target_fraction': False, 'attach_step': 62}, {'path': '/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/c2_replay_valid_rollouts/seed_002_episode_03.npz', 'size': 2773959, 'content_sha256': '76efc2bc4a61ca9f2a80bab1e5a8866568f61eced73592a8ae4213031d8f7031', 'seed': 2, 'episode_index': 3, 'success': True, 'ever_attached': True, 'max_drawer_fraction': 1.0, 'branch_id': 'branch4_densified_attach_window', 'contract_mode': None, 'teacher_mode': 'closed_loop_native', 'uses_planned_segment': False, 'uses_target_fraction': False, 'attach_step': 72}, {'path': '/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/c2_replay_valid_rollouts/seed_002_episode_04.npz', 'size': 2172482, 'content_sha256': '98b00db696e4de3fbadb23d4fe0e0132bf4c2c3f260b516a08361b9b0d8ba1fc', 'seed': 2, 'episode_index': 4, 'success': True, 'ever_attached': True, 'max_drawer_fraction': 1.0, 'branch_id': 'branch4_densified_attach_window', 'contract_mode': None, 'teacher_mode': 'closed_loop_native', 'uses_planned_segment': False, 'uses_target_fraction': False, 'attach_step': 63}], 'fingerprint': '20f1c5c64425d752517e82719857f7c9b96e3ef07b8d6440eed9f98e1d25ee39'}`
- Invalidated Results: `['d2_invalid_for_claim_due_to_single_rollout_seed', 'd3_invalid_for_claim_due_to_soft_gate', 'e1_diagnostic_only_after_gate_drift', {'step_id': 'd1_single_rollout_overfit', 'reason': 'contaminated_by_concurrent_control', 'details': {'since': '2026-03-26T18:26:00+08:00', 'note': 'Concurrent supervisor/D1/train processes detected; do not treat post-cutoff D1 artifacts as clean evidence.'}, 'recorded_at': '2026-03-26T19:17:59+08:00'}, {'step_id': 'd1_single_rollout_overfit', 'reason': 'contaminated_by_concurrent_control', 'details': {'since': '2026-03-26T19:19:02+08:00', 'note': 'Concurrent supervisor/D1/train processes detected; do not treat post-cutoff D1 artifacts as clean evidence.'}, 'recorded_at': '2026-03-26T19:20:27+08:00'}]`
- Env Contract Audit Passed: `True`
- Env Contract Failed Seeds: `[]`
- Verdict: `awaiting_execution`
- Decision: `run_experiments`
- Workflow Score: `7`/10
- Evidence Score: `2`/10
- Claim assessment: The strict-valid teacher pool now distinguishes diagnostic and mainline candidates; 3 D2-feasible rollout(s) remain eligible for the mainline D1->D2 path.

## Lanes

- Strict Replay Lane: `repairing`
- Strict Replay Best Branch: `branch4_densified_attach_window`
- Strict Replay Metrics: `{'selected_branch': 'branch4_densified_attach_window', 'strict_replay_passed': False, 'successful_seed_count': 0, 'successful_replays': 0, 'successful_seeds': []}`
- Learnability Lane: `ready_for_overfit`
- Learnability Contract Mode: `teacher_success_fallback`
- Learnability Source Branch: `branch4_densified_attach_window`
- Learnability Current Step: `d1_single_rollout_overfit`
- Strict-Valid Teacher Rollouts: `15`
- Strict-Valid Teacher Seeds: `5`
- D2-Feasible Seeds: `[2, 8, 10]`
- D1 Candidate Order: `['/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/c2_replay_valid_rollouts/seed_002_episode_01.npz', '/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/c2_replay_valid_rollouts/seed_008_episode_04.npz', '/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/c2_replay_valid_rollouts/seed_008_episode_01.npz']`
- D1 Promoted Candidate: `None`
- Upstream Audit Lane: `suspect`
- Upstream Root Cause: `Handle semantics are weakly exported or heuristically inferred, so upstream asset/export metadata remains a real suspect.`
- Upstream Blocking Step: `u3_handle_region_audit`
- Upstream Affected Seeds: `[]`
- Infinigen Fix Required: `False`
- Infinigen Fix Allowed: `False`
- Infinigen Patch Gate Reason: `await_d1_candidate_search`
- Weakness: Handle semantics are weakly exported or heuristically inferred, so upstream asset/export metadata remains a real suspect.
- Next: Advance the learnability lane into D1 overfit instead of waiting for strict replay to turn fully green.
- Next: Keep the strict replay lane running in parallel so the teacher/native contract becomes scientifically cleaner even if the learnability lane advances first.
- Next: Keep the upstream audit lane explicit: only modify `infinigen/**` if the audited evidence points to a specific export or geometry defect.
- Next: Upstream remains suspect, but do not patch `infinigen/**` yet; wait for D1 candidate search to finish under strict success.
- Next: Run or resume the next pending queue step: `d1_single_rollout_overfit`.

## Recent History

- 2026-03-29T12:14:29+08:00: step_reconciled | d1_single_rollout_overfit => failed
- 2026-03-29T12:59:36+08:00: step_reconciled | d1_single_rollout_overfit => failed
- 2026-03-30T13:30:56+08:00: step_reconciled | d1_single_rollout_overfit => failed
- 2026-03-30T13:34:32+08:00: step_reconciled | d1_single_rollout_overfit => failed
- 2026-03-30T13:39:52+08:00: step_reconciled | d1_single_rollout_overfit => failed
- 2026-03-30T14:17:05+08:00: step_reconciled | d1_single_rollout_overfit => failed
- 2026-03-30T14:26:56+08:00: step_reconciled | d1_single_rollout_overfit => failed
- 2026-03-30T14:36:05+08:00: step_reconciled | d1_single_rollout_overfit => failed
- 2026-03-30T16:28:42+08:00: step_reconciled | d1_single_rollout_overfit => failed
- 2026-03-30T20:02:33+08:00: step_reconciled | d1_single_rollout_overfit => failed
- 2026-03-30T21:41:54+08:00: step_reconciled | d1_single_rollout_overfit => failed
- 2026-03-30T23:37:09+08:00: step_reconciled | d1_single_rollout_overfit => failed
