# MINT Drawer Campaign Dashboard

**Updated**: 2026-03-27T14:00:00+08:00
**Campaign**: `mint_drawer_v1`
**Phase**: `mint_robot_trajectory_claim_push`
**Gate**: `clean_c2_available_ready_for_d1`
**Claim**: `Infinigen-generated AnyGrasp-conditioned robot-arm drawer trajectories improve MINT success on held-out drawer variants in simulation relative to pretrained MINT.`
**Revision**: `robot_revision_v3_claim_push`
**Resolution Plan**: `RESOLUTION_PLAN.md` v2 — corrected after B1/U5/C2 full evidence review
**History Archived**: `history/ARCHIVE_INDEX.md`
**Verdict**: `rebuild_in_progress` — clean rebuild under resolution_plan_v2

## Key Finding (2026-03-27 audit)

The previous `strong_rollout_audit.json` pointed at **archived** `diagnostic_runs/2026-03-26_gate_drift_reset/` which has been cleaned out. The clean `c2_replay_valid_rollouts/` directory contains **25 rollouts across seeds 2, 7, 8, 9, 10** with the following JSON-criteria results:

| Seed | JSON-passing eps | Coherent? | Best ep pre_attach | Best ep persist |
|------|-----------------|-----------|-------------------|-----------------|
| 2    | ep01, ep02, ep04 | **YES** (3 rollouts) | 0.0009 (ep04) | 28 |
| 7    | ep00 only        | no (1 rollout)      | 0.1177         | 20 |
| 8    | ep00, ep01, ep02, ep03 | **YES** (4 rollouts) | 0.0 (ep00,ep03) | 48 (ep03) |
| 9    | none             | no                  | —              | —  |
| 10   | ep03 only        | no (1 rollout)      | 0.0            | 20 |

**Provisional `strong_coherent_seeds`: [2, 8]** — pending NPZ-based grasp/pull verification

**Previous blocker resolved**: The archived `seed_010_ep04` had `pre_attach=0.1215` (rejected). In the clean C2, `seed_010_ep03` has `pre_attach=0.0` — but only 1 rollout passes, so seed_010 is not yet coherent. Seeds 2 and 8 are the primary D1 targets.

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
- `c2_action_contract_repair`: **completed** *(reclassified: BrokenPipeError was in final write only; 25 clean rollouts present in `c2_replay_valid_rollouts/`)*
- `c3_single_rollout_replay_gate`: completed
- `d1_single_rollout_overfit`: **pending** *(reset — previous run used archived `diagnostic_runs/` path; must re-run against clean C2 with corrected audit)*
- `d2_single_seed_overfit`: **pending** *(reset — previous `data_quality_floor_not_met` was against stale source)*
- `d3_train_seed_probe`: pending
- `e1_heldout_eval`: pending
- `write_claim_memo`: pending

## Runtime

- Controller ID: `None` *(stale `manual_lineage_bridge` lease released)*
- Run ID: `None`
- Active Step: `d1_single_rollout_overfit` (pending — not yet launched)
- Worker PID: `None`
- GPU: free (0 MiB / 81154 MiB)
- Screen sessions: none
- Last Error: *(cleared)*

## Stale Files Cleared This Session

| File | Action | Reason |
|------|--------|--------|
| `artifacts/d1_candidate_progress.json` | **deleted** | pointed at archived `diagnostic_runs/` path |
| `artifacts/strong_rollout_audit.json` | **rebuilt** | corrected `source_dir` to `c2_replay_valid_rollouts/` |
| `runtime/controller_lease.json` | **released** | stale `manual_lineage_bridge` lease |
| `state.json` | **patched** | c2→completed, d1/d2→pending, active_job cleared |

## Pre-D1 Checklist (must ALL pass before launching D1)

- [ ] Run `run_strong_rollout_audit.py --source_dir artifacts/c2_replay_valid_rollouts/` to verify NPZ grasp/pull steps and confirm `strong_coherent_seeds=[2,8]`
- [ ] Verify `strong_coherent_seeds` non-empty in output (per GPT gate A, accepted in RESOLUTION_PLAN v2)
- [ ] Verify no competing python processes: `ps aux | grep python | grep -v grep`
- [ ] Verify GPU free: `nvidia-smi | grep MiB`
- [ ] Verify `diagnostic_runs/` is empty: `ls artifacts/diagnostic_runs/ | wc -l` → expect 0
- [ ] Set fresh `MINT_CONTROLLER_ID` and `MINT_RUN_ID` before launching screen job

## Review

- Evidence Score: `6`/10 *(upgraded from 4: clean C2 data confirmed, 2 provisional coherent seeds)*
- Workflow Score: `7`/10
- Verdict: `rebuild_in_progress`
- Decision: `run_strong_rollout_audit_then_d1`
- Claim assessment: Provisional — seeds 2 and 8 have strong coherent rollout sets. Seeds 2 and 10 are AnyGrasp-only (oracle=0/6), making them the scientifically strongest evidence for the claim. D1 rank-1 target: `seed_002_ep04` (pre=0.0009, persist=28, post=1.0).

## Invalidated Results (do not use for claim)

- `d1_single_rollout_overfit` (prior): contaminated by concurrent control + wrong source path
- `d2_single_seed_overfit` (prior): `data_quality_floor_not_met` against stale archived data
- `strong_rollout_audit.json` (prior, until 2026-03-27T14:00): pointed at `diagnostic_runs/` not `c2_replay_valid_rollouts/`

## Recent History

- 2026-03-26T23:47:31+08:00: `d2_stop_loss` — `strong_rollout_count=1, strong_coherent_seeds=[]` (audit was against wrong source_dir)
- 2026-03-27T02:38:40+08:00: `c2` last recorded as failed (BrokenPipeError in final write; rollout data already written)
- 2026-03-27T14:00:00+08:00: **resolution_plan_v2 audit** — root cause identified: `strong_rollout_audit.json` pointed at cleaned-out `diagnostic_runs/`. Rebuilt against correct `c2_replay_valid_rollouts/`. `d1_candidate_progress.json` deleted. `state.json` reset. Stale lease released. Provisional `strong_coherent_seeds=[2,8]`.
