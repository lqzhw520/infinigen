# MINT Step Review Guide

This guide is the review index for the active campaign:

- Campaign: `mint_drawer_v1`
- Active claim: `Infinigen-generated AnyGrasp-conditioned robot-arm drawer trajectories improve MINT success on held-out drawer variants in simulation relative to pretrained MINT.`
- Active phase: `mint_robot_trajectory_claim_push`
- Current success semantics: `v3_attached_open_contract`

## How To Review This Campaign

For each step, review in this order:

1. Step definition in `manifest.yaml` and `acceptance_criteria.json`
2. Step implementation in `scripts/mint/run_<step>.py` or the step helper noted below
3. Step artifact in `artifacts/<step>.json`
4. Step runtime truth in `state.json`, `review.json`, `summary.json`, `campaign_status.md`, and `runtime/watch_status.json`
5. Step-specific logs:
   - queue launcher logs in `runtime/<step>/launcher.log`
   - per-variant D1/D2 logs in `artifacts/d1_*.log` or `artifacts/d2_*.log`
6. Supporting artifacts listed per step

## Pass Definition Precedence

To avoid `manifest / acceptance_criteria / guide` conflicts, use this rule:

1. `manifest.yaml` is the source of truth for:
   - queue order
   - train vs. held-out object split
   - evaluator baselines
2. `acceptance_criteria.json` is the source of truth for queue-step pass definitions.
3. `strict_success.py` plus `env_contract_audit.json` are global hard gates for `D1/D2/D3/E1`, even though they are not queue rows.
4. `step_review_guide.md` is the reviewer checklist and enhanced evidence guide. It may be stricter or more explicit than `acceptance_criteria.json`, but it must not contradict items 1-3.

In short:

- Queue/split/baselines come from `manifest.yaml`
- Machine pass/fail comes from `acceptance_criteria.json`
- `D1-E1` also require the global strict/env contracts
- The guide explains how to review the evidence without redefining the pass contract

## Cross-Check Checklist

When auditing the campaign, explicitly check:

- `manifest.queue` matches the queue in `state.json` and `campaign_status.md`, including `write_claim_memo`
- `manifest.object_split` keeps train seeds `1-10` and held-out seeds `11-15`, and this matches `E1` and the criteria text
- `manifest.baselines` matches the actual evaluator comparison set: `random`, `pretrained_mint`, `finetuned_mint`
- every `acceptance_criteria.json` pass string has corresponding evidence in the step artifact and/or `review.json`
- `U1-U5` are reviewed via both `acceptance_criteria.json` and their dedicated `artifacts/u*.json`
- `strict_success.py` version and `env_contract_audit.json` status are archived together with any `D1/D2/D3/E1` conclusion
- for `D1-D3`, note explicitly that:
  - `acceptance_criteria.json` defines the minimum pass contract
  - this guide defines the enhanced reviewer checks and expected evidence

## Source Of Truth Files

- `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/manifest.yaml`
- `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/acceptance_criteria.json`
- `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/state.json`
- `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/review.json`
- `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/summary.json`
- `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/campaign_status.md`
- `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/runtime/watch_status.json`

## Global Contract Checks

These are not queue steps, but they now gate all `D1/D2/D3/E1` conclusions:

### Strict Success Contract

- Implementation:
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/strict_success.py`
- Current version:
  - `v3_attached_open_contract`
- Required semantics:
  - `drawer_open >= 0.90`
  - `ever_attached = true`
  - `attach_persistence >= 3`
  - `post_attach_drawer_delta >= 0.35`
  - `attach_before_major_open = true`

### Environment Contract Audit

- Purpose: verify the simulator no longer allows unattached drawer opening to masquerade as success.
- Implementation:
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/env_contract_audit.py`
- Artifact:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/env_contract_audit.json`
- What to check:
  - `passed = true`
  - `no_attach_probe.unattached_drawer_delta_max <= 0.02`
  - `attach_probe.strict_success = true`
  - `attach_before_major_open = true`

## Step Index

### U1 `u1_asset_geometry_audit`

- Purpose: audit drawer/cabinet geometry consistency across sampled seeds.
- Implementation:
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_u1_asset_geometry_audit.py`
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/upstream_audit_helpers.py`
- Artifact:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/u1_asset_geometry_audit.json`
- Supporting:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/urdf_asset_audit.json`

### U2 `u2_joint_semantics_audit`

- Purpose: verify URDF joint axis, limits, and open-direction semantics.
- Implementation:
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_u2_joint_semantics_audit.py`
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/upstream_audit_helpers.py`
- Artifact:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/u2_joint_semantics_audit.json`

### U3 `u3_handle_region_audit`

- Purpose: verify exported handle semantics and heuristic handle center quality.
- Implementation:
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_u3_handle_region_audit.py`
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/upstream_audit_helpers.py`
- Artifact:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/u3_handle_region_audit.json`
- Supporting:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/handle_region_audit.json`
- Current significance:
  - strongest upstream suspect, but only becomes patch-authorizing if `D1 top-3` all fail under the strict attach-centered contract

### U4 `u4_export_consistency_audit`

- Purpose: verify `infinigen -> simulator` export consistency.
- Implementation:
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_u4_export_consistency_audit.py`
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/upstream_audit_helpers.py`
- Artifact:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/u4_export_consistency_audit.json`

### U5 `u5_seed_outlier_audit`

- Purpose: distinguish systemic issues from a few bad seeds.
- Implementation:
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_u5_seed_outlier_audit.py`
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/upstream_audit_helpers.py`
- Artifact:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/u5_seed_outlier_audit.json`

### A1 `a1_asset_scene_audit`

- Purpose: confirm robot + drawer scene loads and scene-level audits are sane.
- Implementation:
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_a1_asset_scene_audit.py`
- Artifact:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/urdf_asset_audit.json`

### A2 `a2_frame_transform_audit`

- Purpose: confirm camera, depth, robot frame, and handle-region frame alignment.
- Implementation:
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_a2_frame_transform_audit.py`
- Artifact:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/scene_frame_audit.json`

### B1 `b1_oracle_scripted_baseline`

- Purpose: prove the task is physically solvable in sim with an oracle handle grasp.
- Implementation:
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_b1_oracle_scripted_baseline.py`
- Artifact:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/b1_oracle_scripted_baseline.json`
- Supporting:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/g4_oracle_vs_anygrasp_rollouts.json`

### B2 `b2_anygrasp_scripted_baseline`

- Purpose: prove AnyGrasp is not zero-signal on this task.
- Implementation:
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_b2_anygrasp_scripted_baseline.py`
- Artifact:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/b2_anygrasp_scripted_baseline.json`
- Supporting:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/g3_grasp_audit.json`
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/g3_grasp_plan.json`

### C1 `c1_teacher_native_rollout_rebuild`

- Purpose: rebuild teacher rollouts through native robot control.
- Implementation:
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_c1_teacher_native_rollout_rebuild.py`
- Artifact:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/c1_teacher_native_rollout_rebuild.json`
- Supporting:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/g4_robot_trajectory.json`

### C2 `c2_action_contract_repair`

- Purpose: repair replay-faithful action contract or produce teacher-success fallback.
- Implementation:
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_c2_action_contract_repair.py`
- Artifact:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/c2_action_contract_repair.json`
- Supporting:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/active_action_contract.json`
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/c_action_contract_branch_progress.json`
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/g5_contract_audit.json`

### C3 `c3_single_rollout_replay_gate`

- Purpose: select the single strict-valid rollout that becomes the `D1` candidate source.
- Implementation:
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_c3_single_rollout_replay_gate.py`
- Artifact:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/c3_single_rollout_replay_gate.json`
- Supporting:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/c3_single_rollout/`

### D1 `d1_single_rollout_overfit`

- Purpose: search top-3 mainline, strict-valid, learnability-ranked teacher rollouts and find the first D2-feasible candidate/variant that shows strict attach-open learning.
- Implementation:
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_d1_single_rollout_overfit.py`
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/strict_teacher_dataset.py`
- Primary artifacts:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/d1_candidate_progress.json`
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/d1_candidate_search.json`
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/d1_single_rollout_overfit.json`
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/d1_single_rollout_strict_eval.json`
- Supporting:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/strict_teacher_dataset_audit.json`
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/d1_seed_*.log`
- Important review note:
  - the per-variant training/eval logs are under `artifacts/d1_seed_*.log`, not only `runtime/d1_single_rollout_overfit/launcher.log`
  - `acceptance_criteria.json` is the minimum pass contract; this guide is the enhanced reviewer checklist
- What to check:
  - the active `candidate_mode` is `mainline`, not `diagnostic`
  - current `candidate_rank / variant_id`
  - whether `base`, `extended`, `phase_balanced`, or `attach_curriculum` is active
  - whether a `promoted_candidate` exists
  - whether the promoted candidate comes from a `D2-feasible` seed
  - whether the strict eval records show `ever_attached = true` before major opening

### D2 `d2_single_seed_overfit`

- Purpose: verify single-seed reproducibility using the coherent subset from the promoted D1 seed.
- Implementation:
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_d2_single_seed_overfit.py`
- Artifact:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/d2_single_seed_overfit.json`
- Expected supporting logs:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/d2_*.log`
- What to check:
  - `episodes_per_seed = 5`
  - source seed equals the promoted **mainline** D1 seed
  - if `source_rollout_count < 2`, D2 must fail-fast without training
  - strict success, grasp success rate, and pull distance all improve over pretrained
  - `acceptance_criteria.json` is the minimum pass contract; this guide is the enhanced reviewer checklist

### D3 `d3_train_seed_probe`

- Purpose: verify train-side reproducibility before any held-out claim.
- Implementation:
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_d3_train_seed_probe.py`
- Artifact:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/d3_train_seed_probe.json`
- What to check:
  - D2 is hard-passed before D3 starts
  - covered strict-valid seeds probe
  - full train split probe
  - dataset floor is met: `>=4 seeds / >=8 rollouts / >=300 frames`
  - improvement is not dominated by a single seed
  - `acceptance_criteria.json` is the minimum pass contract; this guide is the enhanced reviewer checklist

### E1 `e1_heldout_eval`

- Purpose: evaluate the actual claim on held-out seeds only after train reproducibility passes.
- Implementation:
  - `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_e1_heldout_eval.py`
- Artifact:
  - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/e1_heldout_eval.json`
- What to check:
  - held-out seeds are fixed and leakage-free
  - `episodes_per_seed = 3`
  - D3 hard-passed before E1 starts
  - same evaluator settings across `random`, `pretrained_mint`, `finetuned_mint`
  - `finetuned_mint.success_rate > pretrained_mint.success_rate`

## Review Logic Between D1 And E1

- `D1` pass means:
  - at least one strict-valid single rollout is learnable
- `D2` pass means:
  - that skill generalizes within the source seed
- `D3` pass means:
  - that skill generalizes across train-side strict-valid seeds
- `E1` pass means:
  - the main claim is supported on held-out seeds

So yes, there is a real `generalization gap` between `D1` and `E1`, but it is only the active blocker after `D1` itself passes.
