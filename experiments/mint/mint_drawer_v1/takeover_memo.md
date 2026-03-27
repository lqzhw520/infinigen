# MINT AnyGrasp Robot-Trajectory Takeover Memo

**Updated**: 2026-03-23T20:15:37+08:00

## Locked non-repeat failures

- Do not reuse the old proxy campaign as the active claim; archive it first.
- Do not trust smoke-pass G4/G5/G6 artifacts as robot-trajectory evidence.
- Do not route AnyGrasp through the `mint` env; use `graspnet` after explicit staging.
- Do not use object-joint sweep or proxy deltas as MINT training actions.
- Do not infer Python 3.12 incompatibility before validating wrapper/API usage.
- Do not allow broken/truncated artifacts to count as completed steps.
- Do not use `force_attach` in G5-generated training data.
- Do not call the held-out claim before train-seed reproducibility turns positive.
- Do not let the PRD describe the old proxy phase as the active execution truth.

## Reusable prior work

- The proxy-control result remains a valid archived baseline.
- The depth/render split is still useful for AnyGrasp staging.
- Existing project-memory integration should be reused from day one.
- Current exported drawer assets behave like front-lip pull drawers, not cabinet handles.

## Active execution truth

- Current target is **simulation-only** held-out evaluation.
- This revision uses AnyGrasp detection only; tracking is out of scope.
- G4/G5 root-cause repair must prove that natural robot rollouts remain valid without `force_attach`.
- Held-out evaluation is deferred until train-seed reproducibility is positive.
- The active action/state contract is real robot EEF control:
  - action(7) = delta_xyz + delta_rxyz + gripper_command
  - state(8) = eef_pos_xyz + eef_quat_xyzw + gripper_open
