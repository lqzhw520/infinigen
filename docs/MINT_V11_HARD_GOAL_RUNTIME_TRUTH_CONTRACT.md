# V11 Hard Goal Runtime Truth Contract

**Contract ID**: `V11_HARD_GOAL_RUNTIME_TRUTH_CONTRACT`
**Version**: v2
**Created**: 2026-04-29
**Repaired**: 2026-04-29

---

## Hard Goal Statement

Remote A800 Infinigen/MuJoCo pipeline must generate an **untrimmed, truth-preserving, full robot-in-scene Franka/Panda drawer-opening teacher rollout**.

The rollout must contain:
- Visible simulated robot + gripper + Infinigen drawer interaction
- Legal gripper pad/fingertip (link5/6/7 collision geoms) to drawer-handle contact
- No forbidden robot/drawer/cabinet penetration
- Drawer opening progress
- A replayable truth bundle

Local macOS strict replay/render must load the remote bundle **without changing truth** and produce inspectable scene/reference/wrist visual artifacts comparable to the uploaded reference image.

---

## Important Clarifications

| Claim | Value |
|-------|-------|
| Simulation-only | YES — no real robot data required |
| Real robot data required | NO |
| Proxy drawer-only runtime claim-bearing | NO — G4/RCA/VR proxy-runtime artifacts are historical diagnostic surfaces only |
| V11 success claimable from proxy runtime | NO — proxy runtime cannot satisfy gates G4-G8 |
| Stage B / MINT training blocked until V11 gates pass | YES |

---

## Success Levels

**All 8 gates are required for both visual_artifact_pass and teacher_success_pass.**

### Visual Artifact Pass

- drawer_fraction_min = 0.25
- **Requires all 8 gates (G1–G8).** Local render is the **final step only** — it loads the strict teacher rollout bundle (which already satisfies G1–G7) and renders it.
- A visual artifact produced from a non-gate-compliant rollout is **NOT a V11 visual artifact pass**.
- requires_local_render: true
- requires_all_gates: true

### Teacher Success Pass

- drawer_fraction_min = 0.80
- **Requires all 8 gates (G1–G8).**
- requires_untrimmed_rollout: true
- requires_all_gates: true

---

## 8 Gates

| Gate | Name | Required for Visual | Required for Teacher |
|------|------|---------------------|---------------------|
| V11-G1 | FULL_ROBOT_IN_SCENE_RUNTIME | YES | YES |
| V11-G2 | RIGHT_SIDE_REACHABILITY_LAYOUT | YES | YES |
| V11-G3 | GEOMETRY_OWNERSHIP_CONTRACT | YES | YES |
| V11-G4 | LEGAL_GRIPPER_PAD_TO_HANDLE_CONTACT | YES | YES |
| V11-G5 | FORBIDDEN_PENETRATION_REJECTION | YES | YES |
| V11-G6 | UNTRIMMED_TEACHER_ROLLOUT_TRUTH | YES | YES |
| V11-G7 | DRAWER_OPENING_THRESHOLD | YES | YES |
| V11-G8 | LOCAL_STRICT_REPLAY_RENDER | YES | YES |

---

## Geometry Contract Evidence

- **Remote commit**: `ec2054b81fbe45d91b93b23a97adc2aaf8b95a3c`
- **Status**: `GEOMETRY_CONTRACT_READY_FOR_PHASE1H`
- **Artifact root**: `experiments/mint/mint_drawer_v1/artifacts/phase1h_geometry_contract`
- **Authority**: `geometry_ownership_contract.json` v2 at `ec2054b8`
- **Caveat**: `classification_audit.json` is a historical reconciliation audit containing pre-patch observations plus reconciliation note. Current machine-readable Geometry Ownership Contract authority is `geometry_ownership_contract.json` v2 at `ec2054b8`.

---

## Gate Status Summary

| Gate | Status |
|------|--------|
| V11-G1 FULL_ROBOT_IN_SCENE | PASS |
| V11-G2 RIGHT_SIDE_REACHABILITY | UNKNOWN |
| V11-G3 GEOMETRY_OWNERSHIP_CONTRACT | PASS |
| V11-G4 LEGAL_GRIPPER_PAD_TO_HANDLE | UNKNOWN |
| V11-G5 FORBIDDEN_PENETRATION_REJECTION | UNKNOWN |
| V11-G6 UNTRIMMED_ROLLOUT_TRUTH | FAIL |
| V11-G7 DRAWER_OPENING_THRESHOLD | FAIL |
| V11-G8 LOCAL_STRICT_REPLAY_RENDER | UNKNOWN |

---

## Reference Image Role

The uploaded reference image establishes the **visual artifact target**, not the final success frame. Visual pass (G8) requires artifacts comparable to reference image in scene composition, camera framing, and robot/drawer visibility. The bundle loaded by G8 must already satisfy G1–G7.
