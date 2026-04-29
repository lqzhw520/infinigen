# V11 Hard Goal Runtime Truth Contract

**Contract ID**: `V11_HARD_GOAL_RUNTIME_TRUTH_CONTRACT`
**Version**: v1
**Created**: 2026-04-29

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
| V11 success claimable from proxy runtime | NO |
| Stage B / MINT training blocked until V11 gates pass | YES |

---

## 8 Gates

| Gate | Name | Status |
|------|------|--------|
| V11-G1 | FULL_ROBOT_IN_SCENE_RUNTIME | PASS |
| V11-G2 | RIGHT_SIDE_REACHABILITY_LAYOUT | UNKNOWN |
| V11-G3 | GEOMETRY_OWNERSHIP_CONTRACT | PASS |
| V11-G4 | LEGAL_GRIPPER_PAD_TO_HANDLE_CONTACT | UNKNOWN |
| V11-G5 | FORBIDDEN_PENETRATION_REJECTION | UNKNOWN |
| V11-G6 | UNTRIMMED_TEACHER_ROLLOUT_TRUTH | FAIL |
| V11-G7 | DRAWER_OPENING_THRESHOLD | FAIL |
| V11-G8 | LOCAL_STRICT_REPLAY_RENDER | UNKNOWN |

---

## Success Levels

- **Visual artifact pass**: drawer_fraction_min = 0.25; requires G8
- **Teacher success pass**: drawer_fraction_min = 0.80; requires G6 + G7

---

## Geometry Contract Evidence

- **Remote commit**: `c20266062a159e448ab60f776256c00070d72165`
- **Status**: `GEOMETRY_CONTRACT_READY_FOR_PHASE1H`
- **Artifact root**: `experiments/mint/mint_drawer_v1/artifacts/phase1h_geometry_contract`

---

## Reference Image Role

The uploaded reference image establishes the **visual artifact target**, not the final success frame. Visual pass (G8) requires artifacts comparable to reference image.
