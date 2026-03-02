# Session Checkpoint

**Date**: 2026-02-26
**Branch**: feature/3d-assets
**Last Commit**: 96cd1016 just fix self-collision mailerbox-simple bug 555555 then just update md

## Completed

### Phase-1 Data Engine (MailerBox_Simple 3.1.3 + 3.1.4) — DONE
- **3.1.3 Domain-Randomized Materials**: `infinigen/assets/sim_objects/box_material_domain_randomization.py` — deterministic sampling across 4 material families (corrugated/kraft/glossy/plastic). Physics params (ρ/μ/e) coupled to URDF via `physics_density` override. Fixed: physics sampling now uses local RNG from material min/max ranges (not global `np.random`).
- **3.1.4 Auto-labeling Pipeline**: `scripts/export_mailerbox_simple_phase1_data_engine.py` — exports per-sample folders with RGB-D + segmentation + instance + keypoints + URDF GT + camera intrinsics + metadata + `segmentation_label_map.json` (readable mapping with stable colors and `present_in_frame`).
- **Prismatic FK fix**: `compute_link_poses_world()` now uses `Translation(axis * q)` for prismatic joints (was incorrectly using rotation).
- **ralph-loop**: 3/5 iterations used. All 3 user stories pass (`docs/user-stories/mailerbox-simple-phase1.json`).
- **Skill upgrade**: `infinigen-scientific-dev` updated with DR materials + auto-labeling protocols.

### Phase-1.1 Box Topology Extension — DONE
- `modular_box_factory.py`: Added `DrawerBoxFactory` and `SlipLidBoxFactory` (thin-shell panel geometry). `TuckEndBoxFactory` verified.
- URDF export + PyBullet load verified for MAILER / TUCK_END / DRAWER / SLIP_LID.

### P0 URDF Inertia Fix — DONE
- `urdf_exporter.py` correctly merges sub-mesh inertias via Parallel Axis Theorem (`thin_shell_inertia.combine_multiple_inertias`). One `<inertial>` per physical link.
- 11 MailerBox seeds verified (structure + PyBullet `URDF_USE_INERTIA_FROM_FILE`).

### Phase-2 Perception Skeleton — DONE (skeleton only, no training)
- `infinigen/perception/topo_box_net/` — schema, URDF parser, dataset loader, model heads, losses.
- `docs/Topo_Box_Net_Architecture.md` — architecture document.
- Kinematic consistency loss: NumPy + PyTorch implementations + 6 unit tests passing.

### Phase-3 Scaffolding — DONE (design + stubs, no real experiments)
- Sim2Real ablation design: `docs/Sim2Real_Ablation_Experiment_Design.md` + `scripts/run_sim2real_ablation_matrix.py`.
- Motion planner API: `infinigen/planning/` — baseline joint-space planner (PyBullet collision checking) + VAMP adapter stub.

## Verified Artifacts
| Artifact | Verification Command | Last Result |
|----------|---------------------|-------------|
| `sim_exports/urdf/mailerbox_simple/` (11 seeds) | `python scripts/verify_urdf_inertia_fix.py --urdf-dir sim_exports/urdf/mailerbox_simple --quiet` | 11/11 PASS (2026-02-03) |
| Phase-1 sample folder (mailer) | `python scripts/verify_phase1_sample_folder.py sim_exports/data_engine/_smoke_test15_labelmap_complete_mailer/dataset/train/000001` | PASS (2026-02-03) |
| Kinematic consistency unit tests | `python -m pytest -q tests/perception/test_kinematic_consistency.py` | 6 passed (2026-02-03) |
| Planner demo (drawer) | `python scripts/planning/run_motion_planner_demo.py --sample-dir sim_exports/data_engine/_smoke_test16_prismatic_fk_drawer/dataset/train/000001 --out /tmp/planner_test.json` | success=true (2026-02-03) |
| PyBullet URDF load (seed 42) | `python -c "import pybullet as p; pid=p.connect(p.DIRECT); p.setAdditionalSearchPath('sim_exports/urdf/mailerbox_simple/42'); p.loadURDF('sim_exports/urdf/mailerbox_simple/42/mailerbox_simple.urdf', useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE); p.disconnect(pid); print('OK')"` | OK (2026-02-26, re-verified) |

## Next
- [P0] Commit all uncommitted work on `feature/3d-assets` (5 modified + ~25 untracked files — risk of loss)
- [P1] Generate production-scale dataset (10k+ samples) for actual training
- [P1] Extend Phase-1 batch exporter to TuckEndBox / DrawerBox / SlipLidBox
- [P2] Implement Topo-Box-Net training script (Phase-2 skeleton exists but no train loop)
- [P2] Upgrade planner from baseline linear to real VAMP (or other collision-aware planner)

## Risks
- **25+ files uncommitted** — single biggest risk of work loss
- `box_material_domain_randomization.py` imports chain pulls `bpy`; cannot unit-test outside Blender
- MailerBox naive open trajectory triggers self-collision (expected; needs smarter planner)
- Phase-2/3 are scaffolds only — no trained models or real Sim2Real experiments yet

## Lessons
- Shell env in some containers lacks `git` in PATH; use `conda run -n infinigen git ...` as workaround
- PyBullet requires `p.setAdditionalSearchPath(urdf_parent_dir)` to resolve relative mesh paths
- `BaseMaterial.sample_parameters()` uses global `np.random` — breaks determinism; always sample from min/max with local RNG
- Prismatic joints need `Translation(axis * q)`, NOT rotation — was a silent FK bug until DrawerBox testing caught it
- When multiple `<inertial>` tags appeared to exist, it was actually opening+closing tags being counted; actual structure was correct
- `tests/conftest.py` must conditionally import `bpy` or pure-Python tests fail outside Blender

## Key Files
- `infinigen/assets/sim_objects/modular_box_factory.py` — all box type factories
- `infinigen/assets/sim_objects/box_material_domain_randomization.py` — DR material sampler
- `scripts/export_mailerbox_simple_phase1_data_engine.py` — Phase-1 data exporter (generic for all box types)
- `infinigen/core/sim/exporters/urdf_exporter.py` — URDF generation with inertia merging
- `infinigen/core/sim/physics/thin_shell_inertia.py` — Parallel Axis Theorem implementation
- `sim_exports/urdf/mailerbox_simple/` — downstream URDF artifacts (DO NOT modify without re-verifying)
- `infinigen/perception/topo_box_net/` — Phase-2 perception skeleton
- `infinigen/planning/` — Phase-3 planner API + baseline

## Context Pointers
- `docs/Retrospective_Roadmap_Completion_2026-02-03.md` — detailed retrospective of all 8 roadmap to-dos
- `docs/Physics_Aligned_Data_Engine_Research_Plan.md` — overall research roadmap (Phase 1→3)
- `docs/Modular_Box_Factory_Architecture.md` — box factory extension guide
- `task_plan.md` / `progress.md` / `findings.md` — planning-with-files state (MailerBox Phase 1.3/1.4)
- `scripts/ralph/log.md` — ralph-loop iteration log (3/5 used)
- `docs/user-stories/mailerbox-simple-phase1.json` — acceptance criteria (all pass)
