# Infinigen-AnyBox Project Status
<!-- Auto-maintained by infinigen-project-memory skill. Last updated: 2026-03-03 17:59 -->

## Architecture

```
Phase 1: Data Engine          Phase 2: Perception           Phase 3: Planning
┌─────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
│ 1.1 Box Topology    │      │ 2.1 Topo-Box-Net     │      │ 3.1 Sim2Real Ablation│
│ 1.2 Inertia Fix     │─────>│ 2.2 Kin-Consist Loss │─────>│ 3.2 VAMP Integration │
│ 1.3 DR Materials    │      │                      │      │                      │
│ 1.4 Annotation Pipe │      └──────────────────────┘      └──────────────────────┘
└─────────────────────┘
```

## Current State

- **Branch**: feature/3d-assets
- **Last Commit**: 42ce8dff Update planning files with iteration protocol
- **Phase**: 1K batch dataset v2: 1000/1000 PASS (250 per box type) with collision + material
- **Active Work**: [P0] Verify 1K dataset usable by downstream Planner (collision check, Topo-Box-Net load)

## Completed Milestones

| # | Date | Milestone | Key Output | Commit |
|---|------|-----------|------------|--------|
| 1 | 2026-01-10 | Phase-0/1 initial implementation: MailerBox geometry + URDF export + modular factory | sim_exports/urdf/mailerbox_simple/ | 8f22ba48 |
| 2 | 2026-01-13 | TuckEndBox first successful generation (8-DOF) | sim_exports/urdf/tuckendbox/ | 550c721c |
| 3 | 2026-01-30 | MailerBox self-collision fix + PyBullet verification | sim_exports/urdf/mailerbox_simple/ (11 seeds verified) | 96cd1016 |
| 4 | 2026-03-02 | Phase-1 Data Engine complete + Phase-2/3 scaffolding for all 4 box types | scripts/export_mailerbox_simple_phase1_data_engine.py, infinigen/assets/sim_objects/box_material_domain_randomization.py, infinigen/assets/sim_objects/modular_box_factory.py, infinigen/perception/topo_box_net/, scripts/run_sim2real_ablation_matrix.py | 56bc0e43 |
| 5 | 2026-03-02 | Phase-1 URDF fix: collision meshes + material color tags + 1K dataset (v2) | sim_exports/data_engine/phase1_1k_mailer/, sim_exports/data_engine/phase1_1k_drawer/, sim_exports/data_engine/phase1_1k_sliplid/, sim_exports/data_engine/phase1_1k_tuckend/, sim_exports/urdf/mailerbox_simple/ (11 seeds, untouched), sim_exports/data_engine/material_diversity_proof.png | 35b99593 |

## Bug Fixes & Lessons

| # | Date | Bug | Root Cause | Fix | Commit |
|---|------|-----|-----------|-----|--------|
| 1 | 2026-01-30 | Multiple <inertial> tags per link in URDF | urdf_exporter wrote one <inertial> per Blender object instead of merging per link | Implemented Parallel Axis Theorem merging in urdf_exporter.py | 96cd1016 |
| 2 | 2026-01-30 | Self-collision not properly configured | URDF lacked self-collision flags; PyBullet soft joint limits allow penetration under external force | Added URDF_USE_SELF_COLLISION flag + verified joint limits | 96cd1016 |
| 3 | 2026-03-02 | URDF missing collision meshes -- joints spin infinitely in online viewers | visual_only=True in urdf_exporter.export() call | Changed to visual_only=False to include <collision> tags and *_col0.obj assets | 35b99593 |
| 4 | 2026-03-02 | No material/color in urdf_gt.urdf -- shows white in URDF viewers | urdf_exporter had no <material> tag injection | Post-processing: _inject_material_color_into_urdf() adds <material><color rgba> from DR config | 35b99593 |
| 5 | 2026-03-02 | imageio SystemError: tile cannot extend outside image at 1024x768 | imageio.imwrite bug with high-res segmentation PNGs | Replaced with PIL.Image.fromarray().save() | 35b99593 |
| 6 | 2026-03-02 | Parallel batch processes overwrite samples (counter collision) | Multiple box types sharing same out_root with overlapping sample_counter | Separate output directories per box type (phase1_1k_mailer, etc.) | 35b99593 |

## Verified Artifacts

| Artifact | Path | Verification Command | Last Result |
|----------|------|---------------------|-------------|
| 1k_dataset_mailer | `sim_exports/data_engine/phase1_1k_mailer/` | (run verification) | - |
| 1k_dataset_drawer | `sim_exports/data_engine/phase1_1k_drawer/` | (run verification) | - |
| 1k_dataset_sliplid | `sim_exports/data_engine/phase1_1k_sliplid/` | (run verification) | - |
| 1k_dataset_tuckend | `sim_exports/data_engine/phase1_1k_tuckend/` | (run verification) | - |
| downstream_urdfs | `sim_exports/urdf/mailerbox_simple/ (11 seeds, untouched)` | (run verification) | - |
| material_proof | `sim_exports/data_engine/material_diversity_proof.png` | (run verification) | - |

## Next Steps

- [P0] Verify 1K dataset usable by downstream Planner (collision check, Topo-Box-Net load)
- [P1] Scale to 10K+ samples: more seeds + random joint states + more viewpoints
- [P1] Phase 2.1: Topo-Box-Net training script + baseline training
- [P2] Joint state diversity: random sampling instead of 2 fixed values

## Key Files

- `sim_exports/data_engine/phase1_1k_mailer/` -- 1k_dataset_mailer
- `sim_exports/data_engine/phase1_1k_drawer/` -- 1k_dataset_drawer
- `sim_exports/data_engine/phase1_1k_sliplid/` -- 1k_dataset_sliplid
- `sim_exports/data_engine/phase1_1k_tuckend/` -- 1k_dataset_tuckend
- `sim_exports/urdf/mailerbox_simple/ (11 seeds, untouched)` -- downstream_urdfs
- `sim_exports/data_engine/material_diversity_proof.png` -- material_proof


## Lessons Learned (cumulative)

- Blender coordinate system differs from URDF -- transform required
- TuckEndBox has 8 DOF -- joint_states parameter must exactly match DOF count
- PyBullet joint limits are 'soft' -- external forces can exceed them
- Each URDF link must have exactly one <inertial> tag with merged properties
- Always verify with PyBullet URDF_USE_INERTIA_FROM_FILE | URDF_USE_SELF_COLLISION
- Use local RNG (not global np.random) for deterministic DR material sampling
- planning-with-files skill keeps task-level context across iterations
- visual_only=True causes URDF to lack collision meshes -- downstream planner cannot use
- URDF needs <material><color rgba> tags for color display in online viewers
- imageio crashes on high-res segmentation PNGs -- use PIL instead
- Never share out_root between parallel batch processes -- sample counter collides
- Always verify with both online URDF viewer AND PyBullet
