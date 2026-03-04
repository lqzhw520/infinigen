# Infinigen-AnyBox Project Status
<!-- Auto-maintained by infinigen-project-memory skill. Last updated: 2026-03-04 -->

## Architecture

```
Phase 1: Data Engine          Phase 2: Perception           Phase 3: Planning
┌─────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
│ 1.1 Box Topology ✅ │      │ 2.1 Topo-Box-Net     │      │ 3.1 Sim2Real Ablation│
│ 1.2 Inertia Fix  ✅ │─────>│ 2.2 Kin-Consist Loss │─────>│ 3.2 VAMP Integration │
│ 1.3 DR Materials ✅ │      │                      │      │                      │
│ 1.4 Annotation   ✅ │      └──────────────────────┘      └──────────────────────┘
│ 1.5 Validation   ✅ │
│ 1.6 CAPNet Bridge✅ │
└─────────────────────┘
         │
    ┌────┴────────────────────────────────────────────┐
    │         6-Level Validation Framework             │
    │ L1: File Presence  L4: Physics (PyBullet+inertia)│
    │ L2: Array Integrity L5: Cross-Consistency        │
    │ L3: URDF Structure  L6: Batch Statistics         │
    └──────────────────────────────────────────────────┘
         │
    ┌────┴────────────────────────────────────────────┐
    │         CAPNet Data Bridge                       │
    │ FK → 6D pose (R,t) per link                      │
    │ Mesh AABB → 3D size per link                     │
    │ NPCS normalization params per link                │
    │ Metrics: Re/Te/Se/A5/A10/mIoU                    │
    └──────────────────────────────────────────────────┘
```

## Current State

- **Branch**: feature/3d-assets
- **Phase**: Phase 1 COMPLETE -- 6-level validated + CAPNet-aligned
- **Dataset**: 1000 samples (4 box types × 250), all 6-level validated, CAPNet annotations generated
- **Active Work**: [P0] Scale to 10K + CAPNet training

## Completed Milestones

| # | Date | Milestone | Key Output |
|---|------|-----------|------------|
| 1 | 2026-01-10 | Phase-0/1 MailerBox geometry + URDF export | sim_exports/urdf/mailerbox_simple/ |
| 2 | 2026-01-13 | TuckEndBox first gen (8-DOF) | sim_exports/urdf/tuckendbox/ |
| 3 | 2026-01-30 | MailerBox self-collision fix + PyBullet verification | 11 seeds verified |
| 4 | 2026-03-02 | Phase-1 Data Engine + Phase-2/3 scaffolding | 4 box type pipelines |
| 5 | 2026-03-02 | URDF fix: collision + material + 1K dataset | 1000/1000 PASS |
| 6 | 2026-03-04 | 6-level validation framework + CAPNet bridge | validate_dataset.py, capnet_data_bridge.py, eval_capnet_metrics.py |

## Validation Results (6-Level)

| Dataset | Samples | L1-L5 Pass | L6 Anomalies | Seeds | Materials | JS Std |
|---------|---------|-----------|-------------|-------|-----------|--------|
| phase1_1k_mailer | 250 | 250/250 ✅ | 0 | 25 | 9 | 0.5196 |
| phase1_1k_drawer | 250 | 250/250 ✅ | 0 | 25 | 8 | 0.0250 |
| phase1_1k_sliplid | 250 | 250/250 ✅ | 0 | 25 | 7 | 0.0150 |
| phase1_1k_tuckend | 250 | 250/250 ✅ | 0 | 25 | 9 | 0.2915 |
| **TOTAL** | **1000** | **1000/1000** | **0** | - | - | - |

## CAPNet Alignment Status

| Output | Samples | Part Annotations | Self-Consistency |
|--------|---------|-----------------|-----------------|
| link_pos_quat_aabb.json | 1000 | 4000 | Re=0, Te=0, Se=0, IoU=1.0 ✅ |
| obj_pos_quat_aabb.json | 1000 | 1000 | A5=1.0, A10=1.0 ✅ |

## Bug Fixes & Lessons (8 total)

| # | Date | Bug | Fix |
|---|------|-----|-----|
| 1 | 01-30 | Multiple <inertial> per link | Parallel Axis Theorem merge |
| 2 | 01-30 | Self-collision not configured | URDF_USE_SELF_COLLISION flag |
| 3 | 03-02 | Missing collision meshes | visual_only=False |
| 4 | 03-02 | No material color | _inject_material_color_into_urdf() |
| 5 | 03-02 | imageio crash | PIL replacement |
| 6 | 03-02 | Parallel counter collision | Separate output dirs |
| 7 | 03-04 | joint_positions dict/list ambiguity | Handle both formats |
| 8 | 03-04 | Depth background flagged as error | Foreground-only range check |

## Verified Artifacts

| Artifact | Path | Verification | Result |
|----------|------|-------------|--------|
| 1k_mailer | sim_exports/data_engine/phase1_1k_mailer/ | validate_dataset.py --level 6 | 250/250 PASS |
| 1k_drawer | sim_exports/data_engine/phase1_1k_drawer/ | validate_dataset.py --level 6 | 250/250 PASS |
| 1k_sliplid | sim_exports/data_engine/phase1_1k_sliplid/ | validate_dataset.py --level 6 | 250/250 PASS |
| 1k_tuckend | sim_exports/data_engine/phase1_1k_tuckend/ | validate_dataset.py --level 6 | 250/250 PASS |
| CAPNet annotations | sim_exports/data_engine/phase1_1k_*/capnet_annotations/ | eval_capnet_metrics.py (self-consistency) | 4/4 perfect |
| Regression tests | tests/sim/test_validation_regression.py | pytest | 7/7 PASS |
| Downstream URDFs | sim_exports/urdf/mailerbox_simple/ | verify_urdf_inertia_fix.py | 11 seeds OK |

## Next Steps

- [P0] Scale to 10K+ samples per box type for CAPNet training
- [P0] Train CAPNet on Infinigen-AnyBox, compare Se with PartNet-Mobility baseline
- [P1] Full NPCS map rendering (per-pixel normalized coords)
- [P1] Depth noise simulation for Sim2Real
- [P2] Phase 2.1: Topo-Box-Net training

## Key Files

| Category | File | Purpose |
|----------|------|---------|
| Export | scripts/export_mailerbox_simple_phase1_data_engine.py | Phase-1 pipeline (generates + validates) |
| Validation | scripts/validate_dataset.py | 6-level comprehensive validator |
| CAPNet | scripts/capnet_data_bridge.py | Convert to CAPNet format (6D pose, size, NPCS) |
| CAPNet | scripts/eval_capnet_metrics.py | Re/Te/Se/A5/A10/mIoU evaluation |
| Academic | docs/CAPNet_Academic_Alignment_Analysis.md | Innovation space + gap closure |
| Tests | tests/sim/test_validation_regression.py | 6 historical bug regression tests |
| Architecture | docs/Phase_Architecture_Overview.md | Phase 1-3 architecture diagram |
| Research | docs/Physics_Aligned_Data_Engine_Research_Plan.md | Original research proposal |

## Lessons Learned (cumulative, 14 total)

1. Blender coordinate system differs from URDF -- transform required
2. TuckEndBox has 8 DOF -- joint_states must exactly match DOF count
3. PyBullet joint limits are 'soft' -- external forces can exceed them
4. Each URDF link must have exactly one <inertial> tag with merged properties
5. Always verify with URDF_USE_INERTIA_FROM_FILE | URDF_USE_SELF_COLLISION
6. Use local RNG for deterministic DR material sampling
7. visual_only=True causes URDF to lack collision meshes
8. URDF needs <material><color rgba> for color in viewers
9. imageio crashes on high-res PNGs -- use PIL instead
10. Never share out_root between parallel batch processes
11. Always verify with both online URDF viewer AND PyBullet
12. joint_positions varies between dict and list formats -- handle both
13. Blender depth Z-pass uses ~1e10 for background -- not an error
14. Self-consistency check (GT vs GT) is the best first test for metrics
