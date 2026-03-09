# Infinigen-AnyBox Project Status
<!-- Auto-maintained by infinigen-project-memory skill. Last updated: 2026-03-09 -->

## Architecture

```
Phase 1: Data Engine          Phase 2: Perception           Phase 3: Planning
┌─────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
│ 1.1 Box Topology ✅ │      │ 2.1 Topo-Box-Net     │      │ 3.1 Sim2Real Ablation│
│ 1.2 Inertia Fix  ✅ │─────>│ 2.2 Kin-Consist Loss │─────>│ 3.2 VAMP Integration │
│ 1.3 DR Materials ✅ │      │                      │      │                      │
│ 1.4 Annotation   ✅ │      └──────────────────────┘      └──────────────────────┘
│ 1.5 Validation   ✅ │
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
    │       PhysNAP Integration Pipeline               │
    │ infinigen_to_nap.py: URDF+OBJ → NAP graph       │
    │ 750 samples (3 types) converted, verified        │
    │ physnap conda env: PyTorch 2.0 + CUDA on A800    │
    │ Training scripts ready (pending data download)   │
    └──────────────────────────────────────────────────┘
```

## Current State

- **Branch**: feature/3d-assets
- **Phase**: Phase 1 COMPLETE + PhysNAP integration in progress
- **Dataset**: 1000 samples (4 box types x 250), all 6-level validated
- **PhysNAP**: Env ready, 750/1000 samples converted to NAP format, awaiting Google Drive data download
- **Active Work**: [P0] PhysNAP reproduction + retraining with Infinigen data

## Completed Milestones

| # | Date | Milestone | Key Output |
|---|------|-----------|------------|
| 1 | 2026-01-10 | Phase-0/1 MailerBox geometry + URDF export | sim_exports/urdf/mailerbox_simple/ |
| 2 | 2026-01-13 | TuckEndBox first gen (8-DOF) | sim_exports/urdf/tuckendbox/ |
| 3 | 2026-01-30 | MailerBox self-collision fix + PyBullet verification | 11 seeds verified |
| 4 | 2026-03-02 | Phase-1 Data Engine + Phase-2/3 scaffolding | 4 box type pipelines |
| 5 | 2026-03-02 | URDF fix: collision + material + 1K dataset | 1000/1000 PASS |
| 6 | 2026-03-04 | 6-level validation framework | validate_dataset.py, test_validation_regression.py |
| 7 | 2026-03-09 | PhysNAP integration: CAPNet rollback + env + data bridge | external/physnap/, scripts/infinigen_to_nap.py |

## PhysNAP Integration Status

| Component | Status | Details |
|-----------|--------|---------|
| CAPNet rollback | ✅ DONE | 9 files deleted, STATUS/evolution cleaned |
| Academic review | ✅ DONE | docs/PhysNAP_Academic_Review_and_Integration.md |
| PhysNAP clone | ✅ DONE | external/physnap/ (MIT license) |
| Conda env | ✅ DONE | physnap: Python 3.9, PyTorch 2.0+cu118, PyG 2.5.2 |
| NAP data download | ⏳ BLOCKED | Google Drive unreachable from server |
| Data bridge | ✅ DONE | 750 samples → NAP graph format (verified compact_pack) |
| PhysNAP training | ⏳ BLOCKED | Waiting for NAP pretrained AE + PartNet-Mobility |
| Infinigen retraining | ⏳ BLOCKED | Waiting for baseline training |

## Validation Results (6-Level)

| Dataset | Samples | L1-L5 Pass | L6 Anomalies |
|---------|---------|-----------|-------------|
| phase1_1k_mailer | 250 | 250/250 | 0 |
| phase1_1k_drawer | 250 | 250/250 | 0 |
| phase1_1k_sliplid | 250 | 250/250 | 0 |
| phase1_1k_tuckend | 250 | 250/250 | 0 |
| **TOTAL** | **1000** | **1000/1000** | **0** |

## NAP Conversion Results

| Box Type | Samples | Parts/sample | Joints/sample | NAP K | Status |
|----------|---------|-------------|---------------|-------|--------|
| MAILER | 250 | 3 | 2 | ≤8 | ✅ converted |
| DRAWER | 250 | 2 | 1 | ≤8 | ✅ converted |
| SLIP_LID | 250 | 2 | 1 | ≤8 | ✅ converted |
| TUCK_END | 250 | 9 | 8 | >8 | ⚠ converted (K=10, excluded from combined) |

## Bug Fixes & Lessons (8 total)

| # | Bug | Fix |
|---|-----|-----|
| 1 | Multiple <inertial> per link | Parallel Axis Theorem merge |
| 2 | Self-collision not configured | URDF_USE_SELF_COLLISION flag |
| 3 | Missing collision meshes | visual_only=False |
| 4 | No material color | _inject_material_color_into_urdf() |
| 5 | imageio crash | PIL replacement |
| 6 | Parallel counter collision | Separate output dirs |
| 7 | joint_positions dict/list ambiguity | Handle both formats |
| 8 | Depth background flagged as error | Foreground-only range check |

## Next Steps

- [P0-BLOCKED] Download NAP data from Google Drive (scripts/download_nap_data.sh)
- [P0-BLOCKED] Train PhysNAP on PartNet-Mobility baseline (Part 2.3)
- [P0-BLOCKED] Retrain PhysNAP with Infinigen data (Part 4)
- [P1] Scale to 10K+ samples per box type
- [P1] Encode Infinigen part shapes via NAP's pretrained AE
- [P2] Phase 2.1: Topo-Box-Net training

## Key Files

| Category | File | Purpose |
|----------|------|---------|
| Export | scripts/export_mailerbox_simple_phase1_data_engine.py | Phase-1 pipeline |
| Validation | scripts/validate_dataset.py | 6-level validator |
| Tests | tests/sim/test_validation_regression.py | Regression tests |
| NAP Bridge | scripts/infinigen_to_nap.py | URDF+OBJ → NAP graph |
| NAP Merge | scripts/merge_infinigen_nap_datasets.py | Multi-type dataset merge |
| Training | scripts/train_physnap_infinigen.sh | PhysNAP training launcher |
| Download | scripts/download_nap_data.sh | NAP data download helper |
| Review | docs/PhysNAP_Academic_Review_and_Integration.md | Academic analysis |
| PhysNAP | external/physnap/ | PhysNAP codebase (ICCV 2025) |

## Lessons Learned (cumulative, 17 total)

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
14. URDF viewers cannot render OBJ textures -- only <material><color> tags
15. PyBullet loadURDF resolves mesh paths relative to URDF file location
16. NAP uses max K=8 nodes -- TuckEndBox (9 parts) exceeds this and needs K=10 or exclusion
17. Google Drive is unreachable from AFS cluster -- need manual download or proxy
