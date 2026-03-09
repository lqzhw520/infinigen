# Session Checkpoint
<!-- Auto-synced from .project-memory/STATUS.md by infinigen-project-memory skill -->

**Date**: 2026-03-04
**Branch**: feature/3d-assets
**Last Commit**: 15a02e60 Upgrade project-memory skill v2: full auto-pipeline hook + self-evolution + archive cleanup

## Status

See `.project-memory/STATUS.md` for the full project status.

## Completed
- Comprehensive 6-level validation framework (validate_dataset.py)
- 7 regression tests encoding all 6 historical bugs (test_validation_regression.py)
- Full 6-level validation: 1000/1000 samples PASS across 4 box types
- Export pipeline upgraded to auto-run full 6-level validation post-export
- CAPNet data bridge: per-link 6D pose, 3D size, AABB, NPCS from URDF+FK
- CAPNet metrics evaluator: Re/Te/Se/A5/A10/mIoU + self-consistency verified
- CAPNet annotations generated for all 1K samples (4000 part annotations)
- Academic alignment analysis: innovation space + CAPNet gap closure

## Next
- [P0] Scale to 10K+ samples per box type for robust CAPNet training
- [P0] Train CAPNet on Infinigen-AnyBox data, compare Se with PartNet-Mobility baseline
- [P1] Full NPCS map rendering (per-pixel normalized coords in camera frame)
- [P1] Depth noise simulation (Gaussian + dropout) for Sim2Real gap reduction
- [P2] Phase 2.1: Topo-Box-Net training with CAPNet pre-training

## Lessons
- URDF viewers (online) cannot render OBJ textures -- material colors visible only as URDF <material><color> tags
- PyBullet loadURDF resolves mesh paths relative to URDF file location
- joint_positions format varies by export version (dict vs list) -- always handle both
- Blender depth Z-pass uses ~1e10 for background -- not an error, just far-plane value
- Self-consistency check (GT vs GT) is the best first test for any metrics pipeline
- CAPNet NPCS normalization: center at AABB center, divide by max extent
