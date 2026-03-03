# Session Checkpoint
<!-- Auto-synced from .project-memory/STATUS.md by infinigen-project-memory skill -->

**Date**: 2026-03-03
**Branch**: feature/3d-assets
**Last Commit**: 42ce8dff Update planning files with iteration protocol

## Status

See `.project-memory/STATUS.md` for the full project status.

## Completed
- Fix: URDF collision meshes (visual_only=False)
- Fix: Material color injection into urdf_gt.urdf <material><color> tags
- Fix: imageio crash replaced with PIL for segmentation PNG
- 1K batch dataset v2: 1000/1000 PASS (250 per box type) with collision + material

## Next
- [P0] Verify 1K dataset usable by downstream Planner (collision check, Topo-Box-Net load)
- [P1] Scale to 10K+ samples: more seeds + random joint states + more viewpoints
- [P1] Phase 2.1: Topo-Box-Net training script + baseline training
- [P2] Joint state diversity: random sampling instead of 2 fixed values

## Lessons
- visual_only=True causes URDF to lack collision meshes -- downstream planner cannot use
- URDF needs <material><color rgba> tags for color display in online viewers
- imageio crashes on high-res segmentation PNGs -- use PIL instead
- Never share out_root between parallel batch processes -- sample counter collides
- Always verify with both online URDF viewer AND PyBullet
