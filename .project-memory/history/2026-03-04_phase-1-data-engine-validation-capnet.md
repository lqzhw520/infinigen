# Iteration Archive: Phase-1 Data Engine Validation CAPNet

**Archived**: 2026-03-04 19:26

## Task Plan

# Task Plan: Phase-1 Data Engine Completion

**Status**: COMPLETE  
**Last Updated**: 2026-03-02  
**Managed by**: planning-with-files skill

## Iteration Protocol

### When to use task_plan.md + findings.md

These files are per-task working memory: created at start of complex task, updated during execution, archived when done.

| File | Purpose | Lifecycle |
|------|---------|-----------|
| task_plan.md | Current task phases + progress + errors | Created per task, archived on completion |
| findings.md | Root cause analysis, discoveries | Created per task, archived on completion |
| .session/checkpoint.md | Cross-session persistent state | Always overwritten with latest state |

### Iteration Flow

1. New task arrives: Create task_plan.md + findings.md
2. During execution: Update both, re-read plan before decisions
3. Task complete: Mark phases done, commit, update checkpoint
4. Next task: Overwrite with new content OR archive to docs/archived_plans/

### vs .session/checkpoint.md

- task_plan.md = what I am doing RIGHT NOW (tactical, per-task)
- checkpoint.md = where the project IS (strategic, cross-session)

## Current Task (COMPLETE)

Fix Phase-1 URDF quality: collision meshes + material color tags

| Phase | Status | Output |
|-------|--------|--------|
| Fix URDF export | DONE | visual_only=False + material injection |
| Validate vs original | DONE | PyBullet match, 7 collision + 7 material |
| Re-generate 1K | DONE | 1000/1000 PASS |
| Commit | DONE | 35b99593, d7c06f72 |

## Errors

| Error | Fix |
|-------|-----|
| imageio crash 1024x768 | PIL.Image.fromarray().save() |
| visual_only=True | Changed to False |
| No URDF material | Inject material color tags |
| URDF viewer white | Expected: viewers dont support PBR, check rgb.png |


## Findings

# Findings

## Issue 1: Phase-1 URDF missing collision meshes (downstream unusable)

**Original** URDF (`sim_exports/urdf/mailerbox_simple/102/mailerbox_simple.urdf`):
- Has `<collision>` with `geom_*_col0.obj` for every link
- Has `<visual>` with `geom_*.obj` for every link
- Full simulation support: collision checking, self-collision, planner usable

**Phase-1** `urdf_gt.urdf` (`phase1_1k_mailer/dataset/train/000001/`):
- Has ONLY `<visual>` tags, NO `<collision>` at all
- **Root cause**: `visual_only=True` on line 572 of export script
- Without collision meshes, downstream planner cannot check collisions

**Fix**: Change `visual_only=True` to `visual_only=False` in Phase-1 export

## Issue 2: No material/color in URDF

**Root cause chain**:
1. `sample_box_material_config()` returns color/roughness/density
2. `factory._apply_box_material(obj, mat_cfg)` applies Blender shader
3. RGB render (`rgb.png`) captures material correctly
4. BUT URDF exporter never writes `<material>` or `<color>` tags
5. `.obj` files are plain geometry

**Fix**: Post-process urdf_gt.urdf to inject `<material>` with RGBA from DR config


## Checkpoint Snapshot

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

