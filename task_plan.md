# Task Plan: Fix Phase-1 URDF Quality Issues

## Goal
Fix two critical issues found by user when validating Phase-1 dataset URDFs in online viewer:
1. **Joint limits not enforced** — joints can rotate infinitely vs original URDF has proper limits
2. **No materials visible** — white/default appearance, domain-randomized materials not visible in URDF

## Root Cause Analysis (Findings)

### Issue 1: Missing Collision Meshes
- Phase-1 exports with `visual_only=True` → NO `<collision>` tags in URDF
- Original `sim_exports/urdf/mailerbox_simple/` has `<collision>` with `*_col0.obj`
- Joint limits ARE present in both (`<limit lower="-3.141593" upper="3.141593"/>`)
- The online viewer may not respect limits without collision meshes, OR the user's
  observation about "infinite rotation" may relate to the full -π to +π range behavior

### Issue 2: No Material in URDF
- Domain randomization applies Blender shader materials for RGB rendering only
- The exported `.obj` files are plain geometry with no material/color
- The URDF has NO `<material>` tags → online viewer shows white
- Need to add `<material>` with color from DR config to URDF `<visual>` elements

## Phases

### Phase 1: Fix URDF export — collision meshes + material tags [status: complete]
- [x] 1a. Changed `visual_only=False` in Phase-1 pipeline (line 572)
- [x] 1b. Added `_inject_material_color_into_urdf()` for `<material><color rgba>` injection
- [x] 1c. Added `_copy_urdf_gt_with_assets()` for self-contained sample folders

### Phase 2: Validate fix against original [status: complete]
- [x] 2a. Exported single seed MAILER (102), confirmed collision + material present
- [x] 2b. PyBullet loaded with `URDF_USE_INERTIA_FROM_FILE | URDF_USE_SELF_COLLISION` — OK
- [x] 2c. Joint limits identical to original (`[-3.1416, 3.1416]`)

### Phase 3: Re-generate 1K dataset with fix [status: complete]
- [x] 3a. Re-ran all 4 box types (25 seeds × 2 joint states × 5 views = 250 each)
- [x] 3b. Full verification: **1000/1000 PASS**
- [x] 3c. Spot-checked: MAILER=7 collision/material, DRAWER=9, TUCKEND=12

### Phase 4: Commit + Checkpoint [status: complete]
- [x] 4a. Committed: `35b99593 Fix Phase-1 URDF: add collision meshes + material color tags`
- [x] 4b. Checkpoint updated below

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| imageio SystemError at 1024x768 | 1 | Replaced with PIL.Image.fromarray().save() |
| visual_only=True → no collision | 1 | Changed to False |
| No material in URDF | 1 | Post-process inject `<material><color>` tags |
