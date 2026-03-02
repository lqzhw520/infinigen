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

### Phase 1: Fix URDF export — collision meshes + material tags [status: in_progress]
- [ ] 1a. Change `visual_only=False` in Phase-1 pipeline 
- [ ] 1b. Add `<material>` tag with DR color to each `<visual>` element
- [ ] 1c. Verify fixed URDF matches original structure (collision + material)

### Phase 2: Validate fix against original [status: pending]
- [ ] 2a. Export single seed MAILER with fix, diff against `sim_exports/urdf/mailerbox_simple/102`
- [ ] 2b. Verify in PyBullet with self-collision + joint limits
- [ ] 2c. Verify `urdf_gt.urdf` structure matches

### Phase 3: Re-generate 1K dataset with fix [status: pending]
- [ ] 3a. Re-run all 4 box types
- [ ] 3b. Full verification (1000/1000)
- [ ] 3c. Spot-check structure of urdf_gt.urdf

### Phase 4: Commit + Checkpoint [status: pending]
- [ ] 4a. Commit fix
- [ ] 4b. Update .session/checkpoint.md

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| (none yet) | | |
