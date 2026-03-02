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
