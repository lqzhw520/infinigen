# GOC-v3 Exact-ID Review for V11-G4 Task Rewrite

Source model commit: `7a7d5c43748822d9c38d948c6d1d39ec2895fe87`

## Legal Gripper Surfaces
- geom_id `63` `link5_collision` body `link5` role `legal_gripper_surface` contype=1 conaffinity=1 group=0 mesh=`link5`
- geom_id `81` `link6_collision` body `link6` role `legal_gripper_surface` contype=1 conaffinity=1 group=0 mesh=`link6`
- geom_id `90` `link7_collision` body `link7` role `legal_gripper_surface` contype=1 conaffinity=1 group=0 mesh=`link7`

## Drawer Handle Surfaces
- geom_id `0` body `drawer_base` role `handle_surface` contype=1 conaffinity=1 group=1 mesh=`drawer_base_visual_0.obj`
- geom_id `1` body `drawer_base` role `handle_surface` contype=1 conaffinity=1 group=1 mesh=`drawer_base_visual_1.obj`
- geom_id `2` body `drawer_base` role `handle_surface` contype=1 conaffinity=1 group=1 mesh=`drawer_base_visual_2.obj`
- geom_id `3` body `drawer_base` role `handle_surface` contype=1 conaffinity=1 group=0 mesh=`drawer_base_collision_0_0.obj`
- geom_id `4` body `drawer_base` role `handle_surface` contype=1 conaffinity=1 group=0 mesh=`drawer_base_collision_0_1.obj`
- geom_id `5` body `drawer_base` role `handle_surface` contype=1 conaffinity=1 group=0 mesh=`drawer_base_collision_1_0.obj`
- geom_id `6` body `drawer_base` role `handle_surface` contype=1 conaffinity=1 group=0 mesh=`drawer_base_collision_2_0.obj`
- geom_id `7` body `drawer_base` role `handle_surface` contype=1 conaffinity=1 group=0 mesh=`drawer_base_collision_2_1.obj`
- geom_id `8` body `drawer_base` role `handle_surface` contype=1 conaffinity=1 group=0 mesh=`drawer_base_collision_2_2.obj`

## Review Decision

The task rewrite may reference GOC-v3 exact IDs, but future runtime/contact-report smoke must validate that runtime contact reports emit these exact IDs rather than GOC-v2 counts or body-based 31/27 sets.
