# Stage 3 Route Entrypoint Preflight

- run_full_robot_teacher_probe.py importable after patch: `True`
- grasp_pose_shape: `[4, 4]`
- reset/preaction gate still passes: `True`
- route_entrypoint_preflight_passed: `True`
- runtime patch files: `['scripts/mint/run_full_robot_teacher_probe.py']`

The legacy entrypoint main was not invoked here because it refreshes status surfaces; bounded attempts use the same rollout builder directly and write only run_dir evidence.
