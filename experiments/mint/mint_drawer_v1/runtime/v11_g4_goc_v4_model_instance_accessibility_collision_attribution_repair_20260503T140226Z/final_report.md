# V11-G4 GOC-v4 Model-Instance Accessibility / Collision Attribution Repair Closeout

Closeout: `FORBIDDEN_COLLISION_PROXY_POLICY_REPAIR_REQUIRED`

This phase did not repeat the previous 117-case matrix or controller CEM loop as its first action. It replayed prior best near-IK endpoint candidates and attributed exact contact-pair violations by GOC-v4 semantic role.

- harness preflight passed: `True`
- task spec lock bound: `True`
- prior failure ingested: `True`
- pair-level attribution completed: `True`
- dominant offending pair classes: `{'broad_hand_wrist_proxy_scene_collision': 620, 'pad_drawer_body_or_cabinet_non_target_contact': 46, 'physical_arm_link_scene_collision': 18}`
- dominant offending geom pairs: `{'drawer_base_collision_1_0.obj|link5_collision': 112, 'drawer_base_collision_1_0.obj|link7_collision': 36, 'drawer_base_collision_1_0.obj|hand_collision': 36, 'drawer_base_collision_1_0.obj|link6_collision': 34, 'drawer_door_collision_1_3.obj|link5_collision': 23, 'drawer_door_collision_1_7.obj|link6_collision': 21, 'drawer_door_collision_1_6.obj|link5_collision': 20, 'drawer_base_collision_0_1.obj|link7_collision': 20, 'drawer_door_collision_1_8.obj|hand_collision': 18, 'drawer_door_collision_1_5.obj|link5_collision': 16, 'drawer_base_collision_0_1.obj|link5_collision': 16, 'drawer_door_collision_0_1.obj|link5_collision': 13, 'drawer_door_collision_0_3.obj|link5_collision': 12, 'drawer_door_collision_1_2.obj|hand_collision': 12, 'drawer_door_collision_1_7.obj|link5_collision': 12, 'drawer_door_collision_1_2.obj|link5_collision': 12, 'drawer_door_collision_1_1.obj|hand_collision': 12, 'drawer_base_collision_0_2.obj|link5_collision': 12, 'drawer_base_collision_0_2.obj|link6_collision': 12, 'drawer_base_collision_0_2.obj|link7_collision': 12}`
- dominant blocker: `BROAD_PROXY_COLLISION_POLICY`
- handle accessibility audited: `True`
- repair cycles run: `1`
- safe source repair identified: `False`
- source patch applied: `False`
- endpoint feasible after repair: `False`
- feasible seed/case count: `0` / `0`
- next gate: `SAFE_COLLISION_PROXY_POLICY_REPAIR_WITH_VISUAL_PHYSICAL_CONSISTENCY_GUARD`

Layer5 pull rollout, export, local replay/render, and MINT training were not run.

## Post-Push Verification
- evidence_commit_hash: `e20c58d16e810d045e382e80256770de08584f76`
- origin_head_at_verification: `e20c58d16e810d045e382e80256770de08584f76`
- all_required_paths_origin_visible: `True`
