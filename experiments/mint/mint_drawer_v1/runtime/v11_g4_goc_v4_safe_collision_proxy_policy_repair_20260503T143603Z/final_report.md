# SAFE Collision Proxy Policy Repair With Visual/Physical Consistency Guard

Generated: `2026-05-03T14:41:14Z`

## Decision

- closeout_classification: `SAFE_PROXY_REPAIR_REJECTED_PHYSICAL_COLLISION_ESSENTIAL`
- next_gate: `ROBOT_MOUNT_LAYOUT_OR_DRAWER_ACCESSIBILITY_REDESIGN_WITH_PHYSICAL_COLLISION_GUARD`
- safe_source_proxy_repair_identified: `False`

## Why

Dominant offending geoms have same-body visual counterparts of comparable size and are named physical palm/wrist/arm collision bodies; demoting them would create a hidden pass-through rather than a valid Layer4R repair.

## Prior Pair Classes

- `broad_hand_wrist_proxy_scene_collision`: 620
- `pad_drawer_body_or_cabinet_non_target_contact`: 46
- `physical_arm_link_scene_collision`: 18

## Offender Audit

- `link5_collision` body=`link5` count=562 collision_rbound=0.193834 max_visual_rbound=0.191533 ratio=1.012 physical_essential=True safe_demote_candidate=False reason=`collision_geom_matches_visible_physical_robot_body`
- `hand_collision` body=`right_gripper` count=296 collision_rbound=0.119058 max_visual_rbound=0.119563 ratio=0.996 physical_essential=True safe_demote_candidate=False reason=`collision_geom_matches_visible_physical_robot_body`
- `link6_collision` body=`link6` count=212 collision_rbound=0.132899 max_visual_rbound=0.139848 ratio=0.950 physical_essential=True safe_demote_candidate=False reason=`collision_geom_matches_visible_physical_robot_body`
- `link7_collision` body=`link7` count=170 collision_rbound=0.090883 max_visual_rbound=0.068812 ratio=1.321 physical_essential=True safe_demote_candidate=False reason=`collision_geom_matches_visible_physical_robot_body`
- `finger2_pad_collision` body=`finger_joint2_tip` count=54 collision_rbound=0.012000 max_visual_rbound=0.000000 ratio=inf physical_essential=False safe_demote_candidate=False reason=`no_same_body_visual_evidence_for_safe_demote`
- `finger1_pad_collision` body=`finger_joint1_tip` count=38 collision_rbound=0.012000 max_visual_rbound=0.000000 ratio=inf physical_essential=False safe_demote_candidate=False reason=`no_same_body_visual_evidence_for_safe_demote`
- `link4_collision` body=`link4` count=28 collision_rbound=0.165694 max_visual_rbound=0.166860 ratio=0.993 physical_essential=True safe_demote_candidate=False reason=`collision_geom_matches_visible_physical_robot_body`
- `link3_collision` body=`link3` count=8 collision_rbound=0.164639 max_visual_rbound=0.165368 ratio=0.996 physical_essential=True safe_demote_candidate=False reason=`collision_geom_matches_visible_physical_robot_body`
- `finger1_collision` body=`leftfinger` count=0 collision_rbound=0.035996 max_visual_rbound=0.036432 ratio=0.988 physical_essential=False safe_demote_candidate=True reason=`None`
- `finger2_collision` body=`rightfinger` count=0 collision_rbound=0.035996 max_visual_rbound=0.036432 ratio=0.988 physical_essential=False safe_demote_candidate=True reason=`None`

## Scientific Boundary

This phase does not claim Layer4R, teacher rollout, export, local replay/render, or MINT success. It closes the proxy-policy gate by proving whether demotion/resizing is a scientifically valid source repair. In this run, unsafe physical-body demotion is rejected.
