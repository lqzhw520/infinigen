# Physical Accessibility Invariant

{
  "I0_model_load": [
    "full Panda/gripper",
    "drawer/cabinet/handle",
    "dedicated pad binding",
    "handle binding",
    "no unknown contact-relevant geoms"
  ],
  "I1_reset": {
    "forbidden_contacts": 0,
    "max_force_n": 1000000.0,
    "max_penetration_m": 0.02
  },
  "I2_semantic_binding": "dedicated pads and true handle only; broad shell/body/name-only rejected",
  "I3_two_pad_grasp_geometry": {
    "max_residual_m": 0.02,
    "preferred_residual_m": 0.005
  },
  "I4_full_body_keepout": "nonlegal robot surfaces have zero forbidden contact at endpoint and are audited for visual/physical proxy clipping",
  "I5_approach_corridor": "collision-legal swept path from reset/pregrasp to hold/contact",
  "I6_pull_corridor": "small pull-axis robot motion feasible without forbidden contact, no direct drawer qpos opening",
  "I7_visual_physical_consistency": "no new fake collision demotion; drawer/handle visual and collision semantics preserved"
}
