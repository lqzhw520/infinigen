# Stage 4 Collision Policy Classification

{
  "basis": "exact MuJoCo contact pairs are dominated by broad hand/wrist/link proxy against drawer/cabinet; coarse accessibility envelope is secondary evidence only",
  "coarse_handle_accessibility_blocked": true,
  "dominant_blocker": "BROAD_PROXY_COLLISION_POLICY",
  "next_gate_if_unrepaired": "FORBIDDEN_COLLISION_PROXY_POLICY_REPAIR_REQUIRED",
  "recommended_repair_operator": "demote_or_resize_proven_duplicate_proxy_only_after_visual_physical_consistency_check",
  "safe_source_patch_allowed_now": false
}
