# V11 G4 GOC-v4 Layer4R On Accessible Instance Pool

closeout_classification: DYNAMIC_CONTACT_POLICY_FAILED_ON_ACCESSIBLE_POOL
readiness_smoke_passed: True
accepted_pool_bound: True
layer4r_cases: 0/30 passed
failure_histogram: {"forbidden_contact_present": 6, "target_contact_consecutive_lt_30": 28, "target_contact_frames_lt_50": 30}
target_contact_frames_min: 0
target_contact_consecutive_min: 0
forbidden_contact_frames_max: 50
handle_nonlegal_contact_frames_max: 0
max_penetration_m: 0.0030038893253817034
max_force_n: 597.575811329058
next_gate: CONTACT_POLICY_REPAIR_ON_ACCESSIBLE_POOL

Readiness smoke re-bound the committed accepted generated pool before matrix execution. Legacy pool fallback and empty matrix execution were disallowed.
The accepted-pool structural gate is not the blocker here: smoke recertified all 5 accepted instances. The remaining blocker is dynamic contact policy on the accessible pool: target contact is too short across all 30 cases and 6 cases also produce forbidden contact, while penetration and force remain bounded.
