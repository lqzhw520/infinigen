# Prior Failure Attribution

Source Layer4R run: `experiments/mint/mint_drawer_v1/runtime/v11_g4_goc_v4_layer4r_on_accessible_instance_pool_20260504T065000Z`
Rows: 30
Failure histogram: `{"forbidden_contact_present": 6, "target_contact_consecutive_lt_30": 28, "target_contact_frames_lt_50": 30}`

Interpretation: the fixed accepted pool and GOC-v4 binding were already re-certified; the unresolved blocker is dynamic execution policy on that pool.
The old policy produced legal target contact only briefly on candidate 001 and produced forbidden hand_collision contact on candidate 002. Candidates 003-005 remained contact-short without forbidden contact, which is consistent with actuation/tracking/timing rather than structural accessibility.
