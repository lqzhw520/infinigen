# Route Repair Telemetry Diagnosis

source_worktree_head: `251714b2b52f8f8b296997e131c5ce2893b359a1`
closeout_classification: `ROUTE_STILL_NO_GOC_V3_TARGET_CONTACT`
root_cause_classification: `route_does_not_bring_eef_close_to_handle`

## Key Distances
- min EEF-to-handle distance: 0.134439 m
- min legal-pad-to-handle centroid distance: 0.142450 m
- target contact observed: False
- forbidden contact observed: False

## MV2 Force Explanation
The prior mv2 qpos traces were replayed against the current GOC-v3 model to recover exact contact pairs. See `prior_mv2_replay_contact_diagnosis.json` for max-force contact records and exact categories.

## Governance
No strict candidate is claimed. No current_truth or next_actions mutation is performed. No GOC-v3 authority change is performed. Any future positive candidate attempt must run from a clean committed tree.
