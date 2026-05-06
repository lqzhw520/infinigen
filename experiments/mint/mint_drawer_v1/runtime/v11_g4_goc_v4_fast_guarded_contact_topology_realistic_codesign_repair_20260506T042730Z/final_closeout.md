
# Fast Guarded Contact Topology-Realistic Repair Closeout

- closeout_classification: `FAST_GUARDED_CONTACT_TARGETED_REPAIR_FAILED`
- targeted_shard: `6/7`
- fast_guarded_contact_passed: `False`
- fast_forensic_best_opening_drawer_fraction: `0.956438122722968`
- fast_forensic_best_legal_drawer_fraction: `0.956438122722968`
- fast_forensic_best_legal_two_pad_target_frames: `6`
- fast_forensic_forbidden_contact_frame_count_max: `0`
- blocking_failure_reasons: `{'pull_phase_two_pad_target_contact_frames_lt_30': 1}`
- full30_certification_passed: `False`
- strict_export_complete: `False`
- local_state_replay_render_passed: `False`
- action_only_spot_check_passed: `False`
- next_gate: `FAST_GUARDED_CONTACT_REPAIR_CONTINUATION`

The bounded fast-only Pareto sweep found legal high-opening fast attempts, but no strict fast attempt with both drawer fraction >= 0.80 and pull-phase two-pad target contact frames >= 30. The best legal opening reached drawer_fraction 0.956438122722968 with zero forbidden and zero handle-nonlegal frames, but only 6 two-pad target frames, so the targeted shard remains 6/7 and full30/export/replay were not claim-bearing attempted.
