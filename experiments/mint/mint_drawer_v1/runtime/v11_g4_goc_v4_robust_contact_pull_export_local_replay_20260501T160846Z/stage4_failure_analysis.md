# Stage 4 Robust Layer4R Failure Analysis

Robust Layer4R did not pass. The run completed 117 cases over all currently available drawer seeds and perturbations. Passing single-instance cases exist, including seed 13, but 90 cases failed, so bounded pull rollout was correctly blocked.

Failure reason counts:
- forbidden_contact_present: 28
- max_penetration_gt_0p02m: 27
- model_load_or_probe_error: 18
- pass: 27
- target_contact_consecutive_lt_30: 65
- target_contact_frames_lt_50: 65
