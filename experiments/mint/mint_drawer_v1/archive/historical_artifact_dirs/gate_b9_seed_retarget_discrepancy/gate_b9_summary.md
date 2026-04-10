# E033 - Gate B Seed-vs-Retarget Discrepancy Audit

- timestamp: 2026-04-07T21:40:05+08:00
- attach_threshold_m: 0.05
- sample: Gate A action-quality top-10
- mapping_target: success-selected legacy rollout subset
- population_claim: No population-wide claim over full 240 episodes

## Mode Summary
- open_loop: attach=0.300, success=0.100, dominant_failure=never_near_handle
- seeded_initial_physical_pose: attach=0.600, success=0.200, dominant_failure=never_near_handle
- seeded_initial_physical_and_command: attach=0.300, success=0.300, dominant_failure=never_near_handle
- retarget_first_20_steps: attach=0.400, success=0.100, dominant_failure=never_near_handle
- retarget_until_source_reference_step: attach=0.400, success=0.100, dominant_failure=never_near_handle
- retarget_until_near_handle: attach=0.400, success=0.000, dominant_failure=never_near_handle

## Key Findings
- best_seed_mode: seeded_initial_physical_and_command (0.300 success)
- best_retarget_mode: retarget_first_20_steps (0.100 success)
- seeded_command_alignment_success_delta: 0.100
- continuous_minus_best_seed_success_delta: -0.300
- continuous_minus_finite_best_success_delta: -0.100

## Final Branch
- one-shot seeding outperforms retargeting; the seed-vs-retarget discrepancy is consistent with continuous pre-handle intervention being too intrusive
