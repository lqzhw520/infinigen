# Reformulation Root-Cause Controller Cycle Memo

- cycle_id: cycle_20260412_011110
- authority_phase: v63_ROOT_CAUSE_REFRAMED_AFTER_OVERNIGHT_NEGATIVE_RESULT
- selected_experiments: ['RF4', 'RF5', 'RF6']
- scientific_terminal_state: BENCHMARK_ALIGNMENT_UNSUPPORTED_UNDER_CURRENT_FORMULATION
- route_next_branch: environment_reformulation

## Gates
- G0_baseline_integrity: passed=True summary=Authority readable and verified p1j/p1l baseline anchors confirm a healthy control baseline and non-blocking parity.
- G1_observation_contract: passed=True summary=Observation contract requires wrist-like second camera, no overlay, no diagnostic trick, and provenance-complete canonical lane.
- G2_action_causality: passed=True summary=Rotation only counts if aligned rotation measurably beats zero/random under orientation-sensitive physics.
- G3_state_semantics: passed=True summary=State gate requires a telemetry-compatible, provenance-complete, non-duplicated state candidate with non-trivial measured adequacy over M0.
- G4_visual_canonicality: passed=False summary=Visual gate requires clean canonical observation semantics plus material perceptual readability improvement.
- G5_training_eligibility: passed=False summary=Training is eligible only after G1-G4 pass and at least one causal gain threshold is met.

## Hypotheses
- H0_same_problem_identity: posterior=0.1152
- H1_embodiment_causal_contract: posterior=0.7868
- H2_state_representability: posterior=0.5545
- H3_observation_visual_contract: posterior=0.8245
- H4_unique_data_scale_only: posterior=0.2500
- H5_optimization_only: posterior=0.1500
- H6_environment_invalid_for_claim: posterior=0.9003

## Route
- why: Repeated evidence supports same-problem identity failure under the current formulation.
