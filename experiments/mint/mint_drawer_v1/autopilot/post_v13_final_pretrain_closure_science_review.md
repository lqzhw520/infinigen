# Post-v13 Final Pretrain Closure Review Memo

## Status
- diagnostic-only
- non-claim-bearing
- authoritative final verdict: `STOP_PRETRAIN_INTERFACE_UNSTABLE`

## Authority
- repo: `/mnt/afs2/zhuhaowu/infinigen`
- branch: `feature/mint-env-reformulation-v1-visual-fidelity`
- head: `bc802069816e6c81beab48ba1f2141c9f3a264d4`
- vendor: `4eab5795345721001c412ff1ca2c886a11eab606`
- run_instance_id: `postv13c_20260420T140833Z_bc802069_e2c27eee`
- execution_scope: `post_v13_final_pretrain_closure`

## Completed Scope
- v13 G1-G4 state/action interface tranche completed
- v13-slice-2 live wiring tranche completed
- post-v13 admission tranche completed
- post-v13 final pretrain closure tranche completed

## Final Closure Outcome
- C0: `PASS`
- C1: `STOP_support_expansion_failed`
- C2: `STOP_parity_mismatch`
- C3: `STOP_signal_band_unstable`
- C4: `PASS` with final verdict `STOP_PRETRAIN_INTERFACE_UNSTABLE`

## What The Closure Proved
1. Unique live support did not expand enough.
   - raw success attempts: `18`
   - selected unique live episodes: `6`
   - selected seed coverage: `6`
   - episode-index scanning for the same seed collapses onto repeated rollout fingerprints rather than yielding meaningfully new live support.
2. Live consumed state is not parity-clean against offline diagnostic reconstruction.
   - continuous-feature blockers:
- distance_to_handle_norm_p95_abs_diff_gt_1e-3
- approach_alignment_cos_p95_abs_diff_gt_1e-3
- orientation_alignment_cos_p95_abs_diff_gt_1e-3
- orientation_error_sin_p95_abs_diff_gt_1e-3
- orientation_error_cos_p95_abs_diff_gt_1e-3
- attach_eligible_proxy_exact_agreement_lt_99pct
3. Signal-band stability does not survive deterministic repeats on the current live support.
   - signal-band blockers:
- effective_unique_live_support_lt_16
- repeat_11_fit_not_ok
- repeat_11_orientation_gain_nonpositive
- repeat_23_fit_not_ok
- repeat_23_orientation_gain_nonpositive
- live_fit_below_matched_support_distribution

## What This Means
### Supported
- The current live-wired interface is **not** stable enough to approve canary train/probe.
- This is no longer just an admission ambiguity; it is a concrete pretrain stop.

### Not Supported
- attach-bridge success
- drawer success
- any claim-bearing conclusion
- a proof that MINT is fundamentally incapable on this task
- a final scientific ranking among all remaining hypotheses

## Carry-forward Context
- v13 slice-2 one-step fit was positive:
  - status: `one_step_action_fit_ok`
  - loss_drop_fraction: `0.8489133031498879`
  - close_accuracy_gain_over_baseline: `0.203125`
  - orientation_mse_improvement_fraction: `0.3788208777278708`
- post-v13 admission still held on signal-band ambiguity:
  - final_verdict: `STOP_PRETRAIN_ADMISSION_AMBIGUOUS`
  - blockers:
- slice2_live_fit_does_not_retain_prior_signal_band
- slice2_live_rollout_support_is_only_4_episodes
- current_evidence_cannot_separate_benign_live_distribution_shift_from_unresolved_signal_instability
- final closure upgraded that ambiguity into a hard stop through support, parity, and stability evidence.

## Questions For Science Agent
1. How should we interpret the same-fingerprint collapse across increasing `episode_index` within fixed seeds: deterministic environment behavior, teacher-selection artifact, or a deeper support-generation flaw?
2. Among the parity mismatches now observed, which are most scientifically meaningful:
   - `distance_to_handle_norm`
   - `approach_alignment_cos`
   - `orientation_alignment_cos`
   - `orientation_error_sin`
   - `orientation_error_cos`
   - `attach_eligible_proxy`
3. Should `attach_eligible_proxy` now be treated as a primary semantic mismatch suspect rather than just a downstream metric field?
4. What is the most defensible remaining root-cause ranking now:
   - live feature-definition mismatch
   - attach-eligible proxy mismatch
   - support entropy / effective-sample collapse
   - state decomposition mismatch
   - something else
5. What single bounded next execution spec should replace open-ended micro-tranches from here?

## What We Need Back
- one ranked root-cause interpretation
- one single bounded next spec
- explicit stop conditions
- explicit success/failure interpretations
- no automatic return to long training without a new admission contract
