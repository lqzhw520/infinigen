# Archived Task Plan: average-fit improved, diversity worsened, physical consistency broken

**Archived**: 2026-03-20 20:56 (auto by post-commit hook)

# Task Plan

**Status**: claim_not_supported
**Last Updated**: 2026-03-20T19:36:21+08:00

## Campaign

- campaign_dir=/mnt/afs2/zhuhaowu/infinigen/experiments/physnap/box_conditioning_v2
- phase=phase2_conditioning_design
- phase_gate=ready_for_writeup

## Queue

- `prepare_conditioning_v2`: completed
- `export_cond_zero_singleview`: completed
- `export_cond_zero_multiview`: completed
- `export_cond_multistate_singleview`: completed
- `export_cond_multistate_multiview`: completed
- `run_guided_zero_singleview`: completed
- `run_guided_zero_multiview`: completed
- `run_guided_multistate_singleview`: completed
- `run_guided_multistate_multiview`: completed
- `evaluate_conditioning_v2`: completed
- `write_phase2_claim_memo`: completed

## Review

- verdict=claim_not_supported
- decision=revise_claim
- workflow_score=10
- evidence_score=5

---
## Findings

# Findings

- campaign=box_conditioning_v2
- phase=phase2_conditioning_design
- gate=ready_for_writeup
- verdict=claim_not_supported
- Weakness: Baseline, scratch, and fine-tune comparison arms are incomplete.
- Weakness: `zero_multiview` does not consistently beat the zero-state single-view anchor.
- Weakness: `multistate_singleview` does not consistently beat the zero-state single-view anchor.
- Weakness: `multistate_multiview` does not consistently beat the zero-state single-view anchor.
- Weakness: Claim ladder `multi-state + multi-view beats zero-state single-view` via `multistate_multiview` => not supported.
- Weakness: Claim ladder `multi-view only beats zero-state single-view` via `zero_multiview` => not supported.
- Weakness: Claim ladder `multi-state only beats zero-state single-view` via `multistate_singleview` => not supported.
