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
