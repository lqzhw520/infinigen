# Findings

- Phase 1 final diagnosis: `forgetting-dominant`.
- Phase 2 final verdict: `claim_not_supported`.
- Strongest true claim: Mixed replay is the strongest validated transfer improvement, but richer conditioning under the current PhysNAP parameterization does not consistently beat the zero-state single-view anchor.
- Root-cause ranking:
  1. conditioning interface mismatch
  2. PhysNAP conditional architecture mismatch
  3. lossy merged-observation construction risk
  4. metrics expose the failure pattern but are not the primary cause
- Locked interpretation: `average-fit improved, diversity worsened, physical consistency broken` is a real signal of interface/architecture mismatch, not evidence that multi-state information has no value.
