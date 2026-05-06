# Handle Strategy Registry

- candidate_count: `39`
- round_knob_candidate_count: `39`
- non_knob_candidate_count: `0`
- current_phase_claim: `ROUND_KNOB` only
- non_knob_dispatch: `HANDLE_AFFORDANCE_BRANCH_NOT_IMPLEMENTED`

| handle_class | strategy | implemented |
|---|---:|---:|
| `ROUND_KNOB` | `RoundKnobBilateralPinchLatchStrategy` | `True` |
| `LINE_BAR_HANDLE` | `LineBarHookOrPinchStrategy` | `False` |
| `ARC_OR_C_HANDLE` | `ArcHandleHookStrategy` | `False` |
| `RECESSED_GROOVE_OR_LIP` | `RecessedLipInsertionStrategy` | `False` |
| `FLAT_FRONT_NO_GRASPABLE_HANDLE` | `UnsupportedHandleStrategy` | `False` |
| `UNKNOWN_OR_UNSUPPORTED` | `UnsupportedHandleStrategy` | `False` |
