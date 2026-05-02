# V11-G4 GOC-v4 Handle-Frame Grasp Trajectory Planner Closeout

Closeout: `HANDLE_FRAME_IK_INFEASIBLE`

This phase reviewed and executed the object-centric planner remedy for the
previous four-cycle Layer4R stall. It does not reuse fixed drawer handle IDs
or body-based 31/27 authority. The planner builds a per-instance handle frame,
constructs two-pad grasp targets, solves constrained two-pad IK keyframes, and
uses contact-mode execution before any full Layer4R promotion.

- handle-frame planner built: `True`
- two-pad grasp frame built: `True`
- constrained IK attempted: `True`
- constrained IK feasible cases: `0`
- target-short before/after: `{'before': 117, 'after': 12}`
- forbidden before/after: `{'before': 36, 'after': 2}`
- penetration before/after: `{'before': 18, 'after': 0}`
- Layer4R cases passed: `0`
- Layer4R cases failed: `12`
- next gate: `MODEL_OR_PLACEMENT_REPAIR`

Layer5 rollout remains prohibited unless the full Layer4R matrix passes.
