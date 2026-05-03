# V11-G4 GOC-v4 Model/Layout Redesign With Physical Accessibility Invariant Closeout

Closeout: `MODEL_OR_LAYOUT_REDESIGN_ACCESSIBILITY_INVARIANT_FAILED`

This phase executed a repair-style model/layout search rather than another raw
controller parameter retry. It tested whether changing the robot mount relation
(base position, yaw, and height) while preserving visual/physical consistency and
GOC-v4 exact-ID authority can satisfy the joint condition:

`dedicated two-pad handle-frame reachability + reset clearance + full robot forbidden-contact keepout`.

Key results:

- harness preflight passed: `True`
- model/layout repair applied: `True`
- visual/physical consistency preserved: `True`
- GOC-v4 exact IDs regenerated: `False`
- dedicated pads: `unchanged; GOC-v4 per-instance exact IDs reverified`
- accepted accessible instances: `0`
- rejected inaccessible instances: `13`
- endpoint candidates evaluated: `23920`
- endpoint feasible cases/seeds: `0` / `0`
- best reset-clean two-pad residual m: `0.0011411312852820784`
- signed clearance improved: `False`
- best full robot corridor clearance m: `None`
- full robot corridor exists: `False`
- dynamic probe attempted/passed: `False` / `False`
- Layer4R certified: `False`
- Layer4R cases passed/failed: `0` / `0`
- next gate: `MODEL_OR_LAYOUT_REDESIGN_WITH_STRONGER_STRUCTURAL_CHANGE`

Layer5 pull rollout, export bundle, local replay/render, and MINT training were not run.
