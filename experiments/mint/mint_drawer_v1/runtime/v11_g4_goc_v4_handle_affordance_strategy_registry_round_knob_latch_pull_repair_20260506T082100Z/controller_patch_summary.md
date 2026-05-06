# Controller Patch Summary

- Added handle-affordance dispatch before controller selection.
- Added round-knob grasp-frame computation around knob center/radius/closing axis.
- Added exact-vs-geometric bilateral grasp forensic so contact discretization is separated from the GOC-v4 gate.
- Fast solver still requires exact two-pad GOC-v4 contact; geometric proxy cannot promote success.
- Existing bounded-pull runner lacks a true explicit latch/reseat mode-transition interface; failures are classified instead of relabeled.
