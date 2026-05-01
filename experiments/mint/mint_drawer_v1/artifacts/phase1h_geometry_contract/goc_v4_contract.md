
# GOC-v4 Dedicated Finger-Pad Contract

This contract supersedes the previous broad-link GOC-v3 target-contact semantics for V11-G4 finger-pad handle contact. The legal target contact set is now the per-instance dedicated pad IDs `[102, 105]`. Broad link geoms `[70, 88, 97]` remain robot geometry and must not be counted as target contact.

Layer 4 stable contact dynamics status: `True`.

Best probe:
- target contact max consecutive frames: `7`
- forbidden contact frames: `0`
- handle nonlegal contact frames: `0`
- max penetration m: `0.0015811238301962627`
- max force N: `67.937573403025`

This is not a drawer-opening, local visual replay, or MINT training success claim. It is the Layer 4 contact-dynamics gate needed before bounded teacher pull rollout.
