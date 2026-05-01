
# GOC-v4 Dedicated Finger-Pad Geometry Ownership Contract

GOC-v4 creates dedicated collision-only fingertip pad geoms for V11-G4 contact-rich drawer manipulation. The previous GOC-v3 legal geoms [63, 81, 90] remain historical broad-link authority, but they are demoted from target contact for this phase.

Legal dedicated finger-pad geom IDs: `[98, 99]`.
Forbidden robot surface geom IDs: `[52, 54, 56, 61, 66, 70, 88, 97]`.
Drawer handle geom IDs: `[0, 1, 2, 3, 4, 5, 6, 7, 8]`.

The contract rejects body-based 31/27 authority and count-only target contact. It does not mutate current_truth, next_actions, or the GOC-v3 artifact.
