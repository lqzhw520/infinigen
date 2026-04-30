# Model Builder Repair Decision

The reset blocker was physical plausibility, not GOC-v3 authority. The repair mounts the Panda at `[-0.9, 0.0, 0.0]` and suppresses only closed-drawer internal drawer self-collision (`drawer_base` with `link_1`/`link_2`). Robot-drawer and robot-handle collision remain active.

After repair:

- forbidden contacts at reset: `0`
- max reset penetration: `0.0` m
- max reset contact force: `0.0` N
- exact GOC-v3 authority emitted: `True`
- body-based 31/27 rejected: `True`
- legacy name-only rejected: `True`

No rollout, render, training, direct-qpos drawer opening, threshold lowering, or GOC-v3 authority change was used.
