# Box Conditioning v2 Positioning

Phase 2 directly targets PhysNAP's limitation: zero-articulation, single-observation, limited-view conditioning.

The Phase 2 v1 path stays PhysNAP-compatible:
- build PhysNAP-style `cond_dir` datasets
- run `run_guided.py --cond_dir ...` on one selected Phase 1 checkpoint
- compare four conditioning settings on the same generator

The active hypothesis is that Infinigen-generated multi-state, multi-view observations provide a stronger conditioning signal than the original zero-state single-view setup.
