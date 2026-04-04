## Handoff — mint_drawer_v1

**Updated**: 2026-04-03T16:52:48+00:00
**Sovereign version**: 2

### Current Verdict

**Status**: `blocked`
**Blockers**: `C_PHYSICS_LEGAL_TEACHER: P4 gate failed (0/6 physics-legal seeds)`
**Reason**: `P4 FAIL: 0/6 physics-legal seeds (action_delta ~1m >> 0.05m threshold — all teacher rollouts are physically illegal). P1a PASS: state[7] continuous ∈ [-0.042,+0.001]. P1b PASS: DrawerRobotEnvMujoco dual-camera 256x256. BLOCKER: physics-illegal teacher data is upstream of everything.`
**Decision**: `fix_physics_legality_gate_P0`

### What the Next Agent Must Do

1. Run: `python3 scripts/harness/sovereign_cli.py bootstrap`
2. Read: `sovereign/claims.yaml` — ONLY sovereign files are canonical
3. DO NOT edit `sovereign/claims.yaml` directly — use `sovereign_cli.py revise-claim`
4. DO NOT close experiments directly — use `sovereign_cli.py close-experiment`
5. Read: `sovereign/handoff.md` (this file) for latest state
