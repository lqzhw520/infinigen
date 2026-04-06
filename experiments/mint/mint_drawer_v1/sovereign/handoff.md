## Handoff — mint_drawer_v1

**Updated**: 2026-04-06T14:35:00+08:00
**Sovereign version**: 3
**STALE WARNING**: Do NOT use this file as primary truth. Read SESSION_BOOTSTRAP.20260406.md instead.

---

### Current Verdict

```
V59_RCA123_ALL_ELIMINATED — P0b_DRAWER_COLOR_ELIMINATED — SIMULATOR_PHYSICS_NOW_PRIMARY
Phase: v59_LEARNABILITY_AUDIT
```

### What Changed Since Last Handoff

| Time | Event | Evidence |
|------|--------|---------|
| 2026-04-06 00:15 | RCA2 action normalization ELIMINATED | E024 |
| 2026-04-06 01:30 | RCA3 state representation ELIMINATED | E025 |
| 2026-04-06 02:30 | P0b drawer-only matched A/B ELIMINATED | E026 |
| 2026-04-06 14:23 | SESSION_BOOTSTRAP written | sovereign/SESSION_BOOTSTRAP.20260406.md |
| 2026-04-06 14:35 | Run ledger + docs updated | sovereign/run_ledger.yaml |

### What the Next Agent Must Do

1. Read `sovereign/SESSION_BOOTSTRAP.20260406.md` FIRST — this supersedes all other docs
2. Run sovereign reconcile + lints
3. **Do NOT claim "physics confirmed" or "retrain justified"** without completing Gate A + Gate B
4. Gate A (Episode Admissibility) is the immediate next step
5. Gate B (Teacher Replayability) is the critical gap
6. **Model load fidelity is F1** — not upstream-faithful. All current conclusions are for the patched variant.

### Critical Warnings

- `handoff.md` is STALE — do not use
- `HARNESS_USAGE_GUIDE.md` and `HARNESS_HYGIENE.md` updated to v1.5 as of 2026-04-06
- All V58-era narrative in `state.json` is LEGACY — do not cite
- External/MINT submodule is dirty — do not trust as "original upstream MINT"

### Learnability Audit Gates

| Gate | Status | Blocking |
|------|--------|---------|
| Gate A: Episode Admissibility | ⏳ PENDING | All replay + retrain |
| Gate B: Teacher Replayability | ⏳ PENDING (CRITICAL) | MINT retrain |
| Gate C: Visual Sufficiency | ⏳ PENDING | Image-domain conclusions |
| Gate D: Model Load Fidelity | ⚠️ Partially done (F1) | Scientific conclusions |
| Gate E: Task Learnability | ⏳ PENDING | Full retrain auth |

### What is NOT Justified Yet

- Full MINT retrain (Gate E not passed)
- "Physics is the root cause" (Gate B not passed)
- "Images are not the issue" (Gate C not passed)
- Any "original upstream MINT" claims (MINT is F1-patched)
