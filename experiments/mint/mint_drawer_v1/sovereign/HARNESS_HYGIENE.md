# Harness Hygiene Rules — mint_drawer_v1

**Version**: 1.5
**Generated**: 2026-04-05T14:35+08:00
**Last Updated**: 2026-04-06T14:35+08:00 (Learnability Audit gates; E026 updated; SESSION_BOOTSTRAP + run_ledger added)
**Supersedes**: Ad-hoc operations that caused V58 provenance collapse

---

## Core Principles

1. **Timestamp-tagged operations** — Every destructive operation (pack, clean, delete) MUST be tagged with ISO timestamp
2. **No-overwrite policy** — Never overwrite `dataset/` without first archiving. `remove_existing=True` in `build_dataset_from_rollouts` is OK ONLY if prior dataset was already archived
3. **Provenance-locked artifacts** — Pack results (`.json`) and logs (`.log`) are written BEFORE the operation, and are read-only after
4. **Clean dataset truth** — The `dataset/` directory is the authoritative "current" dataset. Prior versions live in `artifacts/v58_archive_YYYY-MM-DD_HHMMSS/`
5. **Sovereign state follows reality** — `sovereign/state.json` phase must match the actual disk state

---

## Dataset Operations

### Before Any Pack

```bash
# Always archive first if dataset/ exists
TS=$(date '+%Y-%m-%d_%H%M%S')
if [ -d "experiments/mint/mint_drawer_v1/dataset" ]; then
    mv experiments/mint/mint_drawer_v1/dataset "experiments/mint/mint_drawer_v1/artifacts/v58_archive_${TS}/dataset/"
fi
```

### Pack Naming Convention

| Run | Artifact | Source | Status |
|-----|---------|--------|--------|
| V58 original (Apr 4 23:20) | `v58_lerobot_pack.json` | 229 rollouts | Archive: `v58_archive_2026-04-05_1435/` |
| V59 (Apr 5) | `v59_lerobot_pack.log` | 240 rollouts | **CURRENT** |

### After Any Pack

1. Check `dataset_loads: true` in the result JSON
2. Run `python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; d = LeRobotDataset(...)"`.
3. Verify `len(d) == expected_frames`
4. Update sovereign `state.json` phase and `dataset_provenance` field
5. Write log with provenance header:

```bash
{
    echo "=========================================="
    echo "[V59 pack] CLEAN REPACK — provenance-tagged"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Source: v58_physics_legal_rollouts/ + p4_physics_legal_rollouts/"
    echo "Previous dataset: artifacts/v58_archive_${TS}/dataset/"
    echo "=========================================="
} > "experiments/mint/mint_drawer_v1/outputs/v59_lerobot_pack.log"
```

---

## What Went Wrong in V58 (Apr 4→Apr 5)

| Time | Action | Consequence |
|------|--------|-------------|
| Apr 4 23:20 | `v58_lerobot_pack.log` → 229/18701, `dataset_loads=true` | ✅ Clean dataset truth |
| Apr 4 23:34 | Training starts | Reads 229/18701 |
| Apr 4 23:58 | Training completes | Success |
| Apr 5 13:32 | `run_v58_lerobot_pack.py` re-run | Finds 240 rollouts |
| Apr 5 13:38 | Pack interrupted | Leaves 36/3305 + corrupted parquet |
| Result | **Overwrote** original dataset/ | Original 229/18701 parquet files GONE |

**Root cause**: No archive step before re-running pack. `remove_existing=True` silently deleted the working dataset.

---

## Archive Structure

```
experiments/mint/mint_drawer_v1/artifacts/
├── v58_archive_2026-04-05_1435/     ← Apr 5 cleanup
│   ├── PROVENANCE.txt                 ← What happened + lessons
│   ├── dataset_corrupted_from_interrupted_pack/   ← The overwritten residue
│   ├── v58_lerobot_pack_original.json ← Original 229/18701 truth
│   ├── v58_lerobot_pack_fix.log      ← Failed re-pack log
│   └── v58_gripper_patch.log        ← Failed gripper patch log
├── v58_physics_legal_rollouts/        ← NPZ source files (UPSTREAM — do not delete)
└── p4_physics_legal_rollouts/        ← NPZ source files (UPSTREAM — do not delete)
```

---

## Sovereign State Hygiene

- `sovereign/state.json` phase must reflect actual disk state
- After packing: update `phase`, `dataset_provenance`, `last_updated`
- Do NOT write conclusions about dataset state without checking both pack artifact AND live `dataset/meta/info.json` timestamp
- Verify timestamps: `stat dataset/meta/info.json` vs pack log timestamp

---

## Checkpoint Hygiene

- Training checkpoints: `artifacts/v58_training_outputs/checkpoints/`
- Checkpoints are frozen after eval — do NOT delete after eval
- Each training run should have a corresponding `vXX_train.log` and `vXX_eval.log`
- Always archive old outputs before new run

---

## NPZ Source Files Are the Upstream Truth

- `v58_physics_legal_rollouts/` and `p4_physics_legal_rollouts/` are the **source of truth**
- If `dataset/` is ever corrupted, regenerate from these
- Never modify or delete NPZ files — they are the immutable upstream
- Each NPZ file has a `.json` metadata file (task, seed, branch, success flag)

---

## Version History

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0 | 2026-04-05T14:35 | harness_agent | Initial rules after V58 provenance collapse |
| 1.4 | 2026-04-05T17:00 | harness_agent | V59 pack hygiene |
| 1.5 | 2026-04-06T14:35 | harness_agent | Learnability Audit gates; SESSION_BOOTSTRAP + run_ledger added |

---

## Learnability Audit Hygiene

Every new RCA or diagnostic before launching MINT retrain MUST complete these gates:

| Gate | Name | Must Pass Before |
|------|------|-----------------|
| Gate A | Episode Admissibility | Any replay or retrain |
| Gate B | Teacher Replayability | MINT retrain decision |
| Gate C | Visual Sufficiency | Image-domain conclusions |
| Gate D | Model Load Fidelity | Scientific conclusions about MINT |
| Gate E | Task Learnability | Full retrain authorization |

**The "just run more RCA" pattern is forbidden.** Do not write "root cause narrowed to X" without completing Gates A/B first.

---

## Session Bootstrap Protocol

Each new session MUST read and follow `sovereign/SESSION_BOOTSTRAP.20260406.md` FIRST. It supersedes all stale docs including this file.
