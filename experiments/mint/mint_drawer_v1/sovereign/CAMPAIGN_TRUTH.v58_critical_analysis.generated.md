# V58 Campaign Truth — Generated 2026-04-05T14:35+08:00

## EXECUTIVE SUMMARY

**V58 0% is NOT evidence that MINT cannot learn.**
**V58 0% is evidence of a BROKEN DATA PIPELINE at multiple levels.**
**The critical blocker is NOT single root cause — it is DATA PROVENANCE COLLAPSE.**

---

## PART 1 — THE DATA PROVENANCE COLLAPSE (Critical Context)

### The Four Truths Have Diverged

| Source | Date | Episodes | Frames | Status |
|--------|------|----------|--------|--------|
| **Train-Time Truth** (v58_train.log) | Apr 4 23:34-23:58 | 229 | 18,701 | USED for V58 training |
| **Pack Artifact Truth** (v58_lerobot_pack.log) | Apr 4 23:20 | 229 | 18,701 | 2 parquet files, `dataset_loads: true` |
| **Current Live Truth** (dataset/) | Apr 5 13:38 | 36 | 3,305 | **OVERWRITTEN**, corrupted parquet, unreadable |
| **Sovereign Written Truth** (prior sessions) | Apr 5 14:00 | — | — | Had WRONG conclusions on P0a |

### Timeline Reconstruction

```
2026-04-04 23:20  → v58_lerobot_pack.log: SUCCESS. 229 episodes, 18701 frames.
                    Wrote dataset/data/chunk-000/file-000.parquet (12,480 frames)
                    Wrote dataset/data/chunk-000/file-001.parquet  ( 6,221 frames)
                    Wrote dataset/meta/info.json (229 episodes, 18701 frames)
                    Wrote dataset/meta/stats.json (18701 frames stats)
                    Result: dataset_loads=true

2026-04-04 23:34  → v58_train.log: Training STARTS
                    dataset.num_frames=18701, dataset.num_episodes=229
                    Uses dataset/ path — reads original parquet files

2026-04-04 23:58  → v58_train.log: Training COMPLETES
                    1000 steps, loss 6.52→4.43
                    Checkpoint: artifacts/v58_training_outputs/checkpoints/001000/

2026-04-05 13:32  → run_v58_lerobot_pack.py RESTARTS (misnamed as "pack_fix")
                    Finds 240 legal rollouts (11 more than original 229)
                    Starts writing to SAME dataset/ path — OVERWRITES original
                    INTERRUPTED mid-pack

2026-04-05 13:38  → Interrupted pack leaves residual:
                    dataset/meta/info.json: 36 episodes, 3,305 frames
                    dataset/meta/stats.json: 3,305 frames stats
                    dataset/data/chunk-000/file-000.parquet: CORRUPTED (Parquet magic bytes error)
                    dataset/data/chunk-000/file-001.parquet: NOT WRITTEN
```

### Key Conclusion on Data Provenance

- **ORIGINAL parquet files are GONE.** The Apr 5 interrupted pack overwrote them.
- **We cannot directly verify** what the original parquet files (used for V58 training) contained.
- **However**, we CAN infer from NPZ source files: `gripper_joint` was CONTINUOUS (std=0.0208, 40 unique values).
- **Recommendation**: Regenerate dataset from NPZ source files using corrected code. Do NOT trust current `dataset/` contents.

---

## PART 2 — ROOT CAUSE CORRECTIONS

### P0a (Gripper Joint Frozen): CANCELLED

**Prior claim**: "gripper_joint frozen at -0.042 across all 18,701 frames"
**REVISED**: This claim was NOT verified against the actual training parquet files (they were overwritten). However, NPZ source files confirm `gripper_joint` was CONTINUOUS:

| Source | gripper_joint std | Unique Values | Range |
|--------|-------------------|---------------|-------|
| NPZ raw (seed_007_ep0.npz) | 0.0208 | 40 | [-0.050, +0.014] |
| stats.json (live, 3,305 frames) | 0.0196 | continuous | [-0.042, +0.001] |

**Both sources agree**: gripper_joint is CONTINUOUS, not frozen.
P0a addressed a NON-EXISTENT problem. CANCELLED.

### P0b (Near-White Images): CONFIRMED AS ONE OF MULTIPLE ROOT CAUSES

**Evidence**:
- stats.json (3,305 frames): `image mean=0.977, std=0.0028` — near-white
- NPZ raw (seed_007_ep0.npz): `image mean=248/255=0.973, std=23/255` — bright but with variation
- Root cause: empty `.mtl` files in `sim_exports/urdf/drawerbox/1/assets/` (confirmed by direct inspection)

**IMPORTANT CAVEAT**: MINT 0% is likely MULTI-CAUSE. Near-white images are ONE confirmed root cause, NOT the ONLY root cause.

Remaining plausible factors:
1. **Near-white images** (confirmed — vision encoder gets no useful signal)
2. **State[3:7] format mismatch** — stats.json shows these are JOINT POSITIONS (mean norm ~4.0), not quaternions (norm ~1.0). MINT state tokenizer expects quaternion-like input.
3. **Insufficient training** — 18,701 frames / 1,000 steps for a 3B model is marginal.
4. **Teacher demo quality** — AnyGrasp physics violations in demonstration trajectories.

### Recommendation: Incremental Fix + Small-Scale Validation

**Do NOT do full retrain on potentially still-broken pipeline.**

1. **P0b**: Fix near-white images (add materials to drawerbox MTL files)
2. **Small-scale validation**: 500 frames, 200 steps — verify MINT can overfit
3. **P1a**: Fix state[3:7] format (joint positions → quaternion alignment)
4. **Scale up** if step 2 passes

---

## PART 3 — THE ENVIRONMENT CONTAMINATION PROBLEM

### What Went Wrong

Multiple sessions ran with inconsistent data and made conflicting claims:

1. **Session 1** (pre-V58): Claimed gripper_joint frozen. Code fix applied preemptively.
2. **Session 2** (V58 training): Packed 229/18701. Training used this dataset.
3. **Session 3** (post-V58): Ran "pack_fix" (actually a re-pack). Overwrote dataset/.
4. **Session 4** (prior to current): Made conclusions based on corrupted live data and wrong gripper narrative.
5. **Codex Review**: Caught the contamination. User correctly escalated.

### Lessons for Future Sessions

1. **Never trust live dataset/ when pack artifacts show different counts.**
2. **Timestamp-check all files before drawing conclusions.**
3. **Verify root causes against ORIGINAL pack logs, not residual artifacts.**
4. **Sovereign state must reflect provenance, not just current disk state.**
5. **P0a/P0b fixes must be verified against ACTUAL training data, not inferred.**

---

## SOVEREIGN VERDICT

**Verdict**: `v58_DATA_PROVENANCE_COLLAPSE`
**Phase**: `await_P0b_fix_and_dataset_regeneration`

**Priority 1**: Fix near-white images (P0b) — add materials to drawerbox MTL
**Priority 2**: Regenerate dataset from NPZ source files (P1)
**Priority 3**: Small-scale MINT overfit validation (500 frames, 200 steps)
**Priority 4**: Scale up if step 3 passes

**NOT the blocker**: frozen gripper (was never frozen)
**NOT the blocker**: dataset quantity (18,701 frames were prepared and used)
**The blocker**: data pipeline integrity + multi-cause observation bugs

**Path to MINT success**:
1. Clean dataset from NPZ sources
2. Fix images (P0b)
3. Fix state format (P1a)
4. Validate incrementally
5. Full retrain