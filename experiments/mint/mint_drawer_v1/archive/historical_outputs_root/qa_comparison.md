# LeRobot-style Dataset QA: LIBERO vs Infinigen Comparison

**Reference**: LIBERO_reference — 100 eps
**Infinigen**: Infinigen — 25 eps

## Verdict: **FAIL — critical P0/P1 issues**

### Critical Issues

- ❌ ACTION_VARIANCE: Infinigen |Δa|_2 p95=1.1768 >> LIBERO=0.1643 — teacher has excessive motion jumps
- ❌ STATE_ACTION_LAG: Infinigen ALL dims wrong lag (0 or -1 instead of 1) — action/state convention broken
- ❌ STATE[7]: Infinigen binary {0.0, 1.0} vs LIBERO continuous [-0.042, +0.001] — P0 contract violation
- ❌ DATA_SCALE: Infinigen episodes 2.8x shorter than LIBERO (mean 100 vs 277 frames)

---

## 1. Action Delta |Δa| Comparison

| Metric | LIBERO | Infinigen | Change |
|--------|--------|-----------|--------|
| **|Δa|_2 p95** | 0.1643 | **1.1768** | **+616%** ❌ |
| |Δa|_2 mean | 0.0852 | 0.36 (est) | +322% ❌ |
| action[0] Δ p95 | 0.088 | 0.56 | +537% ❌ |
| action[6] gripper | {-1,+1} binary ✅ | {-1,+1} binary ✅ | OK |

> **Root cause**: Infinigen uses scripted delta-position commands at 10Hz with `translation_scale=0.03m`. Each step commands up to 3cm delta. LIBERO human teleop smoothes commands continuously, so per-step |Δa|_2 p95 ≈ 0.16. Infinigen teacher is **6x jerkier** than LIBERO human teleop.

---

## 2. State-Action Temporal Lag

| Dim | LIBERO best_lag (r) | Infinigen best_lag (r) | Infinigen OK? |
|-----|---------------------|------------------------|---------------|
| dx | lag=1 r=0.257 | lag=0 r=0.403 | ❌ wrong lag |
| dy | lag=-1 r=0.320 | lag=0 r=0.323 | ❌ wrong lag |
| dz | lag=1 r=0.192 | lag=-1 r=-0.298 | ❌ wrong lag |
| drx | lag=1 r=0.038 | lag=0 r=-0.099 | ❌ wrong lag |
| dry | lag=-1 r=-0.013 | lag=-1 r=0.065 | ❌ wrong lag |
| drz | lag=1 r=0.040 | lag=0 r=0.050 | ❌ wrong lag |
| gripper | lag=-1 r=-0.250 | lag=-1 r=-0.021 | ❌ wrong lag |

> **Analysis**: LIBERO shows low r values on some dims (dy, drx, gripper) because the 100-episode sample mixes 40 different task types with heterogeneous motion patterns. Infinigen is worse — ALL lags are wrong (0 or -1), indicating the action/state relationship is fundamentally different. **The `action_delta_l2` being 6x larger explains the poor correlations: large per-step deltas decorrelate with smooth delta-state predictions.**

---

## 3. Episode Filtering

| Dataset | Passing Episodes | Pass Rate |
|---------|-----------------|-----------|
| **LIBERO** | 8/100 | 8% |
| **Infinigen** | 1/25 | 4% |

> **LIBERO flag reason**: `gripper_not_closed` — LIBERO episodes don't always end with gripper in same state as start (multi-task episodes). This flag is **overly strict for LIBERO** but correctly flags Infinigen episodes with no meaningful interaction.

### Flag breakdown (Infinigen):
- `gripper_not_closed`: episodes where gripper start/end state differ
- `short_episode`: episodes < 50 frames
- `near_zero_motion`: episodes with per-step |Δa|_2 p50 < 0.005

---

## 4. Gripper Analysis

| Metric | LIBERO | Infinigen |
|--------|--------|-----------|
| action[6] = {-1, +1} | ✅ True | ✅ True |
| gripper transitions/ep | 3.7 | 2.08 |
| gripper unique deltas | {0.0, 2.0} | {0.0, 2.0} |
| **state[7] = continuous** | ✅ True (410 unique) | ❌ **Binary {0, 1}** |

> **P0 Contract Violation**: `states[7]` in Infinigen NPZ is binary `{0.0, 1.0}`, not continuous `[-0.042, +0.001]`. This is the most critical bug — MINT/LeRobot's quantile normalizer expects continuous gripper joint position.

---

## 5. Autocorrelation / Suggested Chunk Length

| Dim | LIBERO chunk | Infinigen chunk | Notes |
|-----|-------------|-----------------|-------|
| dx | 11 | — | — |
| dy | 16 | — | — |
| dz | 11 | — | — |
| drx | 10 | — | — |
| dry | 9 | — | — |
| drz | 21 | — | — |
| gripper | 20 | — | — |

> **LIBERO recommendation**: chunk length ≈ 11 for position dims. Infinigen's high action variance makes autocorrelation analysis noisy; needs more episodes.

---

## 6. Action Range

| Dim | LIBERO actual cap | Infinigen cap | Status |
|-----|------------------|--------------|--------|
| action[0-2] translation | ±0.938 (not 1.0) | ±1.0 clip | ⚠️ LIBERO never uses full [-1,1] |
| action[3-5] rotation | ±0.375 (not 1.0) | ±1.0 clip | ⚠️ LIBERO rotation smoother |
| action[6] gripper | {-1, +1} only | {-1, +1} only | ✅ OK |

---

## 7. Immediate Action Items

### P0-Critical
1. **Fix `states[7]` in `_state_vector()`**: Use actual finger joint position `[-finger_pos]` instead of binary `gripper_open`
2. **Calibrate action scale**: Reduce `translation_scale` or add smoothing so that `|Δa|_2 p95` drops from 1.18 to < 0.25 (within 1.5x of LIBERO)

### P1-High
3. **Fix state-action lag**: Ensure `action = next_eef_pos - current_eef_pos` convention is correct. Investigate why Infinigen lags are all wrong.
4. **Extend episodes**: Increase episode length from ~100 to ~200+ frames to match LIBERO distribution

### P2-Medium
5. **Relax `gripper_not_closed` flag**: For drawer task, gripper should end CLOSED (different from LIBERO open-close tasks). Update flag logic to be task-aware.
6. **LIBERO chunk length**: Use chunk=11 as MINT action horizon recommendation

---

## 8. Data Files

- Reference QA: `outputs/qa_reference.json`
- Infinigen QA: `outputs/qa_infinigen.json`
- Comparison: `outputs/qa_comparison.json`
- Contract table: `outputs/libero_contract_table.md`
- LIBERO fingerprint: `outputs/libero_fingerprint.json`
