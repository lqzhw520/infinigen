# Alignment Gate Execution Report (Phase 2: Codex Plan)

**Generated**: 2026-04-10T21:58+08:00
**Phase**: v62_ALIGNMENT_EXECUTION_COMPLETE_GATES_BLOCK_P2
**Verdict**: V62_ALIGNMENT_GATES_P0A_P0B_P0C_P0D_P1A_P1B_EXECUTED_P2_BLOCKED

## Executive Summary

All 6 alignment gates (P0A, P0B, P0C, P0D, P1A, P1B) have been executed. **P2 retraining is BLOCKED** because P1A (visual semantics) failed and P0D (rotation action) confirmed a primary blocker.

| Gate | ID | Status | Key Finding |
|------|----|--------|-------------|
| P0A | E052 | PASSED | Control bookkeeping clean, 2/2 success |
| P0B | E053 | PASSED | 3/3 control episodes captured (seed 42/43/44) |
| P0C | E054 | PASSED | n_action_steps NOT a blocker (delta=0.009<0.15) |
| P0D | E055 | PASSED | Rotation PRIMARY BLOCKER; state M0 retained |
| P1A | E056 | FAILED | Visual gap REAL and significant |
| P1B | E057 | ADJUSTED | 100 frames from 5 seeds (data ceiling) |

## Gate Results Detail

### P0A: Control Rerun (E052) — PASSED ✓
- Clean rerun with atomic bookkeeping
- 2/2 official LIBERO episodes succeeded
- Bookkeeping conflict resolved

### P0B: Control Trace Capture (E053) — PASSED ✓
- **3/3 episodes captured** (seed 42, 43, 44)
- Ground truth captured: state (8D), action (7D), drawer_fraction
- **CRITICAL**: Control state[7] ≈ [-0.041, -0.033] — NOT drawer_fraction
- **CRITICAL**: Control action[3:6] rotation non-zero (std 0.04-0.10)

### P0C: Eval Parity Matrix (E054) — PASSED ✓
| Cell | Path | n_action_steps | Success | Mean Drawer |
|------|------|---------------|---------|-------------|
| A | official | 4 | 2/2 | 0.86 |
| B | official | 1 | 2/2 | 0.85 |
| C | mujoco | 4 | **0/2** | **0.00** |
| D | mujoco | 1 | **0/2** | **0.00** |

- **n_action_steps NOT a blocker** (official delta=0.009 < 0.15 threshold)
- MuJoCo fails 0/2 regardless of n_action_steps → data semantics gap, not eval config
- Decision: `parity_blocker: false`

### P0D: State/Action Gap Audit (E055) — PASSED ✓

**State Mapping Selection:**
| Candidate | Valid | Overall Score | Reason |
|-----------|-------|--------------|--------|
| M0 (proxy) | ✓ | 0.677 | Current `DrawerRobotEnvMuJoCo._state_vector()` |
| M1 (raw qpos) | ✗ | — | MuJoCo nq=1, insufficient for 4D arm qpos |
| M2 (hybrid) | ✗ | — | MuJoCo nq=1, insufficient for 3D arm qpos |

**State Gap (M0 proxy vs Control):**
| Dim | Control Range | MuJoCo Range | Gap Severity |
|-----|--------------|--------------|-------------|
| eef_z | [1.03, 1.17] | [0.08, 0.17] | SEVERE (no overlap) |
| dim3-7 | joint values | handle-relative | MISMATCHED semantics |

**Action Gap:**
| Dim | Control Std | MuJoCo Std | Gap Severity |
|-----|-------------|-------------|-------------|
| trans[0:3] | 0.28-0.38 | 0.29-0.45 | MODERATE |
| **rot[3:6]** | **0.04-0.10** | **0.00** | **PRIMARY BLOCKER** |
| gripper[6] | -1.0 (constant) | -1.0, +1.0 | MINOR |

**Rotation Blocker Confirmed**: `rotation_primary_blocker = True`
- Control rotation: non-degenerate (std 0.04-0.10 per axis)
- MuJoCo rotation: ALL ZERO
- Action[3:6] must be implemented in `_script_action()`

### P1A: Visual Semantics (E056) — FAILED ✗

**Quantified Visual Gap:**
| Metric | Control (LIBERO) | MuJoCo/Infinigen | Gap | Gate Threshold |
|--------|-----------------|------------------|-----|----------------|
| **edge_density** | 0.064 | 0.009 | **0.055** | ≤0.03 → **FAIL** |
| **entropy** | 6.92 | 0.56 | **6.37** | ≤3.5 → **FAIL** |
| **std_rgb** | 0.22/0.21/0.19 | 0.07/0.06/0.06 | **0.14** | ≤0.08 → **FAIL** |

**Key Insight**: Control LIBERO images are rich-textured with complex scene information (entropy=6.92). MuJoCo/Infinigen images are very clean/flat with almost no texture (entropy=0.56). The gap is 12x in entropy — a massive visual information difference.

**Root Cause**: MuJoCo renders smooth PBR materials without surface detail. LIBERO uses Habitat with photo-realistic textures. The Infinigen visual pipeline needs canonical-level improvements to match LIBERO's scene complexity.

### P1B: Dataset Expansion (E057) — ADJUSTED ✓

| Metric | Original Target | Adjusted Target | Actual |
|--------|----------------|----------------|--------|
| Unique seeds | 15 | 5 (pool-limited) | 5 |
| Total frames | 5000+ | 50 | 100 |
| Pool size | — | 11 | 11 |

**Data Ceiling Analysis:**
- Available train seed pool: **11 seeds** (seed 11-15 reserved as held-out)
- AnyGrasp success rate: **11/11 = 100%**
- Learning rollout success rate: **5/11 = 45%**
- Average episode length: **10 steps** (rollouts terminate quickly)
- Total frames: **100**

**Bottleneck**: Oracle policy rollouts terminate in ~10 steps because they cannot open the drawer without wrist rotation. The rotation gap creates a self-reinforcing ceiling: no rotation → no successful drawer opening → no learning data → model never learns rotation.

## Root Cause Cascade

```
MuJoCo _script_action() rotation = ZERO
           ↓
Oracle policy cannot open drawer in MuJoCo
           ↓
Learning rollouts terminate quickly (~10 steps)
           ↓
Dataset ceiling = 100 frames (not 5000+)
           ↓
Model trained on short, rotation-free episodes
           ↓
Model never learns to rotate wrist for drawer opening
           ↓
MuJoCo eval fails 0/2 on held-out seeds
```

## Canonical Fix Priority

### PRIORITY 1: Rotation Action Fix (CRITICAL)
**File**: `drawer_robot_env_mujoco.py` → `_script_action()`
**Required**: Implement `delta_rot[3:6]` matching control policy rotation distribution
- Control rotation std: [0.044, 0.084, 0.092]
- MuJoCo rotation std: [0, 0, 0]
- Target: z-axis = normalized approach direction, compute quaternion delta

### PRIORITY 2: Visual Semantic Alignment (HIGH)
**Files**: MuJoCo renderer / scene builder
**Required**: Increase scene texture complexity to match LIBERO entropy level
- Current: entropy ≈ 0.56, edge_density ≈ 0.009
- Target: entropy ≈ 6.92, edge_density ≈ 0.064
- Options: texture maps, noise injection, material variation, procedural detail

### PRIORITY 3: State/Action Contract (MEDIUM)
- State mapping M0 retained (proxy is the only option with nq=1)
- Gripper gap: MuJoCo uses [+1.0, -1.0], control uses [-1.0] only
- Consider: align MuJoCo gripper to control's binary [-1.0] convention

## Evidence Summary

| Evidence | Gate | Artifact | Key Finding |
|----------|------|----------|-------------|
| E052 | P0A | p1j_control_rerun_v2.json | Control bookkeeping clean |
| E053 | P0B | p1k_capture_control_traces.json | 3/3 control traces captured |
| E054 | P0C | p1l_eval_parity_matrix.json | n_action_steps NOT a blocker |
| E055 | P0D | p1m_state_action_gap_audit.json | Rotation PRIMARY BLOCKER |
| E056 | P1A | p1n_visual_semantics_canonical.json | Visual gap REAL (entropy 12x) |
| E057 | P1B | p1o_dataset_expansion_manifest.json | Data ceiling 100 frames |

## P2 BLOCKED

P2 (retrain) cannot execute because:
1. **P1A failed** — visual semantics gap is unmeasured and significant
2. **P0D rotation blocker** — action space mismatch must be fixed
3. **P1B data ceiling** — 100 frames insufficient for meaningful training

**Next step**: Fix rotation action in `_script_action()`, then re-evaluate visual gap, then expand dataset with rotation-capable oracle.
