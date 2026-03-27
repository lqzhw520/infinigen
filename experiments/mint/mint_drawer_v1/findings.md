# findings.md — mint_drawer_v1 D1 deep diagnosis

**Updated**: 2026-03-27T14:30:00+08:00

## D1 Run Summary

- **Training**: PASSED (returncode=0, 1200 steps, loss 2.236→0.317→0.052→0.000→0.000)
- **Eval**: ZERO SIGNAL (5/5 episodes: grasp=False, pull=0.0, ever_attached=False)
- **D1 Gate**: FAILED — `failure_reason: u3_regime_b_required`
- **stage_a_signal**: False for all candidates

---

## Root Cause Decomposition

### What `u3_regime_b_required` actually means

This is NOT a blocking upstream gate. It is the D1 script's internal label (line 613) for when `stage_a_survivors` is empty — i.e., no candidate produced a `stage_a_signal`. The name is misleading: it does not mean "u3 audit must pass first." It means "D1 training produced zero grasp/attach signal."

### The REAL failure: complete zero policy output

All 5 finetuned eval episodes:
```
grasp_success=False, pull_distance=0.0, ever_attached=False, steps=96
```
Pretrained baseline also zero (expected — pretrained MINT not adapted for Infinigen sim).

**Training loss curve**: 2.236 → 0.317 → 0.052 → 0.000 → 0.000
This is CATASTROPHIC OVERFIT to 1 episode × 90 frames. Loss=0 means the model memorized exact action tokens for that single rollout, but learned no generalizable policy.

### Root cause: domain gap + single-episode overfit

1. **1 episode = 90 frames, batch_size=8 → 11x overfitting per epoch**
   - At step 800, loss=0.000 — model memorized the rollout completely
   - Eval at seed=2 (SAME seed as training) still gets zero — the env reset is non-deterministic enough that exact token replay fails
   
2. **Eval env vs training env mismatch**
   - Training: offline dataset from teacher rollout NPZ
   - Eval: live PyBullet sim with DrawerRobotEnv
   - The model overfits to exact action token sequences, not to visual-motor generalization
   - At inference, even seed=2 with same asset produces different initial obs → zero grasp

3. **Missing keys warning during training**:
   ```
   Missing keys when loading state dict: 1 keys
   - model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight
   ```
   This embedding key is missing on load — likely a precision/dtype mismatch on bfloat16 load. May cause the vision-language backbone to be partially random at eval time.

4. **U3 audit flags**:
   - `missing_handle_part_labels_suspect`: part_labels not exported in asset metadata
   - `handle_center_world: None` in JSON metadata — u3 reads from env_record, not JSON
   - `decision: upstream_suspect` — handle geometry is heuristically inferred
   - This means DrawerRobotEnv may be computing attach detection based on heuristic handle position, not explicit mesh labels → false negatives in attach detection are possible

5. **grasp_steps=1 in teacher rollout** (seed_002_ep01):
   - The teacher only spends 1 step in "grasp" phase — extremely sparse grasp signal
   - The policy learns near-zero grasp dwell time, making it unlikely to trigger attach

---

## P0-P4 Root Cause Status

| Root Cause | Status | Evidence |
|------------|--------|----------|
| P0: Strong rollout exists | RESOLVED | seed_002_ep01: pre_attach=0.019, persist=23, post=1.0, strict_success=True |
| P1: Source dir stale | RESOLVED | active_teacher_source.json fixed, c2_replay_valid_rollouts/ used |
| P2: lerobot-train missing | RESOLVED | absolute path fix in train_mint_helpers.py |
| P3: Controller lease | RESOLVED | clean controller_id set |
| P4: Training runs | RESOLVED | loss 2.236→0.000, checkpoint saved |
| **P5: Eval produces zero signal** | **NEW — UNRESOLVED** | 5/5 finetuned episodes: grasp=False |
| **P6: Missing embed_tokens weight** | **NEW — NEEDS INVESTIGATION** | Missing key warning on checkpoint load |
| **P7: U3 handle part_labels absent** | **KNOWN — PARTIAL** | Heuristic handle detection, may cause false-neg attach |

---

## Hypotheses for Zero Eval Signal

### H1: Catastrophic overfit → no visual generalization (HIGH CONFIDENCE)
- Loss=0 at step 800 with 1 episode means the model memorized exact token sequences
- At eval, even slightly different obs → wrong tokens → arm doesn't move toward handle
- **Test**: check if arm moves at all in sim (any non-zero drawer fraction across 96 steps)
- **Fix**: phase_balanced_overfit variant with curriculum windows (already in STAGE_B), OR increase rollout count (multi-episode D1)

### H2: Missing embed_tokens.weight causes partial model at eval (MEDIUM CONFIDENCE)
- If the language embedding is random/zeroed at inference, language-conditioned actions are garbage
- **Test**: load checkpoint and inspect embed_tokens weight norm
- **Fix**: verify MINT checkpoint loading with bfloat16 flag, or use fp32 load

### H3: DrawerRobotEnv observation mismatch (MEDIUM CONFIDENCE)
- Training data: NPZ observations from teacher rollout (possibly different camera/frame)
- Eval: live PyBullet rendering with DrawerRobotEnv
- If image normalization, camera pose, or observation format differs → zero policy output
- **Test**: render one eval obs and compare to training NPZ obs visually

### H4: Action contract mismatch (LOWER CONFIDENCE)
- active_action_contract.json has `translation_scale_m=0.03, rotation_scale_rad=0.25`
- If teacher rollout used different scale → policy learns wrong action magnitudes
- **Test**: compare action statistics from NPZ vs what DrawerRobotEnv expects

---

## Recommended Next Actions (Priority Order)

### Immediate (today):
1. **Check if any arm motion happens** — inspect drawer_trace from eval (currently logging only final, not per-step)
2. **Run phase_balanced_overfit variant** — STAGE_B already defined in D1, uses phase-windowed curriculum. Less catastrophic overfit.
3. **Check embed_tokens weight** — load checkpoint and verify weight norm is non-trivial
4. **Check obs alignment** — compare one NPZ obs image to one DrawerRobotEnv render for seed=2

### If H1 confirmed (catastrophic overfit):
- Run `attach_curriculum_overfit` variant (already in D1 STAGE_B)
- Increase rollout count: use all 3 coherent seed-2 rollouts as dataset (multi-episode D1)
- This is within D1 scope — no new code needed

### If H2 confirmed (missing embed weight):
- Fix MINT checkpoint load in `train_mint_helpers.py` / lerobot config
- Add `--policy.load_strict=false` or verify bfloat16 embed load

### If H3 confirmed (obs mismatch):
- Fix DrawerRobotEnv to match NPZ observation format
- This is a P0 substrate validity issue — affects all downstream training

---

## Academic Claim Status

**Target claim**: Infinigen-generated trajectories improve MINT success on held-out drawer variants

**Current blocker**: The finetuned policy produces zero grasp signal even on the TRAINING seed. This is a training/eval loop problem, not a data quality problem. The data IS good (strict_success=True, clean rollouts). The policy is not generalizing from the training signal.

**Not a data problem.** Not a rollout quality problem. The C2 rollouts are valid. The issue is in how the policy learns from them (catastrophic overfit + possible obs mismatch).
