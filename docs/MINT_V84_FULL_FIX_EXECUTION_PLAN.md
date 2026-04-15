# MINT v8.4 Full-Fix Execution Plan

**Companion to**: `/mnt/afs2/zhuhaowu/infinigen/docs/MINT_V84_FULL_FIX_SPEC.md`

This plan is the concrete implementation order for the full fix. The ordering matters because later phases depend on earlier correctness repairs.

---

## 1. Execution principles

1. Fix harness correctness before reinterpreting any teacher/readiness result.
2. Fix controller validity before rerunning `T4/T5/T6`.
3. Fix retrain/runtime compatibility before using `B0/B1/B2/B3/B4` as learnability evidence.
4. Keep all work in `REMOTE_LIVE` authority:
   - host: `ssh -p 30017 root@10.210.0.88`
   - repo: `/mnt/afs2/zhuhaowu/infinigen`
   - branch: `feature/mint-env-reformulation-v1-visual-fidelity`

---

## 2. Phase A — Evidence correctness repair

### Objective

Remove harness bugs that currently invalidate part of the evidence.

### Files

- `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_v84_teacher_abstraction_full.py`
- `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_tiny_retrain_confirmation.py`

### Tasks

1. Add explicit zero-safe helper functions.
2. Replace all `value or sentinel` logic where `0` is valid.
3. Add run-scoped artifact validation before cross-phase reuse.
4. Add diagnostic shadow execution path for `B1-B4`.

### Required checks

1. Unit-level grep audit:
   - no remaining `truthful_window_longest_interior_gap ... or 9999`
   - no remaining superiority tie-breaks using falsy zero
2. Phase-level validation:
   - rerun `t3b`
   - rerun `t4`
   - confirm seed `4` gap `0` is preserved as `0`
3. Resume validation:
   - `--resume-from t4`
   - `--resume-from b1`

### Deliverables

- corrected `t3b` comparator semantics
- corrected `t4` truth-gap clause semantics
- `B1-B4` diagnostic shadow execution available

---

## 3. Phase B — Rebuild `P1B`

### Objective

Replace the current weak `P1B` candidate with a real fixed-grasp upper-bound assay.

### Files

- `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/drawer_robot_env_mujoco.py`
- `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_v84_teacher_abstraction_full.py`

### Tasks

1. Replace the current `embodiment_bound_quasistatic` controller logic.
2. Implement explicit quasi-static state variables:
   - `s_t`
   - `s_n`
   - `lock_q`
   - plateau reason
3. Remove reseat from `P1B`.
4. Add telemetry traces for `P1B` controller internals.

### Required checks

1. Run:
   - `python scripts/mint/run_v84_teacher_abstraction_full.py --phase p1a`
   - `python scripts/mint/run_v84_teacher_abstraction_full.py --phase p1b`
2. Compare:
   - seed2/4 `max_drawer_fraction`
   - `phase_locked_rate`
   - `effective_pull_progress_peak`
   - `late_phase_drawer_delta_effective_sum`
3. Only accept `P1B` as a valid frontier candidate if:
   - it is not weaker than `P1A`
   - it is strictly better on at least one hard seed

### Deliverables

- corrected `p1b` artifact
- explicit proof whether the new candidate dominates `p1a`

---

## 4. Phase C — Rebuild `T3-A`

### Objective

Restore a scientifically valid matched world-frame reference.

### Files

- `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/drawer_robot_env_mujoco.py`
- `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_v84_teacher_abstraction_full.py`

### Tasks

1. Implement `t3a_matches_frozen_v83_baseline` clause.
2. Reconstruct `world_frame_matched` so it preserves frozen `v8.3 X2` character.
3. Match all non-opening-law settings with `T3-B`.

### Required checks

Run:

- `python scripts/mint/run_v84_teacher_abstraction_full.py --phase t3a`

Verify:

- seed `2` `step_96_max_drawer_fraction >= 0.54`
- seed `4` `step_96_max_drawer_fraction >= 0.45`

If not, do not interpret `T3-B` superiority yet.

### Deliverables

- faithful `T3-A`
- new fidelity clause in artifact and matrix

---

## 5. Phase D — Rebuild `T3-B`

### Objective

Implement a real interaction-frame hybrid controller.

### Files

- `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/drawer_robot_env_mujoco.py`
- `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_v84_teacher_abstraction_full.py`

### Tasks

1. Add explicit local frame decomposition:
   - `t`, `n`, `b`
2. Add explicit controller subphases:
   - `grasp_seat`
   - `interaction_lock`
   - `hybrid_open`
   - `reseat_once`
   - `hybrid_open_final`
3. Separate tangential progress from preload control.
4. Log controller internals and reasons for plateau / reseat.

### Required checks

Run:

- `python scripts/mint/run_v84_teacher_abstraction_full.py --phase t3b --resume-from t3a`

Verify:

1. `matched_superiority_over_t3a` is computed with zero-safe tie-breaks
2. `phase_locked_rate` is not degraded relative to `T3-A`
3. one of:
   - teacher tier improves, or
   - `max_drawer_fraction` improves while truth does not regress

### Deliverables

- real `T3-B` artifact
- corrected superiority result

---

## 6. Phase E — Recompute `T4/T5/T6`

### Objective

Re-evaluate truth/generalization/readiness from corrected upstream evidence.

### Files

- `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_v84_teacher_abstraction_full.py`

### Tasks

1. Rerun `t4` with zero-safe handling.
2. Rerun `t5` from corrected accepted kernel set.
3. Rerun `t6` after corrected upstream phases.

### Required checks

Run:

- `python scripts/mint/run_v84_teacher_abstraction_full.py --phase t4 --resume-from t3b`
- `python scripts/mint/run_v84_teacher_abstraction_full.py --phase t5 --resume-from t4`
- `python scripts/mint/run_v84_teacher_abstraction_full.py --phase t6 --resume-from t5`

Verify:

- `T4` only fails for real truth/generalization causes
- `T5` counts reflect corrected accepted kernel pool
- `T6` clauses reflect corrected upstream phases, not stale artifacts

### Deliverables

- trustworthy `T4/T5/T6`

---

## 7. Phase F — Retrain/runtime compatibility repair

### Objective

Repair the MINT/PaliGemma stack so training can run diagnostically and authoritatively.

### Files

- `/mnt/afs2/zhuhaowu/infinigen/external/MINT/lerobot_policy_mint/src/lerobot_policy_mint/modeling_mint.py`
- `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_g8_mint_train.py`
- `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_tiny_retrain_confirmation.py`

### Tasks

1. Add a PaliGemma compatibility adapter:
   - top-level API path
   - legacy `.model` path
   - explicit failure mode if neither exists
2. Add checkpoint-layout probe and artifact recording.
3. Preserve run metadata stamping in `g8` summaries.
4. Enable shadow `B1-B4` execution when `T6` is false.

### Required checks

1. Static:
   - `python -m py_compile` on modified files
2. Diagnostic train:
   - `python scripts/mint/run_v84_teacher_abstraction_full.py --phase b0 --resume-from t6`
3. Shadow retrain:
   - `python scripts/mint/run_v84_teacher_abstraction_full.py --phase b1 --resume-from b0`
   - and similarly for `b2/b3/b4`

Verify:

- no `AttributeError ... has no attribute 'model'`
- checkpoint mismatch evidence is written if compatibility is still partial

### Deliverables

- working diagnostic retrain path
- working authoritative retrain path when readiness eventually passes

---

## 8. Phase G — Full authoritative rerun

### Objective

After Phases A-F are complete, rerun the entire `v8.4` evidence program on corrected code.

### Command

```bash
source /root/anaconda3/etc/profile.d/conda.sh
conda activate infinigen
cd /mnt/afs2/zhuhaowu/infinigen
python scripts/mint/run_v84_teacher_abstraction_full.py --phase all
```

### Required outputs

- corrected `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/v84_evidence_matrix.json`
- updated phase artifacts for `t0..t6`, `b0`, and shadow/authoritative `b1..b4`

### Final interpretation tree

1. If `P1B` dominates `P1A` and still fails `0.85/0.90`
   - embodiment-limit evidence becomes strong
2. If `T3-A` matches frozen baseline and `T3-B` wins
   - teacher-abstraction condition is supported
3. If `T4/T5/T6` then pass
   - teacher-readiness condition is supported
4. If retrain stages also run
   - training-side evidence becomes interpretable

---

## 9. Exit criteria for the implementation phase

The implementation phase is complete when all of the following are true:

1. harness zero-value bugs are gone,
2. `P1B` is a valid frontier candidate,
3. `T3-A` passes fidelity,
4. `T3-B` is structurally rebuilt,
5. `T4/T5/T6` rerun on corrected evidence,
6. `B0` no longer crashes on the old PaliGemma API assumption,
7. `B1-B4` can run diagnostically when readiness fails.

Only then should we treat a new full `v8.4` run as the decisive rerun.
