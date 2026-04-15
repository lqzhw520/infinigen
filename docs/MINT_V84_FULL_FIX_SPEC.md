# MINT v8.4 Full-Fix Spec

**Status**: Proposed authoritative remediation spec  
**Scope**: `teacher-abstraction sufficient condition` + `teacher-readiness sufficient condition` + `diagnostic/authoritative retrain compatibility`  
**Authority**:
- Host: `ssh -p 30017 root@10.210.0.88`
- Repo: `/mnt/afs2/zhuhaowu/infinigen`
- Branch: `feature/mint-env-reformulation-v1-visual-fidelity`
- Frozen failure baseline head: `89bfed8d23dab432c5964e437324060bcd84b32f`

---

## 1. Objective

This spec defines the **full fix** for the current `v8.4` failure state. It is not a minimal patch and it is not a threshold-relaxation proposal.

The goal is to repair the current `v8.4` stack so that:

1. the evidence program no longer misclassifies valid outcomes because of harness bugs,
2. `P1B` becomes a real fixed-grasp upper-bound assay instead of a weaker controller variant,
3. `T3-A` becomes a faithful matched world-frame baseline instead of a degraded comparison target,
4. `T3-B` becomes a real interaction-frame hybrid controller instead of a world-frame controller with a small normal bias,
5. `T4/T5/T6` become trustworthy downstream gates rather than mixtures of real failures and harness artifacts,
6. the retrain path is repaired so `B0` and shadow `B1-B4` can actually exercise the MINT stack rather than dying on a runtime/API mismatch.

This spec explicitly **does not**:

- relax `near_strict` threshold `max_drawer_fraction >= 0.85`,
- relax strict drawer-open threshold `>= 0.90`,
- reinterpret `P1A` plateau as embodiment proof,
- accept a controller that only "passes enough" without matched evidence.

---

## 2. Current authoritative evidence

The current `v8.4` evidence program completed the teacher/readiness chain and produced:

- `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/v84_evidence_matrix.json`

The key authoritative outcomes are:

### 2.1 `P1A` warm-start baseline plateau

- seed `2`: `max_drawer_fraction = 0.5467456798467271`
- seed `4`: `max_drawer_fraction = 0.450036065101593`
- both runs plateau by `step_96`
- `late_phase_drawer_delta_effective_sum = 0.0`

Interpretation:

> The current baseline warm-start controller reaches a plateau early. Extending horizon alone does not solve the hard seeds.

### 2.2 `P1B` current candidate is weaker than `P1A`

- seed `2`: `0.522453250821915`
- seed `4`: `0.4300410366911081`
- delta vs `P1A`:
  - seed `2`: `-0.0242924290248121`
  - seed `4`: `-0.019995028410484894`
- `phase_locked_rate` drops from `0.421875` to `0.22265625`
- `effective_pull_progress_peak` does not increase
- `late_phase_drawer_delta_effective_sum = 0.0`

Interpretation:

> The current `P1B` candidate does not increase tangential pull authority; it reduces contact stability. It is not a valid embodiment-bound assay.

### 2.3 `T3-A` matched world-frame is too weak

- seed `2`: `0.4455426644977961`
- seed `4`: `0.36702807546891103`
- both are below frozen `v8.3 X2` reset baseline (`~0.54 / ~0.45`)
- both plateau by `step_96`

Interpretation:

> The current `T3-A` is not a faithful matched baseline. It is a degraded comparison target.

### 2.4 `T3-B` current interaction-frame implementation is worse than `T3-A`

- seed `2`: `0.41521106692612914`
- seed `4`: `0.3441386472844368`
- superiority vs `T3-A`: `{'2': -1, '4': -1}`
- both plateau by `step_96`

Interpretation:

> The current `interaction_frame_hybrid` implementation is not a real structural upgrade. It is worse than the world-frame comparison target.

### 2.5 `T4` currently contains a harness false fail

`/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_v84_teacher_abstraction_full.py`

- line using `int(... or 9999)` on `truthful_window_longest_interior_gap`
- when gap is `0`, the expression produces `9999`

This causes:

- false `seed4_truth_gap_le_2 = false`
- incorrect truth-gap ranking in superiority tie-breaks

Interpretation:

> The current `T4` failure is not fully trustworthy. Zero-valued truth gaps are being misread as worst-case.

### 2.6 `T5/T6` are downstream consequences, not primary root causes

`T5` currently reports:

- `accepted_unique_teacher_families_ge_18 = false`
- `strict_unique_teacher_families_ge_8 = false`
- `near_strict_unique_teacher_families_ge_6 = false`

Observed corpus:

- accepted strict teachers only from seeds `1,3,5,6,7,8`
- no hard-seed accepted teachers

Interpretation:

> `T5/T6` are correctly reporting that the corpus is not ready, but the root cause is upstream hard-seed controller failure.

### 2.7 `B0` exposes a real retrain/runtime integration failure

`B0` diagnostic microtrain reached the training layer and failed with:

- `AttributeError: 'PaliGemmaForConditionalGeneration' object has no attribute 'model'`

Additional evidence:

- large state-dict mismatch (`missing`/`unexpected` keys both in the ~600 range)

Interpretation:

> The current MINT/PaliGemma integration is API- and checkpoint-layout-incompatible with the runtime environment.

---

## 3. Root-cause decomposition

The current failure state splits into four root-cause families.

### RC-A. Evidence correctness / harness validity failures

These are engineering failures that make the evidence untrustworthy:

1. zero-valued truth gaps are treated as falsy and replaced by `9999`,
2. run-global artifacts are reused too loosely across phases/runs,
3. downstream retrain shadow execution is still gated too aggressively by readiness.

These failures must be fixed first because they contaminate scientific interpretation.

### RC-B. `P1B` is not an upper-bound assay

The current `embodiment_bound_quasistatic` branch in

- `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/drawer_robot_env_mujoco.py`

changes ladder offsets and speed limits, but it does **not** implement a real quasi-static fixed-grasp upper-bound law. It therefore cannot justify:

- `FIXED_GRASP_ACCEPTANCE_BAR_NOT_REACHED...`

unless it first beats `P1A`.

### RC-C. `T3-A/T3-B` are not a valid matched controller pair

Current problems:

1. `T3-A` is weaker than frozen `v8.3` baseline, so it is not a faithful matched world-frame reference.
2. `T3-B` does not implement a true interaction-frame hybrid controller. It mostly adds a small normal offset on top of world-frame targets.

Therefore the current `matched superiority` result does not answer the intended scientific question.

### RC-D. Retrain/runtime integration is broken independently of teacher quality

The current MINT wrapper expects a `PaliGemma` object layout that is not present in the runtime environment. This is not a teacher RCA. It is an integration RCA in:

- `/mnt/afs2/zhuhaowu/infinigen/external/MINT/lerobot_policy_mint/src/lerobot_policy_mint/modeling_mint.py`

---

## 4. Non-regression rules

The full fix must obey these invariants:

1. **Zero is valid.**
   - `0` is not "missing".
   - No logic may use `value or sentinel` when `0` is a valid measurement.

2. **`P1A` is diagnostic only.**
   - `P1A` measures plateau.
   - It never emits an embodiment-limit conclusion.

3. **`P1B` may issue a strong negative only if it dominates `P1A`.**
   - If it does not beat `P1A`, it stays diagnostic.

4. **`T3-A` must first match frozen `v8.3`.**
   - If it cannot reproduce the old baseline ceiling, `T3-A/T3-B` comparison is invalid.

5. **`T3-B` must differ from `T3-A` only in the opening law.**
   - Not in hidden budget or extra recovery complexity.

6. **`T4/T5/T6` may report downstream failures, but they must not be blamed for upstream controller defects.**

7. **`B0/B1/B2/B3/B4` must support diagnostic shadow execution.**
   - Even if `T6` fails, downstream retrain stages must be able to run in `diagnostic_only` mode.

---

## 5. Required code fixes by file

### 5.1 `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_v84_teacher_abstraction_full.py`

#### A. Zero-safe helpers

Add explicit helpers:

- `_none_or_int(value, default)`
- `_none_or_float(value, default)`

Required replacements:

- `_best_record(...)` truth-gap tie-break
- `_per_seed_summary(...)` best gap extraction
- `_seed_superiority(...)`
- `_run_phase_t4(...)` seed-4 truth-gap clause

Rule:

```python
9999 if value is None else int(value)
```

Never:

```python
int(value or 9999)
```

#### B. `T3-A` fidelity clause

Add a new clause in `T3-A`:

- `t3a_matches_frozen_v83_baseline`

Definition:

- seed `2` `step_96_max_drawer_fraction >= 0.54`
- seed `4` `step_96_max_drawer_fraction >= 0.45`

If this clause fails:

- `T3-A` is still allowed to complete,
- but `T3-B matched_superiority_over_t3a` becomes **scientifically non-final**,
- evidence must explicitly report:
  - `matched_reference_invalid = true`

#### C. `B1-B4` shadow mode

Current behavior:

- `B1/B2/B3/B4` are skipped when `T6` fails

Required behavior:

- always allow `B1/B2/B3/B4` execution,
- mark them as:
  - `diagnostic_only = true`
  - `unsupported_by_readiness_contract = true`
  when `T6` is false.

This is necessary to complete the full evidence tree.

#### D. Run-scoped artifact isolation

Every phase artifact and summary must be linked to the current:

- `run_id`
- `head_commit`
- `branch`

No phase may consume stale global artifacts without confirming matching run metadata.

### 5.2 `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/drawer_robot_env_mujoco.py`

This file requires the core controller reconstruction.

#### A. Rebuild `P1B` as a real quasi-static upper-bound assay

Current implementation is not acceptable because it does not outperform `P1A`.

`P1B` must become:

- fixed-grasp only,
- warm-start only (`attached_phase_locked`),
- no reseat,
- no family sweep,
- quasi-static tangential/preload/orientation controller.

State variables:

- `s_t`: tangential progress along drawer axis
- `s_n`: normal preload amount
- `lock_q`: orientation lock state
- `stall_count_low_slip`
- `stall_count_high_slip`

Control law:

1. If `phase_locked` and `grasp_slip_norm <= 0.55` and `drawer_delta_effective > eps`
   - increment `s_t`
2. If `phase_locked` and `grasp_slip_norm > 0.55`
   - hold `s_t`
   - increase `s_n`
3. If low slip but no progress for `k` steps
   - slightly increase tangential increment size
4. If high slip and no progress for `k` steps
   - declare `fixed_grasp_plateau`

`P1B` must log:

- `p1b_s_t_trace`
- `p1b_s_n_trace`
- `p1b_lock_q_trace`
- `p1b_plateau_reason`

#### B. Rebuild `T3-A` as a faithful matched world-frame baseline

`T3-A` must not invent a weaker baseline. It must preserve the frozen `v8.3 X2` world-frame opening character while matching `T3-B` on:

- reset start,
- `max_steps = 144`,
- same settle budget,
- same reseat budget,
- same logging,
- same `step_96` checkpoint.

Only the opening law may remain world-frame:

```text
target = handle + axis * s_t + z_offset
```

#### C. Rebuild `T3-B` as a real interaction-frame hybrid controller

The current implementation is insufficient because it only adds small local normal offsets.

`T3-B` must explicitly decompose control into:

- tangential advance `t`
- preload normal `n`
- optional binormal stabilization `b`

Local frame:

- `t = drawer motion axis`
- `n = handle-contact/preload normal projected orthogonal to t`
- `b = t x n`

Target form:

```text
x_target = handle + s_t * t + s_n * n + s_b * b + z_offset
```

Required phases:

- `pregrasp`
- `contact`
- `close`
- `grasp_seat`
- `interaction_lock`
- `hybrid_open`
- `reseat_once`
- `hybrid_open_final`
- `retreat`

Required controller behavior:

1. `grasp_seat`
   - build stable contact without tangential growth
2. `interaction_lock`
   - require sustained `phase_locked`
   - stabilize orientation and preload
3. `hybrid_open`
   - grow `s_t` only under stable lock and low/moderate slip
   - grow `s_n` when slip rises
   - allow only one `reseat_once`
4. `hybrid_open_final`
   - final monotone attempt after reseat

Required telemetry:

- `interaction_s_t_trace`
- `interaction_s_n_trace`
- `interaction_s_b_trace`
- `interaction_lock_score_trace`
- `controller_subphase_trace`
- `reseat_reason`

### 5.3 `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_tiny_retrain_confirmation.py`

This file already supports diagnostic-only stages, but must be extended to support **shadow execution** for all retrain stages.

Required changes:

1. `_run_stage(...)` remains reusable for both authoritative and diagnostic execution.
2. Add a loop runner that can execute:
   - `B1`
   - `B2`
   - `B3`
   - `B4`
   even when readiness is false.
3. All stage summaries must include:
   - `diagnostic_only`
   - `require_teacher_readiness`
   - `unsupported_by_readiness_contract`
   - `upstream_teacher_verdict`

### 5.4 `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_g8_mint_train.py`

The run metadata stamping fix is already in place and must be preserved.

Additionally, train summaries must also record:

- `transformers_version`
- `torch_version`
- `mint_model_class`
- `paligemma_runtime_layout`

This ensures future runtime mismatches are diagnosable from artifacts.

### 5.5 `/mnt/afs2/zhuhaowu/infinigen/external/MINT/lerobot_policy_mint/src/lerobot_policy_mint/modeling_mint.py`

This file requires a compatibility adapter, not a one-line hack.

Current fragile assumptions include:

- `self.paligemma.model.get_image_features(...)`
- `paligemma.model.language_model.rotary_emb(...)`

Required fix:

#### A. Runtime compatibility adapter

Introduce wrapper helpers:

- `_paligemma_language_model(module)`
- `_paligemma_get_image_features(module, image)`
- `_paligemma_rotary_emb(module, dummy_tensor, position_ids)`

Resolution order:

1. use modern top-level API if present,
2. else use legacy `.model.*` path,
3. else raise an explicit compatibility error with detected object layout.

#### B. Checkpoint compatibility gate

Before normal training, probe checkpoint compatibility and record:

- missing key count,
- unexpected key count,
- first `N` mismatched prefixes,
- resolved model/config versions.

If mismatch exceeds tolerance:

- stage may continue diagnostically,
- but evidence must say:
  - `checkpoint_layout_incompatible = true`

This prevents silent crashes masquerading as train failures.

---

## 6. Corrected controller specs

### 6.1 `P1B` corrected embodiment-bound assay

This phase is not a repair controller. It is a **bounded upper-bound probe**.

#### Inputs

- warm start: `attached_phase_locked`
- `max_steps = 192` (optionally `256` for sensitivity check)
- fixed grasp
- no reseat

#### Success interpretation

`P1B` may only support strong embodiment conclusions if:

1. it is not weaker than `P1A`,
2. it is strictly stronger on at least one hard seed,
3. it still fails to reach `0.85 / 0.90`.

Then and only then may failure be interpreted as meaningful fixed-grasp bound evidence.

### 6.2 `T3-A` corrected matched world-frame baseline

This phase exists to answer:

> Can a faithful world-frame baseline repair hard seeds from reset under the same budgets and phase structure as the new controller?

It must first satisfy `t3a_matches_frozen_v83_baseline`.

### 6.3 `T3-B` corrected interaction-frame hybrid controller

This phase exists to answer:

> Does a true interaction-frame hybrid controller outperform the matched world-frame baseline without changing the acceptance contract?

This answer is valid only if:

- `T3-A` fidelity clause holds,
- `T4` is zero-safe and run-scoped,
- the superiority comparator is zero-safe.

---

## 7. `T4/T5/T6` after the full fix

### 7.1 `T4`

After zero-safe fixes, `T4` becomes a legitimate downstream truth/generalization gate.

Expected interpretation:

- if `T4` still fails, it is a real controller-truth failure,
- if `T4` passes, current hard-seed controller no longer causes measurement regression.

### 7.2 `T5`

`T5` must remain strict. It is not solved by code tricks.

The only valid way to satisfy `T5` is:

- produce hard-seed accepted deterministic kernels,
- then diversify from accepted kernels only.

### 7.3 `T6`

`T6` should continue to serve as a clause-by-clause readiness summary.

It is not separately "fixed"; it becomes satisfiable only after:

- harness correctness,
- real `P1B`,
- faithful `T3-A`,
- true `T3-B`,
- successful `T4/T5`.

---

## 8. B-stage execution semantics after the full fix

The final `v8.4` evidence program must support two retrain paths.

### 8.1 Diagnostic shadow retrain

Always runnable:

- `B0`
- `B1`
- `B2`
- `B3`
- `B4`

when `T6` is false, with:

- `diagnostic_only = true`
- `unsupported_by_readiness_contract = true`

### 8.2 Authoritative retrain

Runnable only when `T6` is true.

This separation lets the evidence tree finish even if teacher readiness is not established.

---

## 9. Scientific interpretation rules

To avoid repeating the same category of mistakes, the evidence program must obey these interpretation rules:

1. `P1A` plateau is not embodiment proof.
2. `P1B` failure is not embodiment proof unless it dominates `P1A`.
3. `T3-B` superiority is not scientifically meaningful if `T3-A` is not a faithful baseline.
4. `T4` truth-gap conclusions are invalid if zero-safe handling is broken.
5. `T5/T6` deficiencies caused by missing hard-seed kernels must be attributed upstream, not to diversity logic itself.
6. `B0/B1/B2/B3/B4` training crashes caused by runtime incompatibility must be attributed to retrain integration, not teacher corpus quality.

---

## 10. Definition of done

The `v8.4` full fix is complete only when all of the following are true:

1. zero-safe evidence handling is implemented and validated,
2. `P1B` is a valid upper-bound candidate assay,
3. `T3-A` passes fidelity-to-frozen-baseline,
4. `T3-B` is a true interaction-frame hybrid controller with explicit tangential/preload/orientation decomposition,
5. `T4` is run on corrected truth-gap handling,
6. `T5/T6` are recomputed from corrected upstream evidence,
7. `B0` no longer crashes on PaliGemma API/layout mismatch,
8. `B1-B4` support diagnostic shadow execution even when readiness fails.

Until then, current `v8.4` should be treated as an **intermediate evidence run**, not a completed full fix.
