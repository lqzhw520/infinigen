# MINT v8.4 Unified Execution Spec

**Purpose**: single canonical execution spec for the current `v8.4` mainline.

**Authority**
- Repo: `/mnt/afs2/zhuhaowu/infinigen`
- Branch: `feature/mint-env-reformulation-v1-visual-fidelity`
- Canonical vendor baseline: `external/MINT@4eab5795345721001c412ff1ca2c886a11eab606`
- Live top-level sovereign head must be read from git at runtime; this document governs execution order, not a frozen top-level commit id.

**Only active companion audit**
- `/mnt/afs2/zhuhaowu/infinigen/docs/MINT_V84_SYSTEM_AUDIT_2026-04-16.md`

## 0. Canonical stance

1. `external/MINT` is now treated as a **frozen vendor**.
2. The exploratory vendor commits `4abe218a62d262ef37a64b986c208a42b0eb7813` and `137b42d627c308d4fc1cb6b1f84e92a5a7892b74` are **superseded experiments**, not part of the canonical path.
3. Historical truth must remain anchored to the already verified success path.
4. No new edits are allowed under `external/MINT` for this round.
5. All remaining adaptation must happen in the outer Infinigen harness, data path, wrapper layer, environment selection, or invocation contract.

## 1. Current state

### 1.1 Step 1 / Line A
Completed.

`T3B` is no longer the active blocker. Hard-seed strict teachers were restored in the authoritative bounded run. The old diagnosis that hard-seed teacher supply is impossible under the current regime is no longer the active verdict.

### 1.2 Step 2 / Line B
Completed.

Truth-contract alignment is no longer the active blocker. The `teacher_truth_predicate` path is authoritative, `prepare` is valid, and schema fallback behavior has been closed.

### 1.3 Step 3 / Line C
Completed for the current round as a code-enforced baseline-reproduction governance pass.

The current Line C object was:

> Reproduce the historically successful original MINT baseline contract on unmodified `external/MINT@4eab579`, then align Infinigen outside the vendor boundary only.

This has now been enforced in code and verified on the current sovereign:
- `run_g8_runtime_compat_smoke.py` is explicitly `preflight_only`
- `run_p1c10_release_runtime_matched_ab.py` now launches `--variant-run` children with `/root/anaconda3/envs/mint/bin/python`, not ambient `sys.executable`
- `run_p1c7_official_libero_goal_drawer_baseline.py` is explicitly `diagnostic_only` and cannot override the authoritative verdict
- `run_g8_authoritative_baseline_smoke.py` now runs a real one-episode authoritative baseline sanity on frozen vendor + historical `mint` env
- that authoritative smoke passes with `pc_success=100.0`
- `run_g8_mint_train.py` now requires both preflight and authoritative baseline smoke before training, and launches `lerobot-train` from `/root/anaconda3/envs/mint/bin/lerobot-train`
- a current-sovereign one-step integrated `g8` smoke passes through dataset validation, preflight, authoritative baseline smoke, and train launcher execution

Therefore Line C is no longer an unresolved wrapper/env drift blocker for this round.
A current-head downstream readiness sync has also now been rerun using `interaction_frame_hybrid` canonical teacher materialization. That sync produced a valid dataset on the current sovereign (`96` accepted strict rollouts across seeds `1..8`, `5196` effective frames), but `teacher_readiness_passed` remains `false`. The remaining failed clauses are `accepted_unique_teacher_families_ge_18` and `near_strict_unique_teacher_families_ge_6`.
The next active object is therefore not immediate authoritative full train/probe/eval claim validation; it is downstream readiness resolution, with full-train runs treated as authoritative only if readiness is raised to `true`, and otherwise as diagnostic-only.

## 2. Historical anchor that must not be forgotten

The following are already verified historical facts and remain authoritative:

1. Official MINT baseline on LIBERO drawer succeeded.
2. Historical training on the original path completed successfully.
3. We have previously seen baseline success in generated outputs and recorded it in:
   - `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/archive/historical_outputs_root/p1i_infinigen_mint_alignment_solution_plan.md`
4. We have now re-verified baseline success on current host using:
   - frozen vendor `4eab579`
   - historical `mint` env
   - historical API path via `run_p1c10_release_runtime_matched_ab.py --variant-run`

## 3. Canonical execution object

The current canonical object is:

> **Keep vendor frozen, preserve solved Line A/B/C baseline-governance work, and use the current-head downstream readiness sync to decide whether the next train/probe/eval run is authoritative or diagnostic.**

The current primary question is now:

> **Given that baseline reproduction governance is aligned and current-head downstream readiness has been refreshed, can teacher readiness be raised to authoritative `true`, or do diversity / near-strict contract clauses remain the blocker for claim-bearing train/probe/eval?**

## 4. Canonical execution order

### Step 0 — Sovereign synchronization
1. Verify repo root, branch, top-level head, and `external/MINT` head.
2. Verify contracts for acceptance, truth, frontier, and runtime.
3. Reject stale artifacts whose recorded sovereign identity does not match current branch/head/vendor baseline.

### Step 1 — Freeze vendor baseline
1. `external/MINT` must remain at `4eab5795345721001c412ff1ca2c886a11eab606`.
2. Do not modify files under `external/MINT`.
3. The old exploratory vendor commits remain visible in git history but are not part of the execution path.

### Step 2 — Preserve completed upstream work
1. Keep Line A results as solved for this round.
2. Keep Line B results as solved for this round.
3. Do not reopen `T3B`, truth-contract fallback, old `P1B` debates, or vendor-runtime patching while downstream validation is active.

### Step 3 — Line C outer preflight
Use `scripts/mint/run_g8_runtime_compat_smoke.py` only as an **outer preflight**. It must verify:
1. frozen vendor head
2. checkpoint and tokenizer paths
3. exact `p1c7` command construction
4. exact `p1c7` environment construction
5. vendor import under baseline PYTHONPATH
6. historical `mint` runtime availability and versions

This smoke must not introspect or patch vendor internals.

### Step 4 — Authoritative historical baseline reproduction
The authoritative reproduction path is:
- historical `mint` env
- frozen vendor `4eab579`
- `run_p1c10_release_runtime_matched_ab.py --variant-run`

`p1c7` remains a useful CLI-wrapper diagnostic, but not the sole authoritative verdict source.

Interpretation:
- If authoritative historical reproduction fails, the blocker is environment/vendor-baseline reproduction mismatch.
- If authoritative historical reproduction passes but current wrappers fail, the blocker is current outer harness / wrapper / invocation drift.

### Step 5 — Outer-shim alignment only
Only after Step 4 is kept green may we continue adapting Infinigen for MINT retrain/eval. All such alignment must stay outside `external/MINT`.

Permitted places:
- `scripts/mint/*`
- top-level contracts
- top-level harness/wrappers
- dataset/materialization/CLI orchestration
- environment-selection / subprocess entry wrappers

Forbidden place:
- `external/MINT/*`

### Step 6 — Downstream readiness sync
1. Re-materialize canonical train rollouts on the current sovereign head using the current solved `interaction_frame_hybrid` teacher controller.
2. Rebuild the canonical dataset and rewrite the downstream readiness contract on that same sovereign identity.
3. Record the synced result as authoritative only if `run_instance_id`, `working_head_commit`, and contract hashes all match the current head.

### Step 7 — Integrated rerun gate
Proceed to authoritative integrated rerun only if all are true:
1. Line A remains solved
2. Line B remains solved
3. vendor-frozen Line C preflight passes
4. authoritative historical baseline reproduction passes
5. current outer wrapper path is aligned to that same baseline contract
6. current-head downstream dataset is valid
7. current-head `teacher_readiness_passed` is true

If items 1-5 pass but item 7 fails, integrated train/probe/eval may still run, but only as `diagnostic_only`.

## 5. Detailed Line C Execution Plan

### 5.1 Authoritative baseline identity
The only authoritative baseline for Line C is now:
- vendor: `external/MINT@4eab5795345721001c412ff1ca2c886a11eab606`
- environment: `/root/anaconda3/envs/mint/bin/python`
- runtime family: `python 3.12.13 / torch 2.7.1+cu126 / transformers 4.53.3`
- invocation path: `run_p1c10_release_runtime_matched_ab.py --variant-run`
- execution primitive: direct `eval_policy_all()` path, not the current `p1c7` CLI wrapper

### 5.2 Non-authoritative paths
The following are diagnostic only:
- `run_p1c7_official_libero_goal_drawer_baseline.py`
- `run_g8_runtime_compat_smoke.py`

They may detect drift, but they may not overrule the authoritative baseline verdict when the authoritative baseline has already been reproduced successfully.

### 5.3 Required outer-wrapper repair tasks

#### Task C1 — Freeze baseline selection in code
Update outer harnesses so they stop inferring the MINT runtime from ambient PATH.
They must explicitly record and use:
- `vendor_commit = 4eab579...`
- `authoritative_python = /root/anaconda3/envs/mint/bin/python`
- `authoritative_entrypoint = run_p1c10_release_runtime_matched_ab.py --variant-run`

Target files:
- `scripts/mint/run_g8_runtime_compat_smoke.py`
- `scripts/mint/run_g8_mint_train.py`
- any current wrapper that still assumes ambient `lerobot-eval`

#### Task C2 — Split diagnostic CLI wrapper from authoritative runtime runner
`p1c7` must be demoted to `diagnostic_only` in semantics.
It should remain useful for checking:
- command construction drift
- PATH drift
- CLI parser drift
But it must not be treated as the final baseline verdict.

The authoritative runner must stay aligned to the `p1c10` API path.
If a dedicated single-purpose baseline runner is created later, it must reuse the same runtime contract as `p1c10`, not invent a third path.

#### Task C3 — Eliminate current wrapper drift
Repair the current outer wrapper so it no longer diverges from the historical successful contract in either of these ways:
- wrong environment selection (`infinigen` env instead of `mint` env for vendor MINT runtime)
- wrong invocation shape (`lerobot-eval` CLI arguments that do not match the historical parser contract)

This means:
- no ambient PATH reliance
- no unconditional `--env.task_ids=[0]` assumption
- no vendor-internal runtime probing as a substitute for baseline reproduction

#### Task C4 — Rewire training/eval gates around the authoritative contract
Any train/eval gate that currently uses Line C should first verify:
1. authoritative vendor baseline is frozen
2. authoritative runtime environment is available
3. authoritative baseline reproduction is green
Only after that may it proceed to Infinigen-specific data or retrain logic.

This gate must live outside vendor code.

#### Task C5 — Only then align Infinigen data consumption
After Tasks C1-C4 are complete, the remaining question becomes:
- can current Infinigen data, wrappers, and train/eval orchestration consume the already-working MINT baseline contract?

Only at this stage is it meaningful to debug batch/materialization/train-eval issues.

### 5.4 Strict execution order
1. Keep Line A frozen as solved.
2. Keep Line B frozen as solved.
3. Keep `external/MINT` frozen at `4eab579`.
4. Preserve `p1c10 + mint env + frozen vendor` as the authoritative baseline contract.
5. Demote `p1c7` to diagnostic-only status in interpretation.
6. Repair current outer wrappers so they delegate to the authoritative contract instead of re-implementing it.
7. Re-run authoritative baseline reproduction.
8. Re-run current wrapper path.
9. Only if both are green, re-enter integrated Infinigen train/eval flow.

### 5.5 Stop rules
- If authoritative baseline reproduction turns red again, stop and debug baseline reproduction only.
- If authoritative baseline stays green while wrapper path stays red, do not touch vendor code; debug wrapper/env drift only.
- If both baseline and wrapper path are green but Infinigen train/eval remains red, classify as outer data/method contract mismatch.

### 5.6 Current round completion status
- `run_g8_runtime_compat_smoke.py`: PASS (`verdict_scope = preflight_only`)
- `run_g8_authoritative_baseline_smoke.py`: PASS (`pc_success = 100.0`, authoritative python matches expected)
- `run_p1c7_official_libero_goal_drawer_baseline.py`: diagnostic-only, current `infinigen` env correctly classified as `environment_selection_drift`
- `run_tiny_retrain_confirmation.py --phase prepare`: refreshed active plan and dataset provenance to current sovereign head
- `run_g8_mint_train.py` minimal smoke (`steps=1`): PASS through explicit authoritative gates and explicit `/root/anaconda3/envs/mint/bin/lerobot-train` launcher

## 6. Decision table

### Case A
- authoritative historical reproduction fails on frozen vendor
- Verdict: **baseline reproduction mismatch**
- Action: fix environment or invocation path outside vendor; do not patch MINT

### Case B
- authoritative historical reproduction passes on frozen vendor, but current wrappers fail
- Verdict: **outer wrapper / invocation drift**
- Action: fix current wrapper and environment selection outside vendor

### Case C
- authoritative historical reproduction passes and outer wrappers align, but Infinigen path still fails
- Verdict: **outer data/method/train-eval contract mismatch**
- Action: fix batch construction, dataset contract, CLI orchestration, or checkpoint selection outside vendor

## 7. Prohibitions

1. No new edits under `external/MINT`.
2. No new parallel execution specs.
3. No more stale root-level `MINT*.md` accumulation.
4. No reinterpreting Line A or Line B as current blockers unless new contradictory evidence is produced under the current sovereign identity.
5. No vendor-internal runtime shims as the first response to Line C.

## 8. Canonical files for this round

- `/mnt/afs2/zhuhaowu/infinigen/docs/MINT_V84_UNIFIED_EXECUTION_SPEC.md`
- `/mnt/afs2/zhuhaowu/infinigen/docs/MINT_V84_SYSTEM_AUDIT_2026-04-16.md`
- `/mnt/afs2/zhuhaowu/infinigen/docs/contracts/acceptance_contract_v84.json`
- `/mnt/afs2/zhuhaowu/infinigen/docs/contracts/truth_contract_v84.json`
- `/mnt/afs2/zhuhaowu/infinigen/docs/contracts/runtime_compatibility_contract_v84.json`
- `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_g8_runtime_compat_smoke.py`
- `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_p1c7_official_libero_goal_drawer_baseline.py`
- `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_p1c10_release_runtime_matched_ab.py`
