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
Active blocker, but now sharply defined.

The current Line C object is:

> Reproduce the historically successful original MINT baseline contract on unmodified `external/MINT@4eab579`, then align Infinigen outside the vendor boundary only.

Current live findings:
- outer preflight on frozen vendor passes
- the current `p1c7` CLI wrapper is **not** authoritative by itself
- `p1c7` under current `infinigen` env fails with `.model` layout drift
- `p1c7` under historical `mint` env fails earlier because the current wrapper sends a CLI argument (`--env.task_ids=[0]`) that the historical `lerobot-eval` does not accept
- the authoritative historical runtime reproduction is instead:
  - frozen vendor `external/MINT@4eab579`
  - historical `mint` env (`python 3.12.13`, `torch 2.7.1+cu126`, `transformers 4.53.3`)
  - `run_p1c10_release_runtime_matched_ab.py --variant-run`
- that authoritative reproduction **passes 3/3 with pc_success=100.0**

Therefore the current Line C blocker is now precisely:
- **our present outer baseline reproduction path is misaligned**
- not a renewed Line A or Line B failure
- not evidence that vendor MINT itself is broken
- not permission to resume patching `external/MINT`

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

> **Keep vendor frozen, preserve solved Line A/B, and repair only the outer baseline reproduction / alignment path until it matches the historically successful MINT contract.**

The current primary question is now:

> **How do we make current Infinigen harnesses reproduce and then consume the already-working frozen-vendor MINT baseline contract, without editing vendor code?**

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
3. Do not reopen `T3B`, truth-contract fallback, or old `P1B` debates while Line C is active.

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

### Step 6 — Integrated rerun gate
Proceed to integrated rerun only if all are true:
1. Line A remains solved
2. Line B remains solved
3. vendor-frozen Line C preflight passes
4. authoritative historical baseline reproduction passes
5. current outer wrapper path is aligned to that same baseline contract

## 5. Decision table

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

## 6. Prohibitions

1. No new edits under `external/MINT`.
2. No new parallel execution specs.
3. No more stale root-level `MINT*.md` accumulation.
4. No reinterpreting Line A or Line B as current blockers unless new contradictory evidence is produced under the current sovereign identity.
5. No vendor-internal runtime shims as the first response to Line C.

## 7. Canonical files for this round

- `/mnt/afs2/zhuhaowu/infinigen/docs/MINT_V84_UNIFIED_EXECUTION_SPEC.md`
- `/mnt/afs2/zhuhaowu/infinigen/docs/MINT_V84_SYSTEM_AUDIT_2026-04-16.md`
- `/mnt/afs2/zhuhaowu/infinigen/docs/contracts/acceptance_contract_v84.json`
- `/mnt/afs2/zhuhaowu/infinigen/docs/contracts/truth_contract_v84.json`
- `/mnt/afs2/zhuhaowu/infinigen/docs/contracts/runtime_compatibility_contract_v84.json`
- `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_g8_runtime_compat_smoke.py`
- `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_p1c7_official_libero_goal_drawer_baseline.py`
- `/mnt/afs2/zhuhaowu/infinigen/scripts/mint/run_p1c10_release_runtime_matched_ab.py`
