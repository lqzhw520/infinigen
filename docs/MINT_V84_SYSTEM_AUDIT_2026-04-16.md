# MINT v8.4 System Audit (2026-04-16)

**Purpose**: auditable current-state snapshot after vendor freeze, Line A/B completion, and Line C reframing.

**Authority**
- Host: `ssh -p 30017 root@10.210.0.88`
- Repo: `/mnt/afs2/zhuhaowu/infinigen`
- Branch: `feature/mint-env-reformulation-v1-visual-fidelity`
- Canonical vendor baseline: `external/MINT@4eab5795345721001c412ff1ca2c886a11eab606`
- Canonical spec: `/mnt/afs2/zhuhaowu/infinigen/docs/MINT_V84_UNIFIED_EXECUTION_SPEC.md`

## 1. Executive summary

### 1.1 What is already solved in the current round

1. **Line A is solved for the current round.**
   - Hard-seed `T3B` strict teachers were restored in the bounded run.
   - Teacher rollout supply is no longer the current main blocker.

2. **Line B is solved for the current round.**
   - Truth-contract alignment is authoritative.
   - `prepare` is valid and no longer depends on legacy fallback predicate reads.

### 1.2 What has now been cleared

3. **Line C baseline-reproduction governance has been cleared for the current round.**
   - The project is no longer blocked on ambiguous wrapper/env drift.
   - The key fixes are now code-enforced, not just documented:
     - `p1c10` child launches use `/root/anaconda3/envs/mint/bin/python`
     - `p1c7` is marked `diagnostic_only`
     - `g8_runtime_compat_smoke` is marked `preflight_only`
     - `g8_authoritative_baseline_smoke` runs a real one-episode authoritative baseline check
     - `g8_mint_train` requires both gates and launches `/root/anaconda3/envs/mint/bin/lerobot-train` explicitly

Current live result under frozen vendor:
- outer preflight passes
- authoritative one-episode baseline smoke passes with `pc_success = 100.0`
- current `p1c7` in `infinigen` env is correctly classified as `environment_selection_drift` and no longer overrules baseline authority
- current-sovereign `prepare` refreshed active plan / dataset provenance
- current-sovereign one-step `g8` integrated smoke passes through dataset validation, baseline gates, and explicit train launcher execution

So the current active object is no longer baseline reproduction governance.
The next object is full train/probe/eval claim validation on top of the now-aligned baseline contract.

### 1.3 Historical anchor that remains authoritative

We already had strong historical evidence that the original MINT baseline path worked, and we have now reconfirmed it under the frozen vendor baseline using the historical runtime path.

## 2. Vendor status

`external/MINT` is now treated as frozen vendor code.

### 2.1 Canonical vendor commit
- `4eab5795345721001c412ff1ca2c886a11eab606`

### 2.2 Superseded exploratory vendor patches
These commits remain in git history but are no longer canonical:
- `4abe218a62d262ef37a64b986c208a42b0eb7813`
- `137b42d627c308d4fc1cb6b1f84e92a5a7892b74`

## 3. Updated causal conclusion

The project state has shifted again.

### Old active blocker
- outer baseline reproduction / invocation / environment mismatch relative to the already working vendor baseline

### Current active object
- downstream full train/probe/eval claim validation on top of the aligned frozen-vendor MINT baseline contract

This means the correct present-tense question is now:

> **Given that frozen-vendor baseline reproduction is aligned, does the current Infinigen train/probe/eval flow deliver claim-supporting behavior under that contract?**

## 4. Immediate implications

1. do **not** patch `external/MINT`
2. do **not** reopen Line A or Line B as primary blockers
3. do **not** regress the newly aligned Line C gates back to ambient PATH or vendor-patching behavior
4. do use the historical `mint` env + `p1c10` API path as the authoritative baseline reference
5. do treat `p1c7` as a useful wrapper diagnostic, not the sole baseline authority
6. do continue with full train/probe/eval validation on top of the aligned contract


## 5. Current execution plan after Line C hardening

1. Keep the authoritative baseline identity frozen at `p1c10 + mint env + frozen vendor`.
2. Keep `p1c7` diagnostic-only and `g8_runtime_compat_smoke` preflight-only.
3. Keep `g8_authoritative_baseline_smoke` as the hard gate before integrated flow.
4. Continue full train/probe/eval validation using the explicit `/root/anaconda3/envs/mint/bin/lerobot-train` launcher path.
5. Treat any new failure beyond these gates as downstream train/probe/eval claim-validation work, not renewed baseline reproduction confusion.

This plan is intentionally narrow: baseline-governance drift has been cleared, so the remaining work is downstream validation rather than renewed wrapper theory churn.
