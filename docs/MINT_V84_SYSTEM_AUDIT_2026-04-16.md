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

### 1.2 What is now the active blocker

3. **Line C is the only active blocker.**
   - But Line C must now be understood correctly.
   - The active issue is not “keep editing MINT internals until runtime smoke passes”.
   - The active issue is:

> **reproduce the historically successful original MINT baseline on unmodified vendor code, then align Infinigen outside the vendor boundary**

Current live result under frozen vendor:
- outer preflight passes
- current `p1c7` in `infinigen` env fails with `.model` layout drift
- current `p1c7` in historical `mint` env fails because the wrapper sends a CLI argument unsupported by that historical `lerobot-eval`
- authoritative historical runtime reproduction via `run_p1c10_release_runtime_matched_ab.py --variant-run` on frozen vendor + `mint` env succeeds with `pc_success = 100.0`

So the current blocker is now sharper than before:
- **current outer baseline wrapper / environment selection is misaligned**
- vendor MINT itself is not the blocker
- teacher data is not the current blocker

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

The project state has shifted.

### Old active blocker
- hard-seed teacher rollout supply under the current regime

### Current active blocker
- outer baseline reproduction / invocation / environment mismatch relative to the already working vendor baseline

This means the correct present-tense question is no longer “can MINT work?” or “can hard-seed teachers be supplied?”.

The correct question is now:

> **How do we make the current Infinigen harness reproduce and then consume the already-working frozen-vendor MINT baseline contract?**

## 4. Immediate implications

1. do **not** patch `external/MINT`
2. do **not** reopen Line A or Line B as primary blockers
3. do **fix** current wrapper/env drift outside vendor
4. do use the historical `mint` env + `p1c10` API path as the authoritative baseline reference
5. do treat `p1c7` as a useful wrapper diagnostic, not the sole baseline authority
