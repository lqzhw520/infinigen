Current authoritative status:
- `P1B` remains frozen as the valid hard-seed frontier: seed2 `0.6899748044108651`, seed4 `0.5679305979991468`.
- `T3A` frozen-reference fidelity remains exact via contract-backed epsilon checks.
- `T3B` Line A rebuild remains active; current reset supply still sits below both `T3A` and the frozen frontier.
- Line B now reaches `dataset_valid = true` with `errors = []`; `teacher_truth_gate`, `truth_contract_path`, and `truth_contract_hash` are now aligned through the active plan, dataset build, and readiness summaries.
- Line C smoke now reaches `image_features_resolved`; the `.model` layout bug and the first image-feature dtype conflict are repaired, while config manual fallback and large checkpoint key mismatch remain active evidence.

# MINT v8.4 Bounded Mainline Execution Spec

**Authority**
- Repo: `/mnt/afs2/zhuhaowu/infinigen`
- Branch: `feature/mint-env-reformulation-v1-visual-fidelity`
- Current sovereign review anchor: `511026e4942ad7446b08978e33e314ee1522b577`
- Current latest validated hard-seed frontier anchor: `468fe31aa1e0f113f8e824e6d352ad16f89cb730`

**Primary companions**
- `/mnt/afs2/zhuhaowu/infinigen/docs/MINT_V84_SYSTEM_AUDIT_2026-04-16.md`
- `/mnt/afs2/zhuhaowu/infinigen/docs/MINT_V84_FULL_FIX_SPEC.md`
- `/mnt/afs2/zhuhaowu/infinigen/docs/MINT_V84_FULL_FIX_EXECUTION_PLAN.md`

---

## 1. Mainline position

The project is no longer centrally blocked by harness existence, orchestration existence, or by the claim that "nothing in MINT runs".

The main upstream bottleneck is now:

> **hard-seed accepted teacher supply under the current regime**
>
> current regime = `truthful measurement + fixed-grasp scripted teacher + hard seeds + current acceptance contract`

The current authoritative interpretation is:

1. `P1A` is a real warm-start plateau.
2. The latest `P1B` is now a valid hard-seed frontier candidate.
3. The latest `P1B` frontier remains below current acceptance.
4. Reset results remain below the current frontier.
5. Downstream readiness remains false, but should be treated as downstream until upstream is repaired.
6. Runtime compatibility is independently broken and must not be conflated with teacher science.

---

## 2. The only three bounded lines of work

Only the following three lines remain justified.

### Line A
One correct `T3B` rebuild whose only scientific question is:

> Can reset teachers on hard seeds approach the frozen hard-seed frontier under the current regime?

### Line B
One truth-contract alignment audit whose only scientific question is:

> Do teacher acceptance, dataset inclusion, train-probe interpretation, and held-out claim use the same truth object family?

### Line C
One runtime compatibility repair whose only engineering question is:

> Can the current MINT/PaliGemma stack load, forward, and take a training step without changing the scientific object?

Anything outside these three lines is off-mainline.

---

## 3. Frozen references and governance rules

### 3.1 Frozen references

Freeze all of the following:
- latest validated `P1B` as the current hard-seed frontier reference,
- frozen `v8.3 X2` reset baseline as the `T3A` reference,
- current acceptance contract (`near = 0.85`, `strict = 0.90`),
- current truth contract family, once Line B freezes it.

### 3.2 Governance rules

1. No new branch.
2. No new evidence family.
3. No new threshold variants.
4. No open-ended `P1B` tweaking.
5. No reading downstream readiness or runtime failures as proof about upstream controller science.
6. Every line answers exactly one question and emits exactly one bounded verdict family.

---

## 4. Repo-level execution order

### Phase 1 — freeze semantics and contracts
1. Create contract files in `docs/contracts/`:
   - `acceptance_contract_v84.json`
   - `truth_contract_v84.json`
   - `runtime_compatibility_contract_v84.json`
   - `teacher_frontier_reference_v84.json`
   - `t3a_frozen_reference_v84.json`
2. Patch `scripts/mint/run_v84_teacher_abstraction_full.py` so every phase artifact records:
   - `truth_contract_hash`
   - `acceptance_contract_hash`
   - frozen frontier reference metadata when applicable

### Phase 2 — Line A
1. Freeze `P1B` as frontier reference.
2. Repair `T3A` fidelity semantics against the frozen reset baseline with epsilon tolerance.
3. Rebuild `T3B` once, correctly, as a real interaction-frame hybrid controller.
4. Measure reset-to-frontier closure.

### Phase 3 — Line B
1. Freeze a single truth contract family.
2. Stamp teacher materialization, dataset validation, train probe, and held-out evaluation with the same truth contract hash.
3. Audit whether they are aligned.

### Phase 4 — Line C
1. Add runtime compatibility adapter and smoke test.
2. Distinguish wrapper-layout failure, checkpoint-layout failure, and forward-contract failure.
3. Only then let full training proceed.

### Phase 5 — integrated rerun
Rerun only after Lines A/B/C have completed their bounded execution.

---

## 5. Allowed final interpretation tree

Only the following final interpretation tree is allowed.

### Case 1
`T3B` still fails to approach the frozen frontier.

Interpretation:
- reset abstraction remains the active upstream bottleneck.

### Case 2
`T3B` approaches the frozen frontier, but the frontier remains below acceptance.

Interpretation:
- regime/acceptance mismatch becomes the main claim-level pressure.

### Case 3
`T3B` approaches the frontier and accepted hard-seed reset teachers reappear.

Interpretation:
- upstream teacher supply is restored enough for readiness to become scientifically meaningful again.

### Case 4
Line B reports truth-contract misalignment and/or Line C reports runtime incompatibility.

Interpretation:
- no downstream trainability or held-out claim conclusion is globally interpretable yet.

---

## 6. Stop rules

1. Stop open-ended controller redesign after one correct `T3B` rebuild.
2. Stop truth-contract work after one aligned-or-misaligned audit result.
3. Stop runtime work after one clean smoke classification and one least-risk repair path.
4. Do not interpret integrated rerun results until all three lines have produced bounded outputs.
