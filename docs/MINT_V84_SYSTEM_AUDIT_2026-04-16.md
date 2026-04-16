# MINT v8.4 System Audit (2026-04-16)

**Purpose**: auditable current-state summary after the v8.4 evidence-program iterations.

**Authority**:
- Host: `ssh -p 30017 root@10.210.0.88`
- Repo: `/mnt/afs2/zhuhaowu/infinigen`
- Branch: `feature/mint-env-reformulation-v1-visual-fidelity`
- Current sovereign head: `fea75fb436b40bd5f854d7a74063b8ecff663cd4`
- Current `external/MINT` runtime adapter head: `137b42d627c308d4fc1cb6b1f84e92a5a7892b74`
- Current authoritative Line A run id: `v84_20260416T090534Z_fea75fb4_66017839`

**Canonical companion spec**:
- `/mnt/afs2/zhuhaowu/infinigen/docs/MINT_V84_UNIFIED_EXECUTION_SPEC.md`

---

## 1. Executive summary

## 1A. Latest authoritative causal update

After the most recent bounded iterations, the current state is:

1. **Line B is materially aligned.**
   - `run_tiny_retrain_confirmation.py --phase prepare` now yields `dataset_valid = true` and `errors = []`.
   - `teacher_truth_gate`, `truth_contract_path`, and `truth_contract_hash` are now aligned through the active tiny-retrain plan, dataset provenance, and readiness summaries.

2. **Line C has crossed the first-forward barrier.**
   - `run_g8_runtime_compat_smoke.py` now reaches `stage = image_features_resolved`.
   - The `.model` runtime-layout failure and the first vision-stack dtype mismatch are repaired.
   - Remaining Line C risk is now narrower: config manual fallback plus large checkpoint `missing/unexpected keys`.

3. **Line A has improved, but the remaining causal bottleneck is now sharper.**
   - Current `T3B` reached: seed2 `0.4646884555833809`, seed4 `0.3862671700179796`.
   - This materially improves over the earlier rebuilt `T3B` (`0.3811 / 0.3157`) and increases `attach_persistence` from `35/36` to `64/65`.
   - `hybrid_open_hold` also grows from about `23` steps to `47/48` steps.

4. **The remaining Line A problem is not just weaker per-step opening authority.**
   - In open-phase only, current `T3B` mean step delta is already stronger than `T3A`.
   - The failure is instead a **continuation failure**: after the first partially-open burst, `T3B` cannot reconnect attach/lock and resume a second opening burst.

5. **Therefore the current Line A object is now more precise than `post-open hold / detach recovery`.**
   - The active scientific object is:
     > **reset-to-frontier continuation under partially-open geometry**
   - Or equivalently:
     > **half-open reattach / relock manifold under the current scripted fixed-grasp regime**

This matters because even if Line A is further improved, the higher-level frontier-vs-acceptance pressure remains:
- frozen `P1B` frontier: seed2 `0.6899748044108651`, seed4 `0.5679305979991468`
- current acceptance: near `0.85`, strict `0.90`

So the project is now split into two distinct upstream pressures:
- **Line A pressure**: reset continuation still below frontier
- **Regime pressure**: frontier itself remains below acceptance

The central question is no longer “can the harness run?” or “can baseline MINT complete the test chain?”. Those have already been answered.

The central question is now:

> **Is hard-seed teacher data supply feasible under the current regime?**
>
> Current regime = `truthful measurement + fixed-grasp scripted teacher + hard seeds + current acceptance contract`.

The current evidence supports the following high-confidence statements:

1. The baseline chain is real and operational.
   - Environment, sidecars, strict semantics, artifacts, and train/eval orchestration are not imaginary.
   - The project is not failing because “the line fundamentally cannot run”.

2. The upstream teacher bottleneck is real.
   - The current hardest issue is not retrain hyperparameters or held-out eval.
   - It is the ability of the teacher-generation mechanism to produce accepted reset teachers for hard seeds `2/4`.

3. `P1B` has now crossed an important threshold.
   - Earlier `P1B` implementations were not valid upper-bound assays.
   - The latest `P1B` is now a **valid fixed-grasp frontier candidate** because it strictly dominates `P1A` on both hard seeds.

4. The latest validated `P1B` frontier is still below the current acceptance bar.
   - This does **not** mathematically prove impossibility.
   - But it is strong negative evidence that the current regime may be mismatched to the acceptance contract.

5. Downstream `T3/T5/T6` conclusions must now be treated carefully.
   - Some older artifacts were produced before the latest `P1B` refinement and before key harness fixes.
   - Therefore not every old `FAIL` should be treated as current scientific truth.

6. A separate retrain/runtime integration problem also exists.
   - The MINT/PaliGemma stack currently fails at runtime with an API/layout mismatch.
   - This is independent of teacher RCA and must be tracked separately.

In short:

> We have **not** yet proven the original claim.
> We have **not** yet proven the current regime impossible.
> We **have** shown that the current teacher-data supply mechanism is the causal bottleneck, and that this bottleneck may be fundamentally inconsistent with the current acceptance contract.

---

## 2. What is current and authoritative vs. what is stale

This distinction matters. One major source of confusion over the last week has been mixing current evidence with stale artifacts.

### 2.1 Current and authoritative

These are current enough to use as present-state evidence.

#### `P1A` latest baseline plateau
Artifact:
- `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/v84_phase1a_warmstart_baseline_plateau.json`

Latest values:
- seed `2`: `max_drawer_fraction = 0.5467456798467271`
- seed `4`: `max_drawer_fraction = 0.450036065101593`
- both plateau by `step_96`
- both have `late_phase_drawer_delta_effective_sum = 0.0`

Interpretation:
- The old warm-start baseline stalls early.
- More horizon alone does not solve the hard seeds.

#### `P1B` latest valid frontier candidate
Artifact:
- `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/v84_phase1b_fixed_grasp_embodiment_bound.json`

Latest values:
- seed `2`: `max_drawer_fraction = 0.6899748044108651`
- seed `4`: `max_drawer_fraction = 0.5679305979991468`
- improvements over `P1A`:
  - seed `2`: `+0.14322912456413806`
  - seed `4`: `+0.1178945328975538`
- `candidate_frontier_valid = true`
- failed clauses are only:
  - `both_seeds_ge_085`
  - `at_least_one_seed_ge_090`

Interpretation:
- `P1B` is no longer a weaker controller variant.
- It is now a frontier-valid fixed-grasp candidate.
- But it still does not hit the current acceptance thresholds.

#### Harness correction: `T4` zero-gap bug
Artifact:
- `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/v84_t4_truth_generalization.json`

Current status:
- `PASS`

Interpretation:
- The old “seed4 truth gap fail” was at least partly a harness bug caused by falsy-zero handling.
- `0` is now preserved as `0` instead of being converted to a sentinel.

#### Retrain/runtime failure is real
Artifact:
- `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/g8_train_summary.json`

Observed runtime failure:
- `AttributeError: 'PaliGemmaForConditionalGeneration' object has no attribute 'model'`
- plus a large checkpoint mismatch:
  - ~604 missing keys
  - ~603 unexpected keys

Interpretation:
- This is a genuine runtime/API/checkpoint compatibility problem.
- It should not be conflated with teacher-controller RCA.

### 2.2 Stale or partially stale evidence

These artifacts are still useful historically, but they are **not** the same thing as current integrated truth.

#### `T3A`
Artifact:
- `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/v84_t3a_matched_world_frame.json`

Head in artifact:
- `cdf0157c5427f376c0ce006d959493edb9407846`

Key values:
- seed `2` `step_96 = 0.539918839931488`
- seed `4` `step_96 = 0.44551390409469604`

Interpretation:
- These values are very close to the frozen `v8.3` baseline and are scientifically useful.
- But the current `t3a_matches_frozen_v83_baseline` clause was set too coarsely (`>= 0.54` and `>= 0.45`), so the artifact records `false` even though the seed `4` value is essentially the actual frozen baseline.
- Therefore the current `T3A` failure mixes a real non-acceptance result with a **mis-specified fidelity clause**.

#### `T3B`
Artifact:
- `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/v84_t3b_interaction_frame.json`

Head in artifact:
- `cdf0157c5427f376c0ce006d959493edb9407846`

Interpretation:
- This artifact was produced before the current `P1B` refinement and before the latest evidence interpretation cleanup.
- Its superiority bookkeeping is not trustworthy enough to use as the final statement on the current `T3B` question.
- In particular, it should not be treated as a definitive verdict on whether a properly rebuilt interaction-frame controller can or cannot close the reset-to-frontier gap.

#### `T5/T6`
Artifacts:
- `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/v84_t5_diversity_dedupe.json`
- `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/v84_t6_readiness_contract.json`
- `/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/v84_evidence_matrix.json`

Interpretation:
- These correctly show that readiness is not established.
- But they are still downstream summaries built from upstream phases that have since changed in meaning or quality.
- Therefore they remain valid as “not ready” indicators, but not as the final, current, globally consistent integrated verdict.

---

## 3. Mathematical framing of the bottleneck

Define:

- `A_near = 0.85`
- `A_strict = 0.90`
- `U_s` = best hard-seed fixed-grasp warm-start frontier under the current regime
- `R_s` = best hard-seed reset controller result under the current regime
- `G_s = U_s - R_s` = reset-to-frontier gap

For the latest available values:

- `U_2 = 0.6899748044108651`
- `U_4 = 0.5679305979991468`
- `R_2 ≈ 0.5448020868360974`
- `R_4 ≈ 0.4498273453341148`

Therefore:

- `G_2 = U_2 - R_2 ≈ 0.14517271757476768`
- `G_4 = U_4 - R_4 ≈ 0.11810325266503202`

Acceptance gaps from the current frontier are:

- `A_near - U_2 ≈ 0.16002519558913486`
- `A_near - U_4 ≈ 0.2820694020008532`
- `A_strict - U_2 ≈ 0.2100251955891349`
- `A_strict - U_4 ≈ 0.3320694020008532`

### What this means

1. There is a **real reset gap**.
   - `R_s` is materially below `U_s` for both hard seeds.
   - So a reset-controller mismatch still exists.

2. But even the latest valid frontier candidate is still below the current acceptance bar.
   - So solving the reset gap alone does **not** automatically solve acceptance.

3. Therefore the `T3B` rebuild has a very specific value:
   - it can test whether reset teacher construction can approach the current hard-seed frontier,
   - but by itself it cannot guarantee `0.85 / 0.90` if the current frontier remains where it is.

This is the key scientific distinction that was repeatedly blurred during earlier iterations.

---

## 4. Why this did not surface early enough

This should have surfaced earlier. It did not, for several reasons.

### 4.1 Objective drift

The project drifted from:
- “verify the claim under the MINT baseline test chain”

to:
- “keep repairing one controller variant after another until hard seeds work”.

That drift made process metrics look like progress even when they were not the decisive question.

### 4.2 Process signals were mistaken for claim signals

Things like:
- attach persistence,
- warm-start family pass,
- phase-locked rate,
- variant pass,

were all informative, but they are not the same as:
- accepted hard-seed reset teachers,
- truthful teacher readiness,
- retrainable hard-seed data supply.

### 4.3 Upper-bound assay and candidate assay were not separated cleanly

Early `P1B` variants were weaker than `P1A`, but they were still dangerously close to being overinterpreted.

That is a design mistake:
- an upper-bound assay must first prove that it is a stronger frontier candidate than the baseline,
- otherwise it is just another candidate controller.

### 4.4 The first `T3B` was not a faithful implementation of the intended theory

The literature-supported diagnosis was that contact-phase control should be interaction-frame and hybrid in structure.

But the implementation drifted into:
- world-frame pulling,
- plus a local normal bias.

That is not the same thing.

### 4.5 Harness bugs contaminated scientific interpretation

Examples:
- `truth_gap = 0` being treated as falsy,
- stale cross-run artifacts,
- incomplete resume semantics,
- overgated downstream diagnostic execution.

Those bugs created fake failures and made real failures harder to localize.

---

## 5. Current root-cause tree

The present state decomposes into four major families.

### RC-1. Teacher feasibility under the current regime

Question:
- Can the current regime supply accepted hard-seed truthful teachers at all?

Evidence:
- Latest valid `P1B` still remains below `0.85 / 0.90`.

Status:
- **Not yet falsified in the mathematical sense**,
- but **under substantial negative pressure from current evidence**.

### RC-2. Reset-controller mismatch

Question:
- Even if the frontier is below acceptance, can reset teacher construction at least approach that frontier?

Evidence:
- `R_2/R_4` are still materially below `U_2/U_4`.

Status:
- **Open**.
- This is the real scientific role of the `T3B` rebuild.

### RC-3. Teacher-data sufficiency / readiness

Question:
- Does the teacher corpus satisfy readiness conditions?

Evidence:
- No hard-seed accepted reset teachers,
- insufficient accepted/strict/near-strict family counts.

Status:
- **Currently false**, but mainly as a downstream consequence of RC-1/RC-2.

### RC-4. Retrain/runtime compatibility

Question:
- Even if the teacher corpus were ready, would the current MINT stack train?

Evidence:
- PaliGemma API mismatch,
- checkpoint-layout mismatch,
- training halts at step 0.

Status:
- **Broken independently** of teacher quality.

---

## 6. What is already proven vs. not proven vs. effectively falsified

### 6.1 Already proven

- The baseline test chain works end-to-end.
- `P1A` early plateau is real.
- The current `P1B` is stronger than `P1A`.
- The current acceptance bar is still not reached by the latest `P1B` frontier.
- The retrain/runtime stack has a real incompatibility bug.

### 6.2 Not yet proven

- That the current regime is mathematically impossible.
- That a properly rebuilt `T3B` cannot close much of the reset gap.
- That the acceptance contract must be changed.
- That the original scientific claim is globally false in every neighboring regime.

### 6.3 Effectively falsified or invalidated

- The old claim that early `P1B` was a meaningful embodiment-bound assay.
- The old `T4` interpretation that seed `4` necessarily failed because of truth gap under the corrected harness.
- The earlier implementation claim that the first `T3B` was already a true interaction-frame hybrid controller.

---

## 7. Should we continue experimenting on the “possibly fatal” issue?

### Short answer

**Yes, but only in a bounded, root-cause-aligned way.**

### What should stop immediately

The following should stop:
- open-ended controller tweaking,
- repeated `P1B` redesign now that `P1B` has become a valid frontier candidate,
- interpreting downstream readiness/retrain failures as if they were independent upstream science.

### What should continue

Only three bounded lines of work are justified.

#### Line A — hard-seed teacher feasibility falsification

Goal:
- determine whether the current regime is genuinely incompatible with acceptance, or merely underperforming due to reset-controller mismatch.

This is now a **claim-level** question, not a tuning question.

#### Line B — reset-to-frontier closure

Goal:
- rebuild `T3B` as a real interaction-frame hybrid controller and test whether reset teachers can approach the current frontier.

This is a necessary but not sufficient condition for restoring the teacher supply chain.

#### Line C — regime alignment back to the MINT baseline chain

Goal:
- audit whether the teacher-generation regime is actually aligned with the train/eval claim.

The most suspicious alignment gap is:
- training dataset construction and probe semantics are not obviously identical to the final truthful acceptance regime.

This needs to be treated as a first-class scientific question, not a side note.

---

## 8. What the infinigen data line is really missing

The core issue is not “the simulator is fake”.
The core issue is that the current data line may be missing one or more of the following three contracts.

### 8.1 Measurement contract

We need the same underlying notion of truth to govern:
- teacher acceptance,
- dataset inclusion,
- train-probe interpretation,
- held-out claim.

If training is fed on a different effective truth contract than evaluation, then the model is learning the wrong target.

### 8.2 Teacher mechanism contract

We need a teacher mechanism that can actually produce accepted examples in the regime we care about.

If the mechanism is:
- truthful,
- fixed-grasp,
- scripted,
- hard-seed constrained,

and that mechanism cannot produce enough accepted examples, then no amount of dataset plumbing downstream can compensate.

### 8.3 Runtime contract

Even a perfect teacher corpus is useless if:
- the model wrapper cannot embed images correctly,
- checkpoint loading is incompatible,
- training crashes at step 0.

These are separate contracts and all three have to hold.

---

## 9. The most likely scientific interpretation right now

The strongest current interpretation is:

> The project is not primarily failing because MINT “cannot learn a simple drawer task”.
>
> It is failing because the current teacher-data generation contract for the hardest seeds appears misaligned with the current acceptance contract, while the training stack is independently broken.

That is a much narrower and more actionable diagnosis than saying:
- “the whole project is wrong”, or
- “the simulator is fake”, or
- “we just need another controller tweak”.

---

## 10. Recommended next-step decision rule

Use the following decision rule.

### Step 1
Freeze `P1B` as the current valid hard-seed frontier candidate.

Do **not** keep tweaking it unless a new candidate has a clearly different scientific rationale.

### Step 2
Freeze `T3A` as the precise frozen-baseline reference.

Correct the fidelity clause to reference the actual frozen baseline values with epsilon tolerance instead of the current coarse thresholds.

### Step 3
Rebuild `T3B` once, correctly.

Not “world-frame plus normal bias”, but a true interaction-frame hybrid opening law with explicit:
- tangential progress state,
- preload state,
- lock score,
- one reseat budget,
- matched non-opening-law conditions with `T3A`.

### Step 4
Interpret the outcome using the decomposition below.

#### Case A
If rebuilt `T3B` still cannot approach the current `P1B` frontier:
- the reset-controller abstraction is the active bottleneck.

#### Case B
If rebuilt `T3B` approaches `P1B`, but `P1B` remains below acceptance:
- the regime/acceptance mismatch becomes the main scientific issue.

#### Case C
If rebuilt `T3B` approaches or exceeds the current frontier and accepted hard-seed teachers emerge:
- then the main bottleneck moves downstream to `T5/T6` and retrain/runtime compatibility.

This is the cleanest path back to scientific interpretability.

---

## 11. Questions to send to other agents for battle / corroboration

These are the questions most worth putting in front of other agents.

### Q1. Regime-feasibility question

Given the latest authoritative frontier:
- seed2 `0.68997`
- seed4 `0.56793`

and acceptance:
- near `0.85`
- strict `0.90`

is the current evidence already strong enough to treat the current `truthful + fixed-grasp + scripted + hard-seed` regime as scientifically non-viable, or do we still need one more stronger falsification-grade upper-bound control?

### Q2. Measurement-alignment question

Is the teacher-generation contract actually aligned with the training/evaluation contract?

Specifically:
- Are dataset inclusion and train-probe semantics governed by the same truth notion as final acceptance?
- If not, where is the effective label mismatch introduced?

### Q3. Interaction-frame implementation question

What is the minimal faithful interaction-frame hybrid controller that is justified in a position-controlled MuJoCo setting with only slip/progress/lock proxies and no real force-torque sensor?

What structure is essential, and what would still count as an invalid “world-frame controller with preload bias” imitation?

### Q4. Teacher-paradigm question

If the current fixed-grasp scripted paradigm cannot meet hard-seed acceptance, what is the smallest justified paradigm shift?

Options to assess:
- grasp-family diversification,
- one-shot regrasp,
- demonstration-based local contact policy,
- learned contact-phase refinement,
- acceptance/claim re-scoping.

### Q5. Runtime-compatibility question

For the current MINT/PaliGemma stack, what is the least risky repair?

Options to assess:
- wrapper compatibility adapter,
- version pinning to the checkpoint’s original Transformers layout,
- checkpoint conversion/remapping,
- model-family replacement.

---

## 12. Bottom-line position

The honest current bottom line is:

1. We are **not** done.
2. We are **not** entitled to claim success.
3. We are **not** entitled to claim global impossibility yet.
4. We **are** entitled to say that the project’s central bottleneck is now clearly located:
   - hard-seed teacher data supply under the current regime.
5. The next justified work is:
   - one correct `T3B` rebuild,
   - one alignment audit of the teacher/train/eval truth contract,
   - one runtime compatibility repair.

Anything broader than that risks repeating the same week of local RCA churn.
