# MINT v8.4 — P0 Infrastructure Repair Spec
## post-v13 Pretrain Admission → P0 Infrastructure Fix

**Purpose**: Define the minimal bounded execution for resolving post-v13 infrastructure blockers before any further canary training or probe is authorized.

**What this spec supersedes**: This spec does NOT supersede the Layer 1/2/3 framework or the v12/v13/v13-slice-2 scientific findings. It supersedes the ambiguous framing of "post-v13 admission repair" by explicitly separating infrastructure bugs (P0) from deeper scientific blockers (P1/P2).

**What this spec does NOT reopen**:
- v12 G1-G6 findings (orientation audit, state conditioning audit, training results)
- v13 G1-G4 findings (state/action interface RCA)
- v13-slice-2 findings (live wiring one-step fit)
- post-v13 admission verdict: `STOP_PRETRAIN_INTERFACE_UNSTABLE`

---

## 1. Authority and Context Freeze

| Field | Value |
|-------|-------|
| Repo | `/mnt/afs2/zhuhaowu/infinigen` |
| Branch | `feature/mint-env-reformulation-v1-visual-fidelity` |
| Working HEAD | `bc802069816e6c81beab48ba1f2141c9f3a264d4` |
| Vendor HEAD | `4eab5795345721001c412ff1ca2c886a11eab606` |
| Truth contract hash | `17ae4d1554899b843c80b064a2c5f3d6331b04dd169819ca5f2b0bfba729a6f6` |
| Acceptance contract hash | `3e63d1c21004dae9a8ded0c966f69112f59e340f7d0aada219eb66ac8facd53c` |
| Run instance | `postv13c_20260420T140833Z_bc802069_e2c27eee` |

**Active authoritative gate artifacts (post-v13 closure)**:
- `experiments/mint/mint_drawer_v1/autopilot/gates_post_v13_closure/C0_authority_refreeze.json` — PASS
- `experiments/mint/mint_drawer_v1/autopilot/gates_post_v13_closure/C1_live_support_expansion.json` — STOP (selected_unique_live_episode_count=6, seed 2/4 no support)
- `experiments/mint/mint_drawer_v1/autopilot/gates_post_v13_closure/C2_live_diagnostic_parity.json` — STOP (approach_alignment_cos p95_abs_diff ~1.97, near sign-opposite)
- `experiments/mint/mint_drawer_v1/autopilot/gates_post_v13_closure/C3_signal_band_stability.json` — STOP (orientation action fit unstable across splits, close action fit stable)
- `experiments/mint/mint_drawer_v1/autopilot/gates_post_v13_closure/C4_final_pretrain_closure_verdict.json` — PASS with verdict `STOP_PRETRAIN_INTERFACE_UNSTABLE`

**Active v12.1 findings (reference, not reopened)**:
- `G6_orientation_first_probe.json`: pretrained close_cmd_rate=0.998, finetuned close_cmd_rate=0.456 (fine-tuning suppresses gripper close action by 54%)
- `G2_state_conditioning_audit.json`: orientation_features_consumed_by_model=false; EEF orientation absent from 8D state vector
- Checkpoint sweep: approach_gate Δ=+0.207 (reliable positive); orientation_gate Δ=−0.106 (regression); close_cmd Δ=−0.542 (severe collapse)

---

## 2. The Three-Layer Mechanism Hierarchy

This section defines the explicit boundary that previous specs blurred.

```
Layer 1 — Claim Layer
  "Can MINT learn Infinigen drawer?"
  → NOT YET ANSWERED. Requires P0 + P1 + canary to answer.

Layer 2 — Diagnostic Layer
  "Where does the failure chain break locally?"
  → Answered in v12/v13. C1/C2/C3 are diagnostic findings.

Layer 3 — Admission Layer
  "Is training authorized right now?"
  → Answered: NO. Reason = P0 infrastructure bugs unresolved.

Layer 4 — Infrastructure Layer (NEW, previously implicit)
  "Are there bugs in the evaluation/execution infrastructure?"
  → Answered: YES. C1 = live rollout protocol bug. C2 = feature computation path bug.
```

**The critical distinction this spec enforces**:

> **P0 bugs are infrastructure bugs. P1 blockers are dataset/objective/science problems. These two categories must not be conflated. Conflation causes premature training authorization or misdirected scientific effort.**

| Category | Items | Category Name | Authorization Effect |
|----------|-------|---------------|---------------------|
| **P0** | C1, C2 | Infrastructure bugs | Must be fixed BEFORE C3 re-run; BEFORE canary |
| **P1** | C3, close_cmd collapse | Dataset/objective instability | Must be diagnosed AFTER P0; BEFORE canary |
| **P2** | Paradigm risk | scripted teacher insufficient | Diagnosed only if P0+P1 done AND canary has no attach signal |

---

## 3. P0a — Live Rollout Family Dispatch Audit and Fix

### 3.1 What C1 Proved

C1 scanned 8 seeds × 3 episode_index = 24 live rollouts. Result: **100% deterministic collapse**.

Every seed produced identical fingerprints, identical step counts, and identical final drawer fractions across all three episode_index values:

| Seed | 3 Attempts | 3 Unique Fingerprints | Outcome |
|------|-----------|----------------------|---------|
| 1 | 3/3 successful | **1** unique | final_fraction=0.9031 |
| 2 | 0/3 successful | **1** unique | final_fraction=0.4647 |
| 3 | 3/3 successful | **1** unique | final_fraction=0.9169 |
| 4 | 0/3 successful | **1** unique | final_fraction=0.3863 |
| 5 | 3/3 successful | **1** unique | final_fraction=0.9317 |
| 6 | 3/3 successful | **1** unique | final_fraction=0.9315 |
| 7 | 3/3 successful | **1** unique | final_fraction=0.9077 |
| 8 | 3/3 successful | **1** unique | final_fraction=0.9176 |

Selected unique live episodes: **6** (seeds 1,3,5,6,7,8). Seeds 2 and 4 produced zero successful unique episodes.

### 3.2 Why This Is P0 Infrastructure, Not a Paradigm Failure

**Evidence against paradigm failure**: The diagnostic corpus (v11/v12) already demonstrated teacher family diversity through `teacher_family` switching (not `episode_index`). That corpus has 31 unique teacher families and 6,696 effective frames. This proves the teacher CAN produce behavioral diversity when the correct dispatch mechanism is used.

**Evidence for infrastructure bug**: episode_index produced zero diversity across all seeds. The most parsimonious explanation is that the live rollout runner does not actually dispatch different teacher family variants — it only varies episode_index (which is not an entropy source in the current configuration).

### 3.3 Audit Requirements

The audit must answer **all three** of these questions separately:

**Question A** — Is `episode_index` passed as a parameter to the live rollout runner?
- If NO → Bug: episode_index is not forwarded to teacher controller.
- If YES → Proceed to Question B.

**Question B** — Is `teacher_family_variant` or equivalent passed to the live rollout runner?
- If NO → Bug: the rollout runner has no teacher family dispatch mechanism.
- If YES → Proceed to Question C.

**Question C** — Does changing `teacher_family_variant` produce different effective fingerprints?
- If NO → This escalates to a teacher capacity problem (P2 risk).
- If YES → Infrastructure bug confirmed and fixable.

### 3.4 Fix Requirements

After the audit identifies the bug layer (A/B/C above):

**If bug is in A**: Forward `episode_index` correctly through the rollout runner call stack.

**If bug is in B**: Implement `teacher_family_variant` dispatch in the live rollout runner. This must mirror how the diagnostic corpus achieved family diversity (see `active_tiny_retrain_plan.json` field `teacher_family_grid_version: v11_support_family_grid_v1`).

**If bug is in C**: Diagnose why teacher family variants are not effective. This is P2 territory.

### 3.5 P0a Pass/Fail Criteria

```
P0a PASS if ALL of:
  1. live rollout runner correctly dispatches teacher_family_variant (not just episode_index)
  2. at least 2 distinct effective fingerprints are produced across scanned variants for at least seeds 1,3,5,6,7,8
  3. seeds 2 and 4 are explicitly diagnosed:
     - If they produce successful support under ANY teacher family → report which families work
     - If they produce zero support under ALL teacher families → mark as confirmed hard seeds (P2 territory)

P0a FAIL if:
  - No teacher family dispatch mechanism exists in live rollout runner
  - Teacher family dispatch exists but produces no diversity across scanned variants
  - Audit reveals root cause is in the teacher controller itself (not the dispatch layer)
```

### 3.6 Hard Seeds Explicit Handling

Seeds 2 and 4 have been hard seeds since v11 (effective support families = 1 and 3 respectively in diagnostic corpus). Their failure in C1 is consistent with prior findings.

**They must be handled as follows**:
- P0a audit must identify whether they fail because of: (a) no family dispatch, or (b) all families fail.
- If (a): fix dispatch, re-evaluate.
- If (b): document as confirmed hard seeds, do not block P0a PASS.
- The hard seed question does NOT block P0a PASS if at least 6 seeds pass the P0a criteria.

---

## 4. P0b — Feature Computation Path Unification

### 4.1 What C2 Proved

C2 compared live rollout features against diagnostic reconstruction features across 6 episodes. Every continuous feature failed parity:

| Feature | Typical p95_abs_diff | Typical max_abs_diff | Pass Threshold | Severity |
|---------|--------------------|--------------------|--------------|----------|
| `distance_to_handle_norm` | ~0.11–0.12 | ~0.12 | `< 1e-3` | Moderate |
| `approach_alignment_cos` | **~1.97** | **~2.0** | `< 1e-3` | **CRITICAL** |
| `orientation_alignment_cos` | ~0.11 | ~1.99 | `< 1e-3` | High |
| `orientation_error_sin` | ~0.11 | ~0.11 | `< 1e-3` | Moderate |
| `orientation_error_cos` | ~0.11 | ~1.99 | `< 1e-3` | High |
| `attach_eligible_proxy` | 91–98% exact | — | `> 99%` | High |

`attach_eligible_proxy` exact agreement (worst seed: seed 6 at 90.9% = 5/55 frames disagree).

### 4.2 Why This Is P0 Infrastructure, Not Scientific Evidence

For `approach_alignment_cos` (range [-1, +1]): a p95_abs_diff of 1.97 means that at the 95th percentile of frames, live and diagnostic systems give **near-opposite verdicts** on whether the end-effector is approaching the handle along the correct direction. A max_abs_diff of 2.0 means at least some frames give **exactly opposite** values.

This is not a statistical fluctuation. It is not a calibration drift. It is evidence of a **systematic feature definition mismatch** — most likely one or more of:
1. **Sign flip**: one path computes `handle → EEF` and the other computes `EEF → handle`
2. **Frame convention mismatch**: one uses world frame, the other uses handle local frame
3. **Vector direction convention**: approach vector normalized in opposite directions
4. **Anchor point mismatch**: distance computed from gripper tip vs. gripper base vs. palm

### 4.3 Audit Requirements

For each failing feature, the audit must answer:

**Step 1 — Identify the canonical source function.**
Determine which function is the **authoritative** source for each feature in the diagnostic pipeline. This function becomes the canonical source for P0b.

**Step 2 — Compare live path implementation.**
Audit every code path that produces these features in live rollouts. Compare line-by-line against the canonical source function. Document every difference in:
- Input coordinate frames
- Normalization method
- Sign conventions
- Anchor/reference point definitions

**Step 3 — Rank differences by severity.**
- **Severity A** (blocker): Sign flip or frame convention flip → produces near-maximum errors
- **Severity B** (moderate): Normalization method difference → produces moderate errors
- **Severity C** (minor): Numerical precision differences → acceptable

`approach_alignment_cos` near-maximum diff → Severity A. This means at least one code path has a sign or frame convention flip.

### 4.4 Fix Requirements

**Mandatory**: All features listed in section 4.1 must be computed by **one and only one** canonical function. All downstream consumers (live rollout, diagnostic reconstruction, dataset loader, one-step fit evaluator, probe evaluator) must call this function — not re-implement it.

**Forbidden**: No live/diagnostic split in feature computation. No separate implementations for "live" vs "offline evaluation."

**Target features for single-source enforcement**:
```
1. distance_to_handle_norm
2. approach_alignment_cos
3. orientation_alignment_cos
4. orientation_error_sin
5. orientation_error_cos
6. attach_eligible_proxy
```

### 4.5 P0b Pass/Fail Criteria

**Phase 1 — From catastrophic to moderate (MUST pass before Phase 2)**:
```
approach_alignment_cos p95_abs_diff < 0.5
orientation_alignment_cos p95_abs_diff < 0.5
orientation_error_cos p95_abs_diff < 0.5
```
Reason: Current diff is ~1.97-2.0 (near sign-opposite). Reducing to < 0.5 confirms sign/frame flip is fixed. This is a bug fix gate, not a quality gate.

**Phase 2 — From moderate to acceptable**:
```
distance_to_handle_norm p95_abs_diff < 1e-3
approach_alignment_cos p95_abs_diff < 1e-3
orientation_alignment_cos p95_abs_diff < 1e-3
orientation_error_sin p95_abs_diff < 1e-3
orientation_error_cos p95_abs_diff < 1e-3
attach_eligible_proxy exact_agreement_rate > 99%
```
Reason: Once sign/frame bugs are fixed, this confirms full parity. The 1e-3 threshold is acknowledged as strict and requires justification; the justification is that the current evaluator must be consistent across live and diagnostic to make any downstream comparison trustworthy.

**P0b does NOT pass until Phase 1 AND Phase 2 both pass.**

---

## 5. Post-P0 Execution Order — Mandatory Gate Sequence

**The following order is strict. No step may be skipped. No later step may run before earlier steps pass.**

```
[P0a] Live rollout family dispatch audit + fix
        ↓ (P0a PASS)
[P0b] Feature computation path unification
        ↓ (P0b Phase 1 + Phase 2 PASS)
[Re-C1] Re-run C1 — live support expansion audit
        ↓ (Re-C1 PASS)
[Re-C2] Re-run C2 — live diagnostic parity audit
        ↓ (Re-C2 PASS)
[P0c] close_cmd collapse preservation check
        (no new training; verify fix does not regress close_cmd)
        ↓ (P0c PASS)
[Re-C3] Re-run C3 — signal band stability
        ↓ (Re-C3 PASS)
[CANARY] Tiny canary ≤ 2000 steps, diagnostic-only
        ↓
[STOP or PROCEED based on canary outcome]
```

**No training, no probe, no v14, no new RCA spec until Re-C2 PASS.**

---

## 6. P0c — close_cmd Collapse Preservation Check

### 6.1 The Finding

v12.1 G6 (checkpoint sweep at step 10,000):

| Metric | Pretrained | Finetuned | Delta |
|--------|-----------|-----------|-------|
| `close_cmd_rate_mean` | 0.998 | 0.456 | **−0.542** |
| `min_dist_to_handle_mean` | 0.288 | 0.073 | **−0.215** (4× closer) |
| `approach_gate_pass_rate_mean` | 0.073 | 0.280 | +0.207 |
| `orientation_gate_pass_rate_mean` | 0.148 | 0.042 | −0.106 |

### 6.2 Why This Is a P0/P1 Boundary Item

P0c is listed here but is not a gate in the same sense as P0a/P0b. It is a **mandatory monitoring requirement** that must be verified during Re-C3 and the canary.

**Classification**: This is a P1 action-interface/objective finding (fine-tuning suppresses gripper close action), but it must be monitored during P0 because:
1. It is a trained behavior collapse, not an infrastructure bug
2. It must not be allowed to regress again in future canary runs
3. It is the mechanism that makes attach bridge impossible (no close → no attach)

### 6.3 P0c Criteria

```
P0c PASS if: Re-C3 or canary monitoring shows close_cmd_rate_delta ≥ −0.2
P0c FAIL if: close_cmd_rate_delta < −0.2 in any Re-C3 or canary run
P0c is NOT a hard STOP for Re-C3, but it IS a hard STOP for canary authorization.
```

---

## 7. Re-C1 — Live Support Expansion Re-audit

After P0a and P0b are fixed, re-run the C1 audit with the fixed infrastructure.

### 7.1 What Re-C1 Must Verify

1. Live rollout runner now dispatches teacher_family_variant (not just episode_index)
2. Scanning at least 12 variants per seed (matching C1 original scan depth of 12 episodes per seed)
3. Effective unique fingerprints are produced
4. Seeds 2 and 4 explicitly diagnosed (hard seed vs. dispatch failure)

### 7.2 Re-C1 Pass Criteria

```
Re-C1 PASS if ALL:
  1. selected_unique_live_episode_count ≥ 8
  2. selected_seed_coverage ≥ 6 (at least 6 seeds produce ≥1 effective unique support)
  3. seeds 2 and 4 documented as: (a) confirmed hard (all families fail) or (b) recoverable (some families succeed)
  4. No dispatch mechanism failure (episode_index used as sole entropy source)

Re-C1 FAIL if:
  - selected_unique_live_episode_count < 8
  - Seeds 2 and 4 fail AND dispatch mechanism is confirmed working (escalates to P2)
```

---

## 8. Re-C2 — Live-Diagnostic Feature Parity Re-audit

After P0a and P0b are fixed, re-run the C2 parity audit.

### 8.1 Re-C2 Pass Criteria

Identical to P0b Phase 2 criteria:

```
All continuous features: p95_abs_diff < 1e-3
attach_eligible_proxy: exact_agreement_rate > 99%
```

**Note**: Re-C2 MUST run on the same episode set as the original C2 (6 episodes, seeds 1,3,5,6,7,8 at episode_index=0) to ensure comparability.

---

## 9. Re-C3 — Signal Band Stability Re-audit

After Re-C1 and Re-C2 pass, re-run C3 with fixed infrastructure.

### 9.1 What Re-C3 Must Diagnose Separately

The original C3 conflated two different phenomena under "signal band unstable." Re-C3 must report them separately:

**Phenomenon A — close action signal stability**:
- Check: Is close_dim_accuracy gain stable across different split seeds?
- Target: close_accuracy_gain > 0.15 in ≥ 3/4 of repeat fits

**Phenomenon B — orientation action signal stability**:
- Check: Is orientation action fit stable across different split seeds?
- Current finding: highly unstable (−0.45 to +0.79 across repeat seeds)
- Target: `orientation_mse_improvement_fraction > 0` in ≥ 3/4 repeat fits

**Phenomenon B must be reported separately because it determines the P1 repair direction.**

### 9.2 Re-C3 Pass Criteria

```
Re-C3 PASS if ALL:
  1. effective_unique_live_support ≥ 16 (C3 original blocker)
  2. repeat_11_fit: one_step_fit_status != "fit_not_ok"
  3. repeat_23_fit: one_step_fit_status != "fit_not_ok"
  4. close_accuracy_gain consistent across splits (std < 0.05)
  5. orientation_mse_improvement_fraction > 0 in ≥ 3/4 repeat fits
     OR: orientation instability is explicitly attributed to a feature-parity residual (Re-C2 residual) rather than a corpus/objective problem

Re-C3 FAIL if:
  - effective_unique_live_support < 16
  - repeat fits consistently fail (fit_not_ok)
  - orientation instability persists even after Re-C2 parity is clean → P1 diagnosis required
```

---

## 10. Canary Authorization

### 10.1 Canary Preconditions (ALL must be true)

```
1. P0a PASS
2. P0b Phase 1 + Phase 2 PASS
3. Re-C1 PASS
4. Re-C2 PASS
5. P0c monitoring PASS (close_cmd_rate_delta ≥ −0.2)
6. Re-C3 PASS
```

**No training and no probe are authorized until all six preconditions are true.**

### 10.2 Canary Scope

```
Steps: ≤ 2000 (strict hard cap, NOT 10000)
Purpose: Diagnostic only — does NOT authorize full retrain
Policy: Finetune from pretrained checkpoint
Dataset: Live support (post-P0a unique episodes only)
Evaluation: Must include close_cmd_rate monitoring on held-out episodes
```

### 10.3 Canary Success Criteria — Three-Tier Outcome

**Tier 1 — Full Authorization (stop here first)**:
```
CANARY attach_bridge_gain > 0
  OR: ever_attach_eligible_fraction_gain > 0
  OR: ever_attached_rate_gain > 0
```
→ Proceed to full retrain authorization (new admission spec required)

**Tier 2 — Partial Signal (STOP, new P1 admission required)**:
```
Tier 1 NOT met, BUT:
  approach_gate_gain > 0 AND
  distance_pass_rate_gain > 0 AND
  close_cmd_rate_delta ≥ −0.2
```
→ A new P1 admission spec is required before any further training. Do NOT proceed to full retrain.

**Tier 3 — No Signal (P2 paradigm review)**:
```
All Tier 1 and Tier 2 conditions NOT met
  OR: close_cmd_rate_delta < −0.2 (P0c regression)
```
→ Escalate to paradigm review. P2 is activated.

---

## 11. Explicit Non-Regression Requirements

This spec introduces the following explicit non-regression constraints that previous specs lacked:

### 11.1 Diagnostic Pass ≠ Training Authorization
Any diagnostic gate passing (G*, C*, Re-C*) does NOT authorize training.
Training requires explicit canary authorization per section 10.

### 11.2 Pass / Completed / Signal / Claim Separation

| Status | Definition |
|--------|-----------|
| `script_completed` | All code executed without RuntimeError |
| `gate_passed` | Gate's stop condition reached |
| `scientific_signal_detected` | Credible, repeatable, cross-split metric improvement |
| `training_authorized` | ALL six canary preconditions (section 10.1) are true |
| `claim_supported` | Held-out attach_bridge_gain > 0 on canonical acceptance thresholds |

### 11.3 No Automatic Continuation
If Re-C3 PASS but orientation_mse_improvement_fraction < 0 in most splits:
→ Report this as a P1 finding. Do NOT authorize canary without explicitly addressing orientation signal instability.

If canary is authorized but close_cmd_rate collapses again (P0c regression):
→ Immediately STOP canary. Diagnose before re-authorizing.

---

## 12. What This Spec Is NOT

- **NOT a scientific hypothesis test.** P0a/P0b are bug fixes. P1 is dataset quality. P2 is paradigm risk. These are categorically different.
- **NOT a version progression.** This is not v14. Version numbers caused confusion. This spec is "P0 infrastructure repair."
- **NOT reopening v12/v13 findings.** The v12.1 close_cmd collapse finding and G2 orientation_features_missing finding remain authoritative and must not be revised retroactively.
- **NOT training authorization.** No training or probe is authorized by this spec. Canary is authorized only by section 10.
- **NOT an answer to "can MINT learn Infinigen drawer."** That question is answered only after P0 + P1 + Tier 1 canary success.

---

## 13. Canonical Files for This Round

```
Authoritative spec:
  docs/MINT_V84_P0_INFRASTRUCTURE_REPAIR_SPEC.md  ← this file

Authoritative closure artifacts:
  experiments/mint/mint_drawer_v1/autopilot/gates_post_v13_closure/C0_authority_refreeze.json
  experiments/mint/mint_drawer_v1/autopilot/gates_post_v13_closure/C1_live_support_expansion.json
  experiments/mint/mint_drawer_v1/autopilot/gates_post_v13_closure/C2_live_diagnostic_parity.json
  experiments/mint/mint_drawer_v1/autopilot/gates_post_v13_closure/C3_signal_band_stability.json
  experiments/mint/mint_drawer_v1/autopilot/gates_post_v13_closure/C4_final_pretrain_closure_verdict.json

Reference artifacts (NOT reopened):
  experiments/mint/mint_drawer_v1/autopilot/gates_v12/G5_two_stage_orientation_bridge_training.json
  experiments/mint/mint_drawer_v1/autopilot/gates_v12/G6_orientation_first_probe.json
  experiments/mint/mint_drawer_v1/artifacts/v12_state_conditioning_audit.json
  experiments/mint/mint_drawer_v1/artifacts/g8_train_seed_probe.json
  experiments/mint/mint_drawer_v1/artifacts/g8_attach_bridge_summary.json

Contracts (frozen):
  docs/contracts/truth_contract_v84.json
  docs/contracts/acceptance_contract_v84.json
```

---

## 14. Strict Execution Order Summary

```
STEP 0: Verify authority (repo, branch, HEAD, vendor, contract hashes match section 1)
          ↓
STEP 1: P0a — Audit live rollout runner
          Does it dispatch teacher_family_variant? Does it produce diversity?
          If bug found → fix → re-verify
          ↓
STEP 2: P0a PASS confirmed → P0b Phase 1
          Fix approach_alignment_cos sign/frame flip
          Target: p95_abs_diff < 0.5
          ↓
STEP 3: P0b Phase 1 PASS → P0b Phase 2
          Unify all 6 feature computation paths to single canonical function
          Target: p95_abs_diff < 1e-3, attach_eligible_proxy agreement > 99%
          ↓
STEP 4: P0b Phase 2 PASS → Re-C1
          Re-run live support expansion with fixed infrastructure
          Target: selected_unique_live_episode_count ≥ 8, seed_coverage ≥ 6
          ↓
STEP 5: Re-C1 PASS → Re-C2
          Re-run parity audit with unified feature path
          Target: all features p95_abs_diff < 1e-3
          ↓
STEP 6: Re-C2 PASS → Re-C3
          Re-run signal band stability with fixed infrastructure
          Report close vs orientation signal separately
          Target: close stable, orientation MSE improvement > 0 in ≥ 3/4 splits
          ↓
STEP 7: Re-C3 PASS → CANARY AUTHORIZATION
          Only at this point is training authorized
          Canary: ≤ 2000 steps, diagnostic only
          Monitor P0c close_cmd_rate throughout
          ↓
STEP 8: Canary outcome → TIER DETERMINATION
          Tier 1 (attach_bridge_gain > 0) → proceed to full retrain admission
          Tier 2 (approach/distance gain only) → new P1 admission spec
          Tier 3 (no signal or close_cmd collapse) → P2 paradigm review
```

---

*This spec is authoritative. It was written based on all post-v13 closure artifacts (C0-C4), v12.1 G6 findings, and the v12 state conditioning audit. No prior v* spec context has been reintroduced. No version number has been assigned. No training is authorized.*
