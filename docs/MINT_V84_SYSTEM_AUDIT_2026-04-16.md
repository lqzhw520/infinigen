# MINT v8.4 System Audit (2026-04-18)

**Purpose**: authoritative current-state snapshot after the full `v9` gate-controlled diagnostic tiny-retrain execution order completed on the remote sovereign.

**Authority**
- Host: `ssh -p 30017 root@10.210.0.88`
- Repo: `/mnt/afs2/zhuhaowu/infinigen`
- Branch: `feature/mint-env-reformulation-v1-visual-fidelity`
- Canonical vendor baseline: `external/MINT@4eab5795345721001c412ff1ca2c886a11eab606`
- Canonical bounded spec: `/mnt/afs2/zhuhaowu/infinigen/docs/MINT_V84_UNIFIED_EXECUTION_SPEC.md`
- Canonical v9 gate spec: `/mnt/afs2/zhuhaowu/infinigen/docs/MINT_V84_DIAGNOSTIC_TINY_RETRAIN_GATE_EXECUTION_SPEC.md`

## 1. Executive summary

1. **Line A remains operationally frozen and is not the current blocker.**
   - Hard-seed scripted teacher supply was already restored before `v9`.
   - `v9` did not reopen Line A.

2. **Line B remains frozen and is not the current blocker.**
   - Truth-contract alignment stayed intact through the `v9` run.

3. **Line C baseline reproduction governance is no longer the current blocker.**
   - Frozen vendor MINT stayed unmodified.
   - Authoritative baseline identity stayed: `p1c10 + /root/anaconda3/envs/mint/bin/python + external/MINT@4eab579...`.
   - Probe and eval were delegated through the authoritative `mint` environment rather than re-opening vendor patching.

4. **The full `v9` gate-controlled diagnostic run completed end-to-end.**
   - Execution run head: `08a1235bc229a6f31e7c8f248d26df1af8fbc06d`
   - Gate result sequence:
     - `G0 PASS`
     - `G1 PASS`
     - `G2 PASS`
     - `G3 PASS`
     - `G4 DIAGNOSTIC_PASS`
     - `G5 PASS`
     - `G6 STOP`
     - `G7 DIAGNOSTIC_PASS`
     - `G8 DIAGNOSTIC_PASS`

5. **The current final verdict is negative and diagnostic, not claim-bearing.**
   - `final_verdict = diagnostic_no_learning_signal_readiness_likely_causal`

## 2. What `v9` proved

### 2.1 Technical path is open
The negative result is **not**:
- wrapper/env still broken
- vendor MINT still broken
- training cannot start
- checkpoint writing failed

The run proved all of the following under the frozen sovereign:
- dataset materialization and validation passed
- train launched and completed to `10000` steps
- checkpoints were written through `010000`
- train-seed probe executed under authoritative `mint` runtime
- held-out eval was skipped intentionally because the run remained diagnostic-only

### 2.2 Remaining readiness failures survived synchronization
Current synchronized dataset/readiness state:
- `dataset_valid = true`
- `used_rollout_count = 96`
- `effective_frame_count = 5196`
- `used_successful_seed_count = 8`
- `accepted_seed_coverage = [1,2,3,4,5,6,7,8]`
- `accepted_unique_teacher_family_count = 8`
- `strict_unique_teacher_family_count = 8`
- `near_strict_unique_teacher_family_count = 0`
- `family_collapse_suspected = true`

Current readiness outcome:
- `teacher_readiness_passed = false`
- failed clauses:
  - `accepted_unique_teacher_families_ge_18`
  - `near_strict_unique_teacher_families_ge_6`

So the old stale-artifact explanation is no longer available.
The remaining readiness failure is now a live property of the current synchronized corpus.

## 3. Why the result is negative

### 3.1 Trainability bridge was not established
The diagnostic train finished, but probe showed no usable behavioral signal.

Observed probe result:
- `trend_passed = false`
- `attach_bridge_pass = false`
- `probe_classification = no_learning_signal`
- `selected_checkpoint_step = 2000`

Important interpretation note:
- the selected checkpoint being `2000` is a tie-break artifact
- it does **not** mean only the first checkpoint failed
- the authoritative per-checkpoint summaries remained behaviorally zero-signal through `2000/4000/6000/8000/10000`

### 3.2 Behavioral failure mode is upstream of held-out eval
Across the authoritative train-seed probe summaries:
- pretrained success remained `0.0`
- finetuned success remained `0.0`
- pretrained grasp/attach/phase-lock remained `0.0`
- finetuned grasp/attach/phase-lock remained `0.0`
- dominant failure mode remained `never_reach_attach_distance`

The attach-bridge summary is the clearest compression:
- pretrained:
  - `ever_attached_rate = 0.0`
  - `stable_attach_rate = 0.0`
  - `phase_locked_rate = 0.0`
  - `max_drawer_fraction_mean = 0.0`
  - `strict_success_rate = 0.0`
- finetuned:
  - `ever_attached_rate = 0.0`
  - `stable_attach_rate = 0.0`
  - `phase_locked_rate = 0.0`
  - `max_drawer_fraction_mean = 0.0`
  - `strict_success_rate = 0.0`

So the learning system did not even rebuild the attach bridge on the train seeds.
Held-out evaluation therefore remained correctly ineligible under diagnostic-only scope.

## 4. Current causal interpretation

The strongest current interpretation is:

> **The project is no longer blocked on environment/wrapper/vendor execution.**
> **It is now blocked on trainability under a corpus that still fails diversity and near-strict readiness.**

That is why the final diagnostic verdict is:
- `diagnostic_no_learning_signal_readiness_likely_causal`

This does **not** prove a formal impossibility theorem.
It does mean the remaining readiness failures are now the leading causal explanation, because:
- the run is synchronized to current Line A/B/C state
- the run is technically executable
- the train completed
- but the policy still did not learn to attach or open

## 5. What is no longer a valid explanation

The following explanations are now weaker than before and should not be used as the primary story:
- "the environment still isn't wired correctly"
- "MINT runtime is still fundamentally broken"
- "the vendor code itself is the blocker"
- "the old stale G6/G8 artifacts were the reason readiness looked false"

Those are no longer the main bottleneck explanation after `v9`.

## 6. Current active object

The current active object is:

> **root-cause analysis for why the synchronized but low-diversity / no-near-strict corpus still yields zero attach-learning signal under diagnostic tiny retrain**

This is narrower than re-opening Line A/B/C, and narrower than generic MINT runtime debugging.
