<!-- LEGACY DOCUMENT — DO NOT USE AS SOURCE OF TRUTH -->
<!-- Canonical sources: sovereign/ -->
<!-- Generated truth: sovereign/CAMPAIGN_TRUTH.generated.md -->
<!-- Archived location: archive/20260331_legacy/ -->

# MINT Drawer v1 — Session Progress Log

**Session:** 2026-03-27 (resolution_plan_v2 rebuild session)
**Goal:** Unblock D2 by fixing handle metadata → re-run C2 → get strong_rollout_count ≥ 2

**Major Update 2026-03-31 v3.0**: V57 experiment confirmed dual P0 blockers (SigLIP + VQ-VAE). Proxy Baseline claim REJECTED as evidence for SigLIP generalization. All campaign documents now synced to CAMPAIGN_TRUTH.md v3.0.

---

## Session Log

### 2026-03-27T12:00 — Environment Assessment
- [x] Confirmed no stray Python/training processes running on A800
- [x] Confirmed no active screen sessions
- [x] Controller lease: absent (stale lease from 2026-03-26T22:30 already expired)
- [x] `d1_candidate_progress.json` still pointing to archived diagnostic path → needs clearing
- [x] `diagnostic_runs/2026-03-26_gate_drift_reset/` still present in artifacts/ (mv timed out on NFS)

### 2026-03-27T12:30 — Root Cause Analysis Complete
- [x] Read all relevant artifacts: strong_rollout_audit, b1_oracle, u5_outlier, u3_audit, u3_patch, mainline_failure_matrix, d2_artifact, state.json, decision_memo.md
- [x] Read all relevant scripts: run_c2, run_d1, run_d2, run_u3
- [x] **Critical finding F1:** seeds 2,10 are AnyGrasp-only (oracle=0/6) — changes P0 root cause framing
- [x] **Critical finding F4:** held-out seeds 11-15 have 0 successes across ALL modalities — latent E1 blocker
- [x] **Finding F3:** D1 oracle filter missing pre_attach_motion≤0.12 pre-filter — gate incoherence confirmed in code
- [x] **Finding F4:** rollout identity drift — archived vs current seed_010_ep04 are different rollouts

### 2026-03-27T13:00 — Plan and Cleanup Artifacts Written
- [x] `RESOLUTION_PLAN.md` v2 written (8 phases, corrected priority table, GPT suggestions reviewed)
- [x] `findings.md` written (6 findings with evidence)
- [x] `history/2026-03-26_pre_patch_artifacts/` — pre-patch D1/D2 artifacts archived
- [x] `history/ARCHIVE_INDEX.md` written
- [x] `campaign_status.md` updated with verdict=rebuild_in_progress
- [x] `runtime/watch_status.json` reset to clean state

---

## Next Steps (v3.0 — updated after V57)

### BLOCKED (waiting for external resources):
- [ ] ⏳ **Contact PhD师兄**: Request VQ-VAE training code + SigLIP fine-tuning code
- [ ] ⏳ **Generate more rollouts**: Target 5,000+ frames (currently 1,301)

### NOT BLOCKED (can proceed independently):
- [ ] **Verify held-out seed solvability**: Oracle probe on seeds 11-15 (B1 only covered 1-10)
- [ ] **Consider SigLIP alternatives**: CLIP, DINOv2, or other vision encoders that may generalize to synthetic images
- [ ] **Evaluate self-implementation**: Can VQ-VAE training be self-implemented from mint_utils.py reverse-engineering?

---

## Known Risks

| Risk | Mitigation |
|------|------------|
| seed_010 pre_attach_motion still > 0.12 after Phase 1+2 | Use g3_anygrasp_grasps/seed_010.json raw AnyGrasp grasp position as fallback handle_center source |
| Held-out seeds 11-15 physically unsolvable | Narrow claim to train-seed improvement only, or fix u3 patch for held-out seeds |
| D1 no-attach-signal stop-loss triggers | Means C2 rollout quality is still insufficient — diagnose individual rollout pre_attach_motion values |
| NFS timeout on large file operations | Use cp+rm instead of mv; split large operations |

---

## Contamination Prevention Checklist (run before every screen job)

```bash
# 1. No competing Python processes
ps aux | grep python | grep -v grep | grep -v vscode | grep -v code-server
# expect: empty

# 2. GPU free
nvidia-smi | grep MiB
# expect: > 10 GB free

# 3. Correct controller ID set
echo $MINT_CONTROLLER_ID $MINT_RUN_ID
# expect: resolution_plan_v2:...

# 4. diagnostic_runs gate_drift_reset gone
ls experiments/mint/mint_drawer_v1/artifacts/diagnostic_runs/
# expect: empty or no 2026-03-26_gate_drift_reset entry

# 5. d1_candidate_progress cleared
ls experiments/mint/mint_drawer_v1/artifacts/d1_candidate_progress.json 2>/dev/null && echo EXISTS_CHECK_CONTENT || echo CLEARED_OK
```
