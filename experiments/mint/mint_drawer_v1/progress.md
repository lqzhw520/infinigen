# MINT Drawer v1 — Session Progress Log

**Session:** 2026-03-27 (resolution_plan_v2 rebuild session)  
**Goal:** Unblock D2 by fixing handle metadata → re-run C2 → get strong_rollout_count ≥ 2

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

## Next Steps (in order)

- [ ] **Phase 0:** Complete diagnostic_runs archive (cp+rm, NFS-safe); clear d1_candidate_progress.json; reset state.json queue
- [ ] **Phase 1:** Fix seed_010 handle_center_local — use c2 rollout with LOWEST pre_attach_motion as source (not ep04 circular ref)
- [ ] **Phase 2:** Re-run C2 in screen with new MINT_CONTROLLER_ID → verify strong_rollout_count ≥ 2 AND strong_coherent_seeds non-empty
- [ ] **Phase 2.5:** Oracle probe on held-out seeds 11-15 to verify E1 physical solvability
- [ ] **Phase 3:** Add pre_attach_motion≤0.12 filter in strict_teacher_dataset.py; add no_attach_signal stop-loss in run_d1
- [ ] **Phase 4:** Re-run D1 with clean controller lease → verify promoted_candidate.candidate_role=mainline
- [ ] **Phase 5:** Re-run D2 → verify success_gain≥0.2, finetuned_successes≥3
- [ ] **Phase 6:** D3 train seed probe (only after D2 hard-pass)
- [ ] **Phase 7:** E1 held-out eval (only after D3 hard-pass AND Phase 2.5 solvability confirmed)

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
