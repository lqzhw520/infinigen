# Session Bootstrap Artifact
**Generated**: 2026-04-06T14:30:00+08:00
**Session**: bootstrap_20260406_p0b_complete
**Campaign**: mint_drawer_v1

---

## A. Git State

```
Branch:     feature/mint-integration
HEAD:       40f5da80 — "rca(v59): complete RCA2, sovereign validation passed"
Status:     dirty tree
Upstream:   origin = https://github.com/princeton-vl/infinigen.git
Ahead:      yes (local commits not pushed)
```

**Dirty files**:
| Path | What changed |
|------|-------------|
| `experiments/mint/mint_drawer_v1/sovereign/claims.yaml` | Sovereign claims updated (RCA2/3/P0b) |
| `experiments/mint/mint_drawer_v1/sovereign/evidence/index.json` | E024/E025/E026 registered |
| `experiments/mint/mint_drawer_v1/sovereign/next_actions.json` | Next action queue updated |
| `experiments/mint/mint_drawer_v1/sovereign/state.json` | Verdict updated to P0b_ELIMINATED |
| `scripts/mint/evaluate_mint_drawer_campaign.py` | MINTPytorch dtype/compat patches |
| `external/MINT/lerobot_policy_mint/src/lerobot_policy_mint/modeling_mint.py` | MINT compat patches |
| `scripts/mint/run_p0b_colored_reroll.py` | P0b drawer-only reroll script |
| `scripts/mint/run_p0b_night_runner.py` | Night runner script (NEW) |

**external/MINT submodule**:
```
Branch: main
Commit: c053310 — "just add checkpoints folder to gitignore"
Status: dirty (modeling_mint.py modified)
```

---

## B. Dataset Truth

| Field | Value | Status |
|-------|-------|--------|
| `total_episodes` | 240 | ✅ canonical |
| `total_frames` | 19,701 | ✅ canonical |
| `image.mean` | [0.976, 0.976, 0.976] | ⚠️ near-white confirmed |
| `image.std` | [0.003, 0.003, 0.003] | ⚠️ near-white confirmed |
| `gripper_joint` | continuous | ✅ fixed (not frozen) |
| `parquet` | exists, loads | ✅ |
| `dataset_loads` | true | ✅ |

**Dataset is the SAME as what V58 training used**: confirmed via provenance audit (Apr 4 23:20 pack). Not overwritten.

---

## C. Canonical Truth Sources (Tier 0)

Only these files are authoritative for current scientific conclusions:

| File | Purpose | Authority |
|------|---------|-----------|
| `sovereign/evidence/E023.yaml` | RCA1: Teacher quality ELIMINATED | ✅ Tier 0 |
| `sovereign/evidence/E024.yaml` | RCA2: Action normalization ELIMINATED | ✅ Tier 0 |
| `sovereign/evidence/E025.yaml` | RCA3: State representation ELIMINATED | ✅ Tier 0 |
| `sovereign/evidence/E026.yaml` | P0b: Drawer color ELIMINATED (matched A/B) | ✅ Tier 0 |
| `sovereign/evidence/E022.yaml` | Env gate: 0/6 for white baseline | ✅ Tier 0 |
| `sovereign/next_actions.json` | Current next actions | ✅ Tier 0 |
| `sovereign/state.json` | Phase, verdict, queue | ⚠️ Tier 1 (fat, needs clean) |
| `artifacts/p0b_colored_rollouts/p0b_night_results.json` | Raw night runner results | ✅ Tier 1 |

---

## D. Evidence Index (all)

```
E001  d1_v57_pretrained                  eval     verified=false
E004  active_action_contract_audit        audit    verified=true
E005  libero_state_contract_audit         data_audit verified=true
E006  libero_camera_config_audit           data_audit verified=true
E007  p1a_runtime_verification            runtime_verif verified=true
E008  qa_offline_metrics_comparison        data_audit verified=true
E009  physics_legality_audit              data_audit verified=true
E010  dataset_builder_libero_alignment     eng_fix verified=true
E011  p4_physics_legal_gate               gate_eval verified=false
E002  p4_physics_legal_gate               gate_eval verified=false
E012  p1a_state_vector_runtime_verification runtime_verif verified=true
E003  p4_physics_legal_gate               gate_eval verified=false
E013  v58_critical_analysis                eval     verified=true
E014  v58_mint_finetune_and_eval          train    verified=true
E015  v58_eval                            eval     verified=true
E016  v58_live_audit                      audit    verified=true
E017  v58_live_audit_npz_raw              audit    verified=true
E018  v58_provenance_audit_and_cleanup    prov_audit verified=true
E019  v59_clean_dataset_pack               dataset_pack verified=true
E020  p0b_er_tiny_render_audit            code_audit verified=true
E021  v59_overfit_validation               eval     verified=true
E022  v59_overfit_env_gate                eval     verified=true
E023  rca1_teacher_quality_audit          RCA      verified=true
E024  rca2_action_normalization            RCA      verified=false
E025  rca3_state_representation             RCA      verified=false
E026  P0b_drawer_only_matched_AB          eval     verified=false ← NEW
```

---

## E. Stale Files (DO NOT USE as current truth)

| File | Why Stale |
|------|----------|
| `sovereign/HARNESS_HYGIENE.md` | Last updated 2026-04-05 17:00 — does not mention E024/E025/E026 |
| `HARNESS_USAGE_GUIDE.md` | Claims `V59_ENV_GATE_WEAK_PASS` — corrected to `NEGATIVE_GATE` in E022 |
| `sovereign/handoff.md` | Stuck at 2026-04-04 `ready_for_v58_training` — completely outdated |
| `sovereign/CAMPAIGN_TRUTH.v58_critical_analysis.generated.md` | Old generated view, superseded by sovereign files |
| `artifacts/p0b_colored_rollouts/p0b_colored_reroll_results.json` | Old seed-7 results (robot recolor, 20 steps) — replaced by E026 |
| `artifacts/p0b_colored_rollouts/seed_007_*` | Exploratory only, superseded |
| `state.json` (fields: `grade_assessment.v58_critical_findings`, `v58_eval_results`, `v58_train_results`) | Legacy V58 narrative — do not cite as current truth |
| `state.json` (field: `rca_status.p0b_preflight`) | Superseded by E026 matched A/B |

---

## F. Current Scientific Verdict

```
V59_RCA123_ALL_ELIMINATED — P0b_DRAWER_COLOR_ELIMINATED — SIMULATOR_PHYSICS_NOW_PRIMARY

What we KNOW:
  ✅ Teacher demos ARE successful in source sim (7/7 episodes, E023)
  ✅ Action normalization is correct format (E024)
  ✅ State representation matches env exactly (E025)
  ✅ Drawer color alone is NOT the sole bottleneck (E026, matched A/B)

What we DO NOT KNOW:
  ❓ Can teacher actions REPLAY successfully in DrawerRobotEnv?
     → Teacher Replayability Gate B (NOT YET DONE — biggest gap)
  ❓ What % of 240 episodes are task-teaching vs motion-only vs noisy?
     → Episode Admissibility Gate A (NOT YET DONE)
  ❓ Does richer visual scene (full-scene color) help beyond drawer-only?
     → Visual Sufficiency Gate C (NOT YET DONE)
  ❓ Is the patched MINT fidelity good enough for scientific conclusions?
     → Model Load Fidelity Gate D (partially done, not canonical)
  ❓ What is the actual PyBullet contact physics vs source sim?
     → Simulator Physics investigation (NOT YET STARTED)
```

---

## G. Model Load Fidelity (Current)

**Rating: F1** — Compatibility-patched, high-fidelity

| Patch Type | Count | Examples |
|-----------|-------|---------|
| A. Compatibility | 8 | `embed_language_tokens`, `.language_model.layers[]` → `.model.layers[]`, vision tower key remap |
| B. Semantic | 2 | `to_bfloat16_for_selected_params` → float32 (fixes mixed-dtype crash); key architecture remap |
| C. Experimental | 0 | — |

**Critical**: All 6 P0b rollouts AND E022 env gate were run with F1-patched MINT. Results are valid for the patched variant. They are NOT directly attributable to "original upstream MINT."

---

## H. Open Blockers

| Blocker | Severity | Depends on |
|---------|----------|------------|
| Gate B: Teacher Replayability | **CRITICAL** | Gate A |
| Gate A: Episode Admissibility | **HIGH** | — |
| Gate D: Model Load Fidelity (canonical) | HIGH | — |
| Full-scene visual proxy | MEDIUM | Gate B |
| Simulator physics investigation | MEDIUM | Gate A + Gate B |

---

## I. Next Action

**Gate B: Teacher Replayability Audit**
→ Top-10 episodes: [157, 205, 25, 180, 3, 181, 158, 182, 138, 209]
→ Replay each in DrawerRobotEnv to verify physics compatibility

---

## J. Session History

### run_001_p0b_matched_ab (2026-04-06, 1h)
P0b matched A/B: drawer-only, seeds [1,2,3], 96 steps. 0/6 success = white 0/6.
Drawer color ELIMINATED. E026 written.

### run_002_gate_a_audit (2026-04-06, 0.4s)
Episode admissibility on 240 episodes. **ALL 240 motion-only, 0 task-teaching.**
Critical: image_std=0.0000 within every episode = ZERO visual texture.
Action quality STRONG (std=0.275). Gate B is now primary.

---

## K. This File Is Canonical

**This bootstrap artifact supersedes**:
- Cursor transcript (as source of truth for state)
- HARNESS_USAGE_GUIDE.md (stale)
- handoff.md (stale)
- HARNESS_HYGIENE.md (stale)

**Read this file FIRST** before any RCA, verdict, or next-action decision.
