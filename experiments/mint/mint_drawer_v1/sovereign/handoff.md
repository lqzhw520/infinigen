## Bootstrap Handoff — 2026-04-02T01:30:00+0800

### Git Rebase Result ✅
- `git rebase origin/main` → 3 本地 commits 已 rebase 到 bc796ba 之上
- SDAT/: 完整引入（core.py, model/vqvae.py, trainer.py, configs/train.yaml）
- modeling_mint.py: origin/main 权威版本（V57 本地 Python 改动已清除）
- test_mint.py / verify_mint.py: 作为独立脚本文件重新添加（不在 upstream）
- 本地 V57 改动完整备份: `git branch mint_v57_backup`
- 当前状态: `main` 与 `origin/main` 对齐（`git rev-list --left-right --count HEAD...origin/main` → 1 0）
- 推送: `git push -u origin HEAD`（需要时执行）

### MINT Architecture — Full Component Map
MINT = PaliGemma (Vision Encoder) + MultiScaleVQVAE (Action Tokenizer) + Gemma (AR Expert)
- VQ-VAE: action-only, camera-independent (SDAT/core.py confirms: seq_dim → tokens, no image input)
- Vision Encoder: PaliGemma/SigLIP, camera-perspective-sensitive (PaliGemma + vision tower)
- VQ-VAE pretrained on LIBERO action sequences (7-dim, 16-step chunks, 4 patch levels [1,2,4])
- SDAT/train.yaml: `action_norm_mode: none`, `codebook_size=512`, `codebook_dim=32`

### V58 Multi-Factor Alignment (from advisor feedback)
Prioritized data quality dimensions:
- P0: Physics-legal teacher rollouts (gripper cannot penetrate drawer; compliant pull)
- P1: Camera viewpoint alignment (LIBERO eye-in-hand vs Infinigen third-person)
- P1: Sim-to-Real gap (LIBERO photorealistic vs Infinigen synthetic rendering)
- P2: VQ-VAE action distribution alignment (AnyGrasp vs human teleop action distributions)
- P3: Lighting, mesh diversity, object variety

### Claim Update
- C_PHYSICS_LEGAL_TEACHER@1 (candidate): Supersedes C_VQVAE_USABLE, C_DATASET_SUFFICIENT
- v57 lesson: VQ-VAE learns motion (EER 8.6x) but gripper never closes → vision encoder is the primary blocker
- V58 priority: P0 (physics) → P1 (vision) → P2 (tokenizer) → P4 (joint fine-tune)

---

<!-- Bootstrap Handoff — 2026-04-02T00:00:00+0800 -->

### Git Sync Status (New Evidence: 2026-04-02)
- `origin/main` = `bc796ba fix ckpt save bug` — contains full SDAT tokenizer training code (SDAT/core.py 499 lines, SDAT/model/vqvae.py, SDAT/trainer.py, SDAT/configs/train.yaml)
- `local/main` = `fdf2a79 Modifications to the fallback decoder` — V57 changes WITHOUT SDAT
- Local commits `fdf2a79`, `c43099b`, `b305240` are ABOVE `4d04a88` (grafted Release) but NOT on origin/main
- Action: Rebase local V57 changes onto origin/main (preserving quantizer-only unfreeze in modeling_mint.py)

### Physics Penetration Claim (New: C_PHYSICS_LEGAL_TEACHER)
- Evidence: `active_action_contract.json` — ALL 25 replay rollouts failed (passed: false)
- Root cause A: Grasp phase — `max_position_error=0.33-1.03m` (gripper clips INTO drawer panel, not onto handle)
- Root cause B: Pull phase — `max_drawer_error=1.0`, drawer fraction stays 0.0 (world-frame delta action physically impossible when drawer is constrained)
- Root cause C: LIBERO (real robot) vs Infinigen (sim) physics — real robot stops at collision; Infinigen delta action ignores drawer constraint

### V58 Data Pipeline Solution
- Stage 0: Physics feasibility check for AnyGrasp candidates (contact force threshold)
- Stage 1: Compliant pull controller (impedance control, not world-frame delta)  
- Stage 2: Frame-by-frame collision filtering
- Academic value: Demonstrates Infinigen can generate PHYSICS-LEGAL robot manipulation data

### Updated Claims
- C_PHYSICS_LEGAL_TEACHER@1: "Teacher rollouts from Infinigen + AnyGrasp contain gripper trajectories that violate physical collision constraints (gripper penetrates drawer panel; max_position_error 0.33-1.03m)."
- Supersedes: C_VQVAE_USABLE@1, C_DATASET_SUFFICIENT@1 (root cause now shifted upstream to data generation, not tokenizer)

---

<!-- MORNING HANDOFF — 2026-04-01T20:14:17+0800 -->
## Morning Handoff — 2026-04-01T20:14:17+0800

### System Status
**Watcher**: [2026-04-01T20:14:16+0800] [INFO] Starting: night_watcher

### Claim Changes (since last session)
  Recent Claim Changes (last 5 revisions):
    2026-03-31T19:00  C_SIGLIP_GENERALIZATION@3  [reframed]  status=candidate ←CURRENT
              Reframed after advisor review: D1 seed2 0% vs E1 1/5=20% inconsistency means treating SigLIP as confirmed P0 is prematur...

    2026-03-31T19:00  C_VQVAE_USABLE@1  [create]  status=contradicted ←CURRENT
              Established from D1 V57: pretrained_mint=0% grasp (independent of fine-tuning); no VQ-VAE training code in workspace; V5...

    2026-03-31T19:00  C_DATASET_SUFFICIENT@1  [create]  status=contradicted ←CURRENT
              LIBERO uses ~50,000 frames. 1,301 frames = 2.6% of LIBERO. Target 5,000+ for meaningful adaptation. No training code exi...

    2026-03-31T19:00  C_FINETUNE_IMPROVES_HELDOUT@1  [create]  status=contradicted ←CURRENT
              E1 held-out eval: pretrained 1/5=20%, finetuned 0/5=0% — finetuned is strictly worse than pretrained. V57 D1: pretrained...

    2026-03-31T12:00  C_SIGLIP_GENERALIZATION@2  [scope_narrowed]  status=contradicted
              Proxy baseline diagnosis was over-broad; narrowed to DrawerRobotEnv task-specific evaluation after seed 14 evidence emer...

### Night Planner Recommendations
    ## RECOMMENDED NEXT STEPS (for human review)
    ─────────────────────────────────────────────────────────────
      Current decision: await_external_resources
      Verdict: root_cause_confirmed_dual_blocker
      Next phase: v58_end_to_end_finetune_with_siglip_solution
      Awaiting: VQ-VAE training code (from PhD师兄), SigLIP fine-tuning code or alternative vision encoder solution (from PhD师兄 or self-research), 5,000+ frames
    
      Active contradicted claims: C_VQVAE_USABLE, C_DATASET_SUFFICIENT, C_FINETUNE_IMPROVES_HELDOUT
        → Review: Are these still accurate? Should they be superseded or archived?
    
      Active candidate claims: C_SIGLIP_GENERALIZATION
        → Review: What evidence is needed to promote to 'supported' or 'contradicted'?
    
      Recommended actions for next session:
        1. Run: python scripts/harness/sovereign_cli.py bootstrap
        2. Review: sovereign/handoff.md for recent activity
        3. Review: sovereign/CAMPAIGN_TRUTH.generated.md for current state
        4. If blocked by external resources, verify status of:
           - VQ-VAE training code (from PhD师兄)
           - SigLIP fine-tuning code or alternative vision encoder solution (from PhD师兄 or self-research)
           - 5,000+ frames

---

<!-- MORNING HANDOFF — 2026-04-01T20:14:10+0800 -->
## Morning Handoff — 2026-04-01T20:14:10+0800

### System Status
**Watcher**: [2026-04-01T20:14:09+0800] [INFO] Starting: night_watcher

### Claim Changes (since last session)
  Recent Claim Changes (last 5 revisions):
    2026-03-31T19:00  C_SIGLIP_GENERALIZATION@3  [reframed]  status=candidate ←CURRENT
              Reframed after advisor review: D1 seed2 0% vs E1 1/5=20% inconsistency means treating SigLIP as confirmed P0 is prematur...

    2026-03-31T19:00  C_VQVAE_USABLE@1  [create]  status=contradicted ←CURRENT
              Established from D1 V57: pretrained_mint=0% grasp (independent of fine-tuning); no VQ-VAE training code in workspace; V5...

    2026-03-31T19:00  C_DATASET_SUFFICIENT@1  [create]  status=contradicted ←CURRENT
              LIBERO uses ~50,000 frames. 1,301 frames = 2.6% of LIBERO. Target 5,000+ for meaningful adaptation. No training code exi...

    2026-03-31T19:00  C_FINETUNE_IMPROVES_HELDOUT@1  [create]  status=contradicted ←CURRENT
              E1 held-out eval: pretrained 1/5=20%, finetuned 0/5=0% — finetuned is strictly worse than pretrained. V57 D1: pretrained...

    2026-03-31T12:00  C_SIGLIP_GENERALIZATION@2  [scope_narrowed]  status=contradicted
              Proxy baseline diagnosis was over-broad; narrowed to DrawerRobotEnv task-specific evaluation after seed 14 evidence emer...

### Night Planner Recommendations
    ## RECOMMENDED NEXT STEPS (for human review)
    ─────────────────────────────────────────────────────────────
      Current decision: await_external_resources
      Verdict: root_cause_confirmed_dual_blocker
      Next phase: v58_end_to_end_finetune_with_siglip_solution
      Awaiting: VQ-VAE training code (from PhD师兄), SigLIP fine-tuning code or alternative vision encoder solution (from PhD师兄 or self-research), 5,000+ frames
    
      Active contradicted claims: C_VQVAE_USABLE, C_DATASET_SUFFICIENT, C_FINETUNE_IMPROVES_HELDOUT
        → Review: Are these still accurate? Should they be superseded or archived?
    
      Active candidate claims: C_SIGLIP_GENERALIZATION
        → Review: What evidence is needed to promote to 'supported' or 'contradicted'?
    
      Recommended actions for next session:
        1. Run: python scripts/harness/sovereign_cli.py bootstrap
        2. Review: sovereign/handoff.md for recent activity
        3. Review: sovereign/CAMPAIGN_TRUTH.generated.md for current state
        4. If blocked by external resources, verify status of:
           - VQ-VAE training code (from PhD师兄)
           - SigLIP fine-tuning code or alternative vision encoder solution (from PhD师兄 or self-research)
           - 5,000+ frames

---

<!-- MORNING HANDOFF — 2026-04-01T18:29:26+0800 -->
## Morning Handoff — 2026-04-01T18:29:26+0800

### System Status
**Watcher**: [2026-04-01T18:29:25+0800] [INFO] Starting: night_watcher

### Claim Changes (since last session)
  Recent Claim Changes (last 5 revisions):
    2026-03-31T19:00  C_SIGLIP_GENERALIZATION@3  [reframed]  status=candidate ←CURRENT
              Reframed after advisor review: D1 seed2 0% vs E1 1/5=20% inconsistency means treating SigLIP as confirmed P0 is prematur...

    2026-03-31T19:00  C_VQVAE_USABLE@1  [create]  status=contradicted ←CURRENT
              Established from D1 V57: pretrained_mint=0% grasp (independent of fine-tuning); no VQ-VAE training code in workspace; V5...

    2026-03-31T19:00  C_DATASET_SUFFICIENT@1  [create]  status=contradicted ←CURRENT
              LIBERO uses ~50,000 frames. 1,301 frames = 2.6% of LIBERO. Target 5,000+ for meaningful adaptation. No training code exi...

    2026-03-31T19:00  C_FINETUNE_IMPROVES_HELDOUT@1  [create]  status=contradicted ←CURRENT
              E1 held-out eval: pretrained 1/5=20%, finetuned 0/5=0% — finetuned is strictly worse than pretrained. V57 D1: pretrained...

    2026-03-31T12:00  C_SIGLIP_GENERALIZATION@2  [scope_narrowed]  status=contradicted
              Proxy baseline diagnosis was over-broad; narrowed to DrawerRobotEnv task-specific evaluation after seed 14 evidence emer...

### Night Planner Recommendations
    ## RECOMMENDED NEXT STEPS (for human review)
    ─────────────────────────────────────────────────────────────
      Current decision: await_external_resources
      Verdict: root_cause_confirmed_dual_blocker
      Next phase: v58_end_to_end_finetune_with_siglip_solution
      Awaiting: VQ-VAE training code (from PhD师兄), SigLIP fine-tuning code or alternative vision encoder solution (from PhD师兄 or self-research), 5,000+ frames
    
      Active contradicted claims: C_VQVAE_USABLE, C_DATASET_SUFFICIENT, C_FINETUNE_IMPROVES_HELDOUT
        → Review: Are these still accurate? Should they be superseded or archived?
    
      Active candidate claims: C_SIGLIP_GENERALIZATION
        → Review: What evidence is needed to promote to 'supported' or 'contradicted'?
    
      Recommended actions for next session:
        1. Run: python scripts/harness/sovereign_cli.py bootstrap
        2. Review: sovereign/handoff.md for recent activity
        3. Review: sovereign/CAMPAIGN_TRUTH.generated.md for current state
        4. If blocked by external resources, verify status of:
           - VQ-VAE training code (from PhD师兄)
           - SigLIP fine-tuning code or alternative vision encoder solution (from PhD师兄 or self-research)
           - 5,000+ frames

---

<!-- MORNING HANDOFF — 2026-04-01T18:28:42+0800 -->
## Morning Handoff — 2026-04-01T18:28:42+0800

### System Status
**Watcher**: [2026-04-01T18:28:41+0800] [INFO] Starting: night_watcher

### Claim Changes (since last session)
  Recent Claim Changes (last 5 revisions):
    2026-03-31T19:00  C_SIGLIP_GENERALIZATION@3  [reframed]  status=candidate ←CURRENT
              Reframed after advisor review: D1 seed2 0% vs E1 1/5=20% inconsistency means treating SigLIP as confirmed P0 is prematur...

    2026-03-31T19:00  C_VQVAE_USABLE@1  [create]  status=contradicted ←CURRENT
              Established from D1 V57: pretrained_mint=0% grasp (independent of fine-tuning); no VQ-VAE training code in workspace; V5...

    2026-03-31T19:00  C_DATASET_SUFFICIENT@1  [create]  status=contradicted ←CURRENT
              LIBERO uses ~50,000 frames. 1,301 frames = 2.6% of LIBERO. Target 5,000+ for meaningful adaptation. No training code exi...

    2026-03-31T19:00  C_FINETUNE_IMPROVES_HELDOUT@1  [create]  status=contradicted ←CURRENT
              E1 held-out eval: pretrained 1/5=20%, finetuned 0/5=0% — finetuned is strictly worse than pretrained. V57 D1: pretrained...

    2026-03-31T12:00  C_SIGLIP_GENERALIZATION@2  [scope_narrowed]  status=contradicted
              Proxy baseline diagnosis was over-broad; narrowed to DrawerRobotEnv task-specific evaluation after seed 14 evidence emer...

### Night Planner Recommendations
    ## RECOMMENDED NEXT STEPS (for human review)
    ─────────────────────────────────────────────────────────────
      Current decision: await_external_resources
      Verdict: root_cause_confirmed_dual_blocker
      Next phase: v58_end_to_end_finetune_with_siglip_solution
      Awaiting: VQ-VAE training code (from PhD师兄), SigLIP fine-tuning code or alternative vision encoder solution (from PhD师兄 or self-research), 5,000+ frames
    
      Active contradicted claims: C_VQVAE_USABLE, C_DATASET_SUFFICIENT, C_FINETUNE_IMPROVES_HELDOUT
        → Review: Are these still accurate? Should they be superseded or archived?
    
      Active candidate claims: C_SIGLIP_GENERALIZATION
        → Review: What evidence is needed to promote to 'supported' or 'contradicted'?
    
      Recommended actions for next session:
        1. Run: python scripts/harness/sovereign_cli.py bootstrap
        2. Review: sovereign/handoff.md for recent activity
        3. Review: sovereign/CAMPAIGN_TRUTH.generated.md for current state
        4. If blocked by external resources, verify status of:
           - VQ-VAE training code (from PhD师兄)
           - SigLIP fine-tuning code or alternative vision encoder solution (from PhD师兄 or self-research)
           - 5,000+ frames

---

<!-- MORNING HANDOFF — 2026-04-01T18:26:13+0800 -->
## Morning Handoff — 2026-04-01T18:26:13+0800

### System Status
**Watcher**:   [no watcher log found — watcher may not have run]

### Claim Changes (since last session)
  Recent Claim Changes (last 5 revisions):
    2026-03-31T19:00  C_SIGLIP_GENERALIZATION@3  [reframed]  status=candidate ←CURRENT
              Reframed after advisor review: D1 seed2 0% vs E1 1/5=20% inconsistency means treating SigLIP as confirmed P0 is prematur...

    2026-03-31T19:00  C_VQVAE_USABLE@1  [create]  status=contradicted ←CURRENT
              Established from D1 V57: pretrained_mint=0% grasp (independent of fine-tuning); no VQ-VAE training code in workspace; V5...

    2026-03-31T19:00  C_DATASET_SUFFICIENT@1  [create]  status=contradicted ←CURRENT
              LIBERO uses ~50,000 frames. 1,301 frames = 2.6% of LIBERO. Target 5,000+ for meaningful adaptation. No training code exi...

    2026-03-31T19:00  C_FINETUNE_IMPROVES_HELDOUT@1  [create]  status=contradicted ←CURRENT
              E1 held-out eval: pretrained 1/5=20%, finetuned 0/5=0% — finetuned is strictly worse than pretrained. V57 D1: pretrained...

    2026-03-31T12:00  C_SIGLIP_GENERALIZATION@2  [scope_narrowed]  status=contradicted
              Proxy baseline diagnosis was over-broad; narrowed to DrawerRobotEnv task-specific evaluation after seed 14 evidence emer...

### Night Planner Recommendations
    ## RECOMMENDED NEXT STEPS (for human review)
    ─────────────────────────────────────────────────────────────
      Current decision: await_external_resources
      Verdict: root_cause_confirmed_dual_blocker
      Next phase: v58_end_to_end_finetune_with_siglip_solution
      Awaiting: VQ-VAE training code (from PhD师兄), SigLIP fine-tuning code or alternative vision encoder solution (from PhD师兄 or self-research), 5,000+ frames
    
      Active contradicted claims: C_VQVAE_USABLE, C_DATASET_SUFFICIENT, C_FINETUNE_IMPROVES_HELDOUT
        → Review: Are these still accurate? Should they be superseded or archived?
    
      Active candidate claims: C_SIGLIP_GENERALIZATION
        → Review: What evidence is needed to promote to 'supported' or 'contradicted'?
    
      Recommended actions for next session:
        1. Run: python scripts/harness/sovereign_cli.py bootstrap
        2. Review: sovereign/handoff.md for recent activity
        3. Review: sovereign/CAMPAIGN_TRUTH.generated.md for current state
        4. If blocked by external resources, verify status of:
           - VQ-VAE training code (from PhD师兄)
           - SigLIP fine-tuning code or alternative vision encoder solution (from PhD师兄 or self-research)
           - 5,000+ frames

---

## Handoff — mint_drawer_v1 Governance Recovery

**Created**: 2026-03-31T22:10:00+08:00
**Sovereign version**: 1
**Created by**: harness Phase 1 bootstrap

### Current Verdict

**Status**: `blocked`
**Blockers**: `C_SIGLIP_GENERALIZATION` + `C_VQVAE_USABLE`
**Reason**: `dual_p0_unresolved`
**Decision**: `await_phd_brother_vqvae_and_siglip_code_and_data_expansion`
**Next Phase**: `v58_end_to_end_finetune_with_siglip_solution`

### Campaign State

- **Active experiment**: `null` (V57 exhausted; waiting for external resources)
- **Last experiment**: D1 V57 (pretrianed=0% grasp, finetuned=0% grasp, 3000 steps)
- **Evidence chain**: E001 established (d1_v57_pretrained eval)
- **Pending**: E002 (E1 held-out eval), E003 (D1 V57 finetuned eval — same result as E001)

### Claim Evolution (why we are here)

C_SIGLIP_GENERALIZATION evolved through 3 revisions:
- Rev 1 (2026-03-30): broad contradicted — pretrained=0% grasp on seed 2
- Rev 2 (2026-03-31): narrowed to DrawerRobotEnv — E1 seed14 1/5=20% emerged
- Rev 3 (2026-03-31): reframed as candidate — D1 vs E1 inconsistency means system-level check needed

Key lesson: "Proxy baseline 100% success" was rejected as SigLIP evidence because DrawerProxyEnv is trivially simple task (fixed joint-delta).

### What the Next Agent Must Do

1. **ALWAYS** run `python scripts/harness/sovereign_cli.py bootstrap` before any analysis
2. **ALWAYS** read `sovereign/claims.yaml` — do not trust any other document as claim source
3. **DO NOT** promote any claim without running `sovereign_cli.py revise-claim`
4. **DO NOT** close any experiment without running `sovereign_cli.py close-experiment`
5. **V58 path**: await PhD师兄 VQ-VAE training code + SigLIP solution before resuming learnability lane

### Files That Are NOT Canonical

The following files are LEGACY — do not use as truth sources:
- `campaign_status.md`, `findings.md`, `progress.md`, `decision_memo.md`
- `takeover_memo.md`, `step_review_guide.md`, `campaign_spec.md`
- `review_prompt.md`, `acceptance_criteria.json`
- `summary.json`, `review.json`

### V58 Preparation (unblocked tasks)

- [ ] Oracle probe seeds 11-15 (held-out solvability — B1 only covered 1-10)
- [ ] Evaluate SigLIP alternatives if PhD师兄 cannot provide solution
- [ ] Self-implementation feasibility: can VQ-VAE training be reverse-engineered from mint_utils.py?
