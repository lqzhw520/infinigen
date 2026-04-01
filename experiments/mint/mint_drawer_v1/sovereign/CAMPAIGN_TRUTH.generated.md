<!-- GENERATED FILE — DO NOT EDIT -->
<!-- Source: sovereign/state.json + sovereign/claims.yaml + sovereign/evidence/index.json -->
<!-- Generated at: 2026-04-01T07:29:48+00:00 -->
<!-- To regenerate: sovereign_cli.py render-truth -->
<!-- HARNESS VERSION: v1.3 -->

# CAMPAIGN_TRUTH — mint_drawer_v1

## Part A — Current Active Claims

| Claim ID | Status | Scope | Statement |
|----------|--------|-------|-----------|
C_SIGLIP_GENERALIZATION | candidate | current_mint_drawer_v1_pipeline | Current failure is more likely system-level misalignment acr...
C_VQVAE_USABLE | contradicted | simulation_only | Current VQ-VAE action quantizer encodes usable drawer-manipu...
C_DATASET_SUFFICIENT | contradicted | data_scale | 1,301 frames is sufficient for VQ-VAE or vision encoder adap...
C_FINETUNE_IMPROVES_HELDOUT | contradicted | heldout_eval_only | Finetuned MINT improves held-out E1 success over pretrained ...

### Current Verdict

**Status**: `root_cause_confirmed_dual_blocker`
**Blockers**: `none`
**Reason**: `—`
**Decision**: `—`

---

## Part B — Recent Claim Revisions

| Time | Claim | Rev | Change | Reason |
|------|-------|-----|--------|--------|
2026-03-31T19:00:00+08:00 | C_SIGLIP_GENERALIZATION@3 | reframed | Reframed after advisor review: D1 seed2 0% vs E1 1...
2026-03-31T19:00:00+08:00 | C_VQVAE_USABLE@1 | create | Established from D1 V57: pretrained_mint=0% grasp ...
2026-03-31T19:00:00+08:00 | C_DATASET_SUFFICIENT@1 | create | LIBERO uses ~50,000 frames. 1,301 frames = 2.6% of...
2026-03-31T19:00:00+08:00 | C_FINETUNE_IMPROVES_HELDOUT@1 | create | E1 held-out eval: pretrained 1/5=20%, finetuned 0/...
2026-03-31T12:00:00+08:00 | C_SIGLIP_GENERALIZATION@2 | scope_narrowed | Proxy baseline diagnosis was over-broad; narrowed ...
2026-03-30T10:00:00+08:00 | C_SIGLIP_GENERALIZATION@1 | create | Initial diagnosis from D1 V57 failed runs: pretrai...

---

## Evidence Chain

| ID | Experiment | Type | Timestamp |
|----|-----------|------|-----------|
E001 | d1_v57_pretrained | eval | 2026-03-30T20:30:00+08:00

---

## Campaign Queue Summary

- **Pending**: 4 steps
- **Completed**: 12 steps
- **Blocked**: 0 steps
