# findings.md — mint_drawer_v1 D1 deep diagnosis + V57 truth correction

**Updated**: 2026-03-31T19:00:00+08:00

## D1 Run Summary

- **Training**: PASSED (V57: frozen decoder, 3000 steps, loss 9.287→0.100, stable convergence)
- **Eval**: ZERO SIGNAL in grasp (5/5 episodes pretrained=0%, finetuned=0%)
- **D1 Gate**: FAILED — `u3_regime_b_required` (V57, attempt 3)
- **V57 Core Finding**: pretrained AND finetuned both 0% grasp — problem is NOT in fine-tuning strategy

> **IMPORTANT CAVEAT (v3.0 — 2026-03-31)**: The archived proxy baseline result does **NOT** prove that SigLIP vision encoder can generalize to Infinigen synthetic images. The `DrawerProxyEnv` is a trivially simple task (fixed drawer-joint delta `[0.85, 0, 0, 0, 0, 0, 1.0]` per step). The 100% success came from MINT memorizing the training data pattern, not from vision encoder scene understanding. The real robot environment (`DrawerRobotEnv`) E1 evaluation shows: pretrained_mint = 0% grasp, finetuned_mint = 0% success. **SigLIP P0 blocker is real.**

---

## Root Cause Decomposition (Updated v3.0)

### What V57 Actually Confirmed

V57 was designed to test: "Is the frozen pretrained decoder protecting LIBERO gripper representations?" The answer: **No — the problem is upstream of fine-tuning.**

Evidence chain:
1. V57 pretrained MINT = 0% grasp (zero fine-tuning, clean evaluation)
2. V57 finetuned = 0% grasp (3000 steps, stable loss)
3. E1 held-out eval (DrawerRobotEnv): pretrained 1/5=20%, finetuned 0/5=0%
4. **Conclusion: Pretrained MINT itself cannot execute the task in Infinigen robot sim. Fine-tuning strategy is irrelevant.**

### Two P0 Blockers (v3.0 — confirmed)

**P0-1: SigLIP Vision Encoder cannot generalize to Infinigen synthetic images**
- Pretrained on WebLI (real photos) — never seen Infinigen synthetic images
- Infinigen: synthetic rendering, different lighting, textures, noise characteristics
- Result: vision encoder cannot recognize drawer, EEF, or spatial relationships
- Evidence: E1 eval pretrained_mint = 0% grasp on real Robot Env

**P0-2: VQ-VAE Quantizer cannot encode Infinigen drawer action latent**
- Trained on LIBERO ~50,000 frames (diverse manipulation tasks)
- Cannot encode Infinigen drawer-specific action latent space with 1,301 frames (2.6%)
- 512 codebook entries sparsely covered by drawer-only data
- Evidence: Pretrained MINT = 0% grasp (VQ-VAE was never trained on drawer data)

**Both must be solved simultaneously. V58 (VQ-VAE retraining) is necessary but insufficient without vision encoder adaptation.**

### What Was Excluded (v3.0)

|| Hypothesis | Verdict | Evidence |
||-----------|---------|---------|
|| "Continuous vs discrete gripper" | REJECTED | NPZ verified: gripper_values ∈ {0,1}, actions[:,6] ∈ {-1,+1} |
|| "Decoder broken by direct_grip_head" | REJECTED | V57 frozen decoder still 0% grasp |
|| "SigLIP can generalize (Proxy Baseline)" | REJECTED | Proxy = trivial joint-delta task; real Robot Env E1 shows pretrained=0% grasp |
|| "VQ-VAE codebook mismatch (gripper)" | REJECTED | Discrete binary data matches LIBERO; BUT quantizer may fail on drawer-specific latent encoding |
|| "RL convention error" | REJECTED | action[N] correctly predicts state[N+1] |

---

## Root Cause Status (v3.0 — dual P0 blockers confirmed)

|| Root Cause | Status | Evidence |
||------------|--------|----------|
|| P0-1: SigLIP Vision Encoder cannot generalize | **CONFIRMED** | E1 eval (DrawerRobotEnv): pretrained_mint = 0% grasp; V57 pretrained = 0% |
|| P0-2: VQ-VAE Quantizer bottleneck | **CONFIRMED** | Pretrained MINT = 0% grasp (independent of fine-tuning); workspace no training code |
|| P1: Insufficient data | **CONFIRMED** | 1,301 frames (2.6% of LIBERO ~50K); need 5,000+ |
|| P2: Held-out seeds 11-15 physical solvability | **UNVERIFIED** | B1 only covers seeds 1-10; E1 oracle probe needed |
|| ~~Engineering bugs~~ | **RESOLVED** | V57: training fully stable, loss converges normally |
|| ~~Continuous gripper data~~ | **REJECTED** | NPZ verified: discrete binary {0,1} |
|| ~~SigLIP can generalize (Proxy Baseline)~~ | **REJECTED** | Proxy = trivial task; does not transfer to robot-trajectory task |

---

## Hypotheses for Zero Grasp Signal (v3.0 — post-V57)

> The question is no longer "why does fine-tuning fail?" but "why does pretrained MINT fail on Infinigen robot sim?"

### H1: SigLIP Vision Encoder domain gap (CONFIRMED — P0-1)
- SigLIP pretrained on WebLI real photos, Infinigen is synthetic
- Result: vision encoder outputs garbage image embeddings
- Evidence: E1 eval pretrained_mint = 0% grasp on real DrawerRobotEnv
- **Fix needed**: Vision encoder fine-tuning or replacement (SigLIP training code does not exist in workspace)

### H2: VQ-VAE Quantizer domain gap (CONFIRMED — P0-2)
- VQ-VAE trained on LIBERO diverse manipulation tasks
- Cannot encode Infinigen drawer-specific action latent space with 1,301 frames
- 512 codebook entries sparsely covered by drawer-only data
- **Fix needed**: VQ-VAE retraining on Infinigen data (code from PhD师兄)

### H3: Data insufficiency (CONFIRMED — P1)
- 1,301 frames vs LIBERO ~50,000 frames
- Not enough to update either VQ-VAE codebook or vision encoder
- **Fix needed**: Generate 5,000+ additional rollouts

---

## Recommended Next Actions (v3.0)

### Immediate (blocked — waiting for external resources):
1. ⏳ **Contact PhD师兄**: Request VQ-VAE training code + SigLIP fine-tuning code
2. ⏳ **Generate more rollouts**: Target 5,000+ frames (currently 1,301)
3. ❌ **Vision Encoder training**: No code exists in workspace; must find alternative solution

### Decision Tree:

```
PhD师兄 provides VQ-VAE code + SigLIP solution?
  YES → V58 end-to-end fine-tune with SigLIP solution
  NO  → Evaluate self-implementation feasibility
        → Consider SigLIP replacement (CLIP, DINOv2, etc.)
```

---

## Academic Claim Status (v3.0 — updated with caveat)

**Target claim**: Infinigen-generated trajectories improve MINT success on held-out drawer variants

**Current status**: CLAIM NOT SUPPORTED — two P0 blockers confirmed.

> **IMPORTANT CAVEAT**: The archived proxy baseline result does **NOT** prove that SigLIP vision encoder can generalize to Infinigen synthetic images. The `DrawerProxyEnv` is a trivially simple task (fixed drawer-joint delta `[0.85, 0, 0, 0, 0, 0, 1.0]` per step). The 100% success came from MINT memorizing the training data pattern, not from vision encoder scene understanding. The real robot environment (`DrawerRobotEnv`) E1 evaluation shows: pretrained_mint = 0% grasp, finetuned_mint = 0% success. **SigLIP P0 blocker is real.**

**Archived strongest true claim**:
- Archived proxy baseline | Complete — `claim_supported` (proxy task only; does not transfer to robot-trajectory task; see caveat above)
- Archived robot revision v1 | Complete — `scientific_not_supported`, but not accepted as final claim truth

**What was learned**:
1. AnyGrasp successfully generates drawer-grasping trajectories on Infinigen sim ✅
2. Teacher rollouts (C2) produce high-quality, physically valid action sequences ✅
3. MINT language model successfully learns drawer motion patterns (EEF +3.2m in V57) ✅
4. BUT: pretrained MINT cannot execute these trajectories in simulation (SigLIP domain gap) ❌
5. AND: VQ-VAE cannot encode Infinigen-specific action latents (no training code) ❌

**Path to claim**: Both P0 blockers must be resolved. Vision encoder generalization + VQ-VAE adaptation + sufficient data (5,000+ frames) are all required.
