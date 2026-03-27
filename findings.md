# Findings

## PhysNAP Box Conditioning (Phase 1-2)
- Phase 1 final diagnosis: `forgetting-dominant`.
- Phase 2 final verdict: `claim_not_supported`.
- Strongest true claim: Mixed replay is the strongest validated transfer improvement, but richer conditioning under the current PhysNAP parameterization does not consistently beat the zero-state single-view anchor.
- Root-cause ranking:
  1. conditioning interface mismatch
  2. PhysNAP conditional architecture mismatch
  3. lossy merged-observation construction risk
  4. metrics expose the failure pattern but are not the primary cause
- Locked interpretation: `average-fit improved, diversity worsened, physical consistency broken` is a real signal of interface/architecture mismatch, not evidence that multi-state information has no value.

## MINT Drawer Pipeline (mint_drawer_v1) — Full Pipeline Run 2026-03-25

### Academic Claim Under Test
> Infinigen-generated AnyGrasp-conditioned robot-arm drawer trajectories improve MINT success on held-out drawer variants in simulation relative to pretrained MINT.

### Pipeline Gate Results
| Gate | Status | Key Metric |
|------|--------|-----------|
| D1 (single rollout overfit) | PASSED | seed_009_ep02 extended_overfit, success_gain=0.2 |
| D2 (single seed overfit) | FAILED | extended_overfit showed 20% success (1/5) but below MIN_FINETUNED_SUCCESSES=3 |
| D3 (train-seed probe) | FAILED | finetuned 0% success vs pretrained 33%; success_gain=-0.333 |
| E1 (held-out claim eval) | COMPLETED | verdict=`scientific_not_supported` |

### E1 Held-Out Results (Seeds 11-15)
| Policy | Success Rate | Grasp Rate | Pull Distance | Best Seed |
|--------|-------------|------------|---------------|-----------|
| Random | 20% (1/5) | 60% | 0.524 | Seed 13: 100% |
| Pretrained MINT | 20% (1/5) | 40% | 0.196 | Seed 14: 100% |
| Finetuned MINT | **0% (0/5)** | **0%** | **0.000** | None |

### D3 Train-Seed Results (Seeds 2, 7, 9, 10)
| Policy | Success Rate | Pull Distance |
|--------|-------------|---------------|
| Pretrained MINT | 33% (4/12) | 0.331 |
| Finetuned MINT | **0% (0/12)** | **0.000** |

### Critical Finding: Catastrophic Forgetting
Fine-tuning MINT on our teacher data makes the model **strictly worse** than pretrained on every metric. Pretrained MINT already succeeds on some seeds (seed 7: 100%, seed 14: 100%) but fine-tuning destroys this capability entirely.

### Root Cause Analysis
1. **Layer 0 (FIXED)**: `attachment_local` was never set during evaluation — fixed by building per-seed grasp cache
2. **Layer 1 (ACTIVE)**: Catastrophic forgetting — fine-tuning on 4-6 rollouts from 4 seeds collapses pretrained generalization
3. **Layer 2 (ACTIVE)**: Teacher data quality — only 6 strict-valid rollouts from 4 seeds; far too little for meaningful BC
4. **Layer 3 (ACTIVE)**: Privileged teacher MDP mismatch — teacher uses `_planned_segment` + `target_fraction` that learner cannot observe
5. **Layer 4 (STRUCTURAL)**: Oracle grasp data for non-AnyGrasp seeds uses fixed quaternion, may not match per-seed handle geometry

### Verdict
`scientific_not_supported` — The current pipeline cannot support the academic claim. The fundamental bottleneck is insufficient training data (6 rollouts from 4 seeds) combined with catastrophic forgetting during fine-tuning.
