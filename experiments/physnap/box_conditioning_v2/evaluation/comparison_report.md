# Phase 2 Conditioning Comparison

**Updated**: 2026-03-20T19:35:24+08:00
**Base Experiment**: `mixed_finetune_k10`
**Base Checkpoint**: `/mnt/afs2/zhuhaowu/infinigen/external/physnap/log/v6.1_diffusion_mixed_finetune_infinigen_k10/checkpoint/425_latest.pt`
**Claim Verdict**: `claim_not_supported`

| Group | MMD | COV | 1NN | E_pen | E_mob | Cond Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| zero_singleview | 0.5589997228235006 | 0.0908203125 | 0.9250000044703484 | 0.0007139240296964999 | 0.0005797943085781299 | 0.0022055536246625707 |
| zero_multiview | 0.5496429605409503 | 0.08984375 | 0.9241071473807096 | 0.0007759372820146382 | 0.0005648964070132934 | 0.0014058477245271206 |
| multistate_singleview | 0.5471300091594458 | 0.0908203125 | 0.9250000044703484 | 0.0022372949169948697 | 0.0006206683465279639 | 0.0013572014092157285 |
| multistate_multiview | 0.5368264261633158 | 0.0888671875 | 0.9214285761117935 | 0.0091311283952867 | 0.010760995797075642 | 0.0014382785884663463 |

## Interpretation

None of the richer conditioning settings consistently beat the zero-state single-view anchor under the bounded Phase 2 recipes.

## Claim Ladder

- `multi-state + multi-view beats zero-state single-view` via `multistate_multiview`: not supported
- `multi-view only beats zero-state single-view` via `zero_multiview`: not supported
- `multi-state only beats zero-state single-view` via `multistate_singleview`: not supported

## Strongest True Claim

Richer conditioning did not consistently beat the zero-state single-view anchor under the bounded Phase 2 recipes.
