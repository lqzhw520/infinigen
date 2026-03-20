# PhysNAP Box Prior Evaluation

- Generated: 2026-03-18T20:58:35+08:00
- Campaign: box_prior_v1
- Box reference export: infinigen_box_k10_val
- Box metric reference: infinigen_box_k10_val_eval32
- Box metric reference size: 32
- PartNet sanity reference: GTval_eval64
- PartNet sanity reference size: 64

## Box-Domain Metrics

| Experiment | split | MMD | COV | 1NN-acc | E_pen | E_mob |
|---|---:|---:|---:|---:|---:|---:|
| PartNet baseline | multi | 0.3631 | 0.5312 | 0.9792 | 0.442120 | 0.454581 |
| PartNet baseline | zero | 0.3531 | 0.1250 | 0.9792 | - | - |
| Infinigen scratch K=10 (repaired) | multi | 0.4775 | 0.2812 | 1.0000 | 0.005582 | 0.009745 |
| Infinigen scratch K=10 (repaired) | zero | 0.5564 | 0.0938 | 1.0000 | - | - |
| PartNet to Infinigen fine-tune K=10 (repaired) | multi | 0.4406 | 0.3750 | 1.0000 | 0.014982 | 0.004753 |
| PartNet to Infinigen fine-tune K=10 (repaired) | zero | 0.5215 | 0.0938 | 0.9792 | - | - |
| PartNet + Infinigen mixed replay fine-tune K=10 | multi | 0.3211 | 0.5625 | 0.9583 | 0.130906 | 0.103068 |
| PartNet + Infinigen mixed replay fine-tune K=10 | zero | 0.2780 | 0.1250 | 0.9479 | - | - |
| PartNet to Infinigen partial fine-tune K=10 | multi | 0.4809 | 0.3125 | 0.9896 | 0.007567 | 0.004471 |
| PartNet to Infinigen partial fine-tune K=10 | zero | 0.5646 | 0.0938 | 1.0000 | - | - |

## PartNet Sanity

| Experiment | split | MMD | COV | 1NN-acc |
|---|---:|---:|---:|---:|
| PartNet baseline | multi | 0.3171 | 0.4844 | 0.5547 |
| Infinigen scratch K=10 (repaired) | multi | 0.5306 | 0.1562 | 1.0000 |
| PartNet to Infinigen fine-tune K=10 (repaired) | multi | 0.4945 | 0.2031 | 0.9375 |
| PartNet + Infinigen mixed replay fine-tune K=10 | multi | 0.3394 | 0.4062 | 0.7188 |
| PartNet to Infinigen partial fine-tune K=10 | multi | 0.5247 | 0.2188 | 0.9766 |

## Conversion Audit

- Decision: `healthy`
- Box GT self-eval pen mean: `0.0`
- Box GT self-eval mob mean: `4.3893143768514165e-06`
- PartNet GT self-eval pen mean: `3.8836544102287576`
- PartNet GT self-eval mob mean: `3.9299919274159834`
- Saved bbox vs fixed converter max error: `2.9802322387695312e-08`
- Visual-origin bbox effect max: `0.0019005537033081055`

## Sample Galleries

- `PartNet baseline`: `/mnt/afs2/zhuhaowu/infinigen/external/physnap/log/box_prior_v1__baseline/res_chnk01.png`
- `Infinigen scratch K=10 (repaired)`: `/mnt/afs2/zhuhaowu/infinigen/external/physnap/log/box_prior_v1__infinigen_k10/res_chnk01.png`
- `PartNet to Infinigen fine-tune K=10 (repaired)`: `/mnt/afs2/zhuhaowu/infinigen/external/physnap/log/box_prior_v1__finetune_k10/res_chnk01.png`
- `PartNet + Infinigen mixed replay fine-tune K=10`: `/mnt/afs2/zhuhaowu/infinigen/external/physnap/log/box_prior_v1__mixed_finetune_k10/res_chnk01.png`
- `PartNet to Infinigen partial fine-tune K=10`: `/mnt/afs2/zhuhaowu/infinigen/external/physnap/log/box_prior_v1__partial_finetune_k10/res_chnk01.png`
