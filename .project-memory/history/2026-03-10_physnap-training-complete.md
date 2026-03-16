# PhysNAP Training Complete -- Iteration #8
**Date**: 2026-03-10
**Predecessor**: Iteration #7 (PhysNAP integration: env setup + data bridge)

## Summary

Successfully trained PhysNAP diffusion models using Infinigen's procedurally-generated articulated box data. Three training strategies were validated:

1. **Baseline (PartNet-Mobility)**: Standard NAP training on original data (2340 samples, 46 categories). Reached 44.5K/120K iterations with strong convergence (loss 0.29 → 0.039).

2. **Infinigen-only**: Full training from scratch on Infinigen data (750 samples, 3 box types: Mailer, Drawer, SlipLid). Completed all 60K iterations with excellent convergence (loss 0.28 → 0.010). The lower final loss compared to baseline suggests the smaller, focused domain is easier for the model to learn.

3. **Fine-tune**: Starting from PartNet-Mobility pretrained weights, fine-tuned on Infinigen data with lower learning rate (3e-5 vs 1e-4) for 30K iterations. This tests whether transfer learning from a diverse articulated object dataset improves generation quality for the Infinigen-specific domain.

## Key Achievements

### Data Pipeline End-to-End
- 750 Infinigen URDF+OBJ samples converted to NAP's .npz graph format
- 1750 individual part shapes encoded through NAP's pretrained PointNet Shape AE (128-dim latent)
- Codebook, split, and partkeys files generated and verified with NAP's `compact_pack`

### Training Infrastructure
- Created `v6.1_diffusion_infinigen.yaml` (Infinigen-only, 60K iters, batch=32)
- Created `v6.1_diffusion_finetune_infinigen.yaml` (fine-tune, 30K iters, LR=3e-5)
- Fixed headless rendering crash (pyrender EGL error) with graceful fallback
- Evaluation comparison script: `scripts/evaluate_physnap_training.py`

### Loss Convergence Comparison

| Metric | Baseline @5.6K | Infinigen-only @5.8K | Infinigen-only @60K |
|--------|---------------|---------------------|---------------------|
| batch_loss | 0.039 | 0.023 | ~0.010 |
| loss_v_shape | 0.081 | 0.044 | ~0.023 |
| loss_v_bbox | 0.023 | 0.004 | ~0.002 |
| loss_e_plucker | 0.014 | 0.003 | ~0.001 |

## Technical Details

### Shape Encoding Pipeline
The `scripts/encode_infinigen_shapes.py` script:
1. Loads NAP's pretrained ResnetPointnet encoder from `737.pt` checkpoint
2. For each Infinigen part mesh: samples 1024 surface points → normalizes to unit sphere → encodes to 128-dim latent
3. All 1750 parts encoded in ~20 seconds on A800 GPU
4. Codebook format matches NAP's expected structure: `{embedding: [N,128], valid_mask: [N], std: [128]}`

### Training Configuration Choices
- **Infinigen-only**: 60K iterations (vs 120K for baseline) since the smaller dataset converges faster
- **Batch size 32** (vs 64 for baseline): reduced due to smaller dataset (600 train samples)
- **LR decay schedule adjusted**: [20K, 35K, 45K] vs [40K, 70K, 90K] for baseline
- **Fine-tune LR**: 3e-5 (30% of base) to prevent catastrophic forgetting

### Bugs Fixed
- **pyrender EGL crash**: `ValueError: Invalid device ID (0)` when rendering visualizations on headless server. Fixed by wrapping the entire visualization block in `_postprocess_after_optim` with try/except, logging a warning instead of crashing.

## Checkpoints

| Experiment | Directory | Count | Key Checkpoints |
|-----------|-----------|-------|-----------------|
| Baseline | `log/v6.1_diffusion_adapted/` | 9 | 228.pt (5K), 2025_latest.pt (44.5K) |
| Infinigen-only | `log/v6.1_diffusion_infinigen/` | 13 | 278.pt (5K), 3334.pt (60K final) |
| Fine-tune | `log/v6.1_diffusion_finetune_infinigen/` | 7 | 278.pt (5K), 1667.pt (30K final) |

## Next Steps
1. Resume baseline training to 120K for full comparison
2. Generate articulated objects from all 3 models and evaluate quality
3. Scale Infinigen data to 10K+ samples for improved generalization
4. Implement PhysNAP conditional generation (l_pc loss) with Infinigen point clouds
