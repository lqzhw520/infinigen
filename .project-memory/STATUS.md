# Infinigen-AnyBox Project Status
<!-- Auto-maintained by infinigen-project-memory skill. Last updated: 2026-03-10 17:30 -->

## Architecture

```
Phase 1: Data Engine          Phase 2: Perception           Phase 3: Planning
┌─────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
│ 1.1 Box Topology    │      │ 2.1 Topo-Box-Net     │      │ 3.1 Sim2Real Ablation│
│ 1.2 Inertia Fix     │─────>│ 2.2 Kin-Consist Loss │─────>│ 3.2 VAMP Integration │
│ 1.3 DR Materials    │      │                      │      │                      │
│ 1.4 Annotation Pipe │      └──────────────────────┘      └──────────────────────┘
└─────────────────────┘
```

## Current State

- **Branch**: feature/3d-assets
- **Last Commit**: d6c74c0a PhysNAP integration: CAPNet rollback + academic review + data bridge pipeline
- **Phase**: PhysNAP training completed with Infinigen data (60K iters Infinigen-only + 30K fine-tune)
- **Active Work**: Evaluation and comparison of trained models

## Completed Milestones

| # | Date | Milestone | Key Output | Commit |
|---|------|-----------|------------|--------|
| 1 | 2026-01-10 | Phase-0/1 initial implementation: MailerBox geometry + URDF export + modular factory | sim_exports/urdf/mailerbox_simple/ | 8f22ba48 |
| 2 | 2026-01-13 | TuckEndBox first successful generation (8-DOF) | sim_exports/urdf/tuckendbox/ | 550c721c |
| 3 | 2026-01-30 | MailerBox self-collision fix + PyBullet verification | sim_exports/urdf/mailerbox_simple/ (11 seeds verified) | 96cd1016 |
| 4 | 2026-03-02 | Phase-1 Data Engine complete + Phase-2/3 scaffolding for all 4 box types | scripts/export_mailerbox_simple_phase1_data_engine.py, infinigen/assets/sim_objects/box_material_domain_randomization.py, infinigen/assets/sim_objects/modular_box_factory.py, infinigen/perception/topo_box_net/, scripts/run_sim2real_ablation_matrix.py | 56bc0e43 |
| 5 | 2026-03-02 | Phase-1 URDF fix: collision meshes + material color tags + 1K dataset (v2) | sim_exports/data_engine/phase1_1k_mailer/, sim_exports/data_engine/phase1_1k_drawer/, sim_exports/data_engine/phase1_1k_sliplid/, sim_exports/data_engine/phase1_1k_tuckend/, sim_exports/urdf/mailerbox_simple/ (11 seeds, untouched), sim_exports/data_engine/material_diversity_proof.png | 35b99593 |
| 6 | 2026-03-04 | Phase-1 validation framework (6-level) -- comprehensive automated quality assurance | scripts/validate_dataset.py, tests/sim/test_validation_regression.py, sim_exports/data_engine/phase1_1k_*/verification_report.json | pending |
| 7 | 2026-03-09 | PhysNAP integration: CAPNet rollback + academic review + env setup + data bridge | external/physnap/, docs/PhysNAP_Academic_Review_and_Integration.md, scripts/infinigen_to_nap.py | d6c74c0a |
| 8 | 2026-03-10 | PhysNAP training complete: baseline (44K iters) + Infinigen-only (60K iters, loss 0.28→0.010) + fine-tune (30K iters) + shape encoding (1750 parts) | external/physnap/log/v6.1_diffusion_infinigen/ (13 ckpts), scripts/encode_infinigen_shapes.py, scripts/evaluate_physnap_training.py | pending |

## PhysNAP Training Results

| Experiment | Config | Data | Iterations | Final Loss | Checkpoints |
|-----------|--------|------|-----------|------------|-------------|
| Baseline (PartNet-Mobility) | v6.1_diffusion_adapted | 2340 samples, 46 categories | 63.2K / 120K | ~0.039 (at 5.6K) | 13 saved (228-2875) |
| Infinigen-only | v6.1_diffusion_infinigen | 750 samples, 3 box types | 60K / 60K (COMPLETE) | ~0.010 | 13 saved (278-3334) |
| Fine-tune | v6.1_diffusion_finetune_infinigen | 750 samples, pretrained init | 30K / 30K (COMPLETE) | adapting from 0.37 | 7 saved (278-1667) |

## Bug Fixes & Lessons

| # | Date | Bug | Root Cause | Fix | Commit |
|---|------|-----|-----------|-----|--------|
| 1 | 2026-01-30 | Multiple <inertial> tags per link in URDF | urdf_exporter wrote one <inertial> per Blender object instead of merging per link | Implemented Parallel Axis Theorem merging in urdf_exporter.py | 96cd1016 |
| 2 | 2026-01-30 | Self-collision not properly configured | URDF lacked self-collision flags; PyBullet soft joint limits allow penetration under external force | Added URDF_USE_SELF_COLLISION flag + verified joint limits | 96cd1016 |
| 3 | 2026-03-02 | URDF missing collision meshes -- joints spin infinitely in online viewers | visual_only=True in urdf_exporter.export() call | Changed to visual_only=False to include <collision> tags and *_col0.obj assets | 35b99593 |
| 4 | 2026-03-02 | No material/color in urdf_gt.urdf -- shows white in URDF viewers | urdf_exporter had no <material> tag injection | Post-processing: _inject_material_color_into_urdf() adds <material><color rgba> from DR config | 35b99593 |
| 5 | 2026-03-02 | imageio SystemError: tile cannot extend outside image at 1024x768 | imageio.imwrite bug with high-res segmentation PNGs | Replaced with PIL.Image.fromarray().save() | 35b99593 |
| 6 | 2026-03-02 | Parallel batch processes overwrite samples (counter collision) | Multiple box types sharing same out_root with overlapping sample_counter | Separate output directories per box type (phase1_1k_mailer, etc.) | 35b99593 |
| 7 | 2026-03-04 | joint_positions stored as dict but accessed as list in validator | export pipeline writes {name: val} dict, validator assumed list | handle both dict and list formats in validate_dataset.py | pending |
| 8 | 2026-03-04 | depth background pixels (1e10) flagged as error | Blender Z-pass uses large values for sky/background | filter foreground pixels (< 100m) before range check, allow bg ratio | pending |

## Verified Artifacts

| Artifact | Path | Status |
|----------|------|--------|
| physnap_repo | `external/physnap/` | PASS -- cloned + env setup |
| academic_review | `docs/PhysNAP_Academic_Review_and_Integration.md` | PASS |
| data_bridge | `scripts/infinigen_to_nap.py` | PASS -- 750 samples converted |
| shape_encoder | `scripts/encode_infinigen_shapes.py` | PASS -- 1750 parts encoded |
| evaluation_script | `scripts/evaluate_physnap_training.py` | PASS |
| nap_data_combined | `external/physnap/data/infinigen_graph_combined/` | PASS -- 750 samples |
| infinigen_codebook | `infinigen_graph_combined/infinigen_codebook.npz` | PASS -- 1750x128 |
| training_baseline | `external/physnap/log/v6.1_diffusion_adapted/` | PASS -- 63.2K iters (53%) |
| training_infinigen | `external/physnap/log/v6.1_diffusion_infinigen/` | PASS -- 60K iters COMPLETE |
| training_finetune | `external/physnap/log/v6.1_diffusion_finetune_infinigen/` | PASS -- 30K iters COMPLETE |

## Next Steps

- [P0] Resume PartNet-Mobility baseline training to 120K iters for full comparison
- [P0] Generate and evaluate articulated objects from all 3 trained models
- [P1] Scale Infinigen data to 10K+ samples for improved generalization
- [P1] Implement PhysNAP conditional generation with Infinigen point clouds
- [P2] Phase 2.1: Topo-Box-Net training

## Key Files

- `external/physnap/` -- PhysNAP repo (cloned + env)
- `docs/PhysNAP_Academic_Review_and_Integration.md` -- Academic review
- `scripts/infinigen_to_nap.py` -- URDF+OBJ -> NAP graph converter
- `scripts/encode_infinigen_shapes.py` -- Shape AE encoder for Infinigen meshes
- `scripts/evaluate_physnap_training.py` -- Cross-experiment evaluation
- `scripts/merge_infinigen_nap_datasets.py` -- Multi-type dataset merger
- `external/physnap/data/infinigen_graph_combined/` -- 750 samples + codebook
- `external/physnap/configs/nap/v6.1_diffusion_infinigen.yaml` -- Infinigen-only config
- `external/physnap/configs/nap/v6.1_diffusion_finetune_infinigen.yaml` -- Fine-tune config
- `external/physnap/log/v6.1_diffusion_infinigen/` -- Infinigen training (60K, COMPLETE)
- `external/physnap/log/v6.1_diffusion_adapted/` -- Baseline training (44.5K/120K)


## Lessons Learned (cumulative)

- Blender coordinate system differs from URDF -- transform required
- TuckEndBox has 8 DOF -- joint_states parameter must exactly match DOF count
- PyBullet joint limits are 'soft' -- external forces can exceed them
- Each URDF link must have exactly one <inertial> tag with merged properties
- Always verify with PyBullet URDF_USE_INERTIA_FROM_FILE | URDF_USE_SELF_COLLISION
- Use local RNG (not global np.random) for deterministic DR material sampling
- planning-with-files skill keeps task-level context across iterations
- visual_only=True causes URDF to lack collision meshes -- downstream planner cannot use
- URDF needs <material><color rgba> tags for color display in online viewers
- imageio crashes on high-res segmentation PNGs -- use PIL instead
- Never share out_root between parallel batch processes -- sample counter collides
- Always verify with both online URDF viewer AND PyBullet
- URDF viewers (online) cannot render OBJ textures -- material colors visible only as URDF <material><color> tags
- PyBullet loadURDF resolves mesh paths relative to URDF file location
- joint_positions format varies by export version (dict vs list) -- always handle both
- Blender depth Z-pass uses ~1e10 for background -- not an error, just far-plane value
- Self-consistency check (GT vs GT) is the best first test for any metrics pipeline
- NAP uses max K=8 nodes -- TuckEndBox (9 parts) exceeds this and needs K=10 or exclusion
- Google Drive is unreachable from AFS cluster -- need manual download or proxy
- Infinigen-only (750 samples) converges to lower loss than PartNet-Mobility (2340) -- focused domain is easier
- pyrender EGL fails on headless cluster -- wrap viz in try/except for graceful degradation
- Fine-tuning from pretrained needs lower LR (3e-5 vs 1e-4) to avoid catastrophic forgetting
- NAP shape AE encoder (ResnetPointnet) takes 1024 surface points normalized to unit sphere
