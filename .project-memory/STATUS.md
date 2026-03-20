# Infinigen-AnyBox Project Status
<!-- Auto-maintained by infinigen-project-memory skill. Last updated: 2026-03-20 21:25 -->

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
- **Last Commit**: fe945088 average-fit improved, diversity worsened, physical consistency broken
- **Phase**: box_conditioning_v2::phase2_conditioning_design gate=ready_for_writeup
- **Active Work**: Unknown

## Completed Milestones

| # | Date | Milestone | Key Output | Commit |
|---|------|-----------|------------|--------|
| 1 | 2026-01-10 | Phase-0/1 initial implementation: MailerBox geometry + URDF export + modular factory | sim_exports/urdf/mailerbox_simple/ | 8f22ba48 |
| 2 | 2026-01-13 | TuckEndBox first successful generation (8-DOF) | sim_exports/urdf/tuckendbox/ | 550c721c |
| 3 | 2026-01-30 | MailerBox self-collision fix + PyBullet verification | sim_exports/urdf/mailerbox_simple/ (11 seeds verified) | 96cd1016 |
| 4 | 2026-03-02 | Phase-1 Data Engine complete + Phase-2/3 scaffolding for all 4 box types | scripts/export_mailerbox_simple_phase1_data_engine.py, infinigen/assets/sim_objects/box_material_domain_randomization.py, infinigen/assets/sim_objects/modular_box_factory.py, infinigen/perception/topo_box_net/, scripts/run_sim2real_ablation_matrix.py | 56bc0e43 |
| 5 | 2026-03-02 | Phase-1 URDF fix: collision meshes + material color tags + 1K dataset (v2) | sim_exports/data_engine/phase1_1k_mailer/, sim_exports/data_engine/phase1_1k_drawer/, sim_exports/data_engine/phase1_1k_sliplid/, sim_exports/data_engine/phase1_1k_tuckend/, sim_exports/urdf/mailerbox_simple/ (11 seeds, untouched), sim_exports/data_engine/material_diversity_proof.png | 35b99593 |
| 6 | 2026-03-04 | Phase-1 validation framework (6-level) -- comprehensive automated quality assurance | scripts/validate_dataset.py, tests/sim/test_validation_regression.py, sim_exports/data_engine/phase1_1k_*/verification_report.json | pending |
| 7 | 2026-03-09 | PhysNAP integration: CAPNet rollback + academic review + env setup + data bridge | external/physnap/, docs/PhysNAP_Academic_Review_and_Integration.md, scripts/infinigen_to_nap.py, scripts/merge_infinigen_nap_datasets.py, scripts/train_physnap_infinigen.sh, scripts/download_nap_data.sh, external/physnap/data/infinigen_graph_mailer/ (250 samples), external/physnap/data/infinigen_graph_drawer/ (250 samples), external/physnap/data/infinigen_graph_sliplid/ (250 samples), external/physnap/data/infinigen_graph_combined/ (750 samples) | pending |
| 8 | 2026-03-10 | PhysNAP training complete: baseline (44.5K iters) + Infinigen-only (60K complete) + fine-tune (30K) + shape encoding | scripts/encode_infinigen_shapes.py, scripts/evaluate_physnap_training.py, external/physnap/data/infinigen_graph_combined/infinigen_codebook.npz (1750x128), external/physnap/configs/nap/v6.1_diffusion_infinigen.yaml, external/physnap/configs/nap/v6.1_diffusion_finetune_infinigen.yaml, external/physnap/log/v6.1_diffusion_adapted/ (9 ckpts, 44.5K iters), external/physnap/log/v6.1_diffusion_infinigen/ (13 ckpts, 60K complete), external/physnap/log/v6.1_diffusion_finetune_infinigen/ | pending |
| 9 | 2026-03-17 | Implemented the box_prior_v1 K=10 workflow scaffold and launched the scratch Infinigen training job on the A800 host. | experiments/physnap/box_prior_v1/manifest.yaml, experiments/physnap/box_prior_v1/campaign_spec.md, scripts/physnap_experiment_launcher.py, scripts/physnap_auto_review_loop.py, scripts/evaluate_physnap_training.py | b240e09c |
| 10 | 2026-03-17 | Reconciled the completed K=10 scratch run, launched the true fine-tune run, and enabled run-to-completion polling for the box_prior_v1 campaign. | experiments/physnap/box_prior_v1/campaign_status.md, experiments/physnap/box_prior_v1/state.json, experiments/physnap/box_prior_v1/runtime/auto_loop_watch.log, experiments/physnap/box_prior_v1/runtime/v6.1_diffusion_finetune_infinigen_k10/run_meta.json | b240e09c |
| 11 | 2026-03-17 | Box-prior v1 evaluation pipeline completed with K=10 training/eval/loop wiring | experiments/physnap/box_prior_v1/manifest.yaml, experiments/physnap/box_prior_v1/evaluation/comparison_report.md, scripts/physnap_auto_review_loop.py | b240e09c |
| 12 | 2026-03-17 | Cleared the Infinigen->PhysNAP representation-fidelity blocker, regenerated repaired box data, and launched repaired scratch retraining. | experiments/physnap/box_prior_v1/diagnostics/conversion_audit.md, experiments/physnap/box_prior_v1/runtime/prepare_k10_summary.json, experiments/physnap/box_prior_v1/runtime/v6.1_diffusion_infinigen_k10_fixed/run_meta.json | b240e09c |
| 13 | 2026-03-18 | box_prior_v1 phase1_diagnostics verdict=phase1_diagnosis_ready decision=advance_phase | experiments/physnap/box_prior_v1/manifest.yaml, experiments/physnap/box_prior_v1/campaign_status.md, experiments/physnap/box_prior_v1/decision_memo.md | b240e09c |
| 17 | 2026-03-20 | box_conditioning_v2 phase2 progress 2/4 groups completed; multistate_singleview rerun active after invalid summary repair | experiments/physnap/box_conditioning_v2/state.json, experiments/physnap/box_conditioning_v2/review.json, experiments/physnap/box_conditioning_v2/campaign_status.md, .project-memory/history/2026-03-20_box-conditioning-v2-phase2-multistate-rerun-progress.md | b240e09c |
| 18 | 2026-03-20 | box_conditioning_v2 phase2_conditioning_design verdict=claim_not_supported decision=revise_claim | experiments/physnap/box_conditioning_v2/manifest.yaml, experiments/physnap/box_conditioning_v2/campaign_status.md, experiments/physnap/box_conditioning_v2/decision_memo.md | b240e09c |
| 20 | 2026-03-20 | Synchronized final Phase 2 verdict into project memory and recorded the strict root-cause tree | experiments/physnap/box_conditioning_v2/root_cause_tree.md, .project-memory/history/2026-03-20_box-conditioning-v2-phase2-final-verdict-claim-not-supported.md, .project-memory/STATUS.md | fe945088 |

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
| 9 | 2026-03-10 | pyrender OffscreenRenderer EGL crash on headless server -- wrapped viz in try/except |  |  | pending |
| 10 | 2026-03-17 | Fine-tune initialization failed for K=10 because the checkpoint positional embedding shape still assumed K=8. | network_dict.denoiser.p_pe depends on max_K and cannot be loaded verbatim from the K=8 checkpoint. | Added p_pe to logging.ignore_loading_key in the K=10 fine-tune config so the pretrained backbone still loads cleanly. | b240e09c |
| 11 | 2026-03-17 | Launcher metadata vanished after training started, making the loop think experiment status was missing. | PhysNAP recreates log/<exp> at startup and deleted run_meta.json/resolved_config.yaml written there ahead of launch. | Moved authoritative run metadata and resolved configs into campaign runtime sidecars and added lock-based status recovery. | b240e09c |
| 12 | 2026-03-17 | The first scratch run finished under the legacy launcher path and was left in a conservative stopped state. | That run predated exit-code sidecars, so completion had to be inferred from checkpoints and log evidence. | Added checkpoint/log-based completion reconciliation and promoted the scratch run to completed when it matched the expected full-duration trace. | b240e09c |
| 13 | 2026-03-17 | The loop still required manual retriggering after each long training stage. | The original run-until-blocked mode exited as soon as a background job started. | Added run-to-completion polling and launched it as a background watcher for the active campaign. | b240e09c |
| 14 | 2026-03-17 | Old fine-tune path did not initialize from the PartNet baseline checkpoint | The prior train script left option_c unimplemented and wrote only scratch configs | Added explicit K=10 fine-tune config + launcher metadata flow | b240e09c |
| 15 | 2026-03-17 | Per-box and combined K=10 datasets were still derived from the pre-fix converter that ignored URDF visual origins. | prepare_k10 only merged existing .npz files and never re-converted raw Phase-1 data after the converter changed. | prepare_k10 now rebuilds per-box datasets from raw exports and invalidates stale box reference caches when forced. | b240e09c |
| 16 | 2026-03-17 | Conversion audit treated visual-origin sensitivity itself as a blocker instead of checking whether the saved dataset matched the fixed converter. | The audit compared fixed vs buggy conversion variants but did not measure saved-vs-fixed fidelity. | Audit now reports saved-vs-fixed errors, saved alignment, and PartNet GT control metrics. | b240e09c |
| 17 | 2026-03-17 | PhysNAP audit scripts could crash on libstdc++/PIL binary mismatch before GT evaluation started. | The physnap environment sometimes loaded the system libstdc++ instead of the conda copy required by Pillow/libLerc. | Added an auto re-exec preload guard so audit/evaluator processes start with the conda libstdc++.so.6 preloaded. | b240e09c |
| 18 | 2026-03-20 | Phase 2 guided runs for multistate groups completed without required genfull_per_* metrics, which caused evaluate_conditioning_v2 to fail with invalid_phase2_summary. | The accelerated run_guided.py path skipped writing genfull metrics for multistate groups due to a misplaced code block, and the loop treated those runs as completed too early. | Patched run_guided.py to emit the full Phase 2 metric set again and tightened physnap_auto_review_loop.py so guided steps are only marked completed when stats.json contains the required conditioning metrics. | b240e09c |
| 19 | 2026-03-20 | Project-memory sync was only partially repaired: final campaign truth reached STATUS.md but not a dedicated terminal history snapshot, and post-commit still rewrote tracked files. | The memory contract mixed milestone sync with post-commit mutation, and patrol only checked for generic phase2 history instead of a final terminal snapshot. | Moved the hook to validate-only/log-only, tightened terminal history requirements, and backfilled the final Phase 2 verdict plus root-cause record into canonical memory artifacts. | fe945088 |

## Verified Artifacts

| Artifact | Path | Verification Command | Last Result |
|----------|------|---------------------|-------------|
| phase2_root_cause_tree | `experiments/physnap/box_conditioning_v2/root_cause_tree.md` | (run verification) | - |
| phase2_final_history | `.project-memory/history/2026-03-20_box-conditioning-v2-phase2-final-verdict-claim-not-supported.md` | (run verification) | - |
| project_status | `.project-memory/STATUS.md` | (run verification) | - |

## Next Steps

- Write the strongest true claim memo for Phase 2 instead of the original strong claim.
- Record which conditioning groups failed to improve the anchor and why.

## Key Files

- `experiments/physnap/box_conditioning_v2/root_cause_tree.md` -- phase2_root_cause_tree
- `.project-memory/history/2026-03-20_box-conditioning-v2-phase2-final-verdict-claim-not-supported.md` -- phase2_final_history
- `.project-memory/STATUS.md` -- project_status
- `experiments/physnap/box_conditioning_v2/state.json` -- active campaign state
- `experiments/physnap/box_conditioning_v2/review.json` -- active campaign review
- `experiments/physnap/box_conditioning_v2/decision_memo.md` -- active decision memo
- `experiments/physnap/box_conditioning_v2/campaign_status.md` -- active dashboard

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
- Training on 750 Infinigen samples converges to lower loss than 2340 PartNet-Mobility -- smaller, focused domain
- EGL rendering fails on headless AFS cluster -- osmesa also broken, need graceful fallback
- Fine-tuning from pretrained checkpoint needs lower LR (3e-5 vs 1e-4) to avoid catastrophic forgetting
- For box-domain K upgrades, checkpoint loading should explicitly ignore K-shaped tensors instead of pretending the run is a true full-parameter resume.
- Experiment status for long PhysNAP jobs must live outside the training log directory because the framework mutates that directory during startup.
- For unattended research loops, training orchestration needs both per-job exit tracking and a higher-level poller that advances the queue when GPUs go idle.
- A reviewable campaign dashboard is essential; otherwise the queue can be technically correct but still opaque to the human researcher.
- Box-domain evaluation needs custom GT export; PartNet save_gt.py does not cover Infinigen meshes
- K=10 can remain box-scoped without reopening the whole dynamic-K design space
- A converter bug that changes local frames invalidates all downstream training and evaluation artifacts; those artifacts must be treated as archival after regeneration.
- Box GT physics can be healthy even when PartNet GT exceeds an absolute E_pen/E_mob threshold, so those thresholds are not domain-universal.
- The repaired bridge is healthy, and the dominant remaining failure signal is catastrophic forgetting during pure fine-tune.
- Phase 2 claim review must be artifact-first: guided stats must be validated before the queue can advance to evaluate_conditioning_v2.
- The current strongest true claim is still pending because only 2 of 4 conditioning groups have finished with valid metrics.
- All bounded Phase 2 conditioning recipes completed, but none produced a consistent gain over the zero-state single-view anchor.
- Canonical tracked memory sync must happen at milestone finalization, not in post-commit hooks.
- Terminal campaign verdicts need a dedicated final history snapshot even if STATUS.md is already correct.
- The current Phase 2 negative result is best explained by conditioning-interface and architecture mismatch, not raw Infinigen asset failure.

## Live Campaign Snapshot

### box_conditioning_v2

- Phase: `phase2_conditioning_design`
- Gate: `ready_for_writeup`
- Verdict: `claim_not_supported`
- Decision: `revise_claim`
- Active Step: `None`
- Next Incomplete Step: `None`
- Queue: prepare_conditioning_v2=completed, export_cond_zero_singleview=completed, export_cond_zero_multiview=completed, export_cond_multistate_singleview=completed, export_cond_multistate_multiview=completed, run_guided_zero_singleview=completed, run_guided_zero_multiview=completed, run_guided_multistate_singleview=completed ...
- Progress: `11` / `11` steps complete
- Last Updated: `2026-03-20T19:36:21+08:00`
- Claim Assessment: All bounded Phase 2 conditioning recipes completed, but none produced a consistent gain over the zero-state single-view anchor.

### box_prior_v1

- Phase: `phase1_diagnostics`
- Gate: `auto_handoff_ready`
- Verdict: `phase1_diagnosis_ready`
- Decision: `advance_phase`
- Active Step: `None`
- Next Incomplete Step: `None`
- Queue: prepare_k10=completed, train_infinigen_k10=completed, train_finetune_k10=completed, prepare_mixed_replay=completed, train_mixed_finetune_k10=completed, train_partial_finetune_k10=completed, evaluate_all=completed
- Progress: `7` / `7` steps complete
- Last Updated: `2026-03-19T16:04:12+08:00`
- Claim Assessment: The repaired bridge is healthy, and the dominant remaining failure signal is catastrophic forgetting during pure fine-tune.
