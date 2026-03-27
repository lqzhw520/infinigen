# PRD: Infinigen Sim Data → MINT Training Pipeline

**Project**: feature/mint-integration
**Branch**: `feature/mint-integration` (from `feature/3d-assets`)
**Note**: Infinigen export code remains on `feature/3d-assets`. All MINT integration code on `feature/mint-integration`.
**Author**: zhuhaowu
**Last Updated**: 2026-03-26
**Status**: Active phase = `robot_revision_v3_claim_push` in automatic-pipeline recovery mode; old `D2/D3/E1` run archived as `diagnostic_only_after_gate_drift`; mainline reset to `teacher_contract_rebuild`

---

## 1. Objective

Use Infinigen-generated articulated objects as simulation inputs to generate manipulation-relevant trajectories, then use these trajectories to fine-tune MINT for improved task transfer on novel articulated objects.

**Phase scope clarification**: This PRD defines the **simulation-only** pipeline. Real-robot transfer is a future phase.

**Deliverable (active phase)**:
1. Infinigen articulated drawer objects loadable in the validated robot simulation path
2. AnyGrasp detection produces handle-region grasp candidates from rendered robot-scene depth / point clouds
3. Oracle grasp and AnyGrasp grasp are both tested with scripted robot open baselines
4. Natural robot-arm trajectories are recorded and converted to MINT-compatible delta actions
5. Only natural replay-valid rollouts are packaged as a LeRobot v3 dataset
6. MINT is fine-tuned on this dataset for 1000 steps
7. **Held-out sim eval only after train-seed reproducibility is positive**

**What this does NOT deliver (future work)**:
- Real-robot deployment
- True demo-conditioned one-shot transfer (see Section 11 for the gap)
- Tracking-conditioned grasp persistence
- A publishable held-out claim before train-seed reproducibility is demonstrated

### 1.1 Archived Prior Result (proxy-control baseline)

The earlier proxy-control simulation campaign remains archived as a baseline:

- Train split: drawerbox seeds `1-10`
- Held-out split: drawerbox seeds `11-15`
- Dataset built from generated rollouts: `20` episodes, `220` total frames
- Fine-tuning: `1000` training steps completed, checkpoint written
- Held-out simulation comparison:

| Policy | Successes | Success Rate | Pull Distance | Time to Completion |
| --- | ---: | ---: | ---: | ---: |
| Random | 0 / 5 | 0.000 | 0.332 | 24.0 |
| Pretrained MINT | 0 / 5 | 0.000 | 0.432 | 24.0 |
| Fine-tuned MINT | 5 / 5 | 1.000 | 0.912 | 12.0 |

**Archived strongest true claim**:

> Fine-tuned MINT improves held-out drawer success in the proxy simulation.

### 1.2 Current Active Phase (robot_revision_v3_claim_push)

The active campaign is no longer the proxy path above. It is now the **AnyGrasp + robot-trajectory claim-push** phase, with a dual-track execution model:

- `strict replay lane`: keep repairing `absolute -> delta -> native replay` fidelity
- `learnability lane`: once teacher-success fallback is contract-preflight-clean, allow `C3 -> D1 -> D2 -> D3 -> E1`

Current archived negative evidence from `robot_revision_v1_failed`:

- Held-out seeds `11-15`: `finetuned_mint = 0/5`, `pretrained_mint = 0/5`
- Train-seed probe: `finetuned_mint = 1/6`, `pretrained_mint = 0/6`
- Root-cause hypotheses:
  - G4/G5 data contract was partially non-physical
  - `force_attach` contaminated training rollouts
  - held-out evaluation was attempted before train-seed reproducibility was established

**Current execution rule**:

> No held-out claim will be called until train-seed reproducibility is positive, and teacher-success fallback is now treated as a legitimate learnability source while strict replay continues in parallel.

**Additional current rule after gate-drift review**:

> `D1` diagnostic success is not sufficient to advance the mainline. `D2` may only start from a `D2-feasible` seed with at least two strict-valid rollouts, `D3` requires a hard `D2` pass, and `E1` is invalid as a final verdict unless that hard gate chain remains intact.

---

## 2. System Architecture

```
Archived baseline path:

Phase 1: Asset Gen          Phase 2: Proxy Sim         Phase 3: Trajectory        Phase 4: Training/Eval
┌──────────────┐         ┌──────────────────┐       ┌──────────────────┐      ┌────────────────────────┐
│  Infinigen    │  URDF   │  DrawerProxyEnv  │  obs  │ Proxy expert      │      │  LeRobot + MINT        │
│  DrawerBox    │──────→  │  + dual camera   │──────→│ rollouts          │─────→│  fine-tune + held-out  │
│  seeds 1-15   │         │  + EGL headless  │       │ delta actions     │      │  sim evaluation        │
└──────────────┘         └──────────────────┘       └──────────────────┘      └────────────────────────┘
  ✅ Archived baseline      ✅ Archived baseline       ✅ Archived baseline       ✅ Archived baseline

Active root-cause path:
Infinigen drawer asset -> robot sim scene -> AnyGrasp / oracle grasp audit -> natural robot rollout -> G5 replay contract -> MINT overfit ladder -> held-out sim eval
```

---

## 3. Environment

| Component | Version | Location | conda env | Status |
|-----------|---------|----------|-----------|--------|
| A800 GPU | NVIDIA A800-SXM4-80GB | ssh -p 30017 root@10.210.0.88 | — | ✓ |
| Infinigen | 1.19.0 | /mnt/afs2/zhuhaowu/infinigen/ | infinigen | ✓ |
| MINT | 0.1.0 | external/MINT/ | mint | ✓ |
| lerobot | 0.4.3 | pip (mint) | mint | ✓ |
| robosuite | 1.4.0 | pip (mint) | mint | ✓ |
| MuJoCo | 3.6.0 | pip (mint) | mint | ✓ |
| PyTorch (mint) | 2.3+ | pip (mint) | mint | ✓ CUDA |
| AnyGrasp SDK | official SDK | external/anygrasp_sdk/ | graspnet | ✓ staged |
| Contact-GraspNet | PyTorch ver. | external/contact_graspnet_pytorch/ | graspnet | archived auxiliary |
| PyTorch (graspnet) | 2.5.1+cu121 | pip (graspnet) | graspnet | ✓ CUDA |

---

## 4. Cross-Environment Artifact Contract

**This is the strict contract for what each environment can produce and consume. Violations will be caught at these boundaries.**

### Environment A: infinigen (conda: infinigen)
- **PRODUCES ONLY**: URDF files, OBJ meshes, metadata.json, joint_state.json
- **MUST NOT**: install MINT/robosuite deps, modify external/physnap/
- **Output dir**: `sim_exports/urdf/{asset_type}/{seed}/`

### Environment B: mint (conda: mint)
- **CONSUMES**: URDF + OBJ from env A
- **PRODUCES**: robosuite obs (images + state), action sequences, LeRobot datasets
- **Contains**: robosuite, MuJoCo, lerobot, MINT, libero

### Environment C: graspnet (conda: graspnet)
- **CONSUMES**: point clouds (N×3 float32) from env B
- **PRODUCES**: grasp candidates — `pred_grasps_cam` (B,N,4,4) + `pred_scores` (B,N,1)
- **Contains**: AnyGrasp SDK, open3d, torch, and supporting perception deps
- **MUST NOT**: touch env A or env B code/data

### Artifact format specifications:

| From → To | Format | File/Variable |
|-----------|--------|---------------|
| A → B | URDF + OBJ + metadata.json | `sim_exports/urdf/.../` |
| B → C | point_cloud.npz (N×3 float32) + intrinsics.npy (3×3) | temp files |
| C → B | grasp_candidates.json (list of {pose_4x4, score, width}) | temp files |
| B → LeRobot | v3 parquet + MP4 videos | `infinigen_articulated_dataset/` |

---

## 5. Verified Foundations

### 5.1 MINT: 25/26 checks passed

| Check | Result | Detail |
|-------|--------|--------|
| paligemma_with_expert | PASS | PaliGemma 2B backbone + Gemma 300M expert |
| multi_scale_vqvae | PASS | SDAT multi-scale VQ-VAE tokenizer |
| codebook | PASS | 512 codes × 32 dim |
| patch_nums | PASS | [1, 2, 4] scale hierarchy |
| safetensors | PASS | 7.05GB, 1034 keys |
| weights_loaded | PASS | 0 unexpected; embed_tokens missing (expected) |
| weights_not_random | PASS | std=0.0229 |
| tokenizer | PASS | vocab 257152 |
| inference | PASS | 3.7 steps/s, GPU 7.1GB |
| **action_grip_varies** | **FAIL** | Gripper constant in 10-step test (see Known Issues) |
| **closed-loop success** | **FAIL** | Task success = False in test (see Known Issues) |

### 5.2 Infinigen URDF: VERIFIED ✓

| Asset | Joints | Types | Ranges | Status |
|-------|--------|-------|--------|--------|
| drawerbox/42 | 1 | slide | [0, 0.254m] | ✓ |
| mailerbox_simple/101 | 2 | hinge | [-π, π] | ✓ |
| 1000+ drawer variants | — | — | — | ✓ available |

### 5.3 URDF → robosuite MJCF: VERIFIED ✓

- MuJoCo loads URDF via VFS ✓
- MJCF conversion preserves joint properties ✓
- robosuite MujocoXMLObject loads ✓
- Joint movement verified ✓

### 5.4 AnyGrasp: DRAWER-SCENE STAGING VERIFIED ✓

| Check | Result |
|-------|--------|
| SDK staging | PASS — `gsnet.so`, `lib_cxx.so`, license, checkpoint path staged |
| `load_net()` | PASS |
| Smoke inference | PASS — non-empty grasp candidates returned |
| Active contract | score threshold and handle-region hit are enforced in root-cause repair |

Archived auxiliary note: Contact-GraspNet remains available as prior perception groundwork, but the active robot-trajectory phase uses AnyGrasp detection.

### 5.5 Headless Rendering: VERIFIED ✓
- EGL rendering on A800, no display needed

### 5.6 Current Active Phase Status

| Gate | Result |
|------|--------|
| Archived proxy baseline | Complete — `claim_supported` |
| Archived robot revision v1 | Complete — `scientific_not_supported`, but not accepted as final claim truth |
| Archived gate-drift recovery run | `D1` passed diagnostically, but `D2/D3/E1` were reclassified as `diagnostic_only_after_gate_drift` |
| Active robot revision v3 | In automatic-pipeline recovery mode; `teacher_contract_rebuild -> action_contract_repair -> replay_gate -> D1 mainline -> D2 -> D3 -> E1` |

---

## 6. Current Root-Cause Focus

### Historical Negative Signals

These are no longer treated as “warnings only”; they directly define the active repair ladder.

| Signal | Detail | Implication |
|--------|--------|-------------|
| **Gripper action constant (10-step smoke test)** | Early smoke tests observed no gripper change in the first 10 executed steps. | This is **not by itself a data-normalization bug**. Current teacher actions are already in `[-1, 1]`, and the gripper channel is binary `{-1, +1}`. For representative oracle / AnyGrasp rollouts, the first close event occurs around step `34`, so a 10-step smoke test can miss the grasp phase entirely. Future checks must evaluate windows that include the `close -> attach_hold -> pull` segment rather than only the first few approach steps. |
| **Train seeds only 1/6** | The archived robot-trajectory v1 revision only reached `1/6` fine-tuned success on train seeds. | Held-out failure cannot yet be treated as the main conclusion; train reproducibility is the first blocker. |
| **force_attach contamination** | G5 used `force_attach=True` while generating training data. | Training data may have violated the real closed-loop contract. |
| **G4 partial non-physicality** | G4 injected scripted drawer updates during rollout generation. | A runnable rollout is not enough; the rollout must be naturally replay-valid. |

### Remaining Caveats

1. The active phase is still **simulation-only**, not real robot.
2. The archived proxy result is useful baseline evidence, but it does not satisfy the advisor’s robot-trajectory requirement.
3. The active robot-trajectory phase still needs to prove:
   - oracle scripted open success
   - AnyGrasp scripted open success
   - natural G5 replay validity
   - train-seed positive trend
4. `u3_handle_region_audit` remains an **upstream suspect**:
   - Infinigen exports do not currently provide reliable handle part labels.
   - Current AnyGrasp handle localization still depends on heuristic handle-center inference.
   - Observed handle-center offsets on successful seeds are on the order of `0.126m - 0.279m`.
5. `B1/B2` currently prove **task feasibility**, but not report-grade video quality:
   - the stereo videos can show washed-out rendering and clipped wrist-camera geometry,
   - so teacher videos now need a separate `report_safe` rendering path for presentation.
4. Only after those pass does held-out transfer become a meaningful claim target.

### 6.1 Risk Register (Current Top 4)

#### Confirmed Fixed

1. **Evaluation attach-state bug**
   - Historical issue: evaluation never restored `attachment_local`, making strict success structurally impossible during policy rollout.
   - Current status: fixed; no longer the active blocker.

2. **Gate-drift execution bug**
   - Historical issue: a diagnostic `D1` winner was allowed to flow into `D2`, and `D3/E1` advanced after soft gates.
   - Current status: fixed in the recovery path; `D2/D3/E1` are no longer allowed to advance under that relaxed contract.

#### Confirmed Open

3. **Strict replay remains fully failing**
   - Current evidence: `C2` completed only via `teacher_success_fallback`, while open-loop strict replay still fails on every rollout.
   - Implication: learnability can continue, but the scientific cleanliness of the action contract remains unresolved.

4. **Upstream handle semantics remain weak**
   - Current evidence: `u3_handle_region_audit` still shows missing handle part labels and heuristic handle-center offsets of roughly `0.126m - 0.279m`.
   - Implication: AnyGrasp-conditioned attach behavior is still being stabilized by compensatory branch logic rather than precise semantic handle localization.

5. **Control-plane concurrency can contaminate live evidence**
   - Current evidence: multiple supervisors / `run_d1_single_rollout_overfit.py` wrappers / `lerobot-train` jobs have overlapped on the same campaign.
   - Implication: post-contamination D1 results must be forensically reclassified and cannot be used as clean scientific evidence until a single-controller lease is enforced.

6. **Historical rollout drift can invalidate naive data-driven ranking**
   - Current evidence: the archived diagnostic `seed_010_episode_04` that produced a real positive D1 signal is not identical to the current `c2_replay_valid_rollouts/seed_010_episode_04.npz`.
   - Archived diagnostic copy: `96` steps, phase counts `{grasp: 15, hold_close: 10, pull: 7}`, `handle_distance_min ~= 0.0997`.
   - Current active copy: `80` steps, phase counts `{grasp: 1, hold_close: 10, pull: 2}`, `handle_distance_min ~= 0.0250`.
   - Implication: historical D1 success cannot be attributed to current `handle_distance_min` alone, and ranking changes must compare like-for-like rollout versions.

7. **Teacher lineage was not part of the earlier recovery contract**
   - Historical mistake: previous recovery plans tracked `controller_id/run_id` and dataset/checkpoint fingerprints, but not immutable teacher-rollout lineage.
   - Concrete failure mode: `(seed, episode_index)` was implicitly treated as a stable teacher identity even though rebuilt `c1/c2` rollouts can change step count, phase counts, and handle-distance trace.
   - Required correction: future D1/D2 decisions, historical comparisons, and checkpoint reuse must be keyed by rollout fingerprint / semantic summary, not seed-episode name alone.

#### Watch But Not Primary

- **Render / camera report quality**
  - Current evidence: `B1/B2` stereo videos can still show washed-out primary views and clipped wrist-camera geometry.
- **Handle-distance-only ranking**
  - Current evidence: smaller `handle_distance_min` does correlate with easier attach in some runs, but the strongest apparent support mixed archived and current versions of `seed_010_episode_04`.
  - Implication: `handle_distance_min` should be treated as an attachment-oriented signal, not yet as a standalone learnability metric.

### 6.2 Mainline Recovery Rule

The automatic mainline now follows a staged validation rule rather than an all-or-nothing D1 search:

1. **Single-controller clean run only**
   - Any post-cutoff results produced under concurrent control are archived as contaminated and are not valid claim evidence.
2. **Stage A candidate screen**
   - Evaluate only grasp-rich, `D2-feasible` mainline candidates with `base_overfit`.
   - If neither candidate shows any attach-positive signal, immediately trigger the `u3` handle-metadata A/B lane.
3. **Stage B attach-focused screen**
   - Run `phase_balanced_overfit` and `attach_curriculum_overfit` only on the stronger Stage A candidate.
4. **Promotion**
   - Only a clean Stage B positive result may produce a `promoted_mainline_candidate`.
5. **D2 hard gate**
   - D2 may only start from a clean promoted mainline candidate whose seed has at least two strict-valid rollouts.
  - Implication: this is important for reporting quality and may reflect scene/camera debt, but it is not currently the primary claim blocker.

### 6.2 Engineering Failure Modes

| Failure mode | Detection | Current status |
|---|---|---|
| Supervisor loop dead while queue still shows pending steps | no live `mint_auto_review_loop` / worker PIDs, stale `overnight_loop.log`, stale `state.active_runtime` | **Open operational risk** |
| Artifact truth diverges from dashboard truth | compare step artifact JSON against `state.json`, `watch_status.json`, `campaign_status.md` | **Partially fixed** |
| Invalid gate progression | verify `source_rollout_count`, hard-gate prerequisites, and queue transitions | **Fixed in recovery path** |
| Scientific failure confused with engineering outage | require process check + artifact timestamp check before interpreting a failed/pending step | **Partially fixed** |

### 6.3 Strict Replay Interpretation

The project now treats strict replay as a **scientific lane**, not as a hidden prerequisite that can be silently weakened.

- The authoritative strict-replay metric remains the original **open-loop replay**.
- Additional replay variants may be used for diagnosis only.
- In particular, state-restoring replay can help distinguish cumulative drift from deeper contract mismatch, but it must **not** replace the open-loop strict-replay definition in claim-facing conclusions.

### Open Questions (from GPT-5.4 review)

| # | Question | Impact |
|---|----------|--------|
| 1 | Does the current MINT open-source stack support custom LeRobot fine-tuning? | **Resolved** — yes, both proxy and robot-trajectory pipelines can train and write checkpoints |
| 2 | Where is the main failure: assets/sim, AnyGrasp, G5 contract, or learning? | **Active** — this is the purpose of the current root-cause ladder |
| 3 | How many natural successful robot rollouts are needed before train-seed trend stabilizes? | P1 — only investigate after overfit passes |
| 4 | After train-seed trend is positive, how much held-out scale is needed? | P1 — only meaningful after L3 is reached |

---

## 7. MINT Training Data Specification

### 7.1 Per-Frame Data

| Key | Type | Shape | Range | Notes |
|-----|------|-------|-------|-------|
| observation.images.image | uint8 | (H, W, 3) | [0, 255] | agentview camera |
| observation.images.image2 | uint8 | (H, W, 3) | [0, 255] | eye_in_hand camera |
| observation.state | float32 | (8,) | real | real robot contract: `eef_pos_xyz (3) + eef_quat_xyzw (4) + gripper_open (1)` |
| action | float32 | (7,) | [-1, 1] | real robot contract: `delta_xyz + delta_rxyz + gripper_command` |
| task | string | — | — | e.g. "open the drawer" |
| timestamp | float32 | (1,) | — | frame_index / fps |
| frame_index | int64 | (1,) | — | 0-indexed within episode |
| episode_index | int64 | (1,) | — | global episode ID |
| task_index | int64 | (1,) | — | index into tasks.parquet |

### 7.2 Normalization
- VISUAL: IDENTITY (no normalization)
- STATE: QUANTILES (q01/q99 → [-1, 1])
- ACTION: IDENTITY (must already be in [-1, 1])

### 7.3 Action Space (Active Root-Cause Repair Path)
- **Type**: robot end-effector delta control for the simulated manipulator
- **Semantics**:
  - `delta_x, delta_y, delta_z`
  - `delta_rx, delta_ry, delta_rz`
  - `gripper_command`
- **Replay rule**: only natural successful robot rollouts may enter the training dataset
- **Important note**: any rollout requiring `force_attach` is considered invalid for training

---

## 8. One-Shot Transfer Contract (Section addressing P0 review point)

### 8.1 What "one-shot" means in this project

**Current phase (simulation only)**: We do NOT claim true demo-conditioned one-shot transfer. We claim:

> "MINT fine-tuned on Infinigen-generated articulated manipulation data achieves higher success rate on held-out drawer variants than pretrained MINT."

The "one-shot" in MINT's original paper refers to:
- Given: a language instruction + one demonstration video
- Do: the task on a new object instance

**Our evaluation (simulation)**: 
- Pretrained MINT (zero-shot) vs. fine-tuned MINT (ours) on held-out objects
- "One-shot" = the task description is the intent signal (not a video demo)

### 8.2 MINT's actual input interface (from code analysis)

MINT does NOT take a demonstration video as input at inference time. It takes:
```
observation.images.image    # current RGB image
observation.images.image2   # current wrist camera
observation.state           # robot state (8D)
task                        # language description string
→ action                    # 7D delta action
```

The "intent tokenization" happens internally (state → text tokens → VQ codes). There is no external demo input in the current open-source implementation.

### 8.3 What we CAN test

| Evaluation | Description | Baseline |
|------------|-------------|----------|
| **Sim zero-shot** | Pretrained MINT (trained on LIBERO) on our drawer tasks | Random policy |
| **Sim fine-tuned** | MINT fine-tuned on Infinigen drawer data, evaluated on held-out drawer variants | Pretrained MINT |
| **Sim novel task** | Fine-tuned MINT on unseen task description (e.g., "close the drawer" after only training on "open the drawer") | Fine-tuned MINT on seen task |

### 8.4 What we CANNOT test (future work)

- Real-robot deployment
- Video-conditioned demonstration following
- Cross-domain transfer (drawer → microwave)

---

## 9. Validation Gates

| Gate | Input | Output | Pass Criteria | Status |
|------|-------|--------|---------------|--------|
| G1: Asset Load | Infinigen URDF + OBJ | robot + drawer scene | env loads, joint moves, URDF audit passes | ⏳ ACTIVE ROOT-CAUSE |
| G2: Obs Format | robot scene obs | MINT-compatible obs dict | image/image2/depth/state/task shapes valid, scene-frame audit passes | ⏳ ACTIVE ROOT-CAUSE |
| G3a: AnyGrasp Ready | SDK files + graspnet env | loadable detector | `.so`, license, checkpoint, `load_net()` all pass | ⏳ ACTIVE ROOT-CAUSE |
| G3b: Grasp Plan | depth from robot scene | SE(3) grasp poses on drawer lip | AnyGrasp score >= 0.5 with handle-region hit | ⏳ ACTIVE ROOT-CAUSE |
| G4: Robot Trajectory | oracle/AnyGrasp grasp + scripted open | robot-arm rollout | oracle scripted open must be stable; learning rollouts copied only after audit | ⏳ ACTIVE ROOT-CAUSE |
| G5: Delta Actions | natural robot rollout | MINT action contract | replay without `force_attach`, bounded error, bounded action stats | ⏳ ACTIVE ROOT-CAUSE |
| G6: LeRobot Pack | natural robot rollouts | LeRobot v3 dataset | finalized metadata, dataset integrity passes | ⏳ ACTIVE ROOT-CAUSE |
| G7: MINT Load | LeRobot dataset | training batch | `next(dataloader)` correct shapes, no NaN | ⏳ ACTIVE ROOT-CAUSE |
| G8a: Single-rollout overfit | one natural rollout | fine-tuned model | must beat pretrained on the source seed | ⏳ ACTIVE ROOT-CAUSE |
| G8b: Single-seed overfit | one train seed | fine-tuned model | must beat pretrained on the source seed | ⏳ ACTIVE ROOT-CAUSE |
| G8c: Train-seed probe | full train split | reproducibility summary | train-seed trend must turn positive before held-out eval | ⏳ ACTIVE ROOT-CAUSE |
| G9: Sim Eval | fine-tuned model | success rate on held-out objects | only runs after G8c passes | ⏳ DEFERRED UNTIL TRAIN TREND |

---

## 10. Evaluation Protocol (Section addressing P1 review point)

### 10.1 Object Splits

| Split | Objects | Purpose |
|-------|---------|---------|
| **Train** | 10 drawer variants (seed 1-10) | Generate trajectories, fine-tune MINT |
| **Held-out sim** | 5 drawer variants (seed 11-15) | Evaluate generalization in sim |
| **Held-out task** | same objects, different task (close drawer) | Evaluate task generalization |

### 10.2 Baselines

| Baseline | Description |
|----------|-------------|
| **Random policy** | Random action selection |
| **Pretrained MINT** | Zero-shot MINT (trained on LIBERO only) |
| **No conditioning** | Not executed in the current campaign; keep as future ablation |

### 10.3 Metrics

| Metric | Definition |
|--------|------------|
| **Success rate** | Binary: did the task complete? (drawer opened > threshold) |
| **Pull distance** | How far the drawer was opened (meters) |
| **Grasp success** | Did the robot successfully grasp the handle? |
| **Time to completion** | Steps to achieve success |

### 10.4 "One-shot" definition

**In this project**: one-shot = the language task description ("open the drawer") is the only conditioning signal at inference time. No demonstration video is provided. The "one" in "one-shot" refers to the single language instruction, not a visual demonstration.

---

## 11. Execution Plan

### Phase 1: Asset Pipeline

| Task | Description | Status |
|------|-------------|--------|
| 1a | URDF→proxy-sim load path | ✅ Done |
| 1b | DrawerProxyEnv + EGL | ✅ Done |
| 1c | Obs/action format alignment (G2) | ✅ Done |
| 1d | Batch generate drawer variants 1-15 | ✅ Done |

### Phase 2: Grasp Planning

| Task | Description | Status |
|------|-------------|--------|
| 2a | Stage AnyGrasp SDK in `graspnet` env | ✅ Done |
| 2a' | Verify `load_net()` + smoke inference | ✅ Done |
| 2b | Render depth from robot drawer scene | ⏳ Re-run under root-cause ladder |
| 2c | Run AnyGrasp on drawer lip / pull region (G3) | ⏳ Active |
| 2d | Compare AnyGrasp vs oracle handle grasp | ⏳ Active |

### Phase 3: Trajectory Generation

| Task | Description | Status |
|------|-------------|--------|
| 3a | Scripted robot-arm rollout with natural attach | ⏳ Active |
| 3b | Oracle grasp + scripted open baseline | ⏳ Active |
| 3c | AnyGrasp grasp + scripted open baseline | ⏳ Active |
| 3d | Natural delta-action replay contract (G5) | ⏳ Active |
| 3e | Batch clean natural trajectories | ⏳ Active |

### Phase 4: Dataset & Training

| Task | Description | Status |
|------|-------------|--------|
| 4a | LeRobot dataset builder from natural robot rollouts (G6) | ⏳ Active |
| 4b | Single-rollout / single-seed overfit checks | ⏳ Active |
| 4c | 1000-step MINT fine-tune (G8) | ⏳ Deferred until overfit checks pass |
| 4d | Train-seed reproducibility probe | ⏳ Required before held-out |
| 4e | Sim eval: held-out objects (G9) | ⏳ Deferred until train trend passes |

---

## 12. Pre-Download Status

| File | Source | Location | Size | Status |
|------|--------|----------|------|--------|
| MINT-libero | HuggingFace | external/MINT/checkpoints/ | ~7GB | ✅ |
| MINT-tokenizer-libero | HuggingFace | external/MINT/checkpoints/ | ~1GB | ✅ |
| paligemma tokenizer | HuggingFace (gated) | external/MINT/checkpoints/ | ~7MB | ✅ |
| libero-assets | HuggingFace | mint env site-packages | ~766MB | ✅ |
| AnyGrasp SDK assets | official request bundle | external/anygrasp_sdk/ | staged | ✅ |
| Contact-GraspNet PyTorch | git clone | external/contact_graspnet_pytorch/ | ~30MB | archived auxiliary |
| TF checkpoints (4 zips) | Google Drive | external/ (not needed) | ~250MB | ⚠️ Can delete |

---

## 13. Resolved Issues

| # | Issue | Resolution |
|---|-------|-----------|
| 1 | A800 cannot access OpenAI API | SSH reverse tunnel + local proxy |
| 2 | A800 cannot access HuggingFace | Local download → scp → extract |
| 3 | PaliGemma gated model | HF token + local tokenizer files |
| 4 | MINT weights not loading | Manual `load_state_dict` from safetensors |
| 5 | robosuite needs MJCF not URDF | MuJoCo native URDF→MJCF conversion |
| 6 | Contact-GraspNet: import error (cv2) | Installed opencv-python-headless |
| 7 | Contact-GraspNet: wrong class name | `ContactGraspnet` (lowercase n) |
| 8 | Contact-GraspNet: wrong constructor | `(global_config, device)` not `(config, grasp_dim)` |
| 9 | Contact-GraspNet: weights in nested dict | `ckpt["model"]` not `ckpt` |
| 10 | Contact-GraspNet: forward() signature | Just `model(point_cloud)` |
| 11 | TF checkpoints not needed | PyTorch ver has self-contained model.pt |

---

## 14. Safety Rules

1. **conda: infinigen is ISOLATED** — never install MINT/robosuite deps into it
2. **external/physnap/ is READ-ONLY**
3. **experiments/physnap/box_conditioning_v2/ is IMMUTABLE**
4. **Infinigen export code stays on feature/3d-assets**
5. **All MINT integration code on feature/mint-integration**
6. **Do not delete TF checkpoint zips without user confirmation**

---

## Appendix A: Overnight PhysNAP Results (reference)

| Group | MMD ↓ | COV ↑ | E_pen ↓ | E_mob ↓ | Cond Err ↓ |
|-------|-------|-------|---------|---------|-----------|
| zero_singleview | 0.559 | 0.091 | 0.00071 | 0.00058 | 0.00221 |
| best_single_view | 0.547 | 0.090 | 0.00103 | 0.00049 | 0.00141 |
| fixed_state_mv | 0.548 | 0.091 | 0.00102 | 0.00059 | 0.00169 |
| pts_500 | 0.535 | 0.089 | 0.00132 | 0.00076 | 0.00201 |
| pts_1000 | 0.537 | 0.090 | 0.00913 | 0.01076 | 0.00144 |
| pts_2000 | — | — | — | — | — (75% done, paused) |
| pts_5000 | — | — | — | — | — (not started) |

Resume: `_overnight_sandbox/CHECKPOINT_20260321.md`

---

## Appendix B: File Locations

```
/mnt/afs2/zhuhaowu/infinigen/  (branch: feature/3d-assets for export, feature/mint-integration for integration)
├── docs/MINT_INTEGRATION_PRD.md
├── external/
│   ├── MINT/
│   │   ├── checkpoints/{MINT-libero, MINT-tokenizer-libero, paligemma-3b-pt-224}
│   │   └── lerobot_policy_mint/
│   └── contact_graspnet_pytorch/
│       └── checkpoints/contact_graspnet/checkpoints/model.pt
└── sim_exports/urdf/{drawerbox, mailerbox_simple, sliplidbox, tuckendbox}/
```

## Appendix C: Conda Env Activation

```bash
ssh -p 30017 root@10.210.0.88

# Infinigen (do NOT install other deps here)
source /root/anaconda3/etc/profile.d/conda.sh && conda activate infinigen

# MINT + robosuite + LeRobot
source /root/anaconda3/etc/profile.d/conda.sh && conda activate mint && export MUJOCO_GL=egl

# Contact-GraspNet
source /root/anaconda3/etc/profile.d/conda.sh && conda activate graspnet
```
