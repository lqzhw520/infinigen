# PRD: Infinigen Sim Data → MINT Training Pipeline

**Project**: feature/mint-integration
**Branch**: `feature/mint-integration` (from `feature/3d-assets`)
**Note**: Infinigen export code remains on `feature/3d-assets`. All MINT integration code on `feature/mint-integration`.
**Author**: zhuhaowu
**Last Updated**: 2026-03-22
**Status**: Phase 1a & 2a Complete — URDF Loads & GraspNet Inference Verified (NOT end-to-end grasp on drawer yet)

---

## 1. Objective

Use Infinigen-generated articulated objects (drawers, cabinets) as simulation inputs to generate robot manipulation trajectories, then use these trajectories to fine-tune MINT for improved task transfer on novel articulated objects.

**Phase scope clarification**: This PRD defines the **simulation-only** pipeline. Real-robot transfer is a future phase.

**Deliverable (current phase)**:
1. Infinigen articulated objects loadable in robosuite sim with correct physics
2. Contact-GraspNet generates grasp candidates from rendered depth
3. Scripted motion planner produces open-drawer trajectories
4. Trajectories are converted to delta actions and replay-verified in sim
5. Trajectories are packaged as a LeRobot v3 dataset
6. MINT is fine-tuned on this dataset
7. **Sim eval**: success rate on held-out objects in simulation (NOT real robot)

**What this does NOT deliver (future work)**:
- Real-robot deployment
- True demo-conditioned one-shot transfer (see Section 11 for the gap)

---

## 2. System Architecture

```
Phase 1: Asset Gen          Phase 2: Sim Env           Phase 3: Trajectory        Phase 4: Training
┌──────────────┐         ┌──────────────────┐       ┌──────────────────┐      ┌───────────────┐
│  Infinigen    │  URDF   │  robosuite       │  obs  │ Contact-GraspNet │      │  MINT         │
│  DrawerBox    │──────→  │  + Franka Panda  │──────→│ + Motion Planner │─────→│  Fine-tune    │
│  CabinetBox   │  MJCF   │  + dual camera   │  act  │ expert trajectory│      │  (LeRobot)    │
│  MailerBox    │  (auto) │  + EGL headless  │       │ delta actions    │      │               │
└──────────────┘         └──────────────────┘       └──────────────────┘      └───────────────┘
  ✅ Verified ✓            ✅ Partially ✓            ✅ Model loads ✓          ✅ Model loads ✓
  URDF loads in MuJoCo     robosuite 1.4.0           Contact-GraspNet          MINT eval 25/26
  joints correct           EGL headless              generic inference OK      (see Known Issues)
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
| Contact-GraspNet | PyTorch ver. | external/contact_graspnet_pytorch/ | graspnet | ✓ |
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
- **Contains**: Contact-GraspNet, opencv, open3d
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

### 5.4 Contact-GraspNet: MODEL LOAD VERIFIED ✓

| Check | Result |
|-------|--------|
| Model load | PASS — 2,194,097 params |
| Weight load | PASS — `ckpt["model"]` |
| Inference | PASS — 793ms for 2048 points |
| Output format | PASS — (B, 2048, 4, 4) SE(3) |
| **Grasp on drawer** | **NOT YET TESTED** — only generic inference |

### 5.5 Headless Rendering: VERIFIED ✓
- EGL rendering on A800, no display needed

---

## 6. Known Issues Before Execution

### Negative Signals (must be resolved or acknowledged before claiming pipeline works)

| Signal | Detail | Implication |
|--------|--------|-------------|
| **Gripper action constant** | In 10-step test, `action_grip_varies` = FAIL. Gripper stays at one value. | May indicate MINT doesn't output varied gripper commands in first 10 steps, OR the model needs more steps to change gripper state. Needs investigation. |
| **Closed-loop success = False** | In 100-step test, `success` = False. | Expected for random initial state without proper task setup, BUT must verify MINT can achieve success when given correct task instruction and initial state from LIBERO eval protocol. |
| **Contact-GraspNet on drawer** | Model loads and infers, but has NOT been tested on actual rendered depth from a drawer scene. | Grasp candidates on real drawer geometry may fail or be poor quality. |

### Open Questions (from GPT-5.4 review)

| # | Question | Impact |
|---|----------|--------|
| 1 | Does the current MINT open-source stack truly support fine-tuning on custom LeRobot datasets? (README claims it, but we haven't tried `lerobot-train` on our data) | P0 — if not, whole plan needs re-architecture |
| 2 | What is "one-shot" in our context? (see Section 11) | P1 — defines the final evaluation claim |
| 3 | How many demos per task? (LIBERO uses ~50) | P1 — affects data generation cost |
| 4 | What is the baseline for comparison? | P1 — needed for academic claim |

---

## 7. MINT Training Data Specification

### 7.1 Per-Frame Data

| Key | Type | Shape | Range | Notes |
|-----|------|-------|-------|-------|
| observation.images.image | uint8 | (H, W, 3) | [0, 255] | agentview camera |
| observation.images.image2 | uint8 | (H, W, 3) | [0, 255] | eye_in_hand camera |
| observation.state | float32 | (8,) | real | eef_pos(3) + eef_quat(4) + gripper(1) |
| action | float32 | (7,) | [-1, 1] | delta: dx,dy,dz,drx,dry,drz,gripper |
| task | string | — | — | e.g. "open the drawer" |
| timestamp | float32 | (1,) | — | frame_index / fps |
| frame_index | int64 | (1,) | — | 0-indexed within episode |
| episode_index | int64 | (1,) | — | global episode ID |
| task_index | int64 | (1,) | — | index into tasks.parquet |

### 7.2 Normalization
- VISUAL: IDENTITY (no normalization)
- STATE: QUANTILES (q01/q99 → [-1, 1])
- ACTION: IDENTITY (must already be in [-1, 1])

### 7.3 Action Space (CRITICAL — Gate G5)
- **Type**: relative delta in EEF space (OSC_POSE controller)
- **Max output per step**: 0.05m position, 0.5 rad rotation
- **action=1.0** → +0.05m (pos) or +0.5rad (rot) or gripper open
- **action=-1.0** → -0.05m (pos) or -0.5rad (rot) or gripper close

---

## 8. One-Shot Transfer Contract (Section addressing P0 review point)

### 8.1 What "one-shot" means in this project

**Current phase (simulation only)**: We do NOT claim true demo-conditioned one-shot transfer. We claim:

> "MINT fine-tuned on Infinigen-generated articulated manipulation data achieves higher success rate on held-out drawer/cabinet variants than pretrained MINT."

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
| G1: Asset Load | Infinigen URDF + OBJ | robosuite env with articulated object | env.reset() returns obs; joint moves | ✅ DONE |
| G2: Obs Format | robosuite obs | MINT-compatible obs dict | obs shapes match LIBERO | ⬜ NEXT |
| G3: Grasp Plan | depth from drawer scene | SE(3) grasp poses on handle | ≥1 grasp with score > 0.5 on handle region | ⬜ NEXT |
| G4: Trajectory | grasp pose + task goal | (obs_seq, action_seq) | Replay achieves task success in sim | ⬜ TODO |
| G5: Delta Actions | absolute trajectory | delta actions in [-1, 1] | Reconstruct from deltas matches original (< 1mm error) | ⬜ TODO |
| G6: LeRobot Pack | (obs, act) sequences | LeRobot v3 dataset | `LeRobotDataset(root=path)` loads OK | ⬜ TODO |
| G7: MINT Load | LeRobot dataset | training batch | `next(dataloader)` correct shapes, no NaN | ⬜ TODO |
| G8: MINT Train | dataset + pretrained | fine-tuned model | Training loss decreases over 1000 steps | ⬜ TODO |
| G9: Sim Eval | fine-tuned model | success rate on held-out objects | fine-tuned > pretrained MINT on drawer tasks | ⬜ TODO |

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
| **No conditioning** | MINT with empty task string |

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
| 1a | URDF→robosuite MJCF auto-converter | ✅ Verified |
| 1b | InfinigenDrawerEnv (robosuite) | ⬜ TODO |
| 1c | Obs/action format alignment (G2) | ⬜ TODO |
| 1d | Batch generate 10 drawer variants | ⬜ TODO |

### Phase 2: Grasp Planning

| Task | Description | Status |
|------|-------------|--------|
| 2a | Install Contact-GraspNet | ✅ Done |
| 2a' | Verify model loads + inference | ✅ Done (generic) |
| 2b | Render depth from drawer scene | ⬜ TODO |
| 2c | Run grasp prediction on drawer (G3) | ⬜ TODO |
| 2d | Abstract GraspPlanner interface | ⬜ TODO |

### Phase 3: Trajectory Generation

| Task | Description | Status |
|------|-------------|--------|
| 3a | IK solver for Franka | ⬜ TODO |
| 3b | Open-drawer trajectory script | ⬜ TODO |
| 3c | Delta action conversion (G5) | ⬜ TODO |
| 3d | Replay verification (G4) | ⬜ TODO |
| 3e | Batch 20-50 trajectories | ⬜ TODO |

### Phase 4: Dataset & Training

| Task | Description | Status |
|------|-------------|--------|
| 4a | LeRobot dataset builder (G6) | ⬜ TODO |
| 4b | Normalization stats | ⬜ TODO |
| 4c | MINT fine-tune (G8) | ⬜ TODO |
| 4d | Sim eval: train objects | ⬜ TODO |
| 4e | Sim eval: held-out objects (G9) | ⬜ TODO |

---

## 12. Pre-Download Status

| File | Source | Location | Size | Status |
|------|--------|----------|------|--------|
| MINT-libero | HuggingFace | external/MINT/checkpoints/ | ~7GB | ✅ |
| MINT-tokenizer-libero | HuggingFace | external/MINT/checkpoints/ | ~1GB | ✅ |
| paligemma tokenizer | HuggingFace (gated) | external/MINT/checkpoints/ | ~7MB | ✅ |
| libero-assets | HuggingFace | mint env site-packages | ~766MB | ✅ |
| Contact-GraspNet PyTorch | git clone | external/contact_graspnet_pytorch/ | ~30MB | ✅ |
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
