# PRD: Infinigen Sim Data → MINT Training Pipeline

**Project**: feature/mint-integration
**Branch**: `feature/mint-integration` (from `feature/3d-assets`)
**Author**: zhuhaowu
**Last Updated**: 2026-03-22
**Status**: Planning Complete, Ready for Execution

---

## 1. Objective

Use Infinigen-generated articulated objects (drawers, cabinets) as simulation inputs to generate robot manipulation trajectories, then use these trajectories to fine-tune MINT for improved one-shot task transfer on novel articulated objects.

**Deliverable**: A working pipeline that:
1. Generates diverse articulated objects via Infinigen
2. Places them in a robosuite simulation with a Franka robot
3. Automatically generates expert manipulation trajectories (grasp + operate)
4. Packages trajectories as a LeRobot dataset
5. Fine-tunes MINT on this dataset
6. Demonstrates one-shot transfer to unseen objects/tasks

---

## 2. System Architecture

```
Phase 1: Asset Gen          Phase 2: Sim Env           Phase 3: Trajectory        Phase 4: Training
┌──────────────┐         ┌──────────────────┐       ┌──────────────────┐      ┌───────────────┐
│  Infinigen    │  URDF   │  robosuite       │  obs  │ Contact-GraspNet │      │  MINT         │
│  DrawerBox    │──────→  │  + Franka Panda  │──────→│ + Motion Planner │─────→│  Fine-tune    │
│  CabinetBox   │  MJCF   │  + dual camera   │  act  │ expert trajectory│      │  (LeRobot)    │
│  MailerBox    │  (auto)  │  + EGL headless  │       │ delta actions    │      │               │
└──────────────┘         └──────────────────┘       └──────────────────┘      └───────────────┘
     已有 ✓              已验证: robosuite         抓取: Contact-GraspNet      已验证: MINT
                          1.4.0 + MuJoCo 3.6.0     (PyTorch, headless)       eval 通过 ✓
                          EGL headless ✓             规划: IK + task-space
```

---

## 3. Environment

| Component | Version | Location | Status |
|-----------|---------|----------|--------|
| A800 GPU | NVIDIA A800-SXM4-80GB | ssh -p 30017 root@10.210.0.88 | ✓ |
| Infinigen | 1.19.0 | /mnt/afs2/zhuhaowu/infinigen/ | ✓ conda: infinigen |
| MINT | 0.1.0 | external/MINT/ | ✓ conda: mint |
| lerobot | 0.4.3 | pip (mint env) | ✓ |
| robosuite | 1.4.0 | pip (mint env) | ✓ |
| MuJoCo | 3.6.0 | pip (mint env) | ✓ |
| PyTorch | 2.3+ | pip (mint env) | ✓ CUDA |
| Contact-GraspNet | PyTorch ver. | TBD (external/contact_graspnet/) | ⬜ conda: graspnet |
| Rendering | EGL headless | MUJOCO_GL=egl | ✓ verified |

### Network Constraints
- A800 cannot access: huggingface.co, api.openai.com, Google Drive
- Workaround: local Mac download → scp to A800
- SSH tunnel proxy available for OpenAI API (codex MCP)

---

## 4. Verified Foundations

### 4.1 MINT Eval: VERIFIED ✓
- 25/26 checks passed (verify_report.json)
- Model: PaliGemma 2B + Gemma expert + Multi-scale VQVAE
- Weights: 7.05GB safetensors, 1034 keys loaded correctly
- Inference: 3.7 steps/s, GPU 7.1GB
- Action output: (7,) float32, range [-1, 1]

### 4.2 Infinigen URDF Export: VERIFIED ✓
- drawerbox: 1 slide joint, range [0, 0.254m]
- mailerbox: 2 hinge joints, full rotation
- 1000+ variants in phase1_1k_drawer dataset
- MuJoCo loads URDF correctly via VFS

### 4.3 URDF → robosuite: VERIFIED ✓
- URDF → MJCF conversion via MuJoCo native support
- robosuite MujocoXMLObject loading successful
- Joint behavior preserved (slide axis, range, damping)
- Required modifications identified and tested:
  - Nested body structure (`<body><body name="object">`)
  - Free joint for object placement
  - Collision/visual geom duplication
  - Default site addition

### 4.4 Headless Rendering: VERIFIED ✓
- MUJOCO_GL=egl on A800 with NVIDIA driver
- robosuite offscreen rendering: 360×360 RGB ✓
- No display/GUI needed

---

## 5. MINT Training Data Specification

### 5.1 Per-Frame Data (from code-level verification)

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

### 5.2 Normalization
- VISUAL: IDENTITY (no normalization)
- STATE: QUANTILES (q01/q99 → [-1, 1])
- ACTION: IDENTITY (must already be in [-1, 1])

### 5.3 Action Space (CRITICAL)
- **Type**: relative delta in EEF space (OSC_POSE controller)
- **action[:3]**: delta position relative to current EEF
- **action[3:6]**: delta rotation (axis-angle) relative to current EEF
- **action[6]**: gripper command (-1=close, 1=open)
- **action_scale**: must match robosuite OSC controller config

### 5.4 LeRobot Dataset Structure

```
infinigen_articulated_dataset/
├── meta/
│   ├── info.json           # fps=30, features, total_episodes, etc.
│   ├── stats.json          # q01/q99 for state normalization
│   ├── tasks.parquet       # task descriptions
│   └── episodes/chunk-000/ # per-episode metadata
├── data/chunk-000/         # parquet: state + action + indices
└── videos/
    ├── observation.images.image/chunk-000/   # agentview MP4
    └── observation.images.image2/chunk-000/  # eye_in_hand MP4
```

### 5.5 MINT Processing Pipeline (code-verified)
```
state (8D) → pad to 32D → quantize to 256 bins → text
     ↓
"Task: open the drawer, State: 128,130,...;\nAction: "
     ↓
PaliGemma tokenizer → token IDs (max_length=200)
     ↓
images → SiglipVisionModel → visual tokens
     ↓
[visual tokens + text tokens] → PaliGemma LM → VQ codes
     ↓
VQ codes → Multi-scale VQVAE decoder → action (7D × 16 steps)
```

---

## 6. Grasp Planning

### 6.1 Primary: Contact-GraspNet (PyTorch)
- Repo: https://github.com/elchun/contact_graspnet_pytorch
- Python: 3.9 (separate conda env: graspnet)
- Input: depth image (HxW, meters) + camera intrinsics (3x3)
- Output: N × (4x4 SE(3) pose, confidence, gripper_width)
- Headless: ✓ (PYOPENGL_PLATFORM=egl)
- **Model files needed** (download from Google Drive, scp to A800):
  - `checkpoints/scene_test_2048_bs3_hor_sigma_001/` — default checkpoint

### 6.2 Future: AnyGrasp (after license approval)
- Applied: 2026-03-22, awaiting approval
- Machine-locked license

### 6.3 Interface Abstraction
```python
class GraspPlanner(ABC):
    @abstractmethod
    def predict(self, depth: np.ndarray, K: np.ndarray, 
                rgb: np.ndarray = None) -> List[GraspPose]:
        """
        Args:
            depth: (H, W) float32, meters
            K: (3, 3) camera intrinsics
            rgb: optional (H, W, 3) uint8
        Returns:
            List of GraspPose(pose_4x4, score, width)
        """

class ContactGraspNetPlanner(GraspPlanner): ...
class AnyGraspPlanner(GraspPlanner): ...  # drop-in replacement
```

---

## 7. Critical Data Flow & Validation Gates

### 7.1 End-to-End Data Flow

```
Gate 1          Gate 2          Gate 3          Gate 4          Gate 5
Infinigen   →   robosuite   →   Trajectory  →   LeRobot     →   MINT
URDF            env.reset()     generation      dataset         training
                obs dict        (obs, act)      load OK         loss ↓
```

### 7.2 Validation Gates (each must pass before proceeding)

| Gate | Input | Output | Pass Criteria |
|------|-------|--------|---------------|
| G1: Asset Load | Infinigen URDF + OBJ | robosuite env with articulated object | `env.reset()` returns obs with correct shapes; joint moves correctly |
| G2: Obs Format | robosuite obs | MINT-compatible obs dict | `obs["agentview_image"].shape == (360,360,3)`, state shape (8,) |
| G3: Grasp Plan | depth + intrinsics | SE(3) grasp poses | ≥1 grasp with score > 0.5 on drawer handle region |
| G4: Trajectory | grasp pose + task goal | (obs_seq, action_seq) | Replay trajectory achieves task success in sim |
| G5: Delta Actions | absolute trajectory | delta actions in [-1, 1] | Reconstruct trajectory from deltas matches original (error < 1mm) |
| G6: LeRobot Pack | (obs, act) sequences | LeRobot v3 dataset | `LeRobotDataset(repo_id, root=path)` loads without error |
| G7: MINT Load | LeRobot dataset | training batch | `next(dataloader)` returns correct shapes, no NaN |
| G8: MINT Train | dataset + pretrained | fine-tuned model | Training loss decreases over 1000 steps |

### 7.3 Highest Risk: Gate 5 (Delta Action Conversion)

**Why**: robosuite LIBERO uses OSC_POSE controller with specific action scaling. Motion planner outputs absolute poses. Conversion must be exact.

**Conversion formula**:
```python
# At each timestep t:
delta_pos = (target_eef_pos[t+1] - current_eef_pos[t]) / pos_action_scale
delta_rot = quat_to_axis_angle(
    quat_multiply(target_eef_quat[t+1], quat_inverse(current_eef_quat[t]))
) / rot_action_scale
gripper_cmd = 1.0 if open else -1.0
action = np.concatenate([delta_pos, delta_rot, [gripper_cmd]])
action = np.clip(action, -1.0, 1.0)
```

**Verification**: replay the delta actions in robosuite, compare resulting EEF trajectory with planned trajectory. Max error must be < 1mm position, < 1° rotation.

**OSC controller config** (from robosuite defaults):
```python
controller_config = {
    "type": "OSC_POSE",
    "input_max": 1,
    "input_min": -1,
    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],  # m, rad
    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
    "kp": 150,
    "damping_ratio": 1,
    "impedance_mode": "fixed",
    "kp_limits": [0, 300],
    "damping_ratio_limits": [0, 10],
    "position_limits": None,
    "orientation_limits": None,
    "uncouple_pos_ori": True,
    "control_delta": True,
    "interpolation": None,
    "ramp_ratio": 0.2,
}
```

This means: action=1.0 → 0.05m position delta or 0.5 rad rotation delta per step.

---

## 8. Execution Plan

### Phase 1: Asset Pipeline (Day 1)

| Task | Description | Validation | Owner |
|------|-------------|------------|-------|
| 1a | URDF→robosuite MJCF auto-converter | drawerbox loads in robosuite env | Agent |
| 1b | InfinigenDrawerEnv (robosuite) | env.reset() → correct obs shapes | Agent |
| 1c | Obs/action format alignment | Match LIBERO exactly | Agent |
| 1d | Batch generate 10 drawer variants | 10 URDF → 10 env | Agent |

**Gate G1 + G2 must pass.**

### Phase 2: Grasp Planning (Day 2)

| Task | Description | Validation | Owner |
|------|-------------|------------|-------|
| 2a | Install Contact-GraspNet (conda: graspnet) | import succeeds | Agent |
| 2b | Render depth from robosuite env | depth image correct | Agent |
| 2c | Run grasp prediction on drawer | ≥1 valid grasp on handle | Agent |
| 2d | Abstract GraspPlanner interface | AnyGrasp can drop-in later | Agent |

**Gate G3 must pass.**
**Pre-download needed**: Contact-GraspNet model checkpoint from Google Drive → scp to A800.

### Phase 3: Trajectory Generation (Day 3-4)

| Task | Description | Validation | Owner |
|------|-------------|------------|-------|
| 3a | IK solver for Franka in robosuite | Reach grasp pose | Agent |
| 3b | Implement open-drawer trajectory | approach→grasp→pull→release | Agent |
| 3c | Delta action conversion | Reconstruct error < 1mm | Agent |
| 3d | Trajectory replay verification | Replay achieves task success | Agent |
| 3e | Batch generate 20-50 trajectories | All pass replay check | Agent |

**Gate G4 + G5 must pass. This is the highest-risk phase.**

### Phase 4: Dataset & Training (Day 5-6)

| Task | Description | Validation | Owner |
|------|-------------|------------|-------|
| 4a | LeRobot dataset builder script | Dataset loads with correct schema | Agent |
| 4b | Compute normalization stats | stats.json valid | Agent |
| 4c | MINT fine-tune (10k steps) | Loss decreases | Agent |
| 4d | MINT eval on training tasks | Better than random | Agent |
| 4e | One-shot transfer test | Novel object + task | Agent |

**Gate G6 + G7 + G8 must pass.**

---

## 9. Files to Pre-Download (Local Mac → scp to A800)

| File | Source | Dest on A800 | Size | Status |
|------|--------|-------------|------|--------|
| MINT-libero | HuggingFace | external/MINT/checkpoints/ | ~7GB | ✅ Done |
| MINT-tokenizer-libero | HuggingFace | external/MINT/checkpoints/ | ~1GB | ✅ Done |
| paligemma-3b-pt-224 tokenizer | HuggingFace (gated) | external/MINT/checkpoints/ | ~7MB | ✅ Done |
| libero-assets | HuggingFace | mint env site-packages | ~700MB | ✅ Done |
| Contact-GraspNet checkpoint | Google Drive | external/contact_graspnet/checkpoints/ | ~200MB | ⬜ TODO |

---

## 10. Resolved Issues

| Issue | Resolution | Date |
|-------|-----------|------|
| A800 cannot access OpenAI API | SSH reverse tunnel + local HTTP CONNECT proxy (aris-proxy.py + autossh) | 2026-03-20 |
| A800 cannot access HuggingFace | Local Mac download → scp tarball → extract on A800 | 2026-03-22 |
| PaliGemma is gated model | HF token auth + local download of tokenizer files | 2026-03-22 |
| MINT policy weights not loading | cfg.pretrained_path pointed to HF repo; manual safetensors load via load_state_dict | 2026-03-22 |
| robosuite needs MJCF not URDF | MuJoCo native URDF→MJCF + add sites/naming/geom duplication | 2026-03-22 |
| screen not available on A800 | /run/screen/S-root missing; use nohup as fallback | 2026-03-21 |
| Overnight skill: codex review skipped | SKILL.md lacks mandatory review hook; to be fixed | 2026-03-21 |
| Overnight skill: STATE.json not updated | Agent didn't follow state persistence rule; to be fixed | 2026-03-21 |
| Contact-GraspNet needs Python 3.9 | Separate conda env (graspnet) for grasp planning | 2026-03-22 |
| AnyGrasp requires license | Applied; use Contact-GraspNet first; abstract GraspPlanner interface for swap | 2026-03-22 |

---

## 11. Open Questions for Advisor/Senior Review

1. **Task scope**: Start with "open drawer" only, or include "close drawer", "open cabinet" in first iteration?
2. **Trajectory quality**: Scripted trajectories vs learned (RL) demonstrations? Scripted is faster but less natural.
3. **Data quantity**: 20-50 demos per task sufficient for MINT fine-tune? LIBERO uses ~50 demos/task.
4. **Object diversity**: How many unique drawer/cabinet variants needed? Current plan: 10 variants × 5 demos each = 50 demos.
5. **One-shot evaluation**: What novel objects/tasks should we test transfer on?
6. **AnyGrasp priority**: When license arrives, should we switch immediately or after first pipeline validation?

---

## 12. Safety Rules

1. **infinigen env (conda: infinigen) is ISOLATED** — never install MINT/robosuite dependencies into it
2. **external/physnap/ is READ-ONLY** — no modifications
3. **experiments/physnap/box_conditioning_v2/ is IMMUTABLE** — completed baseline
4. **feature/overnight-review branch** has checkpoint for resuming physNAP ablations later
5. **All MINT work stays on feature/mint-integration branch**

---

## Appendix A: Overnight PhysNAP Ablation Results (for reference)

| Group | MMD ↓ | COV ↑ | E_pen ↓ | E_mob ↓ | Cond Err ↓ |
|-------|-------|-------|---------|---------|-----------|
| zero_singleview (baseline) | 0.559 | 0.091 | 0.00071 | 0.00058 | 0.00221 |
| best_single_view | 0.547 | 0.090 | 0.00103 | 0.00049 | 0.00141 |
| fixed_state_mv | 0.548 | 0.091 | 0.00102 | 0.00059 | 0.00169 |
| pts_500 | 0.535 | 0.089 | 0.00132 | 0.00076 | 0.00201 |
| pts_1000 | 0.537 | 0.090 | 0.00913 | 0.01076 | 0.00144 |
| pts_2000 | — | — | — | — | — |
| pts_5000 | — | — | — | — | — |

Remaining: pts_2000 (75% done, paused), pts_5000 (not started).
Resume instructions in `_overnight_sandbox/CHECKPOINT_20260321.md`.

---

## Appendix B: MINT Verified Components

| Check | Result | Detail |
|-------|--------|--------|
| paligemma_with_expert | PASS | PaliGemma backbone + Gemma expert |
| multi_scale_vqvae | PASS | SDAT multi-scale VQ-VAE tokenizer |
| codebook | PASS | 512 codes × 32 dim |
| patch_nums | PASS | [1, 2, 4] scale hierarchy |
| safetensors | PASS | 7.05GB, 1034 keys |
| weights_loaded | PASS | 0 unexpected, 1 missing (embed_tokens) |
| weights_not_random | PASS | std=0.0229 |
| tokenizer | PASS | vocab 257152 |
| pre_processor | PASS | 6 steps |
| post_processor | PASS | 2 steps |
| LIBERO-10 env | PASS | 10 tasks |
| inference | PASS | 3.7 steps/s, GPU 7.1GB |
| action_shape | PASS | (10, 7) |
| action_range | PASS | [-1.0, 0.42] |
