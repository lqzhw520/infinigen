# Infinigen-AnyBox × CAPNet: Academic Alignment Analysis

**Date**: 2026-03-04  
**Context**: CAPNet (CVPR'25 Highlight, arXiv:2504.11230) + Research Group Discussion ([2025.12.09] Updates.pdf)

---

## 1. CAPNet Overview and Key Findings

CAPNet is a unified, single-stage RGB-D network for **6D part pose** (rotation R, translation t) and **3D part size** (scale s) estimation of articulated objects. It uses **NPCS** (Normalized Part Coordinate Space) as its core representation.

### Group Finding (from [2025.12.09] discussion)
> "pose似乎没啥问题，但是size一般"  
> (Pose seems fine, but size estimation is mediocre)

This finding is central to positioning Infinigen-AnyBox's contribution.

### CAPNet Evaluation Protocol
| Metric | Definition |
|--------|-----------|
| Re (°) | Geodesic rotation error between predicted and GT rotation matrices |
| Te (cm) | L2 translation error |
| Se | Mean absolute relative scale error |
| mIoU | 3D axis-aligned bounding box IoU |
| A5 | Accuracy at 5°/5cm threshold |
| A10 | Accuracy at 10°/10cm threshold |

---

## 2. Gap Analysis: Why Size Estimation Is Mediocre

### Hypothesis 1: Training Data Scale/Diversity Bottleneck
CAPNet trains on PartNet-Mobility (~2000 models). For packaging boxes specifically:
- Very few box-type models exist in PartNet-Mobility
- Limited size variation: existing datasets have narrow dimensional distributions
- No parametric control over dimensions (width, depth, height, thickness)

**Infinigen-AnyBox directly addresses this**: procedural generation produces unlimited parametric variations. Our 1K dataset already covers 25 seeds × 10 views × 4 box types = 1000 samples with diverse dimensions.

### Hypothesis 2: Physical Scale Ambiguity
CAPNet's NPCS normalizes parts to [-0.5, 0.5]^3, which inherently removes absolute scale. Size must then be regressed separately. Without physically-grounded training data with accurate metric dimensions, the size regressor has insufficient supervision signal.

**Infinigen-AnyBox addresses this**: every generated box has exact metric dimensions from the parametric generator. The URDF contains precise collision geometries, and our CAPNet bridge exports exact per-part AABB sizes in meters.

### Hypothesis 3: Topology Sensitivity
Different box topologies (MAILER 2-DOF, TUCK_END 8-DOF, DRAWER 1-DOF, SLIP_LID 1-DOF) have fundamentally different part configurations. A model trained primarily on doors/drawers from PartNet-Mobility may struggle with the articulation patterns of packaging boxes.

**Infinigen-AnyBox addresses this**: explicit topology diversity across 4+ box types, with planned expansion to 16+ types (RSCBox, LockBottomBox, etc.).

---

## 3. Innovation Space for Infinigen-AnyBox

### 3.1 Data-Level Contribution (Direct CAPNet Enhancement)

**Claim**: Infinigen-AnyBox is the first large-scale, physics-aligned, procedurally-generated articulated packaging box dataset with NPCS-compatible annotations.

| Comparison | PartNet-Mobility | Infinigen-AnyBox |
|-----------|-----------------|-----------------|
| Scale | ~2000 models (few boxes) | Unlimited (currently 1K, easily 10K+) |
| Box topologies | ~0 dedicated | 4 types, expanding to 16+ |
| Physical properties | No inertia/friction | Full URDF (mass, inertia, friction, collision) |
| Size accuracy | Manual annotation | Exact from parametric generator |
| Material diversity | Limited textures | 10+ domain-randomized PBR materials |
| Articulation states | Usually 1 state | Multiple randomized joint states per seed |
| NPCS annotations | Available | Now available (via capnet_data_bridge.py) |

### 3.2 Physics-Aligned Advantage (Beyond CAPNet)

CAPNet is perception-only. Infinigen-AnyBox's URDF-based pipeline enables:
- **Perception-to-Manipulation closed loop**: predicted poses + URDF → PyBullet/Isaac Sim → motion planning
- **Self-collision guarantees**: generated URDFs have collision meshes verified in PyBullet
- **Dynamic simulation**: inertia tensors satisfy positive-definiteness and triangle inequality

This positions the contribution beyond "just another dataset" toward a **sim-ready data engine** for the full perception-planning pipeline.

### 3.3 Topology-Aware Prediction (Novel Research Direction)

The planned Topo-Box-Net (Phase 2) predicts:
1. Box type (topology classification)
2. Joint state (articulation regression)
3. Per-part 6D pose and 3D size

This extends CAPNet's per-part prediction with **topology inference** — predicting which links/joints exist, not just their poses. No existing work addresses this for packaging boxes.

---

## 4. Data Bridge: Current Implementation Status

### 4.1 CAPNet-Format Annotations (Implemented)

`capnet_data_bridge.py` converts Phase-1 outputs to CAPNet-compatible format:

| Output | Description | Status |
|--------|------------|--------|
| Per-link 6D pose (R, t) | From FK on URDF + joint_state | Verified (self-consistency: Re=0, Te=0) |
| Per-link 3D size | From mesh AABB in world frame | Verified |
| Per-link AABB | World-frame bounding box | Verified |
| NPCS parameters | Per-link normalization (scale, offset) | Implemented |
| link_pos_quat_aabb.json | All link annotations | Generated for 1000 samples |
| obj_pos_quat_aabb.json | Object-level annotations | Generated for 1000 samples |

### 4.2 CAPNet Evaluation Metrics (Implemented)

`eval_capnet_metrics.py` implements the full CAPNet evaluation protocol:
- Rotation error (Re) via geodesic distance
- Translation error (Te) in centimeters
- Scale error (Se) as mean absolute relative error
- 3D IoU (mIoU) for axis-aligned bounding boxes
- Accuracy thresholds A5 (5°/5cm) and A10 (10°/10cm)
- Umeyama alignment for predicted-vs-GT point set matching

### 4.3 Dataset Statistics

| Dataset | Samples | Part Annotations | Seeds | Materials |
|---------|---------|-----------------|-------|-----------|
| phase1_1k_mailer | 250 | 750 (3 parts each) | 25 | 9 types |
| phase1_1k_drawer | 250 | 500 (2 parts each) | 25 | 8 types |
| phase1_1k_sliplid | 250 | 500 (2 parts each) | 25 | 7 types |
| phase1_1k_tuckend | 250 | 2250 (9 parts each) | 25 | 9 types |
| **Total** | **1000** | **4000** | 25 | 10+ |

---

## 5. Addressing "Size Is Mediocre"

### 5.1 Direct Data Improvement Path

The size estimation weakness in the group's CAPNet evaluation likely stems from:
1. **Limited box-specific training data**: few packaging box models in existing datasets
2. **Narrow size distribution**: insufficient variation in training dimensions
3. **Domain gap**: real boxes vs synthetic training data have different texture/geometry distributions

Infinigen-AnyBox can improve size estimation through:
- **Scale diversity**: parametric generation covers a wide range of box dimensions (e.g., width ∈ [0.05, 0.5]m)
- **Exact GT sizes**: per-part AABB computed from precise mesh geometry, not noisy annotations
- **Sim2Real fine-tuning**: domain-randomized materials + depth noise reduce the sim-to-real gap

### 5.2 Experimental Plan

To validate whether Infinigen data improves CAPNet's size estimation:

1. **Baseline**: Train CAPNet on PartNet-Mobility, evaluate on real box images
2. **+Infinigen**: Add Infinigen-AnyBox to training, re-evaluate
3. **Ablation on scale**:
   - Train on Infinigen with restricted size range vs full range
   - Measure Se improvement as a function of training size diversity

Expected result: Se (scale error) should decrease significantly when Infinigen-AnyBox data is added, because the model sees many more box instances with varied sizes.

---

## 6. Academic Positioning

### 6.1 Contribution Framing (for CVPR/ICRA/RSS)

**Title direction**: *Physics-Aligned Procedural Data Engine for Generalizable Articulated Box Manipulation*

**Key contributions**:
1. **Infinigen-AnyBox**: first large-scale, physics-aligned, procedurally-generated dataset for articulated packaging boxes with 4+ topologies and NPCS-compatible annotations
2. **Data-driven improvement**: demonstrate that procedural synthetic data with parametric size control significantly improves 3D part size estimation (addressing the "size一般" finding)
3. **Sim-ready pipeline**: generated URDFs directly usable in physics simulators, enabling a perception-to-manipulation closed loop

### 6.2 Differentiation from Related Work

| Work | Scope | Limitation Infinigen-AnyBox Addresses |
|------|-------|--------------------------------------|
| CAPNet (CVPR'25) | Perception (pose+size from RGBD) | Limited to existing datasets; no physics simulation |
| PartNet-Mobility | Dataset (articulated objects) | ~2000 models, few boxes, no parametric control |
| PhysNAP | Physics-based articulation | Focuses on prediction, not data generation |
| Infinigen (original) | Procedural natural scenes | No articulated packaging boxes |
| **Infinigen-AnyBox (ours)** | **Data engine + perception** | **Unlimited scale, physics-aligned, topology-diverse** |

### 6.3 Haowu's Pipeline Position

From the [2025.12.09] discussion, the full pipeline:
```
Data Generation (Infinigen-AnyBox) → Perception (CAPNet/Topo-Box-Net) → Manipulation (Planner)
```

Infinigen-AnyBox is the **foundational data layer** that feeds both perception training and simulation-based manipulation. The academic value compounds across the full pipeline.

---

## 7. Next Steps

### Immediate (Data Engine)
- [ ] Scale to 10K+ samples per box type for robust training
- [ ] Add depth noise simulation (Gaussian + dropout) for Sim2Real
- [ ] Implement full NPCS map rendering (per-pixel normalized coordinates)

### Medium-term (Perception)
- [ ] Train CAPNet on Infinigen-AnyBox data
- [ ] Compare Se metrics: PartNet-Mobility-only vs +Infinigen
- [ ] Implement Topo-Box-Net architecture (Phase 2.1)

### Long-term (Full Pipeline)
- [ ] Sim2Real validation with real packaging boxes
- [ ] Motion planner integration (Phase 3.2)
- [ ] Paper submission targeting CVPR 2027 / ICRA 2027
