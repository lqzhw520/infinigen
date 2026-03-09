# PhysNAP Academic Review & Infinigen Integration Analysis

**Date**: 2026-03-09  
**Paper**: *Guiding Diffusion-Based Articulated Object Generation by Partial Point Cloud Alignment and Physical Plausibility Constraints*  
**Authors**: Jens U. Kreber, Joerg Stueckler (University of Augsburg)  
**Venue**: ICCV 2025  
**arXiv**: 2508.00558v1

---

## 1. Paper Summary

PhysNAP extends NAP (NeurIPS 2023) -- an unconditional diffusion model that generates articulated object graphs (parts + joints + SDFs) -- by adding **training-free** loss-guided sampling with three physics-based losses:

- **l_pc** (Point Cloud Alignment): EM-style soft correspondence between a partial point cloud and the predicted SDF surfaces. Uses inverse-temperature softmax (tau=1000) over per-part SDF distances, weighted by continuous node existence indicators.
- **l_pen** (Non-penetration): SDF collision detection in bounding box intersection regions. Samples ~1000 grid points in each pairwise BB intersection volume, accumulates squared negative-sum SDF errors.
- **l_mob** (Mobility): Same collision detection but at randomly sampled joint articulation states, using Plucker-coordinate-derived screw transforms. Only evaluated for edge-connected parts (existence > 0.1 threshold).

**Key architectural property**: PhysNAP does NOT retrain the underlying diffusion model for conditioning on observations. It injects gradient-based guidance during the reverse diffusion via the Tweedie-estimated clean state x_hat_0, following DPS/LGD theory:

∇_{x_t} log p_t(y|x_t) ≈ -∇_{x_t} l_y(x_hat_0)

The only retraining is for category-aware generation: a learnable category embedding added to the AGNN's node/edge embeddings.

### 1.1 NAP Parameterization (inherited by PhysNAP)

An articulated object is represented as a fully-connected graph with max K=8 nodes:

**Node attributes** (per part i):
- o_tilde_i ∈ R: continuous existence indicator (thresholded to binary at inference)
- T_gi ∈ R^3: 3D translation only (zero-articulation canonical orientation assumed)
- b_i ∈ R^3: bounding box extents
- s_i ∈ R^128: shape latent code (decoded by pretrained DeepSDF-like autoencoder)

**Edge attributes** (per pair i,j):
- c_tilde_{i,j} ∈ R^3: existence + direction (one-hot encoded for {-1,0,1})
- p_{i,j} = (l_{i,j}, m_{i,j}): 6D Plucker coordinates for joint axis
- rho_{i,j} ∈ R^{2x2}: joint state ranges (screw angle + extension limits)

All attributes are concatenated into a flat real vector for DDPM noise prediction via an Attentional Graph Neural Network (AGNN).

### 1.2 Training Data

NAP/PhysNAP trains on **PartNet-Mobility** (~2000 objects, 17 categories). The pretrained shape autoencoder (stage 1.5) encodes per-part meshes into 128-dim latent codes. Point clouds for evaluation are rendered via GAPartNet pipeline (1000 points per object, sampled within segmentation mask).

### 1.3 Key Quantitative Results (Table 1)

| Model | Guidance | E_pc | D_pc | E_pen | E_mob | MMD | 1-NNA |
|-------|----------|------|------|-------|-------|-----|-------|
| NAP (no cat) | uncond | 0.0564 | 0.2063 | 0.0035 | 0.0033 | **0.0915** | **0.9440** |
| PhysNAP (no cat) | pc+pen+mob | 0.0024 | 0.0705 | **0.0000** | **0.0003** | 0.1435 | 0.9547 |
| NAP (cat) | uncond | 0.0076 | 0.0974 | 0.0019 | 0.0027 | 0.1687 | 0.9709 |
| PhysNAP (cat) | pc+pen+mob | **0.0012** | **0.0483** | **0.0000** | **0.0003** | 0.1970 | 0.9774 |

**Generative dilemma**: Stronger guidance reduces alignment/physics errors but degrades diversity (MMD, 1-NNA increase). This is a mathematical consequence of posterior sampling narrowing the latent manifold.

---

## 2. Critical Academic Review

### 2.1 Strengths

1. **Training-free conditioning**: The LGD approach means any new loss function can be plugged in without retraining. This is architecturally elegant and practically important -- the entire physics module is differentiable but external to the learned model.

2. **Unified physics framework**: The penetration and mobility losses operate on the same SDF representation used for shape generation, creating a mathematically consistent system. The volume-element scaling and existence-weighted formulation ensure gradients flow through the full graph structure.

3. **No graph structure assumption**: Unlike PhysPart (which assumes known base + graph topology), PhysNAP discovers the graph structure during denoising via continuous existence indicators. This is fundamentally harder and more general.

4. **Plucker coordinate choice**: Using 6D Plucker coordinates for joint axes avoids gimbal lock and unifies revolute/prismatic joints in a single differentiable representation.

### 2.2 Limitations

1. **Zero-articulation assumption**: All parts are assumed to share a canonical orientation at zero joint state. The model predicts only 3D translation, not SE(3) rotation per part. This breaks when the object is observed at a non-canonical global pose. The authors acknowledge this requires a separate pose estimation module.

2. **128-dim shape bottleneck**: The compact latent vector cannot capture high-frequency geometric details (handles, buttons, thin features). Generated meshes exhibit "blobby artifacts" compared to retrieved meshes (Table 2, supplementary D).

3. **Computational cost**: Full guidance (pc+pen+mob) takes ~2 min/sample on A40. The mobility loss alone accounts for most of this overhead due to multi-state SDF queries. Point-cloud-only guidance is 15 sec.

4. **PartNet-Mobility data scale**: ~2000 objects across 17 categories is small by modern standards. Box-like categories (Storage Furniture) have limited diversity. This constrains the generative prior's coverage.

5. **Single-point-cloud conditioning**: Only frontal-view partial point clouds tested. No multi-view fusion, no depth noise modeling, no real-world sensor artifacts.

### 2.3 Observations Specific to Infinigen Integration

The user identified two crucial limitations in PhysNAP's training protocol:

1. **Training URDFs use only closed/zero-articulation state** -- no diversity in joint configurations during training data preparation.
2. **Observations are from a single (frontal) viewpoint** -- limited viewpoint diversity.

Additional observations after deep review:

- **Data bottleneck for box categories**: PartNet-Mobility's "Storage Furniture" and similar categories have few box-like objects. Infinigen can generate parametrically unlimited box variants with physics-validated properties, directly addressing this gap.

- **SDF representation gap**: NAP uses a pretrained DeepSDF-like autoencoder (128-dim latent). Infinigen provides OBJ meshes. The conversion path is: OBJ mesh → watertight mesh → SDF point sampling → encode via NAP's pretrained encoder → 128-dim latent.

- **Joint parameterization compatibility**: NAP uses Plucker coordinates (l, m) where l is the axis direction and m = origin × direction. Infinigen's URDF provides `<axis xyz>` (direction) and `<origin xyz>` (point on axis), which are directly convertible.

- **Graph size constraint**: NAP uses max K=8 nodes. Infinigen boxes have 2-9 parts: MAILER=3, DRAWER=2, SLIP_LID=2, TUCK_END=9. TuckEndBox (9 parts) exceeds K=8 and would need to be either excluded or have the model extended.

- **Translation-only assumption holds**: Infinigen's boxes are axis-aligned in their canonical (closed) state, so the zero-rotation assumption is naturally satisfied.

---

## 3. Feasibility: Retraining PhysNAP with Infinigen Data

### 3.1 Data Format Mapping

NAP expects per-object data in `partnet_mobility_graph_v4/` format:

| NAP Requirement | Infinigen Phase-1 Output | Conversion Needed |
|-----------------|--------------------------|-------------------|
| Node existence (binary per K=8) | Link count from URDF | Pad to K=8 with zeros |
| 3D translation per node | `<joint><origin xyz>` | Extract from URDF parse |
| Bounding box extents (R^3) | Computed from OBJ vertices | Per-link mesh AABB |
| Shape latent (R^128) | Per-link OBJ meshes | Merge OBJs per link → watertight → SDF sample → encode with NAP's pretrained AE |
| Edge existence + direction | `<joint>` parent-child | Map to NAP's edge format |
| Plucker coordinates (R^6) | `<axis xyz>` + `<origin>` | l = axis_xyz, m = origin × axis_xyz |
| Joint state ranges (R^{2x2}) | `<limit lower upper>` | Direct mapping |
| Category label | `box_type` in metadata.json | New category index |

### 3.2 Conversion Pipeline Architecture

```
Infinigen Phase-1 Sample
├── urdf_gt.urdf          ──→  Parse links, joints, origins, axes, limits
├── assets/geom_*.obj     ──→  Group by link → merge → make watertight → normalize
├── metadata.json         ──→  Category label, seed, material info
└── segmentation_label_map.json ──→  Link name ↔ ID mapping

                    ↓ infinigen_to_nap.py

NAP Graph Format
├── node_attrs: [K, D_v]  (existence, translation, bbox, shape_latent)
├── edge_attrs: [K(K-1)/2, D_e]  (existence, plucker, joint_ranges)
├── category: int
└── mesh_per_part: {i: trimesh.Trimesh}
```

### 3.3 Technical Challenges

**Challenge 1: Per-link mesh merging**
Infinigen exports multiple OBJ files per link (e.g., link_0 has geom_0..geom_4 = 5 visual meshes). NAP expects a single watertight mesh per part. Solution: use trimesh to merge all visual OBJs for a link, then apply watertight conversion (e.g., trimesh.repair or PyMeshLab's Screened Poisson reconstruction).

**Challenge 2: SDF encoding**
NAP's pretrained shape autoencoder expects normalized point clouds on part surfaces. Conversion: sample 2048 points on each watertight mesh surface → normalize to unit sphere → encode via NAP's PointNet encoder → 128-dim latent. The encoder quality on Infinigen's box geometries (flat surfaces, sharp edges) versus PartNet-Mobility's organic shapes is the primary academic risk.

**Challenge 3: Coordinate normalization**
Infinigen uses physical meters; NAP uses normalized coordinates (max extent = 1). All translations, bounding boxes, and Plucker moment vectors must be rescaled.

**Challenge 4: TuckEndBox (K=9 > K_max=8)**
Options: (a) exclude TuckEndBox from training, (b) extend NAP to K=10, (c) merge small flaps into parent links. Recommend (a) initially for clean baseline.

### 3.4 Gap Analysis: Infinigen Phase-1 Augmentation Needs

| Requirement | Status | Action |
|-------------|--------|--------|
| Per-link merged watertight mesh | Not available (multiple OBJs per link) | Add merge+watertight step to conversion |
| Normalized coordinates | Physical meters | Normalization in converter |
| Plucker coordinates | URDF axis + origin available | Conversion formula in converter |
| 128-dim shape latent | Not available | Encode using NAP's pretrained AE |
| Multiple articulation states | 1-2 states per seed | Not needed for NAP training (zero-state only), but valuable for PhysNAP evaluation |
| Multi-viewpoint point clouds | Multiple viewpoints available in Phase-1 | Already satisfied |

---

## 4. Academic Value Assessment

### 4.1 Contribution Dimensions

```
┌─────────────────────────────────────────────────────────────────┐
│                   Academic Value: HIGH                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DATA AUGMENTATION                                           │
│     PartNet-Mobility: ~2000 objects (fixed)                     │
│     Infinigen: unlimited parametric boxes (procedural)          │
│     → Tests "does more diverse training data improve NAP?"      │
│                                                                 │
│  2. PHYSICS-ALIGNED TRAINING DATA                               │
│     Infinigen URDFs: collision-free joints (PyBullet verified)  │
│     → Directly aligned with PhysNAP's l_pen/l_mob objectives   │
│     → Hypothesis: physics-valid training reduces guidance cost  │
│                                                                 │
│  3. NOVEL CATEGORY                                              │
│     "Packaging boxes" not in PartNet-Mobility                   │
│     → Domain transfer validation for NAP framework              │
│                                                                 │
│  4. MULTI-STATE TRAINING (future)                               │
│     Infinigen can render at arbitrary joint states               │
│     → Enables articulation-state-aware priors                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Experimental Design for Publication

**Experiment A: Mixed Training (PartNet + Infinigen)**
- Merge PartNet-Mobility + Infinigen 1K box dataset
- Add "box" as new category
- Retrain NAP diffusion model (120K steps)
- Evaluate: does Infinigen data improve E_pen/E_mob without hurting MMD/1-NNA on existing categories?

**Experiment B: Infinigen-Only Training**
- Train NAP solely on Infinigen box data (3 types: MAILER, DRAWER, SLIP_LID)
- Evaluate generative quality within box domain
- Tests: can NAP learn from procedurally generated data alone?

**Experiment C: Physics Pre-training Hypothesis**
- Compare PhysNAP guidance efficiency (steps to convergence) between:
  - Model trained on standard PartNet-Mobility
  - Model trained on physics-validated Infinigen data
- Hypothesis: if training data already satisfies physics constraints, fewer guidance steps are needed → faster inference

### 4.3 Academic Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Shape AE quality on box geometries | MEDIUM | Fine-tune AE on box shapes if reconstruction error > threshold |
| Scale mismatch (meters vs normalized) | LOW | Standard normalization procedure |
| Limited geometric diversity (boxes are simple) | MEDIUM | Argue simplicity tests framework generalization |
| TuckEndBox exceeds K=8 | LOW | Exclude from initial experiments |
| Infinigen mesh quality (watertight conversion) | MEDIUM | Validate SDF reconstruction error per sample |

---

## 5. Implementation Plan

### 5.1 PhysNAP Reproduction

```bash
# Environment (separate from infinigen due to Python 3.9 requirement)
conda create -n physnap python=3.9 gcc_linux-64=9 gxx_linux-64=9 -y
conda activate physnap
# Install PyTorch 2.0 + CUDA 11.7
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu117
# Install PyTorch3D, PyG, etc. via env.sh
bash external/physnap/env.sh
```

**Data**: Download from NAP GitHub releases:
- `partnet_mobility_graph_v4/` (preprocessed graph data)
- `s1.5_partshape_ae/` (pretrained shape autoencoder weights)

**Training**: `python run.py --config ./configs/nap/v6.1_diffusion_adapted.yaml -f`
- 120K steps, batch=64, ~9 hours on single A800 GPU

### 5.2 Infinigen → NAP Converter

New script: `scripts/infinigen_to_nap.py`

Pipeline per sample:
1. Parse URDF → extract links, joints, axes, limits, parent-child structure
2. Load per-link OBJs → merge sub-meshes per link
3. Make watertight (trimesh repair + fill holes)
4. Normalize to unit cube (center at origin, max extent = 1)
5. Compute 3D translations from URDF joint origins (in normalized space)
6. Compute bounding boxes from normalized mesh vertices
7. Convert joint axes to Plucker: l = axis_xyz (normalized), m = origin × l
8. Encode part shapes: sample 2048 surface points → NAP's pretrained PointNet encoder → 128-dim latent
9. Build graph: nodes (parts, padded to K=8) + edges (joints) in NAP format
10. Write to `partnet_mobility_graph_v4/` structure

### 5.3 Retraining Strategy

**Recommended sequence**:
1. **Option A first** (mixed PartNet + Infinigen) -- validates data pipeline end-to-end
2. **Option B** (Infinigen-only) -- clean academic signal for contribution claim
3. **Option C** (fine-tune from PartNet pretrained) -- best practical performance

### 5.4 Evaluation Protocol

Generate samples for 30 test objects (Infinigen boxes, diverse seeds):
- E_pc, D_pc: point cloud alignment quality
- E_pen, E_mob: physics plausibility
- MMD, 1-NNA: generative diversity/quality
- Compare against PartNet-Mobility-only baseline

### 5.5 Expected Outcomes

- If E_pen/E_mob improve with Infinigen data → validates physics-aligned data engine value
- If MMD/1-NNA maintained or improved → validates diversity contribution
- If guidance steps to convergence decrease → demonstrates physics pre-training hypothesis
- If box-category generation quality is high → proves domain-specific data engine utility

---

## 6. Connection to Broader Research Context

### 6.1 Positioning in the Field

```
              Articulated Object Generation Landscape (2024-2025)
              ────────────────────────────────────────────────────

  Visual Input                          Geometric Input
  ┌─────────────────────┐              ┌─────────────────────┐
  │ SINGAPO (ICLR 2025) │              │ PhysNAP (ICCV 2025) │ ← this work
  │ DIPO (CVPR 2025)    │              │ PhysPart (ICRA 2025)│
  │ CAGE (CVPR 2024)    │              │ MIDGaRD (NeurIPS 24)│
  │ URDFormer (RSS 2024)│              │ NAP (NeurIPS 2023)  │
  └─────────────────────┘              └─────────────────────┘
         │                                       │
         └──── Both need ──── Training Data ─────┘
                                 │
                          PartNet-Mobility
                           (~2000 objects)
                                 │
                     ┌───────────┴───────────┐
                     │ LIMITATION: fixed,    │
                     │ small-scale, no boxes  │
                     └───────────┬───────────┘
                                 │
                     ┌───────────┴───────────┐
                     │  Infinigen Data Engine │ ← our contribution
                     │  Unlimited parametric  │
                     │  Physics-validated     │
                     │  Domain-randomized     │
                     └───────────────────────┘
```

### 6.2 Infinigen's Unique Academic Value

Infinigen occupies a unique niche: it is **not** a perception model, **not** a generation model, but a **data engine** that can amplify the training signal for both. For PhysNAP specifically:

1. **Quantity**: Procedural generation can produce 10K+ samples vs PartNet-Mobility's ~2000
2. **Quality**: Physics-validated URDFs (PyBullet verified, 6-level validation framework)
3. **Diversity**: Domain randomization across geometry, material, joint states, viewpoints
4. **Novelty**: "Packaging box" category not represented in existing datasets

This positions Infinigen as a **synthetic data augmentation engine** for articulated object understanding, with PhysNAP retraining as the primary validation experiment.
