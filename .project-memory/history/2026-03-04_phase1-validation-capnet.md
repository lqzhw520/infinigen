# Phase-1 Data Engine Validation + CAPNet Academic Alignment Report

**Date**: 2026-03-04  
**Iteration**: #6  
**Branch**: feature/3d-assets

---

## Executive Summary

This iteration delivers two major capabilities for the Infinigen-AnyBox Phase-1 Data Engine:

1. **Comprehensive 6-Level Validation Framework** -- automated, integrated into the export pipeline, covering all 9 historically-identified validation gaps. Result: **1000/1000 samples pass** across all 4 box types.

2. **CAPNet Academic Alignment Bridge** -- converts Phase-1 output to CAPNet-compatible format with per-link 6D pose, 3D size, NPCS parameters, and evaluation metrics (Re/Te/Se/A5/A10). Self-consistency verified at perfect scores.

---

## Part 1: What Was Done (Step-by-Step)

### Step 1: Created `scripts/validate_dataset.py` -- Comprehensive 6-Level Validator

**Need addressed**: Existing validation was fragmented across two scripts (`verify_phase1_sample_folder.py` and `verify_urdf_inertia_fix.py`), with 9 concrete gaps identified from historical bugs.

**What it does**: Runs ALL checks on a full Phase-1 dataset directory. Fail-fast per sample (if Level N fails, skip N+1).

| Level | Check | What It Catches |
|-------|-------|----------------|
| L1 | File Presence | Missing rgb.png, depth.npy, urdf_gt.urdf, etc. (12 files) |
| L2 | Array Integrity | NaN depth, shape mismatches, invisible objects (>99% background) |
| L3 | URDF Structure | Multiple inertials (Bug #1), missing collision (Bug #2), missing limits (Bug #3), missing material (Bug #4), DOF mismatch |
| L4 | Physics Validation | PyBullet load, joint states within limits, inertia positive-definiteness + triangle inequality |
| L5 | Cross-Consistency | Keypoint names vs URDF joints, label_map links vs URDF links, joint_state names vs URDF |
| L6 | Batch Statistics | Seed diversity, material distribution, joint state coverage, anomaly detection |

**Output**: `verification_report.json` per dataset with per-sample pass/fail and batch-level statistics.

### Step 2: Created `tests/sim/test_validation_regression.py` -- 7 Regression Tests

**Need addressed**: 6 historical bugs were fixed in code but NOT encoded as automated tests.

**What it does**: For each bug, constructs a minimal pathological URDF sample and verifies the validator correctly detects it.

| Test | Historical Bug | Assertion |
|------|---------------|-----------|
| TestBug1 | Multiple `<inertial>` per link | check_level3 fails |
| TestBug2 | Missing `<collision>` tags | check_level3 fails |
| TestBug3 | Missing `<limit>` on revolute joints | check_level3 fails |
| TestBug4 | Missing `<material><color>` | check_level3 warns |
| TestBug5 | DOF mismatch (MAILER URDF claimed as TUCK_END) | check_level3 fails |
| TestBug6 | Joint state outside URDF limits | check_level4 fails |
| TestGoodSample | Well-formed sample | Passes all levels |

**Result**: 7/7 PASS.

### Step 3: Ran Full 6-Level Validation on All 4 Datasets

**Need addressed**: Prove that the existing 1K dataset is "usable without thinking" for downstream.

| Dataset | Samples | L1-L5 | L6 Anomalies | Seeds | Materials | JS Std |
|---------|---------|-------|-------------|-------|-----------|--------|
| phase1_1k_mailer | 250 | 250/250 PASS | 0 | 25 | 9 types | 0.5196 |
| phase1_1k_drawer | 250 | 250/250 PASS | 0 | 25 | 8 types | 0.0250 |
| phase1_1k_sliplid | 250 | 250/250 PASS | 0 | 25 | 7 types | 0.0150 |
| phase1_1k_tuckend | 250 | 250/250 PASS | 0 | 25 | 9 types | 0.2915 |
| **TOTAL** | **1000** | **1000/1000** | **0** | - | 10+ | - |

### Step 4: Upgraded Export Pipeline to Full 6-Level Auto-Validation

**Need addressed**: User asked "why only L1-3?" -- there was no fundamental blocker for L4-L6.

**Analysis of "blockers"**:
- L4 (Physics): Inertia checks are pure XML parsing -- no PyBullet needed. PyBullet load is a bonus check that gracefully degrades if unavailable.
- L5 (Cross-consistency): Pure JSON + XML comparison -- zero dependencies.
- L6 (Batch stats): Simple aggregation after all samples processed.

**Solution**: Upgraded `_run_post_export_validation()` in the export script to run all 6 levels, generate `verification_report.json`, and report batch anomalies. Performance: ~35s for 250 samples (0.14s/sample).

### Step 5: Created `scripts/capnet_data_bridge.py` -- CAPNet Format Converter

**Need addressed**: Infinigen-AnyBox Phase-1 output needs to feed into CAPNet (CVPR'25) for the research group's "整个流程" pipeline.

**What it computes per sample**:
- Per-link 6D pose (R, t) via forward kinematics from URDF + joint_state
- Per-link 3D size from mesh vertex AABB in world frame
- Per-link NPCS normalization parameters (scale, offset)
- Object-level pose and bounding box

**Output files**:
- `capnet_annotations/<sample_id>.json` -- per-sample full annotation
- `capnet_annotations/link_pos_quat_aabb.json` -- all link annotations
- `capnet_annotations/obj_pos_quat_aabb.json` -- all object annotations
- `capnet_annotations/summary.json` -- dataset statistics

### Step 6: Created `scripts/eval_capnet_metrics.py` -- CAPNet Evaluation Protocol

**Need addressed**: The research group uses specific metrics (Re/Te/Se/A5/A10) from the CAPNet paper.

**Metrics implemented**:
| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| Re (°) | arccos((tr(R_pred^T R_gt) - 1) / 2) | Geodesic rotation error |
| Te (cm) | ‖t_pred - t_gt‖₂ × 100 | Translation error |
| Se | mean(|s_pred - s_gt| / s_gt) | Relative scale error |
| mIoU | 3D AABB intersection / union | Bounding box overlap |
| A5 | 1{Re < 5° ∧ Te < 5cm} | Strict accuracy |
| A10 | 1{Re < 10° ∧ Te < 10cm} | Relaxed accuracy |

Also includes Umeyama alignment and self-consistency check mode.

### Step 7: Generated CAPNet Annotations for All 1K Samples

| Dataset | Samples | Part Annotations | Self-Consistency |
|---------|---------|-----------------|-----------------|
| mailer | 250 | 750 (3 parts each) | Re=0, Te=0, Se=0, IoU=1.0 |
| drawer | 250 | 500 (2 parts each) | Re=0, Te=0, Se=0, IoU=1.0 |
| sliplid | 250 | 500 (2 parts each) | Re=0, Te=0, Se=0, IoU=1.0 |
| tuckend | 250 | 2250 (9 parts each) | Re=0, Te=0, Se=0, IoU=1.0 |
| **Total** | **1000** | **4000** | **All perfect** |

### Step 8: Academic Alignment Analysis

Created `docs/CAPNet_Academic_Alignment_Analysis.md` covering:
- Why "size is mediocre" (data bottleneck, scale ambiguity, topology sensitivity)
- Innovation space (data contribution, physics-aligned advantage, topology-aware prediction)
- Experimental plan to validate Infinigen improves CAPNet's Se metric
- Academic positioning for CVPR/ICRA submission

---

## Part 2: Files Created/Modified

### New Files

| File | Purpose | Lines |
|------|---------|-------|
| `scripts/validate_dataset.py` | 6-level comprehensive validator | ~500 |
| `scripts/capnet_data_bridge.py` | Phase-1 → CAPNet format converter | ~300 |
| `scripts/eval_capnet_metrics.py` | CAPNet metrics (Re/Te/Se/A5/A10) | ~250 |
| `tests/sim/test_validation_regression.py` | 7 regression tests for 6 historical bugs | ~200 |
| `docs/CAPNet_Academic_Alignment_Analysis.md` | Academic alignment analysis | ~200 |
| `docs/Phase1_DataEngine_Validation_CAPNet_Report_2026-03-04.md` | This report | - |

### Modified Files

| File | Change |
|------|--------|
| `scripts/export_mailerbox_simple_phase1_data_engine.py` | Upgraded post-export validation from L1-3 to full L1-6 with batch stats and report generation |

### Generated Artifacts

| Path | Content |
|------|---------|
| `sim_exports/data_engine/phase1_1k_*/verification_report.json` | Full 6-level validation reports (4 files) |
| `sim_exports/data_engine/phase1_1k_*/capnet_annotations/` | CAPNet-format annotations (4 dirs, 1000+ files) |

---

## Part 3: Verification Commands

```bash
# Full 6-level validation on any dataset
python scripts/validate_dataset.py sim_exports/data_engine/phase1_1k_mailer/ --level 6

# Regression tests (7 tests encoding 6 historical bugs)
python -m pytest tests/sim/test_validation_regression.py -v

# CAPNet self-consistency check (GT vs GT = perfect scores)
python scripts/eval_capnet_metrics.py sim_exports/data_engine/phase1_1k_mailer/

# Generate CAPNet annotations for a dataset
python scripts/capnet_data_bridge.py sim_exports/data_engine/phase1_1k_mailer/
```

---

## Part 4: Bugs Fixed in This Iteration

| # | Bug | Root Cause | Fix |
|---|-----|-----------|-----|
| 7 | joint_positions dict/list ambiguity | Export writes `{name: val}` dict, old code assumed list | Handle both formats in validator and bridge |
| 8 | Depth background (1e10) flagged as error | Blender Z-pass uses large values for sky | Foreground-only range check (< 100m) |

---

## Part 5: What This Enables (Next Steps)

1. **Downstream planner**: The 1000 validated samples can be used "without thinking" -- all 6 levels guarantee file integrity, URDF correctness, physics validity, and cross-consistency.

2. **CAPNet training**: The `capnet_annotations/` directories contain per-link 6D pose, 3D size, and NPCS parameters ready for CAPNet's training pipeline. Addressing the "size一般" finding by providing parametrically diverse box sizes.

3. **10K scaling**: The validated pipeline can now be scaled to 10K+ samples with confidence -- the auto-validation catches any issues immediately.

4. **Academic submission**: The alignment analysis positions Infinigen-AnyBox as a novel data contribution to the CAPNet pipeline, with clear innovation space and experimental plan.
