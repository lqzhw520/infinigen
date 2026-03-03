# Ralph Agent Log (MailerBox_Simple Phase 1)

This file tracks each iteration (max 5) and what was completed.

---

## Template

**Iteration:** N/5  
**User story:** (copy exact `description`)  
**Status:** Completed | In Progress | Blocked  

**Changes:**
- `path/to/file` - what changed

**Verification:**
- command(s) run + brief result summary

**Notes:**
- blockers / follow-ups

---

## 2026-02-03 - Iteration 1/5

**User story:** MailerBox_Simple domain-randomized material system (Phase 1.3)  
**Status:** Completed  

**Changes:**
- `infinigen/assets/sim_objects/modular_box_factory.py` - Extended `BoxMaterialConfig` (physics + shader params), improved `sample_material`, and upgraded `_apply_box_material` (Principled BSDF + optional bump + `_deepcopy_<variant_id>` naming).
- `infinigen/assets/sim_objects/box_material_domain_randomization.py` - Added deterministic material sampler + JSON metadata helper.

**Verification:**
- End-to-end smoke test later confirms `metadata.json` includes sampled \(ρ, μ, e\) and URDF masses change accordingly.

---

## 2026-02-03 - Iteration 2/5

**User story:** MailerBox_Simple auto-labeling pipeline exports RGB-D + masks + keypoints + URDF (Phase 1.4)  
**Status:** Completed  

**Changes:**
- `scripts/export_mailerbox_simple_phase1_data_engine.py` - Added Phase-1 data engine demo script:
  - exports URDF GT (visual-only) with dom-rand materials
  - imports per-link visual meshes, applies FK poses, renders RGB
  - renders depth + object-index segmentation via compositor OutputFile, converts EXR to numpy via Blender image IO
  - writes `depth.npy`, `segmentation.npy/png`, `instance.npy/png`, `keypoints.json`, `joint_state.json`, `camera_intrinsics.json`, `metadata.json`, `urdf_gt.urdf`

**Verification:**
- Command:
  - `python -m infinigen.launch_blender -s scripts/export_mailerbox_simple_phase1_data_engine.py -- --seeds 101 --n_views 1 --joint_states "0,0" --width 160 --height 120 --out_root sim_exports/data_engine/_smoke_test5`
- Result:
  - Created sample folder with the full Phase-1 outputs and `manifest.json`.
  - Follow-up fix: set render resolution + `camera.adjust_camera_sensor` before saving `camview*.npz`, so `camera_intrinsics.json` matches `(W,H)` and 2D keypoints project in-frame.

---

## 2026-02-03 - Iteration 3/5

**User story:** Update `infinigen-scientific-dev` skill with materials + auto-labeling capabilities  
**Status:** Completed  

**Changes:**
- `./.cursor/skills/infinigen-scientific-dev/SKILL.md`
  - Added Phase 1.3 DR materials protocol (determinism + physics coupling + shader contract)
  - Added Phase 1.4 auto-labeling protocol (RGB-D + seg + keypoints + camera params + URDF GT schema)
  - Added code pointers to the new sampler + exporter script
- `./.cursor/skills/infinigen-scientific-dev/verification-templates.md`
  - Added sample-folder integrity checker template
  - Added DR determinism test template

**Verification:**
- Updated `docs/user-stories/mailerbox-simple-phase1.json` → all stories `passes=true`.

---

