# Overnight sandbox log — box_conditioning_v2 Phase 2 ablations

## 2026-03-20

- Pre-flight (`aris-workspace/skills/infinigen-overnight-review/scripts/preflight.py`): **PASS**
- Extended `scripts/build_conditioning_v2_dataset.py` (Infinigen repo, **not** PhysNAP source):
  - `aggregation`: `merge_all` | `fixed_state_multiview` | `best_single_view`
  - `best_single_view`: chamfer proxy vs subsampled mesh vertices from `assets/*.obj`
  - Per-group `points_per_cloud` for point-budget sweep
  - CLI overrides: `--points-per-cloud`, `--aggregation`, `--fixed-state-index`
- Created sandbox campaign `campaign_phase2_ablations/manifest.yaml` (6 groups)
- Added `scripts/evaluate_overnight_ablations.py` and driver `runs/phase2_ablations_20260320/overnight_phase2_ablations.sh`
- Launched overnight driver via **nohup** (`runs/phase2_ablations_20260320/nohup.log`): this host had no usable `screen` socket dir (`/run/screen/S-root`). Skill prefers `screen`; use `screen` on machines where it works.
- Fixed launcher bugs: bash **`GROUPS` is reserved** (was iterating `0`); `build_conditioning_v2_dataset.py` needs `--campaign-dir` **before** the `export` subcommand.

**PhysNAP source** (`external/physnap/**/*.py`): **unchanged**. Guided outputs use existing `run_guided.py` → `external/physnap/log/overnight_box_cond_v2__*`.

**Frozen campaign** `experiments/physnap/box_conditioning_v2/`: **not modified** (only read + copied plan/acceptance/state into sandbox).

## Resume

- `tail -f experiments/physnap/_overnight_sandbox/runs/phase2_ablations_20260320/output.log`
- `tail -f .../nohup.log` (same pipeline; nohup wrapper)
- `ps aux | grep overnight_phase2_ablations` / `pgrep -af phase2_ablations`
- Results: `experiments/physnap/_overnight_sandbox/evaluation/overnight_ablations_report.md`
