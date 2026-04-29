# Campaign Harness Publication Fix Report

## Summary

Publication fix executed at 2026-04-29T14:15:00+08:00.

Root cause: Previous attempts claimed production_ready but files were only staged locally and never committed to the authoritative A800 repo or pushed to GitHub.

## What Was Wrong

| Issue | Previous State | Fixed State |
|-------|---------------|-------------|
| Remote A800 HEAD | `39593b19` (stale) | `350045fa` (harness installed) |
| GitHub visibility | No harness files | All 9 paths return HTTP 200 |
| Preflight callable | Not run from A800 | V01+V02 both PASS |
| Lock file hashes | Local mirror hashes | A800 committed file hashes |

## Commits on A800

```
350045fa fix(harness): add from __future__ import annotations for Python 3.8 compatibility
cc716aa7 chore(mint): install campaign-native agent execution harness
39593b19 chore: update V11 status surfaces with hygiene sprint + pre-commit uv fix
```

## Files Published

| File | Commit | GitHub Raw |
|------|--------|------------|
| `CLAUDE.md` | cc716aa7 | ✅ HTTP 200 |
| `experiments/mint/mint_drawer_v1/CLAUDE.md` | cc716aa7 | ✅ HTTP 200 |
| `experiments/mint/mint_drawer_v1/AGENT_EXECUTION_GUARDRAILS.md` | cc716aa7 | ✅ HTTP 200 |
| `experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py` | 350045fa | ✅ HTTP 200 |
| `experiments/mint/mint_drawer_v1/scripts/harness/validators/validate_task_authority.py` | 350045fa | ✅ HTTP 200 |
| `experiments/mint/mint_drawer_v1/scripts/harness/validators/validate_diff_scope.py` | 350045fa | ✅ HTTP 200 |
| `experiments/mint/mint_drawer_v1/scripts/harness/validators/validate_closeout.py` | 350045fa | ✅ HTTP 200 |
| `experiments/mint/mint_drawer_v1/sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml` | cc716aa7 | ✅ HTTP 200 |
| `experiments/mint/mint_drawer_v1/autopilot/agent_execution_harness_lock.json` | cc716aa7 | ✅ HTTP 200 |

Also committed (integration evidence):
- `experiments/mint/mint_drawer_v1/runtime/campaign_harness_integration/inventory.json`
- `experiments/mint/mint_drawer_v1/runtime/campaign_harness_integration/inventory.md`
- `experiments/mint/mint_drawer_v1/runtime/campaign_harness_integration/real_regression_results.json`

## Python 3.8 Compatibility Fix

A800 runs Python 3.8 which does not support `list[str]`, `dict[str, Any]` etc. in function annotations. Fixed by adding `from __future__ import annotations` to all four Python files. This is a mechanical compatibility fix, not a semantic change.

## Remote Preflight Dry-Run Results

```
campaign_root: /mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1
[V01] validate_task_authority — PASS
[V02] validate_diff_scope — PASS
ALL PRE-FLIGHT VALIDATORS PASSED.
```

## Validation

- `harness_status = production_ready`
- `publication_complete = true`
- `github_visibility_complete = true`
- `campaign_preflight_callable = true`
- `remote_commit_hash = 350045fa`
- `remote_pushed_to_origin = true`
- `runtime_code_modified = false`
- `rollout_render_train_run = false`

## Why It Failed Before

Previous attempts used `git add .` or `git add experiments/mint/...` on the LOCAL workspace, then claimed success. But the local workspace is NOT the authoritative A800 repo. The actual authoritative repo at `/mnt/afs2/zhuhaowu/infinigen` on the A800 server never received the files.

Fix: explicitly `scp` each file from local to A800, then `git add` on A800, commit on A800, push from A800.
