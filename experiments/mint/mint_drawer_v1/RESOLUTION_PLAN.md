# MINT Drawer v1 — Complete Root-Cause Resolution Plan (v2)

**Date:** 2026-03-27 (revised after full B1/U5/C2 evidence review)  
**Status:** ACTIVE — Phase 0 complete, ready for strong_rollout_audit + D1  
**Claim:** `Infinigen-generated AnyGrasp-conditioned robot-arm drawer trajectories improve MINT success on held-out drawer variants in simulation relative to pretrained MINT.`  
**Immediate Blocker:** ~~D2 `strong_rollout_count=1`~~ → **resolved**: clean `c2_replay_valid_rollouts/` contains 25 rollouts; provisional `strong_coherent_seeds=[2,8]`. Next step: run `run_strong_rollout_audit.py` to verify NPZ grasp/pull criteria, then launch D1.

---

## AUDIT UPDATE — 2026-03-27T14:00:00+08:00

**Root cause of 4-day confusion identified and resolved:**

The `strong_rollout_audit.json` was pointing at the **archived** path `artifacts/diagnostic_runs/2026-03-26_gate_drift_reset/c2_replay_valid_rollouts/` which had been cleaned out. The live `artifacts/c2_replay_valid_rollouts/` directory contains **25 rollouts (seeds 2, 7, 8, 9, 10)** that were never audited against the strong_rollout_criteria.

JSON-criteria audit results:
- **seed_002**: ep01 (pre=0.019), ep02 (pre=0.006), ep04 (pre=0.0009) — 3 passing → **coherent**
- **seed_008**: ep00 (pre=0.0), ep01 (pre=0.0001), ep02 (pre=0.003), ep03 (pre=0.0) — 4 passing → **coherent**
- **seed_007**: ep00 only (pre=0.118) — 1 passing → not coherent
- **seed_010**: ep03 only (pre=0.0, post=1.0) — 1 passing → not coherent
- **seed_009**: none pass

**P0 and P1 status revision:**
- P0 (data floor): **partially resolved** — strong_coherent_seeds=[2,8] provisionally available
- P1 (handle heuristic): **lower priority than stated** — seeds 2 and 8 do not require the handle patch; seed_010 needs only 1 more coherent rollout which ep03 provides if npz criteria pass
- P2 (D1 gate code fix): **still required** before D1 launch
- P3 (engineering cleanup): **complete** — d1_candidate_progress.json deleted, state.json reset, stale lease released

**Remaining work before D1:**
1. Run `run_strong_rollout_audit.py --source_dir artifacts/c2_replay_valid_rollouts/` (NPZ grasp/pull verification)
2. Confirm `strong_coherent_seeds` non-empty in output
3. Apply P2 code fix (pre_attach_motion filter + coherence hard gate in `run_d1_single_rollout_overfit.py`)
4. Launch D1 with `seed_002_ep04` as rank-1 mainline candidate

---

## 0. Corrected Priority Table

Current priority fixes are **engineering and data problems only**.  
MINT model architecture and all its hyperparameters must NOT be modified until D2 passes cleanly on correct data. This is an absolute scientific constraint — changing the model before the data is clean conflates data quality with model capacity.

| Priority | Layer | Root Cause | Status | Fix |
|----------|-------|-----------|--------|-----|
| P0 | L3: Data Floor | `strong_rollout_audit.json` pointed at cleaned-out `diagnostic_runs/` — missed all 25 clean C2 rollouts | **RESOLVED** — audit rebuilt; provisional `strong_coherent_seeds=[2,8]`; NPZ verification pending | Run `run_strong_rollout_audit.py` against `c2_replay_valid_rollouts/` to confirm |
| P1 | L2: Gate Design | `run_d1_single_rollout_overfit.py` silently falls back to `_ordered_mainline_candidates()` when `strong_coherent_seeds=[]`; D1 also lacks `pre_attach_motion≤0.12` pre-filter | **Pending code fix** — must apply before D1 launch | Add hard coherence gate + pre_attach filter in `strict_teacher_dataset.py` and remove fallback in `run_d1_single_rollout_overfit.py` |
| P2 | L4: Upstream | Infinigen handle heuristic: `part_labels_present=false` on all seeds; affects AnyGrasp→metadata alignment on seeds 2, 10 | **Lower priority** — seeds 2 and 8 are coherent without handle patch; seed_010 needs only ep03 NPZ confirmation | Apply only if NPZ audit shows seed_010_ep03 fails grasp/pull; otherwise defer to post-D2 |
| P3 | L1: Engineering | Stale `d1_candidate_progress.json`, stale `strong_rollout_audit.json`, stale controller lease | **COMPLETE** — all stale files deleted or rebuilt; `state.json` reset; lease released | Done |
| P4 | L0: Held-Out | Seeds 11-15 have ZERO successes across ALL modalities | **Unresolved** — latent E1 blocker | Run oracle probe on seeds 11-15 before E1; apply u3 patch to held-out seeds if needed |

---

## 1. Critical New Evidence: Seeds 2 and 10 Are AnyGrasp-Only Seeds

**From `b1_oracle_scripted_baseline.json` and `u5_seed_outlier_audit.json`:**

| Seed | oracle | anygrasp | c1_teacher | c2_teacher | Type |
|------|--------|----------|------------|------------|------|
| 1    | 6      | 0        | 5          | 0          | oracle-only |
| 2    | **0**  | 3        | 0          | 2          | **AnyGrasp-only** |
| 3    | 4      | 0        | 5          | 0          | oracle-only |
| 4    | 3      | 0        | 0          | 0          | oracle-only (weak) |
| 5    | 5      | 0        | 4          | 0          | oracle-only |
| 6    | 2      | 0        | 0          | 0          | oracle-only (weak) |
| 7    | 3      | 2        | 0          | 3          | mixed |
| 8    | 3      | 1        | 0          | 0          | mixed |
| 9    | 5      | 2        | 5          | 4          | both strong |
| 10   | **0**  | 3        | 0          | 2          | **AnyGrasp-only** |
| 11-15| 0      | 0        | 0          | 0          | **unsolved — latent E1 blocker** |

**U5 root_cause field:** `oracle_handle_heuristic_mismatch`  
**U5 affected_seeds:** `[2, 10]`

**Mechanistic interpretation:**
- The oracle controller and the `handle_center_world` heuristic both use the same bounding-box estimate
- On seeds 2 and 10, this estimate is wrong enough that the scripted oracle arm completely misses the handle (0/6 success)
- AnyGrasp uses visual point cloud detection and finds the actual handle, achieving 3/6 despite the wrong metadata
- AnyGrasp success generates `pre_attach_motion` because the arm then adjusts from the metadata-predicted pose to the actual grasp pose
- Fixing `handle_center_local` in metadata to match the AnyGrasp actual grasp → robot starts closer to the true handle → fewer corrections → `pre_attach_motion` drops below 0.12

**Scientific significance of seeds 2 and 10 for the claim:**  
These are the cases where the Infinigen→AnyGrasp pipeline adds value over pure oracle: AnyGrasp succeeds where oracle fails. If MINT learns from these rollouts, it demonstrates exactly the core claim of the paper.

**Why the previous AI treated handle semantics as secondary:**  
The previous reasoning was "AnyGrasp succeeds despite poor metadata, so metadata is not the primary blocker." This is correct that AnyGrasp compensates for poor metadata at the grasp detection level. It is wrong that the metadata issue has no downstream effect: the downstream effect is `pre_attach_motion > 0.12`, which causes `strong_rollout_criteria` to reject the rollout, which causes D2 to fail at data intake. The u3 fix is not about making AnyGrasp work — AnyGrasp already works. It is about making the resulting trajectory quality high enough for D2.

## 2. GPT Suggestions — Critical Academic Review

### Suggestion A: Gate between Phase 3 and D1 on same-seed coherence
> "If strong_rollout_count >= 2 but strong_coherent_seeds is still empty, do not enter D1; check same-seed coherence first."

**Verdict: ACCEPT — and strengthen it.**  
This is correct. The current code already has this logic in `run_d1_single_rollout_overfit.py` (it checks `strong_coherent_seeds` first), but it silently falls back to `_ordered_mainline_candidates()` when coherent seeds are empty. This fallback is the source of the D1→D2 incoherence bug. The gate should be:
- `strong_rollout_count >= 2` AND `strong_coherent_seeds non-empty` → proceed to D1
- `strong_rollout_count >= 2` but `strong_coherent_seeds == []` → STOP, diagnose why rollouts from the same seed don't meet coherence threshold before running D1
- `strong_rollout_count < 2` → STOP, fix data (Phase 1+2)

This is not just a monitoring gate — it must be enforced in `run_d1_single_rollout_overfit.py` by removing the `_ordered_mainline_candidates()` fallback when `strong_coherent_seeds` is empty.

### Suggestion B: Hard stop-loss in D1/D2: extended → phase_balanced, then stop
> "D1 only runs extended → phase_balanced. D2 only runs extended → phase_balanced. Two steps with no new signal → stop. No automatic escalation to more variants."

**Verdict: ACCEPT with one modification.**  
The stop-loss logic is scientifically sound: if neither `extended_overfit` nor `phase_balanced_overfit` shows attach signal, the issue is almost certainly data quality, not variant selection. Running more variants is compute waste and creates more contamination surface. The modification: the stop condition should be `no attach signal in EITHER variant` (i.e., `_has_attach_signal() == False` for both), not just `trend_passed == False`. `trend_passed` can be False even with attach signal (e.g., grasp_gain > 0 but success_gain = 0). The stop-loss triggers on zero attach signal, not on imperfect trend.

Concrete implementation:
```python
# In run_d1_single_rollout_overfit.py, after Stage A:
if not any(_has_attach_signal(attempt.get('evaluation',{})) for attempt in [stage_a_attempt]):
    # No attach signal in extended_overfit — data quality issue, not variant issue
    # Do NOT escalate to phase_balanced or attach_curriculum
    # Fail fast with specific error: 'no_attach_signal_stop_loss'
    ...
# In run_d2_single_seed_overfit.py, after first variant:
if not any(bool(r.get('grasp_success')) for r in finetuned_records):
    # No attach signal — fail fast, do not run phase_balanced
    ...
```

---

## 3. Execution Plan — 8 Phases with Hard Gates

### Phase 0: Engineering Cleanup (NOW — 15 min)

```bash
cd /mnt/afs2/zhuhaowu/infinigen

# 0.1 Complete diagnostic_runs archive (mv timed out — use background copy)
mkdir -p experiments/mint/mint_drawer_v1/history/2026-03-26_diagnostic_runs
cp -r experiments/mint/mint_drawer_v1/artifacts/diagnostic_runs/2026-03-26_gate_drift_reset \
   experiments/mint/mint_drawer_v1/history/2026-03-26_diagnostic_runs/ && \
rm -rf experiments/mint/mint_drawer_v1/artifacts/diagnostic_runs/2026-03-26_gate_drift_reset
echo 'diagnostic archive: done'

# 0.2 Clear stale d1_candidate_progress (still points to archived path)
rm -f experiments/mint/mint_drawer_v1/artifacts/d1_candidate_progress.json

# 0.3 Reset c2/d1/d2 queue state to pending via Python
source /root/anaconda3/etc/profile.d/conda.sh && conda activate infinigen
python -c "
import json, pathlib, time
state_path = pathlib.Path('experiments/mint/mint_drawer_v1/state.json')
state = json.loads(state_path.read_text())
for step in state['queue']:
    if step['id'] in ['c2_action_contract_repair','d1_single_rollout_overfit','d2_single_seed_overfit']:
        step['status'] = 'pending'
        step['updated_at'] = '2026-03-27T12:00:00+08:00'
        step.pop('last_error', None)
state['verdict'] = 'rebuild_in_progress'
state['active_job'] = None
state['updated_at'] = '2026-03-27T12:00:00+08:00'
state_path.write_text(json.dumps(state, indent=2))
print('state.json reset OK')
"

# 0.4 Update watch_status.json
cat > experiments/mint/mint_drawer_v1/runtime/watch_status.json << 'EOF'
{
  "updated_at": "2026-03-27T12:00:00+08:00",
  "verdict": "rebuild_in_progress",
  "active_step": "c2_action_contract_repair",
  "active_step_status": "pending",
  "controller_id": null,
  "worker_pid": null,
  "note": "Clean rebuild under RESOLUTION_PLAN v2. Stale lease released. diagnostic_runs archived.",
  "next_action": "Phase 1: fix seed_010 handle patch, then Phase 2: re-run C2"
}
EOF
```

**Phase 0 verification:**
```bash
# Must ALL pass:
ls experiments/mint/mint_drawer_v1/artifacts/diagnostic_runs/ | wc -l  # expect 0 or dir empty
ls experiments/mint/mint_drawer_v1/artifacts/d1_candidate_progress.json 2>/dev/null && echo FAIL || echo OK
python -c "import json; s=json.load(open('experiments/mint/mint_drawer_v1/state.json')); [print(x['id'],x['status']) for x in s['queue'] if x['id'] in ['c2_action_contract_repair','d1_single_rollout_overfit']]"
# Expect: c2_action_contract_repair pending, d1_single_rollout_overfit pending
```

### Phase 1: Fix Seed_010 Handle Patch (30 min)

The existing patch derived `handle_center_local` for seed_010 from ep04 which itself had `pre_attach_motion=0.1215` — circular reference. Use the c2 rollout with the LOWEST `pre_attach_drawer_motion` among seed_010 episodes as the source:

```bash
source /root/anaconda3/etc/profile.d/conda.sh && conda activate infinigen
cd /mnt/afs2/zhuhaowu/infinigen
python -c "
import json, pathlib
c2_dir = pathlib.Path('experiments/mint/mint_drawer_v1/artifacts/c2_replay_valid_rollouts')
best, best_motion = None, float('inf')
for jp in sorted(c2_dir.glob('seed_010_*.json')):
    m = json.loads(jp.read_text())
    motion = float(m.get('pre_attach_drawer_motion', 9.0))
    print(f'  ep{m.get(\"episode_index\")}: pre_attach={motion:.4f} grasp_steps={m.get(\"grasp_steps\")}')
    if motion < best_motion:
        best_motion = motion
        best = m
if best:
    print(f'Best: ep{best[\"episode_index\"]} motion={best_motion:.4f}')
    print(f'handle_center_local: {best.get(\"handle_center_local\")}')
"
```

Then write the corrected center to metadata:
```bash
python -c "
import json, pathlib
# Replace [x,y,z] with value from best rollout above
NEW_CENTER = [x, y, z]  # <-- fill from step above
meta_path = pathlib.Path('/mnt/afs2/zhuhaowu/infinigen/sim_exports/urdf/drawerbox/10/metadata.json')
meta = json.loads(meta_path.read_text())
print('Old handle_center_local:', meta.get('handle_center_local'))
meta['handle_center_local'] = NEW_CENTER
meta_path.write_text(json.dumps(meta, indent=2))
print('New handle_center_local:', meta.get('handle_center_local'))
"
```

**Phase 1 gate:** confirm seed_010 metadata changed from `[-0.122, -0.070, 0.032]`.

### Phase 2: Re-run C2 Rollout Collection (2-3 h GPU)

**Hard pre-run checklist — do not skip any item:**
```bash
# i. No competing Python processes
ps aux | grep python | grep -v grep | grep -v vscode | grep -v code-server
# ii. GPU free memory
nvidia-smi | grep MiB
# iii. Seed_010 metadata updated
python -c "import json; m=json.load(open('/mnt/afs2/zhuhaowu/infinigen/sim_exports/urdf/drawerbox/10/metadata.json')); print(m.get('handle_center_local'))"
# iv. diagnostic_runs gate_drift_reset gone
ls experiments/mint/mint_drawer_v1/artifacts/diagnostic_runs/
```

```bash
export CTL="resolution_plan_v2:$(date +%Y%m%dT%H%M%S)"
export RUN="resolution_plan_v2:$(date +%Y%m%dT%H%M%S)"
screen -dmS mint_c2_rebuild bash -c "
  source /root/anaconda3/etc/profile.d/conda.sh && conda activate infinigen
  cd /mnt/afs2/zhuhaowu/infinigen
  export MINT_CONTROLLER_ID='$CTL'
  export MINT_RUN_ID='$RUN'
  python scripts/mint/run_c2_action_contract_repair.py 2>&1 | tee experiments/mint/mint_drawer_v1/runtime/c2_rebuild.log
"
echo 'C2 launched in screen mint_c2_rebuild'
```

**Phase 2 hard gate (MUST pass — stop if fails):**
```bash
python -c "
import json
audit = json.load(open('experiments/mint/mint_drawer_v1/artifacts/strong_rollout_audit.json'))
count = audit.get('strong_rollout_count', 0)
coherent = audit.get('strong_coherent_seeds', [])
print(f'strong_rollout_count={count} (need>=2)')
print(f'strong_coherent_seeds={coherent} (need non-empty)')
for row in audit.get('strong_rollout_order', []):
    print(f\"  seed={row['seed']} ep={row['episode_index']} grasp={row.get('grasp_steps')} pre_attach={row.get('pre_attach_drawer_motion',0):.4f}\")
"
# IF strong_rollout_count < 2: go to Phase 1 Fallback (read g3_anygrasp_grasps/seed_010.json raw pose)
# IF strong_rollout_count >= 2 but strong_coherent_seeds == []: STOP — diagnose coherence issue
# IF strong_rollout_count >= 2 AND strong_coherent_seeds non-empty: proceed to Phase 3
```

### Phase 2.5: Held-Out Seed Solvability Check (1 h — run in parallel with Phase 3 setup)

U5 shows seeds 11-15 have 0 successes across ALL modalities. E1 will be meaningless if the held-out seeds are physically unsolvable. This must be diagnosed before E1:

```bash
source /root/anaconda3/etc/profile.d/conda.sh && conda activate infinigen
cd /mnt/afs2/zhuhaowu/infinigen
# Run a quick oracle probe on 2 held-out seeds (11, 12) to check physical solvability
# If oracle succeeds on 11-15 with the u3 patch applied: E1 is valid
# If oracle still fails: apply u3 patch to held-out seeds first
python -c "
import json
b1 = json.load(open('experiments/mint/mint_drawer_v1/artifacts/b1_oracle_scripted_baseline.json'))
for r in b1.get('oracle', {}).get('records', []):
    print(f\"seed_{r['seed']}: {r.get('successful_attempts',0)}/{r.get('attempt_count',0)} oracle successes\")
"
# Check if held-out seeds 11-15 are in B1 records
# If not: held-out seeds were never probed by B1 (B1 only covers train seeds 1-10)
# → Need to run a held-out oracle probe explicitly before E1
```

**Held-out seed decision:**
- If oracle probe on 11-15 succeeds: E1 is physically valid, proceed
- If oracle probe on 11-15 fails: must apply u3 patch to seeds 11-15 and re-probe
- If oracle fails even after patch: the claim must be narrowed (held-out seeds are not physically solvable by any method)

### Phase 3: Apply D1 Oracle Filter Fix (30 min — code change)

Add `pre_attach_motion ≤ 0.12` hard filter and remove the silent fallback in D1:

```bash
grep -n 'mainline_candidate_order\|_ordered_mainline\|strong_coherent' \
  /mnt/afs2/zhuhaowu/infinigen/scripts/mint/strict_teacher_dataset.py | head -20
```

In `strict_teacher_dataset.py` mainline candidate ordering function:
```python
PRE_ATTACH_LIMIT = 0.12  # must match strong_rollout_criteria.pre_attach_drawer_motion_lte
candidates = [
    c for c in candidates
    if float(c.get('pre_attach_drawer_motion', 0.0)) <= PRE_ATTACH_LIMIT
]
```

In `run_d1_single_rollout_overfit.py`, enforce hard stop when `strong_coherent_seeds` is empty:
```python
# Replace the current fallback logic:
if not candidates or not strong_coherent_seeds:
    # HARD STOP — do not fall back to _ordered_mainline_candidates()
    result = {"gate": "d1_single_rollout_overfit", "passed": False,
              "error": "strong_coherent_seeds is empty — data quality floor not met",
              "failure_reason": "data_quality_floor_not_met"}
    ...
    return False
```

Also add stop-loss for zero attach signal (GPT suggestion B, accepted):
```python
# After Stage A extended_overfit:
if not _has_attach_signal(stage_a_attempt.get('evaluation', {})):
    # Zero attach signal — data issue, not variant issue
    # Do NOT escalate to phase_balanced or attach_curriculum
    result["failure_reason"] = "no_attach_signal_stop_loss"
    return False
```

**Phase 3 verification:**
```bash
source /root/anaconda3/etc/profile.d/conda.sh && conda activate infinigen
cd /mnt/afs2/zhuhaowu/infinigen
python -c "
from strict_teacher_dataset import rebuild_strict_teacher_dataset
audit = rebuild_strict_teacher_dataset()
for c in audit.get('mainline_candidate_order', []):
    pre = float(c.get('pre_attach_drawer_motion', 0.0))
    status = 'OK' if pre <= 0.12 else 'BLOCKED'
    print(f\"rank={c.get('rank')} seed={c.get('seed')} ep={c.get('episode_index')} pre_attach={pre:.4f} {status}\")
"
# ALL candidates must show OK
```

### Phase 4: Re-run D1 Under Clean Single Controller (1-2 h GPU)

```bash
export CTL="resolution_plan_v2:d1:$(date +%Y%m%dT%H%M%S)"
export RUN="resolution_plan_v2:d1:$(date +%Y%m%dT%H%M%S)"
screen -dmS mint_d1_clean bash -c "
  source /root/anaconda3/etc/profile.d/conda.sh && conda activate infinigen
  cd /mnt/afs2/zhuhaowu/infinigen
  export MINT_CONTROLLER_ID='$CTL'
  export MINT_RUN_ID='$RUN'
  python scripts/mint/run_d1_single_rollout_overfit.py 2>&1 | tee experiments/mint/mint_drawer_v1/runtime/d1_clean.log
"
```

**Phase 4 hard gate:**
```bash
python -c "
import json
d1 = json.load(open('experiments/mint/mint_drawer_v1/artifacts/d1_single_rollout_overfit.json'))
print('D1 passed:', d1.get('passed'))
promoted = d1.get('promoted_candidate')
if promoted:
    print('seed:', promoted.get('selected_seed'), 'role:', promoted.get('candidate_role'))
    ev = promoted.get('winning_attempt',{}).get('evaluation',{})
    ft = ev.get('summary',{}).get('finetuned_mint',{})
    pt = ev.get('summary',{}).get('pretrained_mint',{})
    print(f\"successes ft/pt: {ft.get('successes')}/{pt.get('successes')}\")
else:
    print('failure_reason:', d1.get('failure_reason'))
"
# SUCCESS: passed=True, candidate_role=mainline, seed in [2,10]
# STOP if failure_reason=data_quality_floor_not_met → re-check Phase 2 gate
# STOP if failure_reason=no_attach_signal_stop_loss → data still insufficient → fix upstream
```

### Phase 5: Re-run D2 (1-2 h GPU)

```bash
screen -dmS mint_d2_clean bash -c "
  source /root/anaconda3/etc/profile.d/conda.sh && conda activate infinigen
  cd /mnt/afs2/zhuhaowu/infinigen
  export MINT_CONTROLLER_ID='$CTL'
  export MINT_RUN_ID='$RUN'
  python scripts/mint/run_d2_single_seed_overfit.py 2>&1 | tee experiments/mint/mint_drawer_v1/runtime/d2_clean.log
"
```

**Phase 5 hard gate:**
```bash
python -c "
import json
d2 = json.load(open('experiments/mint/mint_drawer_v1/artifacts/d2_single_seed_overfit.json'))
print('D2 passed:', d2.get('passed'))
w = d2.get('winner') or {}
ev = w.get('evaluation') or {}
ft = ev.get('summary',{}).get('finetuned_mint',{})
pt = ev.get('summary',{}).get('pretrained_mint',{})
print(f\"success_gain={ft.get('success_rate',0)-pt.get('success_rate',0):.3f} ft_successes={ft.get('successes')}\")
if not d2.get('passed'):
    print('failure_reason:', d2.get('failure_reason'))
"
# SUCCESS: passed=True, success_gain >= 0.2, finetuned_successes >= 3
```

### Phase 6: D3 Train Seed Probe (2-3 h GPU — only after D2 hard-passes)

```bash
screen -dmS mint_d3 bash -c "
  source /root/anaconda3/etc/profile.d/conda.sh && conda activate infinigen
  cd /mnt/afs2/zhuhaowu/infinigen
  export MINT_CONTROLLER_ID='$CTL'
  export MINT_RUN_ID='$RUN'
  python scripts/mint/run_d3_train_seed_probe.py 2>&1 | tee experiments/mint/mint_drawer_v1/runtime/d3_clean.log
"
```

D3 floor (from `acceptance_criteria.json`): ≥4 seeds, ≥8 rollouts, ≥300 frames, improvement not single-seed-dominated.

### Phase 7: E1 Held-Out Evaluation (1-2 h GPU — only after D3 hard-passes AND Phase 2.5 confirms held-out solvability)

```bash
screen -dmS mint_e1 bash -c "
  source /root/anaconda3/etc/profile.d/conda.sh && conda activate infinigen
  cd /mnt/afs2/zhuhaowu/infinigen
  export MINT_CONTROLLER_ID='$CTL'
  export MINT_RUN_ID='$RUN'
  python scripts/mint/run_e1_heldout_eval.py 2>&1 | tee experiments/mint/mint_drawer_v1/runtime/e1_clean.log
"
```

---

## 4. Decision Tree

```
Phase 2 gate: strong_rollout_count >= 2 AND strong_coherent_seeds non-empty?
  NO (count < 2)          → fix Phase 1 using g3_anygrasp_grasps/seed_010.json raw pose
  NO (count>=2 but no coherence) → STOP, diagnose same-seed phase distance before D1
  YES                     → Phase 3 code fix → Phase 4 D1

Phase 4 gate: D1 passed AND candidate_role=mainline?
  NO: failure_reason=data_quality_floor_not_met  → recheck Phase 2
  NO: failure_reason=no_attach_signal_stop_loss  → upstream data still bad, re-examine C2 rollout quality
  NO: failure_reason=teacher_mainline_failed      → increase D1 steps 1200→2400 (ONLY this change)
  YES                     → Phase 5 D2

Phase 5 gate: D2 passed?
  NO: failure_reason=data_quality_floor_not_met  → strong_coherent_seeds emptied again → recheck Phase 2
  NO: success_gain < 0.2 → try phase_balanced_overfit with more steps
  YES                     → Phase 6 D3

Phase 2.5 gate: held-out oracle probe passes on ≥3 of seeds 11-15?
  NO → apply u3 patch to seeds 11-15 and re-probe BEFORE running E1
  YES → E1 is valid
```

---

## 5. Absolute Constraints

1. **MINT model architecture must not change** until D2 passes on clean data. No exceptions.
2. **One controller at a time.** Always check `ps aux | grep python` before launching any screen job.
3. **All D1/D2 results must carry the same `MINT_CONTROLLER_ID`.** Mixed controller IDs invalidate comparison.
4. **Rollout identity is SHA256, not seed/episode name.** The two `seed_010_episode_04.npz` files are different rollouts.
5. **After each phase, sync:** `state.json`, `campaign_status.md`, `watch_status.json`, `review.json`, `findings.md`, `progress.md`.

---

## 6. Why the Claim Is Still Achievable

1. AnyGrasp already achieves 3/6 successes on seeds 2 and 10 — the task is physically solved.
2. The `pre_attach_motion=0.1215` failure is a 0.001 precision gap resolvable by a better handle_center patch.
3. Seeds 7 and 9 both have AnyGrasp successes with strong phase structure — these are additional D3 candidates.
4. The oracle baseline failure on seeds 2,10 is a scientific asset: it demonstrates that AnyGrasp adds value over scripted oracles for these geometries.
5. The held-out seeds (11-15) are the only genuine unknown — they must be probed before E1.