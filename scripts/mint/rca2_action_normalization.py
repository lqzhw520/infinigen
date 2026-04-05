#!/usr/bin/env python3
"""
RCA2: Action Normalization Audit
==================================
Verifies that MINT action output format is compatible with DrawerRobotEnv.step() input.

SCOPE GUARDRAIL:
  This audit applies ONLY to the 7 overfit training episodes
  (LeRobot episode indices 0-5, 49) and the current env/inference pipeline.
  Do NOT extrapolate conclusions to the full V59 dataset (19,701 frames).

Key questions:
  1. Dataset action ranges: are they within [-1, 1] for pos/rot dims?
  2. Gripper convention: ±1.0 (env expects) vs 0/1 (dataset might use)?
  3. Action contract: translation_scale/rotation_scale consistency
  4. Teacher action replay: do recorded actions actually open the drawer in simulation?
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
sys.path.insert(0, str(PROJECT_ROOT))

CAMPAIGN_DIR = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1"
ARTIFACT_DIR = CAMPAIGN_DIR / "artifacts"
OUTPUT_FILE = ARTIFACT_DIR / "rca2_action_normalization.json"

# Overfit subset episode indices (from E023 — LeRobot dataset indices)
OVERFIT_EPISODES = [0, 1, 2, 3, 4, 5, 49]

# Env constants from drawer_robot_env.py step() and mint_common.py
TRANSLATION_SCALE_M = 0.03    # action[:3] * 0.03 = world-frame delta position (m)
ROTATION_SCALE_RAD = 0.25     # action[3:6] * 0.25 = world-frame delta rotation (rad)
GRIPPER_OPEN = 1.0           # action[6] >= 0 → open
GRIPPER_CLOSE = -1.0          # action[6] < 0 → close
ACTION_DIM = 7


def load_dataset():
    """Load V59 LeRobot dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    print("Loading V59 LeRobot dataset...")
    ds = LeRobotDataset(
        repo_id="infinigen_drawer_robot_v1",
        root=str(CAMPAIGN_DIR / "dataset"),
        revision="main",
    )
    print(f"  Dataset loaded: {len(ds)} frames")
    return ds


def build_episode_boundaries(ds):
    """Build episode_index → (start_frame, end_frame) mapping."""
    boundaries = {}
    current_ep = None
    start_frame = 0
    for i in range(len(ds)):
        item = ds[i]
        ep = int(item["episode_index"].item())
        if ep != current_ep:
            if current_ep is not None:
                boundaries[current_ep] = (start_frame, i)
            current_ep = ep
            start_frame = i
    if current_ep is not None:
        boundaries[current_ep] = (start_frame, len(ds))
    return boundaries


def compute_dataset_action_stats(ds, episode_indices):
    """Compute per-frame action statistics for specified episodes."""
    print(f"\n[1] Computing dataset action stats for episodes: {episode_indices}")
    boundaries = build_episode_boundaries(ds)

    all_actions = []
    episode_info = []

    for ep_idx in episode_indices:
        if ep_idx not in boundaries:
            print(f"  WARNING: Episode {ep_idx} not found in dataset")
            continue
        start, end = boundaries[ep_idx]
        ep_actions = []
        for i in range(start, end):
            item = ds[i]
            action = item["action"].numpy()
            ep_actions.append(action)
        if not ep_actions:
            continue
        ep_actions = np.array(ep_actions)
        all_actions.append(ep_actions)

        ep_mean = ep_actions.mean(axis=0)
        ep_std = ep_actions.std(axis=0)
        ep_min = ep_actions.min(axis=0)
        ep_max = ep_actions.max(axis=0)
        unique_gripper = np.unique(ep_actions[:, 6])

        episode_info.append({
            "episode_index": int(ep_idx),
            "n_frames": len(ep_actions),
            "action_mean": ep_mean.tolist(),
            "action_std": ep_std.tolist(),
            "action_min": ep_min.tolist(),
            "action_max": ep_max.tolist(),
            "unique_gripper_values": sorted([float(v) for v in unique_gripper]),
        })
        print(f"  Ep {ep_idx}: {len(ep_actions)} frames, "
              f"pos_range=[{ep_min[:3].min():.3f}, {ep_max[:3].max():.3f}], "
              f"rot_range=[{ep_min[3:6].min():.3f}, {ep_max[3:6].max():.3f}], "
              f"gripper_vals={sorted([float(v) for v in unique_gripper])}")

    if not all_actions:
        return None, episode_info

    all_actions = np.concatenate(all_actions, axis=0)
    stats = {
        "n_frames_total": int(len(all_actions)),
        "n_episodes": len(episode_info),
        "mean": all_actions.mean(axis=0).tolist(),
        "std": all_actions.std(axis=0).tolist(),
        "min": all_actions.min(axis=0).tolist(),
        "max": all_actions.max(axis=0).tolist(),
        "p01": np.percentile(all_actions, 1, axis=0).tolist(),
        "p05": np.percentile(all_actions, 5, axis=0).tolist(),
        "p95": np.percentile(all_actions, 95, axis=0).tolist(),
        "p99": np.percentile(all_actions, 99, axis=0).tolist(),
    }

    dim_names = ["dx", "dy", "dz", "drot_x", "drot_y", "drot_z", "gripper"]
    print(f"\n  Combined stats over {stats['n_frames_total']} frames:")
    for i, name in enumerate(dim_names):
        print(f"    {name}: mean={stats['mean'][i]:+.4f}, std={stats['std'][i]:.4f}, "
              f"min={stats['min'][i]:+.4f}, max={stats['max'][i]:+.4f}, "
              f"p01={stats['p01'][i]:+.4f}, p99={stats['p99'][i]:+.4f}")

    return stats, episode_info


def check_gripper_convention(episode_info):
    """Verify gripper convention: env expects >=0→open, <0→close."""
    print("\n[2] Checking gripper action convention...")
    all_gripper_values = set()
    for ep in episode_info:
        all_gripper_values.update(ep["unique_gripper_values"])

    sorted_vals = sorted(all_gripper_values)
    print(f"  All unique gripper values: {sorted_vals}")

    is_binary_01 = all_gripper_values <= {0.0, 1.0}
    is_binary_pm1 = all_gripper_values <= {GRIPPER_OPEN, GRIPPER_CLOSE}
    is_in_env_range = all(abs(v) <= 1.0 for v in all_gripper_values)

    # Check: does 0 appear without ±1?
    has_zero_only = all_gripper_values == {0.0}
    has_zero_with_close = all_gripper_values == {0.0, GRIPPER_CLOSE}
    has_env_convention = all_gripper_values <= {GRIPPER_OPEN, GRIPPER_CLOSE}

    findings = {
        "all_unique_values": sorted_vals,
        "is_binary_01": bool(is_binary_01),
        "is_binary_pm1": bool(is_binary_pm1),
        "is_in_env_range": bool(is_in_env_range),
        "has_zero_only": bool(has_zero_only),
        "has_zero_with_close": bool(has_zero_with_close),
        "has_env_convention": bool(has_env_convention),
    }

    # Diagnose mismatch
    if is_binary_01 and not is_binary_pm1:
        findings["verdict"] = "MISMATCH"
        findings["severity"] = "CRITICAL"
        findings["issue"] = (
            f"Dataset uses 0/1 gripper but env expects >={GRIPPER_OPEN}/<{GRIPPER_OPEN}. "
            f"0 → gripper_open=0.0 (CLOSED in env). "
            f"1 → gripper_open=1.0 (OPEN in env). "
            f"If teacher used 0 (closed) for approach phase, env interprets as gripper closed. "
            f"But if gripper is at 0 while reaching for handle, that might be correct. "
            f"NEED teacher replay to verify if actions actually open the drawer."
        )
    elif has_zero_with_close:
        findings["verdict"] = "POTENTIAL_MISMATCH"
        findings["severity"] = "MEDIUM"
        findings["issue"] = (
            "Dataset uses {0, -1} for gripper. Env interprets 0 as CLOSED, -1 as CLOSED. "
            "If '0' was meant to mean 'maintain current grip' vs '-1' meaning 'actively close', "
            "the env collapses both to gripper_open=0.0. This is a semantic ambiguity."
        )
    elif has_env_convention:
        findings["verdict"] = "OK"
        findings["severity"] = "OK"
        findings["issue"] = None
    else:
        findings["verdict"] = "AMBIGUOUS"
        findings["severity"] = "UNKNOWN"
        findings["issue"] = f"Unexpected gripper values: {sorted_vals}"

    print(f"  Verdict: {findings['verdict']} ({findings['severity']})")
    if findings["issue"]:
        print(f"  Issue: {findings['issue']}")
    return findings


def check_action_range(stats, episode_info):
    """Check if any actions exceed [-1, 1] clipping range."""
    print("\n[3] Checking action range for clipping impact...")
    dim_names = ["dx", "dy", "dz", "drot_x", "drot_y", "drot_z", "gripper"]
    outliers = []

    for i, name in enumerate(dim_names):
        n_above_1 = sum(1 for ep in episode_info if ep["action_max"][i] > 1.0)
        n_below_m1 = sum(1 for ep in episode_info if ep["action_min"][i] < -1.0)
        if n_above_1 > 0 or n_below_m1 > 0:
            outliers.append({
                "dim": name,
                "dim_index": i,
                "global_min": stats["min"][i],
                "global_max": stats["max"][i],
                "p01": stats["p01"][i],
                "p99": stats["p99"][i],
                "episodes_above_1": n_above_1,
                "episodes_below_m1": n_below_m1,
                "clipped_away": (
                    f"[{stats['min'][i]:+.4f}, {stats['max'][i]:+.4f}] "
                    f"clipped to [-1.0, 1.0]"
                ),
            })

    if outliers:
        print(f"  FOUND {len(outliers)} dimensions with out-of-range values:")
        for o in outliers:
            print(f"    {o['dim']}: {o['clipped_away']} "
                  f"({o['episodes_above_1']} eps above 1, {o['episodes_below_m1']} eps below -1)")
    else:
        print("  All action dimensions within [-1, 1] — no clipping distortion.")

    return outliers


def check_action_contract():
    """Load active action contract and compare to env defaults."""
    print("\n[4] Checking active action contract vs env defaults...")
    contract_path = ARTIFACT_DIR / "active_action_contract.json"
    if contract_path.exists():
        with open(contract_path) as f:
            contract = json.load(f)
        print(f"  Active contract: {json.dumps(contract, indent=2)}")
    else:
        contract = None
        print("  No active_action_contract.json — using env defaults only")

    defaults = {
        "translation_scale_m": TRANSLATION_SCALE_M,
        "rotation_scale_rad": ROTATION_SCALE_RAD,
        "gripper_open_value": GRIPPER_OPEN,
        "gripper_close_value": GRIPPER_CLOSE,
        "action_frame": "world",
        "action_dim": ACTION_DIM,
    }
    print(f"  Env defaults: {json.dumps(defaults, indent=2)}")

    # Check contract vs defaults consistency
    checks = {}
    if contract:
        checks["translation_scale_matches"] = (
            contract.get("translation_scale_m") == TRANSLATION_SCALE_M
        )
        checks["rotation_scale_matches"] = (
            contract.get("rotation_scale_rad") == ROTATION_SCALE_RAD
        )
        checks["action_frame_matches"] = (
            contract.get("action_frame") == "world"
        )
        print(f"  Contract vs defaults: {json.dumps(checks, indent=2)}")

    return contract, defaults, checks


def run_teacher_action_replay(episode_info, stats):
    """
    KEY TEST: Replay recorded actions in DrawerRobotEnv.
    Does the teacher actually open the drawer when actions are replayed?

    This is the most definitive test: if teacher actions don't open the drawer
    in simulation, there's a deeper action contract issue.
    """
    print("\n[5] Teacher Action Replay — running in DrawerRobotEnv...")

    # Import inside to avoid loading PyBullet if not needed
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "mint"))
        from drawer_robot_env import DrawerRobotEnv
    except ImportError as e:
        print(f"  FAILED to import DrawerRobotEnv: {e}")
        return None

    results = []

    # Map LeRobot episode index → drawer_robot_env seed
    # Overfit episodes 0-5 are from seed 7, episode 49 is from seed 9
    ep_seed_map = {i: 7 for i in range(6)}
    ep_seed_map[49] = 9

    # Test on 2 episodes from the overfit subset (not all 7 to save time)
    test_episodes = [episode_info[0], episode_info[-1]]  # first and last

    for ep_info in test_episodes:
        ep_idx = ep_info["episode_index"]
        seed = ep_seed_map.get(ep_idx, 7)
        print(f"\n  Replaying ep {ep_idx} ({ep_info['n_frames']} frames, seed={seed})...")

        # Load episode actions from dataset
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            ds = LeRobotDataset(
                repo_id="infinigen_drawer_robot_v1",
                root=str(CAMPAIGN_DIR / "dataset"),
            )
            boundaries = build_episode_boundaries(ds)
            if ep_idx not in boundaries:
                print(f"  SKIP: ep {ep_idx} not in dataset boundaries")
                continue
            start, end = boundaries[ep_idx]

            # Collect actions for this episode
            actions = []
            for i in range(start, end):
                item = ds[i]
                actions.append(item["action"].numpy())
            actions = np.array(actions)

            print(f"    Actions shape: {actions.shape}, "
                  f"gripper vals: {sorted(set(round(float(a[6]), 1) for a in actions))}")

        except Exception as e:
            print(f"  FAILED to load episode: {e}")
            import traceback; traceback.print_exc()
            continue

        # Run env simulation with teacher actions
        try:
            # Use the active action contract for correct scales
            active_contract, _, _ = check_action_contract()
            contract = active_contract or {}
            env = DrawerRobotEnv(
                seed=seed,
                max_steps=len(actions) + 20,
                action_contract=contract if contract else None,
            )
            env.reset()

            step_results = []
            for step_i, action in enumerate(actions):
                obs, reward, done, info = env.step(action)
                step_results.append({
                    "step": step_i,
                    "action": action.tolist(),
                    "drawer_fraction": float(obs.drawer_fraction),
                    "attached": bool(env.attached),
                    "ever_attached": bool(env.ever_attached),
                    "gripper_open": float(obs.gripper_open),
                })
                if done:
                    break

            final_drawer = float(obs.drawer_fraction)
            ever_attached = bool(env.ever_attached)
            max_drawer = max(s["drawer_fraction"] for s in step_results)
            success = bool(info.get("is_success", False))

            result = {
                "episode_index": ep_idx,
                "seed": seed,
                "n_steps": len(step_results),
                "final_drawer_fraction": final_drawer,
                "max_drawer_fraction": max_drawer,
                "ever_attached": ever_attached,
                "strict_success": success,
                "gripper_vals_used": sorted(set(round(float(a[6]), 1) for a in actions)),
                "n_close_steps": int(sum(1 for a in actions if float(a[6]) < 0)),
                "n_open_steps": int(sum(1 for a in actions if float(a[6]) >= 0)),
            }
            results.append(result)

            print(f"    FINAL drawer_fraction: {final_drawer:.3f}")
            print(f"    MAX drawer_fraction:    {max_drawer:.3f}")
            print(f"    ever_attached:          {ever_attached}")
            print(f"    strict_success:         {success}")
            print(f"    gripper close steps:    {result['n_close_steps']}/{len(actions)}")
            print(f"    gripper open steps:     {result['n_open_steps']}/{len(actions)}")

        except Exception as e:
            print(f"  FAILED to run env replay: {e}")
            import traceback; traceback.print_exc()
            results.append({
                "episode_index": ep_idx,
                "seed": seed,
                "error": str(e),
            })

    return results


def check_mint_inference_action_format():
    """Check what actions MINT actually outputs during inference."""
    print("\n[6] Checking MINT action output format from existing eval artifacts...")

    eval_artifact = ARTIFACT_DIR / "v59_overfit_eval.json"
    if eval_artifact.exists():
        with open(eval_artifact) as f:
            data = json.load(f)
        print(f"  V59 overfit eval artifact found:")
        print(f"    pretrained_rmse: {data.get('pretrained_rmse')}")
        print(f"    finetuned_rmse: {data.get('finetuned_rmse')}")
        print(f"    pretrained_gripper_mae: {data.get('pretrained_gripper_mae')}")
        print(f"    finetuned_gripper_mae: {data.get('finetuned_gripper_mae')}")
    else:
        data = None
        print("  V59 overfit eval artifact not found")

    gate_artifact = ARTIFACT_DIR / "v59_overfit_env_gate.json"
    if gate_artifact.exists():
        with open(gate_artifact) as f:
            gate_data = json.load(f)
        print(f"\n  V59 env gate artifact found:")
        for pol in ["pretrained", "finetuned"]:
            if pol in gate_data:
                print(f"    {pol}: non_zero_action_ratio={gate_data[pol].get('non_zero_action_ratio')}, "
                      f"total_eef_motion={gate_data[pol].get('total_eef_motion')}")
                per_rollouts = gate_data.get("per_rollout", {}).get(pol, [])
                if per_rollouts:
                    sample = per_rollouts[0]
                    print(f"    Sample rollout keys: {list(sample.keys())}")
    else:
        gate_data = None
        print("  V59 env gate artifact not found")

    return data, gate_data


def synthesize_verdict(stats, episode_info, gripper_findings, range_outliers,
                       contract_info, replay_results, mint_data):
    """Synthesize all findings into a structured verdict."""
    print("\n[7] Synthesizing RCA2 verdict...")

    contract, defaults, checks = contract_info

    # Key questions to answer:
    # Q1: Are actions within [-1, 1]?
    q1_out_of_range = len(range_outliers) > 0

    # Q2: Gripper convention match?
    q2_gripper_mismatch = gripper_findings["verdict"] in ("MISMATCH", "POTENTIAL_MISMATCH")

    # Q3: Action contract scales consistent?
    q3_contract_ok = (
        contract is None or
        (checks.get("translation_scale_matches", True) and
         checks.get("rotation_scale_matches", True))
    )

    # Q4: Teacher replay — did actions open the drawer?
    if replay_results:
        replay_successes = [r for r in replay_results if "error" not in r]
        if replay_successes:
            any_ever_attached = any(r.get("ever_attached", False) for r in replay_successes)
            any_drawer_opened = any(r.get("max_drawer_fraction", 0) > 0.01 for r in replay_successes)
            any_strict_success = any(r.get("strict_success", False) for r in replay_successes)
        else:
            any_ever_attached = None
            any_drawer_opened = None
            any_strict_success = None
    else:
        any_ever_attached = None
        any_drawer_opened = None
        any_strict_success = None

    # RCA2 verdict logic
    if any_strict_success is False:
        # Teacher replay failed → action contract issue
        if q2_gripper_mismatch:
            root_cause = "GRIPPER_CONVENTION_MISMATCH"
            explanation = (
                "Teacher actions replayed in env do NOT open the drawer. "
                f"Dataset gripper={gripper_findings['all_unique_values']} but env expects >={GRIPPER_OPEN}/<{GRIPPER_OPEN}. "
                "Actions may be using wrong gripper polarity, preventing attachment."
            )
        elif q1_out_of_range:
            root_cause = "ACTION_RANGE_CLIPPING"
            explanation = (
                f"Teacher actions have out-of-range values in dims: {[o['dim'] for o in range_outliers]}. "
                "Actions are clipped by env.step(), distorting the commanded trajectory."
            )
        else:
            root_cause = "ACTION_CONTRACT_UNCLEAR_FROM_REPLAY"
            explanation = (
                "Teacher actions do not open drawer in env but no clear mismatch found in "
                "action format/range. Possible: sim-to-sim gap (LIBERO→Infinigen physics mismatch), "
                "or the actions require correct initial state."
            )
    elif any_strict_success is True:
        root_cause = "NOT_ACTION_CONTRACT"
        explanation = (
            "Teacher actions DO open the drawer in env replay. Action format and contract "
            "are compatible. Root cause is NOT action normalization/mismatch."
        )
    else:
        # Replay didn't run or inconclusive
        if q2_gripper_mismatch:
            root_cause = "LIKELY_GRIPPER_CONVENTION"
            explanation = (
                f"Gripper convention mismatch detected: {gripper_findings['issue']}. "
                "Teacher replay did not confirm or rule this out."
            )
        elif q1_out_of_range:
            root_cause = "LIKELY_ACTION_RANGE"
            explanation = (
                f"Out-of-range actions detected: {[o['dim'] for o in range_outliers]}. "
                "Teacher replay inconclusive."
            )
        else:
            root_cause = "INCONCLUSIVE_NEED_REPLAY"
            explanation = (
                "No clear action mismatch found from static analysis. "
                "Teacher action replay required for definitive verdict."
            )

    verdict = {
        "root_cause": root_cause,
        "explanation": explanation,
        "q1_action_in_range": not q1_out_of_range,
        "q2_gripper_convention_ok": not q2_gripper_mismatch,
        "q3_contract_scales_ok": q3_contract_ok,
        "q4_teacher_replay_ever_attached": any_ever_attached,
        "q4_teacher_replay_drawer_opened": any_drawer_opened,
        "q4_teacher_replay_strict_success": any_strict_success,
        "rca2_eliminates_action_normalization": (
            root_cause == "NOT_ACTION_CONTRACT"
        ),
    }

    print(f"\n  RCA2 Verdict: {root_cause}")
    print(f"  Explanation: {explanation}")
    print(f"  Q1 (range): {'PASS' if verdict['q1_action_in_range'] else 'FAIL'}")
    print(f"  Q2 (gripper): {'PASS' if verdict['q2_gripper_convention_ok'] else 'FAIL'}")
    print(f"  Q3 (contract): {'PASS' if verdict['q3_contract_scales_ok'] else 'FAIL'}")
    print(f"  Q4 (replay attached): {any_ever_attached}")
    print(f"  Q4 (replay opened): {any_drawer_opened}")
    print(f"  Q4 (replay strict_success): {any_strict_success}")

    return verdict


def main():
    t0 = time.time()
    result = {
        "rca_id": "RCA2",
        "scope": "overfit_subset_7_episodes",
        "scope_guardrail": (
            "Conclusions apply ONLY to 7 overfit episodes and current env/inference pipeline. "
            "Do NOT extrapolate to full V59 dataset."
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
    }

    # [1] Dataset action stats
    try:
        ds = load_dataset()
        stats, episode_info = compute_dataset_action_stats(ds, OVERFIT_EPISODES)
        result["dataset_stats"] = stats
        result["episode_info"] = episode_info
    except Exception as e:
        print(f"FAILED at dataset load: {e}")
        import traceback; traceback.print_exc()
        result["dataset_error"] = str(e)
        stats = None
        episode_info = []

    if stats:
        # [2] Gripper convention
        gripper_findings = check_gripper_convention(episode_info)
        result["gripper_convention"] = gripper_findings

        # [3] Action range
        range_outliers = check_action_range(stats, episode_info)
        result["range_outliers"] = range_outliers

        # [4] Action contract
        contract_info = check_action_contract()
        result["action_contract"] = {
            "active_contract": contract_info[0],
            "env_defaults": contract_info[1],
            "consistency_checks": contract_info[2],
        }
    else:
        gripper_findings = {"verdict": "NO_DATA", "issue": "Dataset not loaded"}
        range_outliers = []
        contract_info = (None, None, {})
        result["gripper_convention"] = gripper_findings
        result["range_outliers"] = []

    # [5] Teacher action replay (KEY TEST)
    if stats and episode_info:
        replay_results = run_teacher_action_replay(episode_info, stats)
        result["teacher_replay"] = replay_results
    else:
        replay_results = None
        result["teacher_replay"] = []

    # [6] MINT inference actions
    mint_data = check_mint_inference_action_format()
    result["mint_inference_actions"] = {
        "overfit_eval_artifact_exists": mint_data[0] is not None,
        "env_gate_artifact_exists": mint_data[1] is not None,
    }

    # [7] Synthesize verdict
    if stats:
        verdict = synthesize_verdict(
            stats, episode_info, gripper_findings, range_outliers,
            contract_info, replay_results, mint_data
        )
        result["verdict"] = verdict
    else:
        result["verdict"] = {
            "root_cause": "DATA_LOAD_FAILED",
            "explanation": "Could not load V59 dataset to perform audit.",
        }

    result["elapsed_sec"] = round(time.time() - t0, 1)
    print(f"\nTotal time: {result['elapsed_sec']}s")

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {OUTPUT_FILE}")

    return result


if __name__ == "__main__":
    result = main()
    print("\n" + "="*60)
    print("RCA2 SUMMARY:")
    print(f"  Root Cause: {result['verdict']['root_cause']}")
    print(f"  Gripper OK: {result['verdict']['q2_gripper_convention_ok']}")
    print(f"  Range OK: {result['verdict']['q1_action_in_range']}")
    print(f"  Contract OK: {result['verdict']['q3_contract_scales_ok']}")
    if result.get("teacher_replay"):
        for r in result["teacher_replay"]:
            if "error" not in r:
                print(f"  Replay ep{r['episode_index']}: attached={r.get('ever_attached')}, "
                      f"drawer={r.get('max_drawer_fraction', 0):.3f}, "
                      f"success={r.get('strict_success')}")
