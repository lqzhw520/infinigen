#!/usr/bin/env python3
"""
Phase D: Offline LeRobot-style QA metrics module.

对标 HuggingFace LeRobot visualize_dataset 的关键指标：
  - Action Insights (Δa 平滑度, p95, per-dim σ)
  - State-Action Temporal Lag (action[t] → delta_state[t+1])
  - Action Autocorrelation + suggested chunk length
  - Cross-episode Action Variance heatmap
  - Episode Filtering (flagged indices with reason codes)

双跑模式：
  1. Reference mode: 对 LIBERO 本地 parquet 数据跑，输出参考分布
  2. Infinigen mode: 对 NPZ rollout 文件跑，输出对比报告

用法：
  # Reference: LIBERO
  python scripts/mint/dataset_qa_lerobot_fingerprint.py \
    --mode reference \
    --libero-root external/LIBERO/datasets/libero \
    --output experiments/mint/mint_drawer_v1/outputs/qa_reference.json

  # Infinigen: 自有 rollouts
  python scripts/mint/dataset_qa_lerobot_fingerprint.py \
    --mode infinigen \
    --npz-dir experiments/mint/mint_drawer_v1/artifacts/c2_replay_valid_rollouts \
    --output experiments/mint/mint_drawer_v1/outputs/qa_infinigen.json

  # Compare: 对比两者
  python scripts/mint/dataset_qa_lerobot_fingerprint.py \
    --mode compare \
    --ref experiments/mint/mint_drawer_v1/outputs/qa_reference.json \
    --ours experiments/mint/mint_drawer_v1/outputs/qa_infinigen.json \
    --output experiments/mint/mint_drawer_v1/outputs/qa_comparison.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "mint"))

STATE_NAMES = ["eef_x", "eef_y", "eef_z", "q0", "q1", "q2", "q3", "gripper"]
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]

ACTION_NAMES_7 = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
STATE_NAMES_8 = ["eef_x", "eef_y", "eef_z", "q0", "q1", "q2", "q3", "gripper_joint"]


# ---------------------------------------------------------------------------
# Core metric functions (framework-agnostic, operate on numpy arrays)
# ---------------------------------------------------------------------------

def action_delta_stats(actions: np.ndarray) -> dict[str, Any]:
    """
    Action Insights: per-step |Δa|_2 统计。
    对标 visualize_dataset "Action Insights" 面板。

    Args:
        actions: (T, 7) float32 array

    Returns:
        Dict with per-dim and aggregate delta statistics.
    """
    if len(actions) < 2:
        return {"error": "Need at least 2 steps"}
    deltas = np.abs(np.diff(actions, axis=0))  # (T-1, 7)

    result = {
        "total_frames": int(len(actions)),
        "total_deltas": int(len(deltas)),
        "per_dim": {},
        "l2": {},
    }
    for d in range(7):
        vals = deltas[:, d]
        result["per_dim"][ACTION_NAMES_7[d]] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "p50": float(np.percentile(vals, 50)),
            "p95": float(np.percentile(vals, 95)),
            "p99": float(np.percentile(vals, 99)),
            "max": float(vals.max()),
            "zero_frac": float(np.mean(vals < 1e-6)),
        }

    # Aggregate L2 per step
    l2 = np.linalg.norm(deltas, axis=1)
    result["l2"] = {
        "mean": float(l2.mean()),
        "std": float(l2.std()),
        "p50": float(np.percentile(l2, 50)),
        "p95": float(np.percentile(l2, 95)),
        "p99": float(np.percentile(l2, 99)),
        "max": float(l2.max()),
    }

    # Gripper-specific: should be binary jumps (all or nothing)
    gripper_d = deltas[:, 6]
    unique_deltas = sorted(set(np.round(gripper_d, 4).tolist()))
    is_binary = unique_deltas in [[0.0], [0.0, 2.0], [0.0, 1.0, 2.0]] or (
        len(unique_deltas) <= 2 and all(abs(v) < 0.01 or abs(abs(v) - 2.0) < 0.01 for v in unique_deltas)
    )
    result["gripper_analysis"] = {
        "is_binary_jumps": is_binary,
        "unique_deltas": unique_deltas[:10],
        "num_transitions": int(np.sum(gripper_d > 0.01)),
        "transitions_per_ep": float(np.mean(gripper_d > 0.01)),
    }
    return result


def action_autocorrelation(actions: np.ndarray, max_lag: int = 30) -> dict[str, Any]:
    """
    Action Autocorrelation: 每维自相关，衰减到 0.5 的 lag 即为建议 chunk length。
    对标 visualize_dataset "Action Insights > chunk length" 推断。

    Args:
        actions: (T, 7) float32 array
        max_lag: maximum lag to compute

    Returns:
        Dict with per-dim autocorrelation and suggested chunk length.
    """
    if len(actions) < max_lag + 1:
        return {"error": f"Need > {max_lag} steps"}

    result = {"per_dim": {}, "suggested_chunk_length": {}}
    for d in range(7):
        vals = actions[:, d]
        vals = vals - vals.mean()
        n = len(vals)
        acf = []
        for lag in range(1, max_lag + 1):
            if lag >= n:
                break
            c0 = np.dot(vals, vals) / n
            if c0 < 1e-8:
                acf.append(0.0)
            else:
                acf.append(np.dot(vals[:-lag], vals[lag:]) / n / c0)

        # Find lag where autocorrelation drops below 0.5
        suggested_lag = max_lag
        for lag_i, val in enumerate(acf):
            if val < 0.5:
                suggested_lag = lag_i + 1
                break

        result["per_dim"][ACTION_NAMES_7[d]] = {
            "acf_1": float(acf[0]) if len(acf) > 0 else None,
            "acf_5": float(acf[4]) if len(acf) > 4 else None,
            "acf_10": float(acf[9]) if len(acf) > 9 else None,
            "suggested_chunk_length": suggested_lag,
        }
        result["suggested_chunk_length"][ACTION_NAMES_7[d]] = suggested_lag

    # Overall: median chunk length across position dims
    pos_chunk = [result["suggested_chunk_length"].get(ACTION_NAMES_7[d], max_lag) for d in range(3)]
    result["overall_position_chunk_length"] = int(np.median(pos_chunk))
    return result


def state_action_lag(
    states: np.ndarray, actions: np.ndarray
) -> dict[str, Any]:
    """
    State-Action Temporal Alignment: action[t] → delta_state[t+1] 的最佳 lag。
    对标 visualize_dataset "Temporal alignment" 推断。

    这是 P0 级的合同验证：
      - LIBERO ground truth: lag=1 for position dims, r>0.97
      - 如果 Infinigen lag!=1，说明 action/state 约定不一致

    Args:
        states: (T, 8) float32 array
        actions: (T, 7) float32 array

    Returns:
        Dict with best lag per dimension and Pearson r.
    """
    T = min(len(states), len(actions))
    if T < 5:
        return {"error": f"Need at least 5 steps, got {T}"}

    delta_state = states[1:T] - states[: T - 1]  # (T-1, 8)
    acts = actions[: T - 1]  # (T-1, 7)

    result = {
        "per_dim": {},
        "p0_verdict": "PASS",
        "p0_issues": [],
    }

    for d in range(min(7, delta_state.shape[1])):
        best_lag, best_r = 0, 0.0
        for lag in [-1, 0, 1]:
            if lag == 0:
                a_seg, ds_seg = acts[:, d], delta_state[:, d]
            elif lag == 1:
                a_seg, ds_seg = acts[:-1, d], delta_state[1:, d]
            else:  # lag = -1
                a_seg, ds_seg = acts[1:, d], delta_state[:-1, d]
            min_len = min(len(a_seg), len(ds_seg))
            if min_len < 5:
                continue
            a_seg, ds_seg = a_seg[:min_len], ds_seg[:min_len]
            if np.std(a_seg) < 1e-6 or np.std(ds_seg) < 1e-6:
                continue
            r = float(np.corrcoef(a_seg, ds_seg)[0, 1])
            if abs(r) > abs(best_r):
                best_r, best_lag = r, lag

        expected_lag = 1 if d < 3 else 1  # position and rotation dims expect lag=1
        is_correct = best_lag == expected_lag
        if not is_correct:
            result["p0_verdict"] = "FAIL"
            result["p0_issues"].append(
                f"action[{d}] ({ACTION_NAMES_7[d]}): expected lag=1, got lag={best_lag}, r={best_r:.4f}"
            )

        result["per_dim"][ACTION_NAMES_7[d]] = {
            "best_lag": int(best_lag),
            "r": float(best_r),
            "expected_lag": expected_lag,
            "is_correct": is_correct,
        }

    return result


def cross_episode_variance(
    episode_actions: list[np.ndarray],
    episode_states: list[np.ndarray] | None = None,
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    Cross-Episode Action Variance: 按归一化时间分桶，算每维的方差热图摘要。
    对标 visualize_dataset "Cross-Episode Variance"。

    用于发现：
      - 过度脚本化（各 episode 在同一相位方差过低）
      - AnyGrasp/种子间不一致（某些桶方差过高）

    Args:
        episode_actions: list of (T_i, 7) arrays
        episode_states: optional list of (T_i, 8) arrays
        n_bins: number of time bins for bucketing

    Returns:
        Dict with per-bin variance per dimension.
    """
    if not episode_actions:
        return {"error": "No episodes provided"}

    result = {
        "n_episodes": len(episode_actions),
        "n_bins": n_bins,
        "per_bin": {},
    }

    # Interpolate each episode to n_bins
    for bin_idx in range(n_bins):
        frac = (bin_idx + 0.5) / n_bins
        bucket_vals = []
        for ep in episode_actions:
            if len(ep) < 2:
                continue
            t = int(frac * (len(ep) - 1))
            bucket_vals.append(ep[t])
        if bucket_vals:
            result["per_bin"][f"bin_{bin_idx:02d}_frac_{frac:.1f}"] = {
                dim: {
                    "mean": float(np.mean([v[dim] for v in bucket_vals])),
                    "std": float(np.std([v[dim] for v in bucket_vals])),
                    "n": len(bucket_vals),
                }
                for dim in range(7)
            }

    # Per-dim overall variance across episodes
    all_actions = np.concatenate(episode_actions, axis=0)
    result["overall"] = {
        ACTION_NAMES_7[d]: {
            "std": float(all_actions[:, d].std()),
            "range": [float(all_actions[:, d].min()), float(all_actions[:, d].max())],
        }
        for d in range(7)
    }

    # Check for over-scripted: low variance at grasp phase
    # (Grasp phase is typically in the last 20-50% of episode)
    grasp_frac_start = 0.5
    grasp_bucket = n_bins - 2
    if f"bin_{grasp_bucket:02d}" in result["per_bin"]:
        gv = result["per_bin"][f"bin_{grasp_bucket:02d}"]
        gripper_std = gv.get(6, {}).get("std", None)
        result["over_scripted_analysis"] = {
            "grasp_phase_bin": grasp_bucket,
            "grasp_phase_gripper_std": gripper_std,
            "grasp_phase_gripper_std_threshold": 0.1,
            "is_over_scripted": (
                gripper_std is not None and gripper_std < 0.05
            ),
            "note": "If grasp phase gripper_std < 0.05, all episodes have identical gripper action — over-scripted",
        }

    return result


def flag_episodes(
    episode_actions: list[np.ndarray],
    episode_states: list[np.ndarray] | None = None,
    action_delta_p95_ref: float | None = None,
    max_position_error_ref: float | None = None,
) -> dict[str, Any]:
    """
    Episode Filtering: 综合所有指标，生成排除列表。

    对标 visualize_dataset "Filtering / Flag episodes"。

    排除标准：
      1. action_delta_l2 p95 异常高（>3x reference p95）→ 动作突变/不稳定
      2. action_delta_l2 p95 异常低（<0.1x reference）→ 几乎无运动
      3. gripper transitions 数量为 0 或极少 → 没有抓取动作
      4. episode 长度 < 50 帧 → 过短
      5. gripper action 全程不变 → 缺少交互

    Args:
        episode_actions: list of (T_i, 7) arrays
        episode_states: optional list of (T_i, 8) arrays
        action_delta_p95_ref: reference p95 from LIBERO
        max_position_error_ref: reference max position error

    Returns:
        Dict with per-episode flags and overall exclude list.
    """
    if not episode_actions:
        return {"error": "No episodes"}

    REF_P95 = action_delta_p95_ref if action_delta_p95_ref else 0.161
    REF_MAX = max_position_error_ref if max_position_error_ref else 0.03

    flags = []
    exclude_episodes = []
    exclude_reasons: dict[int, list[str]] = {}

    for ep_idx, actions in enumerate(episode_actions):
        ep_flags = []
        reasons = []

        # 1. Too short
        if len(actions) < 50:
            ep_flags.append("short_episode")
            reasons.append(f"length={len(actions)} < 50")

        # 2. Action delta L2 analysis
        deltas = np.abs(np.diff(actions, axis=0))
        if len(deltas) > 0:
            l2 = np.linalg.norm(deltas, axis=1)
            p95 = float(np.percentile(l2, 95))
            p50 = float(np.percentile(l2, 50))

            if p95 > REF_P95 * 3:
                ep_flags.append("high_action_variance")
                reasons.append(f"action_delta_l2_p95={p95:.4f} > 3x_ref({REF_P95:.4f})")
            if p50 < 0.005:
                ep_flags.append("near_zero_motion")
                reasons.append(f"action_delta_l2_p50={p50:.5f} < 0.005")

        # 3. Gripper analysis
        gripper = actions[:, 6]
        n_transitions = int(np.sum(np.abs(np.diff(gripper)) > 0.01))
        if n_transitions == 0:
            ep_flags.append("no_gripper_transitions")
            reasons.append("gripper never changes")
        elif n_transitions < 2:
            ep_flags.append("too_few_gripper_transitions")
            reasons.append(f"only {n_transitions} transitions")

        # 4. Gripper consistency (should be open→close→open pattern)
        if n_transitions > 0:
            first_val = float(gripper[0])
            last_val = float(gripper[-1])
            if abs(first_val - last_val) < 0.01:
                ep_flags.append("gripper_not_closed")
                reasons.append("gripper at same value at start and end")

        # 5. Action range check (should not exceed [-1, 1])
        for d in range(6):
            dim_max = float(np.abs(actions[:, d]).max())
            if dim_max > 1.1:  # small tolerance
                ep_flags.append(f"action_dim{d}_out_of_range")
                reasons.append(f"action[{d}] max_abs={dim_max:.3f} > 1.1")

        is_flagged = len(ep_flags) > 0
        if is_flagged:
            exclude_episodes.append(ep_idx)
            exclude_reasons[ep_idx] = reasons

        flags.append({
            "episode_index": ep_idx,
            "length": int(len(actions)),
            "n_gripper_transitions": n_transitions,
            "flags": ep_flags,
            "is_flagged": is_flagged,
        })

    # Summary
    total = len(flags)
    flagged = len(exclude_episodes)
    passing = total - flagged

    result = {
        "total_episodes": total,
        "flagged_episodes": flagged,
        "passing_episodes": passing,
        "pass_rate": f"{passing}/{total} ({100*passing/total:.0f}%)",
        "exclude_episode_indices": exclude_episodes,
        "exclude_reasons": exclude_reasons,
        "per_episode": flags,
        "reference_used": {
            "action_delta_l2_p95_ref": REF_P95,
            "max_position_error_ref": REF_MAX,
        },
        "cli_export": (
            f"python scripts/mint/run_g6_lerobot_pack.py "
            f"--exclude_episodes {','.join(map(str, exclude_episodes))}"
            if exclude_episodes else "# No exclusions needed"
        ),
    }
    return result


# ---------------------------------------------------------------------------
# Data loading: LIBERO parquet vs Infinigen NPZ
# ---------------------------------------------------------------------------

def load_libero_episodes(
    libero_root: Path, max_episodes: int = 100
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load episodes from LIBERO parquet files."""
    import pandas as pd

    parquet_files = sorted(libero_root.glob("data/chunk-*/*.parquet"))[:max_episodes]
    states_list, actions_list = [], []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        s = np.stack(df["state"].values).astype(np.float64)
        a = np.stack(df["actions"].values).astype(np.float64)
        states_list.append(s)
        actions_list.append(a)
    return states_list, actions_list


def load_infinigen_episodes(
    npz_dir: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    """Load episodes from Infinigen NPZ rollouts."""
    import json

    npz_files = sorted(npz_dir.glob("*.npz"))
    states_list, actions_list, metas = [], [], []
    for pf in npz_files:
        data = dict(np.load(pf, allow_pickle=True))
        # NPZ may have 'states' or 'state' key
        s_key = "states" if "states" in data else "state"
        a_key = "actions" if "actions" in data else "action"
        s = data.get(s_key, np.zeros((0, 8)))
        a = data.get(a_key, np.zeros((0, 7)))
        if s.ndim == 1:
            s = s.reshape(1, -1)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        states_list.append(s.astype(np.float64))
        actions_list.append(a.astype(np.float64))
        meta_path = pf.with_suffix(".json")
        meta = (
            json.loads(meta_path.read_text())
            if meta_path.exists()
            else {"source": str(pf.name)}
        )
        metas.append(meta)
    return states_list, actions_list, metas


# ---------------------------------------------------------------------------
# Reference computation
# ---------------------------------------------------------------------------

def compute_reference(libero_root: Path, max_episodes: int = 100) -> dict[str, Any]:
    """Compute reference statistics from LIBERO dataset."""
    print(f"Loading LIBERO episodes from {libero_root}...")
    states_list, actions_list = load_libero_episodes(libero_root, max_episodes)
    print(f"  Loaded {len(actions_list)} episodes")

    all_actions = np.concatenate(actions_list, axis=0)
    all_states = np.concatenate(states_list, axis=0) if states_list else None

    result = {
        "source": "LIBERO_reference",
        "dataset": "LIBERO_full",
        "episodes_sampled": len(actions_list),
        "total_frames": int(sum(len(a) for a in actions_list)),
    }

    print("  Computing action_delta_stats...")
    result["action_delta"] = action_delta_stats(all_actions)

    print("  Computing action_autocorrelation...")
    result["autocorrelation"] = action_autocorrelation(all_actions)

    if all_states is not None:
        print("  Computing state_action_lag...")
        result["state_action_lag"] = state_action_lag(all_states, all_actions)

    print("  Computing cross_episode_variance...")
    result["cross_episode_variance"] = cross_episode_variance(actions_list, states_list)

    print("  Computing flag_episodes...")
    ref_p95 = result["action_delta"]["l2"]["p95"]
    result["episode_flags"] = flag_episodes(actions_list, states_list, ref_p95)

    return result


# ---------------------------------------------------------------------------
# Infinigen QA
# ---------------------------------------------------------------------------

def compute_infinigen_qa(npz_dir: Path) -> dict[str, Any]:
    """Compute QA metrics for Infinigen NPZ rollouts."""
    print(f"Loading Infinigen episodes from {npz_dir}...")
    states_list, actions_list, metas = load_infinigen_episodes(npz_dir)
    print(f"  Loaded {len(actions_list)} episodes")

    if not actions_list:
        return {"error": "No episodes found"}

    all_actions = np.concatenate(actions_list, axis=0)
    all_states = np.concatenate(states_list, axis=0) if states_list else None

    result = {
        "source": "Infinigen",
        "episodes_sampled": len(actions_list),
        "total_frames": int(sum(len(a) for a in actions_list)),
        "episode_sources": [m.get("seed", m.get("source", "unknown")) for m in metas],
    }

    print("  Computing action_delta_stats...")
    result["action_delta"] = action_delta_stats(all_actions)

    print("  Computing action_autocorrelation...")
    result["autocorrelation"] = action_autocorrelation(all_actions)

    if all_states is not None:
        print("  Computing state_action_lag...")
        result["state_action_lag"] = state_action_lag(all_states, all_actions)

    print("  Computing cross_episode_variance...")
    result["cross_episode_variance"] = cross_episode_variance(actions_list, states_list)

    print("  Computing flag_episodes...")
    ref_p95 = result["action_delta"]["l2"]["p95"]
    result["episode_flags"] = flag_episodes(actions_list, states_list, ref_p95)

    return result


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def compare(ref: dict[str, Any], ours: dict[str, Any]) -> dict[str, Any]:
    """Generate comparison report between reference and Infinigen metrics."""

    def pct_chg(ours_val: float, ref_val: float) -> str:
        if abs(ref_val) < 1e-9:
            return "inf"
        return f"{100*(ours_val - ref_val)/abs(ref_val):+.1f}%"

    comparison = {
        "summary": {},
        "action_delta": {},
        "autocorrelation": {},
        "state_action_lag": {},
        "episode_filtering": {},
        "verdict": "UNKNOWN",
        "verdict_issues": [],
    }

    # Action delta comparison
    ref_p95 = ref.get("action_delta", {}).get("l2", {}).get("p95", 0.161)
    ours_p95 = ours.get("action_delta", {}).get("l2", {}).get("p95", None)
    if ours_p95 is not None:
        comparison["action_delta"]["l2_p95"] = {
            "libero": ref_p95,
            "infinigen": ours_p95,
            "change": pct_chg(ours_p95, ref_p95),
        }
        if ours_p95 > ref_p95 * 3:
            comparison["verdict_issues"].append(
                f"ACTION_VARIANCE: Infinigen |Δa|_2 p95={ours_p95:.4f} >> LIBERO={ref_p95:.4f} — "
                "teacher has excessive motion jumps"
            )
        elif ours_p95 < ref_p95 * 0.1:
            comparison["verdict_issues"].append(
                f"ACTION_VARIANCE: Infinigen |Δa|_2 p95={ours_p95:.5f} << LIBERO={ref_p95:.4f} — "
                "teacher has near-zero motion (likely scripted/deterministic)"
            )

    # Gripper transitions
    ref_trans = ref.get("episode_flags", {}).get("total_episodes", 0)
    ref_flagged = ref.get("episode_flags", {}).get("flagged_episodes", 0)
    ours_trans = ours.get("episode_flags", {}).get("total_episodes", 0)
    ours_flagged = ours.get("episode_flags", {}).get("flagged_episodes", 0)
    comparison["episode_filtering"] = {
        "libero": f"{ref_trans - ref_flagged}/{ref_trans} passing",
        "infinigen": f"{ours_trans - ours_flagged}/{ours_trans} passing",
    }

    # State-action lag (P0 critical)
    # NOTE: LIBERO multi-task (40 types) reduces individual dim r to 0.2-0.3.
    # The key signal is best_lag — LIBERO consistently shows lag=1 for position
    # dims in single-task episodes; Infinigen's all-lag=0 is a different convention.
    ref_lag = ref.get("state_action_lag", {}).get("per_dim", {})
    ours_lag = ours.get("state_action_lag", {}).get("per_dim", {})
    lag_issues = []
    for d in range(7):
        rn = ACTION_NAMES_7[d]
        r_info = ref_lag.get(rn, {})
        o_info = ours_lag.get(rn, {})
        ref_l = r_info.get("best_lag", 999)
        ours_l = o_info.get("best_lag", 999)
        ref_r = r_info.get("r", 0.0)
        ours_r = o_info.get("r", 0.0)
        # Flag if best_lag differs from expected=1 AND correlation is meaningful
        if ref_l == 1 and ours_l != 1 and abs(ours_r) > 0.1:
            lag_issues.append(
                f"LAG[{d}]({rn}): LIBERO best_lag=1(r={ref_r:.3f}), "
                f"Infinigen best_lag={ours_l}(r={ours_r:.3f}) — convention mismatch"
            )

    comparison["state_action_lag"] = {
        "reference": ref_lag,
        "infinigen": ours_lag,
        "issues": lag_issues,
    }
    if lag_issues:
        comparison["verdict_issues"].extend(lag_issues)

    # Gripper analysis
    ref_grip = ref.get("action_delta", {}).get("gripper_analysis", {})
    ours_grip = ours.get("action_delta", {}).get("gripper_analysis", {})
    comparison["gripper_analysis"] = {
        "libero": ref_grip,
        "infinigen": ours_grip,
    }

    # Overall verdict
    critical_issues = [i for i in comparison["verdict_issues"] if "MISMATCH" in i or "BINARY" in i]
    if critical_issues:
        comparison["verdict"] = "FAIL — critical P0/P1 issues"
    elif comparison["verdict_issues"]:
        comparison["verdict"] = "WARN — non-critical issues"
    else:
        comparison["verdict"] = "PASS"

    return comparison


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LeRobot-style dataset QA")
    parser.add_argument("--mode", choices=["reference", "infinigen", "compare"], required=True)
    parser.add_argument("--libero-root", type=Path)
    parser.add_argument("--npz-dir", type=Path)
    parser.add_argument("--ref", type=Path, help="Reference JSON from --mode reference")
    parser.add_argument("--ours", type=Path, help="Infinigen JSON from --mode infinigen")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "reference":
        assert args.libero_root, "--libero-root required for reference mode"
        result = compute_reference(args.libero_root, args.max_episodes)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\nWritten: {args.output}")

    elif args.mode == "infinigen":
        assert args.npz_dir, "--npz-dir required for infinigen mode"
        result = compute_infinigen_qa(args.npz_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\nWritten: {args.output}")

    elif args.mode == "compare":
        assert args.ref and args.ours, "--ref and --ours required for compare mode"
        ref = json.loads(args.ref.read_text())
        ours = json.loads(args.ours.read_text())
        comparison = compare(ref, ours)

        md_lines = [
            "# LeRobot-style Dataset QA: LIBERO vs Infinigen Comparison\n",
            f"**Reference**: {ref.get('source', 'LIBERO')} — {ref.get('episodes_sampled', 0)} eps\n",
            f"**Infinigen**: {ours.get('source', 'Infinigen')} — {ours.get('episodes_sampled', 0)} eps\n",
            f"\n## Verdict: **{comparison['verdict']}**\n",
        ]

        if comparison["verdict_issues"]:
            md_lines.append("### Critical Issues\n")
            for issue in comparison["verdict_issues"]:
                md_lines.append(f"- ❌ {issue}\n")

        # Action delta
        ad = comparison.get("action_delta", {})
        if ad:
            l2 = ad.get("l2_p95", {})
            if l2:
                md_lines.extend([
                    "\n## Action Delta |Δa|\n",
                    f"| Metric | LIBERO | Infinigen | Change |\n",
                    f"|--------|--------|-----------|--------|\n",
                    f"| |Δa|_2 p95 | {l2.get('libero', 'N/A'):.4f} | "
                    f"{l2.get('infinigen', 'N/A'):.4f} | "
                    f"{l2.get('change', 'N/A')} |\n",
                ])

        # State-action lag
        lag = comparison.get("state_action_lag", {})
        if lag:
            md_lines.append("\n## State-Action Temporal Lag\n")
            md_lines.append(
                "| Dim | LIBERO lag (r) | Infinigen lag (r) | Correct? |\n"
                "|-----|---------------|-------------------|----------|\n"
            )
            ref_lags = lag.get("reference", {})
            ours_lags = lag.get("infinigen", {})
            for d in range(7):
                rn = ACTION_NAMES_7[d]
                ri = ours_lags.get(rn, {})
                r_ref = ref_lags.get(rn, {}).get("r", 0)
                r_ours = ri.get("r", 0)
                lag_ref = ref_lags.get(rn, {}).get("best_lag", "?")
                lag_ours = ri.get("best_lag", "?")
                ok_ref = ref_lags.get(rn, {}).get("is_correct", "?")
                ok_ours = ri.get("is_correct", "?")
                icon = "✅" if ok_ours else "❌"
                md_lines.append(
                    f"| {rn} | lag={lag_ref} r={r_ref:.3f} | "
                    f"lag={lag_ours} r={r_ours:.3f} | {icon} |\n"
                )

        # Episode filtering
        ef = comparison.get("episode_filtering", {})
        if ef:
            md_lines.extend([
                "\n## Episode Filtering\n",
                f"- **LIBERO**: {ef.get('libero', 'N/A')}\n",
                f"- **Infinigen**: {ef.get('infinigen', 'N/A')}\n",
            ])

        # Gripper
        ga = comparison.get("gripper_analysis", {})
        if ga:
            rg = ga.get("libero", {})
            og = ga.get("infinigen", {})
            md_lines.extend([
                "\n## Gripper Analysis\n",
                f"| Metric | LIBERO | Infinigen |\n",
                f"|--------|--------|-----------|\n",
                f"| is_binary_jumps | {rg.get('is_binary_jumps', 'N/A')} | {og.get('is_binary_jumps', 'N/A')} |\n",
                f"| transitions/ep | {rg.get('num_transitions', 'N/A')} | {og.get('num_transitions', 'N/A')} |\n",
            ])

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(md_lines))
        print(f"\nWritten: {args.output}")

        # Also save as JSON
        json_out = args.output.with_suffix(".json")
        json_out.write_text(json.dumps(comparison, indent=2))
        print(f"JSON: {json_out}")


if __name__ == "__main__":
    main()
