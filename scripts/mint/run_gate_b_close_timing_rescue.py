#!/usr/bin/env python3
"""
Stage A.6 / H1 - Close-Timing Rescue Audit
==========================================

Proposal-only follow-up after corrected Stage A and Stage A.5.

Tests whether delaying teacher close actions until the current env is
actually near the handle materially contracts the replay-gap.
"""

from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN_ROOT = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1"
ARTIFACT_ROOT = CAMPAIGN_ROOT / "artifacts"
SOVEREIGN = CAMPAIGN_ROOT / "sovereign"
PROPOSALS_DIR = SOVEREIGN / "proposals"
SCRIPTS_MINT = PROJECT_ROOT / "scripts" / "mint"

sys.path.insert(0, str(SCRIPTS_MINT))
sys.path.insert(0, str(PROJECT_ROOT))

from drawer_robot_env import DrawerRobotEnv  # noqa: E402
from run_gate_b_teacher_replayability import (  # noqa: E402
    current_env_action_contract,
    current_env_handle_local,
    git_head,
    now_ts,
    resolve_npz,
    run_replay_mode,
    selected_episodes,
    source_meta,
    summarize_run_from_traces,
    to_builtin,
)


IMAGE_SIZE = 16
RENDER_OBSERVATIONS = False
MAX_STEPS = 200
WORKERS = max(1, min(4, (os.cpu_count() or 1)))
TOP10_EPISODES = [157, 205, 25, 180, 3, 181, 158, 182, 138, 209]
ATTACH_THRESHOLD = 0.05
H1_DIR = ARTIFACT_ROOT / "gate_b7_close_timing_rescue"
E031_YAML = PROPOSALS_DIR / "E031_gate_b_close_timing_rescue_draft.yaml"
E031_MD = PROPOSALS_DIR / "E031_gate_b_close_timing_rescue_draft.md"

SAMPLE_SCOPE = {
    "sample": "Gate A action-quality top-10",
    "mapping_target": "success-selected legacy rollout subset",
    "population_claim": "No population-wide claim over full 240 episodes",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_builtin(payload), indent=2, ensure_ascii=False))


def save_yaml(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(to_builtin(payload), allow_unicode=True, sort_keys=False))


def save_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_builtin(value) for key, value in row.items()})


def _episode_set() -> list[tuple[int, str]]:
    selected = selected_episodes(",".join(str(v) for v in TOP10_EPISODES))
    return [(int(ep), str(name)) for ep, name in selected]


def _teacher_close_step(actions: np.ndarray) -> int | None:
    for idx, action in enumerate(actions):
        if float(action[6]) < 0.0:
            return int(idx)
    return None


def run_close_timing_rescue(
    *,
    episode_index: int,
    npz_name: str,
    attach_threshold: float,
    max_steps: int,
    image_size: int,
    render_observations: bool,
) -> dict[str, Any]:
    npz_path = resolve_npz(npz_name)
    source_npz = dict(np.load(npz_path, allow_pickle=True))
    meta = source_meta(npz_path)
    action_contract = current_env_action_contract(
        meta,
        attach_threshold_override=attach_threshold,
    )
    env = DrawerRobotEnv(
        seed=int(meta["seed"]),
        image_size=image_size,
        max_steps=max_steps,
        action_contract=action_contract,
        render_observations=render_observations,
    )
    actions = np.asarray(source_npz["actions"], dtype=np.float32)
    eef_positions: list[np.ndarray] = []
    drawer_trace: list[float] = []
    attached_trace: list[bool] = []
    handle_distance_trace: list[float] = []
    success = False
    attach_step = None
    suppressed_close_steps = 0
    first_allowed_close_step = None
    first_near_handle_step = None
    try:
        obs = env.reset()
        local_handle, handle_metadata = current_env_handle_local(env)
        env.attachment_local = local_handle
        near_handle_unlocked = False
        for step_index in range(max_steps):
            handle_pos, _ = env.local_to_world_pose(
                local_handle["pos"],
                local_handle["quat"],
                fraction=obs.drawer_fraction,
            )
            current_handle_distance = float(np.linalg.norm(obs.eef_pos - handle_pos))
            if current_handle_distance <= attach_threshold:
                near_handle_unlocked = True
                if first_near_handle_step is None:
                    first_near_handle_step = int(step_index)
            teacher_action = (
                actions[step_index] if step_index < len(actions) else np.zeros((7,), dtype=np.float32)
            )
            action = teacher_action.copy()
            teacher_wants_close = float(action[6]) < 0.0
            if teacher_wants_close and not near_handle_unlocked:
                action[6] = 1.0
                suppressed_close_steps += 1
            elif teacher_wants_close and near_handle_unlocked and first_allowed_close_step is None:
                first_allowed_close_step = int(step_index)

            obs, _reward, done, info = env.step(action)
            handle_pos, _ = env.local_to_world_pose(
                local_handle["pos"],
                local_handle["quat"],
                fraction=obs.drawer_fraction,
            )
            eef_positions.append(obs.eef_pos.copy())
            drawer_trace.append(float(obs.drawer_fraction))
            attached_trace.append(bool(info.get("attached")))
            handle_distance_trace.append(float(np.linalg.norm(obs.eef_pos - handle_pos)))
            if info.get("attached") and attach_step is None:
                attach_step = int(step_index)
            success = success or bool(info.get("is_success", False))
            if done:
                break
    finally:
        env.close()

    summary = summarize_run_from_traces(
        episode_index=episode_index,
        npz_path=npz_path,
        source_npz=source_npz,
        source_meta_payload=meta,
        replay_mode="close_timing_rescue",
        eef_positions=np.asarray(eef_positions, dtype=np.float32),
        drawer_trace=np.asarray(drawer_trace, dtype=np.float32),
        attached_trace=np.asarray(attached_trace, dtype=np.bool_),
        handle_distance_trace=np.asarray(handle_distance_trace, dtype=np.float32),
        success=success,
        attach_step=attach_step,
        action_contract=action_contract,
        frames=[],
    )
    summary["suppressed_close_steps"] = int(suppressed_close_steps)
    summary["first_allowed_close_step"] = first_allowed_close_step
    summary["first_rescue_near_handle_step"] = first_near_handle_step
    summary["measurement_image_size_px"] = int(image_size)
    summary["render_observations"] = bool(render_observations)
    summary["rescue_definition"] = {
        "name": "close_timing_rescue",
        "description": "Teacher translation/rotation actions unchanged; gripper close commands are suppressed until current-env handle distance is within canonical attach threshold.",
        "close_release_distance_m": float(attach_threshold),
        "teacher_motion_preserved": True,
        "teacher_gripper_modified": True,
    }
    return summary


def _run_episode_pair(
    episode_index: int,
    npz_name: str,
    attach_threshold: float,
    max_steps: int,
    image_size: int,
    render_observations: bool,
) -> dict[str, Any]:
    npz_path = resolve_npz(npz_name)
    source_npz = dict(np.load(npz_path, allow_pickle=True))
    meta = source_meta(npz_path)
    actions = np.asarray(source_npz["actions"], dtype=np.float32)
    baseline = run_replay_mode(
        episode_index=episode_index,
        npz_path=npz_path,
        source_npz=source_npz,
        source_meta_payload=meta,
        replay_mode="open_loop",
        max_steps=max_steps,
        attach_threshold_override=attach_threshold,
        image_size=image_size,
        render_observations=render_observations,
        capture_frames=False,
        capture_trace=False,
    )
    rescue = run_close_timing_rescue(
        episode_index=episode_index,
        npz_name=npz_name,
        attach_threshold=attach_threshold,
        max_steps=max_steps,
        image_size=image_size,
        render_observations=render_observations,
    )
    teacher_close_step = _teacher_close_step(actions)
    return {
        "episode_index": int(episode_index),
        "npz_name": npz_name,
        "npz_path": str(npz_path),
        "teacher_close_step": teacher_close_step,
        "baseline": baseline,
        "rescue": rescue,
    }


def _rate(rows: list[dict[str, Any]], key: str, metric: str) -> float:
    n = max(len(rows), 1)
    if metric == "success":
        return float(sum(1 for row in rows if row[key]["success"]) / n)
    if metric == "attach":
        return float(sum(1 for row in rows if row[key]["ever_attached"]) / n)
    raise ValueError(metric)


def build_h1(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    transition_counts = Counter()
    rescued_success_count = 0
    rescued_attach_count = 0
    for row in bundles:
        baseline = row["baseline"]
        rescue = row["rescue"]
        transition_counts[f"{baseline['failure_stage']} -> {rescue['failure_stage']}"] += 1
        if not baseline["success"] and rescue["success"]:
            rescued_success_count += 1
        if not baseline["ever_attached"] and rescue["ever_attached"]:
            rescued_attach_count += 1
        rows.append(
            {
                "episode_index": row["episode_index"],
                "npz_name": row["npz_name"],
                "teacher_close_step": row["teacher_close_step"],
                "baseline_success": baseline["success"],
                "baseline_ever_attached": baseline["ever_attached"],
                "baseline_failure_stage": baseline["failure_stage"],
                "baseline_first_near_handle_step": baseline["first_near_handle_step"],
                "baseline_attach_step": baseline["attach_step"],
                "baseline_min_handle_distance_m": baseline["min_handle_distance_m"],
                "baseline_max_drawer_fraction": baseline["max_drawer_fraction"],
                "rescue_success": rescue["success"],
                "rescue_ever_attached": rescue["ever_attached"],
                "rescue_failure_stage": rescue["failure_stage"],
                "rescue_first_near_handle_step": rescue["first_near_handle_step"],
                "rescue_attach_step": rescue["attach_step"],
                "rescue_min_handle_distance_m": rescue["min_handle_distance_m"],
                "rescue_max_drawer_fraction": rescue["max_drawer_fraction"],
                "suppressed_close_steps": rescue["suppressed_close_steps"],
                "first_allowed_close_step": rescue["first_allowed_close_step"],
                "first_rescue_near_handle_step": rescue["first_rescue_near_handle_step"],
                "success_rescued": bool((not baseline["success"]) and rescue["success"]),
                "attach_rescued": bool((not baseline["ever_attached"]) and rescue["ever_attached"]),
            }
        )

    baseline_attach = _rate(bundles, "baseline", "attach")
    baseline_success = _rate(bundles, "baseline", "success")
    rescue_attach = _rate(bundles, "rescue", "attach")
    rescue_success = _rate(bundles, "rescue", "success")
    attach_delta = float(rescue_attach - baseline_attach)
    success_delta = float(rescue_success - baseline_success)
    total_suppressed = int(sum(row["suppressed_close_steps"] for row in rows))
    mean_suppressed = float(np.mean([row["suppressed_close_steps"] for row in rows])) if rows else 0.0

    if success_delta > 0.20:
        final_branch = "close timing materially contracts the replay-gap"
    elif success_delta > 0.0 or attach_delta > 0.20 or rescued_success_count > 0:
        final_branch = "close timing partially contributes to the replay-gap but is not sufficient alone"
    else:
        final_branch = "close timing alone does not materially contract the replay-gap"

    artifact = {
        "experiment": "gate_b7_close_timing_rescue",
        "timestamp": now_ts(),
        "sample_scope": dict(SAMPLE_SCOPE),
        "attach_threshold_m": float(ATTACH_THRESHOLD),
        "measurement_image_size_px": int(IMAGE_SIZE),
        "render_observations": bool(RENDER_OBSERVATIONS),
        "episodes": rows,
        "aggregate": {
            "baseline_attach_rate": baseline_attach,
            "baseline_success_rate": baseline_success,
            "rescue_attach_rate": rescue_attach,
            "rescue_success_rate": rescue_success,
            "attach_rate_delta": attach_delta,
            "success_rate_delta": success_delta,
            "rescued_success_count": int(rescued_success_count),
            "rescued_attach_count": int(rescued_attach_count),
            "total_suppressed_close_steps": total_suppressed,
            "mean_suppressed_close_steps_per_episode": mean_suppressed,
            "failure_stage_transitions": dict(transition_counts),
            "final_branch": final_branch,
        },
    }
    save_json(H1_DIR / "gate_b7_summary.json", artifact)
    save_csv(H1_DIR / "gate_b7_episodes.csv", rows)
    save_text(H1_DIR / "gate_b7_summary.md", render_h1_markdown(artifact))
    return artifact


def build_e031(h1: dict[str, Any]) -> tuple[dict[str, Any], str]:
    agg = h1["aggregate"]
    final_branch = str(agg["final_branch"])
    draft = {
        "draft_id": "E031_draft",
        "draft_type": "proposal_only_followup",
        "target_evidence_id": "E031",
        "experiment_id": "gate_b_close_timing_rescue",
        "timestamp": now_ts(),
        "scope": "gate_b_top10_close_timing_rescue",
        "verified": False,
        "sample_scope": dict(SAMPLE_SCOPE),
        "summary": (
            "Stage A.6 / H1 tests whether delaying teacher close commands until current-env near-handle "
            "timing materially contracts the replay-gap while preserving teacher translation and rotation actions."
        ),
        "source_artifacts": {
            "e029": "sovereign/proposals/E029_gate_b_corrected_deconfounded_draft.yaml",
            "e030": "sovereign/proposals/E030_gate_b_replay_gap_contraction_draft.yaml",
            "h1": "artifacts/gate_b7_close_timing_rescue/gate_b7_summary.json",
        },
        "measurement_runtime_note": (
            f"H1 used image_size={IMAGE_SIZE} and render_observations={RENDER_OBSERVATIONS} "
            "for state-only measurement throughput; physics, task semantics, and action contracts were unchanged."
        ),
        "key_findings": {
            "baseline_success_rate": float(agg["baseline_success_rate"]),
            "rescue_success_rate": float(agg["rescue_success_rate"]),
            "success_rate_delta": float(agg["success_rate_delta"]),
            "baseline_attach_rate": float(agg["baseline_attach_rate"]),
            "rescue_attach_rate": float(agg["rescue_attach_rate"]),
            "attach_rate_delta": float(agg["attach_rate_delta"]),
            "rescued_success_count": int(agg["rescued_success_count"]),
            "rescued_attach_count": int(agg["rescued_attach_count"]),
            "mean_suppressed_close_steps_per_episode": float(agg["mean_suppressed_close_steps_per_episode"]),
        },
        "final_branch": final_branch,
        "stage_c_reconsider_allowed": False,
        "allowed_wording": [
            "Close timing can be discussed as a replay-gap contraction factor under canonical current-env semantics.",
            "Teacher motion channels stayed fixed while only premature close commands were delayed.",
        ],
        "disallowed_wording": [
            "policy-only gap proven",
            "root cause solved",
            "source-grasp fidelity proven",
            "close timing fully explains the replay-gap unless the rescue is material and near-complete",
        ],
        "do_not": [
            "Do not treat H1 alone as sufficient to reopen policy-gap Stage C.",
            "Do not generalize beyond the Gate A top-10 sample.",
            "Do not rewrite E030; H1 is a follow-up contraction audit.",
        ],
    }
    save_yaml(E031_YAML, draft)
    save_text(E031_MD, render_e031_markdown(draft))
    return draft, final_branch


def render_h1_markdown(artifact: dict[str, Any]) -> str:
    agg = artifact["aggregate"]
    return "\n".join(
        [
            "# Gate B7 Close-Timing Rescue",
            "",
            "- Proposal-only close-timing rescue audit.",
            f"- Attach threshold: `{artifact['attach_threshold_m']:.2f}`",
            f"- Baseline success rate: `{agg['baseline_success_rate']:.3f}`",
            f"- Rescue success rate: `{agg['rescue_success_rate']:.3f}`",
            f"- Success delta: `{agg['success_rate_delta']:.3f}`",
            f"- Baseline attach rate: `{agg['baseline_attach_rate']:.3f}`",
            f"- Rescue attach rate: `{agg['rescue_attach_rate']:.3f}`",
            f"- Attach delta: `{agg['attach_rate_delta']:.3f}`",
            f"- Rescued success count: `{agg['rescued_success_count']}`",
            f"- Mean suppressed close steps: `{agg['mean_suppressed_close_steps_per_episode']:.2f}`",
            f"- Final branch: `{agg['final_branch']}`",
            "",
        ]
    ) + "\n"


def render_e031_markdown(draft: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# E031 Draft",
            "",
            "Non-canonical close-timing rescue follow-up.",
            "",
            f"- Final branch: `{draft['final_branch']}`",
            f"- Baseline success rate: `{draft['key_findings']['baseline_success_rate']:.3f}`",
            f"- Rescue success rate: `{draft['key_findings']['rescue_success_rate']:.3f}`",
            f"- Success delta: `{draft['key_findings']['success_rate_delta']:.3f}`",
            f"- Attach delta: `{draft['key_findings']['attach_rate_delta']:.3f}`",
            f"- Rescued success count: `{draft['key_findings']['rescued_success_count']}`",
            "",
            "## Guardrails",
            "",
            *[f"- {item}" for item in draft["do_not"]],
            "",
        ]
    ) + "\n"


def main() -> None:
    e030_path = PROPOSALS_DIR / "E030_gate_b_replay_gap_contraction_draft.yaml"
    if not e030_path.exists():
        raise SystemExit("Stage A.5 outputs missing. Run replay-gap contraction before H1.")

    episode_set = _episode_set()
    ensure_dir(H1_DIR)
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = [
            executor.submit(
                _run_episode_pair,
                episode_index,
                npz_name,
                ATTACH_THRESHOLD,
                MAX_STEPS,
                IMAGE_SIZE,
                RENDER_OBSERVATIONS,
            )
            for episode_index, npz_name in episode_set
        ]
        bundles = [future.result() for future in concurrent.futures.as_completed(futures)]
    bundles.sort(key=lambda row: int(row["episode_index"]))
    h1 = build_h1(bundles)
    draft, final_branch = build_e031(h1)
    print(
        json.dumps(
            {
                "h1": str((H1_DIR / "gate_b7_summary.json").relative_to(CAMPAIGN_ROOT)),
                "e031": str(E031_YAML.relative_to(CAMPAIGN_ROOT)),
                "final_branch": final_branch,
                "stage_c_reconsider_allowed": bool(draft["stage_c_reconsider_allowed"]),
                "git_head": git_head(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
