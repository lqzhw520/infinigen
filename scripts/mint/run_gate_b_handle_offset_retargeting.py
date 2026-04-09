#!/usr/bin/env python3
"""
Stage A.6 / H2 - Handle-Offset Retargeting Audit
================================================

Proposal-only follow-up after corrected Stage A, Stage A.5, and H1.

Tests whether a constant world-frame pre-handle retargeting offset materially
contracts the replay-gap. The offset is estimated by aligning the source
closest-to-handle EEF pose to the current env handle center under current-env
handle semantics.
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

from drawer_robot_env import DrawerRobotEnv, _action_toward_pose  # noqa: E402
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
H2_DIR = ARTIFACT_ROOT / "gate_b8_handle_offset_retargeting"
E032_YAML = PROPOSALS_DIR / "E032_gate_b_handle_offset_retargeting_draft.yaml"
E032_MD = PROPOSALS_DIR / "E032_gate_b_handle_offset_retargeting_draft.md"

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


def _rate(rows: list[dict[str, Any]], key: str, metric: str) -> float:
    n = max(len(rows), 1)
    if metric == "success":
        return float(sum(1 for row in rows if row[key]["success"]) / n)
    if metric == "attach":
        return float(sum(1 for row in rows if row[key]["ever_attached"]) / n)
    raise ValueError(metric)


def _safe_argmin(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    finite = np.isfinite(values)
    if not np.any(finite):
        return 0
    return int(np.argmin(np.where(finite, values, np.inf)))


def _source_reference_step(source_npz: dict[str, Any]) -> tuple[int, float]:
    source_handle_distance = np.asarray(
        source_npz["handle_distance_trace"], dtype=np.float32
    )
    step = _safe_argmin(source_handle_distance)
    if len(source_handle_distance) == 0:
        return 0, float("nan")
    return step, float(source_handle_distance[min(step, len(source_handle_distance) - 1)])


def run_handle_offset_retargeting(
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
    source_eef_positions = np.asarray(source_npz["eef_positions"], dtype=np.float32)
    source_reference_step, source_reference_distance = _source_reference_step(source_npz)
    if len(source_eef_positions):
        source_reference_eef_pos = source_eef_positions[
            min(source_reference_step, len(source_eef_positions) - 1)
        ].astype(np.float32)
    else:
        source_reference_eef_pos = np.zeros((3,), dtype=np.float32)

    eef_positions: list[np.ndarray] = []
    drawer_trace: list[float] = []
    attached_trace: list[bool] = []
    handle_distance_trace: list[float] = []
    success = False
    attach_step = None
    retarget_active_steps = 0
    first_near_handle_step = None
    first_retarget_disabled_step = None
    retarget_applied_any = False

    try:
        obs = env.reset()
        local_handle, handle_metadata = current_env_handle_local(env)
        env.attachment_local = local_handle
        handle_center_world = np.asarray(
            handle_metadata["handle_center_world"], dtype=np.float32
        )
        offset_world = (handle_center_world - source_reference_eef_pos).astype(np.float32)

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
                if first_retarget_disabled_step is None:
                    first_retarget_disabled_step = int(step_index)

            teacher_action = (
                actions[step_index]
                if step_index < len(actions)
                else np.zeros((7,), dtype=np.float32)
            )
            action = teacher_action.copy()

            if not near_handle_unlocked:
                source_step = min(step_index, max(len(source_eef_positions) - 1, 0))
                if len(source_eef_positions):
                    retarget_pos = (
                        source_eef_positions[source_step].astype(np.float32) + offset_world
                    )
                else:
                    retarget_pos = handle_center_world.copy()
                retarget_action = _action_toward_pose(
                    obs,
                    retarget_pos,
                    obs.eef_quat.copy(),
                    gripper_open=1.0 if teacher_action[6] >= 0.0 else 0.0,
                    translation_scale=env.translation_scale,
                    rotation_scale=env.rotation_scale,
                    action_frame=env.action_frame,
                )
                action[:3] = retarget_action[:3]
                retarget_active_steps += 1
                retarget_applied_any = True

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
        replay_mode="handle_offset_retarget",
        eef_positions=np.asarray(eef_positions, dtype=np.float32),
        drawer_trace=np.asarray(drawer_trace, dtype=np.float32),
        attached_trace=np.asarray(attached_trace, dtype=np.bool_),
        handle_distance_trace=np.asarray(handle_distance_trace, dtype=np.float32),
        success=success,
        attach_step=attach_step,
        action_contract=action_contract,
        frames=[],
    )
    summary["source_reference_step"] = int(source_reference_step)
    summary["source_reference_min_handle_distance_m"] = float(source_reference_distance)
    summary["source_reference_eef_pos"] = source_reference_eef_pos
    summary["current_env_handle_center_world"] = handle_center_world
    summary["applied_offset_world_m"] = offset_world
    summary["applied_offset_norm_m"] = float(np.linalg.norm(offset_world))
    summary["retarget_active_steps"] = int(retarget_active_steps)
    summary["first_retarget_near_handle_step"] = first_near_handle_step
    summary["first_retarget_disabled_step"] = first_retarget_disabled_step
    summary["retarget_applied_any"] = bool(retarget_applied_any)
    summary["measurement_image_size_px"] = int(image_size)
    summary["render_observations"] = bool(render_observations)
    summary["retarget_definition"] = {
        "name": "handle_offset_retarget",
        "description": (
            "Teacher rotation and gripper channels are preserved. During pre-handle "
            "steps, the translation channel is recomputed toward source EEF positions "
            "shifted by a constant world-frame offset that aligns the source "
            "closest-to-handle EEF pose to the current-env handle center."
        ),
        "constant_world_offset": True,
        "offset_reference": "source_closest_to_handle_eef_pose_to_current_env_handle_center",
        "teacher_translation_modified": True,
        "teacher_rotation_preserved": True,
        "teacher_gripper_preserved": True,
        "pre_handle_only": True,
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
    retarget = run_handle_offset_retargeting(
        episode_index=episode_index,
        npz_name=npz_name,
        attach_threshold=attach_threshold,
        max_steps=max_steps,
        image_size=image_size,
        render_observations=render_observations,
    )
    return {
        "episode_index": int(episode_index),
        "npz_name": npz_name,
        "npz_path": str(npz_path),
        "baseline": baseline,
        "retarget": retarget,
    }


def build_h2(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    transition_counts = Counter()
    rescued_success_count = 0
    rescued_attach_count = 0
    for row in bundles:
        baseline = row["baseline"]
        retarget = row["retarget"]
        transition_counts[f"{baseline['failure_stage']} -> {retarget['failure_stage']}"] += 1
        if not baseline["success"] and retarget["success"]:
            rescued_success_count += 1
        if not baseline["ever_attached"] and retarget["ever_attached"]:
            rescued_attach_count += 1
        rows.append(
            {
                "episode_index": row["episode_index"],
                "npz_name": row["npz_name"],
                "baseline_success": baseline["success"],
                "baseline_ever_attached": baseline["ever_attached"],
                "baseline_failure_stage": baseline["failure_stage"],
                "baseline_first_near_handle_step": baseline["first_near_handle_step"],
                "baseline_attach_step": baseline["attach_step"],
                "baseline_min_handle_distance_m": baseline["min_handle_distance_m"],
                "baseline_max_drawer_fraction": baseline["max_drawer_fraction"],
                "retarget_success": retarget["success"],
                "retarget_ever_attached": retarget["ever_attached"],
                "retarget_failure_stage": retarget["failure_stage"],
                "retarget_first_near_handle_step": retarget["first_near_handle_step"],
                "retarget_attach_step": retarget["attach_step"],
                "retarget_min_handle_distance_m": retarget["min_handle_distance_m"],
                "retarget_max_drawer_fraction": retarget["max_drawer_fraction"],
                "source_reference_step": retarget["source_reference_step"],
                "source_reference_min_handle_distance_m": retarget[
                    "source_reference_min_handle_distance_m"
                ],
                "applied_offset_norm_m": retarget["applied_offset_norm_m"],
                "retarget_active_steps": retarget["retarget_active_steps"],
                "first_retarget_near_handle_step": retarget["first_retarget_near_handle_step"],
                "first_retarget_disabled_step": retarget["first_retarget_disabled_step"],
                "success_rescued": bool((not baseline["success"]) and retarget["success"]),
                "attach_rescued": bool(
                    (not baseline["ever_attached"]) and retarget["ever_attached"]
                ),
            }
        )

    baseline_attach = _rate(bundles, "baseline", "attach")
    baseline_success = _rate(bundles, "baseline", "success")
    retarget_attach = _rate(bundles, "retarget", "attach")
    retarget_success = _rate(bundles, "retarget", "success")
    attach_delta = float(retarget_attach - baseline_attach)
    success_delta = float(retarget_success - baseline_success)
    total_active_steps = int(sum(row["retarget_active_steps"] for row in rows))
    mean_active_steps = (
        float(np.mean([row["retarget_active_steps"] for row in rows])) if rows else 0.0
    )
    mean_offset_norm = (
        float(np.mean([row["applied_offset_norm_m"] for row in rows])) if rows else 0.0
    )

    if success_delta > 0.20:
        final_branch = "handle-offset retargeting materially contracts the replay-gap"
    elif success_delta > 0.0 or rescued_success_count > 0:
        final_branch = "handle-offset retargeting improves end-to-end execution but is not sufficient alone"
    elif attach_delta > 0.10 or rescued_attach_count > 0:
        final_branch = "handle-offset retargeting improves pre-handle/attach entry but does not improve end-to-end success"
    else:
        final_branch = "handle-offset retargeting alone does not materially contract the replay-gap"

    artifact = {
        "experiment": "gate_b8_handle_offset_retargeting",
        "timestamp": now_ts(),
        "sample_scope": dict(SAMPLE_SCOPE),
        "attach_threshold_m": float(ATTACH_THRESHOLD),
        "measurement_image_size_px": int(IMAGE_SIZE),
        "render_observations": bool(RENDER_OBSERVATIONS),
        "episodes": rows,
        "aggregate": {
            "baseline_attach_rate": baseline_attach,
            "baseline_success_rate": baseline_success,
            "retarget_attach_rate": retarget_attach,
            "retarget_success_rate": retarget_success,
            "attach_rate_delta": attach_delta,
            "success_rate_delta": success_delta,
            "rescued_success_count": int(rescued_success_count),
            "rescued_attach_count": int(rescued_attach_count),
            "total_retarget_active_steps": total_active_steps,
            "mean_retarget_active_steps_per_episode": mean_active_steps,
            "mean_applied_offset_norm_m": mean_offset_norm,
            "failure_stage_transitions": dict(transition_counts),
            "final_branch": final_branch,
        },
    }
    save_json(H2_DIR / "gate_b8_summary.json", artifact)
    save_csv(H2_DIR / "gate_b8_episodes.csv", rows)
    save_text(H2_DIR / "gate_b8_summary.md", render_h2_markdown(artifact))
    return artifact


def build_e032(h2: dict[str, Any]) -> tuple[dict[str, Any], str]:
    agg = h2["aggregate"]
    final_branch = str(agg["final_branch"])
    draft = {
        "draft_id": "E032_draft",
        "draft_type": "proposal_only_followup",
        "target_evidence_id": "E032",
        "experiment_id": "gate_b_handle_offset_retargeting",
        "timestamp": now_ts(),
        "scope": "gate_b_top10_handle_offset_retargeting",
        "verified": False,
        "sample_scope": dict(SAMPLE_SCOPE),
        "summary": (
            "Stage A.6 / H2 tests whether a constant world-frame pre-handle retargeting "
            "offset materially contracts the replay-gap while preserving teacher rotation "
            "and gripper channels."
        ),
        "source_artifacts": {
            "e029": "sovereign/proposals/E029_gate_b_corrected_deconfounded_draft.yaml",
            "e030": "sovereign/proposals/E030_gate_b_replay_gap_contraction_draft.yaml",
            "e031": "sovereign/proposals/E031_gate_b_close_timing_rescue_draft.yaml",
            "h2": "artifacts/gate_b8_handle_offset_retargeting/gate_b8_summary.json",
        },
        "measurement_runtime_note": (
            f"H2 used image_size={IMAGE_SIZE} and render_observations={RENDER_OBSERVATIONS} "
            "for state-only measurement throughput; physics, task semantics, and action contracts were unchanged."
        ),
        "key_findings": {
            "baseline_success_rate": float(agg["baseline_success_rate"]),
            "retarget_success_rate": float(agg["retarget_success_rate"]),
            "success_rate_delta": float(agg["success_rate_delta"]),
            "baseline_attach_rate": float(agg["baseline_attach_rate"]),
            "retarget_attach_rate": float(agg["retarget_attach_rate"]),
            "attach_rate_delta": float(agg["attach_rate_delta"]),
            "rescued_success_count": int(agg["rescued_success_count"]),
            "rescued_attach_count": int(agg["rescued_attach_count"]),
            "mean_retarget_active_steps_per_episode": float(
                agg["mean_retarget_active_steps_per_episode"]
            ),
            "mean_applied_offset_norm_m": float(agg["mean_applied_offset_norm_m"]),
        },
        "final_branch": final_branch,
        "stage_c_reconsider_allowed": False,
        "allowed_wording": [
            "Handle-offset retargeting can be discussed as a replay-gap contraction factor under canonical current-env semantics.",
            "Teacher rotation and gripper channels stayed fixed while only pre-handle translation was retargeted.",
        ],
        "disallowed_wording": [
            "policy-only gap proven",
            "root cause solved",
            "source-grasp fidelity proven",
            "handle-offset retargeting fully explains the replay-gap unless the rescue is material and near-complete",
        ],
        "do_not": [
            "Do not treat H2 as a source-grasp fidelity proof.",
            "Do not generalize beyond the Gate A top-10 sample.",
            "Do not reopen policy-gap Stage C from H2 alone.",
        ],
    }
    save_yaml(E032_YAML, draft)
    save_text(E032_MD, render_e032_markdown(draft))
    return draft, final_branch


def render_h2_markdown(artifact: dict[str, Any]) -> str:
    agg = artifact["aggregate"]
    return "\n".join(
        [
            "# Gate B8 Handle-Offset Retargeting",
            "",
            "- Proposal-only handle-offset retargeting audit.",
            f"- Attach threshold: `{artifact['attach_threshold_m']:.2f}`",
            f"- Baseline success rate: `{agg['baseline_success_rate']:.3f}`",
            f"- Retarget success rate: `{agg['retarget_success_rate']:.3f}`",
            f"- Success delta: `{agg['success_rate_delta']:.3f}`",
            f"- Baseline attach rate: `{agg['baseline_attach_rate']:.3f}`",
            f"- Retarget attach rate: `{agg['retarget_attach_rate']:.3f}`",
            f"- Attach delta: `{agg['attach_rate_delta']:.3f}`",
            f"- Rescued success count: `{agg['rescued_success_count']}`",
            f"- Mean retarget active steps: `{agg['mean_retarget_active_steps_per_episode']:.2f}`",
            f"- Mean applied offset norm (m): `{agg['mean_applied_offset_norm_m']:.4f}`",
            f"- Final branch: `{agg['final_branch']}`",
            "",
        ]
    ) + "\n"


def render_e032_markdown(draft: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# E032 Draft",
            "",
            "Non-canonical handle-offset retargeting follow-up.",
            "",
            f"- Final branch: `{draft['final_branch']}`",
            f"- Baseline success rate: `{draft['key_findings']['baseline_success_rate']:.3f}`",
            f"- Retarget success rate: `{draft['key_findings']['retarget_success_rate']:.3f}`",
            f"- Success delta: `{draft['key_findings']['success_rate_delta']:.3f}`",
            f"- Attach delta: `{draft['key_findings']['attach_rate_delta']:.3f}`",
            f"- Rescued success count: `{draft['key_findings']['rescued_success_count']}`",
            f"- Mean applied offset norm (m): `{draft['key_findings']['mean_applied_offset_norm_m']:.4f}`",
            "",
            "## Guardrails",
            "",
            *[f"- {item}" for item in draft["do_not"]],
            "",
        ]
    ) + "\n"


def main() -> None:
    e031_path = PROPOSALS_DIR / "E031_gate_b_close_timing_rescue_draft.yaml"
    if not e031_path.exists():
        raise SystemExit("Stage A.6 / H1 outputs missing. Run close-timing rescue before H2.")

    episode_set = _episode_set()
    ensure_dir(H2_DIR)
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
    h2 = build_h2(bundles)
    draft, final_branch = build_e032(h2)
    print(
        json.dumps(
            {
                "h2": str((H2_DIR / "gate_b8_summary.json").relative_to(CAMPAIGN_ROOT)),
                "e032": str(E032_YAML.relative_to(CAMPAIGN_ROOT)),
                "final_branch": final_branch,
                "stage_c_reconsider_allowed": bool(draft["stage_c_reconsider_allowed"]),
                "git_head": git_head(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
