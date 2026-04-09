#!/usr/bin/env python3
"""
Stage A.6 / H2b - Seed-vs-Retarget Discrepancy Audit
====================================================

Proposal-only follow-up after B5 and H2.

This audit explains why one-shot seeding can improve success while continuous
pre-handle retargeting improves attach but hurts end-to-end success.
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
from run_gate_b_replay_gap_contraction import run_seeded_open_loop  # noqa: E402
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

H2B_DIR = ARTIFACT_ROOT / "gate_b9_seed_retarget_discrepancy"
E033_YAML = PROPOSALS_DIR / "E033_gate_b_seed_retarget_discrepancy_draft.yaml"
E033_MD = PROPOSALS_DIR / "E033_gate_b_seed_retarget_discrepancy_draft.md"

SAMPLE_SCOPE = {
    "sample": "Gate A action-quality top-10",
    "mapping_target": "success-selected legacy rollout subset",
    "population_claim": "No population-wide claim over full 240 episodes",
}

MODES = (
    "open_loop",
    "seeded_initial_physical_pose",
    "seeded_initial_physical_and_command",
    "retarget_first_20_steps",
    "retarget_until_source_reference_step",
    "retarget_until_near_handle",
)


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


def _mode_definition(mode: str) -> dict[str, Any]:
    payloads = {
        "retarget_first_20_steps": {
            "name": mode,
            "description": "Apply pre-handle translation-only retargeting for the first 20 steps, then return fully to teacher actions.",
            "retarget_type": "finite_horizon_steps",
            "max_retarget_steps": 20,
            "retarget_disables_on_near_handle": False,
        },
        "retarget_until_source_reference_step": {
            "name": mode,
            "description": "Apply pre-handle translation-only retargeting only until the source episode's closest-to-handle reference step, then return fully to teacher actions.",
            "retarget_type": "finite_horizon_source_reference",
            "max_retarget_steps": None,
            "retarget_disables_on_near_handle": False,
        },
        "retarget_until_near_handle": {
            "name": mode,
            "description": "Apply translation-only retargeting until current-env handle proximity unlocks, then return fully to teacher actions.",
            "retarget_type": "continuous_until_near_handle",
            "max_retarget_steps": None,
            "retarget_disables_on_near_handle": True,
        },
    }
    return payloads[mode]


def _retarget_active(
    *,
    mode: str,
    step_index: int,
    source_reference_step: int,
    near_handle_unlocked: bool,
) -> bool:
    if mode == "retarget_first_20_steps":
        return step_index < 20
    if mode == "retarget_until_source_reference_step":
        return step_index <= source_reference_step
    if mode == "retarget_until_near_handle":
        return not near_handle_unlocked
    raise ValueError(mode)


def run_retarget_variant(
    *,
    episode_index: int,
    npz_name: str,
    mode: str,
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
        retarget_had_been_active = False
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
                actions[step_index]
                if step_index < len(actions)
                else np.zeros((7,), dtype=np.float32)
            )
            action = teacher_action.copy()

            active = _retarget_active(
                mode=mode,
                step_index=step_index,
                source_reference_step=source_reference_step,
                near_handle_unlocked=near_handle_unlocked,
            )
            if active:
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
                retarget_had_been_active = True
            elif retarget_had_been_active and first_retarget_disabled_step is None:
                first_retarget_disabled_step = int(step_index)

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
        replay_mode=mode,
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
    summary["applied_offset_norm_m"] = float(np.linalg.norm(offset_world))
    summary["retarget_active_steps"] = int(retarget_active_steps)
    summary["first_retarget_near_handle_step"] = first_near_handle_step
    summary["first_retarget_disabled_step"] = first_retarget_disabled_step
    summary["retarget_applied_any"] = bool(retarget_applied_any)
    summary["measurement_image_size_px"] = int(image_size)
    summary["render_observations"] = bool(render_observations)
    summary["retarget_definition"] = _mode_definition(mode)
    return summary


def _pair_episode(
    episode_index: int,
    npz_name: str,
    attach_threshold: float,
    max_steps: int,
    image_size: int,
    render_observations: bool,
) -> list[dict[str, Any]]:
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
    seeded_pose = run_seeded_open_loop(
        episode_index=episode_index,
        npz_name=npz_name,
        seed_mode="open_loop_seeded_initial_physical_pose",
        attach_threshold=attach_threshold,
        max_steps=max_steps,
        image_size=image_size,
        render_observations=render_observations,
    )
    seeded_pose_and_command = run_seeded_open_loop(
        episode_index=episode_index,
        npz_name=npz_name,
        seed_mode="open_loop_seeded_initial_physical_and_command",
        attach_threshold=attach_threshold,
        max_steps=max_steps,
        image_size=image_size,
        render_observations=render_observations,
    )
    retarget_first_20 = run_retarget_variant(
        episode_index=episode_index,
        npz_name=npz_name,
        mode="retarget_first_20_steps",
        attach_threshold=attach_threshold,
        max_steps=max_steps,
        image_size=image_size,
        render_observations=render_observations,
    )
    retarget_until_reference = run_retarget_variant(
        episode_index=episode_index,
        npz_name=npz_name,
        mode="retarget_until_source_reference_step",
        attach_threshold=attach_threshold,
        max_steps=max_steps,
        image_size=image_size,
        render_observations=render_observations,
    )
    retarget_until_near = run_retarget_variant(
        episode_index=episode_index,
        npz_name=npz_name,
        mode="retarget_until_near_handle",
        attach_threshold=attach_threshold,
        max_steps=max_steps,
        image_size=image_size,
        render_observations=render_observations,
    )
    return [
        baseline,
        seeded_pose,
        seeded_pose_and_command,
        retarget_first_20,
        retarget_until_reference,
        retarget_until_near,
    ]


def _aggregate_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(len(rows), 1)
    failure_counts = Counter(str(row["failure_stage"]) for row in rows)
    payload = {
        "attach_rate": float(sum(1 for row in rows if row["ever_attached"]) / n),
        "success_rate": float(sum(1 for row in rows if row["success"]) / n),
        "dominant_failure_stage": (
            max(failure_counts.items(), key=lambda item: item[1])[0]
            if failure_counts
            else None
        ),
        "failure_stage_counts": dict(failure_counts),
        "mean_min_handle_distance_m": float(
            np.mean([float(row["min_handle_distance_m"]) for row in rows]) if rows else 0.0
        ),
    }
    retarget_rows = [row for row in rows if "retarget_active_steps" in row]
    if retarget_rows:
        payload["mean_retarget_active_steps"] = float(
            np.mean([float(row["retarget_active_steps"]) for row in retarget_rows])
        )
        payload["mean_applied_offset_norm_m"] = float(
            np.mean([float(row["applied_offset_norm_m"]) for row in retarget_rows])
        )
    return payload


def _best_mode(mode_payloads: dict[str, dict[str, Any]], candidates: list[str]) -> str:
    return max(
        candidates,
        key=lambda name: (
            float(mode_payloads[name]["success_rate"]),
            float(mode_payloads[name]["attach_rate"]),
            -float(mode_payloads[name].get("mean_retarget_active_steps", 0.0)),
        ),
    )


def build_audit() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    per_episode_rows: list[dict[str, Any]] = []
    jobs = _episode_set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [
            executor.submit(
                _pair_episode,
                episode_index,
                npz_name,
                ATTACH_THRESHOLD,
                MAX_STEPS,
                IMAGE_SIZE,
                RENDER_OBSERVATIONS,
            )
            for episode_index, npz_name in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            rows.extend(future.result())

    rows.sort(key=lambda row: (int(row["episode_index"]), str(row["replay_mode"])))

    mode_rows = {
        "open_loop": [row for row in rows if row["replay_mode"] == "open_loop"],
        "seeded_initial_physical_pose": [
            row for row in rows if row["replay_mode"] == "open_loop_seeded_initial_physical_pose"
        ],
        "seeded_initial_physical_and_command": [
            row for row in rows if row["replay_mode"] == "open_loop_seeded_initial_physical_and_command"
        ],
        "retarget_first_20_steps": [
            row for row in rows if row["replay_mode"] == "retarget_first_20_steps"
        ],
        "retarget_until_source_reference_step": [
            row for row in rows if row["replay_mode"] == "retarget_until_source_reference_step"
        ],
        "retarget_until_near_handle": [
            row for row in rows if row["replay_mode"] == "retarget_until_near_handle"
        ],
    }
    mode_payloads = {mode: _aggregate_mode(mode_rows[mode]) for mode in MODES}
    best_seed_mode = _best_mode(
        mode_payloads,
        ["seeded_initial_physical_pose", "seeded_initial_physical_and_command"],
    )
    best_retarget_mode = _best_mode(
        mode_payloads,
        [
            "retarget_first_20_steps",
            "retarget_until_source_reference_step",
            "retarget_until_near_handle",
        ],
    )

    baseline_success = float(mode_payloads["open_loop"]["success_rate"])
    best_seed_success = float(mode_payloads[best_seed_mode]["success_rate"])
    best_retarget_success = float(mode_payloads[best_retarget_mode]["success_rate"])
    continuous_success = float(mode_payloads["retarget_until_near_handle"]["success_rate"])
    finite_best_success = max(
        float(mode_payloads["retarget_first_20_steps"]["success_rate"]),
        float(mode_payloads["retarget_until_source_reference_step"]["success_rate"]),
    )

    if best_seed_success > best_retarget_success:
        final_branch = (
            "one-shot seeding outperforms retargeting; the seed-vs-retarget discrepancy is "
            "consistent with continuous pre-handle intervention being too intrusive"
        )
    elif finite_best_success > continuous_success:
        final_branch = (
            "finite-horizon retargeting outperforms continuous retargeting; the discrepancy is "
            "consistent with overlong pre-handle retarget distortion"
        )
    else:
        final_branch = (
            "seeding and retargeting both show mixed trade-offs; the discrepancy remains multi-factor "
            "within approach/handle-entry behavior"
        )

    for episode_index, npz_name in jobs:
        episode_payload = {
            "episode_index": int(episode_index),
            "npz_name": str(npz_name),
        }
        baseline_row = next(
            row for row in mode_rows["open_loop"] if int(row["episode_index"]) == int(episode_index)
        )
        episode_payload.update(
            {
                "baseline_success": bool(baseline_row["success"]),
                "baseline_attach": bool(baseline_row["ever_attached"]),
                "baseline_failure_stage": str(baseline_row["failure_stage"]),
                "baseline_min_handle_distance_m": float(baseline_row["min_handle_distance_m"]),
            }
        )
        for mode in MODES[1:]:
            row = next(
                row for row in mode_rows[mode] if int(row["episode_index"]) == int(episode_index)
            )
            prefix = mode
            episode_payload[f"{prefix}_success"] = bool(row["success"])
            episode_payload[f"{prefix}_attach"] = bool(row["ever_attached"])
            episode_payload[f"{prefix}_failure_stage"] = str(row["failure_stage"])
            episode_payload[f"{prefix}_min_handle_distance_m"] = float(row["min_handle_distance_m"])
            if "retarget_active_steps" in row:
                episode_payload[f"{prefix}_retarget_active_steps"] = int(row["retarget_active_steps"])
        per_episode_rows.append(episode_payload)

    aggregate = {
        "modes": mode_payloads,
        "best_seed_mode": best_seed_mode,
        "best_seed_success_rate": best_seed_success,
        "best_retarget_mode": best_retarget_mode,
        "best_retarget_success_rate": best_retarget_success,
        "seeded_command_alignment_success_delta": (
            float(mode_payloads["seeded_initial_physical_and_command"]["success_rate"])
            - float(mode_payloads["seeded_initial_physical_pose"]["success_rate"])
        ),
        "finite_horizon_best_success_rate": finite_best_success,
        "continuous_retarget_success_rate": continuous_success,
        "continuous_minus_best_seed_success_delta": continuous_success - best_seed_success,
        "continuous_minus_finite_best_success_delta": continuous_success - finite_best_success,
        "final_branch": final_branch,
    }

    return {
        "experiment": "gate_b9_seed_retarget_discrepancy",
        "timestamp": now_ts(),
        "sample_scope": SAMPLE_SCOPE,
        "attach_threshold_m": ATTACH_THRESHOLD,
        "measurement_image_size_px": IMAGE_SIZE,
        "render_observations": RENDER_OBSERVATIONS,
        "episodes": per_episode_rows,
        "aggregate": aggregate,
        "git": {"head": git_head()},
        "inputs": {
            "top10_episodes": TOP10_EPISODES,
            "modes": list(MODES),
            "source_artifacts": {
                "b5": str((ARTIFACT_ROOT / "gate_b5_initial_state_seed" / "gate_b5_summary.json")),
                "b8": str((ARTIFACT_ROOT / "gate_b8_handle_offset_retargeting" / "gate_b8_summary.json")),
            },
        },
    }


def build_markdown(summary: dict[str, Any]) -> str:
    agg = summary["aggregate"]
    lines = [
        "# E033 - Gate B Seed-vs-Retarget Discrepancy Audit",
        "",
        f"- timestamp: {summary['timestamp']}",
        f"- attach_threshold_m: {summary['attach_threshold_m']}",
        f"- sample: {summary['sample_scope']['sample']}",
        f"- mapping_target: {summary['sample_scope']['mapping_target']}",
        f"- population_claim: {summary['sample_scope']['population_claim']}",
        "",
        "## Mode Summary",
    ]
    for mode, payload in agg["modes"].items():
        lines.append(
            f"- {mode}: attach={payload['attach_rate']:.3f}, success={payload['success_rate']:.3f}, dominant_failure={payload['dominant_failure_stage']}"
        )
    lines.extend(
        [
            "",
            "## Key Findings",
            f"- best_seed_mode: {agg['best_seed_mode']} ({agg['best_seed_success_rate']:.3f} success)",
            f"- best_retarget_mode: {agg['best_retarget_mode']} ({agg['best_retarget_success_rate']:.3f} success)",
            f"- seeded_command_alignment_success_delta: {agg['seeded_command_alignment_success_delta']:.3f}",
            f"- continuous_minus_best_seed_success_delta: {agg['continuous_minus_best_seed_success_delta']:.3f}",
            f"- continuous_minus_finite_best_success_delta: {agg['continuous_minus_finite_best_success_delta']:.3f}",
            "",
            "## Final Branch",
            f"- {agg['final_branch']}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_e033(summary: dict[str, Any]) -> dict[str, Any]:
    agg = summary["aggregate"]
    return {
        "proposal_id": "E033_gate_b_seed_retarget_discrepancy",
        "status": "proposal_only",
        "title": "Gate B H2b seed-vs-retarget discrepancy audit",
        "sample_scope": summary["sample_scope"],
        "inputs": summary["inputs"],
        "result": {
            "best_seed_mode": agg["best_seed_mode"],
            "best_seed_success_rate": agg["best_seed_success_rate"],
            "best_retarget_mode": agg["best_retarget_mode"],
            "best_retarget_success_rate": agg["best_retarget_success_rate"],
            "seeded_command_alignment_success_delta": agg["seeded_command_alignment_success_delta"],
            "continuous_minus_best_seed_success_delta": agg["continuous_minus_best_seed_success_delta"],
            "continuous_minus_finite_best_success_delta": agg["continuous_minus_finite_best_success_delta"],
            "final_branch": agg["final_branch"],
        },
        "allowed_wording": [
            "One-shot seeding and retargeting interventions produce different trade-offs within approach/handle-entry behavior.",
            "Continuous pre-handle retargeting should be treated as a strong intervention, not a clean geometry-alignment fix.",
        ],
        "disallowed_wording": [
            "constant handle offset fixes the replay-gap",
            "coordinate-system bug proven",
            "policy-only gap established",
            "root cause solved",
        ],
    }


def main() -> None:
    ensure_dir(H2B_DIR)
    summary = build_audit()
    save_json(H2B_DIR / "gate_b9_summary.json", summary)
    save_csv(H2B_DIR / "gate_b9_episodes.csv", summary["episodes"])
    md = build_markdown(summary)
    save_text(H2B_DIR / "gate_b9_summary.md", md)
    e033 = build_e033(summary)
    save_yaml(E033_YAML, e033)
    save_text(E033_MD, md)
    print(json.dumps(to_builtin(summary), indent=2, ensure_ascii=False))
    print(f"\n[gate_b9] wrote: {H2B_DIR / 'gate_b9_summary.json'}")


if __name__ == "__main__":
    main()
