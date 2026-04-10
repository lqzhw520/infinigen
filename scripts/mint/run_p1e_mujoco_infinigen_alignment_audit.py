#!/usr/bin/env python3
"""Audit alignment gaps between the official LIBERO control surface and the current Infinigen MuJoCo dataset."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

import pyarrow.parquet as pq

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN_DIR = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1"
ARTIFACT_DIR = CAMPAIGN_DIR / "artifacts"
OUTPUT_DIR = CAMPAIGN_DIR / "outputs"

CONTROL_DATASET = PROJECT_ROOT / "external" / "LIBERO" / "datasets" / "libero_10_image"
TARGET_DATASET = CAMPAIGN_DIR / "dataset"
CONTROL_BASELINE = ARTIFACT_DIR / "p1c11_official_libero_goal_drawer_baseline_rollback_4eab579.json"
CURRENT_NIGHT = ARTIFACT_DIR / "mujoco_infinigen_mainline_night.json"
M4_PATH = ARTIFACT_DIR / "m4_robot_rollout_gate.json"
M5_PATH = ARTIFACT_DIR / "m5_dataset_pack_gate.json"
M7_PATH = ARTIFACT_DIR / "m7_mint_eval_gate.json"
RUNNER_PATH = PROJECT_ROOT / "scripts" / "mint" / "run_mujoco_infinigen_mainline_night.py"
ENV_PATH = PROJECT_ROOT / "scripts" / "mint" / "drawer_robot_env_mujoco.py"

AUDIT_JSON = ARTIFACT_DIR / "p1e_mujoco_infinigen_alignment_audit.json"
AUDIT_MD = OUTPUT_DIR / "p1e_mujoco_infinigen_alignment_audit.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_tasks(path: Path) -> list[str]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    out = []
    for row in rows:
        if "task" in row:
            out.append(str(row["task"]))
        elif "__index_level_0__" in row:
            out.append(str(row["__index_level_0__"]))
    return out


def channel_stats(stats: dict[str, Any], key: str) -> dict[str, list[float]]:
    entry = stats[key]
    return {
        "mean": [float(entry["mean"][c][0][0]) for c in range(3)],
        "std": [float(entry["std"][c][0][0]) for c in range(3)],
    }


def vector_stats(stats: dict[str, Any], key: str) -> dict[str, list[float]]:
    entry = stats[key]
    return {
        "mean": [float(x) for x in entry["mean"]],
        "std": [float(x) for x in entry["std"]],
        "min": [float(x) for x in entry["min"]],
        "max": [float(x) for x in entry["max"]],
    }


def gap_grade(score: float) -> str:
    if score < 0.15:
        return "low"
    if score < 0.5:
        return "medium"
    return "high"


def build_audit() -> dict[str, Any]:
    control_info = load_json(CONTROL_DATASET / "meta" / "info.json")
    control_stats = load_json(CONTROL_DATASET / "meta" / "stats.json")
    target_info = load_json(TARGET_DATASET / "meta" / "info.json")
    target_stats = load_json(TARGET_DATASET / "meta" / "stats.json")
    control_tasks = load_tasks(CONTROL_DATASET / "meta" / "tasks.parquet")
    target_tasks = load_tasks(TARGET_DATASET / "meta" / "tasks.parquet")

    control_image = channel_stats(control_stats, "observation.images.image")
    target_image = channel_stats(target_stats, "observation.images.image")
    control_wrist = channel_stats(control_stats, "observation.images.wrist_image")
    target_wrist = channel_stats(target_stats, "observation.images.image2")
    control_state = vector_stats(control_stats, "observation.state")
    target_state = vector_stats(target_stats, "observation.state")
    control_action = vector_stats(control_stats, "action")
    target_action = vector_stats(target_stats, "action")

    m4 = load_json(M4_PATH)
    m5 = load_json(M5_PATH)
    m7 = load_json(M7_PATH)

    image_mean_gap = sum(abs(a - b) for a, b in zip(control_image["mean"], target_image["mean"])) / 3.0
    image_std_gap = sum(abs(a - b) for a, b in zip(control_image["std"], target_image["std"])) / 3.0
    wrist_mean_gap = sum(abs(a - b) for a, b in zip(control_wrist["mean"], target_wrist["mean"])) / 3.0
    state_std_gap = mean(abs(a - b) for a, b in zip(control_state["std"], target_state["std"]))
    action_std_gap = mean(abs(a - b) for a, b in zip(control_action["std"], target_action["std"]))

    findings = []
    findings.append(
        {
            "id": "image_brightness_gap",
            "severity": gap_grade(image_mean_gap + image_std_gap),
            "summary": "Primary camera images are much darker and lower-variance than the official control dataset.",
            "control_mean_rgb": control_image["mean"],
            "target_mean_rgb": target_image["mean"],
            "control_std_rgb": control_image["std"],
            "target_std_rgb": target_image["std"],
        }
    )
    findings.append(
        {
            "id": "wrist_camera_gap",
            "severity": gap_grade(wrist_mean_gap),
            "summary": "Secondary camera images are also much darker than the official wrist-image distribution.",
            "control_mean_rgb": control_wrist["mean"],
            "target_mean_rgb": target_wrist["mean"],
        }
    )
    findings.append(
        {
            "id": "task_text_gap",
            "severity": "high",
            "summary": "The current dataset task text is generic and does not match the official control drawer phrasing.",
            "control_drawer_tasks": [task for task in control_tasks if "drawer" in task or "cabinet" in task][:5],
            "target_tasks": target_tasks,
        }
    )
    findings.append(
        {
            "id": "state_variation_gap",
            "severity": gap_grade(state_std_gap),
            "summary": "State dimensions 3:7 are nearly constant in the Infinigen dataset, unlike the official control distribution.",
            "control_state_std": control_state["std"],
            "target_state_std": target_state["std"],
        }
    )
    findings.append(
        {
            "id": "action_degeneracy_gap",
            "severity": gap_grade(action_std_gap),
            "summary": "Rollout actions are far more degenerate than the official control distribution, especially rotation channels.",
            "control_action_std": control_action["std"],
            "target_action_std": target_action["std"],
            "control_action_mean": control_action["mean"],
            "target_action_mean": target_action["mean"],
        }
    )
    findings.append(
        {
            "id": "rollout_quality_gap",
            "severity": "high",
            "summary": "Current robot rollout source quality is still low; the first pilot copied more learning files than truly successful rollouts.",
            "anygrasp_successful_seeds": m4["anygrasp_successful_seeds"],
            "oracle_successful_seeds": m4["oracle_successful_seeds"],
            "copied_learning_files": m4["copied_learning_files"],
            "successful_oracle_rollouts": sum(1 for item in m4["oracle_records"] if item["passed"]),
        }
    )
    findings.append(
        {
            "id": "dataset_size_gap",
            "severity": "high",
            "summary": "The current training dataset is extremely small for adapting a 3B policy.",
            "target_episode_count": m5["episode_count"],
            "target_frame_count": m5["frame_count"],
        }
    )
    findings.append(
        {
            "id": "pilot_outcome",
            "severity": "high",
            "summary": "The first MuJoCo pilot completed end-to-end but produced no held-out improvement.",
            "pretrained_success_count": m7["pretrained"]["success_count"],
            "finetuned_success_count": m7["finetuned"]["success_count"],
            "heldout_seed_count": m7["pretrained"]["seed_count"],
        }
    )

    recommendations = [
        {
            "priority": "P0",
            "action": "Filter the learning set to successful oracle/AnyGrasp rollouts only and regenerate the dataset.",
        },
        {
            "priority": "P0",
            "action": "Expand train rollout generation from seeds 1-6 to the full train pool 1-10 and keep only successful episodes.",
        },
        {
            "priority": "P0",
            "action": "Align the task text to the official drawer phrasing instead of the generic 'open the drawer'.",
        },
        {
            "priority": "P0",
            "action": "Calibrate MuJoCo-rendered images toward the official control brightness/contrast distribution before training and eval.",
        },
        {
            "priority": "P1",
            "action": "Increase overnight training steps substantially beyond the 200-step pilot once the learning set is filtered and expanded.",
        },
        {
            "priority": "P1",
            "action": "Treat the current 0/5 result as a pilot infrastructure result, not as a scientific failure of the MuJoCo/Infinigen line.",
        },
    ]

    remediation_plan = {
        "control": {
            "benchmark_ref": "E036/p1c11 official LIBERO drawer baseline",
            "mujoco_control_task": "open_the_middle_drawer_of_the_cabinet",
        },
        "target_fixes": [
            {
                "id": "fix_task_phrase",
                "why": "Generic task wording weakens task-conditioning alignment against the official control.",
                "change": "Use 'open the middle drawer of the cabinet' throughout rollout generation and dataset packing.",
            },
            {
                "id": "fix_image_distribution",
                "why": "Current MuJoCo images are far darker than the official control surface.",
                "change": "Apply deterministic brightness/background calibration before saving rollout frames.",
            },
            {
                "id": "fix_state_signal",
                "why": "Dims 3:7 are nearly constant and do not carry task-progress signal.",
                "change": "Replace quaternion-like placeholders with task-relevant motor proxy features.",
            },
            {
                "id": "fix_learning_source",
                "why": "Failed rollouts polluted the first pilot dataset.",
                "change": "Copy only successful rollouts into the learning set and repeat successful seeds for overnight training density.",
            },
            {
                "id": "fix_dataset_scale",
                "why": "Six pilot episodes are too small for meaningful adaptation.",
                "change": "Expand rollout generation to train seeds 1-10 and increase overnight train steps beyond the 200-step pilot.",
            },
        ],
        "night_runner_targets": {
            "train_seeds": list(range(1, 11)),
            "heldout_seeds": list(range(11, 16)),
            "success_only_learning": True,
            "long_train_steps_default": 4000,
        },
    }

    return {
        "control_surface": {
            "dataset_root": str(CONTROL_DATASET),
            "task_specific_benchmark": "E036/p1c11 official LIBERO drawer baseline",
        },
        "target_surface": {
            "dataset_root": str(TARGET_DATASET),
            "night_runner_result": str(CURRENT_NIGHT),
            "runner_path": str(RUNNER_PATH),
            "env_path": str(ENV_PATH),
        },
        "findings": findings,
        "recommendations": recommendations,
        "remediation_plan": remediation_plan,
    }


def write_outputs(report: dict[str, Any]) -> None:
    AUDIT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# MuJoCo Infinigen Alignment Audit",
        "",
        "## Summary",
        "This audit compares the current MuJoCo/Infinigen pilot dataset against the official LIBERO image-schema control surface and the restored upstream drawer benchmark.",
        "",
        "## Findings",
    ]
    for item in report["findings"]:
        lines.append(f"- `{item['id']}` [{item['severity']}] {item['summary']}")
    lines.extend(["", "## Recommendations"])
    for item in report["recommendations"]:
        lines.append(f"- `{item['priority']}` {item['action']}")
    lines.extend(["", "## Remediation Plan"])
    for item in report["remediation_plan"]["target_fixes"]:
        lines.append(f"- `{item['id']}` {item['change']} Reason: {item['why']}")
    targets = report["remediation_plan"]["night_runner_targets"]
    lines.extend(
        [
            "",
            "## Next Night-Run Targets",
            f"- train_seeds: {targets['train_seeds']}",
            f"- heldout_seeds: {targets['heldout_seeds']}",
            f"- success_only_learning: {targets['success_only_learning']}",
            f"- long_train_steps_default: {targets['long_train_steps_default']}",
        ]
    )
    AUDIT_MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    report = build_audit()
    write_outputs(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
