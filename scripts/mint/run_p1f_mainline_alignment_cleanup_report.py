#!/usr/bin/env python3
"""Generate a compact canonical summary for the current MuJoCo mainline plus cleanup actions."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1")
ART = ROOT / "artifacts"
OUT = ROOT / "outputs"
ARCHIVE = ROOT / "archive"
SOVEREIGN = ROOT / "sovereign"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    p1c11 = load(ART / "p1c11_official_libero_goal_drawer_baseline_rollback_4eab579.json")
    night = load(ART / "mujoco_infinigen_mainline_night.json")
    m4 = load(ART / "m4_robot_rollout_gate.json")
    m5 = load(ART / "m5_dataset_pack_gate.json")
    m6 = load(ART / "m6_mint_train_gate.json")
    m7 = load(ART / "m7_mint_eval_gate.json")
    status = load(SOVEREIGN / "night" / "mujoco_infinigen_mainline_night_last.json")
    current_truth = load(SOVEREIGN / "current_truth.json")
    next_actions = load(SOVEREIGN / "next_actions.json")

    report = {
        "control": {
            "mint_head": p1c11["external_mint_head"],
            "task": p1c11["task_name"],
            "success_rate": p1c11["eval_info"]["overall"]["pc_success"],
            "n_episodes": p1c11["n_episodes"],
            "artifact": str(ART / "p1c11_official_libero_goal_drawer_baseline_rollback_4eab579.json"),
        },
        "mainline_run": {
            "started_at": night["started_at"],
            "completed_at": night["completed_at"],
            "final_stage": status["current_stage"],
            "final_status": status["status"],
        },
        "alignment_fixes_applied": [
            "task text aligned to 'open the middle drawer of the cabinet'",
            "learning set restricted to successful rollouts only",
            "successful rollout seeds repeated to raise overnight train density",
            "train seed pool expanded from pilot subset to 1..10",
            "MuJoCo image brightness/background calibrated toward official control",
            "state dims 3:7 replaced with task-relevant motor proxy features",
        ],
        "training_dataset": {
            "learning_source": m4["learning_source"],
            "oracle_successful_seeds": m4["oracle_successful_seeds"],
            "anygrasp_successful_seeds": m4["anygrasp_successful_seeds"],
            "successful_learning_seeds": m4.get("successful_learning_seeds", []),
            "success_repeat": m4.get("success_repeat"),
            "copied_learning_files": m4["copied_learning_files"],
            "episode_count": m5["episode_count"],
            "frame_count": m5["frame_count"],
            "task_coverage": m5["task_coverage"],
            "seed_coverage": m5["seed_coverage"],
        },
        "training_result": {
            "steps_requested": m6["steps_requested"],
            "elapsed_sec": m6["elapsed_sec"],
            "checkpoint_path": m6["checkpoint_path"],
        },
        "eval_result": {
            "pretrained_success": m7["pretrained"]["success_count"],
            "finetuned_success": m7["finetuned"]["success_count"],
            "heldout_seed_count": m7["pretrained"]["seed_count"],
            "delta_success": m7["delta_success"],
        },
        "remaining_gaps": [
            "held-out success remains 0/5 for both pretrained and fine-tuned MINT",
            "current success-only dataset still comes from only 5 successful train seeds, repeated 10x",
            "no exact official control state/action trace exists yet for one-to-one distribution matching against the p1c11 benchmark",
            "the p1e audit is still useful, but its task_text finding is stale because the dataset task phrase has already been corrected",
        ],
        "canonical_keep_in_place": [
            "sovereign/",
            "dataset/",
            "artifacts/p1c10_release_runtime_matched_ab.json",
            "artifacts/p1c11_official_libero_goal_drawer_baseline_rollback_4eab579.json",
            "artifacts/p1e_mujoco_infinigen_alignment_audit.json",
            "artifacts/m0_authority_preflight.json",
            "artifacts/m1_proxy_asset_smoke.json",
            "artifacts/m2_true_robot_env_gate.json",
            "artifacts/m3_anygrasp_gate.json",
            "artifacts/m4_robot_rollout_gate.json",
            "artifacts/m5_dataset_pack_gate.json",
            "artifacts/m6_mint_train_gate.json",
            "artifacts/m7_mint_eval_gate.json",
            "artifacts/mujoco_infinigen_mainline_night.json",
            "artifacts/p1f_mainline_alignment_cleanup_report.json",
            "outputs/m6_mujoco_mint_train/",
            "outputs/m7_mujoco_eval/",
            "outputs/mujoco_infinigen_mainline_night.launch.log",
            "outputs/mujoco_infinigen_mainline_night_report.md",
            "outputs/p1e_mujoco_infinigen_alignment_audit.md",
            "outputs/p1f_mainline_alignment_cleanup_report.md",
            "outputs/p1c11_official_libero_goal_drawer_baseline_rollback_4eab579/",
        ],
        "archived_now": {
            "legacy_docs": sorted(p.name for p in (ARCHIVE / "legacy_docs").iterdir()),
            "old_dataset_snapshots": sorted(p.name for p in (ARCHIVE / "old_dataset_snapshots").iterdir()),
            "invalid_runs_artifacts": sorted(p.name for p in (ARCHIVE / "invalid_runs" / "artifacts").iterdir()),
            "invalid_runs_outputs": sorted(p.name for p in (ARCHIVE / "invalid_runs" / "outputs").iterdir()),
            "mujoco_pilot_pre_mainline_artifacts": sorted(p.name for p in (ARCHIVE / "mujoco_pilot_pre_mainline" / "artifacts").iterdir()),
            "mujoco_pilot_pre_mainline_outputs": sorted(p.name for p in (ARCHIVE / "mujoco_pilot_pre_mainline" / "outputs").iterdir()),
        },
        "deleted_now": [
            "._acceptance_criteria.json",
            "artifacts/m3_smoke_payload.npz",
            "artifacts/m3_smoke_result.json",
            "artifacts/drawer_depth.npz",
            "artifacts/drawer_intrinsics.npy",
            "artifacts/drawer_rgb.npy",
        ],
        "top_level_after_cleanup": sorted(p.name for p in ROOT.iterdir()),
        "artifact_count_after_cleanup": len(list((ROOT / "artifacts").iterdir())),
        "output_count_after_cleanup": len(list((ROOT / "outputs").iterdir())),
        "current_truth": {
            "phase": current_truth["current"]["phase"],
            "phase_gate": current_truth["current"]["phase_gate"],
            "next_action": current_truth["current"]["next_action"],
        },
        "top_actions": next_actions.get("actions", [])[:6],
    }

    json_path = ART / "p1f_mainline_alignment_cleanup_report.json"
    md_path = OUT / "p1f_mainline_alignment_cleanup_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    lines: list[str] = []
    lines.append("# P1f Mainline Alignment And Cleanup Report")
    lines.append("")
    lines.append("## Canonical Control")
    lines.append(f"- external/MINT head: `{report['control']['mint_head']}`")
    lines.append(f"- official task: `{report['control']['task']}`")
    lines.append(
        f"- official baseline success: `{report['control']['success_rate']}` over `{report['control']['n_episodes']}` episodes"
    )
    lines.append(f"- control artifact: `{report['control']['artifact']}`")
    lines.append("")
    lines.append("## What Last Night Actually Did")
    lines.append(
        f"- run window: `{report['mainline_run']['started_at']}` -> `{report['mainline_run']['completed_at']}`"
    )
    lines.append(f"- final stage: `{report['mainline_run']['final_stage']}`")
    lines.append(f"- final status: `{report['mainline_run']['final_status']}`")
    lines.append("")
    lines.append("## Alignment Fixes Applied Before Training")
    for item in report["alignment_fixes_applied"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Training Dataset Used Last Night")
    for key in [
        "learning_source",
        "oracle_successful_seeds",
        "anygrasp_successful_seeds",
        "successful_learning_seeds",
        "success_repeat",
        "copied_learning_files",
        "episode_count",
        "frame_count",
        "task_coverage",
        "seed_coverage",
    ]:
        lines.append(f"- {key}: `{report['training_dataset'][key]}`")
    lines.append("")
    lines.append("## Training And Eval Outcome")
    lines.append(f"- train steps: `{report['training_result']['steps_requested']}`")
    lines.append(f"- train elapsed_sec: `{report['training_result']['elapsed_sec']}`")
    lines.append(
        f"- pretrained held-out success: `{report['eval_result']['pretrained_success']}/{report['eval_result']['heldout_seed_count']}`"
    )
    lines.append(
        f"- finetuned held-out success: `{report['eval_result']['finetuned_success']}/{report['eval_result']['heldout_seed_count']}`"
    )
    lines.append(f"- delta_success: `{report['eval_result']['delta_success']}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- The upstream official MINT baseline is intact and successful.")
    lines.append("- The true MuJoCo/Infinigen line now trains end-to-end with a cleaned success-only dataset.")
    lines.append("- The current failure is no longer 'no training happened'; it is 'training happened but did not improve held-out drawer success'.")
    lines.append("")
    lines.append("## Remaining Gaps")
    for item in report["remaining_gaps"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Cleanup Completed")
    lines.append(f"- top-level entries after cleanup: `{report['top_level_after_cleanup']}`")
    lines.append(f"- artifact_count_after_cleanup: `{report['artifact_count_after_cleanup']}`")
    lines.append(f"- output_count_after_cleanup: `{report['output_count_after_cleanup']}`")
    lines.append("")
    lines.append("### Archived Now")
    for key, items in report["archived_now"].items():
        lines.append(f"- `{key}`: {items}")
    lines.append("")
    lines.append("### Deleted Now")
    for item in report["deleted_now"]:
        lines.append(f"- `{item}`")
    lines.append("")
    lines.append("## Canonical Keep-In-Place Working Set")
    for item in report["canonical_keep_in_place"]:
        lines.append(f"- `{item}`")
    md_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({"json": str(json_path), "md": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
