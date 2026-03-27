from __future__ import annotations

import json
import shutil
from pathlib import Path

from mint_common import (
    ARTIFACT_DIR,
    CAMPAIGN_DIR,
    EVAL_DIR,
    append_history,
    build_next_actions,
    load_manifest,
    load_state,
    load_summary,
    now_iso,
    refresh_campaign_views,
    render_campaign_status,
    save_next_actions,
    save_review,
    save_state,
    save_summary,
    update_watch_status,
    write_text_atomic,
)

RUN_TAG = "2026-03-26_gate_drift_reset"

KEEP_COMPLETED = {
    "u1_asset_geometry_audit",
    "u2_joint_semantics_audit",
    "u3_handle_region_audit",
    "u4_export_consistency_audit",
    "u5_seed_outlier_audit",
    "a1_asset_scene_audit",
    "a2_frame_transform_audit",
    "b1_oracle_scripted_baseline",
    "b2_anygrasp_scripted_baseline",
}

ARTIFACT_FILES = [
    "c1_teacher_native_rollout_rebuild.json",
    "c2_action_contract_repair.json",
    "c3_single_rollout_replay_gate.json",
    "strict_teacher_dataset_audit.json",
    "d1_candidate_search.json",
    "d1_candidate_progress.json",
    "d1_single_rollout_overfit.json",
    "d1_single_rollout_strict_eval.json",
    "d2_single_seed_overfit.json",
    "d3_train_seed_probe.json",
    "active_action_contract.json",
    "c_action_contract_branch_progress.json",
]

ARTIFACT_DIRS = [
    "c1_branch_rollouts",
    "c2_replay_valid_rollouts",
    "c2_replay_valid_rollouts_branch2_phase_scaled",
    "c2_replay_valid_rollouts_branch3_explicit_gripper_phases",
    "c2_replay_valid_rollouts_branch4_densified_attach_window",
    "c3_single_rollout",
    "strict_teacher_rollouts",
    "d1_phase_windows",
    "d2_phase_windows",
    "d3_phase_windows",
]

ARTIFACT_PREFIXES = ("d1_", "d2_", "d3_")
ROOT_PREFIXES = (
    "dataset_d1_",
    "dataset_d2_",
    "dataset_d3_",
    "outputs_d1_",
    "outputs_d2_",
    "outputs_d3_",
)


def move_path(src: Path, dst_root: Path) -> None:
    if not src.exists():
        return
    dst_root.mkdir(parents=True, exist_ok=True)
    dst = dst_root / src.name
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.move(str(src), str(dst))


def write_decision_memo(now: str) -> None:
    memo = "\n".join(
        [
            "# MINT Drawer Robot-Trajectory Campaign Decision Memo",
            "",
            f"**Updated**: {now}",
            "**Phase**: `mint_robot_trajectory_claim_push`",
            "**Gate**: `teacher_contract_rebuild`",
            "**Verdict**: `awaiting_execution`",
            "",
            "## Current Assessment",
            "",
            "The most recent D2/D3/E1 sequence is **diagnostic-only after gate drift** and must not be treated as the final verdict for the main claim.",
            "",
            "## Why The Previous Verdict Was Invalid For Claim",
            "",
            "- `D1` passed only as a diagnostic winner on `seed_009_episode_02`.",
            "- `D2` then used a single-rollout seed (`seed 9`), which is invalid for a true single-seed overfit gate.",
            "- `D3` continued after `D2` without a hard pass.",
            "- `E1` therefore became a diagnostic held-out run rather than a final claim evaluation.",
            "",
            "## Current Strongest True Claim",
            "",
            "A diagnostic D1 winner exists, proving at least one strict-valid rollout is learnable, but the mainline automatic pipeline must be rebuilt from `C1` onward before the claim can be tested again.",
            "",
            "## Next Required Work",
            "",
            "1. Rebuild teacher contract from `C1/C2` so teacher and learner share the same control problem.",
            "2. Rebuild strict teacher dataset and require D2-feasible seeds only.",
            "3. Restart `D1` from a mainline candidate, then continue to `D2 -> D3 -> E1` only under hard gates.",
            "",
        ]
    )
    write_text_atomic(CAMPAIGN_DIR / "decision_memo.md", memo)


def main() -> None:
    manifest = load_manifest()
    state = load_state()
    summary = load_summary()
    now = now_iso()

    archive_root = ARTIFACT_DIR / "diagnostic_runs" / RUN_TAG
    campaign_root_archive = archive_root / "campaign_root"
    evaluation_archive = archive_root / "evaluation"

    for name in ARTIFACT_FILES:
        move_path(ARTIFACT_DIR / name, archive_root)
    for name in ARTIFACT_DIRS:
        move_path(ARTIFACT_DIR / name, archive_root)
    for path in list(ARTIFACT_DIR.iterdir()):
        if any(path.name.startswith(prefix) for prefix in ARTIFACT_PREFIXES):
            move_path(path, archive_root)
    for path in list(EVAL_DIR.iterdir()):
        move_path(path, evaluation_archive)
    for path in list(CAMPAIGN_DIR.iterdir()):
        if any(path.name.startswith(prefix) for prefix in ROOT_PREFIXES):
            move_path(path, campaign_root_archive)

    archive_note = {
        "tag": RUN_TAG,
        "at": now,
        "classification": "diagnostic_only_after_gate_drift",
        "reason": {
            "d1": "diagnostic winner existed but came from seed 9",
            "d2": "single-rollout seed used for single-seed overfit",
            "d3": "soft gate continued after D2 failed",
            "e1": "held-out result was diagnostic-only, not final claim verdict",
        },
    }
    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / "archive_note.json").write_text(json.dumps(archive_note, indent=2))

    for item in state.get("queue", []):
        step_id = item.get("id")
        item["status"] = "completed" if step_id in KEEP_COMPLETED else "pending"
        item["attempts"] = 0
        item["updated_at"] = now
        item.pop("last_error", None)

    state["phase_gate"] = "teacher_contract_rebuild"
    state["verdict"] = None
    state["claim_level"] = None
    state["last_error"] = "teacher_contract_rebuild_required"
    state["last_invalidated_result"] = "diagnostic_only_after_gate_drift"
    state["invalidated_results"] = [
        "d2_invalid_for_claim_due_to_single_rollout_seed",
        "d3_invalid_for_claim_due_to_soft_gate",
        "e1_diagnostic_only_after_gate_drift",
    ]
    state["coverage_expansion_attempts"] = 0
    state["retries"] = {}
    state["active_job"] = None
    state["active_runtime"] = {
        "step_id": "c1_teacher_native_rollout_rebuild",
        "active_lane": "learnability",
        "active_branch": "teacher_contract_rebuild",
        "active_fingerprint": {"recovery_tag": RUN_TAG},
        "worker_pid": None,
        "worker_kind": "manual_reconcile",
        "attempt_no": 0,
        "launched_at": now,
        "heartbeat_at": now,
        "latest_artifact": str(archive_root / "archive_note.json"),
        "last_error": "teacher_contract_rebuild_required",
        "stage": "teacher_contract_rebuild",
    }
    state["strict_replay_lane"] = {
        "status": "pending",
        "best_branch": None,
        "best_metrics": {},
        "last_update": now,
    }
    state["learnability_lane"] = {
        "status": "diagnostic_winner_only",
        "source_contract_mode": None,
        "source_branch": None,
        "current_step": "c1_teacher_native_rollout_rebuild",
        "last_positive_step": "d1_single_rollout_overfit",
        "last_update": now,
        "diagnostic_winner_seed": 9,
        "diagnostic_winner_rollout": "seed_009_episode_02",
    }
    append_history(
        state,
        "gate_drift_reset",
        "Archived previous C1-E1 artifacts as diagnostic_only_after_gate_drift and reset mainline to teacher_contract_rebuild.",
    )
    save_state(state)

    review, summary = refresh_campaign_views(state, manifest, summary)
    state = load_state()
    state["learnability_lane"]["status"] = "diagnostic_winner_only"
    state["learnability_lane"]["current_step"] = "c1_teacher_native_rollout_rebuild"
    state["learnability_lane"]["last_positive_step"] = "d1_single_rollout_overfit"
    state["learnability_lane"]["diagnostic_winner_seed"] = 9
    state["learnability_lane"]["diagnostic_winner_rollout"] = "seed_009_episode_02"
    state["upstream_audit_lane"]["patch_allowed"] = False
    state["upstream_audit_lane"]["fix_required"] = False
    state["upstream_audit_lane"]["patch_gate_reason"] = "await_teacher_contract_rebuild"
    save_state(state)
    summary["status"] = "teacher_contract_rebuild"
    summary["artifacts"] = {
        "b1_oracle_scripted_baseline": str(
            ARTIFACT_DIR / "b1_oracle_scripted_baseline.json"
        ),
        "b2_anygrasp_scripted_baseline": str(
            ARTIFACT_DIR / "b2_anygrasp_scripted_baseline.json"
        ),
        "diagnostic_archive": str(archive_root / "archive_note.json"),
    }
    summary["strongest_true_claim"] = (
        "A diagnostic D1 winner exists, but the last D2/D3/E1 sequence is invalid for claim after gate drift and must not be treated as final evidence."
    )
    summary["last_error"] = "teacher_contract_rebuild_required"
    summary["last_invalidated_result"] = "diagnostic_only_after_gate_drift"
    summary["invalidated_results"] = list(state["invalidated_results"])
    summary["diagnostic_archive"] = {
        "tag": RUN_TAG,
        "root": str(archive_root),
        "d1_diagnostic_winner_seed": 9,
        "d1_diagnostic_winner_rollout": "seed_009_episode_02",
    }
    save_summary(summary)

    review["decision"] = "run_experiments"
    review["blocker_type"] = "none"
    review["verdict"] = "awaiting_execution"
    review["claim_assessment"] = (
        "D1 produced a diagnostic winner, but the last D2/D3/E1 sequence is diagnostic-only after gate drift. Teacher contract rebuild is now required before claim evaluation can resume."
    )
    review["strengths"] = [
        "Layer-0 evaluator attachment bug is already fixed; D1 is no longer structurally impossible.",
        "A diagnostic D1 winner exists and proves at least one strict-valid rollout is learnable.",
        "Environment and upstream audits remain archived as completed evidence.",
    ]
    review["weaknesses"] = [
        "The last D2/D3/E1 run drifted past hard gates and cannot serve as final claim evidence.",
        "Teacher contract still needs to be rebuilt so teacher and learner share the same control problem.",
        "Mainline training must come from D2-feasible seeds only (seed 2 or seed 10 under the current audit assumptions).",
    ]
    review["next_actions"] = [
        "Rerun C1 and C2 under the rebuilt closed-loop teacher contract.",
        "Rebuild the strict teacher pool and verify mainline candidates come only from D2-feasible seeds.",
        "Restart D1 only after the rebuilt teacher proves seed 2 and seed 10 each have at least two strict-valid rollouts.",
    ]
    save_review(review)
    save_next_actions(build_next_actions(state, review))
    write_text_atomic(
        CAMPAIGN_DIR / "campaign_status.md",
        render_campaign_status(manifest, state, review),
    )

    update_watch_status(
        {
            "supervisor_pid": None,
            "inner_watcher_pid": None,
            "worker_pid": None,
            "worker_kind": "manual_reconcile",
            "queue_step": "c1_teacher_native_rollout_rebuild",
            "stage": "teacher_contract_rebuild",
            "active_lane": "learnability",
            "active_branch": "teacher_contract_rebuild",
            "active_fingerprint": {"recovery_tag": RUN_TAG},
            "status": "paused",
            "last_heartbeat": now,
            "latest_artifact": str(archive_root / "archive_note.json"),
            "last_successful_milestone": "d1_single_rollout_overfit",
            "retry_count": 0,
            "last_repair_action": "diagnostic_only_after_gate_drift_reset",
            "last_error": "teacher_contract_rebuild_required",
            "verdict": "awaiting_execution",
        }
    )

    write_decision_memo(now)

    hist_dir = Path("/mnt/afs2/zhuhaowu/infinigen/.project-memory/history")
    hist_dir.mkdir(parents=True, exist_ok=True)
    (hist_dir / "2026-03-26_mint-drawer-gate-drift-reset.md").write_text(
        "\n".join(
            [
                "# Gate Drift Reset",
                "",
                f"- At: {now}",
                f"- Archived run: {RUN_TAG}",
                "- Reason: D2/D3/E1 were diagnostic-only after gate drift.",
                "- Next step: rerun C1/C2 under rebuilt closed-loop teacher contract.",
                "",
            ]
        )
    )

    print("RESET_OK")
    print(str(archive_root))


if __name__ == "__main__":
    main()
