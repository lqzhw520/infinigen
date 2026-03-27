#!/usr/bin/env python3
"""Bootstrap or revise the MINT drawer campaign for the AnyGrasp robot-trajectory phase."""

from __future__ import annotations

import json
import shutil

import yaml
from mint_common import (
    ARCHIVE_DIR,
    CAMPAIGN_DIR,
    DEFAULT_HELD_OUT_SEEDS,
    DEFAULT_TRAIN_SEEDS,
    EVAL_DIR,
    QUEUE_ENVS,
    QUEUE_STEPS,
    default_next_actions,
    default_review,
    default_state,
    default_summary,
    ensure_contract_files,
    now_iso,
    run_memory_sync,
    save_next_actions,
    save_review,
    save_state,
    save_summary,
    update_watch_status,
    write_json_atomic,
    write_manifest,
    write_text_atomic,
)


def build_manifest() -> dict:
    return {
        "campaign": {
            "id": "mint_drawer_v1",
            "phase": "mint_robot_trajectory_root_cause_repair",
            "phase_gate": "await_train_reproducibility",
            "scope": "simulation-only",
            "main_claim": "Infinigen-generated AnyGrasp-conditioned robot-arm drawer trajectories improve MINT success on held-out drawer variants in simulation relative to pretrained MINT.",
            "queue": QUEUE_STEPS,
            "env_routing": QUEUE_ENVS,
            "object_split": {
                "train": DEFAULT_TRAIN_SEEDS,
                "held_out_sim": DEFAULT_HELD_OUT_SEEDS,
            },
            "baselines": ["random", "pretrained_mint", "finetuned_mint"],
            "primary_metric": "held_out_sim_success_rate",
            "notes": {
                "takeover_mode": "anygrasp_robot_trajectory",
                "tracking_scope": "out_of_scope",
                "proxy_baseline_archived": True,
                "robot_revision_v1_failed_archived": True,
                "root_cause_repair": True,
            },
        }
    }


def build_acceptance() -> dict:
    return {
        "g1_asset_load": {
            "pass": "robot + drawer scene loads for train and held-out seeds, and URDF/joint audits pass"
        },
        "g2_obs_contract": {
            "pass": "scene emits dual RGB, depth, task, and MINT-compatible robot state(8,), and scene-frame audit passes"
        },
        "g3_render_depth": {
            "pass": "all train and held-out seeds render cached RGB-D / point cloud payloads for AnyGrasp"
        },
        "g3_anygrasp_ready": {
            "pass": "AnyGrasp staging succeeds and AnyGrasp.load_net() returns successfully in the graspnet env"
        },
        "g3_grasp_plan": {
            "pass": ">=6 training seeds produce >=1 AnyGrasp pull-region grasp with score >= 0.5 and handle-region hit"
        },
        "g4_robot_trajectory": {
            "pass": "oracle scripted robot open succeeds stably; canonical learning rollouts are copied from oracle or AnyGrasp only after audit"
        },
        "g5_delta_reconstruction": {
            "pass": "natural robot rollouts replay without force_attach and satisfy MINT action-contract thresholds"
        },
        "g6_lerobot_pack": {
            "pass": "successful natural robot-scene rollouts finalize into a valid LeRobot dataset"
        },
        "g7_mint_batch_load": {
            "pass": "MINT preprocessor loads a NaN-free batch from the robot dataset"
        },
        "g8_single_rollout_overfit": {
            "pass": "single natural rollout overfit succeeds and beats pretrained on the source seed"
        },
        "g8_single_seed_overfit": {
            "pass": "single-seed overfit succeeds and beats pretrained on the source seed"
        },
        "g8_mint_train": {
            "pass": "MINT fine-tune runs 1000 steps and writes a checkpoint"
        },
        "g8_train_seed_probe": {
            "pass": "train-seed reproducibility shows a clear positive trend before held-out evaluation"
        },
        "g9_sim_eval": {
            "pass": "only after train-trend pass, fine-tuned MINT beats pretrained MINT on held-out seeds 11-15"
        },
    }


def build_takeover_memo() -> str:
    return """# MINT AnyGrasp Robot-Trajectory Takeover Memo

**Updated**: {updated}

## Locked non-repeat failures

- Do not reuse the old proxy campaign as the active claim; archive it first.
- Do not trust smoke-pass G4/G5/G6 artifacts as robot-trajectory evidence.
- Do not route AnyGrasp through the `mint` env; use `graspnet` after explicit staging.
- Do not use object-joint sweep or proxy deltas as MINT training actions.
- Do not infer Python 3.12 incompatibility before validating wrapper/API usage.
- Do not allow broken/truncated artifacts to count as completed steps.
- Do not use `force_attach` in G5-generated training data.
- Do not call the held-out claim before train-seed reproducibility turns positive.
- Do not let the PRD describe the old proxy phase as the active execution truth.

## Reusable prior work

- The proxy-control result remains a valid archived baseline.
- The depth/render split is still useful for AnyGrasp staging.
- Existing project-memory integration should be reused from day one.
- Current exported drawer assets behave like front-lip pull drawers, not cabinet handles.

## Active execution truth

- Current target is **simulation-only** held-out evaluation.
- This revision uses AnyGrasp detection only; tracking is out of scope.
- G4/G5 root-cause repair must prove that natural robot rollouts remain valid without `force_attach`.
- Held-out evaluation is deferred until train-seed reproducibility is positive.
- The active action/state contract is real robot EEF control:
  - action(7) = delta_xyz + delta_rxyz + gripper_command
  - state(8) = eef_pos_xyz + eef_quat_xyzw + gripper_open
""".format(updated=now_iso())


def ensure_robot_v1_archive() -> dict:
    archive_root = ARCHIVE_DIR / "robot_revision_v1_failed"
    archive_index = archive_root / "archive_index.json"
    if archive_index.exists():
        return json.loads(archive_index.read_text())

    archive_root.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    eval_archive = archive_root / "evaluation"
    if EVAL_DIR.exists():
        if eval_archive.exists():
            shutil.rmtree(eval_archive)
        shutil.copytree(EVAL_DIR, eval_archive)
        copied["evaluation"] = str(eval_archive)

    comparison_summary = {}
    train_probe = {}
    comparison_path = EVAL_DIR / "comparison_summary.json"
    train_probe_path = EVAL_DIR / "train_seed_probe.json"
    if comparison_path.exists():
        comparison_summary = json.loads(comparison_path.read_text())
    if train_probe_path.exists():
        train_probe = json.loads(train_probe_path.read_text())

    manifest = {
        "campaign": {
            "id": "mint_drawer_v1",
            "phase": "mint_robot_trajectory_pipeline",
            "phase_gate": "ready_for_writeup",
            "scope": "simulation-only",
            "main_claim": "Infinigen-generated AnyGrasp-conditioned robot-arm drawer trajectories improve MINT success on held-out drawer variants in simulation relative to pretrained MINT.",
        }
    }
    state = {
        "phase": "mint_robot_trajectory_pipeline",
        "phase_gate": "ready_for_writeup",
        "verdict": "scientific_not_supported",
        "claim_level": comparison_summary.get(
            "claim_level", "scientific_not_supported"
        ),
        "revision": {
            "active": "robot_trajectory_revision",
            "archived": ["proxy_revision"],
        },
        "history": [
            {
                "at": now_iso(),
                "event": "robot_revision_v1_failed_archived",
                "details": "Reconstructed archive from the previous robot-trajectory failure using saved evaluation artifacts.",
            }
        ],
    }
    review = {
        "phase": "mint_robot_trajectory_pipeline",
        "phase_gate": "ready_for_writeup",
        "workflow_score": 10,
        "evidence_score": 8,
        "decision": "revise_claim",
        "blocker_type": "none",
        "verdict": "scientific_not_supported",
        "claim_assessment": comparison_summary.get(
            "strongest_true_claim",
            "The previous robot-trajectory revision did not show stable train-seed or held-out success.",
        ),
        "strengths": [
            "The previous robot-trajectory revision executed end-to-end and produced a valid negative summary."
        ],
        "weaknesses": [
            "Train-seed reproducibility remained weak, so the robot-trajectory claim was not supported."
        ],
        "next_actions": [
            "Repair the G4/G5 physical contract before rerunning train or held-out evaluation."
        ],
        "generated_at": now_iso(),
    }
    summary = {
        "campaign": "mint_drawer_v1",
        "phase": "mint_robot_trajectory_pipeline",
        "status": "scientific_not_supported",
        "evaluation": comparison_summary,
        "train_probe": train_probe,
        "strongest_true_claim": comparison_summary.get(
            "strongest_true_claim",
            "The robot-trajectory pipeline executed, but the policy did not show stable train-seed or held-out success.",
        ),
        "generated_at": now_iso(),
    }
    decision_memo = (
        "# Archived Robot Revision V1 Decision Memo\n\n"
        f"**Archived At**: {now_iso()}\n"
        "**Verdict**: `scientific_not_supported`\n\n"
        f"{summary['strongest_true_claim']}\n"
    )
    campaign_status = (
        "# Archived MINT Drawer Campaign Dashboard\n\n"
        f"**Updated**: {now_iso()}\n"
        "**Campaign**: `mint_drawer_v1`\n"
        "**Revision**: `robot_revision_v1_failed`\n"
        "**Verdict**: `scientific_not_supported`\n\n"
        "This archive was reconstructed from the previous robot-trajectory evaluation artifacts.\n"
    )

    file_payloads = {
        archive_root / "manifest.yaml": manifest,
        archive_root / "state.json": state,
        archive_root / "review.json": review,
        archive_root / "summary.json": summary,
    }
    write_text_atomic(archive_root / "decision_memo.md", decision_memo)
    write_text_atomic(archive_root / "campaign_status.md", campaign_status)
    for path, payload in file_payloads.items():
        if path.suffix == ".yaml":
            write_text_atomic(path, yaml.safe_dump(payload, sort_keys=False))
        else:
            write_json_atomic(path, payload)

    payload = {
        "archived_at": now_iso(),
        "archive_root": str(archive_root),
        "copied": copied,
        "reconstructed": True,
    }
    write_json_atomic(archive_index, payload)
    return payload


def proxy_archive_payload() -> dict:
    archive_root = ARCHIVE_DIR / "proxy_revision"
    archive_index = archive_root / "archive_index.json"
    if archive_index.exists():
        return json.loads(archive_index.read_text())
    return {"archive_root": str(archive_root), "copied": {}}


def main() -> int:
    manifest = build_manifest()
    robot_v1_archive = ensure_robot_v1_archive()
    proxy_archive = proxy_archive_payload()

    ensure_contract_files()
    write_manifest(manifest)
    write_json_atomic(CAMPAIGN_DIR / "acceptance_criteria.json", build_acceptance())

    state = default_state(manifest)
    state["revision"]["proxy_revision_archived"] = True
    state["revision"]["robot_revision_v1_failed_archived"] = True
    state["revision"]["robot_trajectory_revision_started"] = True
    state["history"] = [
        {
            "at": now_iso(),
            "event": "proxy_revision_archived",
            "details": f"Proxy baseline remains archived at {proxy_archive['archive_root']}.",
        },
        {
            "at": now_iso(),
            "event": "robot_revision_v1_failed_archived",
            "details": f"Archived the previous failed robot revision to {robot_v1_archive['archive_root']}.",
        },
        {
            "at": now_iso(),
            "event": "robot_revision_v2_root_cause_repair_started",
            "details": "Activated the AnyGrasp + robot-trajectory root-cause repair revision.",
        },
    ]
    save_state(state)

    summary = default_summary(manifest)
    summary["proxy_baseline_archived"] = True
    summary["archived_proxy"] = proxy_archive
    summary["archived_robot_revision_v1_failed"] = robot_v1_archive
    summary["status"] = "robot_revision_v2_root_cause_repair_initialized"
    save_summary(summary)

    review = default_review(manifest)
    review["strengths"] = [
        "Proxy baseline and the previous failed robot revision were archived before restarting the active claim."
    ]
    review["weaknesses"] = [
        "Root-cause repair has not yet revalidated the asset/sim, perception, or natural G5 contract layers."
    ]
    review["next_actions"] = [
        "Run the next pending queue step: `g1_asset_load`.",
        "Do not interpret the old held-out 0/5 result as the final claim verdict before train-seed reproducibility is rechecked.",
    ]
    save_review(review)
    next_actions = default_next_actions()
    next_actions["actions"] = [
        {
            "type": "run_experiments",
            "target": "g1_asset_load",
            "reason": "Start the root-cause ladder from asset/simulation audit.",
        },
    ]
    save_next_actions(next_actions)

    write_text_atomic(
        CAMPAIGN_DIR / "decision_memo.md",
        "# Pending\n\nRobot-trajectory root-cause repair revision initialized; no terminal verdict yet.\n",
    )
    write_text_atomic(CAMPAIGN_DIR / "takeover_memo.md", build_takeover_memo())
    write_text_atomic(
        CAMPAIGN_DIR / "campaign_spec.md",
        "# Campaign Spec\n\n"
        "MINT drawer AnyGrasp-conditioned robot-trajectory root-cause repair campaign.\n\n"
        "- Train seeds: `1-10`\n"
        "- Held-out seeds: `11-15`\n"
        "- Active goal: prove oracle/AnyGrasp scripted success, then natural G5 contract, then train-seed reproducibility, then held-out transfer.\n",
    )
    write_text_atomic(
        CAMPAIGN_DIR / "campaign_status.md",
        "# MINT Drawer Campaign Dashboard\n\n"
        f"**Updated**: {now_iso()}\n"
        "**Campaign**: `mint_drawer_v1`\n"
        "**Phase**: `mint_robot_trajectory_root_cause_repair`\n"
        "**Gate**: `await_train_reproducibility`\n"
        "**Claim**: `Infinigen-generated AnyGrasp-conditioned robot-arm drawer trajectories improve MINT success on held-out drawer variants in simulation relative to pretrained MINT.`\n\n"
        "## Revision\n\n"
        f"- Archived proxy baseline: `{proxy_archive['archive_root']}`\n"
        f"- Archived failed robot revision: `{robot_v1_archive['archive_root']}`\n"
        "- Active revision: `robot_revision_v2_root_cause_repair`\n"
        "- Next focus: `asset/sim audit -> oracle/AnyGrasp scripted audit -> natural G5 contract -> train-seed reproducibility`\n",
    )
    update_watch_status(
        {
            "supervisor_pid": None,
            "inner_watcher_pid": None,
            "worker_pid": None,
            "worker_kind": None,
            "queue_step": None,
            "stage": "bootstrap",
            "status": "initialized",
            "last_heartbeat": now_iso(),
            "latest_artifact": None,
            "last_repair_action": None,
            "last_error": None,
        }
    )
    run_memory_sync(
        title="mint_drawer_v1 robot trajectory root cause repair started",
        summary="Archived the previous failed robot revision and initialized the root-cause repair ladder.",
        completed=["proxy_revision_archived", "robot_revision_v1_failed_archived"],
        next_actions=[
            "Run g1-g3 to validate the AnyGrasp + robot-scene setup.",
            "Do not rerun held-out evaluation until train-seed reproducibility turns positive.",
        ],
        artifacts={
            "archived_proxy": proxy_archive["archive_root"],
            "archived_robot_revision_v1_failed": robot_v1_archive["archive_root"],
        },
        lessons=[
            "The previous proxy result remains a valid archived baseline, but it no longer defines the active campaign claim.",
            "The previous robot revision failed because a runnable pipeline was mistaken for a trustworthy scientific result.",
            "The active claim now requires true robot-arm / end-effector trajectories and natural G5 replay.",
        ],
    )
    print(
        json.dumps(
            {
                "campaign_dir": str(CAMPAIGN_DIR),
                "archive_root": robot_v1_archive["archive_root"],
                "status": "bootstrapped_robot_revision_v2_root_cause_repair",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
