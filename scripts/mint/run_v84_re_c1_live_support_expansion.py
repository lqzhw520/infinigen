#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
MINT_SCRIPTS = PROJECT_ROOT / "scripts" / "mint"
sys.path.insert(0, str(MINT_SCRIPTS))

from drawer_robot_env_mujoco import DrawerEnvContractConfig, build_robot_rollout, save_robot_rollout  # noqa: E402
from root_cause_controller import RootCauseController  # noqa: E402
from v13_audit_common import (  # noqa: E402
    ACCEPTANCE_CONTRACT_PATH,
    ARTIFACT_DIR,
    AUTOPILOT_DIR,
    TRUTH_CONTRACT_PATH,
    current_repo_identity,
    load_json,
    make_run_instance,
    save_active_plan,
    sha256_file,
    teacher_family_dispatch_payload,
    write_gate,
    write_json_atomic,
)

SPEC_REFERENCE = str(PROJECT_ROOT / "docs" / "MINT_V84_P0_INFRASTRUCTURE_REPAIR_SPEC.md")
PLAN_OUTPUT_PATH = ARTIFACT_DIR / "p0_infrastructure_repair_plan.json"
OUTPUT_PATH = ARTIFACT_DIR / "re_c1_live_support_expansion.json"
ROLL_OUT_DIR = ARTIFACT_DIR / "re_c1_live_support_rollouts"
GATE_PATH = AUTOPILOT_DIR / "gates_p0_infrastructure" / "Re_C1_live_support_expansion.json"
SOURCE_P0A_GATE = AUTOPILOT_DIR / "gates_p0_infrastructure" / "P0a_live_rollout_family_dispatch_audit.json"
SOURCE_P0B_GATE = AUTOPILOT_DIR / "gates_p0_infrastructure" / "P0b_feature_computation_path_unification.json"
TARGET_SEEDS = list(range(1, 9))
SCAN_DEPTH = 12


def _sha1_rollout(rollout: dict[str, Any]) -> str:
    h = hashlib.sha1()
    h.update(np.asarray(rollout["states"], dtype=np.float32).tobytes())
    h.update(np.asarray(rollout["actions"], dtype=np.float32).tobytes())
    return h.hexdigest()


def _make_plan() -> dict[str, Any]:
    ident = current_repo_identity()
    run_instance_id = make_run_instance("rec1", ident["working_head_commit"])
    plan = {
        "run_instance_id": run_instance_id,
        "plan_version": "tiny_retrain_confirmation_p0_infrastructure_repair",
        "slice_name": "re-c1-live-support-expansion",
        "working_head_commit": ident["working_head_commit"],
        "vendor_head_commit": ident["vendor_head_commit"],
        "branch": ident["branch"],
        "execution_scope": "re_c1_live_support_expansion",
        "bridge_stage": "p0_infrastructure_repair",
        "diagnostic_only": True,
        "claim_bearing": False,
        "training_allowed": False,
        "probe_allowed": False,
        "spec_reference": SPEC_REFERENCE,
        "truth_contract_hash": sha256_file(TRUTH_CONTRACT_PATH),
        "acceptance_contract_hash": sha256_file(ACCEPTANCE_CONTRACT_PATH),
        "allowed_next_phases": ["Re-C1", "Re-C2"],
    }
    write_json_atomic(PLAN_OUTPUT_PATH, plan)
    save_active_plan(plan)
    return plan


def _live_contract() -> DrawerEnvContractConfig:
    controller = RootCauseController()
    payload = dict(
        controller._frozen_matrix_contracts_v5_pro()["V1cT2S3"]["env_contract_config"]
    )
    return DrawerEnvContractConfig(**payload)


def run() -> int:
    plan = _make_plan()
    p0a_gate = load_json(SOURCE_P0A_GATE, {})
    p0b_gate = load_json(SOURCE_P0B_GATE, {})
    blocking: list[str] = []
    if str(p0a_gate.get("status") or "") != "PASS":
        blocking.append("p0a_not_passed")
    if str(p0b_gate.get("status") or "") != "PASS":
        blocking.append("p0b_not_passed")

    if ROLL_OUT_DIR.exists():
        shutil.rmtree(ROLL_OUT_DIR)
    ROLL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_rows: list[dict[str, Any]] = []
    seed_summary: dict[str, Any] = {}
    raw_attempt_rows: list[dict[str, Any]] = []
    if not blocking:
        contract = _live_contract()
        for seed in TARGET_SEEDS:
            seen_success_fingerprints: set[str] = set()
            attempts_for_seed: list[dict[str, Any]] = []
            successful_variants: set[str] = set()
            for episode_index in range(SCAN_DEPTH):
                payload = teacher_family_dispatch_payload(episode_index)
                rollout = build_robot_rollout(
                    seed=seed,
                    grasp_pose_world=np.eye(4, dtype=np.float32),
                    episode_index=episode_index,
                    image_size=64,
                    max_steps=96,
                    contract=contract,
                    rotation_source="aligned",
                    claim_policy="diagnostic",
                    teacher_controller_mode="interaction_frame_hybrid",
                    interventions=payload,
                )
                rollout["run_instance_id"] = plan["run_instance_id"]
                rollout["plan_version"] = plan["plan_version"]
                rollout["working_head_commit"] = plan["working_head_commit"]
                rollout["bridge_stage"] = plan["execution_scope"]
                rollout["bridge_attempt"] = "re_c1_live_support_expansion"
                fingerprint = _sha1_rollout(rollout)
                success = bool(rollout.get("success"))
                variant_name = str(payload.get("teacher_family_variant") or "base")
                attempt = {
                    "seed": seed,
                    "episode_index": episode_index,
                    "teacher_family_variant": variant_name,
                    "success": success,
                    "fingerprint": fingerprint,
                    "final_drawer_fraction": float(rollout.get("final_drawer_fraction", 0.0) or 0.0),
                    "max_drawer_fraction": float(rollout.get("max_drawer_fraction", 0.0) or 0.0),
                }
                attempts_for_seed.append(attempt)
                raw_attempt_rows.append(attempt)
                if success:
                    successful_variants.add(variant_name)
                    if fingerprint not in seen_success_fingerprints:
                        seen_success_fingerprints.add(fingerprint)
                        base = ROLL_OUT_DIR / f"seed_{seed:03d}_episode_{episode_index:02d}"
                        save_robot_rollout(base, rollout)
                        selected_rows.append(
                            {
                                "seed": seed,
                                "episode_index": episode_index,
                                "teacher_family_variant": variant_name,
                                "fingerprint": fingerprint,
                                "path": str(base.with_suffix(".json")),
                                "final_drawer_fraction": attempt["final_drawer_fraction"],
                            }
                        )
            seed_summary[str(seed)] = {
                "successful_variants": sorted(successful_variants),
                "selected_unique_success_count": len(
                    [row for row in selected_rows if row["seed"] == seed]
                ),
                "attempts": attempts_for_seed,
                "status": (
                    "recoverable_under_some_teacher_family"
                    if successful_variants
                    else "confirmed_hard_seed_under_all_scanned_teacher_families"
                ),
            }

        selected_unique_live_episode_count = len(selected_rows)
        selected_seed_coverage = len({row["seed"] for row in selected_rows})
        if selected_unique_live_episode_count < 8:
            blocking.append("selected_unique_live_episode_count_lt_8")
        if selected_seed_coverage < 6:
            blocking.append("selected_seed_coverage_lt_6")
        if seed_summary["4"]["status"] not in {
            "confirmed_hard_seed_under_all_scanned_teacher_families",
            "recoverable_under_some_teacher_family",
        }:
            blocking.append("seed_4_status_missing")
        if seed_summary["2"]["status"] not in {
            "confirmed_hard_seed_under_all_scanned_teacher_families",
            "recoverable_under_some_teacher_family",
        }:
            blocking.append("seed_2_status_missing")
        if not any(
            row["teacher_family_variant"] != "base" for row in raw_attempt_rows
        ):
            blocking.append("dispatch_mechanism_failure")
    else:
        selected_unique_live_episode_count = 0
        selected_seed_coverage = 0

    artifact = {
        "audit": "re_c1_live_support_expansion",
        "run_instance_id": plan["run_instance_id"],
        "plan_version": plan["plan_version"],
        "working_head_commit": plan["working_head_commit"],
        "execution_scope": plan["execution_scope"],
        "diagnostic_only": True,
        "claim_bearing": False,
        "scan_depth_per_seed": SCAN_DEPTH,
        "selected_unique_live_episode_count": selected_unique_live_episode_count,
        "selected_seed_coverage": selected_seed_coverage,
        "selected_episode_list": selected_rows,
        "seed_summary": seed_summary,
        "raw_attempts": raw_attempt_rows,
        "re_c1_status": "PASS" if not blocking else "FAIL",
    }
    write_json_atomic(OUTPUT_PATH, artifact)

    gate = write_gate(
        gate_path=GATE_PATH,
        gate_id="Re-C1",
        gate_name="live_support_expansion",
        run_instance_id=plan["run_instance_id"],
        status="PASS" if not blocking else "STOP",
        blocking_reasons=blocking,
        allowed_next_phases=["Re-C2"] if not blocking else [],
        extra={
            "audit_artifact_path": str(OUTPUT_PATH),
            "selected_unique_live_episode_count": selected_unique_live_episode_count,
            "selected_seed_coverage": selected_seed_coverage,
            "seed_summary": seed_summary,
            "live_rollout_dir": str(ROLL_OUT_DIR),
        },
        spec_reference=SPEC_REFERENCE,
    )
    print(json.dumps({"artifact": artifact, "gate": gate}, indent=2))
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(run())
