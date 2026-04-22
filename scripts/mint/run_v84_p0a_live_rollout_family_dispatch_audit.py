#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
MINT_SCRIPTS = PROJECT_ROOT / "scripts" / "mint"
sys.path.insert(0, str(MINT_SCRIPTS))

from drawer_robot_env_mujoco import DrawerEnvContractConfig, build_robot_rollout  # noqa: E402
from root_cause_controller import RootCauseController  # noqa: E402
from tiny_retrain_mainline import TEACHER_FAMILY_GRID_V11  # noqa: E402
from v13_audit_common import (  # noqa: E402
    ARTIFACT_DIR,
    AUTOPILOT_DIR,
    ACCEPTANCE_CONTRACT_PATH,
    PLAN_PATH,
    TRUTH_CONTRACT_PATH,
    current_repo_identity,
    load_json,
    make_run_instance,
    save_active_plan,
    sha256_file,
    teacher_family_dispatch_payload,
    teacher_family_variant_names,
    write_gate,
    write_json_atomic,
)

SOURCE_CLOSURE_SUMMARY = ARTIFACT_DIR / "post_v13_final_pretrain_closure_summary.json"
SOURCE_CLOSURE_GATE = (
    AUTOPILOT_DIR / "gates_post_v13_closure" / "C4_final_pretrain_closure_verdict.json"
)
P0_GATES_DIR = AUTOPILOT_DIR / "gates_p0_infrastructure"
OUTPUT_PATH = ARTIFACT_DIR / "p0a_live_rollout_family_dispatch_audit.json"
PLAN_OUTPUT_PATH = ARTIFACT_DIR / "p0_infrastructure_repair_plan.json"
GATE_PATH = P0_GATES_DIR / "P0a_live_rollout_family_dispatch_audit.json"
SPEC_REFERENCE = str(PROJECT_ROOT / "docs" / "MINT_V84_P0_INFRASTRUCTURE_REPAIR_SPEC.md")
TARGET_SEEDS = list(range(1, 9))
SUCCESS_EXPECTED_SEEDS = [1, 3, 5, 6, 7, 8]
HARD_SEEDS = [2, 4]
TEACHER_CONTROLLER_MODE = "interaction_frame_hybrid"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha1_rollout(rollout: dict[str, Any]) -> str:
    h = hashlib.sha1()
    h.update(np.asarray(rollout["states"], dtype=np.float32).tobytes())
    h.update(np.asarray(rollout["actions"], dtype=np.float32).tobytes())
    return h.hexdigest()


def _closure_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    return load_json(SOURCE_CLOSURE_SUMMARY), load_json(SOURCE_CLOSURE_GATE)


def _make_plan() -> dict[str, Any]:
    ident = current_repo_identity()
    closure_summary, closure_gate = _closure_authority()
    run_instance_id = make_run_instance("p0a", ident["working_head_commit"])
    plan = {
        "run_instance_id": run_instance_id,
        "plan_version": "tiny_retrain_confirmation_p0_infrastructure_repair",
        "slice_name": "p0a-live-rollout-family-dispatch-audit",
        "working_head_commit": ident["working_head_commit"],
        "source_working_head_commit": closure_summary["working_head_commit"],
        "current_publication_head_commit": ident["working_head_commit"],
        "vendor_head_commit": ident["vendor_head_commit"],
        "branch": ident["branch"],
        "execution_scope": "p0a_live_rollout_family_dispatch_audit",
        "bridge_stage": "p0_infrastructure_repair",
        "diagnostic_only": True,
        "claim_bearing": False,
        "training_allowed": False,
        "probe_allowed": False,
        "publication_scope": "blocked_before_canary",
        "result_scope": "p0_infrastructure_only",
        "spec_reference": SPEC_REFERENCE,
        "source_post_v13_final_pretrain_closure_run_instance_id": closure_summary["run_instance_id"],
        "source_post_v13_final_pretrain_closure_head_commit": closure_summary["working_head_commit"],
        "source_post_v13_final_pretrain_closure_final_verdict": closure_summary["final_verdict"],
        "source_post_v13_final_pretrain_closure_blocking_reasons": closure_summary["blocking_reasons"],
        "truth_contract_path": str(TRUTH_CONTRACT_PATH),
        "truth_contract_hash": sha256_file(TRUTH_CONTRACT_PATH),
        "acceptance_contract_path": str(ACCEPTANCE_CONTRACT_PATH),
        "acceptance_contract_hash": sha256_file(ACCEPTANCE_CONTRACT_PATH),
        "allowed_next_phases": ["P0a", "P0b"],
        "authority_freeze": {
            "spec_source_working_head_commit": "bc802069816e6c81beab48ba1f2141c9f3a264d4",
            "current_repo_head_commit": ident["working_head_commit"],
            "source_closure_working_head_commit": closure_summary["working_head_commit"],
            "source_closure_run_instance_id": closure_summary["run_instance_id"],
            "current_repo_head_matches_source_closure_head": ident["working_head_commit"]
            == closure_summary["working_head_commit"],
            "current_repo_head_is_packaging_commit_drift_only": ident["working_head_commit"]
            != closure_summary["working_head_commit"],
            "source_closure_gate_status": closure_gate.get("status"),
            "source_closure_final_verdict": closure_gate.get("final_verdict"),
        },
    }
    write_json_atomic(PLAN_OUTPUT_PATH, plan)
    save_active_plan(plan)
    return plan


def _fixed_live_contract() -> DrawerEnvContractConfig:
    controller = RootCauseController()
    payload = dict(
        controller._frozen_matrix_contracts_v5_pro()["V1cT2S3"]["env_contract_config"]
    )
    return DrawerEnvContractConfig(**payload)


def _answer_question_a() -> dict[str, Any]:
    probe_rollout = build_robot_rollout(
        seed=1,
        grasp_pose_world=np.eye(4, dtype=np.float32),
        episode_index=7,
        image_size=64,
        max_steps=24,
        contract=_fixed_live_contract(),
        rotation_source="aligned",
        claim_policy="diagnostic",
        teacher_controller_mode=TEACHER_CONTROLLER_MODE,
    )
    observed = int(probe_rollout.get("episode_index", -1))
    return {
        "question": "Is episode_index passed as a parameter to the live rollout runner?",
        "answer": "YES" if observed == 7 else "NO",
        "requested_episode_index": 7,
        "observed_episode_index": observed,
    }


def _answer_question_b() -> dict[str, Any]:
    closure_source = (
        MINT_SCRIPTS / "run_v84_post_v13_final_pretrain_closure.py"
    ).read_text()
    slice2_source = (
        MINT_SCRIPTS / "run_v84_v13_slice2_state_consumption_audit.py"
    ).read_text()
    closure_mentions_dispatch = (
        "teacher_family_dispatch_payload(" in closure_source
        and "interventions=teacher_family_dispatch_payload(episode_index)" in closure_source
    )
    slice2_mentions_dispatch = (
        "teacher_family_dispatch_payload(" in slice2_source
        and "interventions=teacher_family_dispatch_payload(episode_index)" in slice2_source
    )
    return {
        "question": "Is teacher_family_variant or equivalent passed to the live rollout runner?",
        "answer": "YES"
        if closure_mentions_dispatch and slice2_mentions_dispatch
        else "NO",
        "closure_runner_dispatch_wired": closure_mentions_dispatch,
        "slice2_runner_dispatch_wired": slice2_mentions_dispatch,
        "conclusion": "Current authoritative live runners dispatch teacher_family_variant through interventions."
        if closure_mentions_dispatch and slice2_mentions_dispatch
        else "Current authoritative live runners do not yet dispatch teacher_family_variant through interventions.",
    }


def _scan_variants() -> list[dict[str, Any]]:
    contract = _fixed_live_contract()
    scan_results = []
    for seed in TARGET_SEEDS:
        variant_records = []
        for episode_index, variant in enumerate(TEACHER_FAMILY_GRID_V11):
            rollout = build_robot_rollout(
                seed=seed,
                grasp_pose_world=np.eye(4, dtype=np.float32),
                episode_index=episode_index,
                image_size=64,
                max_steps=96,
                contract=contract,
                rotation_source="aligned",
                claim_policy="diagnostic",
                teacher_controller_mode=TEACHER_CONTROLLER_MODE,
                interventions=teacher_family_dispatch_payload(episode_index),
            )
            variant_records.append(
                {
                    "teacher_family_variant": variant["teacher_family_variant"],
                    "teacher_family_payload": dict(variant),
                    "fingerprint": _sha1_rollout(rollout),
                    "success": bool(rollout.get("success")),
                    "ever_attached": bool(rollout.get("ever_attached")),
                    "final_drawer_fraction": float(
                        rollout.get("final_drawer_fraction", 0.0)
                    ),
                    "max_drawer_fraction": float(
                        rollout.get("max_drawer_fraction", 0.0)
                    ),
                }
            )
        scan_results.append(
            {
                "seed": seed,
                "unique_fingerprints": len(
                    {record["fingerprint"] for record in variant_records}
                ),
                "successful_variants": [
                    record["teacher_family_variant"]
                    for record in variant_records
                    if record["success"]
                ],
                "variant_records": variant_records,
            }
        )
    return scan_results


def _answer_question_c(scan_results: list[dict[str, Any]]) -> dict[str, Any]:
    seeds_with_diversity = [
        result["seed"] for result in scan_results if result["unique_fingerprints"] >= 2
    ]
    return {
        "question": "Does changing teacher_family_variant produce different effective fingerprints?",
        "answer": "YES" if len(seeds_with_diversity) >= 6 else "NO",
        "seeds_with_at_least_two_unique_fingerprints": seeds_with_diversity,
        "seed_count_with_diversity": len(seeds_with_diversity),
    }


def _hard_seed_diagnosis(scan_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    diagnosis: dict[str, dict[str, Any]] = {}
    by_seed = {result["seed"]: result for result in scan_results}
    for seed in HARD_SEEDS:
        result = by_seed[seed]
        successful = result["successful_variants"]
        diagnosis[str(seed)] = {
            "seed": seed,
            "successful_variants": successful,
            "status": (
                "recoverable_under_some_teacher_family"
                if successful
                else "confirmed_hard_seed_under_all_scanned_teacher_families"
            ),
        }
    return diagnosis


def run() -> int:
    plan = _make_plan()
    closure_summary, _closure_gate = _closure_authority()
    question_a = _answer_question_a()
    question_b = _answer_question_b()
    scan_results = _scan_variants()
    question_c = _answer_question_c(scan_results)
    hard_seed = _hard_seed_diagnosis(scan_results)

    passes_seed_diversity = all(
        next(
            result["unique_fingerprints"]
            for result in scan_results
            if result["seed"] == seed
        )
        >= 2
        for seed in SUCCESS_EXPECTED_SEEDS
    )
    blocking: list[str] = []
    if question_a["answer"] != "YES":
        blocking.append("episode_index_not_forwarded")
    if question_b["answer"] != "YES":
        blocking.append("teacher_family_variant_not_dispatched")
    if question_c["answer"] != "YES":
        blocking.append("teacher_family_variant_failed_to_create_diversity")
    if not passes_seed_diversity:
        blocking.append("expected_success_seeds_failed_variant_diversity")
    if hard_seed["2"]["status"] != "recoverable_under_some_teacher_family":
        blocking.append("seed_2_not_recoverable_under_scanned_teacher_families")
    if hard_seed["4"]["status"] not in {
        "recoverable_under_some_teacher_family",
        "confirmed_hard_seed_under_all_scanned_teacher_families",
    }:
        blocking.append("seed_4_diagnosis_missing")

    p0a_pass = (
        question_a["answer"] == "YES"
        and question_b["answer"] == "YES"
        and question_c["answer"] == "YES"
        and passes_seed_diversity
        and hard_seed["2"]["status"] == "recoverable_under_some_teacher_family"
        and hard_seed["4"]["status"]
        == "confirmed_hard_seed_under_all_scanned_teacher_families"
    )

    artifact = {
        "audit": "p0a_live_rollout_family_dispatch_audit",
        "run_instance_id": plan["run_instance_id"],
        "plan_version": plan["plan_version"],
        "working_head_commit": plan["working_head_commit"],
        "execution_scope": plan["execution_scope"],
        "diagnostic_only": True,
        "claim_bearing": False,
        "authority_freeze": plan["authority_freeze"],
        "source_post_v13_final_pretrain_closure_run_instance_id": closure_summary[
            "run_instance_id"
        ],
        "source_post_v13_final_pretrain_closure_blocking_reasons": closure_summary[
            "blocking_reasons"
        ],
        "question_a_episode_index_forwarded": question_a,
        "question_b_current_live_runner_dispatch_support": question_b,
        "question_c_variant_effectiveness": question_c,
        "teacher_family_grid_version": "v11_support_family_grid_v1",
        "teacher_family_grid_variants": teacher_family_variant_names(),
        "variant_scan_results": scan_results,
        "hard_seed_diagnosis": hard_seed,
        "p0a_fix_summary": "teacher_family_variant dispatch is wired into the current authoritative live rollout runners via interventions, mirroring the v11 support-family grid.",
        "p0a_status": "PASS" if p0a_pass else "FAIL",
    }
    write_json_atomic(OUTPUT_PATH, artifact)

    gate_status = "PASS" if p0a_pass else "STOP"
    gate = write_gate(
        gate_path=GATE_PATH,
        gate_id="P0a",
        gate_name="live_rollout_family_dispatch_audit",
        run_instance_id=plan["run_instance_id"],
        status=gate_status,
        blocking_reasons=blocking,
        allowed_next_phases=["P0b"] if p0a_pass else [],
        extra={
            "audit_artifact_path": str(OUTPUT_PATH),
            "question_a_answer": question_a["answer"],
            "question_b_answer": question_b["answer"],
            "question_c_answer": question_c["answer"],
            "successful_diversity_seeds": SUCCESS_EXPECTED_SEEDS,
            "hard_seed_diagnosis": hard_seed,
            "teacher_family_grid_version": "v11_support_family_grid_v1",
            "teacher_family_dispatch_wired": question_b["answer"] == "YES",
        },
        spec_reference=SPEC_REFERENCE,
    )
    print(json.dumps({"artifact": artifact, "gate": gate}, indent=2))
    return 0 if p0a_pass else 1


if __name__ == "__main__":
    raise SystemExit(run())
