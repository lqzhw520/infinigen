#!/usr/bin/env python3
"""Plan-driven tiny retrain confirmation pipeline for canonical V1cT2S0."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from dataset_builder import (
    build_dataset_from_rollouts,
    validate_built_dataset_provenance,
    validate_rollout_for_canonical_training,
)
from mint_common import (
    ACTIVE_ACTION_CONTRACT_PATH,
    ARTIFACT_DIR,
    CAMPAIGN_DIR,
    DATASET_DIR,
    DATASET_REPO_ID,
    EVAL_DIR,
    PROJECT_ROOT,
    TINY_RETRAIN_DATASET_BUILD_PATH,
    TINY_RETRAIN_EVAL_DIR,
    TINY_RETRAIN_OUTPUT_DIR,
    TINY_RETRAIN_PLAN_PATH,
    TINY_RETRAIN_SUMMARY_PATH,
    load_json,
    write_json_atomic,
)
from tiny_retrain_mainline import (
    DEFAULT_ROLLOUT_SOURCE_DIR,
    MATERIALIZATION_ARTIFACT,
    TERMINAL_COMMIT,
    materialize_canonical_train_rollouts,
)

ROUTE_DECISION_PATH = CAMPAIGN_DIR / "autopilot" / "route_decision.json"
FINAL_RUN_SUMMARY_PATH = CAMPAIGN_DIR / "autopilot" / "final_run_summary.json"
CYCLE_STATE_PATH = CAMPAIGN_DIR / "autopilot" / "cycle_state.json"
RCA1_PATH = ARTIFACT_DIR / "p2rca1_true_handle_measurement_audit.json"
RCA2_PATH = ARTIFACT_DIR / "p2rca2_local_affordance_visual_attack.json"
RCA3_PATH = ARTIFACT_DIR / "p2rca3_affordance_plus_transition_contract.json"
RCA4_PATH = ARTIFACT_DIR / "p2rca4_affordance_plus_transition_state.json"
RCA5_PATH = ARTIFACT_DIR / "p2rca5_frozen_matrix_screen.json"
RCA6_PATH = ARTIFACT_DIR / "p2rca6_best_cell_replicate.json"
RCA7_PATH = ARTIFACT_DIR / "p2rca7_tiny_retrain_if_eligible.json"
REJECTED_ROLLOUTS_PATH = ARTIFACT_DIR / "g8_canonical_dataset_rejected_rollouts.json"
G8_SUMMARY_PATH = ARTIFACT_DIR / "g8_train_summary.json"
G8_PROBE_PATH = ARTIFACT_DIR / "g8_train_seed_probe.json"
G9_SUMMARY_PATH = EVAL_DIR / "comparison_summary.json"
G9_REPORT_PATH = EVAL_DIR / "comparison_report.md"
EXECUTION_MEMO_PATH = EVAL_DIR / "tiny_retrain_execution_memo.md"


def _repo_rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _git(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=PROJECT_ROOT, text=True).strip()


def load_terminal_artifacts() -> dict:
    return {
        "route_decision": load_json(ROUTE_DECISION_PATH, {}),
        "final_run_summary": load_json(FINAL_RUN_SUMMARY_PATH, {}),
        "cycle_state": load_json(CYCLE_STATE_PATH, {}),
        "rca1": load_json(RCA1_PATH, {}),
        "rca2": load_json(RCA2_PATH, {}),
        "rca3": load_json(RCA3_PATH, {}),
        "rca4": load_json(RCA4_PATH, {}),
        "rca5": load_json(RCA5_PATH, {}),
        "rca6": load_json(RCA6_PATH, {}),
        "rca7": load_json(RCA7_PATH, {}),
    }


def assert_terminal_ready(artifacts: dict) -> dict:
    route = artifacts["route_decision"]
    rca1 = artifacts["rca1"]
    rca2 = artifacts["rca2"]
    rca5 = artifacts["rca5"]
    rca6 = artifacts["rca6"]
    rca7 = artifacts["rca7"]
    checks = [
        (route.get("scientific_terminal_state") == "TS_CANONICAL_POSITIVE_ESTABLISHED", "route_terminal_state_mismatch"),
        (route.get("route_next_branch") == "tiny_retrain_confirmation", "route_next_branch_mismatch"),
        (bool(rca7.get("tiny_retrain_permitted", False)) is True, "rca7_tiny_retrain_not_permitted"),
        (str(rca7.get("canonical_train_cell")) == "V1cT2S0", "rca7_canonical_train_cell_mismatch"),
        (str(rca7.get("best_train_state_mode")) == "S0", "rca7_best_train_state_mode_mismatch"),
        (bool(rca6.get("replicate_positive", False)) is True, "rca6_replicate_positive_false"),
        (str(rca5.get("best_transition_cell")) == "V1cT2S0", "rca5_best_transition_cell_mismatch"),
        (bool(rca2.get("v1c_carrier_canonical_ready", False)) is True, "rca2_v1c_not_canonical_ready"),
        (bool(rca1.get("measurement_truthful_available", False)) is True, "rca1_measurement_not_truthful"),
    ]
    failures = [reason for passed, reason in checks if not passed]
    if failures:
        raise SystemExit(f"Terminal tiny retrain prerequisites not satisfied: {failures}")
    return {
        "canonical_train_cell": "V1cT2S0",
        "best_transition_cell": "V1cT2S0",
        "best_train_state_mode": "S0",
        "selector_mode": str(rca5.get("selector_mode") or rca7.get("selector_mode") or "frozen_v5_pro"),
        "frozen_matrix_hash": str(rca5.get("frozen_matrix_hash") or rca7.get("frozen_matrix_hash") or ""),
        "ts_baseline_cell_id": str(rca5.get("ts_baseline_cell_id") or rca7.get("ts_baseline_cell_id") or "V1cT0S0"),
        "measurement_truth_tier": str(artifacts["rca1"].get("measurement_truth_tier") or "manifest_entity_verified"),
        "measurement_backend": str(artifacts["rca1"].get("measurement_backend") or "segmentation_render"),
        "measurement_verifier": str((artifacts["rca1"].get("measurement_report") or {}).get("measurement_verifier") or artifacts["rca1"].get("measurement_verifier") or "isolated_rgb_threshold"),
        "runtime_visible_handle_mapping_source": str(artifacts["rca1"].get("runtime_visible_handle_mapping_source") or "collision_geom"),
    }


def materialize_active_tiny_retrain_plan(artifacts: dict) -> dict:
    ready = assert_terminal_ready(artifacts)
    branch = _git(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    head = _git(["git", "rev-parse", "HEAD"])
    plan = {
        "plan_version": "tiny_retrain_confirmation_v1",
        "source_branch": branch,
        "source_commit": head,
        "terminal_commit": TERMINAL_COMMIT,
        "scientific_terminal_state": "TS_CANONICAL_POSITIVE_ESTABLISHED",
        "route_next_branch": "tiny_retrain_confirmation",
        "canonical_train_cell": ready["canonical_train_cell"],
        "best_transition_cell": ready["best_transition_cell"],
        "best_train_state_mode": ready["best_train_state_mode"],
        "preferred_canonical_train_cell": "V1cT2S2",
        "ts_baseline_cell_id": ready["ts_baseline_cell_id"],
        "selector_mode": ready["selector_mode"],
        "frozen_matrix_hash": ready["frozen_matrix_hash"],
        "train_seeds": [1, 2, 3, 4, 5, 6, 7, 8],
        "heldout_seeds": [11, 12, 13, 14, 15],
        "active_action_contract_path": _repo_rel(ACTIVE_ACTION_CONTRACT_PATH),
        "dataset_root": _repo_rel(DATASET_DIR),
        "dataset_repo_id": DATASET_REPO_ID,
        "train_output_dir": _repo_rel(TINY_RETRAIN_OUTPUT_DIR / ready["canonical_train_cell"]),
        "evaluation_dir": _repo_rel(TINY_RETRAIN_EVAL_DIR / ready["canonical_train_cell"]),
        "measurement_truth_tier": ready["measurement_truth_tier"],
        "measurement_backend": ready["measurement_backend"],
        "measurement_verifier": ready["measurement_verifier"],
        "runtime_visible_handle_mapping_source": ready["runtime_visible_handle_mapping_source"],
        "train_steps": 1000,
        "batch_size": 8,
        "save_freq": 1000,
        "train_probe_min_success_gain": 0.15,
        "train_probe_min_successes": 2,
    }
    write_json_atomic(TINY_RETRAIN_PLAN_PATH, plan)
    return plan


def _find_rollout_refs(payload: Any) -> set[Path]:
    refs: set[Path] = set()
    if isinstance(payload, dict):
        for value in payload.values():
            refs.update(_find_rollout_refs(value))
    elif isinstance(payload, list):
        for value in payload:
            refs.update(_find_rollout_refs(value))
    elif isinstance(payload, str) and payload.endswith('.npz'):
        path = Path(payload)
        refs.add(path if path.is_absolute() else PROJECT_ROOT / path)
    return refs


def _load_validated_rollout_paths(paths: list[Path], plan: dict) -> tuple[list[Path], list[dict[str, Any]]]:
    valid: list[Path] = []
    rejected: list[dict[str, Any]] = []
    for npz_path in sorted(paths):
        meta_path = npz_path.with_suffix('.json')
        if not meta_path.exists():
            rejected.append({"npz_path": str(npz_path), "reason": "missing_meta_json"})
            continue
        meta = json.loads(meta_path.read_text())
        ok, reason = validate_rollout_for_canonical_training(meta, plan)
        if ok:
            valid.append(npz_path)
        else:
            rejected.append({"npz_path": str(npz_path), "reason": reason, "seed": meta.get("seed")})
    return valid, rejected


def build_or_refresh_canonical_dataset(plan: dict) -> dict:
    sources_checked: list[str] = []
    candidate_paths: set[Path] = set()
    for path in [RCA5_PATH, RCA6_PATH, CAMPAIGN_DIR / 'runtime' / 'controller_cycles' / 'cycle_20260414_134515' / 'lane_results.json']:
        payload = load_json(path, {})
        refs = _find_rollout_refs(payload)
        sources_checked.append(f"{path}:{len(refs)}refs")
        candidate_paths.update(refs)
    if DEFAULT_ROLLOUT_SOURCE_DIR.exists():
        sources_checked.append(f"{DEFAULT_ROLLOUT_SOURCE_DIR}:existing")
        candidate_paths.update(DEFAULT_ROLLOUT_SOURCE_DIR.glob('*.npz'))

    valid_rollouts, rejected = _load_validated_rollout_paths(sorted(candidate_paths), plan)
    unique_valid_seeds = {int(json.loads(path.with_suffix('.json').read_text()).get('seed')) for path in valid_rollouts if path.with_suffix('.json').exists()}
    if len(valid_rollouts) == 0 or len(unique_valid_seeds) == 0:
        materialization = materialize_canonical_train_rollouts(
            DEFAULT_ROLLOUT_SOURCE_DIR,
            expected=plan,
            force_rebuild=True,
        )
        if not materialization.get('passed'):
            raise SystemExit(f"Canonical rollout materialization failed: {materialization.get('error')}")
        valid_rollouts, rejected = _load_validated_rollout_paths(sorted(DEFAULT_ROLLOUT_SOURCE_DIR.glob('*.npz')), plan)
    else:
        materialization = load_json(MATERIALIZATION_ARTIFACT, {})

    if not valid_rollouts:
        raise SystemExit('No valid canonical rollouts remain after validation')

    dataset_root = _resolve_repo_path(plan['dataset_root'])
    payload = build_dataset_from_rollouts(valid_rollouts, dataset_root, str(plan['dataset_repo_id']))
    provenance_report = validate_built_dataset_provenance(dataset_root, plan)
    build_report = {
        **provenance_report,
        "dataset_root": str(dataset_root),
        "dataset_repo_id": str(plan['dataset_repo_id']),
        "rejected_rollout_count": len(rejected),
        "sources_checked": sources_checked,
        "materialization": materialization,
        "used_rollout_paths": [str(path) for path in valid_rollouts],
        "integrity": payload.get("integrity", {}),
    }
    write_json_atomic(REJECTED_ROLLOUTS_PATH, {"rejected_rollouts": rejected})
    write_json_atomic(TINY_RETRAIN_DATASET_BUILD_PATH, build_report)
    if not build_report.get("dataset_valid"):
        raise SystemExit(f"Canonical dataset build failed validation: {build_report.get('errors', [])}")
    return build_report


def launch_tiny_retrain(plan: dict) -> dict:
    from run_g8_mint_train import run as run_train

    if not run_train():
        return load_json(G8_SUMMARY_PATH, {})
    return load_json(G8_SUMMARY_PATH, {})


def run_train_probe(plan: dict, train_summary: dict) -> dict:
    from run_g8_train_seed_probe import run as run_probe

    if not bool(train_summary.get("passed", False)):
        raise SystemExit("Cannot run train probe without a passed train summary")
    run_probe()
    return load_json(G8_PROBE_PATH, {})


def run_heldout_eval(plan: dict, train_summary: dict, probe_summary: dict) -> dict | None:
    from run_g9_sim_eval import run as run_eval

    if not bool(probe_summary.get("trend_passed", False)):
        return None
    if not bool(train_summary.get("passed", False)):
        raise SystemExit("Cannot run held-out eval without a passed train summary")
    run_eval()
    return load_json(G9_SUMMARY_PATH, {})


def write_skipped_heldout_eval_summary(plan: dict, probe_summary: dict) -> dict:
    summary = {
        "gate": "g9_sim_eval",
        "training_mode": "tiny_retrain_confirmation",
        "canonical_train_cell": plan.get("canonical_train_cell"),
        "best_train_state_mode": plan.get("best_train_state_mode"),
        "heldout_eval_run": False,
        "claim_supported": False,
        "verdict": "train_probe_not_confirmed",
        "reason": "Train probe failed; held-out eval skipped by v7 stop rule",
        "trend_passed": bool(probe_summary.get("trend_passed", False)),
        "held_out_seeds": plan.get("heldout_seeds", []),
    }
    write_json_atomic(G9_SUMMARY_PATH, summary)
    G9_REPORT_PATH.write_text("# Held-out Eval Skipped\n\nTrain probe did not pass, so held-out eval was not executed under v7.\n")
    return summary


def write_execution_memo(plan: dict, dataset_summary: dict, summary: dict) -> None:
    lines = [
        "# Tiny Retrain Execution Memo",
        "",
        f"- dataset 是否只来自 `V1cT2S0`：{'是' if dataset_summary.get('canonical_train_cell') == 'V1cT2S0' else '否'}",
        f"- 是否严格使用 `S0`：{'是' if plan.get('best_train_state_mode') == 'S0' else '否'}",
        f"- 是否只用 train seeds `[1..8]`：{'是' if plan.get('train_seeds') == [1,2,3,4,5,6,7,8] else '否'}",
        f"- held-out eval 是否只用 `[11..15]`：{'是' if plan.get('heldout_seeds') == [11,12,13,14,15] and summary.get('heldout_eval_run') else '未运行'}",
        f"- 最终 verdict：`{summary.get('final_verdict')}`",
    ]
    EXECUTION_MEMO_PATH.write_text("\n".join(lines) + "\n")


def write_tiny_retrain_confirmation_summary(
    plan: dict,
    dataset_summary: dict,
    train_summary: dict,
    probe_summary: dict,
    eval_summary: dict | None,
) -> dict:
    train_probe_passed = bool(probe_summary.get("trend_passed", False))
    heldout_eval_run = bool(train_probe_passed and eval_summary and eval_summary.get("heldout_eval_run", True))
    claim_supported = bool(heldout_eval_run and eval_summary and eval_summary.get("verdict") == "claim_supported")
    if not train_probe_passed:
        final_verdict = "train_probe_not_confirmed"
        scientific_terminal_state = "TINY_RETRAIN_NOT_CONFIRMED_ON_TRAIN_PROBE"
    elif claim_supported:
        final_verdict = "claim_supported"
        scientific_terminal_state = None
    else:
        final_verdict = "scientific_not_supported"
        scientific_terminal_state = None
    summary = {
        "confirmation_mode": "tiny_retrain_confirmation",
        "source_branch": plan.get("source_branch"),
        "terminal_commit": plan.get("terminal_commit", TERMINAL_COMMIT),
        "canonical_train_cell": plan.get("canonical_train_cell"),
        "best_train_state_mode": plan.get("best_train_state_mode"),
        "dataset_valid": bool(dataset_summary.get("dataset_valid", False)),
        "train_summary_available": bool(train_summary),
        "train_probe_passed": train_probe_passed,
        "heldout_eval_run": heldout_eval_run,
        "claim_supported": claim_supported,
        "final_verdict": final_verdict,
        "scientific_terminal_state": scientific_terminal_state,
        "checkpoint_path": train_summary.get("checkpoint_path"),
        "train_steps": int(plan.get("train_steps", 1000)),
        "train_seeds": plan.get("train_seeds", []),
        "heldout_seeds": plan.get("heldout_seeds", []),
    }
    if heldout_eval_run and eval_summary:
        comparison = eval_summary.get("comparison", {})
        summary.update({
            "finetuned_success_rate": (comparison.get("finetuned_mint") or {}).get("success_rate"),
            "pretrained_success_rate": (comparison.get("pretrained_mint") or {}).get("success_rate"),
            "random_success_rate": (comparison.get("random") or {}).get("success_rate"),
        })
    write_json_atomic(TINY_RETRAIN_SUMMARY_PATH, summary)
    write_execution_memo(plan, dataset_summary, summary)
    return summary


def _phase_prepare() -> tuple[dict, dict]:
    artifacts = load_terminal_artifacts()
    plan = materialize_active_tiny_retrain_plan(artifacts)
    dataset_summary = build_or_refresh_canonical_dataset(plan)
    return plan, dataset_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prepare", "train", "probe", "eval", "finalize", "all"], default="all")
    args = parser.parse_args()

    if args.phase == "prepare":
        plan, dataset_summary = _phase_prepare()
        print(json.dumps({"phase": "prepare", "plan": plan, "dataset_summary": dataset_summary}, indent=2))
        return 0

    plan = load_json(TINY_RETRAIN_PLAN_PATH, {})
    dataset_summary = load_json(TINY_RETRAIN_DATASET_BUILD_PATH, {})
    train_summary = load_json(G8_SUMMARY_PATH, {})
    probe_summary = load_json(G8_PROBE_PATH, {})
    eval_summary = load_json(G9_SUMMARY_PATH, {})

    if args.phase == "train":
        if not plan or not dataset_summary:
            plan, dataset_summary = _phase_prepare()
        train_summary = launch_tiny_retrain(plan)
        print(json.dumps(train_summary, indent=2))
        return 0 if train_summary.get("passed") else 1

    if args.phase == "probe":
        if not train_summary:
            raise SystemExit("Missing g8_train_summary.json before probe")
        probe_summary = run_train_probe(plan, train_summary)
        print(json.dumps(probe_summary, indent=2))
        return 0 if probe_summary.get("trend_passed") else 1

    if args.phase == "eval":
        if not probe_summary or not bool(probe_summary.get("trend_passed", False)):
            raise SystemExit("Train probe must pass before held-out eval")
        eval_summary = run_heldout_eval(plan, train_summary, probe_summary)
        print(json.dumps(eval_summary or {}, indent=2))
        return 0 if eval_summary else 1

    if args.phase == "finalize":
        if probe_summary and not bool(probe_summary.get("trend_passed", False)):
            eval_summary = write_skipped_heldout_eval_summary(plan, probe_summary)
        elif probe_summary and bool(probe_summary.get("trend_passed", False)) and not eval_summary:
            raise SystemExit("Held-out eval summary missing after a passed train probe")
        summary = write_tiny_retrain_confirmation_summary(plan, dataset_summary, train_summary, probe_summary, eval_summary if eval_summary else None)
        print(json.dumps(summary, indent=2))
        return 0

    # all
    plan, dataset_summary = _phase_prepare()
    train_summary = launch_tiny_retrain(plan)
    if not bool(train_summary.get("passed", False)):
        summary = write_tiny_retrain_confirmation_summary(plan, dataset_summary, train_summary, {}, None)
        print(json.dumps(summary, indent=2))
        return 1
    probe_summary = run_train_probe(plan, train_summary)
    if not bool(probe_summary.get("trend_passed", False)):
        eval_summary = write_skipped_heldout_eval_summary(plan, probe_summary)
        summary = write_tiny_retrain_confirmation_summary(plan, dataset_summary, train_summary, probe_summary, eval_summary)
        print(json.dumps(summary, indent=2))
        return 1
    eval_summary = run_heldout_eval(plan, train_summary, probe_summary)
    summary = write_tiny_retrain_confirmation_summary(plan, dataset_summary, train_summary, probe_summary, eval_summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("final_verdict") == "claim_supported" else 1


if __name__ == "__main__":
    raise SystemExit(main())
