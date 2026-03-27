#!/usr/bin/env python3
"""U5: Audit whether failures are concentrated in a few seeds or indicate a systemic upstream issue."""

from __future__ import annotations

import json
import time
from collections import defaultdict

from mint_common import (
    ARTIFACT_DIR,
    DEFAULT_HELD_OUT_SEEDS,
    DEFAULT_TRAIN_SEEDS,
    validate_json_file,
)
from upstream_audit_helpers import classify_from_flags

ARTIFACT = ARTIFACT_DIR / "u5_seed_outlier_audit.json"


def _records_by_seed(block: dict, key: str) -> dict[int, dict]:
    result = {}
    for item in block.get(key, {}).get("records", []):
        result[int(item["seed"])] = item
    return result


def run() -> bool:
    b1 = validate_json_file(ARTIFACT_DIR / "b1_oracle_scripted_baseline.json") or {}
    b2 = validate_json_file(ARTIFACT_DIR / "b2_anygrasp_scripted_baseline.json") or {}
    c1 = (
        validate_json_file(ARTIFACT_DIR / "c1_teacher_native_rollout_rebuild.json")
        or {}
    )
    c2 = validate_json_file(ARTIFACT_DIR / "c2_action_contract_repair.json") or {}
    oracle = _records_by_seed(b1, "oracle")
    anygrasp = _records_by_seed(b2, "anygrasp")
    c1_records = {}
    for leaderboard_row in c1.get("leaderboard", []):
        if (
            leaderboard_row.get("branch_id") == "branch4_densified_attach_window"
            and leaderboard_row.get("grasp_mode") == "oracle_handle"
        ):
            c1_records = {
                int(item["seed"]): item for item in leaderboard_row.get("records", [])
            }
            break
    c2_records = defaultdict(dict)
    for branch in c2.get("results", []):
        if branch.get("branch_id") == c2.get("selected_branch"):
            for item in branch.get("teacher_summary", {}).get("records", []):
                c2_records[int(item["seed"])] = item
            break

    per_seed = []
    flags = []
    failed_oracle = []
    failed_all = []
    failed_anygrasp = []
    oracle_only_fail = []
    for seed in DEFAULT_TRAIN_SEEDS + DEFAULT_HELD_OUT_SEEDS:
        oracle_successes = int(oracle.get(seed, {}).get("successful_attempts", 0))
        anygrasp_successes = int(anygrasp.get(seed, {}).get("successful_attempts", 0))
        c1_successes = int(c1_records.get(seed, {}).get("successes", 0))
        c2_successes = int(c2_records.get(seed, {}).get("successes", 0))
        row = {
            "seed": seed,
            "split": "train" if seed in DEFAULT_TRAIN_SEEDS else "held_out",
            "oracle_successes": oracle_successes,
            "anygrasp_successes": anygrasp_successes,
            "c1_teacher_successes": c1_successes,
            "c2_teacher_successes": c2_successes,
        }
        per_seed.append(row)
        if seed in DEFAULT_TRAIN_SEEDS and oracle_successes == 0:
            failed_oracle.append(seed)
            if anygrasp_successes > 0 or c2_successes > 0:
                oracle_only_fail.append(seed)
        if seed in DEFAULT_TRAIN_SEEDS and anygrasp_successes == 0:
            failed_anygrasp.append(seed)
        if (
            seed in DEFAULT_TRAIN_SEEDS
            and oracle_successes == 0
            and anygrasp_successes == 0
            and c1_successes == 0
            and c2_successes == 0
        ):
            failed_all.append(seed)
    if failed_all:
        flags.append("oracle_failures_suggest_nonlearning_issue_suspect")
    elif failed_anygrasp and len(failed_anygrasp) <= 3:
        flags.append("few_bad_seeds_suspect")
    decision = classify_from_flags(flags)
    root_cause = (
        "systemic_upstream_issue"
        if failed_all
        else "oracle_handle_heuristic_mismatch"
        if oracle_only_fail
        else "few_bad_seeds"
        if failed_anygrasp and len(failed_anygrasp) <= 3
        else "non-upstream_issue"
    )
    result = {
        "gate": "u5_seed_outlier_audit",
        "passed": decision == "upstream_clean",
        "decision": decision,
        "root_cause": root_cause,
        "affected_seeds": failed_all or oracle_only_fail or failed_anygrasp,
        "oracle_only_fail_seeds": oracle_only_fail,
        "failed_all_modalities_seeds": failed_all,
        "flags": flags,
        "per_seed": per_seed,
        "conclusion_text": (
            "Observed failures are not concentrated in a way that forces an upstream Infinigen explanation."
            if decision == "upstream_clean"
            else "Seed-level behavior suggests either a few bad generated assets or a broader upstream/export issue that must be isolated before claim verdicts."
        ),
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return True


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
