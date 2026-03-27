#!/usr/bin/env python3
"""B1: Validate that oracle scripted robot opening is stably achievable in sim."""

from __future__ import annotations

import json
import time

from mint_common import ARTIFACT_DIR, CAMPAIGN_DIR, validate_json_file
from video_reporting import (
    choose_best_success_rollout,
    safe_render_teacher_rollout,
    safe_render_teacher_rollout_report,
)

ARTIFACT = ARTIFACT_DIR / "b1_oracle_scripted_baseline.json"
MIN_SUCCESSFUL_SEEDS = 6
MIN_SUCCESSFUL_ROLLOUTS = 12


def run() -> bool:
    audit = (
        validate_json_file(ARTIFACT_DIR / "g4_oracle_vs_anygrasp_rollouts.json") or {}
    )
    oracle = audit.get("oracle", {})
    passed = bool(
        oracle.get("passed")
        and oracle.get("successful_seed_count", 0) >= MIN_SUCCESSFUL_SEEDS
        and oracle.get("successful_rollouts", 0) >= MIN_SUCCESSFUL_ROLLOUTS
    )
    result = {
        "gate": "b1_oracle_scripted_baseline",
        "passed": passed,
        "oracle": oracle,
        "timestamp": time.time(),
    }
    best_rollout = choose_best_success_rollout({"oracle": oracle}, "oracle")
    result["video_outputs"] = {
        "teacher_oracle": safe_render_teacher_rollout(
            campaign_dir=CAMPAIGN_DIR,
            rollout_path=best_rollout,
            stage="b1",
            policy="oracle_teacher",
            attempt="0",
            variant="scripted_baseline",
        ),
        "teacher_oracle_report_safe": safe_render_teacher_rollout_report(
            campaign_dir=CAMPAIGN_DIR,
            rollout_path=best_rollout,
            stage="b1",
            policy="oracle_teacher",
            attempt="0",
            variant="scripted_baseline",
        ),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return passed


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
