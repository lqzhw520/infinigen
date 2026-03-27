#!/usr/bin/env python3
"""B2: Validate that AnyGrasp provides a usable non-zero scripted baseline."""

from __future__ import annotations

import json
import time

from mint_common import ARTIFACT_DIR, CAMPAIGN_DIR, validate_json_file
from video_reporting import (
    choose_best_success_rollout,
    safe_render_teacher_rollout,
    safe_render_teacher_rollout_report,
)

ARTIFACT = ARTIFACT_DIR / "b2_anygrasp_scripted_baseline.json"
MIN_SUCCESSFUL_SEEDS = 4
MIN_SUCCESSFUL_ROLLOUTS = 8


def run() -> bool:
    audit = (
        validate_json_file(ARTIFACT_DIR / "g4_oracle_vs_anygrasp_rollouts.json") or {}
    )
    anygrasp = audit.get("anygrasp", {})
    passed = bool(
        anygrasp.get("passed")
        and anygrasp.get("successful_seed_count", 0) >= MIN_SUCCESSFUL_SEEDS
        and anygrasp.get("successful_rollouts", 0) >= MIN_SUCCESSFUL_ROLLOUTS
    )
    result = {
        "gate": "b2_anygrasp_scripted_baseline",
        "passed": passed,
        "anygrasp": anygrasp,
        "timestamp": time.time(),
    }
    best_rollout = choose_best_success_rollout({"anygrasp": anygrasp}, "anygrasp")
    result["video_outputs"] = {
        "teacher_anygrasp": safe_render_teacher_rollout(
            campaign_dir=CAMPAIGN_DIR,
            rollout_path=best_rollout,
            stage="b2",
            policy="anygrasp_teacher",
            attempt="0",
            variant="scripted_baseline",
        ),
        "teacher_anygrasp_report_safe": safe_render_teacher_rollout_report(
            campaign_dir=CAMPAIGN_DIR,
            rollout_path=best_rollout,
            stage="b2",
            policy="anygrasp_teacher",
            attempt="0",
            variant="scripted_baseline",
        ),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return passed


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
