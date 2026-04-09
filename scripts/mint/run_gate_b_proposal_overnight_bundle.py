#!/usr/bin/env python3
"""
Proposal-only overnight launcher for Gate B follow-up diagnostics.

This does not mutate canonical truth. It runs proposal-only audits, snapshots
night-runner status, and records a single report JSON under sovereign/night.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN_ROOT = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1"
SOVEREIGN = CAMPAIGN_ROOT / "sovereign"
NIGHT_DIR = SOVEREIGN / "night"
SCRIPTS_MINT = PROJECT_ROOT / "scripts" / "mint"
HARNESS = CAMPAIGN_ROOT / "scripts" / "harness"

REPORT_PATH = NIGHT_DIR / "gate_b_proposal_overnight_last.json"
LOG_PATH = NIGHT_DIR / "gate_b_proposal_overnight.log"


def now_ts() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_shell(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )


def main() -> int:
    NIGHT_DIR.mkdir(parents=True, exist_ok=True)
    steps = [
        {
            "id": "night_runner_status_before",
            "command": f"{sys.executable} {HARNESS / 'v59_night_runner.py'} --status",
        },
        {
            "id": "compile_gate_b9",
            "command": f"{sys.executable} -m py_compile {SCRIPTS_MINT / 'run_gate_b_seed_retarget_discrepancy.py'}",
        },
        {
            "id": "go_before",
            "command": f"{sys.executable} {HARNESS / 'sovereign_cli.py'} go",
        },
        {
            "id": "run_gate_b9",
            "command": f"{sys.executable} {SCRIPTS_MINT / 'run_gate_b_seed_retarget_discrepancy.py'}",
        },
        {
            "id": "go_after",
            "command": f"{sys.executable} {HARNESS / 'sovereign_cli.py'} go",
        },
        {
            "id": "night_runner_status_after",
            "command": f"{sys.executable} {HARNESS / 'v59_night_runner.py'} --status",
        },
    ]

    payload: dict[str, object] = {
        "experiment": "gate_b_proposal_overnight_bundle",
        "timestamp": now_ts(),
        "proposal_only": True,
        "artifacts_expected": [
            str(CAMPAIGN_ROOT / "artifacts" / "gate_b9_seed_retarget_discrepancy" / "gate_b9_summary.json"),
            str(CAMPAIGN_ROOT / "sovereign" / "proposals" / "E033_gate_b_seed_retarget_discrepancy_draft.yaml"),
        ],
        "steps": [],
    }

    with LOG_PATH.open("a") as log_handle:
        log_handle.write(f"[{now_ts()}] gate_b_proposal_overnight_bundle start\n")
        log_handle.flush()
        REPORT_PATH.write_text(json.dumps({**payload, "status": "running"}, indent=2, ensure_ascii=False))
        for step in steps:
            result = run_shell(step["command"])
            step_payload = {
                "id": step["id"],
                "command": step["command"],
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-4000:],
                "stderr_tail": result.stderr[-4000:],
            }
            payload["steps"].append(step_payload)
            log_handle.write(
                f"[{now_ts()}] step={step['id']} rc={result.returncode}\n"
            )
            if result.stdout:
                log_handle.write(result.stdout[-4000:] + "\n")
            if result.stderr:
                log_handle.write(result.stderr[-4000:] + "\n")
            log_handle.flush()
            REPORT_PATH.write_text(
                json.dumps(
                    {
                        **payload,
                        "status": "running" if result.returncode == 0 else "failed",
                        "active_step": step["id"],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            if result.returncode != 0:
                payload["status"] = "failed"
                payload["failed_step"] = step["id"]
                REPORT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
                print(json.dumps(payload, indent=2, ensure_ascii=False))
                return result.returncode

        payload["status"] = "completed"
        payload["completed_at"] = now_ts()
        REPORT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
