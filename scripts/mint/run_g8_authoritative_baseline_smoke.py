#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from mint_common import ARTIFACT_DIR, PROJECT_ROOT, write_json_atomic

OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1" / "outputs"
from run_p1c10_release_runtime_matched_ab import AUTHORITATIVE_MINT_PYTHON, MINT_REPO_ROOT

ARTIFACT_NAME = "g8_authoritative_baseline_smoke.json"
OUTPUT_DIR = OUTPUT_ROOT / "g8_authoritative_baseline_smoke"
VARIANT_OUTPUT_DIR = OUTPUT_DIR / "current_vendor_authoritative_smoke"
SUMMARY_PATH = VARIANT_OUTPUT_DIR / "variant_summary.json"
FAILURE_PATH = VARIANT_OUTPUT_DIR / "variant_failure.json"
EVAL_INFO_PATH = VARIANT_OUTPUT_DIR / "eval_info.json"
ENTRYPOINT = PROJECT_ROOT / "scripts" / "mint" / "run_p1c10_release_runtime_matched_ab.py"
VENDOR_SRC = MINT_REPO_ROOT / "lerobot_policy_mint" / "src"
FROZEN_VENDOR_HEAD = "4eab5795345721001c412ff1ca2c886a11eab606"


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> int:
    started = time.time()
    artifact = ARTIFACT_DIR / ARTIFACT_NAME
    report: dict[str, Any] = {
        "gate": "g8_authoritative_baseline_smoke",
        "verdict_scope": "authoritative_baseline_smoke",
        "authoritative_override_allowed": False,
        "authoritative_python_expected": str(AUTHORITATIVE_MINT_PYTHON),
        "authoritative_entrypoint": str(ENTRYPOINT),
        "vendor_head_expected": FROZEN_VENDOR_HEAD,
        "vendor_src": str(VENDOR_SRC),
        "passed": False,
        "error": None,
    }
    try:
        if not AUTHORITATIVE_MINT_PYTHON.exists():
            raise RuntimeError(f"Authoritative MINT python missing: {AUTHORITATIVE_MINT_PYTHON}")
        head_proc = _run(["git", "-C", str(MINT_REPO_ROOT), "rev-parse", "HEAD"])
        vendor_head = head_proc.stdout.strip()
        report["vendor_head_used"] = vendor_head
        report["vendor_head_matches_expected"] = (head_proc.returncode == 0 and vendor_head == FROZEN_VENDOR_HEAD)
        if not report["vendor_head_matches_expected"]:
            raise RuntimeError(f"Frozen vendor mismatch: {vendor_head!r} != {FROZEN_VENDOR_HEAD}")
        VARIANT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        command = [
            str(AUTHORITATIVE_MINT_PYTHON),
            str(ENTRYPOINT),
            "--variant-run",
            "--label",
            "current_vendor_authoritative_smoke",
            "--mint-src",
            str(VENDOR_SRC),
            "--output-dir",
            str(VARIANT_OUTPUT_DIR),
            "--n-episodes",
            "1",
            "--seed",
            "42",
        ]
        report["command"] = command
        proc = _run(command)
        summary = _load_json(SUMMARY_PATH)
        failure = _load_json(FAILURE_PATH)
        eval_info = _load_json(EVAL_INFO_PATH)
        runtime_env = (summary or {}).get("runtime_env", {})
        authoritative_python_used = runtime_env.get("python_executable")
        pc_success = ((eval_info or {}).get("overall") or {}).get("pc_success")
        report.update({
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "summary_path": str(SUMMARY_PATH) if SUMMARY_PATH.exists() else None,
            "failure_path": str(FAILURE_PATH) if FAILURE_PATH.exists() else None,
            "eval_info_path": str(EVAL_INFO_PATH) if EVAL_INFO_PATH.exists() else None,
            "authoritative_python_used": authoritative_python_used,
            "authoritative_python_matches_expected": authoritative_python_used == str(AUTHORITATIVE_MINT_PYTHON),
            "pc_success": pc_success,
            "summary": summary,
            "failure": failure,
        })
        report["passed"] = bool(
            proc.returncode == 0
            and report["authoritative_python_matches_expected"]
            and pc_success is not None
            and float(pc_success) >= 100.0
        )
        if not report["passed"]:
            raise RuntimeError("authoritative baseline smoke failed")
    except Exception as exc:  # noqa: BLE001
        report["error"] = repr(exc)
    report["elapsed_sec"] = round(time.time() - started, 3)
    write_json_atomic(artifact, report)
    print(json.dumps(report, indent=2))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
