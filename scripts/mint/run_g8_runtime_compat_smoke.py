#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts" / "mint") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "mint"))

from mint_common import ARTIFACT_DIR, write_json_atomic
from run_p1c7_official_libero_goal_drawer_baseline import build_command, build_env

ARTIFACT_NAME = "g8_runtime_compat_smoke.json"
FROZEN_MINT_HEAD = "4eab5795345721001c412ff1ca2c886a11eab606"
MINT_REPO = PROJECT_ROOT / "external" / "MINT"
MINT_CKPT = MINT_REPO / "checkpoints" / "MINT-libero"
TOKENIZER_PATH = MINT_REPO / "checkpoints" / "MINT-tokenizer-libero"
HISTORICAL_MINT_PYTHON = Path('/root/anaconda3/envs/mint/bin/python')


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)


def _default_args() -> argparse.Namespace:
    return argparse.Namespace(
        n_episodes=1,
        task_suite="libero_goal",
        task_id=0,
        seed=42,
        device="cuda",
        height=256,
        width=256,
        n_action_steps=4,
    )


def main() -> int:
    started = time.time()
    report: dict[str, Any] = {
        "gate": "g8_runtime_compat_smoke",
        "vendor_policy": "frozen",
        "vendor_repo": str(MINT_REPO),
        "expected_vendor_head": FROZEN_MINT_HEAD,
        "checkpoint_path": str(MINT_CKPT),
        "tokenizer_path": str(TOKENIZER_PATH),
        "historical_mint_python": str(HISTORICAL_MINT_PYTHON),
        "passed": False,
        "stage": None,
        "error": None,
        "timestamp": time.time(),
    }
    artifact = ARTIFACT_DIR / ARTIFACT_NAME
    try:
        head_proc = _run(["git", "rev-parse", "HEAD"], cwd=MINT_REPO)
        vendor_head = head_proc.stdout.strip()
        report["vendor_head"] = vendor_head
        report["vendor_head_matches_frozen"] = (
            head_proc.returncode == 0 and vendor_head == FROZEN_MINT_HEAD
        )
        if not report["vendor_head_matches_frozen"]:
            raise RuntimeError(
                f"external/MINT head {vendor_head!r} does not match frozen baseline {FROZEN_MINT_HEAD}"
            )
        report["stage"] = "vendor_head_verified"

        report["checkpoint_exists"] = MINT_CKPT.exists()
        report["tokenizer_exists"] = TOKENIZER_PATH.exists()
        report["tokenizer_ckpt_exists"] = (TOKENIZER_PATH / "ms_vqvae.pth").exists()
        if not (report["checkpoint_exists"] and report["tokenizer_exists"] and report["tokenizer_ckpt_exists"]):
            raise RuntimeError("Frozen baseline checkpoint/tokenizer paths are incomplete")
        report["stage"] = "checkpoint_paths_verified"

        args = _default_args()
        command = build_command(args)
        env = build_env()
        report["lerobot_eval"] = shutil.which("lerobot-eval")
        report["baseline_command"] = command
        report["stage"] = "baseline_cli_constructed"
        report["baseline_pythonpath_prefix"] = env.get("PYTHONPATH", "").split(":")[:2]
        report["baseline_runtime"] = {
            "MUJOCO_GL": env.get("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": env.get("PYOPENGL_PLATFORM"),
            "TOKENIZERS_PARALLELISM": env.get("TOKENIZERS_PARALLELISM"),
        }
        report["stage"] = "baseline_env_constructed"

        import_cmd = [
            sys.executable,
            "-c",
            (
                "from lerobot_policy_mint.configuration_mint import MINTConfig; "
                "from lerobot_policy_mint.modeling_mint import MINTPolicy; "
                "print('vendor_import_ok')"
            ),
        ]
        import_proc = _run(import_cmd, cwd=PROJECT_ROOT, env=env)
        report["vendor_import_returncode"] = import_proc.returncode
        report["vendor_import_stdout_tail"] = import_proc.stdout[-1000:]
        report["vendor_import_stderr_tail"] = import_proc.stderr[-2000:]
        if import_proc.returncode != 0:
            raise RuntimeError("Vendor import failed under baseline PYTHONPATH")
        report["stage"] = "vendor_import_resolved"

        report["historical_mint_python_exists"] = HISTORICAL_MINT_PYTHON.exists()
        if not report["historical_mint_python_exists"]:
            raise RuntimeError("Historical mint runtime python is missing")
        version_proc = _run(
            [
                str(HISTORICAL_MINT_PYTHON),
                "-c",
                (
                    "import json,platform,torch,transformers; "
                    "print(json.dumps({'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__}))"
                ),
            ],
            cwd=PROJECT_ROOT,
        )
        report["historical_runtime_returncode"] = version_proc.returncode
        report["historical_runtime_stdout_tail"] = version_proc.stdout[-1000:]
        report["historical_runtime_stderr_tail"] = version_proc.stderr[-2000:]
        if version_proc.returncode != 0:
            raise RuntimeError("Historical mint runtime version probe failed")
        report["stage"] = "historical_runtime_env_detected"
        report["passed"] = True
    except Exception as exc:  # noqa: BLE001
        report["error"] = repr(exc)
    report["elapsed_sec"] = round(time.time() - started, 3)
    write_json_atomic(artifact, report)
    print(json.dumps(report, indent=2))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
