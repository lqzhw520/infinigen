#!/usr/bin/env python3
"""P1b gate: run verify_p1b_mujoco.py and emit artifact JSON.

Exit codes:
  0 = gate passed
  1 = gate failed
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
SCRIPTS_MINT = PROJECT_ROOT / "scripts" / "mint"
sys.path.insert(0, str(SCRIPTS_MINT))
from mint_common import ARTIFACT_DIR

ARTIFACT = ARTIFACT_DIR / "p1b_mujoco_env.json"


def run() -> bool:
    t0 = time.monotonic()

    result = subprocess.run(
        [sys.executable, str(SCRIPTS_MINT / "verify_p1b_mujoco.py")],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    elapsed_s = time.monotonic() - t0
    passed = result.returncode == 0

    payload = {
        "gate": "p1b_mujoco_env",
        "passed": passed,
        "elapsed_s": round(elapsed_s, 1),
        "stdout": result.stdout[-4000:] if result.stdout else "",
        "stderr": result.stderr[-2000:] if result.stderr else "",
        "returncode": result.returncode,
        "timestamp": time.time(),
    }

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))

    verdict = "PASS" if passed else "FAIL"
    print(f"\n[P1b gate] {verdict}  elapsed={elapsed_s:.1f}s")
    return passed


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
