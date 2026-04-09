#!/usr/bin/env python3
"""Sequential night bundle for the MuJoCo / LIBERO drawer pilot."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from mujoco_pilot_common import ARTIFACT_ROOT, PROJECT_ROOT, now_iso, write_json

ARTIFACT_PATH = ARTIFACT_ROOT / "mujoco_pilot_night.json"
SCRIPTS = [
    "scripts/mint/run_p1c0_libero_drawer_smoke.py",
    "scripts/mint/run_p1c2_libero_drawer_oracle.py",
    "scripts/mint/run_p1c4_mint_on_libero.py",
    "scripts/mint/run_p1c5_state_comparison.py",
]


def main() -> int:
    steps = []
    failed = None
    for rel in SCRIPTS:
        started = time.time()
        proc = subprocess.run([sys.executable, rel], cwd=PROJECT_ROOT)
        step = {
            "script": rel,
            "started_at": now_iso(),
            "elapsed_s": round(time.time() - started, 3),
            "returncode": int(proc.returncode),
        }
        steps.append(step)
        if proc.returncode != 0 and failed is None:
            failed = rel
            break
    payload = {
        "generated_at": now_iso(),
        "steps": steps,
        "failed_step": failed,
        "passed": failed is None,
    }
    write_json(ARTIFACT_PATH, payload)
    print(f"[night] wrote {ARTIFACT_PATH}")
    print(f"[night] passed={payload['passed']} failed_step={failed}")
    return 0 if failed is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
