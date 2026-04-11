#!/usr/bin/env python3
"""CLI entrypoint for the v2 root-cause controller."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from mint_common import now_iso
from root_cause_contracts import AUTOPILOT_DIR, CYCLE_STATE_PATH, HYPOTHESIS_BOARD_PATH
from root_cause_controller import run_controller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-cycles", type=int, default=1)
    parser.add_argument("--max-experiments-per-cycle", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset-state", action="store_true", help="Archive cycle state + hypothesis board before starting a fresh controller run.")
    return parser.parse_args()



def reset_state() -> str:
    AUTOPILOT_DIR.mkdir(parents=True, exist_ok=True)
    archive_dir = AUTOPILOT_DIR / "resets" / now_iso().replace(":", "").replace("+", "_")
    archive_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    for path in (CYCLE_STATE_PATH, HYPOTHESIS_BOARD_PATH):
        if path.exists():
            target = archive_dir / path.name
            shutil.move(str(path), str(target))
            moved.append(path.name)
    summary = {"archive_dir": str(archive_dir), "moved": moved}
    (archive_dir / "reset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return str(archive_dir)


def main() -> int:
    args = parse_args()
    reset_archive = reset_state() if args.reset_state else None
    payload = run_controller(
        max_cycles=args.max_cycles,
        max_experiments_per_cycle=args.max_experiments_per_cycle,
        sleep_seconds=args.sleep_seconds,
        dry_run=args.dry_run,
    )
    if reset_archive is not None:
        payload["reset_archive"] = reset_archive
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
