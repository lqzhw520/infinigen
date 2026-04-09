#!/usr/bin/env python3
"""Block launching a runner on the wrong scientific mainline."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CURRENT_TRUTH = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1" / "sovereign" / "current_truth.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, help="Absolute path to the runner script")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    current_truth = json.loads(CURRENT_TRUTH.read_text())
    next_action = current_truth["current"]["next_action"]
    phase = current_truth["current"]["phase"]
    runner_path = Path(args.runner)
    visited: set[Path] = set()

    def collect_sources(path: Path) -> tuple[str, list[str]]:
        if path in visited or not path.exists():
            return "", []
        visited.add(path)
        text = path.read_text()
        referenced: list[str] = []
        for rel in re.findall(r'["\\\']([^"\\\']+\\.py)["\\\']', text):
            candidate = (PROJECT_ROOT / rel).resolve() if not rel.startswith("/") else Path(rel)
            if candidate.exists():
                referenced.append(str(candidate))
                child_text, child_refs = collect_sources(candidate)
                text += "\n" + child_text
                referenced.extend(child_refs)
        return text, referenced

    source, referenced = collect_sources(runner_path)

    canonical_is_mujoco = "MUJOCO" in next_action.get("type", "") or "MUJOCO" in phase
    pybullet_markers = [
        "DrawerRobotEnv(",
        "DrawerRobotEnvLeRobot",
        "run_g9_sim_eval.py",
        "evaluate_mint_drawer_campaign",
    ]
    matched_markers = [marker for marker in pybullet_markers if marker in source]

    payload = {
        "phase": phase,
        "next_action_type": next_action.get("type"),
        "runner": str(runner_path),
        "referenced_python_files": referenced,
        "canonical_is_mujoco": canonical_is_mujoco,
        "matched_pybullet_markers": matched_markers,
        "passed": not (canonical_is_mujoco and matched_markers),
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
