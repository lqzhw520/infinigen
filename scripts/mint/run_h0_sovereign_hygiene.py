#!/usr/bin/env python3
"""Normalize canonical sovereign surfaces before new alignment gates."""

from __future__ import annotations

import json
from pathlib import Path

from p1_execution_common import normalize_sovereign_h0, write_final_execution_plan_surfaces


def main() -> int:
    plan = write_final_execution_plan_surfaces()
    result = normalize_sovereign_h0()
    payload = {
        "passed": True,
        "plan_id": plan["plan_id"],
        "review_surfaces": result["review_surfaces"],
        "pending_next_action": result["pending_next_action"],
        "current_truth_path": result["current_truth_path"],
        "next_actions_path": result["next_actions_path"],
        "evidence": result["evidence"],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
