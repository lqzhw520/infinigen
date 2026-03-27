#!/usr/bin/env python3
"""U2: Audit URDF joint semantics for drawerbox exports."""

from __future__ import annotations

import json
import time

from mint_common import ARTIFACT_DIR
from upstream_audit_helpers import ALL_SEEDS, classify_from_flags, joint_record

ARTIFACT = ARTIFACT_DIR / "u2_joint_semantics_audit.json"


def run() -> bool:
    records = [joint_record(seed) for seed in ALL_SEEDS]
    flags = []
    bad_axis = [
        row["seed"] for row in records if row["axis_alignment_to_negative_y"] < 0.95
    ]
    bad_limit = [
        row["seed"]
        for row in records
        if row["limit_upper"] is None
        or row["limit_upper"] <= 0.05
        or row["limit_upper"] >= 0.35
    ]
    bad_lower = [
        row["seed"]
        for row in records
        if row["limit_lower"] is None or abs(row["limit_lower"]) > 1e-4
    ]
    if bad_axis:
        flags.append("joint_axis_blocked")
    if bad_limit:
        flags.append("joint_limit_suspect")
    if bad_lower:
        flags.append("joint_lower_bound_suspect")
    decision = classify_from_flags(flags)
    result = {
        "gate": "u2_joint_semantics_audit",
        "passed": decision == "upstream_clean",
        "decision": decision,
        "flags": flags,
        "bad_axis_seeds": bad_axis,
        "bad_limit_seeds": bad_limit,
        "bad_lower_bound_seeds": bad_lower,
        "records": records,
        "conclusion_text": (
            "Joint axes and ranges are consistent with the controller's drawer-opening assumption."
            if decision == "upstream_clean"
            else "Joint semantics show export-level inconsistencies or suspicious ranges that can invalidate control assumptions."
        ),
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return True


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
