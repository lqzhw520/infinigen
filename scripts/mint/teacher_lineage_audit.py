#!/usr/bin/env python3
"""Audit teacher-rollout lineage across current and archived campaign runs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from mint_common import ARTIFACT_DIR, CAMPAIGN_DIR, STRICT_TEACHER_AUDIT_PATH, now_iso

LINEAGE_AUDIT_PATH = ARTIFACT_DIR / "teacher_lineage_audit.json"


def _decode_phases(values: np.ndarray | None) -> list[str]:
    if values is None:
        return []
    decoded: list[str] = []
    for value in values.tolist():
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8", errors="ignore"))
        else:
            decoded.append(str(value))
    return decoded


def _strict_summary(npz_path: Path) -> dict[str, Any]:
    meta = json.loads(npz_path.with_suffix(".json").read_text())
    data = np.load(npz_path, allow_pickle=True)
    phases = _decode_phases(
        data["phase_labels"] if "phase_labels" in data.files else None
    )
    drawer = (
        data["absolute_drawer_fraction"].astype(np.float32)
        if "absolute_drawer_fraction" in data.files
        else data["next_drawer_fraction"].astype(np.float32)
    )
    attached = data["attached_trace"].astype(bool)
    handle_min = None
    if "handle_distance_trace" in data.files:
        handle_trace = data["handle_distance_trace"].astype(np.float32)
        if len(handle_trace):
            handle_min = float(np.min(handle_trace))
    first_attach = int(np.argmax(attached)) if bool(attached.any()) else None
    if first_attach is not None and not attached[first_attach]:
        first_attach = None
    attach_fraction = float(drawer[first_attach]) if first_attach is not None else 0.0
    return {
        "path": str(npz_path),
        "sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "seed": int(meta["seed"]),
        "episode_index": int(meta.get("episode_index", -1)),
        "steps": int(len(drawer)),
        "branch_id": meta.get("branch_id") or meta.get("source_branch_id"),
        "teacher_mode": meta.get("teacher_mode"),
        "uses_planned_segment": meta.get("uses_planned_segment"),
        "uses_target_fraction": meta.get("uses_target_fraction"),
        "phase_counts": dict(Counter(phases)),
        "attach_step": meta.get("attach_step"),
        "first_attach_step": first_attach,
        "attach_persistence": int(attached.sum()),
        "pre_attach_drawer_motion": float(
            np.max(np.abs(drawer[: first_attach + 1] - drawer[0]))
        )
        if first_attach is not None
        else float(np.max(np.abs(drawer - drawer[0]))),
        "post_attach_drawer_delta": float(drawer.max() - attach_fraction)
        if first_attach is not None
        else 0.0,
        "handle_distance_min": handle_min,
        "max_drawer_fraction": float(drawer.max()),
    }


def _collect_npz_pairs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("seed_*_episode_*.npz")
        if path.with_suffix(".json").exists()
    )


def run() -> dict[str, Any]:
    current_root = ARTIFACT_DIR / "c2_replay_valid_rollouts"
    archived_root = ARTIFACT_DIR / "diagnostic_runs"
    current = defaultdict(list)
    archived = defaultdict(list)

    for npz_path in _collect_npz_pairs(current_root):
        summary = _strict_summary(npz_path)
        current[(summary["seed"], summary["episode_index"])].append(summary)

    for npz_path in _collect_npz_pairs(archived_root):
        summary = _strict_summary(npz_path)
        archived[(summary["seed"], summary["episode_index"])].append(summary)

    current_rollouts = (
        json.loads(STRICT_TEACHER_AUDIT_PATH.read_text()).get("accepted_rollouts", [])
        if STRICT_TEACHER_AUDIT_PATH.exists()
        else []
    )
    current_keys = {
        tuple(
            int(part)
            for part in (Path(path).stem.split("_")[1], Path(path).stem.split("_")[3])
        )
        for path in current_rollouts
    }

    comparisons = []
    drifted = []
    for key in sorted(set(current) | set(archived)):
        cur = current.get(key, [])
        arc = archived.get(key, [])
        record = {
            "seed": key[0],
            "episode_index": key[1],
            "in_current_active_pool": key in current_keys,
            "current_versions": cur,
            "archived_versions": arc,
            "version_count": len(cur) + len(arc),
        }
        comparisons.append(record)
        hashes = {item["sha256"] for item in cur + arc}
        semantics = {
            (
                item["steps"],
                json.dumps(item["phase_counts"], sort_keys=True),
                item["attach_persistence"],
                round(item["handle_distance_min"], 6)
                if item["handle_distance_min"] is not None
                else None,
            )
            for item in cur + arc
        }
        if len(hashes) > 1 or len(semantics) > 1:
            drifted.append(
                {
                    "seed": key[0],
                    "episode_index": key[1],
                    "in_current_active_pool": key in current_keys,
                    "current_versions": cur,
                    "archived_versions": arc,
                }
            )

    payload = {
        "gate": "teacher_lineage_audit",
        "timestamp": now_iso(),
        "campaign_root": str(CAMPAIGN_DIR),
        "current_root": str(current_root),
        "archived_root": str(archived_root),
        "comparison_count": len(comparisons),
        "drifted_rollout_count": len(drifted),
        "drifted_rollouts": drifted,
        "comparisons": comparisons,
    }
    LINEAGE_AUDIT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
