#!/usr/bin/env python3
"""
Aggregate parallel worker results for the TuckEndBox paper audit and update the paper markdown.

This script is meant to run in normal Python (NOT Blender).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _replace_block(md: str, start_marker: str, end_marker: str, new_block: str) -> str:
    pattern = re.compile(
        rf"({re.escape(start_marker)}\n)([\s\S]*?)(\n{re.escape(end_marker)})",
        re.MULTILINE,
    )
    if not pattern.search(md):
        raise RuntimeError(f"Markers not found: {start_marker} ... {end_marker}")
    return pattern.sub(rf"\1{new_block}\3", md, count=1)


def _format_count_rate(n: int, total: int) -> Tuple[str, str]:
    rate = 0.0 if total == 0 else (n / total)
    return str(n), f"{rate:.3%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Directory containing worker result JSON files")
    ap.add_argument("--md-path", required=True, help="docs/paper/TuckEndBox_Data_Materials.md")
    ap.add_argument("--summary-json-path", required=True, help="Where to write aggregated summary JSON")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    md_path = Path(args.md_path)
    summary_path = Path(args.summary_json_path)
    # Also write a compact per-seed raw results file for downstream analysis / plotting.
    if summary_path.name.endswith(".audit_summary.json"):
        raw_path = summary_path.with_name(
            summary_path.name.replace(".audit_summary.json", ".audit_raw_results.json")
        )
    else:
        raw_path = summary_path.with_name(summary_path.stem + ".audit_raw_results.json")

    worker_files = sorted(input_dir.glob("worker_*.json"))
    if not worker_files:
        raise SystemExit(f"No worker_*.json found in {input_dir}")

    # Read + validate payloads
    payloads: List[Dict[str, Any]] = []
    for f in worker_files:
        payloads.append(json.loads(f.read_text()))

    meta_seed = int(payloads[0]["meta_seed"])
    n_total = int(payloads[0]["n_total"])
    steps = int(payloads[0]["steps"])
    explosion_threshold = float(payloads[0]["explosion_threshold"])

    # Merge results by global idx
    merged: List[Dict[str, Any]] = [None] * n_total  # type: ignore[list-item]
    for p in payloads:
        if int(p["meta_seed"]) != meta_seed or int(p["n_total"]) != n_total:
            raise SystemExit("Inconsistent meta_seed/n_total across worker payloads")
        for r in p["results"]:
            idx = int(r["idx"])
            if not (0 <= idx < n_total):
                raise SystemExit(f"Bad idx={idx}")
            merged[idx] = r

    missing = [i for i, r in enumerate(merged) if r is None]
    if missing:
        raise SystemExit(f"Missing {len(missing)} results (first few: {missing[:10]})")

    total = n_total
    valid_n = sum(1 for r in merged if r.get("valid"))
    collision_n = sum(1 for r in merged if r.get("initial_collision"))
    inertia_pd_n = sum(1 for r in merged if r.get("inertia_pd"))
    explosion_n = sum(1 for r in merged if r.get("explosion"))

    # Scaling curve (prefix, deterministic order)
    scaling_points = [k for k in [10, 20, 50, 100, 200, 500, 1000] if k <= n_total]
    prefix_valid = 0
    scaling_rows = []
    for i, r in enumerate(merged, start=1):
        if r.get("valid"):
            prefix_valid += 1
        if i in scaling_points:
            vr = prefix_valid / i
            scaling_rows.append((i, 1.0 - vr))

    # Update md tables
    md = md_path.read_text()

    audit_rows = []
    cnt, rate = _format_count_rate(valid_n, total)
    audit_rows.append(f"| Validity Rate | {cnt} | {rate} |")
    cnt, rate = _format_count_rate(collision_n, total)
    audit_rows.append(f"| Initial Collision Rate | {cnt} | {rate} |")
    cnt, rate = _format_count_rate(inertia_pd_n, total)
    audit_rows.append(f"| Inertia PD Rate | {cnt} | {rate} |")
    cnt, rate = _format_count_rate(explosion_n, total)
    audit_rows.append(f"| PyBullet Explosion Rate | {cnt} | {rate} |")

    md = _replace_block(md, "<!-- AUDIT_TABLE_START -->", "<!-- AUDIT_TABLE_END -->", "\n".join(audit_rows))
    md = _replace_block(
        md,
        "<!-- SCALING_TABLE_START -->",
        "<!-- SCALING_TABLE_END -->",
        "\n".join([f"| {k} | {gap:.3%} |" for k, gap in scaling_rows]),
    )

    md_path.write_text(md)

    summary = {
        "meta_seed": meta_seed,
        "N": n_total,
        "steps": steps,
        "explosion_threshold": explosion_threshold,
        "counts": {
            "valid": valid_n,
            "initial_collision": collision_n,
            "inertia_pd": inertia_pd_n,
            "explosion": explosion_n,
        },
        "scaling": [{"N": int(k), "gap": float(g)} for k, g in scaling_rows],
        "worker_files": [str(f.name) for f in worker_files],
    }

    summary_path.write_text(json.dumps(summary, indent=2))
    raw_payload = {
        "meta_seed": meta_seed,
        "N": n_total,
        "steps": steps,
        "explosion_threshold": explosion_threshold,
        "results": merged,
    }
    raw_path.write_text(json.dumps(raw_payload, indent=2))
    print(f"Wrote md: {md_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote raw results: {raw_path}")


if __name__ == "__main__":
    main()

