#!/usr/bin/env python3
"""Summarize overnight sandbox guided runs vs frozen Phase 2 zero_singleview anchor."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SANDBOX_ROOT = PROJECT_ROOT / "experiments" / "physnap" / "_overnight_sandbox"
ANCHOR_OUTPUT_NAME = "box_conditioning_v2__zero_singleview"
LOG_ROOT = PROJECT_ROOT / "external" / "physnap" / "log"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def load_manifest(campaign_dir: Path) -> dict:
    return yaml.safe_load((campaign_dir / "manifest.yaml").read_text()) or {}


def report_from_stats(run_dir: Path, group_id: str, output_name: str) -> dict:
    stats = load_json(run_dir / "stats.json", {})
    return {
        "group_id": group_id,
        "output_name": output_name,
        "run_dir": str(run_dir),
        "genfull_per_mmd_mean": stats.get("genfull_per_mmd_mean"),
        "genfull_per_cov_mean": stats.get("genfull_per_cov_mean"),
        "genfull_per_1NN-acc_mean": stats.get("genfull_per_1NN-acc_mean"),
        "pen_error_mean": stats.get("pen_error_mean"),
        "mob_error_mean": stats.get("mob_error_mean"),
        "cond_error_mean": stats.get("cond_error_mean"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign-dir",
        default=str(SANDBOX_ROOT / "campaign_phase2_ablations"),
    )
    args = parser.parse_args()
    campaign_dir = Path(args.campaign_dir).resolve()
    manifest = load_manifest(campaign_dir)
    groups = (manifest.get("conditioning") or {}).get("groups") or {}

    anchor_dir = LOG_ROOT / ANCHOR_OUTPUT_NAME
    if not (anchor_dir / "stats.json").exists():
        raise SystemExit(f"Missing anchor stats: {anchor_dir / 'stats.json'}")

    anchor_stats = load_json(anchor_dir / "stats.json", {})
    anchor_row = {
        "group_id": "zero_singleview_anchor",
        "output_name": ANCHOR_OUTPUT_NAME,
        "run_dir": str(anchor_dir),
        "genfull_per_mmd_mean": anchor_stats.get("genfull_per_mmd_mean"),
        "genfull_per_cov_mean": anchor_stats.get("genfull_per_cov_mean"),
        "genfull_per_1NN-acc_mean": anchor_stats.get("genfull_per_1NN-acc_mean"),
        "pen_error_mean": anchor_stats.get("pen_error_mean"),
        "mob_error_mean": anchor_stats.get("mob_error_mean"),
        "cond_error_mean": anchor_stats.get("cond_error_mean"),
    }

    ablations: dict[str, dict] = {}
    missing: list[str] = []
    for group_id, spec in groups.items():
        name = spec["output_name"]
        run_dir = LOG_ROOT / name
        if not (run_dir / "stats.json").exists():
            missing.append(f"{group_id} -> {run_dir}")
            continue
        ablations[group_id] = report_from_stats(run_dir, group_id, name)

    summary = {
        "updated_at": now_iso(),
        "campaign_dir": str(campaign_dir),
        "anchor": anchor_row,
        "ablations": ablations,
        "missing_runs": missing,
    }

    out_dir = SANDBOX_ROOT / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "overnight_ablations_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    lines = [
        "# Overnight Phase 2 ablations",
        "",
        f"**Updated**: {summary['updated_at']}",
        "",
        "| Run | MMD | COV | 1NN | E_pen | E_mob | Cond |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def row(label: str, p: dict) -> str:
        return (
            f"| {label} | {p.get('genfull_per_mmd_mean')} | {p.get('genfull_per_cov_mean')} | "
            f"{p.get('genfull_per_1NN-acc_mean')} | {p.get('pen_error_mean')} | {p.get('mob_error_mean')} | "
            f"{p.get('cond_error_mean')} |"
        )

    lines.append(row("anchor zero_singleview", anchor_row))
    order = [
        "ablation_best_single_view",
        "ablation_fixed_state_multiview",
        "ablation_pts_500",
        "ablation_pts_1000",
        "ablation_pts_2000",
        "ablation_pts_5000",
    ]
    for gid in order:
        if gid in ablations:
            lines.append(row(gid, ablations[gid]))
    for gid, payload in ablations.items():
        if gid not in order:
            lines.append(row(gid, payload))

    if missing:
        lines.extend(["", "## Missing", ""] + [f"- `{m}`" for m in missing])

    (out_dir / "overnight_ablations_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
