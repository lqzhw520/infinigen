#!/usr/bin/env python3
"""Summarize Phase 2 conditioning runs into a claim-ready report."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN_DIR = PROJECT_ROOT / "experiments" / "physnap" / "box_conditioning_v2"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def load_manifest(campaign_dir: Path) -> dict:
    return yaml.safe_load((campaign_dir / "manifest.yaml").read_text()) or {}


def report_for_group(run_dir: Path, output_name: str) -> dict:
    stats = load_json(run_dir / "stats.json", {})
    config = load_json(run_dir / "config.json", {})
    gallery = sorted(run_dir.glob("res_chnk*.png"))
    return {
        "output_name": output_name,
        "run_dir": str(run_dir),
        "gallery": str(gallery[-1]) if gallery else None,
        "cond_error_mean": stats.get("cond_error_mean"),
        "cond_dist_mean": stats.get("cond_dist_mean"),
        "pen_error_mean": stats.get("pen_error_mean"),
        "mob_error_mean": stats.get("mob_error_mean"),
        "genfull_per_mmd_mean": stats.get("genfull_per_mmd_mean"),
        "genfull_per_cov_mean": stats.get("genfull_per_cov_mean"),
        "genfull_per_1NN-acc_mean": stats.get("genfull_per_1NN-acc_mean"),
        "n_per_condition": config.get("N"),
    }


def invalid_metric(value, *, allow_negative: bool = False) -> bool:
    if value is None:
        return True
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return True
    if math.isnan(numeric) or math.isinf(numeric):
        return True
    if not allow_negative and numeric < 0:
        return True
    return False


def validate_group_metrics(group_id: str, payload: dict) -> None:
    required_fields = [
        "genfull_per_mmd_mean",
        "genfull_per_cov_mean",
        "genfull_per_1NN-acc_mean",
        "pen_error_mean",
        "mob_error_mean",
        "cond_error_mean",
    ]
    invalid = [field for field in required_fields if invalid_metric(payload.get(field))]
    if invalid:
        raise ValueError(f"Invalid or missing metrics for conditioning group `{group_id}`: {invalid}")


def group_beats_anchor(anchor: dict, candidate: dict, physics_tolerance_factor: float, cond_tolerance_factor: float) -> bool:
    required = [
        anchor.get("genfull_per_mmd_mean"),
        anchor.get("genfull_per_cov_mean"),
        anchor.get("pen_error_mean"),
        anchor.get("mob_error_mean"),
        anchor.get("cond_error_mean"),
        candidate.get("genfull_per_mmd_mean"),
        candidate.get("genfull_per_cov_mean"),
        candidate.get("pen_error_mean"),
        candidate.get("mob_error_mean"),
        candidate.get("cond_error_mean"),
    ]
    if any(value is None for value in required):
        return False
    structural_gain = (
        candidate["genfull_per_mmd_mean"] <= anchor["genfull_per_mmd_mean"]
        and candidate["genfull_per_cov_mean"] >= anchor["genfull_per_cov_mean"]
    )
    physics_ok = (
        candidate["pen_error_mean"] <= anchor["pen_error_mean"] * physics_tolerance_factor
        and candidate["mob_error_mean"] <= anchor["mob_error_mean"] * physics_tolerance_factor
    )
    cond_ok = candidate["cond_error_mean"] <= anchor["cond_error_mean"] * cond_tolerance_factor
    return structural_gain and physics_ok and cond_ok


def group_outcome(anchor: dict, candidate: dict, physics_tolerance_factor: float, cond_tolerance_factor: float) -> dict:
    required = {
        "anchor_mmd": anchor.get("genfull_per_mmd_mean"),
        "anchor_cov": anchor.get("genfull_per_cov_mean"),
        "anchor_pen": anchor.get("pen_error_mean"),
        "anchor_mob": anchor.get("mob_error_mean"),
        "anchor_cond": anchor.get("cond_error_mean"),
        "cand_mmd": candidate.get("genfull_per_mmd_mean"),
        "cand_cov": candidate.get("genfull_per_cov_mean"),
        "cand_pen": candidate.get("pen_error_mean"),
        "cand_mob": candidate.get("mob_error_mean"),
        "cand_cond": candidate.get("cond_error_mean"),
    }
    if any(value is None for value in required.values()):
        return {
            "supported": False,
            "structural_gain": False,
            "physics_preserved": False,
            "conditioning_preserved": False,
            "missing_metrics": sorted(key for key, value in required.items() if value is None),
        }

    structural_gain = (
        candidate["genfull_per_mmd_mean"] <= anchor["genfull_per_mmd_mean"]
        and candidate["genfull_per_cov_mean"] >= anchor["genfull_per_cov_mean"]
    )
    physics_preserved = (
        candidate["pen_error_mean"] <= anchor["pen_error_mean"] * physics_tolerance_factor
        and candidate["mob_error_mean"] <= anchor["mob_error_mean"] * physics_tolerance_factor
    )
    conditioning_preserved = candidate["cond_error_mean"] <= anchor["cond_error_mean"] * cond_tolerance_factor
    return {
        "supported": structural_gain and physics_preserved and conditioning_preserved,
        "structural_gain": structural_gain,
        "physics_preserved": physics_preserved,
        "conditioning_preserved": conditioning_preserved,
        "candidate_mmd_delta": candidate["genfull_per_mmd_mean"] - anchor["genfull_per_mmd_mean"],
        "candidate_cov_delta": candidate["genfull_per_cov_mean"] - anchor["genfull_per_cov_mean"],
        "candidate_pen_delta": candidate["pen_error_mean"] - anchor["pen_error_mean"],
        "candidate_mob_delta": candidate["mob_error_mean"] - anchor["mob_error_mean"],
        "candidate_cond_delta": candidate["cond_error_mean"] - anchor["cond_error_mean"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 conditioning results")
    parser.add_argument("--campaign-dir", default=str(DEFAULT_CAMPAIGN_DIR))
    args = parser.parse_args()

    campaign_dir = Path(args.campaign_dir).resolve()
    manifest = load_manifest(campaign_dir)
    acceptance = load_json(campaign_dir / "acceptance_criteria.json", {})
    state = load_json(campaign_dir / "state.json", {})
    conditioning = manifest.get("conditioning", {}) or {}
    groups = conditioning.get("groups", {}) or {}
    log_root = PROJECT_ROOT / "external" / "physnap" / "log"
    phase2_groups = {}
    for group_id, spec in groups.items():
        run_dir = log_root / spec["output_name"]
        if not (run_dir / "stats.json").exists():
            raise FileNotFoundError(f"Missing stats.json for conditioning group {group_id}: {run_dir}")
        phase2_groups[group_id] = report_for_group(run_dir, spec["output_name"])
        validate_group_metrics(group_id, phase2_groups[group_id])

    anchor = phase2_groups["zero_singleview"]
    physics_tolerance_factor = acceptance.get("scoring", {}).get("physics_tolerance_factor", 1.05)
    cond_tolerance_factor = acceptance.get("scoring", {}).get("cond_error_tolerance_factor", 1.0)
    claim_ladder = [
        {
            "claim_level": "multi-state + multi-view beats zero-state single-view",
            "group_id": "multistate_multiview",
        },
        {
            "claim_level": "multi-view only beats zero-state single-view",
            "group_id": "zero_multiview",
        },
        {
            "claim_level": "multi-state only beats zero-state single-view",
            "group_id": "multistate_singleview",
        },
    ]
    group_outcomes = {}
    winning_group = None
    supported_claim_level = None
    for entry in claim_ladder:
        group_id = entry["group_id"]
        outcome = group_outcome(anchor, phase2_groups[group_id], physics_tolerance_factor, cond_tolerance_factor)
        group_outcomes[group_id] = outcome
        if outcome["supported"] and winning_group is None:
            winning_group = group_id
            supported_claim_level = entry["claim_level"]

    if winning_group:
        verdict = "claim_supported"
        strongest_true_claim = supported_claim_level
    else:
        verdict = "claim_not_supported"
        strongest_true_claim = (
            "Richer conditioning did not consistently beat the zero-state single-view anchor under the bounded Phase 2 recipes."
        )

    summary = {
        "campaign_id": campaign_dir.name,
        "phase": "phase2_conditioning_design",
        "updated_at": now_iso(),
        "phase2_base_experiment": state.get("phase2_base_experiment"),
        "phase2_base_checkpoint": state.get("phase2_base_checkpoint"),
        "anchor_group": "zero_singleview",
        "winning_group": winning_group,
        "supported_claim_level": supported_claim_level,
        "strongest_true_claim": strongest_true_claim,
        "claim_verdict": verdict,
        "conditioning_groups": phase2_groups,
        "claim_ladder": claim_ladder,
        "group_outcomes": group_outcomes,
        "phase1_diagnosis": state.get("phase1_diagnosis"),
    }

    evaluation_dir = campaign_dir / "evaluation"
    save_json(evaluation_dir / "comparison_summary.json", summary)
    save_json(campaign_dir / "summary.json", summary)

    lines = [
        "# Phase 2 Conditioning Comparison",
        "",
        f"**Updated**: {summary['updated_at']}",
        f"**Base Experiment**: `{summary.get('phase2_base_experiment')}`",
        f"**Base Checkpoint**: `{summary.get('phase2_base_checkpoint')}`",
        f"**Claim Verdict**: `{verdict}`",
        "",
        "| Group | MMD | COV | 1NN | E_pen | E_mob | Cond Error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group_id in ["zero_singleview", "zero_multiview", "multistate_singleview", "multistate_multiview"]:
        payload = phase2_groups[group_id]
        lines.append(
            f"| {group_id} | {payload.get('genfull_per_mmd_mean')} | {payload.get('genfull_per_cov_mean')} | {payload.get('genfull_per_1NN-acc_mean')} | {payload.get('pen_error_mean')} | {payload.get('mob_error_mean')} | {payload.get('cond_error_mean')} |"
        )
    lines.extend(["", "## Interpretation", ""])
    if winning_group:
        lines.append(
            f"`{winning_group}` beats the zero-state single-view anchor on structural metrics while preserving physics metrics."
        )
    else:
        lines.append(
            "None of the richer conditioning settings consistently beat the zero-state single-view anchor under the bounded Phase 2 recipes."
        )
    lines.extend(["", "## Claim Ladder", ""])
    for entry in claim_ladder:
        group_id = entry["group_id"]
        outcome = group_outcomes[group_id]
        verdict_text = "supported" if outcome["supported"] else "not supported"
        lines.append(f"- `{entry['claim_level']}` via `{group_id}`: {verdict_text}")
    lines.extend(["", "## Strongest True Claim", "", strongest_true_claim])
    (evaluation_dir / "comparison_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
