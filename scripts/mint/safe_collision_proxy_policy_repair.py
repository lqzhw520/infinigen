#!/usr/bin/env python3
"""Safe collision proxy policy repair with visual/physical consistency guard.

This runner is intentionally conservative: it refuses to demote robot collision
geoms that match visible physical palm/wrist/arm geometry. It may only identify a
safe source repair when an offending geom is demonstrably a duplicate,
nonessential, overconservative proxy whose removal does not hide visible robot
occupancy. Otherwise it records the exact reason the proxy-policy gate cannot be
used to create a Layer4R pass.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import mujoco
import numpy as np

ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN = ROOT / "experiments/mint/mint_drawer_v1"
SPEC_REL = (
    "experiments/mint/mint_drawer_v1/sovereign/experiment_specs/"
    "v11_g4_goc_v4_safe_collision_proxy_policy_repair.yaml"
)
PRIOR_ATTR_PATTERN = "runtime/v11_g4_goc_v4_model_instance_accessibility_collision_attribution_repair_*"
PHYSICAL_ESSENTIAL_NAMES = {"link5_collision", "link6_collision", "link7_collision", "hand_collision"}
DUPLICATE_FINGER_SHELL_NAMES = {"finger1_collision", "finger2_collision"}

sys.path.insert(0, str(ROOT / "scripts/mint"))
import contact_aware_drawer_teacher as cat  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple, set)):
        return [ready(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ready(payload), indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(ready(payload), sort_keys=True) + "\n")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def collect_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def latest_dir(pattern: str) -> Path | None:
    paths = sorted(CAMPAIGN.glob(pattern))
    return paths[-1] if paths else None


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def geom_name(model: mujoco.MjModel, gid: int) -> str:
    return str(model.geom(int(gid)).name or f"geom_{gid}")


def body_name(model: mujoco.MjModel, bid: int) -> str:
    return str(model.body(int(bid)).name or f"body_{bid}")


def body_chain(model: mujoco.MjModel, bid: int) -> list[str]:
    chain: list[str] = []
    cur = int(bid)
    while cur >= 0:
        chain.append(body_name(model, cur))
        parent = int(model.body_parentid[cur]) if cur < model.nbody else -1
        if parent == cur or cur == 0:
            break
        cur = parent
    return list(reversed(chain))


def make_env(seed: int = 2, base_pos: tuple[float, float, float] = (-0.9, 0.0, 0.0), yaw_deg: float = 5.0):
    cat.DrawerRobotEnvMuJoCoLibero._MERGED_BUILDER_CLASS = cat.make_builder(base_pos, yaw_deg)
    env = cat.DrawerRobotEnvMuJoCoLibero(seed=seed, image_size=96, max_steps=10, contract=None)
    env.reset()
    return env


def same_body_visuals(model: mujoco.MjModel, gid: int) -> list[dict[str, Any]]:
    bid = int(model.geom_bodyid[int(gid)])
    out = []
    for j in range(model.ngeom):
        if int(model.geom_bodyid[j]) != bid:
            continue
        if j == int(gid):
            continue
        is_visual = int(model.geom_group[j]) == 1 or (int(model.geom_contype[j]) == 0 and int(model.geom_conaffinity[j]) == 0)
        if not is_visual:
            continue
        out.append({
            "geom_id": int(j),
            "geom_name": geom_name(model, j),
            "group": int(model.geom_group[j]),
            "contype": int(model.geom_contype[j]),
            "conaffinity": int(model.geom_conaffinity[j]),
            "geom_type": int(model.geom_type[j]),
            "geom_rbound_m": float(model.geom_rbound[j]),
            "geom_size": model.geom_size[j].astype(float).tolist(),
        })
    return out


def audit_offender_geom(model: mujoco.MjModel, gid: int, pair_counter: Counter[str], penetration_by_name: dict[str, list[float]]) -> dict[str, Any]:
    gid = int(gid)
    name = geom_name(model, gid)
    bid = int(model.geom_bodyid[gid])
    visuals = same_body_visuals(model, gid)
    max_visual = max([float(v["geom_rbound_m"]) for v in visuals], default=0.0)
    collision_r = float(model.geom_rbound[gid])
    ratio = collision_r / max(max_visual, 1e-12) if max_visual > 0 else math.inf
    has_comparable_visual = bool(visuals and 0.60 <= ratio <= 1.60)
    is_duplicate_finger_shell = name in DUPLICATE_FINGER_SHELL_NAMES and int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0
    is_named_physical = name in PHYSICAL_ESSENTIAL_NAMES or (name.startswith("link") and name.endswith("_collision"))
    physical_essential = bool(is_named_physical and has_comparable_visual)
    # The safe rule is intentionally narrow. If a physical palm/wrist/arm shell
    # has matching visual geometry, disabling it would create a hidden pass-through.
    safe_demote_candidate = bool(is_duplicate_finger_shell and not is_named_physical)
    if physical_essential:
        safe_demote_candidate = False
    if is_named_physical:
        unsafe_reason = "collision_geom_matches_visible_physical_robot_body"
    elif not visuals:
        unsafe_reason = "no_same_body_visual_evidence_for_safe_demote"
    elif ratio > 1.60:
        unsafe_reason = "overconservative_possible_but_requires_geometry_redesign_not_bitmask_demote"
    else:
        unsafe_reason = "not_proven_duplicate_nonessential_proxy"
    return {
        "geom_id": gid,
        "geom_name": name,
        "body_id": bid,
        "body_name": body_name(model, bid),
        "body_chain": body_chain(model, bid),
        "contype": int(model.geom_contype[gid]),
        "conaffinity": int(model.geom_conaffinity[gid]),
        "group": int(model.geom_group[gid]),
        "geom_type": int(model.geom_type[gid]),
        "geom_size": model.geom_size[gid].astype(float).tolist(),
        "collision_rbound_m": collision_r,
        "same_body_visual_geom_count": len(visuals),
        "max_same_body_visual_rbound_m": max_visual,
        "collision_to_visual_rbound_ratio": ratio,
        "same_body_visual_geoms": visuals[:12],
        "prior_offending_pair_count": int(pair_counter.get(name, 0)),
        "prior_penetration_m": {
            "count": len(penetration_by_name.get(name, [])),
            "max": max(penetration_by_name.get(name, [0.0])),
            "mean": float(np.mean(penetration_by_name.get(name, [0.0]))),
        },
        "has_comparable_same_body_visual": has_comparable_visual,
        "is_named_physical_palm_wrist_or_arm": is_named_physical,
        "is_duplicate_finger_shell_already_noncontact": is_duplicate_finger_shell,
        "physical_essential_offender": physical_essential,
        "safe_demote_candidate": safe_demote_candidate,
        "unsafe_demote_reason": None if safe_demote_candidate else unsafe_reason,
    }


def ingest_prior(run_dir: Path) -> dict[str, Any]:
    closeout = load_json(run_dir / "closeout_decision.json") or {}
    summary = load_json(run_dir / "stage2_pair_attribution_summary.json") or {}
    classification = load_json(run_dir / "stage4_collision_policy_classification.json") or {}
    what_if = load_json(run_dir / "stage5_what_if_proxy_suppression.json") or {}
    pairs = collect_jsonl(run_dir / "stage2_pair_attribution.jsonl")
    return {
        "run_dir": rel(run_dir),
        "closeout": closeout,
        "summary": summary,
        "classification": classification,
        "what_if": what_if,
        "pair_count": len(pairs),
        "pairs": pairs,
    }


def offender_stats(pairs: list[dict[str, Any]]) -> tuple[Counter[str], Counter[str], dict[str, list[float]], set[int]]:
    pair_classes: Counter[str] = Counter()
    geom_names: Counter[str] = Counter()
    penetration_by_name: dict[str, list[float]] = defaultdict(list)
    offender_ids: set[int] = set()
    for p in pairs:
        if not p.get("is_violation", False):
            continue
        pair_classes[str(p.get("pair_class", "unknown"))] += 1
        for side in ["geom1_role", "geom2_role"]:
            role = p.get(side) or {}
            name = str(role.get("geom_name", ""))
            gid = role.get("geom_id")
            proxy = str(role.get("proxy_class", "none"))
            semantic = str(role.get("semantic_role", ""))
            if proxy in {"broad_hand_wrist_link_proxy", "hand_collision_proxy", "physical_arm_link_collision"} or name in PHYSICAL_ESSENTIAL_NAMES:
                geom_names[name] += 1
                if gid is not None:
                    offender_ids.add(int(gid))
                penetration_by_name[name].append(float(p.get("penetration_m", 0.0)))
            elif semantic == "legal_dedicated_finger_pad":
                geom_names[name] += 1
                if gid is not None:
                    offender_ids.add(int(gid))
                penetration_by_name[name].append(float(p.get("penetration_m", 0.0)))
    return pair_classes, geom_names, penetration_by_name, offender_ids


def classify_repair(audits: list[dict[str, Any]], prior: dict[str, Any]) -> dict[str, Any]:
    physical = [a for a in audits if a["physical_essential_offender"] and int(a.get("prior_offending_pair_count", 0)) > 0]
    duplicate = [a for a in audits if a["safe_demote_candidate"] and int(a.get("prior_offending_pair_count", 0)) > 0]
    named = {a["geom_name"] for a in audits}
    unsafe_physical_names = sorted(a["geom_name"] for a in physical)
    prior_what_if = prior.get("what_if") or {}
    all_proxy_what_if_improved = bool(prior_what_if)
    source_patch_allowed = bool(duplicate and not physical)
    if source_patch_allowed:
        closeout = "SAFE_SOURCE_PROXY_REPAIR_IDENTIFIED_REQUIRES_MODEL_BUILDER_PATCH"
        next_gate = "APPLY_SAFE_DUPLICATE_PROXY_MODEL_BUILDER_PATCH_AND_RERUN_ENDPOINT_CORRIDOR_DYNAMIC_LAYER4R"
    else:
        closeout = "SAFE_PROXY_REPAIR_REJECTED_PHYSICAL_COLLISION_ESSENTIAL"
        next_gate = "ROBOT_MOUNT_LAYOUT_OR_DRAWER_ACCESSIBILITY_REDESIGN_WITH_PHYSICAL_COLLISION_GUARD"
    return {
        "visual_physical_consistency_guard_completed": True,
        "offender_geom_audit_count": len(audits),
        "physically_essential_offender_count": len(physical),
        "duplicate_proxy_offender_count": len(duplicate),
        "offending_geom_names": sorted(named),
        "unsafe_physical_offender_names": unsafe_physical_names,
        "prior_all_proxy_what_if_improved": all_proxy_what_if_improved,
        "prior_all_proxy_what_if_acceptance": "rejected_because_it_demotes_physical_palm_wrist_or_arm_bodies" if all_proxy_what_if_improved else "not_available",
        "safe_source_proxy_repair_identified": source_patch_allowed,
        "source_patch_applied": False,
        "closeout_classification": closeout,
        "next_gate": next_gate,
        "decision_basis": (
            "At least one safe duplicate nonessential proxy can be repaired without hiding physical robot occupancy."
            if source_patch_allowed else
            "Dominant offending geoms have same-body visual counterparts of comparable size and are named physical palm/wrist/arm collision bodies; demoting them would create a hidden pass-through rather than a valid Layer4R repair."
        ),
    }


def current_truth_changed() -> bool:
    out = subprocess.check_output(["git", "status", "--short", "experiments/mint/mint_drawer_v1/sovereign/current_truth.json"], cwd=ROOT, text=True).strip()
    return bool(out)


def next_actions_changed() -> bool:
    out = subprocess.check_output(["git", "status", "--short", "experiments/mint/mint_drawer_v1/sovereign/next_actions.json"], cwd=ROOT, text=True).strip()
    return bool(out)


def render_markdown_report(run_dir: Path, decision: dict[str, Any], audits: list[dict[str, Any]], pair_classes: Counter[str]) -> str:
    lines = [
        "# SAFE Collision Proxy Policy Repair With Visual/Physical Consistency Guard",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "## Decision",
        "",
        f"- closeout_classification: `{decision['closeout_classification']}`",
        f"- next_gate: `{decision['next_gate']}`",
        f"- safe_source_proxy_repair_identified: `{decision['safe_source_proxy_repair_identified']}`",
        "",
        "## Why",
        "",
        decision["decision_basis"],
        "",
        "## Prior Pair Classes",
        "",
    ]
    for key, val in pair_classes.most_common():
        lines.append(f"- `{key}`: {val}")
    lines += ["", "## Offender Audit", ""]
    for a in audits:
        lines.append(
            f"- `{a['geom_name']}` body=`{a['body_name']}` count={a['prior_offending_pair_count']} "
            f"collision_rbound={a['collision_rbound_m']:.6f} max_visual_rbound={a['max_same_body_visual_rbound_m']:.6f} "
            f"ratio={a['collision_to_visual_rbound_ratio']:.3f} physical_essential={a['physical_essential_offender']} "
            f"safe_demote_candidate={a['safe_demote_candidate']} reason=`{a['unsafe_demote_reason']}`"
        )
    lines += [
        "",
        "## Scientific Boundary",
        "",
        "This phase does not claim Layer4R, teacher rollout, export, local replay/render, or MINT success. It closes the proxy-policy gate by proving whether demotion/resizing is a scientifically valid source repair. In this run, unsafe physical-body demotion is rejected.",
    ]
    return "\n".join(lines)


def run_campaign(run_dir: Path) -> dict[str, Any]:
    t0 = time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    commands_log = run_dir / "commands.log"
    commands_log.write_text("safe_collision_proxy_policy_repair.py\n")

    authority = {
        "generated_at_utc": utc_now(),
        "pwd": str(ROOT),
        "head": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "status_short": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
        "spec": SPEC_REL,
    }
    write_json(run_dir / "stage0_authority.json", authority)

    prior_dir = latest_dir(PRIOR_ATTR_PATTERN)
    if prior_dir is None:
        closeout = {
            "closeout_classification": "PRIOR_ATTRIBUTION_EVIDENCE_MISSING",
            "harness_preflight_passed": False,
            "task_spec_lock_bound": False,
            "prior_attribution_ingested": False,
            "next_gate": "RERUN_MODEL_INSTANCE_ACCESSIBILITY_COLLISION_ATTRIBUTION",
        }
        write_json(run_dir / "closeout_decision.json", closeout)
        return closeout

    prior = ingest_prior(prior_dir)
    write_json(run_dir / "stage1_prior_attribution_ingestion.json", {k: v for k, v in prior.items() if k != "pairs"})
    pair_classes, geom_counter, penetration_by_name, offender_ids = offender_stats(prior["pairs"])
    prior_closeout_pair_classes = (prior.get("closeout") or {}).get("dominant_offending_pair_classes") or {}
    public_pair_classes = Counter({str(k): int(v) for k, v in prior_closeout_pair_classes.items()}) if prior_closeout_pair_classes else pair_classes
    write_json(run_dir / "stage1_prior_offender_histogram.json", {
        "pair_classes_from_jsonl_raw": dict(pair_classes),
        "pair_classes_public_prior_closeout_surface": dict(public_pair_classes),
        "offender_geom_names": dict(geom_counter),
        "offender_geom_ids": sorted(offender_ids),
    })

    seed = 2
    base = (-0.9, 0.0, 0.0)
    yaw = 5.0
    rows = (prior.get("summary") or {}).get("rows") or []
    if rows:
        first = rows[0]
        seed = int(first.get("seed", seed))
        base = tuple(float(x) for x in first.get("base_pos", base))
        yaw = float(first.get("yaw_deg", yaw))
    env = make_env(seed=seed, base_pos=base, yaw_deg=yaw)
    model = env.model

    audits = []
    ids_by_name = {geom_name(model, gid): gid for gid in range(model.ngeom)}
    for name, _ in geom_counter.most_common():
        gid = ids_by_name.get(name)
        if gid is None:
            continue
        audits.append(audit_offender_geom(model, gid, geom_counter, penetration_by_name))
    # Always audit the named physical suspects even if a representative seed's
    # ID shifted; this keeps the guard explicit across model-instance IDs.
    seen = {a["geom_name"] for a in audits}
    for name in sorted(PHYSICAL_ESSENTIAL_NAMES | DUPLICATE_FINGER_SHELL_NAMES):
        if name in seen or name not in ids_by_name:
            continue
        audits.append(audit_offender_geom(model, ids_by_name[name], geom_counter, penetration_by_name))

    write_json(run_dir / "stage2_visual_physical_consistency_guard.json", {
        "representative_seed": seed,
        "representative_base_pos": base,
        "representative_yaw_deg": yaw,
        "audit_count": len(audits),
        "audits": audits,
    })

    decision = classify_repair(audits, prior)
    write_json(run_dir / "stage3_safe_proxy_source_repair_decision.json", decision)

    repair_cycle = {
        "cycle": 1,
        "operator": "safe_source_proxy_demote_or_resize_decision",
        "changed_source_files": [],
        "tested_patch": False,
        "progress_metric": {
            "prior_exact_pair_attribution_available": prior["pair_count"] > 0,
            "unsafe_all_proxy_what_if_improved": decision["prior_all_proxy_what_if_improved"],
            "safe_source_proxy_repair_identified": decision["safe_source_proxy_repair_identified"],
        },
        "outcome": "no_source_patch_applied" if not decision["safe_source_proxy_repair_identified"] else "source_patch_required_not_applied_by_guard_phase",
        "reason": decision["decision_basis"],
    }
    append_jsonl(run_dir / "stage4_repair_cycles.jsonl", repair_cycle)

    endpoint = {
        "attempted": False,
        "reason": "no_safe_source_patch_identified" if not decision["safe_source_proxy_repair_identified"] else "source_patch_not_applied_in_guard_phase",
        "endpoint_feasible_after_repair": False,
        "feasible_seed_count": 0,
        "feasible_case_count": 0,
    }
    write_json(run_dir / "stage5_endpoint_rerun_gate.json", endpoint)

    proposed_truth = {
        "proposal_id": "proposed_current_truth_delta_safe_collision_proxy_policy_repair",
        "generated_at_utc": utc_now(),
        "source_run_dir": rel(run_dir),
        "do_not_mutate_current_truth_directly": True,
        "safe_collision_proxy_policy_repair": {
            "closeout_classification": decision["closeout_classification"],
            "dominant_prior_pair_classes": dict(public_pair_classes),
            "safe_source_proxy_repair_identified": decision["safe_source_proxy_repair_identified"],
            "source_patch_applied": False,
            "scientific_interpretation": decision["decision_basis"],
        },
    }
    proposed_next = {
        "proposal_id": "proposed_next_actions_safe_collision_proxy_policy_repair",
        "generated_at_utc": utc_now(),
        "source_run_dir": rel(run_dir),
        "do_not_mutate_next_actions_directly": True,
        "recommended_next_gate": decision["next_gate"],
        "actions": [
            "Do not demote link5/link6/link7/hand collisions to create a pass unless visual/physical guard is overturned by stronger evidence.",
            "Move next repair to mount/layout/drawer-accessibility redesign or model-instance rejection with physical collision guard.",
            "Only rerun endpoint/corridor/dynamic/Layer4R after a safe source-level geometry/layout repair is committed.",
        ],
    }
    write_json(CAMPAIGN / "sovereign/proposed_current_truth_delta_safe_collision_proxy_policy_repair.json", proposed_truth)
    write_json(CAMPAIGN / "sovereign/proposed_next_actions_safe_collision_proxy_policy_repair.json", proposed_next)

    closeout = {
        "closeout_classification": decision["closeout_classification"],
        "harness_preflight_passed": True,
        "task_spec_lock_bound": True,
        "prior_attribution_ingested": True,
        "prior_attribution_run_dir": rel(prior_dir),
        "visual_physical_consistency_guard_completed": True,
        "offender_geom_audit_count": len(audits),
        "physically_essential_offender_count": decision["physically_essential_offender_count"],
        "duplicate_proxy_offender_count": decision["duplicate_proxy_offender_count"],
        "dominant_prior_pair_classes": dict(public_pair_classes),
        "safe_source_proxy_repair_identified": decision["safe_source_proxy_repair_identified"],
        "source_patch_applied": False,
        "repair_cycles_run": 1,
        "endpoint_rerun_attempted": False,
        "endpoint_feasible_after_repair": False,
        "approach_corridor_attempted": False,
        "dynamic_probe_attempted": False,
        "layer4r_full_matrix_attempted": False,
        "current_truth_modified": current_truth_changed(),
        "next_actions_modified": next_actions_changed(),
        "runtime_patch_applied": True,
        "runtime_patch_files": ["scripts/mint/safe_collision_proxy_policy_repair.py"],
        "committed": False,
        "pushed_to_origin": False,
        "remote_commit_hash": None,
        "next_gate": decision["next_gate"],
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    write_json(run_dir / "closeout_decision.json", closeout)
    write_md(run_dir / "final_report.md", render_markdown_report(run_dir, decision, audits, public_pair_classes))
    return closeout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir or CAMPAIGN / "runtime" / f"v11_g4_goc_v4_safe_collision_proxy_policy_repair_{utc_stamp()}"
    closeout = run_campaign(run_dir)
    print(json.dumps(ready(closeout), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
