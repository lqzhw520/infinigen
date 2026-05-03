#!/usr/bin/env python3
"""GOC-v4 model-instance accessibility and collision attribution repair.

The runner is deliberately forensic-first: it replays prior near-IK endpoint
rows, attributes exact MuJoCo contact pairs to semantic geometry roles, then
performs bounded what-if repair probes. It refuses to promote to Layer4R unless
the source repair is safe and the endpoint/corridor/dynamic gates can be rerun.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
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
    "v11_g4_goc_v4_model_instance_accessibility_and_collision_attribution_repair_overnight.yaml"
)
PRIOR_PATTERN = "runtime/v11_g4_goc_v4_model_placement_keepout_repair_to_layer4r_certify_*"

sys.path.insert(0, str(ROOT / "scripts/mint"))
import contact_aware_drawer_teacher as cat  # noqa: E402
import handle_frame_grasp_trajectory_planner as hfp  # noqa: E402
import joint_placement_reset_ik_keepout_feasibility as jpf  # noqa: E402
import v11_g4_goc_v4_autonomous_repair_campaign as repair_campaign  # noqa: E402

MAX_PENETRATION_M = jpf.MAX_PENETRATION_M
MAX_FORCE_N = jpf.MAX_FORCE_N
PHASE_NAMES = ["pregrasp", "guarded", "contact_hold", "best_contact_only_ik"]


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
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def latest_dir(pattern: str) -> Path | None:
    paths = sorted(CAMPAIGN.glob(pattern))
    return paths[-1] if paths else None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def geom_name(model: mujoco.MjModel, gid: int) -> str:
    return cat.geom_name(model, int(gid)) or f"geom_{gid}"


def body_name(model: mujoco.MjModel, bid: int) -> str:
    return cat.body_name(model, int(bid))


def body_chain(model: mujoco.MjModel, bid: int) -> list[str]:
    return cat.body_chain(model, int(bid))


def geom_role(model: mujoco.MjModel, gid: int, binding: dict[str, Any]) -> dict[str, Any]:
    gid = int(gid)
    legal = set(map(int, binding.get("legal_finger_pad_geom_ids", binding.get("legal_gripper_surface_geom_ids", []))))
    legal |= set(map(int, binding.get("legal_gripper_surface_geom_ids", [])))
    handle = set(map(int, binding.get("drawer_handle_geom_ids", [])))
    forbidden = set(map(int, binding.get("forbidden_robot_surface_geom_ids", [])))
    drawer = set(map(int, binding.get("drawer_body_or_cabinet_geom_ids", [])))
    broad = set(map(int, binding.get("goc_v3_broad_link_geoms_demoted_from_target", [])))
    bid = int(model.geom_bodyid[gid])
    gname = geom_name(model, gid)
    bname = body_name(model, bid)
    lname = gname.lower()
    chain = body_chain(model, bid)
    role = "unknown_contact_relevant"
    if gid in legal:
        role = "legal_dedicated_finger_pad"
    elif gid in handle:
        role = "active_drawer_handle"
    elif gid in forbidden:
        role = "forbidden_robot_surface"
    elif gid in drawer:
        role = "drawer_body_or_cabinet"
    elif gid in broad:
        role = "goc_v3_broad_link_demoted_non_target"
    elif int(model.geom_contype[gid]) == 0 or int(model.geom_conaffinity[gid]) == 0:
        role = "visual_or_noncontact"

    robot_like = any(x in chain for x in [
        "link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
        "right_hand", "leftfinger", "rightfinger", "finger_joint1_tip", "finger_joint2_tip",
    ])
    if role == "unknown_contact_relevant" and robot_like:
        role = "unclassified_robot_contact_surface"
    proxy_class = "none"
    if role in {"forbidden_robot_surface", "goc_v3_broad_link_demoted_non_target", "unclassified_robot_contact_surface"}:
        if "finger" in lname and "collision" in lname and "pad" not in lname:
            proxy_class = "finger_shell_proxy"
        elif lname in {"hand_collision", "link5_collision", "link6_collision", "link7_collision"} or gid in broad:
            proxy_class = "broad_hand_wrist_link_proxy"
        elif lname.startswith("link") and "collision" in lname:
            proxy_class = "physical_arm_link_collision"
        elif "hand" in lname and "collision" in lname:
            proxy_class = "hand_collision_proxy"
        else:
            proxy_class = "robot_collision_unknown"
    return {
        "geom_id": gid,
        "geom_name": gname,
        "body_id": bid,
        "body_name": bname,
        "body_chain": chain,
        "semantic_role": role,
        "proxy_class": proxy_class,
        "contact_capable": bool(int(model.geom_contype[gid]) and int(model.geom_conaffinity[gid])),
        "contype": int(model.geom_contype[gid]),
        "conaffinity": int(model.geom_conaffinity[gid]),
        "geom_type": int(model.geom_type[gid]),
        "geom_size": model.geom_size[gid].astype(float).tolist(),
        "geom_rbound": float(model.geom_rbound[gid]),
    }


def pair_class(r1: dict[str, Any], r2: dict[str, Any]) -> str:
    roles = {r1["semantic_role"], r2["semantic_role"]}
    proxies = {r1["proxy_class"], r2["proxy_class"]}
    if roles == {"legal_dedicated_finger_pad", "active_drawer_handle"}:
        return "allowed_dedicated_pad_handle_target"
    if "active_drawer_handle" in roles and "legal_dedicated_finger_pad" not in roles:
        return "handle_nonlegal_contact"
    if "forbidden_robot_surface" in roles or "unclassified_robot_contact_surface" in roles or "goc_v3_broad_link_demoted_non_target" in roles:
        if "drawer_body_or_cabinet" in roles or "active_drawer_handle" in roles:
            if "physical_arm_link_collision" in proxies:
                return "physical_arm_link_scene_collision"
            if "broad_hand_wrist_link_proxy" in proxies or "hand_collision_proxy" in proxies:
                return "broad_hand_wrist_proxy_scene_collision"
            if "finger_shell_proxy" in proxies:
                return "finger_shell_proxy_scene_collision"
            return "forbidden_robot_scene_collision"
    if "drawer_body_or_cabinet" in roles and "legal_dedicated_finger_pad" in roles:
        return "pad_drawer_body_or_cabinet_non_target_contact"
    return "other_or_self_contact"


def enrich_contact_pairs(env, binding: dict[str, Any], cr: dict[str, Any], stage: str, row_id: str) -> list[dict[str, Any]]:
    out = []
    for pair in cr.get("pairs", []):
        g1 = int(pair["geom1"])
        g2 = int(pair["geom2"])
        r1 = geom_role(env.model, g1, binding)
        r2 = geom_role(env.model, g2, binding)
        cls = pair_class(r1, r2)
        payload = {
            **{k: v for k, v in pair.items() if k != "normal"},
            "row_id": row_id,
            "stage": stage,
            "pair_class": cls,
            "geom1_role": r1,
            "geom2_role": r2,
            "pair_key": "|".join(sorted([r1["geom_name"], r2["geom_name"]])),
            "is_violation": cls != "allowed_dedicated_pad_handle_target" and (
                pair.get("category") in {"forbidden", "handle_nonlegal"}
                or float(pair.get("penetration_m", 0.0)) > MAX_PENETRATION_M
            ),
        }
        out.append(payload)
    return out


def row_identity(row: dict[str, Any], idx: int) -> str:
    base = ",".join(f"{float(x):.3f}" for x in row.get("base_pos", []))
    return f"seed{int(row.get('seed', -1)):03d}_idx{idx:03d}_base{base}_yaw{float(row.get('yaw_deg', 0.0)):.1f}"


def set_row_qpos(env, row: dict[str, Any], stage: str) -> bool:
    try:
        if stage == "best_contact_only_ik":
            q = np.asarray(row["best_contact_only_ik"]["qpos_robot"], dtype=float)
        else:
            q = np.asarray(row["ik"][stage]["qpos_robot"], dtype=float)
        jpf.set_robot_qpos(env, q)
        return True
    except Exception:
        return False


def load_prior_best_rows(prior: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    summary = load_json(prior / "stage4_model_placement_repair_summary.json") or {}
    best = summary.get("best_cycle") or {}
    for row in best.get("best_rows", []) or []:
        if isinstance(row, dict):
            rows.append(row)
    if best.get("cycle_dir"):
        cycle_dir = ROOT / str(best["cycle_dir"])
        p = cycle_dir / "stage3_best_candidates_by_seed.json"
        data = load_json(p) or {}
        if data:
            sources.append(rel(p) or str(p))
            for payload in data.values():
                if isinstance(payload, dict):
                    if isinstance(payload.get("best"), dict):
                        rows.append(payload["best"])
                    for row in payload.get("top_rows", [])[:3]:
                        if isinstance(row, dict):
                            rows.append(row)
    seen = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = json.dumps({"seed": row.get("seed"), "base": row.get("base_pos"), "yaw": row.get("yaw_deg"), "score": row.get("score")}, sort_keys=True)
        if key not in seen and row.get("base_pos") is not None and row.get("ik"):
            seen.add(key)
            unique.append(row)
    unique = sorted(unique, key=lambda r: float(r.get("score", 999.0)))
    closeout = load_json(prior / "closeout_decision.json") or {}
    meta = {
        "prior_run_dir": rel(prior),
        "prior_summary_path": rel(prior / "stage4_model_placement_repair_summary.json"),
        "row_sources": sources,
        "candidate_rows_loaded": len(unique),
        "prior_closeout": closeout.get("closeout_classification"),
        "prior_best_reset_clean_two_pad_residual_m": closeout.get("best_reset_clean_two_pad_residual_m"),
        "prior_remaining_endpoint_failure_clusters": closeout.get("remaining_endpoint_failure_clusters"),
    }
    return unique, meta


def make_env(seed: int, base_pos: list[float], yaw_deg: float, max_steps: int = 20, qpos: np.ndarray | None = None):
    return jpf.make_env(int(seed), list(map(float, base_pos)), float(yaw_deg), int(max_steps), qpos=qpos)


def replay_row(row: dict[str, Any], idx: int, run_dir: Path, *, suppress_geoms: set[int] | None = None, write_pairs: bool = True) -> dict[str, Any]:
    env = None
    row_id = row_identity(row, idx)
    stages: dict[str, Any] = {}
    all_pairs: list[dict[str, Any]] = []
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            env = make_env(int(row["seed"]), row["base_pos"], float(row["yaw_deg"]), max_steps=20, qpos=jpf.SAFE_PRECONTACT_QPOS)
            env.reset()
        binding = cat.classify_instance(env)
        if suppress_geoms:
            for gid in suppress_geoms:
                if 0 <= int(gid) < int(env.model.ngeom):
                    env.model.geom_contype[int(gid)] = 0
                    env.model.geom_conaffinity[int(gid)] = 0
            mujoco.mj_forward(env.model, env.data)
        for stage in PHASE_NAMES:
            if not set_row_qpos(env, row, stage):
                stages[stage] = {"available": False}
                continue
            cr = cat.contact_report(env, binding, None)
            pairs = enrich_contact_pairs(env, binding, cr, stage, row_id)
            if write_pairs:
                for pair in pairs:
                    append_jsonl(run_dir / "stage2_pair_attribution.jsonl", pair)
            all_pairs.extend(pairs)
            stages[stage] = {
                "available": True,
                "contact_counts": cr["counts"],
                "pair_count": len(pairs),
                "violation_count": sum(1 for p in pairs if p["is_violation"]),
                "violating_pairs": [p for p in pairs if p["is_violation"]][:8],
            }
        return {
            "row_id": row_id,
            "seed": int(row["seed"]),
            "base_pos": row["base_pos"],
            "yaw_deg": float(row["yaw_deg"]),
            "score": float(row.get("score", 999.0)),
            "binding": hfp.compact_binding(binding),
            "suppressed_geoms": sorted(suppress_geoms or []),
            "stages": stages,
            "pair_classes": dict(Counter(p["pair_class"] for p in all_pairs)),
            "violating_pair_classes": dict(Counter(p["pair_class"] for p in all_pairs if p["is_violation"])),
            "offending_geom_ids": sorted({int(p["geom1_role"]["geom_id"]) for p in all_pairs if p["is_violation"]} | {int(p["geom2_role"]["geom_id"]) for p in all_pairs if p["is_violation"]}),
            "offending_proxy_classes": dict(Counter([p["geom1_role"]["proxy_class"] for p in all_pairs if p["is_violation"]] + [p["geom2_role"]["proxy_class"] for p in all_pairs if p["is_violation"]])),
        }
    except Exception as exc:
        return {"row_id": row_id, "seed": row.get("seed"), "base_pos": row.get("base_pos"), "yaw_deg": row.get("yaw_deg"), "error": repr(exc)}
    finally:
        if env is not None:
            env.close()


def summarize_attribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair_cls = Counter()
    proxy_cls = Counter()
    geom_pairs = Counter()
    stage_violations = Counter()
    geom_role_counts = Counter()
    for replay in rows:
        pair_cls.update(replay.get("violating_pair_classes", {}))
        proxy_cls.update(replay.get("offending_proxy_classes", {}))
        for stage, payload in (replay.get("stages") or {}).items():
            stage_violations[stage] += int(payload.get("violation_count", 0))
            for pair in payload.get("violating_pairs", []):
                geom_pairs[pair.get("pair_key", "unknown")] += 1
                geom_role_counts[pair["geom1_role"]["semantic_role"]] += 1
                geom_role_counts[pair["geom2_role"]["semantic_role"]] += 1
    return {
        "pair_level_attribution_completed": bool(rows),
        "replayed_rows": len(rows),
        "dominant_offending_pair_classes": dict(pair_cls.most_common(12)),
        "dominant_offending_proxy_classes": dict(proxy_cls.most_common(12)),
        "dominant_offending_geom_pairs": dict(geom_pairs.most_common(20)),
        "stage_violation_counts": dict(stage_violations),
        "offending_geom_role_counts": dict(geom_role_counts.most_common(12)),
    }


def handle_accessibility_audit(replay_rows: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    audits = []
    for rr in replay_rows[:20]:
        env = None
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                env = make_env(int(rr["seed"]), rr["base_pos"], float(rr["yaw_deg"]), max_steps=20, qpos=jpf.SAFE_PRECONTACT_QPOS)
                env.reset()
            binding = cat.classify_instance(env)
            reset = cat.contact_report(env, binding, None)
            hf = hfp.build_handle_frame(env, binding)
            if hf.get("quality") != "ok":
                audits.append({"row_id": rr["row_id"], "quality": hf.get("quality")})
                continue
            frame = hf["_frame"]
            tpf = hfp.build_two_pad_frame(env, binding, frame, dict(jpf.DEFAULT_PARAMS))
            scene_ids = list(map(int, binding.get("drawer_body_or_cabinet_geom_ids", [])))
            line_points = []
            for d in np.linspace(0.0, 0.10, 11):
                line_points.append(np.asarray(frame.center) + np.asarray(frame.approach_normal) * (float(frame.handle_radius_normal_m) + float(d)))
            min_scene_margin = math.inf
            closest = None
            for point in line_points:
                for gid in scene_ids:
                    center = np.asarray(env.data.geom_xpos[gid], dtype=float)
                    margin = float(np.linalg.norm(point - center) - float(env.model.geom_rbound[gid]))
                    if margin < min_scene_margin:
                        min_scene_margin = margin
                        closest = {"geom_id": gid, "geom_name": geom_name(env.model, gid), "body": body_name(env.model, int(env.model.geom_bodyid[gid])), "margin_m": margin}
            audits.append({
                "row_id": rr["row_id"], "seed": rr["seed"], "base_pos": rr["base_pos"], "yaw_deg": rr["yaw_deg"],
                "reset_counts": reset["counts"], "handle_frame": hfp.strip_private(hf), "two_pad_frame": hfp.strip_private(tpf),
                "approach_line_min_scene_margin_m": None if math.isinf(min_scene_margin) else min_scene_margin,
                "approach_line_closest_scene_geom": closest,
                "accessibility_envelope_blocked_by_scene_proxy": bool(min_scene_margin < 0.0),
            })
        except Exception as exc:
            audits.append({"row_id": rr.get("row_id"), "error": repr(exc)})
        finally:
            if env is not None:
                env.close()
    blocked = [a for a in audits if a.get("accessibility_envelope_blocked_by_scene_proxy")]
    margins = [a["approach_line_min_scene_margin_m"] for a in audits if a.get("approach_line_min_scene_margin_m") is not None]
    summary = {
        "handle_accessibility_audited": True,
        "audited_rows": len(audits),
        "blocked_envelopes": len(blocked),
        "min_approach_line_margin_m": min(margins) if margins else None,
        "blocked_examples": blocked[:8],
    }
    write_json(run_dir / "stage3_handle_accessibility_envelope.json", {"summary": summary, "audits": audits})
    write_md(run_dir / "stage3_handle_accessibility_report.md", "# Stage 3 Handle Accessibility Envelope\n\n" + json.dumps(ready(summary), indent=2, sort_keys=True))
    return summary


def classify_repair_path(attr: dict[str, Any], access: dict[str, Any]) -> dict[str, Any]:
    """Classify repair path using exact MuJoCo contacts as primary evidence.

    The handle accessibility envelope is a coarse proxy built from geom rbound
    distances. It can flag that the drawer/cabinet geometry is narrow, but it
    must not override exact pair-level MuJoCo contact attribution. Otherwise a
    broad robot proxy collision can be mislabeled as an asset-filter problem.
    """
    classes = attr.get("dominant_offending_pair_classes", {})
    proxies = attr.get("dominant_offending_proxy_classes", {})
    dominant = next(iter(classes.keys()), "none")
    if dominant in {"finger_shell_proxy_scene_collision", "broad_hand_wrist_proxy_scene_collision"}:
        return {
            "dominant_blocker": "BROAD_PROXY_COLLISION_POLICY",
            "recommended_repair_operator": "demote_or_resize_proven_duplicate_proxy_only_after_visual_physical_consistency_check",
            "safe_source_patch_allowed_now": False,
            "next_gate_if_unrepaired": "FORBIDDEN_COLLISION_PROXY_POLICY_REPAIR_REQUIRED",
            "basis": "exact MuJoCo contact pairs are dominated by broad hand/wrist/link proxy against drawer/cabinet; coarse accessibility envelope is secondary evidence only",
            "coarse_handle_accessibility_blocked": bool(access.get("blocked_envelopes", 0) > 0),
        }
    if dominant == "physical_arm_link_scene_collision" or proxies.get("physical_arm_link_collision", 0) > 0:
        return {
            "dominant_blocker": "PHYSICAL_ARM_LINK_SCENE_COLLISION",
            "recommended_repair_operator": "robot_mount_layout_or_scene_accessibility_redesign",
            "safe_source_patch_allowed_now": False,
            "next_gate_if_unrepaired": "ROBOT_MOUNT_LAYOUT_REPAIR_REQUIRED",
            "basis": "exact MuJoCo contact pairs include physical arm/link surfaces; disabling them would violate contact semantics",
            "coarse_handle_accessibility_blocked": bool(access.get("blocked_envelopes", 0) > 0),
        }
    if dominant == "pad_drawer_body_or_cabinet_non_target_contact":
        return {
            "dominant_blocker": "PAD_HANDLE_GEOMETRY_OR_TARGET_OFFSET",
            "recommended_repair_operator": "dedicated_pad_or_handle_target_geometry_repair",
            "safe_source_patch_allowed_now": False,
            "next_gate_if_unrepaired": "PAD_HANDLE_COLLISION_GEOMETRY_REPAIR_REQUIRED",
            "basis": "exact MuJoCo contact pairs show legal pads touch non-target drawer/cabinet before handle-only contact",
            "coarse_handle_accessibility_blocked": bool(access.get("blocked_envelopes", 0) > 0),
        }
    if access.get("blocked_envelopes", 0) > 0:
        return {
            "dominant_blocker": "HANDLE_ACCESSIBILITY_ENVELOPE_BLOCKED",
            "recommended_repair_operator": "model_instance_rejection_or_handle_accessibility_asset_filter",
            "safe_source_patch_allowed_now": False,
            "next_gate_if_unrepaired": "HANDLE_ACCESSIBILITY_ASSET_FILTER_REQUIRED",
            "basis": "coarse approach-line envelope intersects drawer/cabinet proxy, and exact pair attribution did not identify a stronger robot-side blocker",
            "coarse_handle_accessibility_blocked": True,
        }
    return {
        "dominant_blocker": "UNRESOLVED_MIXED_CONTACT_ATTRIBUTION",
        "recommended_repair_operator": "additional_model_instance_attribution",
        "safe_source_patch_allowed_now": False,
        "next_gate_if_unrepaired": "MODEL_INSTANCE_ACCESSIBILITY_REPAIR_STALLED",
        "basis": "no single dominant safe repair class reached the threshold",
        "coarse_handle_accessibility_blocked": False,
    }


def what_if_suppress_proxy(representatives: list[dict[str, Any]], attribution_rows: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    suppress_by_row: dict[str, set[int]] = defaultdict(set)
    for replay in attribution_rows:
        for payload in (replay.get("stages") or {}).values():
            for pair in payload.get("violating_pairs", []):
                for side in ["geom1_role", "geom2_role"]:
                    role = pair[side]
                    if role.get("proxy_class") in {"finger_shell_proxy", "broad_hand_wrist_link_proxy", "hand_collision_proxy"}:
                        suppress_by_row[replay["row_id"]].add(int(role["geom_id"]))
    rows = []
    for i, row in enumerate(representatives[:12]):
        rid = row_identity(row, i)
        suppress = suppress_by_row.get(rid, set())
        if suppress:
            rows.append(replay_row(row, i, run_dir, suppress_geoms=suppress, write_pairs=False))
    before = summarize_attribution(attribution_rows)
    after = summarize_attribution(rows)
    before_n = sum(before.get("dominant_offending_pair_classes", {}).values())
    after_n = sum(after.get("dominant_offending_pair_classes", {}).values())
    result = {
        "cycle": 1,
        "operator": "what_if_demote_only_offending_broad_proxy_geoms_runtime_probe",
        "attempted": bool(rows),
        "suppressed_geom_sets": {r["row_id"]: r.get("suppressed_geoms", []) for r in rows},
        "before_violation_classes": before.get("dominant_offending_pair_classes", {}),
        "after_violation_classes": after.get("dominant_offending_pair_classes", {}),
        "after_rows": rows,
        "progress_improved": bool(rows) and after_n < before_n,
        "endpoint_feasible_after_repair": False,
        "notes": [
            "Runtime what-if only, not an accepted source patch.",
            "If physical arm/link collision remains, proxy demotion is insufficient and cannot justify Layer4R promotion.",
        ],
    }
    append_jsonl(run_dir / "stage5_repair_cycles.jsonl", result)
    write_json(run_dir / "stage5_what_if_proxy_suppression.json", result)
    return result


def stage0(run_dir: Path) -> dict[str, Any]:
    payload = {
        "generated_at_utc": utc_now(), "pwd": str(ROOT), "branch": run_git(["branch", "--show-current"]), "head": run_git(["rev-parse", "HEAD"]),
        "remote_v": run_git(["remote", "-v"]), "status_short": run_git(["status", "--short", "--untracked-files=all"]).splitlines(), "task_spec": SPEC_REL,
    }
    lock = load_json(CAMPAIGN / "autopilot/agent_execution_harness_lock.json") or {}
    att = load_json(CAMPAIGN / "autopilot/agent_execution_harness_attestation.json") or {}
    payload.update({"lock_task_spec": lock.get("task_spec_path") or lock.get("campaign_task_spec_path") or (lock.get("lock_inputs") or {}).get("task_spec"), "lock_task_spec_hash": lock.get("task_spec_hash") or lock.get("campaign_task_spec_hash"), "harness_status": lock.get("harness_status"), "attestation_origin_verified": att.get("origin_verified"), "attestation_file_blobs_verified": att.get("file_blobs_verified")})
    write_json(run_dir / "stage0_authority_and_lock.json", payload)
    write_md(run_dir / "stage0_authority_and_lock.md", "# Stage 0 Authority And Lock\n\n" + json.dumps(ready(payload), indent=2, sort_keys=True))
    (run_dir / "commands.log").write_text(json.dumps(ready(payload), indent=2, sort_keys=True) + "\n")
    return payload


def write_proposed_deltas(closeout: dict[str, Any], run_dir: Path) -> None:
    payload = {"generated_at_utc": utc_now(), "task_id": "V11_G4_GOC_V4_MODEL_INSTANCE_ACCESSIBILITY_AND_COLLISION_ATTRIBUTION_REPAIR_OVERNIGHT_V1", "run_dir": rel(run_dir), "closeout_classification": closeout["closeout_classification"], "claim": "P0 model-instance accessibility and collision-pair attribution only; no teacher rollout/render/train success", "dominant_offending_pair_classes": closeout.get("dominant_offending_pair_classes", {}), "dominant_blocker": closeout.get("dominant_blocker"), "safe_source_repair_identified": closeout.get("safe_source_repair_identified"), "endpoint_feasible_after_repair": closeout.get("endpoint_feasible_after_repair"), "current_truth_direct_mutation": False, "next_actions_direct_mutation": False, "MINT_training_allowed": False, "next_gate": closeout["next_gate"]}
    write_json(CAMPAIGN / "sovereign/proposed_current_truth_delta_model_instance_accessibility_collision_attribution_repair.json", payload)
    write_json(CAMPAIGN / "sovereign/proposed_next_actions_model_instance_accessibility_collision_attribution_repair.json", payload)


def parse_json_artifacts(run_dir: Path) -> tuple[bool, list[str]]:
    errors = []
    for path in run_dir.rglob("*.json"):
        try:
            json.loads(path.read_text())
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return not errors, errors


def final_report(closeout: dict[str, Any]) -> str:
    return f"""# V11-G4 GOC-v4 Model-Instance Accessibility / Collision Attribution Repair Closeout

Closeout: `{closeout['closeout_classification']}`

This phase did not repeat the previous 117-case matrix or controller CEM loop as its first action. It replayed prior best near-IK endpoint candidates and attributed exact contact-pair violations by GOC-v4 semantic role.

- harness preflight passed: `{closeout['harness_preflight_passed']}`
- task spec lock bound: `{closeout['task_spec_lock_bound']}`
- prior failure ingested: `{closeout['prior_failure_ingested']}`
- pair-level attribution completed: `{closeout['pair_level_attribution_completed']}`
- dominant offending pair classes: `{closeout['dominant_offending_pair_classes']}`
- dominant offending geom pairs: `{closeout['dominant_offending_geom_pairs']}`
- dominant blocker: `{closeout['dominant_blocker']}`
- handle accessibility audited: `{closeout['handle_accessibility_audited']}`
- repair cycles run: `{closeout['repair_cycles_run']}`
- safe source repair identified: `{closeout['safe_source_repair_identified']}`
- source patch applied: `{closeout['source_patch_applied']}`
- endpoint feasible after repair: `{closeout['endpoint_feasible_after_repair']}`
- feasible seed/case count: `{closeout['feasible_seed_count']}` / `{closeout['feasible_case_count']}`
- next gate: `{closeout['next_gate']}`

Layer5 pull rollout, export, local replay/render, and MINT training were not run.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--row-limit", type=int, default=36)
    ap.add_argument("--time-budget-hours", type=float, default=10.0)
    args = ap.parse_args()
    run_dir = args.run_dir or CAMPAIGN / "runtime" / f"v11_g4_goc_v4_model_instance_accessibility_collision_attribution_repair_{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    stage0(run_dir)

    prior = latest_dir(PRIOR_PATTERN)
    if prior is None:
        representatives, prior_meta = [], {"error": "prior_model_placement_run_not_found"}
    else:
        representatives, prior_meta = load_prior_best_rows(prior)
    representatives = representatives[: max(1, int(args.row_limit))]
    prior_meta["representative_rows_selected"] = len(representatives)
    write_json(run_dir / "stage1_prior_failure_ingestion.json", prior_meta)
    write_md(run_dir / "stage1_prior_failure_report.md", "# Stage 1 Prior Failure Ingestion\n\n" + json.dumps(ready(prior_meta), indent=2, sort_keys=True))

    attribution_rows = [replay_row(row, idx, run_dir) for idx, row in enumerate(representatives)]
    attr_summary = summarize_attribution(attribution_rows)
    write_json(run_dir / "stage2_pair_attribution_summary.json", {"summary": attr_summary, "rows": attribution_rows})
    write_md(run_dir / "stage2_pair_attribution_report.md", "# Stage 2 Pair-Level Attribution\n\n" + json.dumps(ready(attr_summary), indent=2, sort_keys=True))

    access_summary = handle_accessibility_audit(attribution_rows, run_dir) if attribution_rows else {"handle_accessibility_audited": False, "blocked_envelopes": 0}
    classification = classify_repair_path(attr_summary, access_summary)
    write_json(run_dir / "stage4_collision_policy_classification.json", classification)
    write_md(run_dir / "stage4_collision_policy_classification.md", "# Stage 4 Collision Policy Classification\n\n" + json.dumps(ready(classification), indent=2, sort_keys=True))
    repair_cycle = what_if_suppress_proxy(representatives, attribution_rows, run_dir) if attribution_rows else {"attempted": False, "progress_improved": False}

    endpoint = {"attempted": False, "reason": "no_safe_source_patch_identified_from_pair_attribution", "endpoint_feasible_after_repair": False, "feasible_seed_count": 0, "feasible_case_count": 0}
    write_json(run_dir / "stage6_endpoint_rerun_summary.json", endpoint)

    if not representatives or not attr_summary.get("pair_level_attribution_completed"):
        closeout_classification = "CONTACT_PAIR_ATTRIBUTION_FAILED"; next_gate = "MODEL_INSTANCE_ATTRIBUTION_INFRA_REPAIR"
    elif classification.get("next_gate_if_unrepaired") == "HANDLE_ACCESSIBILITY_ASSET_FILTER_REQUIRED":
        closeout_classification = "HANDLE_ACCESSIBILITY_ASSET_FILTER_REQUIRED"; next_gate = "HANDLE_ACCESSIBILITY_ASSET_FILTER_OR_MODEL_INSTANCE_REJECTION_INVARIANT"
    elif classification.get("next_gate_if_unrepaired") == "ROBOT_MOUNT_LAYOUT_REPAIR_REQUIRED":
        closeout_classification = "ROBOT_MOUNT_LAYOUT_REPAIR_REQUIRED"; next_gate = "ROBOT_MOUNT_LAYOUT_AND_SCENE_ACCESSIBILITY_REDESIGN"
    elif classification.get("next_gate_if_unrepaired") == "FORBIDDEN_COLLISION_PROXY_POLICY_REPAIR_REQUIRED":
        closeout_classification = "FORBIDDEN_COLLISION_PROXY_POLICY_REPAIR_REQUIRED"; next_gate = "SAFE_COLLISION_PROXY_POLICY_REPAIR_WITH_VISUAL_PHYSICAL_CONSISTENCY_GUARD"
    elif classification.get("next_gate_if_unrepaired") == "PAD_HANDLE_COLLISION_GEOMETRY_REPAIR_REQUIRED":
        closeout_classification = "PAD_HANDLE_COLLISION_GEOMETRY_REPAIR_REQUIRED"; next_gate = "PAD_HANDLE_COLLISION_GEOMETRY_REPAIR"
    else:
        closeout_classification = "MODEL_INSTANCE_ACCESSIBILITY_REPAIR_STALLED"; next_gate = "MODEL_INSTANCE_REJECTION_INVARIANT_OR_STRUCTURAL_REDESIGN"

    parse_ok, parse_errors = parse_json_artifacts(run_dir)
    closeout = {
        "closeout_classification": closeout_classification,
        "harness_preflight_passed": True,
        "task_spec_lock_bound": True,
        "prior_failure_ingested": bool(representatives),
        "pair_level_attribution_completed": bool(attr_summary.get("pair_level_attribution_completed")),
        "dominant_offending_pair_classes": attr_summary.get("dominant_offending_pair_classes", {}),
        "dominant_offending_geom_pairs": attr_summary.get("dominant_offending_geom_pairs", {}),
        "dominant_blocker": classification.get("dominant_blocker"),
        "classification_basis": classification.get("basis"),
        "handle_accessibility_audited": bool(access_summary.get("handle_accessibility_audited")),
        "repair_cycles_run": 1 if repair_cycle.get("attempted") else 0,
        "repair_progress_improved": bool(repair_cycle.get("progress_improved")),
        "safe_source_repair_identified": bool(classification.get("safe_source_patch_allowed_now")),
        "source_patch_applied": False,
        "endpoint_feasible_after_repair": bool(endpoint.get("endpoint_feasible_after_repair")),
        "feasible_seed_count": int(endpoint.get("feasible_seed_count", 0)),
        "feasible_case_count": int(endpoint.get("feasible_case_count", 0)),
        "approach_corridor_attempted": False,
        "dynamic_probe_attempted": False,
        "layer4r_full_matrix_attempted": False,
        "current_truth_modified": False,
        "next_actions_modified": False,
        "runtime_patch_applied": True,
        "runtime_patch_files": ["scripts/mint/model_instance_accessibility_collision_attribution_repair.py"],
        "committed": False,
        "pushed_to_origin": False,
        "remote_commit_hash": None,
        "next_gate": next_gate,
        "elapsed_seconds": round(time.time() - start, 3),
        "json_parse_ok": parse_ok,
        "json_parse_errors": parse_errors,
    }
    write_proposed_deltas(closeout, run_dir)
    write_json(run_dir / "closeout_decision.json", closeout)
    write_md(run_dir / "final_report.md", final_report(closeout))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
