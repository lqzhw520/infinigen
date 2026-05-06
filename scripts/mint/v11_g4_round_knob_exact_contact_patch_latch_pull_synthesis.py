#!/usr/bin/env python3
"""Exact round-knob pad-patch and latch-pull synthesis for V11 GOC-v4.

This phase upgrades the fingertip authority from a single point-like exact pad
per finger to explicit finite left/right pad groups, then re-runs strict gates
using left-group AND right-group exact knob contact. Geometric bilateral grasp is
kept diagnostic-only.
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
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import mujoco
import numpy as np

ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN = ROOT / "experiments/mint/mint_drawer_v1"
TASK_ID = "V11_G4_GOC_V4_ROUND_KNOB_EXACT_CONTACT_PATCH_AND_LATCH_PULL_SYNTHESIS_TO_STRICT_REPLAY_TEXTURED_REVIEW_V2"
SPEC_REL = "experiments/mint/mint_drawer_v1/sovereign/experiment_specs/v11_g4_goc_v4_round_knob_exact_contact_patch_latch_pull_synthesis_to_strict_replay_textured_review.yaml"
RUN_PREFIX = "v11_g4_goc_v4_round_knob_exact_contact_patch_latch_pull_synthesis_to_strict_replay_textured_review"
PREV_PREFIX = (
    "v11_g4_goc_v4_handle_affordance_strategy_registry_round_knob_latch_pull_repair_"
)
SUCCESS = (
    "ROUND_KNOB_EXACT_CONTACT_PATCH_LATCH_PULL_STRICT_REPLAY_TEXTURED_REVIEW_READY"
)
STRICT_DRAWER_FRACTION = 0.80
MAX_PENETRATION_M = 0.02
MAX_FORCE_N = 1_000_000.0
MIN_BILATERAL_EXACT_FRAMES = 30

sys.path.insert(0, str(ROOT / "scripts/mint"))
import contact_aware_drawer_teacher as teacher  # noqa: E402
import v11_g4_dynamic_pull_accessible_pool_co_design as cd  # noqa: E402
import v11_g4_fast_guarded_contact_topology_realistic_codesign_repair as fast  # noqa: E402
import v11_g4_round_knob_grasp_latch_pull_controller as rk  # noqa: E402
import v11_g4_visual_topology_realism_and_action_replay_repair_v2 as v2  # noqa: E402


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
        return {
            str(k): ready(v) for k, v in value.items() if not str(k).startswith("_")
        }
    if isinstance(value, (list, tuple, set)):
        return [ready(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ready(payload), indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(ready(payload), sort_keys=True) + "\n")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path)


def run_git(args: list[str]) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    return proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metric_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        value = row.get(key)
        return default if value is None else int(value)
    except Exception:
        return default


def metric_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key)
        out = default if value is None else float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def max_consecutive(values: list[bool]) -> int:
    best = cur = 0
    for value in values:
        cur = cur + 1 if value else 0
        best = max(best, cur)
    return best


def latest_run(prefix: str) -> Path | None:
    runs = sorted((CAMPAIGN / "runtime").glob(f"{prefix}*"))
    runs = [p for p in runs if (p / "final_closeout.json").exists()]
    return runs[-1] if runs else None


def install_patch() -> None:
    rk.install_patch()
    for mod in (rk, fast, cd, v2):
        mod.SPEC_REL = SPEC_REL
        mod.TASK_ID = TASK_ID
        mod.RUN_PREFIX = RUN_PREFIX


def prior_rows(prev: Path) -> list[dict[str, Any]]:
    return read_jsonl(prev / "fast_case_latch_pull_solver_results.jsonl")


def row_rank_open(row: dict[str, Any]) -> tuple[float, float, float]:
    legal = (
        metric_int(row, "forbidden_contact_frames") == 0
        and metric_int(row, "handle_nonlegal_contact_frames") == 0
    )
    return (
        1.0 if legal else 0.0,
        metric_float(row, "max_drawer_fraction"),
        -metric_float(row, "max_penetration_m", 1.0),
    )


def row_rank_latch(row: dict[str, Any]) -> tuple[float, float, float]:
    legal = (
        metric_int(row, "forbidden_contact_frames") == 0
        and metric_int(row, "handle_nonlegal_contact_frames") == 0
    )
    return (
        1.0 if legal else 0.0,
        metric_int(row, "pull_phase_two_pad_target_contact_frames"),
        metric_float(row, "max_drawer_fraction"),
    )


def row_rank_balanced(row: dict[str, Any]) -> tuple[float, float, float]:
    legal = (
        metric_int(row, "forbidden_contact_frames") == 0
        and metric_int(row, "handle_nonlegal_contact_frames") == 0
    )
    exact = min(metric_int(row, "pull_phase_two_pad_target_contact_frames") / 30.0, 1.0)
    frac = min(metric_float(row, "max_drawer_fraction") / STRICT_DRAWER_FRACTION, 1.0)
    return (
        1.0 if legal else 0.0,
        exact + frac,
        metric_float(row, "max_drawer_fraction"),
    )


def choose_prior_evidence(prev: Path) -> dict[str, Any]:
    rows = prior_rows(prev)
    report = load_json(prev / "fast_case_latch_pull_solver_report.json", {}) or {}
    legal = [
        r
        for r in rows
        if metric_int(r, "forbidden_contact_frames") == 0
        and metric_int(r, "handle_nonlegal_contact_frames") == 0
    ]
    best_opening = report.get("best_opening") or (
        max(legal, key=row_rank_open) if legal else {}
    )
    best_exact = max(legal, key=row_rank_latch) if legal else {}
    balanced = max(legal, key=row_rank_balanced) if legal else {}
    return {
        "latest_previous_run": rel(prev),
        "previous_final_closeout": load_json(prev / "final_closeout.json", {}),
        "previous_solver_report": report,
        "rows_total": len(rows),
        "best_high_opening_legal_row": best_opening,
        "best_exact_latch_under_opening_row": best_exact,
        "balanced_row": balanced,
    }


def source_candidates(prev: Path) -> list[dict[str, Any]]:
    candidates = rk.source_candidates()
    by_id = {str(c.get("candidate_id")): c for c in candidates}
    ids = []
    ev = choose_prior_evidence(prev)
    for key in [
        "best_high_opening_legal_row",
        "best_exact_latch_under_opening_row",
        "balanced_row",
    ]:
        cid = str((ev.get(key) or {}).get("candidate_id") or "")
        if cid and cid not in ids:
            ids.append(cid)
    preferred = []
    for cid in ids:
        if cid in by_id:
            preferred.append(by_id[cid])
    for cid in [
        "fg_cycle2_05_d0p9_02",
        "fg_cycle2_05_d1p6_04",
        "v2_short_stub_island_densify_05",
        "v2_short_stub_island_densify_15",
    ]:
        if cid in by_id and by_id[cid] not in preferred:
            preferred.append(by_id[cid])
    return preferred or candidates[:4]


def write_stage0(run_dir: Path, prev: Path) -> dict[str, Any]:
    evidence = choose_prior_evidence(prev)
    payload = {
        "generated_at_utc": utc_now(),
        "task_id": TASK_ID,
        "root_case": "ROUND_KNOB_EXACT_BILATERAL_CONTACT_PATCH_AND_LATCH_PULL_WRENCH_TRANSMISSION_NOT_SYNTHESIZED",
        "previous_baseline_preserved": True,
        "previous_contact_discretization_closeout_is_not_final_failure": True,
        "pwd": str(ROOT),
        "branch": run_git(["branch", "--show-current"]),
        "head": run_git(["rev-parse", "HEAD"]),
        "git_status_short": run_git(["status", "--short"]).splitlines(),
        **evidence,
    }
    write_json(run_dir / "stage0_failure_ingestion.json", payload)
    bopen = evidence.get("best_high_opening_legal_row") or {}
    bexact = evidence.get("best_exact_latch_under_opening_row") or {}
    write_md(
        run_dir / "stage0_failure_report.md",
        f"""# Stage 0 Failure Ingestion

Previous run: `{rel(prev)}`

Previous closeout remains bounded as `CONTACT_DISCRETIZATION_OR_PAD_PATCH_REPAIR_REQUIRED`, not final failure.

Best high-opening legal row:
- candidate: `{bopen.get('candidate_id')}`
- variant: `{bopen.get('variant_name')}`
- drawer_fraction: `{bopen.get('max_drawer_fraction')}`
- exact_two_pad_pull_frames: `{bopen.get('pull_phase_two_pad_target_contact_frames')}`
- forbidden: `{bopen.get('forbidden_contact_frames')}`

Best exact-latch under-opening row:
- candidate: `{bexact.get('candidate_id')}`
- variant: `{bexact.get('variant_name')}`
- drawer_fraction: `{bexact.get('max_drawer_fraction')}`
- exact_two_pad_pull_frames: `{bexact.get('pull_phase_two_pad_target_contact_frames')}`

Current root case: `ROUND_KNOB_EXACT_BILATERAL_CONTACT_PATCH_AND_LATCH_PULL_WRENCH_TRANSMISSION_NOT_SYNTHESIZED`.
""",
    )
    return payload


def geom_name(model: mujoco.MjModel, gid: int) -> str:
    return str(model.geom(int(gid)).name or f"geom_{gid}")


def body_name(model: mujoco.MjModel, bid: int) -> str:
    return str(model.body(int(bid)).name or f"body_{bid}")


def group_contract(
    env: Any, binding: dict[str, Any], patch_kind: str
) -> dict[str, Any]:
    model = env.model
    legal = [int(g) for g in binding.get("legal_gripper_surface_geom_ids", [])]
    left, right = [], []
    for gid in legal:
        name = geom_name(model, gid).lower()
        body = body_name(model, int(model.geom_bodyid[gid])).lower()
        if "finger1" in name or "finger_joint1" in body or "leftfinger" in body:
            left.append(gid)
        elif "finger2" in name or "finger_joint2" in body or "rightfinger" in body:
            right.append(gid)
    handle = [int(g) for g in binding.get("drawer_handle_geom_ids", [])]
    forbidden = [int(g) for g in binding.get("forbidden_robot_surface_geom_ids", [])]
    drawer = [int(g) for g in binding.get("drawer_body_or_cabinet_geom_ids", [])]
    unknown = [int(g) for g in binding.get("unknown_contact_relevant_geom_ids", [])]
    return {
        "patch_kind": patch_kind,
        "left_finger_pad_group_geom_ids": sorted(left),
        "left_finger_pad_group_geom_names": [geom_name(model, g) for g in sorted(left)],
        "right_finger_pad_group_geom_ids": sorted(right),
        "right_finger_pad_group_geom_names": [
            geom_name(model, g) for g in sorted(right)
        ],
        "true_knob_handle_geom_ids": sorted(handle),
        "true_knob_handle_geom_names": [geom_name(model, g) for g in sorted(handle)],
        "forbidden_robot_geom_ids": sorted(forbidden),
        "drawer_or_cabinet_geom_ids": sorted(drawer),
        "unknown_contact_relevant_geom_ids": sorted(unknown),
        "left_group_non_empty": bool(left),
        "right_group_non_empty": bool(right),
        "true_knob_handle_non_empty": bool(handle),
        "left_right_disjoint": not (set(left) & set(right)),
        "target_groups_disjoint_from_forbidden": not (
            (set(left) | set(right)) & set(forbidden)
        ),
        "handle_disjoint_from_robot": not (
            set(handle) & (set(left) | set(right) | set(forbidden))
        ),
        "unknown_contact_relevant_geoms_absent": not unknown,
        "broad_shell_link_wrist_arm_finger_shell_target_rejected": True,
        "geometric_proxy_rejected_as_success": True,
    }


def contract_passed(contract: dict[str, Any]) -> bool:
    return all(
        bool(contract.get(k))
        for k in [
            "left_group_non_empty",
            "right_group_non_empty",
            "true_knob_handle_non_empty",
            "left_right_disjoint",
            "target_groups_disjoint_from_forbidden",
            "handle_disjoint_from_robot",
            "unknown_contact_relevant_geoms_absent",
        ]
    )


def make_env_for_candidate(candidate: dict[str, Any], qpos: list[float] | None = None):
    init = qpos or candidate.get("reset", {}).get("qpos_arm")
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        env = cd.bp.cp.layer4r.make_candidate_env(
            candidate, robot_init_qpos=init, max_steps=20
        )
        env.reset()
    return env


def live_group_state(env: Any, contract: dict[str, Any]) -> dict[str, Any]:
    left = set(contract.get("left_finger_pad_group_geom_ids") or [])
    right = set(contract.get("right_finger_pad_group_geom_ids") or [])
    knob = set(contract.get("true_knob_handle_geom_ids") or [])
    left_exact = right_exact = False
    left_force = right_force = 0.0
    left_impulse = right_impulse = 0.0
    max_force = 0.0
    max_pen = 0.0
    pull_axis = np.asarray([-1.0, 0.0, 0.0], dtype=float)
    wrench_proxy = 0.0
    normal_alignment: list[float] = []
    for ci in range(int(env.data.ncon)):
        c = env.data.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(env.model, env.data, ci, force)
        normal_force = abs(float(force[0]))
        max_force = max(max_force, normal_force)
        max_pen = max(max_pen, max(0.0, -float(c.dist)))
        normal = np.asarray(c.frame, dtype=float).reshape(3, 3)[0]
        if (g1 in left and g2 in knob) or (g2 in left and g1 in knob):
            left_exact = True
            left_force += normal_force
            left_impulse += normal_force * float(env.model.opt.timestep)
            wrench_proxy += normal_force * abs(float(np.dot(normal, pull_axis)))
            pad_gid = g1 if g1 in left else g2
            knob_gid = g2 if pad_gid == g1 else g1
            radial = env.data.geom_xpos[pad_gid] - env.data.geom_xpos[knob_gid]
            nr = np.linalg.norm(radial)
            if nr > 1e-9:
                normal_alignment.append(abs(float(np.dot(normal, radial / nr))))
        if (g1 in right and g2 in knob) or (g2 in right and g1 in knob):
            right_exact = True
            right_force += normal_force
            right_impulse += normal_force * float(env.model.opt.timestep)
            wrench_proxy += normal_force * abs(float(np.dot(normal, pull_axis)))
            pad_gid = g1 if g1 in right else g2
            knob_gid = g2 if pad_gid == g1 else g1
            radial = env.data.geom_xpos[pad_gid] - env.data.geom_xpos[knob_gid]
            nr = np.linalg.norm(radial)
            if nr > 1e-9:
                normal_alignment.append(abs(float(np.dot(normal, radial / nr))))
    return {
        "left_group_exact_contact": left_exact,
        "right_group_exact_contact": right_exact,
        "bilateral_exact_group_contact": bool(left_exact and right_exact),
        "left_contact_force_n": left_force,
        "right_contact_force_n": right_force,
        "left_contact_impulse_proxy": left_impulse,
        "right_contact_impulse_proxy": right_impulse,
        "max_contact_force_n_live": max_force,
        "max_penetration_m_live": max_pen,
        "pad_normal_vs_knob_radial_abs_mean": float(np.mean(normal_alignment))
        if normal_alignment
        else None,
        "pull_axis_wrench_projection_proxy_n": wrench_proxy,
    }


def group_distances(env: Any, contract: dict[str, Any]) -> dict[str, Any]:
    knob = [int(g) for g in contract.get("true_knob_handle_geom_ids") or []]
    left = [int(g) for g in contract.get("left_finger_pad_group_geom_ids") or []]
    right = [int(g) for g in contract.get("right_finger_pad_group_geom_ids") or []]
    if not knob:
        return {}
    knob_center = np.mean(env.data.geom_xpos[knob].astype(float), axis=0)
    knob_radius = max(float(env.model.geom_size[g][0]) for g in knob)

    def min_surface(group: list[int]) -> float | None:
        vals = []
        for gid in group:
            pad_extent = float(max(env.model.geom_size[gid]))
            vals.append(
                float(
                    np.linalg.norm(env.data.geom_xpos[gid] - knob_center)
                    - knob_radius
                    - pad_extent
                )
            )
        return min(vals) if vals else None

    ld = min_surface(left)
    rd = min_surface(right)
    return {
        "knob_center_m": knob_center.tolist(),
        "knob_radius_m": knob_radius,
        "left_pad_distance_to_knob_surface_proxy_m": ld,
        "right_pad_distance_to_knob_surface_proxy_m": rd,
        "both_pads_within_1mm": bool(
            ld is not None and rd is not None and ld <= 0.001 and rd <= 0.001
        ),
        "both_pads_within_2mm": bool(
            ld is not None and rd is not None and ld <= 0.002 and rd <= 0.002
        ),
        "both_pads_within_5mm": bool(
            ld is not None and rd is not None and ld <= 0.005 and rd <= 0.005
        ),
        "knob_between_pads_boolean_proxy": bool(
            ld is not None and rd is not None and ld <= 0.006 and rd <= 0.006
        ),
    }


def record_pair_group_state(
    rec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fallback exact group metrics from compact trace contact_pairs.

    Older traces do not always include qpos/qvel. Their contact_pairs still carry
    exact geom names, contact forces, penetration, and categories from the live
    rollout, which is the authoritative source for those compact traces.
    """

    left_exact = right_exact = False
    left_force = right_force = 0.0
    left_impulse = right_impulse = 0.0
    max_force = 0.0
    max_pen = 0.0
    pull_wrench = 0.0
    normal_alignment: list[float] = []
    for pair in rec.get("contact_pairs") or []:
        if not isinstance(pair, dict):
            continue
        force = abs(float(pair.get("normal_force_n", 0.0) or 0.0))
        pen = max(0.0, float(pair.get("penetration_m", 0.0) or 0.0))
        max_force = max(max_force, force)
        max_pen = max(max_pen, pen)
        if pair.get("category") != "target":
            continue
        names = [
            str(pair.get("geom1_name", "")).lower(),
            str(pair.get("geom2_name", "")).lower(),
        ]
        has_handle = any(
            "drawer_handle" in n or ("handle" in n and "collision" in n) for n in names
        )
        if not has_handle:
            continue
        is_left = any("finger1" in n or "left" in n for n in names)
        is_right = any("finger2" in n or "right" in n for n in names)
        normal = pair.get("normal")
        if isinstance(normal, list) and len(normal) >= 3:
            try:
                pull_wrench += force * abs(float(normal[0]))
                normal_alignment.append(abs(float(normal[0])))
            except Exception:
                pass
        if is_left:
            left_exact = True
            left_force += force
            left_impulse += force * 0.002
        if is_right:
            right_exact = True
            right_force += force
            right_impulse += force * 0.002
    state = {
        "left_group_exact_contact": left_exact,
        "right_group_exact_contact": right_exact,
        "bilateral_exact_group_contact": bool(left_exact and right_exact),
        "left_contact_force_n": left_force,
        "right_contact_force_n": right_force,
        "left_contact_impulse_proxy": left_impulse,
        "right_contact_impulse_proxy": right_impulse,
        "max_contact_force_n_live": max_force,
        "max_penetration_m_live": max_pen,
        "pad_normal_vs_knob_radial_abs_mean": float(np.mean(normal_alignment))
        if normal_alignment
        else None,
        "pull_axis_wrench_projection_proxy_n": pull_wrench,
    }
    distance = rec.get("distance") or {}
    min_legal = metric_float(distance, "min_legal_pad_to_handle_m", 999.0)
    dist = {
        "knob_center_m": distance.get("handle_center"),
        "knob_radius_m": None,
        "left_pad_distance_to_knob_surface_proxy_m": None,
        "right_pad_distance_to_knob_surface_proxy_m": None,
        "both_pads_within_1mm": bool(left_exact and right_exact) or min_legal <= 0.001,
        "both_pads_within_2mm": bool(left_exact and right_exact) or min_legal <= 0.002,
        "both_pads_within_5mm": bool(left_exact and right_exact) or min_legal <= 0.005,
        "knob_between_pads_boolean_proxy": bool(left_exact and right_exact)
        or min_legal <= 0.006,
    }
    return state, dist


def trace_group_metrics(
    trace_path: Path,
    candidate: dict[str, Any],
    out_timeseries: Path | None = None,
    label: str = "trace",
) -> dict[str, Any]:
    rows = read_jsonl(trace_path)
    env = make_env_for_candidate(candidate)
    try:
        binding = teacher.classify_instance(env)
        contract = group_contract(
            env,
            binding,
            str(
                (candidate.get("model_builder_parameters") or {})
                .get("finger_pad_contact_patch", {})
                .get("kind", "baseline")
            ),
        )
        left_frames: list[bool] = []
        right_frames: list[bool] = []
        bilateral_frames: list[bool] = []
        bilateral_pull: list[bool] = []
        bilateral_latch: list[bool] = []
        geo_frames: list[bool] = []
        out_rows = []
        max_frac = 0.0
        max_force = 0.0
        max_pen = 0.0
        forbidden_frames = 0
        handle_nonlegal_frames = 0
        pull_wrench_max = 0.0
        for idx, rec in enumerate(rows):
            qpos = rec.get("qpos")
            if isinstance(qpos, list) and len(qpos) == env.model.nq:
                env.data.qpos[:] = np.asarray(qpos, dtype=float)
                qvel = rec.get("qvel")
                if isinstance(qvel, list) and len(qvel) == env.model.nv:
                    env.data.qvel[:] = np.asarray(qvel, dtype=float)
                ctrl = rec.get("ctrl")
                if isinstance(ctrl, list) and len(ctrl) == env.model.nu:
                    env.data.ctrl[:] = np.asarray(ctrl, dtype=float)
                mujoco.mj_forward(env.model, env.data)
                state = live_group_state(env, contract)
                dist = group_distances(env, contract)
            else:
                state, dist = record_pair_group_state(rec)
            mode = str(rec.get("mode", ""))
            in_pull = mode in {"bounded_teacher_pull", "post_pull_hold"}
            in_latch = mode in {"contact_hold", "post_pull_hold"}
            left = bool(state["left_group_exact_contact"])
            right = bool(state["right_group_exact_contact"])
            bilateral = bool(left and right)
            left_frames.append(left)
            right_frames.append(right)
            bilateral_frames.append(bilateral)
            bilateral_pull.append(bilateral and in_pull)
            bilateral_latch.append(bilateral and in_latch)
            counts = rec.get("contact_counts") or {}
            forbidden_frames += int(int(counts.get("forbidden", 0) or 0) > 0)
            handle_nonlegal_frames += int(
                int(counts.get("handle_nonlegal", 0) or 0) > 0
            )
            max_frac = max(max_frac, metric_float(rec, "drawer_fraction"))
            max_force = max(
                max_force,
                metric_float(counts, "max_contact_force_n"),
                state["max_contact_force_n_live"],
            )
            max_pen = max(
                max_pen,
                metric_float(counts, "max_penetration_m"),
                state["max_penetration_m_live"],
            )
            pull_wrench_max = max(
                pull_wrench_max, state["pull_axis_wrench_projection_proxy_n"]
            )
            aperture = None
            finger = rec.get("finger_qpos") or []
            if isinstance(finger, list) and len(finger) >= 2:
                aperture = abs(float(finger[0]) - float(finger[1]))
            geo = bool(
                aperture is not None
                and aperture <= 0.035
                and dist.get("both_pads_within_5mm", False)
                and mode in {"contact_hold", "bounded_teacher_pull", "post_pull_hold"}
            )
            geo_frames.append(geo)
            if out_timeseries is not None:
                out_rows.append(
                    {
                        "label": label,
                        "frame_index": idx,
                        "mode": mode,
                        "pad1_exact_contact_with_knob": left,
                        "pad2_exact_contact_with_knob": right,
                        "left_pad_group_exact_contact": left,
                        "right_pad_group_exact_contact": right,
                        "bilateral_exact_group_contact": bilateral,
                        "geometric_bilateral_grasp": geo,
                        "gripper_aperture_proxy_m": aperture,
                        "drawer_qpos": rec.get("drawer_qpos"),
                        "drawer_fraction": rec.get("drawer_fraction"),
                        "forbidden_contact_count": counts.get("forbidden", 0),
                        "handle_nonlegal_contact_count": counts.get(
                            "handle_nonlegal", 0
                        ),
                        **dist,
                        **state,
                    }
                )
        if out_timeseries is not None:
            out_timeseries.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if out_timeseries.exists() else "w"
            with out_timeseries.open(mode) as fh:
                for row in out_rows:
                    fh.write(json.dumps(ready(row), sort_keys=True) + "\n")
        summary = {
            "label": label,
            "trace_path": rel(trace_path),
            "trace_frames": len(rows),
            "contract": contract,
            "left_group_exact_contact_frames": int(sum(left_frames)),
            "right_group_exact_contact_frames": int(sum(right_frames)),
            "bilateral_exact_group_contact_frames": int(sum(bilateral_frames)),
            "bilateral_exact_group_contact_max_consecutive_frames": int(
                max_consecutive(bilateral_frames)
            ),
            "bilateral_exact_group_contact_pull_frames": int(sum(bilateral_pull)),
            "bilateral_exact_group_contact_pull_max_consecutive_frames": int(
                max_consecutive(bilateral_pull)
            ),
            "bilateral_exact_group_contact_latch_frames": int(sum(bilateral_latch)),
            "geometric_bilateral_frames": int(sum(geo_frames)),
            "geometric_bilateral_proxy_is_diagnostic_only": True,
            "drawer_fraction_max": float(max_frac),
            "forbidden_contact_frames": int(forbidden_frames),
            "handle_nonlegal_contact_frames": int(handle_nonlegal_frames),
            "max_penetration_m": float(max_pen),
            "max_force_n": float(max_force),
            "pull_axis_wrench_projection_proxy_max_n": float(pull_wrench_max),
        }
        if (
            summary["geometric_bilateral_frames"] >= 100
            and summary["bilateral_exact_group_contact_frames"]
            < MIN_BILATERAL_EXACT_FRAMES
        ):
            summary["classification"] = "TRUE_CONTACT_PATCH_DISCRETIZATION"
        elif (
            summary["bilateral_exact_group_contact_frames"]
            >= MIN_BILATERAL_EXACT_FRAMES
            and summary["drawer_fraction_max"] < STRICT_DRAWER_FRACTION
        ):
            summary["classification"] = "EXACT_LATCH_EXISTS_BUT_PULL_WORK_INSUFFICIENT"
        elif (
            summary["bilateral_exact_group_contact_frames"] < MIN_BILATERAL_EXACT_FRAMES
        ):
            summary["classification"] = "PAD_PATCH_TOO_SMALL_OR_WRONG_ORIENTATION"
        else:
            summary["classification"] = "CONTROLLER_INTERFACE_LIMITATION"
        return summary
    finally:
        env.close()


def write_contact_forensic_pp(
    run_dir: Path, stage0: dict[str, Any], candidates: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    out = run_dir / "contact_forensic_pp_timeseries.jsonl"
    if out.exists():
        out.unlink()
    cases = [
        (
            "high_opening_legal_low_exact",
            stage0.get("best_high_opening_legal_row") or {},
        ),
        (
            "high_exact_under_opening",
            stage0.get("best_exact_latch_under_opening_row") or {},
        ),
        ("balanced", stage0.get("balanced_row") or {}),
    ]
    summaries = []
    for label, row in cases:
        cid = str(row.get("candidate_id") or "")
        trace = row.get("trace_jsonl")
        if cid not in candidates or not trace:
            summaries.append(
                {"label": label, "trace_found": False, "candidate_id": cid}
            )
            continue
        tpath = ROOT / str(trace)
        if not tpath.exists():
            summaries.append(
                {
                    "label": label,
                    "trace_found": False,
                    "candidate_id": cid,
                    "trace_path": trace,
                }
            )
            continue
        summaries.append(trace_group_metrics(tpath, candidates[cid], out, label))
    hist = Counter(s.get("classification", "MISSING") for s in summaries)
    payload = {
        "generated_at_utc": utc_now(),
        "contact_forensic_pp_attempted": True,
        "summaries": summaries,
        "classification_histogram": dict(sorted(hist.items())),
        "do_not_promote_geometric_proxy": True,
    }
    write_json(run_dir / "contact_forensic_pp_summary.json", payload)
    return payload


def patch_space() -> list[dict[str, Any]]:
    return [
        {
            "kind": "current_single_pad_baseline",
            "description": "unchanged single exact pad per finger",
        },
        {
            "kind": "enlarged_inner_pad",
            "base_size": [0.010, 0.0055, 0.010],
            "friction": [3.0, 0.06, 0.0002],
            "margin": 0.0005,
        },
        {
            "kind": "enlarged_inner_pad_margin_1mm",
            "base_size": [0.011, 0.0060, 0.011],
            "friction": [3.5, 0.08, 0.0002],
            "margin": 0.0010,
        },
        {
            "kind": "two_subpatches_z",
            "base_size": [0.009, 0.0050, 0.009],
            "subpatch_count": 2,
            "subpatch_offsets": [[0, 0, -0.006], [0, 0, 0.006]],
            "friction": [3.2, 0.07, 0.0002],
            "margin": 0.0005,
        },
        {
            "kind": "two_subpatches_z_margin_1mm",
            "base_size": [0.010, 0.0055, 0.010],
            "subpatch_count": 2,
            "subpatch_offsets": [[0, 0, -0.0065], [0, 0, 0.0065]],
            "friction": [3.8, 0.08, 0.0002],
            "margin": 0.0010,
        },
        {
            "kind": "three_subpatches_vertical",
            "base_size": [0.009, 0.0050, 0.009],
            "subpatch_count": 3,
            "subpatch_offsets": [[0, 0, -0.007], [0, 0, 0], [0, 0, 0.007]],
            "friction": [3.4, 0.07, 0.0002],
            "margin": 0.0008,
        },
        {
            "kind": "rounded_box_oriented_pad",
            "base_size": [0.012, 0.0065, 0.012],
            "friction": [4.0, 0.10, 0.0002],
            "margin": 0.0015,
            "solref": [0.006, 0.40],
        },
        {
            "kind": "two_subpatches_high_friction_soft",
            "base_size": [0.010, 0.0055, 0.010],
            "subpatch_count": 2,
            "friction": [4.5, 0.10, 0.0003],
            "margin": 0.0010,
            "solref": [0.010, 0.55],
        },
        {
            "kind": "two_subpatches_low_margin_stiff",
            "base_size": [0.010, 0.0050, 0.010],
            "subpatch_count": 2,
            "friction": [3.5, 0.08, 0.0002],
            "margin": 0.00025,
            "solref": [0.004, 0.35],
        },
        {
            "kind": "enlarged_inner_pad_no_margin_high_friction",
            "base_size": [0.012, 0.0065, 0.012],
            "friction": [4.2, 0.10, 0.0002],
            "margin": 0.0,
        },
        {
            "kind": "two_subpatches_xz",
            "base_size": [0.0095, 0.0055, 0.0095],
            "subpatch_count": 2,
            "subpatch_offsets": [[-0.003, 0, -0.0055], [0.003, 0, 0.0055]],
            "friction": [3.8, 0.08, 0.0002],
            "margin": 0.0010,
        },
        {
            "kind": "three_subpatches_xz_soft",
            "base_size": [0.009, 0.0055, 0.009],
            "subpatch_count": 3,
            "subpatch_offsets": [[-0.003, 0, -0.006], [0, 0, 0], [0.003, 0, 0.006]],
            "friction": [4.0, 0.09, 0.0003],
            "margin": 0.0012,
            "solref": [0.010, 0.60],
        },
    ]


def write_patch_space(run_dir: Path) -> list[dict[str, Any]]:
    space = patch_space()
    write_json(
        run_dir / "contact_patch_synthesis_space.json",
        {
            "generated_at_utc": utc_now(),
            "modeling_principle": "finger pad is a finite surface; success still requires left group AND right group exact knob contact",
            "forbidden": [
                "broad_shell_target",
                "arm_wrist_link_target",
                "invisible_giant_catcher_geom",
                "geometric_proxy_as_success",
            ],
            "patch_candidates": space,
        },
    )
    return space


def patch_candidates(
    base_candidates: list[dict[str, Any]], space: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    out = []
    if not base_candidates:
        return out
    for idx, patch in enumerate(space[:12]):
        parent = base_candidates[idx % min(len(base_candidates), 2)]
        cand = deepcopy(parent)
        parent_id = str(parent.get("candidate_id"))
        kind = str(patch.get("kind"))
        cand["candidate_id"] = f"{parent_id}_padpatch_{idx:02d}_{kind[:28]}"
        cand["parent_instance_id"] = parent_id
        cand["co_design_source_type"] = "round_knob_contact_patch_repair"
        cand["source_type_for_spec"] = "generated_repaired_round_knob_pad_patch"
        params = dict(cand.get("model_builder_parameters") or {})
        params["finger_pad_contact_patch"] = dict(patch)
        params["round_knob_exact_contact_patch_repair"] = True
        cand["model_builder_parameters"] = params
        out.append(cand)
    return out


def rebind_patch_groups(
    run_dir: Path, candidates: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    admitted = []
    for cand in candidates:
        patch = (cand.get("model_builder_parameters") or {}).get(
            "finger_pad_contact_patch"
        ) or {}
        try:
            env = make_env_for_candidate(cand)
            try:
                binding = teacher.classify_instance(env)
                contract = group_contract(
                    env, binding, str(patch.get("kind", "baseline"))
                )
                passed = contract_passed(contract)
            finally:
                env.close()
            row = {
                "candidate_id": cand.get("candidate_id"),
                "parent_instance_id": cand.get("parent_instance_id"),
                "patch": patch,
                "contract": contract,
                "passed": passed,
            }
        except Exception as exc:  # noqa: BLE001
            row = {
                "candidate_id": cand.get("candidate_id"),
                "parent_instance_id": cand.get("parent_instance_id"),
                "patch": patch,
                "passed": False,
                "exception": repr(exc),
                "exception_type": type(exc).__name__,
            }
        rows.append(row)
        if row.get("passed"):
            admitted.append(cand)
    report = {
        "generated_at_utc": utc_now(),
        "candidate_count": len(candidates),
        "passed_count": len(admitted),
        "rows": rows,
        "goc_v4_patch_group_rebound": bool(admitted),
    }
    write_json(run_dir / "goc_v4_patch_group_rebinding_report.json", report)
    if admitted:
        first = next(r for r in rows if r.get("passed"))
        write_json(run_dir / "goc_v4_patch_group_contract.json", first.get("contract"))
    else:
        write_json(
            run_dir / "goc_v4_patch_group_contract.json",
            {"contract_created": False, "rows": rows},
        )
    write_json(
        run_dir / "contact_patch_candidate_manifest.json",
        {
            "generated_at_utc": utc_now(),
            "candidate_count": len(candidates),
            "candidates": candidates,
        },
    )
    return admitted, report


def latch_only_variants() -> list[dict[str, Any]]:
    def v(name: str, **kw: Any) -> dict[str, Any]:
        base = {
            "name": name,
            "pull_steps": 0,
            "post_pull_hold_steps": 360,
            "pull_velocity_m_per_step": 0.0,
            "pull_distance_m": 0.0,
            "lead_cap_m": 0.0,
            "pull_press_m": 0.002,
            "op_gain": 8.0,
            "op_vel_limit": 0.055,
            "q_vel_limit": 1.6,
            "null_gain": 0.12,
            "servo_kp": 320.0,
            "servo_kd": 120.0,
            "finger_mode": "binary_close",
        }
        base.update(kw)
        return base

    return [
        v(
            "exact_latch_only_binary_close_low_press",
            pull_press_m=0.0015,
            servo_kp=300.0,
        ),
        v("exact_latch_only_binary_close_firm", pull_press_m=0.0030, servo_kp=360.0),
        v("exact_latch_only_semiclose", finger_mode="semi_close", pull_press_m=0.0025),
        v("exact_latch_only_ikhold", finger_mode="ik_hold", pull_press_m=0.0025),
        v(
            "exact_latch_only_high_friction_hold",
            pull_press_m=0.0035,
            servo_kp=380.0,
            servo_kd=145.0,
        ),
        v(
            "exact_latch_only_soft_hold",
            pull_press_m=0.0012,
            servo_kp=260.0,
            servo_kd=150.0,
        ),
        v("exact_latch_only_null_low", null_gain=0.03, pull_press_m=0.0020),
        v("exact_latch_only_null_zero", null_gain=0.0, pull_press_m=0.0020),
    ]


def pull_variants(stage0: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in [
        "best_high_opening_legal_row",
        "best_exact_latch_under_opening_row",
        "balanced_row",
    ]:
        v = (stage0.get(key) or {}).get("variant")
        if isinstance(v, dict):
            rows.append(dict(v))

    def v(name: str, **kw: Any) -> dict[str, Any]:
        base = {
            "name": name,
            "pull_steps": 6600,
            "post_pull_hold_steps": 0,
            "pull_velocity_m_per_step": 0.000105,
            "pull_distance_m": 0.35,
            "lead_cap_m": 0.032,
            "pull_press_m": 0.0025,
            "op_gain": 13.0,
            "op_vel_limit": 0.085,
            "q_vel_limit": 1.90,
            "null_gain": 0.14,
            "servo_kp": 340.0,
            "servo_kd": 135.0,
            "finger_mode": "binary_close",
        }
        base.update(kw)
        return base

    rows.extend(
        [
            v(
                "patch_locked_pull_high_open_low_press",
                pull_press_m=0.0018,
                lead_cap_m=0.034,
                pull_velocity_m_per_step=0.000115,
            ),
            v(
                "patch_locked_pull_firm_latch_axis_work",
                pull_press_m=0.0032,
                lead_cap_m=0.026,
                pull_velocity_m_per_step=0.000095,
                servo_kp=380.0,
            ),
            v(
                "patch_locked_pull_slow_high_friction",
                pull_steps=7600,
                pull_velocity_m_per_step=0.000085,
                pull_press_m=0.0030,
                lead_cap_m=0.030,
            ),
            v(
                "patch_locked_pull_reseat_like_semiclose",
                finger_mode="semi_close",
                pull_press_m=0.0024,
                lead_cap_m=0.028,
                null_gain=0.18,
            ),
            v(
                "patch_locked_pull_ikhold_axis_work",
                finger_mode="ik_hold",
                pull_press_m=0.0028,
                lead_cap_m=0.036,
                op_gain=14.0,
                op_vel_limit=0.090,
            ),
            v(
                "patch_locked_pull_long_duration_low_slip",
                pull_steps=8200,
                pull_velocity_m_per_step=0.000075,
                pull_press_m=0.0035,
                lead_cap_m=0.024,
                q_vel_limit=1.55,
            ),
        ]
    )
    seen = set()
    out = []
    for item in rows:
        name = str(item.get("name"))
        if name and name not in seen:
            item["round_knob_exact_patch_latch_pull_variant"] = True
            out.append(item)
            seen.add(name)
    return out[:64]


def run_case_with_group_metrics(
    candidate: dict[str, Any],
    perturb: str,
    variant: dict[str, Any],
    run_dir: Path,
    case_idx: int,
    stem: str,
) -> dict[str, Any]:
    try:
        row = cd.run_case(candidate, perturb, variant, run_dir, case_idx, stem)
    except Exception as exc:  # noqa: BLE001
        return {
            "candidate_id": candidate.get("candidate_id"),
            "perturbation": perturb,
            "variant_name": variant.get("name"),
            "passed": False,
            "failure_reasons": ["CASE_EXECUTION_EXCEPTION"],
            "exception": repr(exc),
            "exception_type": type(exc).__name__,
        }
    trace = row.get("trace_jsonl")
    if trace and (ROOT / str(trace)).exists():
        gm = trace_group_metrics(ROOT / str(trace), candidate)
        row.update({f"patch_group_{k}": v for k, v in gm.items() if k != "contract"})
        row["goc_v4_patch_group_contract"] = gm.get("contract")
        row["left_group_exact_contact_frames"] = gm["left_group_exact_contact_frames"]
        row["right_group_exact_contact_frames"] = gm["right_group_exact_contact_frames"]
        row["bilateral_exact_contact_frames"] = gm[
            "bilateral_exact_group_contact_frames"
        ]
        row["bilateral_exact_contact_pull_frames"] = gm[
            "bilateral_exact_group_contact_pull_frames"
        ]
        row["bilateral_exact_contact_latch_frames"] = gm[
            "bilateral_exact_group_contact_latch_frames"
        ]
        row["geometric_bilateral_frames"] = gm["geometric_bilateral_frames"]
        row["drawer_fraction_patch_group"] = gm["drawer_fraction_max"]
        row["forbidden_contact_frames_patch_group"] = gm["forbidden_contact_frames"]
        row["handle_nonlegal_contact_frames_patch_group"] = gm[
            "handle_nonlegal_contact_frames"
        ]
        row["max_penetration_m_patch_group"] = gm["max_penetration_m"]
        row["max_force_n_patch_group"] = gm["max_force_n"]
    return row


def safety_ok(row: dict[str, Any]) -> bool:
    return bool(
        metric_int(
            row,
            "forbidden_contact_frames_patch_group",
            metric_int(row, "forbidden_contact_frames"),
        )
        == 0
        and metric_int(
            row,
            "handle_nonlegal_contact_frames_patch_group",
            metric_int(row, "handle_nonlegal_contact_frames"),
        )
        == 0
        and metric_float(
            row, "max_penetration_m_patch_group", metric_float(row, "max_penetration_m")
        )
        <= MAX_PENETRATION_M
        and metric_float(
            row, "max_force_n_patch_group", metric_float(row, "max_force_n")
        )
        <= MAX_FORCE_N
        and not row.get("direct_qpos_drawer_opening")
        and not row.get("drawer_motor_command_used")
    )


def latch_only_pass(row: dict[str, Any]) -> bool:
    return bool(
        safety_ok(row)
        and metric_int(row, "left_group_exact_contact_frames")
        >= MIN_BILATERAL_EXACT_FRAMES
        and metric_int(row, "right_group_exact_contact_frames")
        >= MIN_BILATERAL_EXACT_FRAMES
        and metric_int(row, "bilateral_exact_contact_latch_frames")
        >= MIN_BILATERAL_EXACT_FRAMES
    )


def latch_pull_pass(row: dict[str, Any]) -> bool:
    return bool(
        safety_ok(row)
        and metric_float(
            row, "drawer_fraction_patch_group", metric_float(row, "max_drawer_fraction")
        )
        >= STRICT_DRAWER_FRACTION
        and metric_int(row, "bilateral_exact_contact_pull_frames")
        >= MIN_BILATERAL_EXACT_FRAMES
        and bool(
            row.get("drawer_qpos_nondecreasing_with_tolerance")
            or row.get("drawer_qpos_nondecreasing_during_pull")
        )
    )


def exact_latch_only_loop(
    run_dir: Path, candidates: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    out = run_dir / "exact_latch_only_results.jsonl"
    if out.exists():
        out.unlink()
    rows, passers = [], []
    variants = latch_only_variants()
    idx = 0
    for cand in candidates[:12]:
        for variant in variants[:8]:
            row = run_case_with_group_metrics(
                cand,
                "fast_guarded_contact",
                variant,
                run_dir,
                910000 + idx,
                "exact_latch_only",
            )
            row["exact_latch_only_passed"] = latch_only_pass(row)
            append_jsonl(out, row)
            rows.append(row)
            if row["exact_latch_only_passed"]:
                passers.append(cand)
                break
            idx += 1
    best = max(
        rows,
        key=lambda r: (
            metric_int(r, "bilateral_exact_contact_latch_frames"),
            metric_int(r, "bilateral_exact_contact_frames"),
            -metric_int(r, "forbidden_contact_frames_patch_group", 999),
        ),
        default={},
    )
    report = {
        "generated_at_utc": utc_now(),
        "attempts_total": len(rows),
        "passing_candidate_count": len(passers),
        "exact_latch_only_passed": bool(passers),
        "best_row": best,
    }
    write_json(run_dir / "exact_latch_only_report.json", report)
    return report, rows, passers


def exact_latch_pull_loop(
    run_dir: Path, candidates: list[dict[str, Any]], stage0: dict[str, Any]
) -> tuple[
    dict[str, Any], list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None
]:
    out = run_dir / "exact_latch_pull_results.jsonl"
    if out.exists():
        out.unlink()
    rows = []
    best_pass = None
    variants = pull_variants(stage0)
    idx = 0
    for cand in candidates:
        for variant in variants:
            if idx >= 64:
                break
            row = run_case_with_group_metrics(
                cand,
                "fast_guarded_contact",
                variant,
                run_dir,
                920000 + idx,
                "exact_latch_pull",
            )
            row["exact_latch_pull_passed"] = latch_pull_pass(row)
            append_jsonl(out, row)
            rows.append(row)
            if row["exact_latch_pull_passed"]:
                best_pass = row
                break
            idx += 1
        if best_pass or idx >= 64:
            break
    best_open = max(
        rows,
        key=lambda r: metric_float(
            r, "drawer_fraction_patch_group", metric_float(r, "max_drawer_fraction")
        ),
        default={},
    )
    best_bilat = max(
        rows,
        key=lambda r: metric_int(r, "bilateral_exact_contact_pull_frames"),
        default={},
    )
    if best_pass:
        classification = "EXACT_LATCH_PULL_PASSED"
    elif (
        metric_int(best_bilat, "bilateral_exact_contact_pull_frames")
        >= MIN_BILATERAL_EXACT_FRAMES
    ):
        classification = "EXACT_LATCH_PULL_UNDER_OPENED"
    elif (
        metric_float(
            best_open,
            "drawer_fraction_patch_group",
            metric_float(best_open, "max_drawer_fraction"),
        )
        >= STRICT_DRAWER_FRACTION
    ):
        classification = "CONTACT_PATCH_OR_RESEAT_REPAIR_REQUIRED"
    else:
        classification = "PULL_AXIS_WORK_UNDER_EXACT_LATCH_FAILED"
    report = {
        "generated_at_utc": utc_now(),
        "attempts_total": len(rows),
        "exact_latch_pull_passed": bool(best_pass),
        "classification": classification,
        "best_pass": best_pass or {},
        "best_opening": best_open,
        "best_bilateral_exact": best_bilat,
    }
    write_json(run_dir / "exact_latch_pull_report.json", report)
    return report, rows, best_pass, best_open


def targeted_7of7(
    run_dir: Path,
    fast_pass: dict[str, Any] | None,
    candidates: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not fast_pass:
        report = {
            "generated_at_utc": utc_now(),
            "targeted_shard_passed": False,
            "targeted_shard_attempted": False,
            "skip_reason": "fast_guarded_contact_not_recertified",
        }
        write_json(run_dir / "targeted_shard_results.json", report)
        return report, []
    cid = str(fast_pass.get("candidate_id"))
    candidate = candidates.get(cid)
    variant = (
        fast_pass.get("variant") if isinstance(fast_pass.get("variant"), dict) else None
    )
    if candidate is None or variant is None:
        report = {
            "generated_at_utc": utc_now(),
            "targeted_shard_passed": False,
            "targeted_shard_attempted": False,
            "skip_reason": "fast_pass_candidate_or_variant_missing",
        }
        write_json(run_dir / "targeted_shard_results.json", report)
        return report, []
    perturbations = list(cd.DEFAULT_TARGETED_PERTURBATIONS)
    rows = []
    out = run_dir / "targeted_shard_results.jsonl"
    if out.exists():
        out.unlink()
    for idx, pname in enumerate(perturbations[:7]):
        row = run_case_with_group_metrics(
            candidate, pname, variant, run_dir, 930000 + idx, "targeted_patch_group"
        )
        row["targeted_patch_group_passed"] = latch_pull_pass(row)
        rows.append(row)
        append_jsonl(out, row)
    report = {
        "generated_at_utc": utc_now(),
        "targeted_shard_attempted": True,
        "cases_total": len(rows),
        "cases_passed": sum(1 for r in rows if r.get("targeted_patch_group_passed")),
        "targeted_shard_passed": bool(
            rows and all(r.get("targeted_patch_group_passed") for r in rows)
        ),
        "fast_guarded_contact_included": any(
            r.get("perturbation") == "fast_guarded_contact" for r in rows
        ),
        "rows": rows,
    }
    write_json(run_dir / "targeted_shard_results.json", report)
    return report, rows


def downstream_placeholders(run_dir: Path, targeted: dict[str, Any]) -> None:
    full30 = {
        "generated_at_utc": utc_now(),
        "full30_attempted": False,
        "full30_passed": False,
        "skip_reason": "targeted_shard_not_passed",
    }
    if targeted.get("targeted_shard_passed"):
        full30["skip_reason"] = (
            "not_executed_in_this_helper_until_targeted_7of7_group_contract_is_stable"
        )
    write_json(
        run_dir / "full30_patch_group_dynamic_pull_certification_report.json", full30
    )
    (run_dir / "full30_patch_group_dynamic_pull_certification.jsonl").write_text("")
    write_json(
        run_dir / "strict_teacher_export_manifest.json",
        {"strict_export_complete": False, "skip_reason": "full30_not_passed"},
    )
    write_json(run_dir / "trace_hash_manifest.json", trace_hash_manifest(run_dir))
    write_json(
        run_dir / "local_replay_handoff_manifest.json",
        {
            "local_state_replay_passed": False,
            "skip_reason": "strict_export_not_complete",
        },
    )
    write_json(
        run_dir / "strict_replay_metrics.json",
        {
            "local_state_replay_passed": False,
            "skip_reason": "strict_export_not_complete",
        },
    )
    write_json(
        run_dir / "render_manifest.json",
        {
            "textured_claim_render_passed": False,
            "skip_reason": "strict_export_not_complete",
        },
    )
    write_md(
        run_dir / "visual_review_packet.md",
        "# Visual Review Packet\n\nNot claim-bearing because strict export/local replay was not reached. Topology realism remains inherited from previous V2 evidence; this phase is blocked at exact patch/latch-pull gate.\n",
    )
    write_json(
        run_dir / "action_only_replay_spot_check.json",
        {
            "action_only_replay_spot_check_passed": False,
            "skip_reason": "local_state_replay_not_passed",
        },
    )
    write_md(
        run_dir / "action_only_replay_spot_check.md",
        "# Action-only Spot Check\n\nNot reached because local strict replay was not reached.\n",
    )


def trace_hash_manifest(run_dir: Path) -> dict[str, Any]:
    items = []
    for path in (
        sorted((run_dir / "traces").glob("*.jsonl"))
        if (run_dir / "traces").exists()
        else []
    ):
        items.append(
            {
                "path": rel(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {"generated_at_utc": utc_now(), "trace_count": len(items), "traces": items}


def final_checks(run_dir: Path) -> dict[str, Any]:
    status = run_git(["status", "--short", "--untracked-files=all"]).splitlines()
    changed = "\n".join(status)
    payload = {
        "generated_at_utc": utc_now(),
        "git_status_short": status,
        "current_truth_modified": "experiments/mint/mint_drawer_v1/sovereign/current_truth.json"
        in changed,
        "next_actions_modified": "experiments/mint/mint_drawer_v1/sovereign/next_actions.json"
        in changed,
        "external_MINT_modified": "external/MINT" in changed,
        "previous_run_dir_modified": PREV_PREFIX in changed,
        "geometric_proxy_used_as_success": False,
        "broad_shell_target_used": False,
    }
    write_json(run_dir / "stage12_final_checks.json", payload)
    return payload


def proposed_deltas(run_dir: Path, closeout: dict[str, Any]) -> None:
    truth = {
        "proposal_id": "proposed_current_truth_delta_round_knob_exact_contact_patch_latch_pull_synthesis",
        "generated_at_utc": utc_now(),
        "do_not_apply_without_manual_review": True,
        "previous_physics_replay_success_preserved": True,
        "previous_contact_discretization_closeout_not_final_failure": True,
        "new_closeout_classification": closeout.get("closeout_classification"),
        "dataset_admission_review_ready": closeout.get("closeout_classification")
        == SUCCESS,
        "mint_train_eval_remains_blocked_unless_manual_review_approves": True,
    }
    actions = {
        "proposal_id": "proposed_next_actions_round_knob_exact_contact_patch_latch_pull_synthesis",
        "generated_at_utc": utc_now(),
        "do_not_apply_without_manual_review": True,
        "next_gate": closeout.get("next_gate"),
        "recommended_action": closeout.get("next_gate"),
    }
    tpath = (
        CAMPAIGN
        / "sovereign/proposed_current_truth_delta_round_knob_exact_contact_patch_latch_pull_synthesis.json"
    )
    apath = (
        CAMPAIGN
        / "sovereign/proposed_next_actions_round_knob_exact_contact_patch_latch_pull_synthesis.json"
    )
    write_json(tpath, truth)
    write_json(apath, actions)
    write_json(
        run_dir / "proposed_sovereign_delta_manifest.json",
        {"current_truth_delta": rel(tpath), "next_actions_delta": rel(apath)},
    )


def closeout_payload(
    run_dir: Path,
    latch_report: dict[str, Any],
    pull_report: dict[str, Any],
    targeted: dict[str, Any],
    checks: dict[str, Any],
) -> dict[str, Any]:
    best_fast = pull_report.get("best_pass") or pull_report.get("best_opening") or {}
    exact_latch_only = bool(latch_report.get("exact_latch_only_passed"))
    exact_latch_pull = bool(pull_report.get("exact_latch_pull_passed"))
    targeted_passed = bool(targeted.get("targeted_shard_passed"))
    if targeted_passed:
        classification = "FULL30_FAILED_AFTER_PATCH_REPAIR"
        next_gate = "PATCH_GROUP_FULL30_GENERALIZATION_REPAIR"
    elif exact_latch_pull:
        classification = "TARGETED_SHARD_FAILED_AFTER_PATCH_REPAIR"
        next_gate = "TARGETED_PATCH_GROUP_REPAIR"
    elif exact_latch_only:
        if pull_report.get("classification") == "EXACT_LATCH_PULL_UNDER_OPENED":
            classification = "EXACT_LATCH_PULL_UNDER_OPENED"
            next_gate = "PULL_WRENCH_TRANSMISSION_UNDER_LATCH_REPAIR"
        else:
            classification = "FAST_GUARDED_CONTACT_RECERTIFICATION_FAILED"
            next_gate = "DEDICATED_PAD_PATCH_OR_CONTACT_MODEL_REPAIR"
    else:
        classification = "EXACT_LATCH_ONLY_FAILED"
        next_gate = "GRIPPER_PAD_CONTACT_MODEL_REDESIGN"
    payload = {
        "generated_at_utc": utc_now(),
        "closeout_classification": classification,
        "contact_patch_repair_attempted": True,
        "pad_geoms_changed": True,
        "pad_group_authority_created": bool(
            (run_dir / "goc_v4_patch_group_contract.json").exists()
        ),
        "knob_geoms_changed": False,
        "goc_v4_patch_group_rebound": bool(
            load_json(run_dir / "goc_v4_patch_group_rebinding_report.json", {}).get(
                "goc_v4_patch_group_rebound"
            )
        ),
        "exact_latch_only_passed": exact_latch_only,
        "exact_latch_pull_passed": exact_latch_pull,
        "fast_guarded_contact_passed": exact_latch_pull,
        "left_group_exact_contact_frames_fast": metric_int(
            best_fast, "left_group_exact_contact_frames"
        ),
        "right_group_exact_contact_frames_fast": metric_int(
            best_fast, "right_group_exact_contact_frames"
        ),
        "bilateral_exact_contact_frames_fast": metric_int(
            best_fast, "bilateral_exact_contact_frames"
        ),
        "geometric_bilateral_frames_fast": metric_int(
            best_fast, "geometric_bilateral_frames"
        ),
        "drawer_fraction_fast": metric_float(
            best_fast,
            "drawer_fraction_patch_group",
            metric_float(best_fast, "max_drawer_fraction"),
        ),
        "forbidden_fast": metric_int(
            best_fast,
            "forbidden_contact_frames_patch_group",
            metric_int(best_fast, "forbidden_contact_frames"),
        ),
        "handle_nonlegal_fast": metric_int(
            best_fast,
            "handle_nonlegal_contact_frames_patch_group",
            metric_int(best_fast, "handle_nonlegal_contact_frames"),
        ),
        "max_penetration_fast": metric_float(
            best_fast,
            "max_penetration_m_patch_group",
            metric_float(best_fast, "max_penetration_m"),
        ),
        "targeted_shard_passed": targeted_passed,
        "full30_passed": False,
        "strict_export_complete": False,
        "local_state_replay_passed": False,
        "textured_claim_render_passed": False,
        "action_only_spot_check_passed": False,
        "current_truth_modified": bool(checks.get("current_truth_modified")),
        "next_actions_modified": bool(checks.get("next_actions_modified")),
        "committed": False,
        "pushed_to_origin": False,
        "remote_commit_hash": None,
        "next_gate": next_gate,
    }
    write_json(run_dir / "final_closeout.json", payload)
    write_md(
        run_dir / "final_closeout.md",
        f"""# Final Closeout

closeout_classification: `{classification}`
next_gate: `{next_gate}`

Exact latch-only passed: `{exact_latch_only}`.
Exact latch+pull passed: `{exact_latch_pull}`.
Targeted shard passed: `{targeted_passed}`.

Geometric bilateral grasp remains diagnostic-only and was not used as success.
""",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    install_patch()
    prev = latest_run(PREV_PREFIX)
    if prev is None:
        raise SystemExit(f"previous run not found: {PREV_PREFIX}")
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else CAMPAIGN / "runtime" / f"{RUN_PREFIX}_{utc_stamp()}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "task_scope.json",
        {
            "task_id": TASK_ID,
            "spec": SPEC_REL,
            "run_dir": rel(run_dir),
            "previous_run": rel(prev),
        },
    )
    stage0 = write_stage0(run_dir, prev)
    bases = source_candidates(prev)
    base_by_id = {str(c.get("candidate_id")): c for c in bases}
    write_json(
        run_dir / "source_candidate_manifest.json",
        {
            "generated_at_utc": utc_now(),
            "candidate_count": len(bases),
            "candidate_ids": list(base_by_id),
            "candidates": bases,
        },
    )
    forensic = write_contact_forensic_pp(run_dir, stage0, base_by_id)
    space = write_patch_space(run_dir)
    candidates = patch_candidates(bases, space)
    admitted, rebinding = rebind_patch_groups(run_dir, candidates)
    latch_report, latch_rows, latch_pass_candidates = exact_latch_only_loop(
        run_dir, admitted
    )
    pass_ids = {str(c.get("candidate_id")) for c in latch_pass_candidates}
    latch_pass_full = [c for c in admitted if str(c.get("candidate_id")) in pass_ids]
    pull_report, pull_rows, fast_pass, best_open = (
        exact_latch_pull_loop(run_dir, latch_pass_full, stage0)
        if latch_pass_full
        else (
            {
                "exact_latch_pull_passed": False,
                "classification": "EXACT_LATCH_ONLY_FAILED",
                "best_opening": {},
            },
            [],
            None,
            None,
        )
    )
    write_json(
        run_dir / "fast_guarded_contact_rec_certification.json",
        {
            "generated_at_utc": utc_now(),
            "fast_guarded_contact_passed": bool(fast_pass),
            "best_pass": fast_pass or {},
            "best_opening": best_open or {},
            "classification": pull_report.get("classification"),
        },
    )
    cand_by_id = {str(c.get("candidate_id")): c for c in admitted}
    targeted, targeted_rows = targeted_7of7(run_dir, fast_pass, cand_by_id)
    downstream_placeholders(run_dir, targeted)
    checks = final_checks(run_dir)
    closeout = closeout_payload(run_dir, latch_report, pull_report, targeted, checks)
    proposed_deltas(run_dir, closeout)
    write_json(
        run_dir / "contact_forensic_pp_taxonomy_summary.json",
        {
            "generated_at_utc": utc_now(),
            "forensic_histogram": forensic.get("classification_histogram"),
            "latch_pull_classification": pull_report.get("classification"),
            "final_closeout_classification": closeout.get("closeout_classification"),
        },
    )
    print(
        json.dumps(
            {
                "run_dir": rel(run_dir),
                "closeout_classification": closeout["closeout_classification"],
                "next_gate": closeout["next_gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
