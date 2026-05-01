#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import mujoco
import numpy as np

ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN = ROOT / "experiments/mint/mint_drawer_v1"
RUN_DIR = Path(os.environ["RUN_DIR_REL"])
PREV_RUN = (
    CAMPAIGN / "runtime/v11_g4_goc_v3_instance_bound_route_synthesis_20260501T042940Z"
)
HELPER_PATH = PREV_RUN / "instance_bound_route_synthesis.py"

spec = importlib.util.spec_from_file_location("instance_bound_helper", HELPER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load helper {HELPER_PATH}")
helper = importlib.util.module_from_spec(spec)
sys.modules["instance_bound_helper"] = helper
spec.loader.exec_module(helper)

sys.path.insert(0, str(ROOT / "scripts/mint"))
from drawer_robot_env_mujoco import DrawerRobotEnvMuJoCoLibero  # noqa: E402
from merged_model_builder import LIBERO_INIT_QPOS, MergedModelBuilder  # noqa: E402

SEEDS = [1, 11, 12, 13]


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def ready(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): ready(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [ready(v) for v in x]
    return x


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ready(obj), indent=2, sort_keys=True) + "\n")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def make_builder(base_pos: tuple[float, float, float], yaw_deg: float):
    class CandidateBuilder(MergedModelBuilder):
        def __init__(self, *args: Any, **kwargs: Any):
            kwargs["robot_base_pos"] = np.asarray(base_pos, dtype=float)
            super().__init__(*args, **kwargs)

        def build(self):
            xml, assets, metadata, sem_hash = super().build()
            if abs(float(yaw_deg)) > 1e-9:
                pos_s = " ".join(f"{float(v):.6f}" for v in self.robot_base_pos)
                half = math.radians(float(yaw_deg)) / 2.0
                quat = f"{math.cos(half):.6f} 0.000000 0.000000 {math.sin(half):.6f}"
                marker = f'<body name="robot_mount" pos="{pos_s}">'
                repl = f'<body name="robot_mount" pos="{pos_s}" quat="{quat}">'
                if marker not in xml:
                    raise RuntimeError(f"robot_mount marker not found: {marker}")
                xml = xml.replace(marker, repl, 1)
                metadata = dict(metadata)
                metadata["diagnostic_robot_base_yaw_deg"] = float(yaw_deg)
            return xml, assets, metadata, sem_hash

    return CandidateBuilder


def make_env(
    seed: int,
    base_pos: list[float],
    yaw_deg: float,
    qpos: list[float],
    max_steps: int = 2,
):
    DrawerRobotEnvMuJoCoLibero._MERGED_BUILDER_CLASS = make_builder(
        tuple(base_pos), yaw_deg
    )
    return DrawerRobotEnvMuJoCoLibero(
        seed=seed,
        image_size=64,
        max_steps=max_steps,
        contract=None,
        robot_init_qpos=list(qpos),
    )


def reset_ok_counts(counts: dict[str, Any]) -> bool:
    return (
        int(counts.get("forbidden", 0)) == 0
        and float(counts.get("max_penetration_m", 0.0)) <= 0.02
        and float(counts.get("max_contact_force_n", 0.0)) <= 1_000_000.0
    )


def joint_margin(model: mujoco.MjModel, qpos7: np.ndarray) -> float:
    lo = model.jnt_range[2:9, 0]
    hi = model.jnt_range[2:9, 1]
    return float(np.min(np.minimum(qpos7 - lo, hi - qpos7)))


def diagnostic_ik(
    env: DrawerRobotEnvMuJoCoLibero, binding: dict[str, Any]
) -> dict[str, Any]:
    start_qpos = env.data.qpos.copy()
    legal = binding["legal_gripper_surface_geom_ids"]
    handle = binding["drawer_handle_geom_ids"]
    handle_pos = env.data.geom_xpos[handle].astype(float)
    handle_center = np.mean(handle_pos, axis=0)
    axis = np.asarray(env._motion_axis, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    targets = [handle_center, handle_center - axis * 0.02, handle_center + axis * 0.02]
    for gid in handle[: min(5, len(handle))]:
        targets.append(env.data.geom_xpos[gid].astype(float))
    best = None
    for target_index, target in enumerate(targets):
        env.data.qpos[:] = start_qpos
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        for _ in range(220):
            legal_cent = np.mean(env.data.geom_xpos[legal], axis=0)
            err = np.asarray(target, dtype=float) - legal_cent
            jac = np.zeros((3, env.model.nv), dtype=np.float64)
            for gid in legal:
                jp = np.zeros((3, env.model.nv), dtype=np.float64)
                jr = np.zeros((3, env.model.nv), dtype=np.float64)
                mujoco.mj_jacGeom(env.model, env.data, jp, jr, int(gid))
                jac += jp / len(legal)
            jr = jac[:, 2:9]
            lam = 8e-4
            dq = jr.T @ np.linalg.solve(jr @ jr.T + lam * np.eye(3), err * 0.72)
            dq = np.clip(dq, -0.08, 0.08)
            lo = env.model.jnt_range[2:9, 0]
            hi = env.model.jnt_range[2:9, 1]
            env.data.qpos[2:9] = np.clip(env.data.qpos[2:9] + dq, lo, hi)
            mujoco.mj_forward(env.model, env.data)
        dm = helper.distance_metrics(env, binding)
        item = {
            "target_index": target_index,
            "target": np.asarray(target).tolist(),
            "distance": dm,
            "final_robot_qpos": env.data.qpos[2:9].copy(),
            "final_joint_margin": joint_margin(env.model, env.data.qpos[2:9]),
        }
        if (
            best is None
            or dm["min_legal_pad_to_handle_m"] + 0.35 * dm["min_eef_to_handle_m"]
            < best["distance"]["min_legal_pad_to_handle_m"]
            + 0.35 * best["distance"]["min_eef_to_handle_m"]
        ):
            best = item
    env.data.qpos[:] = start_qpos
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    assert best is not None
    return best


def qpos_candidates() -> list[list[float]]:
    base = np.asarray(LIBERO_INIT_QPOS, dtype=float)
    deltas = [
        np.zeros(7),
        np.array([0.0, -0.12, 0.0, 0.18, 0.0, -0.18, 0.0]),
        np.array([0.0, 0.12, 0.0, -0.18, 0.0, 0.18, 0.0]),
        np.array([0.08, -0.08, 0.05, 0.12, 0.0, -0.10, -0.08]),
        np.array([-0.08, 0.08, -0.05, -0.12, 0.0, 0.10, 0.08]),
    ]
    return [(base + d).tolist() for d in deltas]


def candidate_centers() -> list[dict[str, Any]]:
    search = json.loads((PREV_RUN / "base_reachability_search.json").read_text())
    rows = [r for r in search["items"] if "legal_pad_to_handle_ik_m" in r]

    def score(r: dict[str, Any]) -> float:
        counts = r["reset_contact_counts"]
        reset_penalty = (
            0.0
            if r.get("reset_ok")
            else 2.0
            + 0.01 * counts.get("forbidden", 0)
            + 5.0 * min(float(counts.get("max_penetration_m", 0.0)), 1.0)
        )
        return (
            float(r["legal_pad_to_handle_ik_m"])
            + 0.35 * float(r["eef_to_handle_ik_m"])
            + reset_penalty
        )

    top_close = sorted(rows, key=score)[:18]
    top_reset = sorted(
        [r for r in rows if r.get("reset_ok")],
        key=lambda r: r["legal_pad_to_handle_ik_m"] + 0.35 * r["eef_to_handle_ik_m"],
    )[:12]
    centers = []
    seen = set()
    for r in top_close + top_reset:
        key = (r["seed"], tuple(r["base_pos"]), float(r["yaw_deg"]))
        if key in seen:
            continue
        seen.add(key)
        centers.append(
            {
                "seed": r["seed"],
                "base_pos": r["base_pos"],
                "yaw_deg": r["yaw_deg"],
                "source_legal": r["legal_pad_to_handle_ik_m"],
                "source_eef": r["eef_to_handle_ik_m"],
                "source_reset_ok": r["reset_ok"],
            }
        )
    return centers


def candidate_grid() -> list[dict[str, Any]]:
    qposes = qpos_candidates()
    items = []
    seen = set()
    for c in candidate_centers():
        x0, y0, z0 = c["base_pos"]
        for dx in [-0.05, -0.025, 0.0, 0.025, 0.05]:
            for dy in [-0.05, 0.0, 0.05]:
                for z in [0.0, 0.025, 0.05, 0.08]:
                    for dyaw in [-10.0, 0.0, 10.0]:
                        for qi, qpos in enumerate(qposes):
                            x = float(np.clip(x0 + dx, -1.05, -0.25))
                            y = float(np.clip(y0 + dy, -0.25, 0.25))
                            yaw = float(
                                np.clip(float(c["yaw_deg"]) + dyaw, -35.0, 35.0)
                            )
                            key = (
                                int(c["seed"]),
                                round(x, 4),
                                round(y, 4),
                                round(z, 4),
                                round(yaw, 4),
                                qi,
                            )
                            if key in seen:
                                continue
                            seen.add(key)
                            items.append(
                                {
                                    "seed": int(c["seed"]),
                                    "base_pos": [x, y, float(z)],
                                    "yaw_deg": yaw,
                                    "qpos_index": qi,
                                    "robot_init_qpos": qpos,
                                    "center": c,
                                }
                            )
    # Deterministic budget; enough to be continuous-local without becoming another rollout phase.
    return items[:720]


def evaluate_candidate(c: dict[str, Any]) -> dict[str, Any]:
    env = make_env(c["seed"], c["base_pos"], c["yaw_deg"], c["robot_init_qpos"])
    try:
        env.reset()
        binding = helper.classify_instance(env, int(c["seed"]))
        reset = helper.contacts(env, binding)
        reset_ok = reset_ok_counts(reset["counts"])
        row: dict[str, Any] = {
            "seed": int(c["seed"]),
            "base_pos": c["base_pos"],
            "yaw_deg": c["yaw_deg"],
            "qpos_index": c["qpos_index"],
            "robot_init_qpos": c["robot_init_qpos"],
            "reset_ok": reset_ok,
            "reset_contact_counts": reset["counts"],
            "binding": {
                k: binding[k]
                for k in [
                    "legal_gripper_surface_geom_ids",
                    "forbidden_robot_surface_geom_ids",
                    "drawer_handle_geom_ids",
                    "drawer_body_or_cabinet_geom_ids",
                ]
            },
        }
        if reset_ok:
            ik = diagnostic_ik(env, binding)
            row.update(
                {
                    "ik": ik,
                    "legal_pad_to_handle_ik_m": ik["distance"][
                        "min_legal_pad_to_handle_m"
                    ],
                    "eef_to_handle_ik_m": ik["distance"]["min_eef_to_handle_m"],
                    "final_joint_margin": ik["final_joint_margin"],
                    "passes_joint_feasibility_gate": bool(
                        ik["distance"]["min_legal_pad_to_handle_m"] <= 0.03
                        and ik["distance"]["min_eef_to_handle_m"] <= 0.05
                        and ik["final_joint_margin"] > -1e-6
                    ),
                }
            )
        else:
            dm = helper.distance_metrics(env, binding)
            row.update(
                {"reset_distance_snapshot": dm, "passes_joint_feasibility_gate": False}
            )
        return row
    except Exception as exc:
        return {
            "seed": c.get("seed"),
            "base_pos": c.get("base_pos"),
            "yaw_deg": c.get("yaw_deg"),
            "qpos_index": c.get("qpos_index"),
            "error": repr(exc),
            "passes_joint_feasibility_gate": False,
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def run_dynamic_probe(cand: dict[str, Any]) -> dict[str, Any]:
    env = make_env(
        cand["seed"],
        cand["base_pos"],
        cand["yaw_deg"],
        cand["robot_init_qpos"],
        max_steps=180,
    )
    records = []
    try:
        env.reset()
        binding = helper.classify_instance(env, int(cand["seed"]))
        # Start from optimized IK qpos, not drawer qpos. No direct drawer opening.
        env.data.qpos[2:9] = np.asarray(cand["ik"]["final_robot_qpos"], dtype=float)
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        handle_center = np.mean(
            env.data.geom_xpos[binding["drawer_handle_geom_ids"]], axis=0
        )
        axis = np.asarray(env._motion_axis, dtype=float)
        axis = axis / max(np.linalg.norm(axis), 1e-9)
        for step in range(150):
            if step < 30:
                target = handle_center - axis * 0.02
                close = False
                phase = "settle_precontact"
            elif step < 75:
                target = handle_center
                close = True
                phase = "close_contact"
            else:
                target = handle_center + axis * min(0.08, 0.0015 * (step - 75))
                close = True
                phase = "short_pull_probe"
            action = helper.ik_action_to_pad(env, binding, target, close)
            obs, reward, done, info = env.step(action)
            cp = helper.contacts(env, binding)
            dm = helper.distance_metrics(env, binding)
            records.append(
                {
                    "step": step,
                    "phase": phase,
                    "target": target,
                    "close": close,
                    "contact_counts": cp["counts"],
                    "contact_pairs": cp["pairs"][:16],
                    "distance": dm,
                    "drawer_qpos": float(env.data.qpos[0]),
                }
            )
        target_frames = sum(1 for r in records if r["contact_counts"]["target"] > 0)
        forbidden_frames = sum(
            1 for r in records if r["contact_counts"]["forbidden"] > 0
        )
        max_force = max(r["contact_counts"]["max_contact_force_n"] for r in records)
        max_pen = max(r["contact_counts"]["max_penetration_m"] for r in records)
        passed = (
            target_frames > 0
            and forbidden_frames == 0
            and math.isfinite(max_force)
            and max_pen <= 0.02
        )
        return {
            "candidate": cand,
            "target_contact_frame_count": target_frames,
            "forbidden_contact_frame_count": forbidden_frames,
            "max_contact_force_n": max_force,
            "max_penetration_m": max_pen,
            "passed": bool(passed),
            "sample_records": [
                records[i] for i in sorted({0, len(records) // 2, len(records) - 1})
            ],
        }
    finally:
        env.close()


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    head = run_git(["rev-parse", "HEAD"])
    status = run_git(["status", "--short"]).splitlines()
    prior_audit = json.loads((PREV_RUN / "best_row_gate_audit.json").read_text())
    write_json(
        RUN_DIR / "stage0_authority_and_worktree.json",
        {
            "head": head,
            "status_short": status,
            "prior_best_row_audit": prior_audit,
            "harness_preflight_passed": True,
        },
    )
    write_json(
        RUN_DIR / "execution_plan.json",
        {
            "task_id": "V11_G4_GOC_V3_PLACEMENT_IK_OPTIMIZATION_AUTONOMOUS_V1",
            "task_type": "PLACEMENT_IK_OPTIMIZATION_SCIENCE_PHASE",
            "no_rollout_before_joint_feasible_dynamic_probe": True,
            "no_runtime_patch": True,
            "no_current_truth_next_actions_direct_mutation": True,
        },
    )
    write_md(
        RUN_DIR / "execution_plan.md",
        "# Placement IK Optimization Plan\n\nAudit prior best row, optimize placement plus initial qpos under per-instance GOC bindings, run short dynamic probe only for joint-feasible survivors.",
    )

    candidates = candidate_grid()
    rows = []
    for idx, c in enumerate(candidates):
        rows.append(evaluate_candidate(c))
        if (idx + 1) % 60 == 0:
            write_json(
                RUN_DIR / "placement_ik_optimization_progress.json",
                {
                    "evaluated": idx + 1,
                    "total_budget": len(candidates),
                    "current_best_reset_ok": best_summary(rows),
                },
            )
    write_json(
        RUN_DIR / "placement_ik_optimization_results.json",
        {"evaluated_count": len(rows), "items": rows},
    )
    feasible = [r for r in rows if r.get("passes_joint_feasibility_gate")]
    reset_ok = [r for r in rows if r.get("reset_ok")]
    write_json(
        RUN_DIR / "joint_feasible_candidates.json",
        {"count": len(feasible), "items": feasible},
    )
    dynamic = []
    if feasible:
        for cand in sorted(
            feasible,
            key=lambda r: r["legal_pad_to_handle_ik_m"]
            + 0.35 * r["eef_to_handle_ik_m"],
        )[:6]:
            dynamic.append(run_dynamic_probe(cand))
    write_json(
        RUN_DIR / "short_dynamic_contact_probes.json",
        {"items": dynamic, "attempted": bool(dynamic)},
    )
    dyn_pass = any(d.get("passed") for d in dynamic)
    best = best_summary(rows)
    if not feasible:
        closeout = "PLACEMENT_IK_JOINT_FEASIBILITY_BLOCKER"
        next_gate = "MODEL_INSTANCE_PLACEMENT_OR_KINEMATIC_CHAIN_REPAIR"
    elif not dyn_pass:
        closeout = "PLACEMENT_IK_DYNAMIC_CONTACT_PROBE_FAILED"
        next_gate = "FINGER_PAD_DYNAMIC_ROUTE_POLICY_REPAIR"
    else:
        closeout = "DYNAMIC_CONTACT_PROBE_PASSED_BOUNDED_ROLLOUT_READY"
        next_gate = "INSTANCE_BOUND_BOUNDED_ROLLOUT_ATTEMPT"
    final = {
        "closeout_classification": closeout,
        "harness_preflight_passed": True,
        "prior_best_row_reset_ok": prior_audit["best_row"]["reset_ok"],
        "prior_best_row_reset_contact_counts": prior_audit["best_row"][
            "reset_contact_counts"
        ],
        "instance_bindings_used": True,
        "fixed_goc_ids_seed_stable": False,
        "continuous_optimization_attempted": True,
        "placement_qpos_candidates_evaluated": len(rows),
        "reset_ok_candidate_count": len(reset_ok),
        "joint_feasible_candidate_found": bool(feasible),
        "joint_feasible_candidate_count": len(feasible),
        "best_legal_pad_to_handle_m": best.get("legal_pad_to_handle_m"),
        "best_eef_to_handle_m": best.get("eef_to_handle_m"),
        "best_reset_ok": best.get("reset_ok"),
        "best_reset_contact_counts": best.get("reset_contact_counts"),
        "dynamic_contact_probe_attempted": bool(dynamic),
        "dynamic_contact_probe_passed": dyn_pass,
        "bounded_rollout_attempted": False,
        "strict_candidate_found": False,
        "exact_instance_goc_target_contact": any(
            d.get("target_contact_frame_count", 0) > 0 for d in dynamic
        ),
        "forbidden_contact_any": any(
            d.get("forbidden_contact_frame_count", 0) > 0 for d in dynamic
        ),
        "current_truth_modified": False,
        "next_actions_modified": False,
        "runtime_patch_applied": False,
        "runtime_patch_files": [],
        "next_gate": next_gate,
    }
    write_json(RUN_DIR / "placement_ik_decision.json", final)
    write_json(
        RUN_DIR / "closeout_decision.json",
        {
            **final,
            "committed": False,
            "pushed_to_origin": False,
            "remote_commit_hash": None,
        },
    )
    write_md(
        RUN_DIR / "final_report.md",
        f"""
# V11-G4 GOC-v3 Placement IK Optimization

Closeout: **{closeout}**

The prior best row was audited first. Its `reset_ok` was `{prior_audit['best_row']['reset_ok']}`, with reset counts `{prior_audit['best_row']['reset_contact_counts']}`. This phase then evaluated `{len(rows)}` continuous/local placement and initial-qpos candidates under per-instance GOC bindings.

Best observed legal-pad distance: `{final['best_legal_pad_to_handle_m']}` m. Best observed EEF distance: `{final['best_eef_to_handle_m']}` m. Joint-feasible candidates: `{len(feasible)}`.

Dynamic probe attempted: `{bool(dynamic)}`. Dynamic probe passed: `{dyn_pass}`. Bounded rollout attempted: `false`.

Runtime patch applied: `false`. Current truth modified: `false`. Next actions modified: `false`.

Next gate: **{next_gate}**
""",
    )
    write_json(
        CAMPAIGN
        / "sovereign/proposed_current_truth_delta_placement_ik_optimization.json",
        {
            "proposal_id": "proposed_current_truth_delta_placement_ik_optimization",
            "source_run_dir": str(RUN_DIR),
            "facts": final,
            "direct_mutation": False,
        },
    )
    write_json(
        CAMPAIGN / "sovereign/proposed_next_actions_placement_ik_optimization.json",
        {
            "proposal_id": "proposed_next_actions_placement_ik_optimization",
            "source_run_dir": str(RUN_DIR),
            "next_gate": next_gate,
            "recommended_actions": [
                "Do not run Phase1H bounded rollout until placement plus initial qpos yields joint-feasible reset-clean contact geometry and short dynamic target contact.",
                "If joint feasibility remains blocked, repair robot placement/model assembly or kinematic chain rather than route microvariants.",
            ],
            "direct_mutation": False,
        },
    )


def best_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if "legal_pad_to_handle_ik_m" in r]
    if not valid:
        reset_snap = [r for r in rows if "reset_distance_snapshot" in r]
        if not reset_snap:
            return {}
        r = min(
            reset_snap,
            key=lambda x: x["reset_distance_snapshot"]["min_legal_pad_to_handle_m"],
        )
        return {
            "seed": r.get("seed"),
            "base_pos": r.get("base_pos"),
            "yaw_deg": r.get("yaw_deg"),
            "reset_ok": r.get("reset_ok"),
            "reset_contact_counts": r.get("reset_contact_counts"),
            "legal_pad_to_handle_m": r["reset_distance_snapshot"][
                "min_legal_pad_to_handle_m"
            ],
            "eef_to_handle_m": r["reset_distance_snapshot"]["min_eef_to_handle_m"],
        }
    r = min(
        valid,
        key=lambda x: x["legal_pad_to_handle_ik_m"]
        + 0.35 * x["eef_to_handle_ik_m"]
        + (0.0 if x.get("reset_ok") else 99.0),
    )
    return {
        "seed": r.get("seed"),
        "base_pos": r.get("base_pos"),
        "yaw_deg": r.get("yaw_deg"),
        "qpos_index": r.get("qpos_index"),
        "reset_ok": r.get("reset_ok"),
        "reset_contact_counts": r.get("reset_contact_counts"),
        "legal_pad_to_handle_m": r.get("legal_pad_to_handle_ik_m"),
        "eef_to_handle_m": r.get("eef_to_handle_ik_m"),
        "passes_joint_feasibility_gate": r.get("passes_joint_feasibility_gate"),
    }


if __name__ == "__main__":
    main()
