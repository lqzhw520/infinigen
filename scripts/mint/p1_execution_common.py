#!/usr/bin/env python3
"""Shared helpers for the Infinigen -> MINT exact-alignment execution plan."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import traceback
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import yaml

from mint_common import now_iso, write_json_atomic, write_text_atomic

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN_DIR = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1"
ARTIFACT_DIR = CAMPAIGN_DIR / "artifacts"
OUTPUT_DIR = CAMPAIGN_DIR / "outputs"
SOVEREIGN_DIR = CAMPAIGN_DIR / "sovereign"
EVIDENCE_DIR = SOVEREIGN_DIR / "evidence"
CURRENT_TRUTH_PATH = SOVEREIGN_DIR / "current_truth.json"
NEXT_ACTIONS_PATH = SOVEREIGN_DIR / "next_actions.json"
EVIDENCE_INDEX_PATH = EVIDENCE_DIR / "index.json"

MINT_REPO_ROOT = PROJECT_ROOT / "external" / "MINT"
MINT_POLICY_SRC = MINT_REPO_ROOT / "lerobot_policy_mint" / "src"
MINT_CKPT = MINT_REPO_ROOT / "checkpoints" / "MINT-libero"
MINT_TOKENIZER = MINT_REPO_ROOT / "checkpoints" / "MINT-tokenizer-libero"

CONTROL_BASELINE_REF = "E036/p1c11_official_libero_goal_drawer_baseline_rollback_4eab579"
CONTROL_TASK_SUITE = "libero_goal"
CONTROL_TASK_ID = 0
CONTROL_EPISODES = 3
CONTROL_SEED = 42
CONTROL_IMAGE_SIZE = 256
CONTROL_N_ACTION_STEPS = 4

P1I_ARTIFACT = ARTIFACT_DIR / "p1i_infinigen_mint_alignment_solution_plan.json"
P1I_REPORT = OUTPUT_DIR / "p1i_infinigen_mint_alignment_solution_plan.md"
P1I_EVIDENCE_ID = "E051"

CANONICAL_READABLE_REPORTS = [
    "experiments/mint/mint_drawer_v1/outputs/p1f_mainline_alignment_cleanup_report.md",
    "experiments/mint/mint_drawer_v1/outputs/p1g_exact_control_alignment_audit.md",
    "experiments/mint/mint_drawer_v1/outputs/p1h_working_set_manifest.md",
    "experiments/mint/mint_drawer_v1/outputs/p1i_infinigen_mint_alignment_solution_plan.md",
]

CANONICAL_REVIEW_SURFACES = [
    {
        "evidence_id": "E045",
        "path": "experiments/mint/mint_drawer_v1/artifacts/p1f_mainline_alignment_cleanup_report.json",
        "role": "run_summary",
    },
    {
        "evidence_id": "E046",
        "path": "experiments/mint/mint_drawer_v1/artifacts/p1g_exact_control_alignment_audit.json",
        "role": "exact_control_alignment_audit",
    },
    {
        "evidence_id": "E047",
        "path": "experiments/mint/mint_drawer_v1/artifacts/p1h_working_set_manifest.json",
        "role": "working_set_manifest",
    },
    {
        "evidence_id": "E051",
        "path": "experiments/mint/mint_drawer_v1/artifacts/p1i_infinigen_mint_alignment_solution_plan.json",
        "role": "root_cause_execution_plan",
    },
]

H0_PENDING_NEXT_ACTION = {
    "type": "ROTATION_ACTION_FIX",
    "id": "implement_control_calibrated_rotation_in_script_action",
    "priority": "P0",
    "target": "Implement control-calibrated delta_rot[3:6] in DrawerRobotEnvMuJoCo._script_action() and verify oracle rollouts break the current ~10-step / 100-frame ceiling.",
    "status": "pending",
    "description": "E055 shows control rotation is non-degenerate while MuJoCo rollout rotation is all-zero. E056 confirms a real visual gap, and E057 shows dataset scale remains capped until action and render semantics improve.",
    "control_baseline_ref": CONTROL_BASELINE_REF,
    "state_action_gap_ref": "E055/p1m_state_action_gap_audit",
    "visual_gap_ref": "E056/p1n_visual_semantics_canonical",
    "dataset_ceiling_ref": "E057/p1o_dataset_expansion_manifest",
}

FINAL_EXECUTION_PLAN = {
    "plan_id": "p1i_infinigen_mint_alignment_solution_plan",
    "control_baseline_ref": CONTROL_BASELINE_REF,
    "supporting_evidence": ["E045", "E046", "E047", "E052", "E053", "E054", "E055", "E056", "E057"],
    "scope": "Post-gate canonical execution plan after P0A-P1B. Focus on the remaining blockers before any retrain.",
    "summary": [
        "P0A-P1B execution established that eval parity is not the primary blocker, but rotation, visual semantics, and dataset ceiling remain real blockers.",
        "The next cycle should not retrain yet. It should first fix MuJoCo rotation in the oracle/control writer, add a control-like state candidate to the measured state audit, and improve canonical render-level visual semantics.",
        "Only after those three fixes improve rollout length, successful unique-seed coverage, and total frames should dataset expansion and retraining resume.",
    ],
    "phases": [
        {
            "id": "R1",
            "title": "Rotation action contract fix",
            "priority": "P0",
            "goal": "Implement control-calibrated delta_rot[3:6] in DrawerRobotEnvMuJoCo._script_action() and verify oracle rollouts can exceed the current ~10-step ceiling.",
            "deliverables": [
                "rotation-capable _script_action()",
                "updated oracle rollout statistics",
                "evidence that max_drawer_fraction and episode length improve beyond the current ceiling",
            ],
            "gate": "MuJoCo rollout rotation is no longer degenerate and oracle/teacher rollouts materially improve in drawer progress and length.",
        },
        {
            "id": "R2",
            "title": "Control-like state candidate and re-audit",
            "priority": "P0",
            "goal": "Extend the measured state audit with a control-like state candidate aligned to live control telemetry instead of treating proxy-vs-qpos as the complete candidate set.",
            "deliverables": [
                "new control-like state candidate",
                "rerun p1m-style audit with the expanded candidate set",
                "updated decision on whether M0 truly remains the closest mapping",
            ],
            "gate": "State mapping retention or replacement is justified against a candidate set that includes a control-like telemetry-aligned option.",
        },
        {
            "id": "R3",
            "title": "Canonical visual render fix",
            "priority": "P1",
            "goal": "Improve render-level camera/material/background/lighting semantics and rerun p1n without relying on diagnostic-only post-render texture injection.",
            "deliverables": [
                "canonical visual changes in the MuJoCo/Infinigen render path",
                "rerun visual audit with updated edge_density / entropy / std_rgb metrics",
            ],
            "gate": "Canonical visual metrics move materially toward control and brightness is no longer the only improved indicator.",
        },
        {
            "id": "R4",
            "title": "Dataset expansion retry after repaired data line",
            "priority": "P1-after-fixes",
            "goal": "Retry dataset expansion only after R1-R3 lift the current 100-frame ceiling.",
            "deliverables": [
                "rerun dataset expansion manifest",
                "updated successful unique seed count and frame count",
            ],
            "gate": "Reach >=15 successful unique train seeds and ~5000+ frames, or explicitly fail the gate if the repaired data line still cannot reach that scale.",
        },
        {
            "id": "R5",
            "title": "Retrain only after repaired data line",
            "priority": "P2",
            "goal": "Rerun pretrained/prior-finetune/new-finetune comparison only after rotation, state contract, visual semantics, and dataset scale gates pass.",
            "deliverables": [
                "retrain eval matrix",
                "delta vs pretrained and prior finetune under the repaired data line",
            ],
            "gate": "Any new training claim is interpreted only after R1-R4 pass.",
        },
    ],
    "assumptions_and_defaults": [
        "Upstream external/MINT@4eab579 remains the canonical control benchmark.",
        "p1j-p1o are execution evidence and gate artifacts, not canonical readable reports.",
        "Control telemetry is the final truth; comments, STATE_NAMES, and old markdown are not.",
        "Post-render texture injection stays diagnostic-only unless later encoder evidence explicitly justifies it.",
        "The current M0 state retention result is provisional because the candidate set did not yet include a control-like state mapping.",
    ],
}



def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return deepcopy(default)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return deepcopy(default)


def stable_config_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def ensure_mint_policy_importable() -> None:
    src = str(MINT_POLICY_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    import lerobot_policy_mint  # noqa: F401


def upsert_evidence(evidence_id: str, experiment_id: str, type_name: str, path: Path, summary: str, *, verified: bool = True) -> dict[str, Any]:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    rel_path = str(path.relative_to(PROJECT_ROOT))
    entry = {
        "evidence_id": evidence_id,
        "experiment_id": experiment_id,
        "type": type_name,
        "timestamp": now_iso(),
        "path": rel_path,
        "verified": bool(verified),
        "summary": summary,
    }
    yaml_path = EVIDENCE_DIR / f"{evidence_id}.yaml"
    write_text_atomic(yaml_path, yaml.safe_dump(entry, sort_keys=False, allow_unicode=True))

    index = load_json(EVIDENCE_INDEX_PATH, {"version": 1, "last_updated": now_iso(), "entries": []})
    entries = [item for item in index.get("entries", []) if item.get("evidence_id") != evidence_id]
    entries.append(entry)
    index["entries"] = entries
    index["last_updated"] = now_iso()
    write_json_atomic(EVIDENCE_INDEX_PATH, index)
    return entry


def dedupe_authority_notes(notes: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for note in notes:
        text = str(note).strip()
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return out


def dedupe_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed: OrderedDict[tuple[Any, Any], dict[str, Any]] = OrderedDict()
    for action in actions:
        key = (action.get("type"), action.get("id"))
        existing = keyed.get(key)
        if existing is None:
            keyed[key] = deepcopy(action)
            continue
        existing_ts = existing.get("updated_at") or ""
        new_ts = action.get("updated_at") or ""
        if new_ts >= existing_ts:
            keyed[key] = deepcopy(action)
    return list(keyed.values())


def normalize_sovereign_h0() -> dict[str, Any]:
    current_truth = load_json(CURRENT_TRUTH_PATH, {"current": {}})
    next_actions = load_json(NEXT_ACTIONS_PATH, {"actions": []})

    current = current_truth.setdefault("current", {})
    current["canonical_readable_reports"] = list(CANONICAL_READABLE_REPORTS)
    current["review_surfaces"] = deepcopy(CANONICAL_REVIEW_SURFACES)
    current["next_action"] = {**deepcopy(H0_PENDING_NEXT_ACTION), "updated_at": now_iso()}

    authority_notes = list(current.get("authority_notes", []))
    authority_notes.append("Use E045/E046/E047/E051 as the canonical review surface for the completed MuJoCo mainline run and the current execution plan.")
    current["authority_notes"] = dedupe_authority_notes(authority_notes)
    current["generated_at"] = now_iso()
    current_truth["current"] = current
    write_json_atomic(CURRENT_TRUTH_PATH, current_truth)

    cleaned_actions = dedupe_actions(next_actions.get("actions", []))
    cleaned_actions = [
        action
        for action in cleaned_actions
        if not (
            action.get("type") == H0_PENDING_NEXT_ACTION["type"] and action.get("id") == H0_PENDING_NEXT_ACTION["id"]
        )
    ]
    next_actions["actions"] = [{**deepcopy(H0_PENDING_NEXT_ACTION), "updated_at": now_iso()}] + cleaned_actions
    next_actions["generated_at"] = now_iso()
    write_json_atomic(NEXT_ACTIONS_PATH, next_actions)

    evidence_entry = upsert_evidence(
        P1I_EVIDENCE_ID,
        "p1i_infinigen_mint_alignment_solution_plan",
        "root_cause_execution_plan",
        P1I_ARTIFACT,
        "Canonical root-cause execution plan for the Infinigen+AnyGrasp+MuJoCo to MINT alignment line. Prioritizes exact control traces, eval parity, measured state/action alignment, visual semantics, and dataset expansion before retraining.",
        verified=True,
    )
    return {
        "current_truth_path": str(CURRENT_TRUTH_PATH),
        "next_actions_path": str(NEXT_ACTIONS_PATH),
        "evidence": evidence_entry,
        "review_surfaces": current["review_surfaces"],
        "pending_next_action": current["next_action"],
    }


def write_final_execution_plan_surfaces() -> dict[str, Any]:
    payload = deepcopy(FINAL_EXECUTION_PLAN)
    payload["generated_at"] = now_iso()

    lines = [
        "# Infinigen -> MINT Alignment Root-Cause Execution Plan (Final Execution Version)",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Summary",
    ]
    for item in payload["summary"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Execution Order"])
    for phase in payload["phases"]:
        lines.append(f"- {phase['id']}: {phase['title']} ({phase['priority']})")
    lines.extend(["", "## Assumptions And Defaults"])
    for item in payload["assumptions_and_defaults"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Phases"])
    for phase in payload["phases"]:
        lines.extend([
            f"### {phase['id']}. {phase['title']}",
            f"Goal: {phase['goal']}",
            "",
            "Deliverables:",
        ])
        for deliverable in phase["deliverables"]:
            lines.append(f"- {deliverable}")
        lines.extend(["", f"Gate: {phase['gate']}", ""])

    write_json_atomic(P1I_ARTIFACT, payload)
    write_text_atomic(P1I_REPORT, "\n".join(lines).rstrip() + "\n")
    return payload


def atomic_finalize_output_dir(staging_dir: Path, final_dir: Path) -> None:
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(staging_dir, final_dir)


def video_integrity_report(video_paths: list[Path]) -> dict[str, Any]:
    records = []
    ok = True
    for path in video_paths:
        exists = path.exists()
        size_bytes = path.stat().st_size if exists else 0
        readable = False
        frame_count = 0
        if exists and size_bytes > 0:
            cap = cv2.VideoCapture(str(path))
            try:
                readable = bool(cap.isOpened())
                if readable:
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            finally:
                cap.release()
        record = {
            "path": str(path),
            "exists": exists,
            "size_bytes": int(size_bytes),
            "readable": readable,
            "frame_count": frame_count,
        }
        ok = ok and exists and size_bytes > 0 and readable and frame_count > 0
        records.append(record)
    return {"passed": ok, "video_count": len(video_paths), "records": records}


def bundle_config_hash(cfg: Any) -> str:
    try:
        payload = json.loads(json.dumps(cfg, default=lambda o: getattr(o, '__dict__', str(o)), sort_keys=True))
    except TypeError:
        payload = str(cfg)
    return stable_config_hash(payload)


def build_official_eval_cfg(*, n_action_steps: int, n_episodes: int, device: str, seed: int, output_dir: Path):
    ensure_mint_policy_importable()
    from lerobot.configs import parser
    from lerobot.configs.eval import EvalPipelineConfig

    def _build(cfg):
        return cfg

    _build.__annotations__ = {"cfg": EvalPipelineConfig}
    wrapped = parser.wrap()(_build)

    argv = [
        "prog",
        f"--policy.path={MINT_CKPT}",
        f"--policy.vqvae_name_or_path={MINT_TOKENIZER}",
        "--env.type=libero",
        f"--env.task={CONTROL_TASK_SUITE}",
        f"--env.observation_height={CONTROL_IMAGE_SIZE}",
        f"--env.observation_width={CONTROL_IMAGE_SIZE}",
        f"--policy.n_action_steps={n_action_steps}",
        f"--policy.device={device}",
        "--eval.batch_size=1",
        f"--eval.n_episodes={n_episodes}",
        f"--seed={seed}",
        f"--output_dir={output_dir}",
    ]
    old_argv = sys.argv[:]
    sys.argv = argv
    try:
        return wrapped()
    finally:
        sys.argv = old_argv


def make_official_control_bundle(*, n_action_steps: int, n_episodes: int, device: str, seed: int, output_dir: Path):
    ensure_mint_policy_importable()
    import gymnasium as gym
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.libero import create_libero_envs
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    cfg = build_official_eval_cfg(
        n_action_steps=n_action_steps,
        n_episodes=n_episodes,
        device=device,
        seed=seed,
        output_dir=output_dir,
    )
    envs = create_libero_envs(
        task=cfg.env.task,
        n_envs=cfg.eval.batch_size,
        camera_name=cfg.env.camera_name,
        init_states=cfg.env.init_states,
        gym_kwargs={
            **cfg.env.gym_kwargs,
            "task_ids": [CONTROL_TASK_ID],
            "observation_height": cfg.env.observation_height,
            "observation_width": cfg.env.observation_width,
        },
        env_cls=gym.vector.AsyncVectorEnv if cfg.eval.use_async_envs else gym.vector.SyncVectorEnv,
        control_mode=cfg.env.control_mode,
        episode_length=cfg.env.episode_length,
    )
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
    return {
        "cfg": cfg,
        "envs": envs,
        "policy": policy,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "env_preprocessor": env_preprocessor,
        "env_postprocessor": env_postprocessor,
    }


def close_bundle(bundle: dict[str, Any]) -> None:
    envs = bundle.get("envs", {})
    for task_map in envs.values():
        for env in task_map.values():
            try:
                env.close()
            except Exception:
                pass


def get_single_vec_env(bundle: dict[str, Any]):
    envs = bundle["envs"]
    return envs[CONTROL_TASK_SUITE][CONTROL_TASK_ID]


def infer_libero_drawer_fraction(vec_env) -> dict[str, Any]:
    env = vec_env.envs[0].unwrapped
    sim = env._env.sim
    preferred = [
        "wooden_cabinet_1_middle_level",
        "wooden_cabinet_1_top_level",
        "wooden_cabinet_1_bottom_level",
    ]
    joint_name = None
    for name in preferred:
        try:
            sim.model.joint_name2id(name)
            joint_name = name
            break
        except Exception:
            continue
    if joint_name is None:
        for idx in range(sim.model.njnt):
            name = sim.model.joint_id2name(idx)
            if "drawer" in name or "cabinet" in name or "level" in name:
                joint_name = name
                break
    if joint_name is None:
        return {"joint_name": None, "fraction": None, "qpos": None, "range": None}
    joint_id = sim.model.joint_name2id(joint_name)
    qpos_addr = int(sim.model.jnt_qposadr[joint_id])
    low, high = [float(x) for x in sim.model.jnt_range[joint_id]]
    qpos = float(sim.data.qpos[qpos_addr])
    closed = high
    opened = low
    span = max(abs(closed - opened), 1e-6)
    fraction = max(0.0, min(1.0, (closed - qpos) / span))
    return {
        "joint_name": joint_name,
        "joint_id": int(joint_id),
        "qpos_addr": qpos_addr,
        "low": low,
        "high": high,
        "qpos": qpos,
        "fraction": fraction,
    }


def failure_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "passed": False,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "timestamp": now_iso(),
    }
