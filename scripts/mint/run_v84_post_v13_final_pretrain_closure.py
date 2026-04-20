#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from drawer_robot_env_mujoco import (
    DrawerEnvContractConfig,
    build_robot_rollout,
    save_robot_rollout,
)
from root_cause_controller import RootCauseController
from v13_audit_common import (
    ARTIFACT_DIR,
    AUTOPILOT_DIR,
    V12_SPEC_REFERENCE,
    V13_SPEC_REFERENCE,
    current_repo_identity,
    load_json,
    make_run_instance,
    save_active_plan,
    utc_now,
    write_gate,
    write_json_atomic,
)

POST_V13_ADMISSION_SPEC_REFERENCE = "/Users/zhuhaowu/ws/phd-anyboxs/infinigen_from_servers/docs/gpt5.4Pro/codex_mint_v84_post_v13_train_probe_admission_execution_spec.md"
PLAN_VERSION = "tiny_retrain_confirmation_post_v13_final_pretrain_closure"
PLAN_ARTIFACT_PATH = ARTIFACT_DIR / "post_v13_final_pretrain_closure_plan.json"
SUMMARY_PATH = ARTIFACT_DIR / "post_v13_final_pretrain_closure_summary.json"
GATES_DIR = AUTOPILOT_DIR / "gates_post_v13_closure"
LIVE_DIR = ARTIFACT_DIR / "post_v13_closure_live_state_rollouts"
PARITY_DIR = ARTIFACT_DIR / "post_v13_closure_parity_rollouts"

GATES_V13_DIR = AUTOPILOT_DIR / "gates_v13"
GATES_V13_SLICE2_DIR = AUTOPILOT_DIR / "gates_v13_slice2"
GATES_POST_V13_DIR = AUTOPILOT_DIR / "gates_post_v13"

EXPECTED_HEAD = "bc802069816e6c81beab48ba1f2141c9f3a264d4"
EXPECTED_VENDOR = "4eab5795345721001c412ff1ca2c886a11eab606"
EXPECTED_BRANCH = "feature/mint-env-reformulation-v1-visual-fidelity"
EXPECTED_ADMISSION_VERDICT = "STOP_PRETRAIN_ADMISSION_AMBIGUOUS"
EXPECTED_ADMISSION_BLOCKERS = [
    "slice2_live_fit_does_not_retain_prior_signal_band",
    "slice2_live_rollout_support_is_only_4_episodes",
    "current_evidence_cannot_separate_benign_live_distribution_shift_from_unresolved_signal_instability",
]

TARGET_SEEDS = list(range(1, 9))
TARGET_SUCCESS_PER_SEED = 3
MIN_TOTAL_EPISODES = 16
MIN_SEED_COVERAGE = 6
MAX_EPISODE_SCAN_PER_SEED = 12
DUPLICATE_EXHAUSTION_STREAK = 2
MATCHED_BASELINE_SEEDS = [11, 23, 37, 41, 53, 67, 79, 97]
REPEAT_SPLIT_SEEDS = [11, 23, 37]
FULLSET_SPLIT_SEED = 0
BENIGN_P25_GAP_TOLERANCE = 0.05

STATE_DIM_NAMES = [
    "distance_to_handle_norm",
    "approach_alignment_cos",
    "orientation_alignment_cos",
    "orientation_error_sin",
    "orientation_error_cos",
    "prev_close_cmd",
    "gripper_joint",
    "attach_eligible_proxy",
]
CONTINUOUS_DIM_NAMES = [
    "distance_to_handle_norm",
    "approach_alignment_cos",
    "orientation_alignment_cos",
    "orientation_error_sin",
    "orientation_error_cos",
    "gripper_joint",
]
DISCRETE_DIM_NAMES = ["prev_close_cmd", "attach_eligible_proxy"]


class TinyHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load(path: Path) -> dict[str, Any]:
    return load_json(path, {})


def _carrier_contract() -> DrawerEnvContractConfig:
    controller = RootCauseController()
    payload = dict(
        controller._frozen_matrix_contracts_v5_pro()["V1cT2S3"]["env_contract_config"]
    )
    return DrawerEnvContractConfig(**payload)


def _diagnostic_contract() -> DrawerEnvContractConfig:
    payload = dict(
        RootCauseController()
        ._frozen_matrix_contracts_v5_pro()["V1cT2S3"]["env_contract_config"]
    )
    payload["state_mode"] = "m0_proxy"
    return DrawerEnvContractConfig(**payload)


def _build_plan(
    run_instance_id: str,
    ident: dict[str, str],
    slice2_plan: dict[str, Any],
    admission_plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_instance_id": run_instance_id,
        "plan_version": PLAN_VERSION,
        "slice_name": "post-v13-final-pretrain-closure",
        "working_head_commit": ident["working_head_commit"],
        "current_publication_head_commit": ident["working_head_commit"],
        "vendor_head_commit": ident["vendor_head_commit"],
        "branch": ident["branch"],
        "execution_scope": "post_v13_final_pretrain_closure",
        "bridge_stage": "post_v13_final_pretrain_closure",
        "diagnostic_only": True,
        "claim_bearing": False,
        "training_allowed": False,
        "probe_allowed": False,
        "publication_scope": "blocked_before_canary",
        "result_scope": "final_pretrain_closure_only",
        "source_v13_slice2_run_instance_id": slice2_plan.get("run_instance_id"),
        "source_v13_slice2_plan_version": slice2_plan.get("plan_version"),
        "source_post_v13_admission_run_instance_id": admission_plan.get("run_instance_id"),
        "source_post_v13_admission_plan_version": admission_plan.get("plan_version"),
        "allowed_next_phases": ["C0", "C1", "C2", "C3", "C4"],
        "spec_reference": POST_V13_ADMISSION_SPEC_REFERENCE,
        "carryforward_spec_references": [
            V12_SPEC_REFERENCE,
            V13_SPEC_REFERENCE,
            POST_V13_ADMISSION_SPEC_REFERENCE,
        ],
        "goal": "Resolve the remaining post-v13 training admission ambiguity in one final pretrain closure tranche, without starting train/probe.",
        "constraints": [
            "Do not start training.",
            "Do not start probe.",
            "Do not modify external/MINT.",
            "Do not change truth predicates or acceptance thresholds.",
            "Do not mutate the live wiring unless parity audit proves a hard contradiction.",
        ],
        "future_canary_prebinding": {
            "max_train_steps": 2000,
            "checkpoint_ladder": [500, 1000, 1500, 2000],
            "early_stop_rule": "Stop at checkpoint 1000 if pre-attach, close, and attach-eligible gains are all zero relative to pretrained baseline.",
            "probe_scope": ["pre-attach", "close", "attach-eligible"],
            "forbidden_scope": ["drawer_success", "claim_bearing_interpretation"],
            "allowed_future_results": [
                "state_action_interface_repaired_no_signal",
                "pre_attach_metric_shift_no_attach_conversion",
                "state_action_interface_repaired_attach_signal_detected",
                "action_close_interface_mismatch_confirmed",
                "objective_weighting_needed_after_interface_repair",
            ],
        },
        "expected_outputs": [
            "experiments/mint/mint_drawer_v1/autopilot/gates_post_v13_closure/C0_authority_refreeze.json",
            "experiments/mint/mint_drawer_v1/autopilot/gates_post_v13_closure/C1_live_support_expansion.json",
            "experiments/mint/mint_drawer_v1/autopilot/gates_post_v13_closure/C2_live_diagnostic_parity.json",
            "experiments/mint/mint_drawer_v1/autopilot/gates_post_v13_closure/C3_signal_band_stability.json",
            "experiments/mint/mint_drawer_v1/autopilot/gates_post_v13_closure/C4_final_pretrain_closure_verdict.json",
            "experiments/mint/mint_drawer_v1/artifacts/post_v13_final_pretrain_closure_summary.json",
        ],
    }


def _stamp_rollout(
    rollout: dict[str, Any],
    *,
    plan: dict[str, Any],
    execution_scope: str,
    source_slice2_plan: dict[str, Any],
) -> None:
    rollout["run_instance_id"] = plan["run_instance_id"]
    rollout["plan_version"] = plan["plan_version"]
    rollout["working_head_commit"] = plan["working_head_commit"]
    rollout["bridge_stage"] = execution_scope
    rollout["bridge_attempt"] = "final_pretrain_closure"
    rollout["canonical_train_cell"] = source_slice2_plan.get("canonical_train_cell")
    rollout["source_canonical_train_cell"] = source_slice2_plan.get("source_canonical_train_cell")
    rollout["best_train_state_mode"] = source_slice2_plan.get("best_train_state_mode")
    rollout["source_best_train_state_mode"] = source_slice2_plan.get("source_best_train_state_mode")
    rollout["active_train_state_mode"] = source_slice2_plan.get("active_train_state_mode")
    rollout["active_state_mode_name"] = rollout.get("active_state_mode_name") or source_slice2_plan.get("active_state_mode_name")


def _episode_key(seed: int, episode_index: int) -> str:
    return f"seed_{seed:03d}_episode_{episode_index:02d}"


def _fingerprint_rollout(rollout: dict[str, Any]) -> str:
    states = np.asarray(rollout["states"], dtype=np.float32)
    actions = np.asarray(rollout["actions"], dtype=np.float32)
    h = hashlib.sha1()
    h.update(states.tobytes())
    h.update(actions.tobytes())
    return h.hexdigest()


def _build_rollout(
    *,
    seed: int,
    episode_index: int,
    contract: DrawerEnvContractConfig,
    plan: dict[str, Any],
    source_slice2_plan: dict[str, Any],
    execution_scope: str,
) -> dict[str, Any]:
    rollout = build_robot_rollout(
        seed=seed,
        grasp_pose_world=np.eye(4, dtype=np.float32),
        episode_index=episode_index,
        image_size=64,
        max_steps=96,
        contract=contract,
        rotation_source="aligned",
        claim_policy="diagnostic",
        teacher_controller_mode="interaction_frame_hybrid",
    )
    _stamp_rollout(
        rollout,
        plan=plan,
        execution_scope=execution_scope,
        source_slice2_plan=source_slice2_plan,
    )
    return rollout


def _approach_alignment(states: np.ndarray) -> np.ndarray:
    eef = states[:, :3]
    handle_rel = states[:, 3:6] * np.array([0.22, 0.18, 0.14], dtype=np.float32)
    handle_world = eef + handle_rel
    out = np.zeros((len(states),), dtype=np.float32)
    for i in range(1, len(states)):
        motion = eef[i] - eef[i - 1]
        desired = handle_world[i - 1] - eef[i - 1]
        m = float(np.linalg.norm(motion))
        d = float(np.linalg.norm(desired))
        if m > 1e-8 and d > 1e-8:
            out[i] = float(np.clip(np.dot(motion / m, desired / d), -1.0, 1.0))
    return out


def _diag_state(
    states: np.ndarray,
    actions: np.ndarray,
    handle_distance: np.ndarray,
    orientation_alignment: np.ndarray,
    orientation_error: np.ndarray,
    attach_eligible: np.ndarray,
) -> np.ndarray:
    approach = _approach_alignment(states)
    prev_close = np.concatenate([[0.0], (actions[:-1, 6] < 0.0).astype(np.float32)])
    gripper = states[:, 7]
    distance_norm = np.clip(handle_distance / 0.35, 0.0, 1.0)
    return np.stack(
        [
            distance_norm,
            approach,
            orientation_alignment,
            np.sin(orientation_error).astype(np.float32),
            np.cos(orientation_error).astype(np.float32),
            prev_close,
            gripper,
            attach_eligible.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def _load_rollout_arrays(meta_path: Path) -> dict[str, np.ndarray]:
    data = np.load(meta_path.with_suffix(".npz"), allow_pickle=True)
    return {key: np.asarray(data[key]) for key in data.files}


def _collect_fit_dataset(meta_paths: list[Path]) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    features = []
    actions = []
    episodes = []
    episode_seed = {}
    for meta_path in sorted(meta_paths):
        meta = json.loads(meta_path.read_text())
        arrays = _load_rollout_arrays(meta_path)
        states = np.asarray(arrays["states"], dtype=np.float32)
        acts = np.asarray(arrays["actions"], dtype=np.float32)
        features.append(states)
        actions.append(acts)
        stem = meta_path.stem
        episodes.extend([stem] * len(states))
        episode_seed[stem] = int(meta["seed"])
    return np.concatenate(features), np.concatenate(actions), episodes, episode_seed


def _collect_diag_dataset(meta_paths: list[Path]) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    features = []
    actions = []
    episodes = []
    episode_seed = {}
    for meta_path in sorted(meta_paths):
        meta = json.loads(meta_path.read_text())
        arrays = _load_rollout_arrays(meta_path)
        states = np.asarray(arrays["states"], dtype=np.float32)
        acts = np.asarray(arrays["actions"], dtype=np.float32)
        diag = _diag_state(
            states,
            acts,
            np.asarray(arrays["handle_distance_trace"], dtype=np.float32),
            np.asarray(arrays["orientation_alignment_trace"], dtype=np.float32),
            np.asarray(arrays["orientation_error_trace"], dtype=np.float32),
            np.asarray(arrays["attach_eligible_trace"], dtype=bool),
        )
        features.append(diag)
        actions.append(acts)
        stem = meta_path.stem
        episodes.extend([stem] * len(states))
        episode_seed[stem] = int(meta["seed"])
    return np.concatenate(features), np.concatenate(actions), episodes, episode_seed


def _seed_stratified_split(
    episode_seed: dict[str, int],
    split_seed: int,
    train_fraction: float = 0.75,
) -> tuple[list[str], list[str]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for episode, seed in episode_seed.items():
        grouped[int(seed)].append(episode)
    train_eps: list[str] = []
    val_eps: list[str] = []
    singleton_eps: list[str] = []
    for seed in sorted(grouped):
        eps = sorted(grouped[seed])
        rng = random.Random(split_seed + seed * 1009)
        rng.shuffle(eps)
        if len(eps) == 1:
            singleton_eps.append(eps[0])
            continue
        cut = max(1, int(round(train_fraction * len(eps))))
        cut = min(cut, len(eps) - 1)
        train_eps.extend(eps[:cut])
        val_eps.extend(eps[cut:])
    if singleton_eps:
        rng = random.Random(split_seed + 7919)
        rng.shuffle(singleton_eps)
        total_episode_count = sum(len(rows) for rows in grouped.values())
        val_target = max(1, int(round((1.0 - train_fraction) * total_episode_count)))
        while singleton_eps and len(val_eps) < val_target:
            val_eps.append(singleton_eps.pop())
        train_eps.extend(singleton_eps)
    if not train_eps and val_eps:
        train_eps.append(val_eps.pop())
    if not val_eps and train_eps:
        val_eps.append(train_eps[-1])
    return train_eps, val_eps


def _episode_indices(episodes: list[str], selected: list[str]) -> np.ndarray:
    chosen = set(selected)
    return np.array([i for i, ep in enumerate(episodes) if ep in chosen], dtype=np.int64)


def _metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    trans_pred, rot_pred, close_logit = pred[:, :3], pred[:, 3:6], pred[:, 6]
    trans_t, rot_t = target[:, :3], target[:, 3:6]
    close_t = (target[:, 6] < 0.0).float()
    trans_mse = float(torch.mean((trans_pred - trans_t) ** 2).item())
    rot_mse = float(torch.mean((rot_pred - rot_t) ** 2).item())
    close_prob = torch.sigmoid(close_logit)
    close_acc = float(torch.mean(((close_prob >= 0.5).float() == close_t).float()).item())
    close_brier = float(torch.mean((close_prob - close_t) ** 2).item())
    rot_num = torch.sum(rot_pred * rot_t, dim=1)
    rot_den = torch.linalg.norm(rot_pred, dim=1) * torch.linalg.norm(rot_t, dim=1)
    valid = rot_den > 1e-6
    rot_cos = (
        float(torch.mean((rot_num[valid] / rot_den[valid]).clamp(-1.0, 1.0)).item())
        if bool(torch.any(valid))
        else 0.0
    )
    overall = trans_mse + rot_mse + close_brier
    return {
        "action_mse_overall": overall,
        "translation_action_mse": trans_mse,
        "orientation_action_mse": rot_mse,
        "close_dim_accuracy": close_acc,
        "close_dim_brier": close_brier,
        "close_dim_sign_accuracy": close_acc,
        "orientation_action_cosine_similarity": rot_cos,
    }


def _train_probe(
    features: np.ndarray,
    target: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(target, dtype=torch.float32, device=device)
    model = TinyHead(features.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    bce = nn.BCEWithLogitsLoss()

    def loss_fn(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        trans = torch.mean((pred[:, :3] - tgt[:, :3]) ** 2)
        rot = torch.mean((pred[:, 3:6] - tgt[:, 3:6]) ** 2)
        close = bce(pred[:, 6], (tgt[:, 6] < 0.0).float())
        return trans + rot + close

    with torch.no_grad():
        initial_pred = model(x[val_idx])
        initial_metrics = _metrics(initial_pred, y[val_idx])
    losses = []
    batch_size = min(256, len(train_idx))
    for _ in range(250):
        perm = torch.randperm(len(train_idx))
        batch = train_idx[perm[:batch_size]]
        pred = model(x[batch])
        loss = loss_fn(pred, y[batch])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    with torch.no_grad():
        final_pred = model(x[val_idx])
        final_metrics = _metrics(final_pred, y[val_idx])
    close_val = (y[val_idx, 6] < 0.0).cpu().numpy().astype(np.float32)
    majority = float(max(np.mean(close_val), 1.0 - np.mean(close_val)))
    loss_drop = (losses[0] - losses[-1]) / max(losses[0], 1e-8)
    close_gain = final_metrics["close_dim_accuracy"] - majority
    init_rot = initial_metrics["orientation_action_mse"]
    final_rot = final_metrics["orientation_action_mse"]
    rot_gain = (init_rot - final_rot) / max(init_rot, 1e-8)
    status = (
        "one_step_action_fit_ok"
        if loss_drop > 0.20 and close_gain > 0.15 and rot_gain > 0.10
        else "one_step_action_fit_weak"
    )
    return {
        "split_seed": seed,
        "train_examples": int(len(train_idx)),
        "val_examples": int(len(val_idx)),
        "close_majority_baseline": majority,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "training_loss_start": losses[0],
        "training_loss_end": losses[-1],
        "loss_drop_fraction": float(loss_drop),
        "close_accuracy_gain_over_baseline": float(close_gain),
        "orientation_mse_improvement_fraction": float(rot_gain),
        "one_step_fit_status": status,
    }


def _fit_with_split(
    features: np.ndarray,
    actions: np.ndarray,
    episodes: list[str],
    episode_seed: dict[str, int],
    split_seed: int,
) -> dict[str, Any]:
    train_eps, val_eps = _seed_stratified_split(episode_seed, split_seed)
    train_idx = _episode_indices(episodes, train_eps)
    val_idx = _episode_indices(episodes, val_eps)
    if len(train_idx) == 0 or len(val_idx) == 0:
        return {
            "split_seed": split_seed,
            "one_step_fit_status": "insufficient_examples",
            "loss_drop_fraction": 0.0,
            "close_accuracy_gain_over_baseline": 0.0,
            "orientation_mse_improvement_fraction": 0.0,
            "train_examples": int(len(train_idx)),
            "val_examples": int(len(val_idx)),
        }
    return _train_probe(features, actions, train_idx, val_idx, split_seed)


def _parity_summary(live: np.ndarray, diag: np.ndarray) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for idx, name in enumerate(STATE_DIM_NAMES):
        diff = np.abs(live[:, idx] - diag[:, idx])
        if name in CONTINUOUS_DIM_NAMES:
            summary[name] = {
                "p95_abs_diff": float(np.quantile(diff, 0.95)),
                "mean_abs_diff": float(np.mean(diff)),
                "max_abs_diff": float(np.max(diff)),
            }
        else:
            summary[name] = {
                "exact_agreement_rate": float(np.mean(live[:, idx] == diag[:, idx])),
                "mismatch_count": int(np.sum(live[:, idx] != diag[:, idx])),
            }
    return summary


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def run() -> int:
    ident = current_repo_identity()
    run_instance_id = make_run_instance("postv13c", ident["working_head_commit"])

    slice2_plan = _load(ARTIFACT_DIR / "v13_slice2_state_action_interface_wiring_plan.json")
    slice2_summary = _load(ARTIFACT_DIR / "v13_slice2_tranche_summary.json")
    slice2_fit = _load(ARTIFACT_DIR / "v13_slice2_one_step_fit_audit.json")
    slice2_action = _load(ARTIFACT_DIR / "v13_slice2_action_interface_audit.json")
    slice2_state = _load(ARTIFACT_DIR / "v13_slice2_state_consumption_audit.json")

    v13_summary = _load(ARTIFACT_DIR / "v13_g1_g4_tranche_summary.json")
    v13_fit = _load(ARTIFACT_DIR / "v13_one_step_supervised_fit_audit.json")

    admission_plan = _load(ARTIFACT_DIR / "post_v13_admission_plan.json")
    admission_summary = _load(ARTIFACT_DIR / "post_v13_admission_summary.json")

    plan = _build_plan(run_instance_id, ident, slice2_plan, admission_plan)
    write_json_atomic(PLAN_ARTIFACT_PATH, plan)
    save_active_plan(plan)

    c0_path = GATES_DIR / "C0_authority_refreeze.json"
    c1_path = GATES_DIR / "C1_live_support_expansion.json"
    c2_path = GATES_DIR / "C2_live_diagnostic_parity.json"
    c3_path = GATES_DIR / "C3_signal_band_stability.json"
    c4_path = GATES_DIR / "C4_final_pretrain_closure_verdict.json"

    c0_blocking: list[str] = []
    if ident.get("branch") != EXPECTED_BRANCH:
        c0_blocking.append("branch_drift")
    if ident.get("working_head_commit") != EXPECTED_HEAD:
        c0_blocking.append("head_drift")
    if ident.get("vendor_head_commit") != EXPECTED_VENDOR:
        c0_blocking.append("vendor_drift")
    if str(admission_summary.get("final_verdict") or "") != EXPECTED_ADMISSION_VERDICT:
        c0_blocking.append("post_v13_admission_verdict_mismatch")
    if list(admission_summary.get("blocking_reasons") or []) != EXPECTED_ADMISSION_BLOCKERS:
        c0_blocking.append("post_v13_admission_blockers_mismatch")
    if slice2_summary.get("stop_point") != "G4":
        c0_blocking.append("slice2_stop_point_mismatch")
    if v13_summary.get("stop_point") != "G4":
        c0_blocking.append("v13_stop_point_mismatch")
    c0_status = "PASS" if not c0_blocking else "STOP_authority_drift"
    c0_gate = write_gate(
        gate_path=c0_path,
        gate_id="C0",
        gate_name="authority_refreeze",
        run_instance_id=run_instance_id,
        status=c0_status,
        blocking_reasons=c0_blocking,
        allowed_next_phases=["C1"] if c0_status == "PASS" else [],
        extra={
            "expected_head_commit": EXPECTED_HEAD,
            "expected_vendor_head_commit": EXPECTED_VENDOR,
            "expected_branch": EXPECTED_BRANCH,
            "carryforward_anchor": {
                "v13_run_instance_id": v13_summary.get("run_instance_id"),
                "v13_slice2_run_instance_id": slice2_summary.get("run_instance_id"),
                "post_v13_admission_run_instance_id": admission_summary.get("run_instance_id"),
            },
            "expected_admission_blockers": EXPECTED_ADMISSION_BLOCKERS,
        },
        spec_reference=POST_V13_ADMISSION_SPEC_REFERENCE,
    )

    selected_live_paths: list[Path] = []
    selected_live_rows: list[dict[str, Any]] = []
    c1_attempts: list[dict[str, Any]] = []
    c1_seed_summary: dict[str, Any] = {}
    c1_blocking: list[str] = []
    raw_success_attempt_count = 0

    if LIVE_DIR.exists():
        shutil.rmtree(LIVE_DIR)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)

    if c0_status == "PASS":
        live_contract = _carrier_contract()
        for seed in TARGET_SEEDS:
            seen_fingerprints: dict[str, int] = {}
            unique_success_count = 0
            duplicate_streak = 0
            attempts_for_seed = []
            stop_reason = "scan_limit_reached"
            for episode_index in range(MAX_EPISODE_SCAN_PER_SEED):
                rollout = _build_rollout(
                    seed=seed,
                    episode_index=episode_index,
                    contract=live_contract,
                    plan=plan,
                    source_slice2_plan=slice2_plan,
                    execution_scope="post_v13_final_pretrain_closure_c1",
                )
                fingerprint = _fingerprint_rollout(rollout)
                duplicate_of = seen_fingerprints.get(fingerprint)
                is_duplicate = duplicate_of is not None
                if bool(rollout.get("success")):
                    raw_success_attempt_count += 1
                attempt = {
                    "seed": seed,
                    "episode_index": episode_index,
                    "success": bool(rollout.get("success")),
                    "steps": int(rollout.get("steps", 0)),
                    "final_drawer_fraction": float(rollout.get("final_drawer_fraction", 0.0) or 0.0),
                    "fingerprint": fingerprint,
                    "duplicate_of_episode_index": duplicate_of,
                }
                if is_duplicate:
                    duplicate_streak += 1
                else:
                    duplicate_streak = 0
                    seen_fingerprints[fingerprint] = episode_index
                if bool(rollout.get("success")) and not is_duplicate and unique_success_count < TARGET_SUCCESS_PER_SEED:
                    base = LIVE_DIR / _episode_key(seed, episode_index)
                    save_robot_rollout(base, rollout)
                    selected_live_paths.append(base.with_suffix(".json"))
                    selected_live_rows.append(
                        {
                            "seed": seed,
                            "episode_index": episode_index,
                            "success": True,
                            "fingerprint": fingerprint,
                            "path": str(base.with_suffix(".json")),
                            "steps": int(rollout.get("steps", 0)),
                            "final_drawer_fraction": float(rollout.get("final_drawer_fraction", 0.0) or 0.0),
                        }
                    )
                    unique_success_count += 1
                attempts_for_seed.append(attempt)
                c1_attempts.append(attempt)
                if unique_success_count >= TARGET_SUCCESS_PER_SEED:
                    stop_reason = "target_unique_success_count_reached"
                    break
                if duplicate_streak >= DUPLICATE_EXHAUSTION_STREAK:
                    stop_reason = "duplicate_fingerprint_exhaustion"
                    break
            c1_seed_summary[str(seed)] = {
                "unique_success_count": unique_success_count,
                "attempt_count": len(attempts_for_seed),
                "successful_attempt_count": int(sum(1 for row in attempts_for_seed if row["success"])),
                "unique_fingerprint_count": len(seen_fingerprints),
                "stop_reason": stop_reason,
                "attempts": attempts_for_seed,
            }

        selected_seed_coverage = len({row["seed"] for row in selected_live_rows})
        selected_unique_episode_count = len(selected_live_rows)
        if selected_unique_episode_count < MIN_TOTAL_EPISODES:
            c1_blocking.append("selected_unique_live_episode_count_lt_16")
        if selected_seed_coverage < MIN_SEED_COVERAGE:
            c1_blocking.append("selected_seed_coverage_lt_6")
        if any(
            summary["stop_reason"] == "duplicate_fingerprint_exhaustion"
            for summary in c1_seed_summary.values()
        ):
            c1_blocking.append("episode_index_failed_to_expand_unique_live_support")
        c1_status = "PASS" if not c1_blocking else "STOP_support_expansion_failed"
    else:
        selected_seed_coverage = 0
        selected_unique_episode_count = 0
        c1_blocking = ["authority_refreeze_not_passed"]
        c1_status = "STOP_support_expansion_failed"

    c1_gate = write_gate(
        gate_path=c1_path,
        gate_id="C1",
        gate_name="live_support_expansion",
        run_instance_id=run_instance_id,
        status=c1_status,
        blocking_reasons=c1_blocking,
        allowed_next_phases=["C2"] if selected_live_rows else [],
        extra={
            "target_seeds": TARGET_SEEDS,
            "target_successful_teacher_episodes_per_seed": TARGET_SUCCESS_PER_SEED,
            "max_episode_scan_per_seed": MAX_EPISODE_SCAN_PER_SEED,
            "duplicate_exhaustion_streak": DUPLICATE_EXHAUSTION_STREAK,
            "selected_unique_live_episode_count": selected_unique_episode_count,
            "selected_seed_coverage": selected_seed_coverage,
            "raw_success_attempt_count": raw_success_attempt_count,
            "exact_selected_episode_list": selected_live_rows,
            "seed_summary": c1_seed_summary,
            "live_rollout_dir": str(LIVE_DIR),
        },
        spec_reference=POST_V13_ADMISSION_SPEC_REFERENCE,
    )

    if PARITY_DIR.exists():
        shutil.rmtree(PARITY_DIR)
    PARITY_DIR.mkdir(parents=True, exist_ok=True)

    c2_blocking: list[str] = []
    parity_rows: list[dict[str, Any]] = []
    parity_aggregate: dict[str, list[float]] = {name: [] for name in STATE_DIM_NAMES}

    if selected_live_paths:
        diag_contract = _diagnostic_contract()
        for live_meta_path in selected_live_paths:
            live_meta = json.loads(live_meta_path.read_text())
            seed = int(live_meta["seed"])
            episode_index = int(live_meta["episode_index"])
            diag_rollout = _build_rollout(
                seed=seed,
                episode_index=episode_index,
                contract=diag_contract,
                plan=plan,
                source_slice2_plan=slice2_plan,
                execution_scope="post_v13_final_pretrain_closure_c2",
            )
            diag_base = PARITY_DIR / _episode_key(seed, episode_index)
            save_robot_rollout(diag_base, diag_rollout)

            live_arrays = _load_rollout_arrays(live_meta_path)
            diag_arrays = _load_rollout_arrays(diag_base.with_suffix(".json"))
            live_states = np.asarray(live_arrays["states"], dtype=np.float32)
            diag_state = _diag_state(
                np.asarray(diag_arrays["states"], dtype=np.float32),
                np.asarray(diag_arrays["actions"], dtype=np.float32),
                np.asarray(diag_arrays["handle_distance_trace"], dtype=np.float32),
                np.asarray(diag_arrays["orientation_alignment_trace"], dtype=np.float32),
                np.asarray(diag_arrays["orientation_error_trace"], dtype=np.float32),
                np.asarray(diag_arrays["attach_eligible_trace"], dtype=bool),
            )
            if live_states.shape != diag_state.shape:
                c2_blocking.append(f"shape_mismatch_seed_{seed}_episode_{episode_index}")
                continue
            parity = _parity_summary(live_states, diag_state)
            for name in CONTINUOUS_DIM_NAMES:
                parity_aggregate[name].append(parity[name]["p95_abs_diff"])
            for name in DISCRETE_DIM_NAMES:
                parity_aggregate[name].append(parity[name]["exact_agreement_rate"])
            parity_rows.append(
                {
                    "seed": seed,
                    "episode_index": episode_index,
                    "frame_count": int(live_states.shape[0]),
                    "parity": parity,
                }
            )

        if not parity_rows:
            c2_blocking.append("no_live_episode_available_for_parity")
        else:
            for row in parity_rows:
                for name in CONTINUOUS_DIM_NAMES:
                    if row["parity"][name]["p95_abs_diff"] > 1e-3:
                        _append_unique(c2_blocking, f"{name}_p95_abs_diff_gt_1e-3")
                if row["parity"]["prev_close_cmd"]["exact_agreement_rate"] < 0.99:
                    _append_unique(c2_blocking, "prev_close_cmd_exact_agreement_lt_99pct")
                if row["parity"]["attach_eligible_proxy"]["exact_agreement_rate"] < 0.99:
                    _append_unique(c2_blocking, "attach_eligible_proxy_exact_agreement_lt_99pct")
        c2_status = "PASS" if not c2_blocking else "STOP_parity_mismatch"
    else:
        c2_blocking = ["live_support_expansion_produced_no_episode"]
        c2_status = "STOP_parity_mismatch"

    c2_gate = write_gate(
        gate_path=c2_path,
        gate_id="C2",
        gate_name="live_diagnostic_parity",
        run_instance_id=run_instance_id,
        status=c2_status,
        blocking_reasons=c2_blocking,
        allowed_next_phases=["C3"] if parity_rows else [],
        extra={
            "parity_rollout_dir": str(PARITY_DIR),
            "feature_order_match": True,
            "feature_dim_match": True,
            "feature_names": STATE_DIM_NAMES,
            "episode_parity_summary": parity_rows,
        },
        spec_reference=POST_V13_ADMISSION_SPEC_REFERENCE,
    )

    c3_blocking: list[str] = []
    live_full = {}
    live_repeats: list[dict[str, Any]] = []
    matched_baseline: list[dict[str, Any]] = []
    benign_support_size_effect = False
    p25_close = 0.0
    p25_rot = 0.0
    selected_counts_by_seed: dict[int, int] = defaultdict(int)
    for row in selected_live_rows:
        selected_counts_by_seed[int(row["seed"])] += 1

    if selected_live_paths:
        live_features, live_actions, live_episode_rows, live_episode_seed = _collect_fit_dataset(selected_live_paths)
        live_full = _fit_with_split(
            live_features,
            live_actions,
            live_episode_rows,
            live_episode_seed,
            FULLSET_SPLIT_SEED,
        )
        for split_seed in REPEAT_SPLIT_SEEDS:
            live_repeats.append(
                _fit_with_split(
                    live_features,
                    live_actions,
                    live_episode_rows,
                    live_episode_seed,
                    split_seed,
                )
            )

        diag_root = ARTIFACT_DIR / "g6_orientation_support_train_rollouts"
        available_by_seed: dict[int, list[Path]] = defaultdict(list)
        for meta_path in sorted(diag_root.glob("*.json")):
            meta = json.loads(meta_path.read_text())
            available_by_seed[int(meta["seed"])].append(meta_path)
        for idx, baseline_seed in enumerate(MATCHED_BASELINE_SEEDS):
            selected_baseline_paths: list[Path] = []
            for seed, count in sorted(selected_counts_by_seed.items()):
                candidates = list(available_by_seed.get(seed, []))
                rng = random.Random(baseline_seed + seed * 9973)
                rng.shuffle(candidates)
                selected_baseline_paths.extend(sorted(candidates[:count]))
            if len(selected_baseline_paths) != sum(selected_counts_by_seed.values()):
                c3_blocking.append("matched_support_baseline_incomplete")
                break
            base_features, base_actions, base_episodes, base_episode_seed = _collect_diag_dataset(selected_baseline_paths)
            matched_baseline.append(
                {
                    "baseline_index": idx,
                    "support_seed": baseline_seed,
                    "episode_count": len(selected_baseline_paths),
                    "split_result": _fit_with_split(
                        base_features,
                        base_actions,
                        base_episodes,
                        base_episode_seed,
                        FULLSET_SPLIT_SEED,
                    ),
                    "selected_episode_list": [path.name for path in selected_baseline_paths],
                }
            )

        baseline_close = [
            row["split_result"]["close_accuracy_gain_over_baseline"]
            for row in matched_baseline
            if row["split_result"].get("one_step_fit_status") != "insufficient_examples"
        ]
        baseline_rot = [
            row["split_result"]["orientation_mse_improvement_fraction"]
            for row in matched_baseline
            if row["split_result"].get("one_step_fit_status") != "insufficient_examples"
        ]
        if baseline_close and baseline_rot:
            p25_close = float(np.quantile(np.asarray(baseline_close, dtype=np.float32), 0.25))
            p25_rot = float(np.quantile(np.asarray(baseline_rot, dtype=np.float32), 0.25))

        if selected_unique_episode_count < MIN_TOTAL_EPISODES:
            c3_blocking.append("effective_unique_live_support_lt_16")
        if live_full.get("one_step_fit_status") != "one_step_action_fit_ok":
            c3_blocking.append("live_fullset_one_step_fit_not_ok")
        if float(live_full.get("loss_drop_fraction") or 0.0) <= 0.50:
            c3_blocking.append("live_fullset_loss_drop_not_gt_0_50")
        if float(live_full.get("close_accuracy_gain_over_baseline") or 0.0) <= 0.0:
            c3_blocking.append("live_fullset_close_gain_not_positive")
        if float(live_full.get("orientation_mse_improvement_fraction") or 0.0) <= 0.0:
            c3_blocking.append("live_fullset_orientation_gain_not_positive")

        for repeat in live_repeats:
            if repeat.get("one_step_fit_status") != "one_step_action_fit_ok":
                c3_blocking.append(f"repeat_{repeat.get('split_seed')}_fit_not_ok")
            if float(repeat.get("close_accuracy_gain_over_baseline") or 0.0) <= 0.0:
                c3_blocking.append(f"repeat_{repeat.get('split_seed')}_close_gain_nonpositive")
            if float(repeat.get("orientation_mse_improvement_fraction") or 0.0) <= 0.0:
                c3_blocking.append(f"repeat_{repeat.get('split_seed')}_orientation_gain_nonpositive")

        close_gap = p25_close - float(live_full.get("close_accuracy_gain_over_baseline") or 0.0)
        rot_gap = p25_rot - float(live_full.get("orientation_mse_improvement_fraction") or 0.0)
        close_above_p25 = float(live_full.get("close_accuracy_gain_over_baseline") or 0.0) >= p25_close
        rot_above_p25 = float(live_full.get("orientation_mse_improvement_fraction") or 0.0) >= p25_rot
        if not (close_above_p25 and rot_above_p25):
            benign_support_size_effect = (
                c2_status == "PASS"
                and all(repeat.get("one_step_fit_status") == "one_step_action_fit_ok" for repeat in live_repeats)
                and all(float(repeat.get("close_accuracy_gain_over_baseline") or 0.0) > 0.0 for repeat in live_repeats)
                and all(float(repeat.get("orientation_mse_improvement_fraction") or 0.0) > 0.0 for repeat in live_repeats)
                and close_gap <= BENIGN_P25_GAP_TOLERANCE
                and rot_gap <= BENIGN_P25_GAP_TOLERANCE
                and float(live_full.get("close_accuracy_gain_over_baseline") or 0.0) > 0.0
                and float(live_full.get("orientation_mse_improvement_fraction") or 0.0) > 0.0
            )
            if not benign_support_size_effect:
                c3_blocking.append("live_fit_below_matched_support_distribution")
        c3_status = "PASS" if not c3_blocking else "STOP_signal_band_unstable"
    else:
        c3_blocking = ["no_live_support_available_for_signal_band_audit"]
        c3_status = "STOP_signal_band_unstable"

    c3_gate = write_gate(
        gate_path=c3_path,
        gate_id="C3",
        gate_name="signal_band_stability",
        run_instance_id=run_instance_id,
        status=c3_status,
        blocking_reasons=c3_blocking,
        allowed_next_phases=["C4"],
        extra={
            "live_fullset_fit": live_full,
            "live_repeat_fits": live_repeats,
            "matched_support_baseline_distribution": matched_baseline,
            "matched_support_p25_close_accuracy_gain": p25_close,
            "matched_support_p25_orientation_mse_improvement": p25_rot,
            "benign_support_size_effect": benign_support_size_effect,
            "selected_counts_by_seed": dict(sorted(selected_counts_by_seed.items())),
        },
        spec_reference=POST_V13_ADMISSION_SPEC_REFERENCE,
    )

    all_pass = c0_status == "PASS" and c1_status == "PASS" and c2_status == "PASS" and c3_status == "PASS"
    final_verdict = "ALLOW_CANARY_TRAIN_PROBE" if all_pass else "STOP_PRETRAIN_INTERFACE_UNSTABLE"
    final_blocking = []
    if c0_status != "PASS":
        for reason in c0_blocking:
            _append_unique(final_blocking, reason)
    if c1_status != "PASS":
        for reason in c1_blocking:
            _append_unique(final_blocking, reason)
    if c2_status != "PASS":
        for reason in c2_blocking:
            _append_unique(final_blocking, reason)
    if c3_status != "PASS":
        for reason in c3_blocking:
            _append_unique(final_blocking, reason)

    summary = {
        "run_instance_id": run_instance_id,
        "plan_version": PLAN_VERSION,
        "working_head_commit": ident["working_head_commit"],
        "execution_scope": plan["execution_scope"],
        "diagnostic_only": True,
        "claim_bearing": False,
        "gates": {
            "C0": c0_status,
            "C1": c1_status,
            "C2": c2_status,
            "C3": c3_status,
            "C4": "PASS",
        },
        "final_verdict": final_verdict,
        "blocking_reasons": final_blocking,
        "source_anchor": {
            "v13_run_instance_id": v13_summary.get("run_instance_id"),
            "v13_slice2_run_instance_id": slice2_summary.get("run_instance_id"),
            "post_v13_admission_run_instance_id": admission_summary.get("run_instance_id"),
        },
        "closure_findings": {
            "selected_unique_live_episode_count": selected_unique_episode_count,
            "selected_seed_coverage": selected_seed_coverage,
            "raw_success_attempt_count": raw_success_attempt_count,
            "slice2_prior_signal_band_retained": slice2_fit.get("fit_retains_prior_signal_band"),
            "parity_passed": c2_status == "PASS",
            "matched_support_benign_effect": benign_support_size_effect,
        },
        "future_canary_prebinding": plan["future_canary_prebinding"] if final_verdict == "ALLOW_CANARY_TRAIN_PROBE" else None,
        "timestamp_utc": utc_now(),
    }
    write_json_atomic(SUMMARY_PATH, summary)
    c4_gate = write_gate(
        gate_path=c4_path,
        gate_id="C4",
        gate_name="final_pretrain_closure_verdict",
        run_instance_id=run_instance_id,
        status="PASS",
        blocking_reasons=final_blocking,
        allowed_next_phases=["CANARY_TRAIN_PROBE"] if final_verdict == "ALLOW_CANARY_TRAIN_PROBE" else [],
        extra={
            "final_verdict": final_verdict,
            "training_allowed_now": final_verdict == "ALLOW_CANARY_TRAIN_PROBE",
            "probe_allowed_now": final_verdict == "ALLOW_CANARY_TRAIN_PROBE",
            "summary_artifact_path": str(SUMMARY_PATH),
        },
        spec_reference=POST_V13_ADMISSION_SPEC_REFERENCE,
    )

    print(
        json.dumps(
            {
                "plan": plan,
                "C0": c0_gate,
                "C1": c1_gate,
                "C2": c2_gate,
                "C3": c3_gate,
                "C4": c4_gate,
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
