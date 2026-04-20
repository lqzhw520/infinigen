#!/usr/bin/env python3
from __future__ import annotations

import json
import random

import numpy as np
import torch
from torch import nn

from v13_audit_common import (
    ARTIFACT_DIR,
    GATES_V13_DIR,
    ensure_v13_plan,
    load_json,
    persist_plan_scope,
    write_gate,
    write_json_atomic,
)

OUTPUT_PATH = ARTIFACT_DIR / "v13_one_step_supervised_fit_audit.json"
SUMMARY_PATH = ARTIFACT_DIR / "v13_g1_g4_tranche_summary.json"
GATE_PATH = GATES_V13_DIR / "G4_one_step_supervised_fit.json"
ROLLOUT_DIR = ARTIFACT_DIR / "g6_orientation_support_train_rollouts"


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


def _diag_state(states: np.ndarray, actions: np.ndarray, handle_distance: np.ndarray, orientation_alignment: np.ndarray, orientation_error: np.ndarray, attach_eligible: np.ndarray) -> np.ndarray:
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


def _collect_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    current_states = []
    diag_states = []
    actions = []
    episodes = []
    for meta_path in sorted(ROLLOUT_DIR.glob("*.json")):
        data = np.load(meta_path.with_suffix(".npz"), allow_pickle=True)
        states = np.asarray(data["states"], dtype=np.float32)
        acts = np.asarray(data["actions"], dtype=np.float32)
        handle_distance = np.asarray(data["handle_distance_trace"], dtype=np.float32)
        orientation_alignment = np.asarray(data["orientation_alignment_trace"], dtype=np.float32)
        orientation_error = np.asarray(data["orientation_error_trace"], dtype=np.float32)
        attach_eligible = np.asarray(data["attach_eligible_trace"], dtype=bool)
        current_states.append(states)
        diag_states.append(_diag_state(states, acts, handle_distance, orientation_alignment, orientation_error, attach_eligible))
        actions.append(acts)
        episodes.extend([meta_path.stem] * len(states))
    return np.concatenate(current_states), np.concatenate(diag_states), np.concatenate(actions), episodes


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


def _episode_split(episodes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(set(episodes))
    rng = random.Random(0)
    rng.shuffle(unique)
    cut = max(1, int(0.8 * len(unique)))
    train_eps = set(unique[:cut])
    train_idx = np.array([i for i, ep in enumerate(episodes) if ep in train_eps], dtype=np.int64)
    val_idx = np.array([i for i, ep in enumerate(episodes) if ep not in train_eps], dtype=np.int64)
    if len(val_idx) == 0:
        val_idx = train_idx[-max(1, len(train_idx) // 5):]
        train_idx = train_idx[: len(train_idx) - len(val_idx)]
    return train_idx, val_idx


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
    rot_cos = float(torch.mean((rot_num[valid] / rot_den[valid]).clamp(-1.0, 1.0)).item()) if bool(torch.any(valid)) else 0.0
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


def _train_probe(name: str, features: np.ndarray, target: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray) -> dict[str, object]:
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
    return {
        "name": name,
        "train_examples": int(len(train_idx)),
        "val_examples": int(len(val_idx)),
        "close_majority_baseline": majority,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "training_loss_start": losses[0],
        "training_loss_end": losses[-1],
    }


def run() -> int:
    plan = persist_plan_scope(ensure_v13_plan(), "v13_one_step_supervised_fit")
    run_instance_id = str(plan["run_instance_id"])
    current_state, diag_state, actions, episodes = _collect_dataset()
    train_idx, val_idx = _episode_split(episodes)
    current_probe = _train_probe("current_m0_proxy_state", current_state, actions, train_idx, val_idx)
    diag_probe = _train_probe("orientation_bridge_state_v1", diag_state, actions, train_idx, val_idx)

    best = diag_probe
    loss_drop = (best["training_loss_start"] - best["training_loss_end"]) / max(best["training_loss_start"], 1e-8)
    close_gain = best["final_metrics"]["close_dim_accuracy"] - best["close_majority_baseline"]
    init_rot = best["initial_metrics"]["orientation_action_mse"]
    final_rot = best["final_metrics"]["orientation_action_mse"]
    rot_gain = (init_rot - final_rot) / max(init_rot, 1e-8)

    if loss_drop > 0.20 and close_gain > 0.15 and rot_gain > 0.10:
        status_label = "one_step_action_fit_ok"
    elif close_gain <= 0.15:
        status_label = "one_step_close_not_fit"
    else:
        status_label = "one_step_orientation_not_fit"

    artifact = {
        "audit": "v13_one_step_supervised_fit_audit",
        "run_instance_id": run_instance_id,
        "plan_version": plan["plan_version"],
        "working_head_commit": plan["working_head_commit"],
        "execution_scope": plan["execution_scope"],
        "dataset_examples": int(len(current_state)),
        "episode_count": int(len(set(episodes))),
        "current_state_probe": current_probe,
        "diagnostic_state_probe": diag_probe,
        "best_probe_name": best["name"],
        "loss_drop_fraction": float(loss_drop),
        "close_accuracy_gain_over_baseline": float(close_gain),
        "orientation_mse_improvement_fraction": float(rot_gain),
        "one_step_fit_status": status_label,
        "training_allowed_after_g4": status_label == "one_step_action_fit_ok",
    }
    write_json_atomic(OUTPUT_PATH, artifact)
    blocking = [] if status_label == "one_step_action_fit_ok" else [status_label]
    gate = write_gate(
        gate_path=GATE_PATH,
        gate_id="G4",
        gate_name="one_step_supervised_fit",
        run_instance_id=run_instance_id,
        status="PASS" if not blocking else "STOP",
        blocking_reasons=blocking,
        allowed_next_phases=["slice_closed"] if not blocking else [],
        extra={
            "audit_artifact_path": str(OUTPUT_PATH),
            "one_step_fit_status": status_label,
            "best_probe_name": best["name"],
            "loss_drop_fraction": float(loss_drop),
            "close_accuracy_gain_over_baseline": float(close_gain),
            "orientation_mse_improvement_fraction": float(rot_gain),
            "training_allowed_after_g4": status_label == "one_step_action_fit_ok",
        },
    )
    summary = {
        "run_instance_id": run_instance_id,
        "plan_version": plan["plan_version"],
        "working_head_commit": plan["working_head_commit"],
        "execution_scope": plan["execution_scope"],
        "diagnostic_only": True,
        "claim_bearing": False,
        "gates": {
            "G0": load_json(GATES_V13_DIR / "G0_sovereign_sync.json", {}).get("status"),
            "G1": load_json(GATES_V13_DIR / "G1_classification_sanity.json", {}).get("status"),
            "G2": load_json(GATES_V13_DIR / "G2_action_interface_audit.json", {}).get("status"),
            "G3": load_json(GATES_V13_DIR / "G3_orientation_state_consumption.json", {}).get("status"),
            "G4": gate.get("status"),
        },
        "stop_point": "G4",
        "one_step_fit_status": status_label,
        "training_allowed_after_g4": status_label == "one_step_action_fit_ok",
        "slice_constraint": "Do not start G5/G6 in this tranche.",
    }
    write_json_atomic(SUMMARY_PATH, summary)
    print(json.dumps({"artifact": artifact, "gate": gate, "summary": summary}, indent=2))
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(run())
