#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from v13_audit_common import (
    ARTIFACT_DIR,
    GATES_V13_SLICE2_DIR,
    ensure_v13_slice2_plan,
    load_json,
    persist_plan_scope,
    write_gate,
    write_json_atomic,
)

ROLLOUT_DIR = ARTIFACT_DIR / "v13_slice2_live_state_rollouts"
OUTPUT_PATH = ARTIFACT_DIR / "v13_slice2_one_step_fit_audit.json"
SUMMARY_PATH = ARTIFACT_DIR / "v13_slice2_tranche_summary.json"
GATE_PATH = GATES_V13_SLICE2_DIR / "G4_one_step_fit.json"
PRIOR_OUTPUT = ARTIFACT_DIR / "v13_one_step_supervised_fit_audit.json"


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


def _collect_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = []
    actions = []
    episodes = []
    for meta_path in sorted(ROLLOUT_DIR.glob("*.json")):
        data = np.load(meta_path.with_suffix(".npz"), allow_pickle=True)
        states = np.asarray(data["states"], dtype=np.float32)
        acts = np.asarray(data["actions"], dtype=np.float32)
        features.append(states)
        actions.append(acts)
        episodes.extend([meta_path.stem] * len(states))
    return np.concatenate(features), np.concatenate(actions), episodes


def _episode_split(episodes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(set(episodes))
    rng = random.Random(0)
    rng.shuffle(unique)
    cut = max(1, int(0.75 * len(unique)))
    train_eps = set(unique[:cut])
    train_idx = np.array([i for i, ep in enumerate(episodes) if ep in train_eps], dtype=np.int64)
    val_idx = np.array([i for i, ep in enumerate(episodes) if ep not in train_eps], dtype=np.int64)
    if len(val_idx) == 0:
        val_idx = train_idx[-max(1, len(train_idx) // 4):]
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


def _train_probe(features: np.ndarray, target: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray) -> dict[str, object]:
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
        "train_examples": int(len(train_idx)),
        "val_examples": int(len(val_idx)),
        "close_majority_baseline": majority,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "training_loss_start": losses[0],
        "training_loss_end": losses[-1],
    }


def run() -> int:
    plan = persist_plan_scope(ensure_v13_slice2_plan(), "v13_slice_2_one_step_fit")
    features, actions, episodes = _collect_dataset()
    train_idx, val_idx = _episode_split(episodes)
    probe = _train_probe(features, actions, train_idx, val_idx)
    loss_drop = (probe["training_loss_start"] - probe["training_loss_end"]) / max(probe["training_loss_start"], 1e-8)
    close_gain = probe["final_metrics"]["close_dim_accuracy"] - probe["close_majority_baseline"]
    init_rot = probe["initial_metrics"]["orientation_action_mse"]
    final_rot = probe["final_metrics"]["orientation_action_mse"]
    rot_gain = (init_rot - final_rot) / max(init_rot, 1e-8)
    prior = load_json(PRIOR_OUTPUT, {})
    prior_close_gain = float(prior.get("close_accuracy_gain_over_baseline", 0.0) or 0.0)
    prior_rot_gain = float(prior.get("orientation_mse_improvement_fraction", 0.0) or 0.0)

    if loss_drop > 0.20 and close_gain > 0.15 and rot_gain > 0.10:
        status_label = "one_step_action_fit_ok"
    elif close_gain <= 0.15:
        status_label = "one_step_close_not_fit"
    else:
        status_label = "one_step_orientation_not_fit"

    artifact = {
        "audit": "v13_slice2_one_step_fit_audit",
        "run_instance_id": plan["run_instance_id"],
        "plan_version": plan["plan_version"],
        "working_head_commit": plan["working_head_commit"],
        "execution_scope": plan["execution_scope"],
        "live_rollout_dir": str(ROLLOUT_DIR),
        "dataset_examples": int(len(features)),
        "episode_count": int(len(set(episodes))),
        "live_state_probe": probe,
        "loss_drop_fraction": float(loss_drop),
        "close_accuracy_gain_over_baseline": float(close_gain),
        "orientation_mse_improvement_fraction": float(rot_gain),
        "prior_diagnostic_close_accuracy_gain": prior_close_gain,
        "prior_diagnostic_orientation_mse_improvement_fraction": prior_rot_gain,
        "fit_retains_prior_signal_band": close_gain >= max(0.15, prior_close_gain - 0.15) and rot_gain >= max(0.10, prior_rot_gain - 0.15),
        "one_step_fit_status": status_label,
        "train_probe_may_be_considered_after_slice2": status_label == "one_step_action_fit_ok",
    }
    write_json_atomic(OUTPUT_PATH, artifact)
    blocking = [] if status_label == "one_step_action_fit_ok" else [status_label]
    gate = write_gate(
        gate_path=GATE_PATH,
        gate_id="G4",
        gate_name="one_step_fit",
        run_instance_id=plan["run_instance_id"],
        status="PASS" if not blocking else "STOP",
        blocking_reasons=blocking,
        allowed_next_phases=["decision_boundary_only"] if not blocking else [],
        extra={
            "audit_artifact_path": str(OUTPUT_PATH),
            "one_step_fit_status": status_label,
            "loss_drop_fraction": float(loss_drop),
            "close_accuracy_gain_over_baseline": float(close_gain),
            "orientation_mse_improvement_fraction": float(rot_gain),
            "train_probe_may_be_considered_after_slice2": status_label == "one_step_action_fit_ok",
        },
    )
    summary = {
        "run_instance_id": plan["run_instance_id"],
        "plan_version": plan["plan_version"],
        "working_head_commit": plan["working_head_commit"],
        "execution_scope": plan["execution_scope"],
        "diagnostic_only": True,
        "claim_bearing": False,
        "gates": {
            "G0": load_json(GATES_V13_SLICE2_DIR / "G0_sovereign_sync.json", {}).get("status"),
            "G2": load_json(GATES_V13_SLICE2_DIR / "G2_action_interface_audit.json", {}).get("status"),
            "G3": load_json(GATES_V13_SLICE2_DIR / "G3_state_consumption.json", {}).get("status"),
            "G4": gate.get("status"),
        },
        "stop_point": "G4",
        "one_step_fit_status": status_label,
        "train_probe_may_be_considered_after_slice2": status_label == "one_step_action_fit_ok",
        "slice_constraint": "Do not start training/probe in this tranche.",
        "decision_boundary": "Only a later separate tranche may consider train/probe, and only because slice-2 G2/G3/G4 all passed.",
    }
    write_json_atomic(SUMMARY_PATH, summary)
    print(json.dumps({"artifact": artifact, "gate": gate, "summary": summary}, indent=2))
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(run())

