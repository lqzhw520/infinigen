#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
MINT_SCRIPTS = PROJECT_ROOT / "scripts" / "mint"
sys.path.insert(0, str(MINT_SCRIPTS))

from v13_audit_common import (  # noqa: E402
    ACCEPTANCE_CONTRACT_PATH,
    ARTIFACT_DIR,
    AUTOPILOT_DIR,
    TRUTH_CONTRACT_PATH,
    current_repo_identity,
    load_json,
    make_run_instance,
    save_active_plan,
    sha256_file,
    write_gate,
    write_json_atomic,
)

SPEC_REFERENCE = str(PROJECT_ROOT / "docs" / "MINT_V84_P0_INFRASTRUCTURE_REPAIR_SPEC.md")
PLAN_OUTPUT_PATH = ARTIFACT_DIR / "p0_infrastructure_repair_plan.json"
OUTPUT_PATH = ARTIFACT_DIR / "re_c3_signal_band_stability.json"
GATE_PATH = AUTOPILOT_DIR / "gates_p0_infrastructure" / "Re_C3_signal_band_stability.json"
SOURCE_RE_C1_GATE = AUTOPILOT_DIR / "gates_p0_infrastructure" / "Re_C1_live_support_expansion.json"
SOURCE_RE_C2_GATE = AUTOPILOT_DIR / "gates_p0_infrastructure" / "Re_C2_live_diagnostic_parity.json"
REPEAT_SPLIT_SEEDS = [11, 23, 37, 41]
MIN_EFFECTIVE_SUPPORT = 16
P0C_CLOSE_DELTA_MIN = -0.2


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


def _make_plan() -> dict[str, Any]:
    ident = current_repo_identity()
    run_instance_id = make_run_instance("rec3", ident["working_head_commit"])
    plan = {
        "run_instance_id": run_instance_id,
        "plan_version": "tiny_retrain_confirmation_p0_infrastructure_repair",
        "slice_name": "re-c3-signal-band-stability",
        "working_head_commit": ident["working_head_commit"],
        "vendor_head_commit": ident["vendor_head_commit"],
        "branch": ident["branch"],
        "execution_scope": "re_c3_signal_band_stability",
        "bridge_stage": "p0_infrastructure_repair",
        "diagnostic_only": True,
        "claim_bearing": False,
        "training_allowed": False,
        "probe_allowed": False,
        "spec_reference": SPEC_REFERENCE,
        "truth_contract_hash": sha256_file(TRUTH_CONTRACT_PATH),
        "acceptance_contract_hash": sha256_file(ACCEPTANCE_CONTRACT_PATH),
        "allowed_next_phases": ["Re-C3", "Canary"],
    }
    write_json_atomic(PLAN_OUTPUT_PATH, plan)
    save_active_plan(plan)
    return plan


def _load_rollout_arrays(meta_path: Path) -> dict[str, np.ndarray]:
    data = np.load(meta_path.with_suffix(".npz"), allow_pickle=True)
    return {key: np.asarray(data[key]) for key in data.files}


def _collect_fit_dataset(meta_paths: list[Path]) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    features = []
    actions = []
    episodes = []
    episode_seed: dict[str, int] = {}
    for meta_path in sorted(meta_paths):
        meta = load_json(meta_path, {})
        arrays = _load_rollout_arrays(meta_path)
        states = np.asarray(arrays["states"], dtype=np.float32)
        acts = np.asarray(arrays["actions"], dtype=np.float32)
        stem = meta_path.stem
        features.append(states)
        actions.append(acts)
        episodes.extend([stem] * len(states))
        episode_seed[stem] = int(meta["seed"])
    return np.concatenate(features), np.concatenate(actions), episodes, episode_seed


def _seed_stratified_split(
    episode_seed: dict[str, int],
    split_seed: int,
    train_fraction: float = 0.75,
) -> tuple[list[str], list[str]]:
    grouped: dict[int, list[str]] = {}
    for episode, seed in episode_seed.items():
        grouped.setdefault(int(seed), []).append(episode)
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
        val_target = max(1, int(round((1.0 - train_fraction) * max(1, len(grouped)))))
        while singleton_eps and len(val_eps) < val_target:
            val_eps.append(singleton_eps.pop())
        train_eps.extend(singleton_eps)
    if not train_eps and val_eps:
        train_eps.append(val_eps[-1])
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
    pred_close = (close_prob >= 0.5).float()
    close_acc = float(torch.mean((pred_close == close_t).float()).item())
    close_brier = float(torch.mean((close_prob - close_t) ** 2).item())
    predicted_close_rate = float(torch.mean(pred_close).item())
    teacher_close_rate = float(torch.mean(close_t).item())

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
        "predicted_close_rate": predicted_close_rate,
        "teacher_close_rate": teacher_close_rate,
        "close_cmd_rate_delta": predicted_close_rate - teacher_close_rate,
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

    status = "fit_ok"
    if not math.isfinite(loss_drop) or loss_drop <= 0.0:
        status = "fit_not_ok"
    if not all(math.isfinite(v) for v in [close_gain, rot_gain]):
        status = "fit_not_ok"

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
        "p0c_close_cmd_rate_delta": float(final_metrics["close_cmd_rate_delta"]),
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
            "one_step_fit_status": "fit_not_ok",
            "loss_drop_fraction": 0.0,
            "close_accuracy_gain_over_baseline": 0.0,
            "orientation_mse_improvement_fraction": 0.0,
            "p0c_close_cmd_rate_delta": -1.0,
            "train_examples": int(len(train_idx)),
            "val_examples": int(len(val_idx)),
        }
    return _train_probe(features, actions, train_idx, val_idx, split_seed)


def run() -> int:
    plan = _make_plan()
    rec1_gate = load_json(SOURCE_RE_C1_GATE, {})
    rec2_gate = load_json(SOURCE_RE_C2_GATE, {})

    blocking: list[str] = []
    if str(rec1_gate.get("status") or "") != "PASS":
        blocking.append("re_c1_not_passed")
    if str(rec2_gate.get("status") or "") != "PASS":
        blocking.append("re_c2_not_passed")

    live_dir = Path(str(rec1_gate.get("live_rollout_dir") or ""))
    if not live_dir.exists():
        blocking.append("re_c1_live_rollout_dir_missing")
        live_meta_paths: list[Path] = []
    else:
        live_meta_paths = sorted(live_dir.glob("*.json"))

    fullset_fit: dict[str, Any] = {}
    repeat_fits: list[dict[str, Any]] = []
    p0c_monitoring = {
        "close_cmd_rate_delta_threshold": P0C_CLOSE_DELTA_MIN,
        "fullset_close_cmd_rate_delta": None,
        "repeat_close_cmd_rate_deltas": {},
        "p0c_monitoring_pass": False,
    }

    if not blocking:
        features, actions, episodes, episode_seed = _collect_fit_dataset(live_meta_paths)
        fullset_fit = _fit_with_split(features, actions, episodes, episode_seed, 0)
        for split_seed in REPEAT_SPLIT_SEEDS:
            repeat_fits.append(
                _fit_with_split(features, actions, episodes, episode_seed, split_seed)
            )

        effective_unique_live_support = len(live_meta_paths)
        if effective_unique_live_support < MIN_EFFECTIVE_SUPPORT:
            blocking.append("effective_unique_live_support_lt_16")

        repeat11 = next((x for x in repeat_fits if x["split_seed"] == 11), None)
        repeat23 = next((x for x in repeat_fits if x["split_seed"] == 23), None)
        if not repeat11 or repeat11.get("one_step_fit_status") == "fit_not_ok":
            blocking.append("repeat_11_fit_not_ok")
        if not repeat23 or repeat23.get("one_step_fit_status") == "fit_not_ok":
            blocking.append("repeat_23_fit_not_ok")

        close_std = float(
            np.std(
                np.asarray(
                    [x["close_accuracy_gain_over_baseline"] for x in repeat_fits],
                    dtype=np.float32,
                )
            )
        )
        if close_std >= 0.05:
            blocking.append("close_accuracy_gain_std_ge_0_05")

        orientation_positive_repeat_count = sum(
            1 for x in repeat_fits if float(x["orientation_mse_improvement_fraction"]) > 0.0
        )
        if orientation_positive_repeat_count < 3:
            blocking.append("orientation_mse_improvement_not_positive_in_ge_3_of_4_repeats")

        p0c_monitoring["fullset_close_cmd_rate_delta"] = float(
            fullset_fit.get("p0c_close_cmd_rate_delta", 0.0)
        )
        p0c_monitoring["repeat_close_cmd_rate_deltas"] = {
            str(x["split_seed"]): float(x.get("p0c_close_cmd_rate_delta", 0.0))
            for x in repeat_fits
        }
        min_close_delta = min(
            [float(p0c_monitoring["fullset_close_cmd_rate_delta"])]
            + [float(v) for v in p0c_monitoring["repeat_close_cmd_rate_deltas"].values()]
        )
        p0c_monitoring["min_observed_close_cmd_rate_delta"] = float(min_close_delta)
        p0c_monitoring["p0c_monitoring_pass"] = bool(min_close_delta >= P0C_CLOSE_DELTA_MIN)

        close_gt_015_count = sum(
            1 for x in repeat_fits if float(x["close_accuracy_gain_over_baseline"]) > 0.15
        )
    else:
        effective_unique_live_support = len(live_meta_paths)
        close_std = None
        orientation_positive_repeat_count = 0
        close_gt_015_count = 0

    artifact = {
        "audit": "re_c3_signal_band_stability",
        "run_instance_id": plan["run_instance_id"],
        "plan_version": plan["plan_version"],
        "working_head_commit": plan["working_head_commit"],
        "execution_scope": plan["execution_scope"],
        "diagnostic_only": True,
        "claim_bearing": False,
        "effective_unique_live_support": effective_unique_live_support,
        "live_support_rollout_dir": str(live_dir) if live_dir else "",
        "re_c3_status": "PASS" if not blocking else "FAIL",
        "phenomenon_a_close_signal_stability": {
            "fullset_fit": fullset_fit,
            "repeat_fits": repeat_fits,
            "close_accuracy_gain_consistency_std": close_std,
            "close_accuracy_gain_gt_0_15_in_repeats": close_gt_015_count,
            "target_close_accuracy_gain_gt_0_15_in_ge_3_of_4_repeats": close_gt_015_count >= 3,
        },
        "phenomenon_b_orientation_signal_stability": {
            "orientation_positive_repeat_count": orientation_positive_repeat_count,
            "target_orientation_positive_in_ge_3_of_4_repeats": orientation_positive_repeat_count >= 3,
            "re_c2_clean_parity": str(rec2_gate.get("status") or "") == "PASS",
            "feature_parity_residual_attribution_used": False,
        },
        "p0c_monitoring": p0c_monitoring,
    }
    write_json_atomic(OUTPUT_PATH, artifact)

    re_c3_pass = not blocking
    allowed_next = ["Canary"] if re_c3_pass and bool(p0c_monitoring["p0c_monitoring_pass"]) else []
    gate = write_gate(
        gate_path=GATE_PATH,
        gate_id="Re-C3",
        gate_name="signal_band_stability",
        run_instance_id=plan["run_instance_id"],
        status="PASS" if re_c3_pass else "STOP",
        blocking_reasons=blocking,
        allowed_next_phases=allowed_next,
        extra={
            "audit_artifact_path": str(OUTPUT_PATH),
            "effective_unique_live_support": effective_unique_live_support,
            "repeat_fit_panel": REPEAT_SPLIT_SEEDS,
            "close_accuracy_gain_consistency_std": close_std,
            "orientation_positive_repeat_count": orientation_positive_repeat_count,
            "p0c_monitoring": p0c_monitoring,
            "canary_authorized_after_re_c3": bool(re_c3_pass and p0c_monitoring["p0c_monitoring_pass"]),
        },
        spec_reference=SPEC_REFERENCE,
    )
    print(json.dumps({"artifact": artifact, "gate": gate}, indent=2))
    return 0 if re_c3_pass else 1


if __name__ == "__main__":
    raise SystemExit(run())
