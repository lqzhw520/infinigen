#!/usr/bin/env python3
"""Run one Phase 2 guided-generation conditioning group."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHYSNAP_ROOT = PROJECT_ROOT / "external" / "physnap"
PHYSNAP_PYTHON = Path("/root/anaconda3/envs/physnap/bin/python")


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


def merged_guide_cfg(conditioning: dict, group: dict) -> dict:
    guide_cfg = dict((conditioning.get("guide") or {}))
    guide_cfg.update(group.get("guide_overrides") or {})
    return guide_cfg


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    prefix = stem.split("_", 1)[0]
    try:
        return int(prefix), stem
    except ValueError:
        return -1, stem


def resolve_checkpoint_path(checkpoint: Path) -> Path:
    if checkpoint.exists():
        return checkpoint
    checkpoint_dir = checkpoint.parent if checkpoint.parent.name == "checkpoint" else None
    if checkpoint_dir and checkpoint_dir.exists():
        latest_named = sorted(checkpoint_dir.glob("*_latest.pt"), key=checkpoint_sort_key)
        if latest_named:
            return latest_named[-1]
        numbered = sorted(checkpoint_dir.glob("*.pt"), key=checkpoint_sort_key)
        if numbered:
            return numbered[-1]
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Phase 2 guided-generation group")
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    campaign_dir = Path(args.campaign_dir).resolve()
    manifest = load_manifest(campaign_dir)
    state = load_json(campaign_dir / "state.json", {})
    conditioning = (manifest.get("conditioning") or {})
    group = (conditioning.get("groups") or {}).get(args.group_id)
    if not group:
        raise SystemExit(f"Unknown conditioning group: {args.group_id}")

    checkpoint = state.get("phase2_base_checkpoint")
    config_path = state.get("phase2_base_config")
    if not checkpoint or not config_path:
        raise SystemExit("Phase 2 base checkpoint/config not populated in state.json")
    checkpoint = Path(checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    checkpoint = resolve_checkpoint_path(checkpoint)
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    cond_dir = Path(group["cond_dir"])
    if not cond_dir.is_absolute():
        cond_dir = PROJECT_ROOT / cond_dir
    run_name = group["output_name"]
    run_dir = PHYSNAP_ROOT / "log" / run_name
    runtime_dir = campaign_dir / "runtime" / run_name
    runtime_dir.mkdir(parents=True, exist_ok=True)
    log_path = runtime_dir / "launcher.log"
    meta_path = runtime_dir / "run_meta.json"

    guide_cfg = merged_guide_cfg(conditioning, group)
    ref_cfg = conditioning.get("reference", {})
    meta = {
        "group_id": args.group_id,
        "output_name": run_name,
        "cond_dir": str(cond_dir),
        "checkpoint_path": str(checkpoint),
        "config_path": str(config_path),
        "status": "running",
        "launched_at": now_iso(),
    }
    save_json(meta_path, meta)

    command = [
        str(PHYSNAP_PYTHON),
        str(PHYSNAP_ROOT / "eval" / "run_guided.py"),
        "--name",
        run_name,
        "--output_name",
        run_name,
        "--config_path",
        str(config_path),
        "--checkpoint_path",
        str(checkpoint),
        "--ref_name",
        ref_cfg["ref_name"],
        "--ref_path",
        str(PROJECT_ROOT / ref_cfg["ref_path"]),
        "--cond_dir",
        str(cond_dir),
        "--cond_fac",
        str(guide_cfg.get("cond_fac", 45)),
        "--pen_fac",
        str(guide_cfg.get("pen_fac", 2)),
        "--mob_fac",
        str(guide_cfg.get("mob_fac", 2)),
        "--cond_temp",
        str(guide_cfg.get("cond_temp", 1000)),
        "--cond_model",
        str(guide_cfg.get("cond_model", "squared")),
        "--N",
        str(guide_cfg.get("n_per_condition", 3)),
        "--bs",
        str(guide_cfg.get("batch_size", 24)),
        "--num_workers",
        str(guide_cfg.get("num_workers", 4)),
        "--guide_fromto",
        f"{guide_cfg.get('guide_fromto', [500, 1000])[0]},{guide_cfg.get('guide_fromto', [500, 1000])[1]}",
        "--dps_weight",
        str(guide_cfg.get("dps_weight", 0.333)),
        "--metrics_for",
        str(guide_cfg.get("metrics_for", "sdf")),
        "--genfull_metrics",
        "--cond_per_metrics",
    ]
    if guide_cfg.get("skip_visualization", False):
        command.append("--skip_visualization")
    if guide_cfg.get("skip_retrieval_exports", False):
        command.append("--skip_retrieval_exports")
    if args.force:
        command.append("--remove")

    env = os.environ.copy()
    existing = env.get("LD_LIBRARY_PATH", "")
    prefixes = [
        "/root/anaconda3/envs/physnap/lib",
        "/root/anaconda3/envs/physnap/lib/python3.9/site-packages/torch/lib",
    ]
    env["LD_LIBRARY_PATH"] = ":".join(prefixes + ([existing] if existing else []))

    with log_path.open("a") as log_handle:
        proc = subprocess.run(command, cwd=PHYSNAP_ROOT / "eval", env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True)

    stats_path = run_dir / "stats.json"
    config_json_path = run_dir / "config.json"
    meta["finished_at"] = now_iso()
    meta["log_path"] = str(log_path)
    meta["run_dir"] = str(run_dir)
    meta["exit_code"] = proc.returncode
    meta["gallery"] = str(sorted(run_dir.glob("res_chnk*.png"))[-1]) if run_dir.exists() and list(run_dir.glob("res_chnk*.png")) else None
    meta["stats_path"] = str(stats_path)
    meta["config_json_path"] = str(config_json_path)
    if proc.returncode == 0 and stats_path.exists() and config_json_path.exists():
        meta["status"] = "completed"
    else:
        meta["status"] = "failed"
    save_json(meta_path, meta)

    if proc.returncode != 0 or meta["status"] != "completed":
        raise SystemExit(proc.returncode or 2)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
