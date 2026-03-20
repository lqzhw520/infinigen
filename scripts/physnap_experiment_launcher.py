#!/usr/bin/env python3
"""Launch and manage PhysNAP box-prior experiments on the A800 host."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(os.environ.get("INFINIGEN_PROJECT_ROOT", "/mnt/afs2/zhuhaowu/infinigen")).resolve()
PHYSNAP_ROOT = PROJECT_ROOT / "external" / "physnap"
CAMPAIGN_DIR = Path(os.environ.get("INFINIGEN_CAMPAIGN_DIR", str(PROJECT_ROOT / "experiments" / "physnap" / "box_prior_v1"))).resolve()
RUNTIME_DIR = CAMPAIGN_DIR / "runtime"
LOCK_PATH = RUNTIME_DIR / "active_train_job.json"
CONDA_SH = Path("/root/anaconda3/etc/profile.d/conda.sh")
INFINIGEN_PYTHON = Path("/root/anaconda3/envs/infinigen/bin/python")
PHYSNAP_PYTHON = Path("/root/anaconda3/envs/physnap/bin/python")
INFINIGEN_PREFIX = "/root/anaconda3/envs/infinigen"
PHYSNAP_PREFIX = "/root/anaconda3/envs/physnap"
GIT_CANDIDATES = [
    Path("/root/anaconda3/envs/infinigen/bin/git"),
    Path("/usr/bin/git"),
    Path("/bin/git"),
]

DEFAULT_PREPARE_DIR = PHYSNAP_ROOT / "data" / "infinigen_graph_combined_k10"
DEFAULT_MIXED_REPLAY_DIR = PHYSNAP_ROOT / "data" / "partnet_infinigen_mixed_k10"
DEFAULT_AE_CHECKPOINT = PHYSNAP_ROOT / "log" / "s1.5_partshape_ae" / "checkpoint" / "737.pt"
DEFAULT_PHASE2_CAMPAIGN_DIR = PROJECT_ROOT / "experiments" / "physnap" / "box_conditioning_v2"
BUILD_CONDITIONING_SCRIPT = PROJECT_ROOT / "scripts" / "build_conditioning_v2_dataset.py"
RUN_CONDITIONING_GROUP_SCRIPT = PROJECT_ROOT / "scripts" / "run_conditioning_v2_group.py"

BOX_DATA_SOURCES = [
    {
        "key": "mailer",
        "box_type": "MAILER",
        "input_dir": PROJECT_ROOT / "sim_exports" / "data_engine" / "phase1_1k_mailer" / "dataset" / "train",
        "output_dir": PHYSNAP_ROOT / "data" / "infinigen_graph_mailer",
    },
    {
        "key": "drawer",
        "box_type": "DRAWER",
        "input_dir": PROJECT_ROOT / "sim_exports" / "data_engine" / "phase1_1k_drawer" / "dataset" / "train",
        "output_dir": PHYSNAP_ROOT / "data" / "infinigen_graph_drawer",
    },
    {
        "key": "slip_lid",
        "box_type": "SLIP_LID",
        "input_dir": PROJECT_ROOT / "sim_exports" / "data_engine" / "phase1_1k_sliplid" / "dataset" / "train",
        "output_dir": PHYSNAP_ROOT / "data" / "infinigen_graph_sliplid",
    },
    {
        "key": "tuck_end",
        "box_type": "TUCK_END",
        "input_dir": PROJECT_ROOT / "sim_exports" / "data_engine" / "phase1_1k_tuckend" / "dataset" / "train",
        "output_dir": PHYSNAP_ROOT / "data" / "infinigen_graph_tuckend",
    },
]

BOX_REFERENCE_PATTERNS = [
    "infinigen_box_k10_val",
    "infinigen_box_k10_val_eval*",
]

EXPERIMENT_SPECS = {
    "infinigen_k10": {
        "config_path": PHYSNAP_ROOT / "configs" / "nap" / "v6.1_diffusion_infinigen_k10.yaml",
        "log_dir": "v6.1_diffusion_infinigen_k10",
        "source_checkpoint": None,
    },
    "infinigen_k10_fixed": {
        "config_path": PHYSNAP_ROOT / "configs" / "nap" / "v6.1_diffusion_infinigen_k10_fixed.yaml",
        "log_dir": "v6.1_diffusion_infinigen_k10_fixed",
        "source_checkpoint": None,
    },
    "finetune_k10": {
        "config_path": PHYSNAP_ROOT / "configs" / "nap" / "v6.1_diffusion_finetune_infinigen_k10.yaml",
        "log_dir": "v6.1_diffusion_finetune_infinigen_k10",
        "source_checkpoint": str(
            PHYSNAP_ROOT / "log" / "v6.1_diffusion_adapted" / "checkpoint" / "5455.pt"
        ),
    },
    "finetune_k10_fixed": {
        "config_path": PHYSNAP_ROOT / "configs" / "nap" / "v6.1_diffusion_finetune_infinigen_k10_fixed.yaml",
        "log_dir": "v6.1_diffusion_finetune_infinigen_k10_fixed",
        "source_checkpoint": str(
            PHYSNAP_ROOT / "log" / "v6.1_diffusion_adapted" / "checkpoint" / "5455.pt"
        ),
    },
    "mixed_finetune_k10": {
        "config_path": PHYSNAP_ROOT / "configs" / "nap" / "v6.1_diffusion_mixed_finetune_infinigen_k10.yaml",
        "log_dir": "v6.1_diffusion_mixed_finetune_infinigen_k10",
        "source_checkpoint": str(
            PHYSNAP_ROOT / "log" / "v6.1_diffusion_adapted" / "checkpoint" / "5455.pt"
        ),
    },
    "partial_finetune_k10": {
        "config_path": PHYSNAP_ROOT / "configs" / "nap" / "v6.1_diffusion_partial_finetune_infinigen_k10.yaml",
        "log_dir": "v6.1_diffusion_partial_finetune_infinigen_k10",
        "source_checkpoint": str(
            PHYSNAP_ROOT / "log" / "v6.1_diffusion_adapted" / "checkpoint" / "5455.pt"
        ),
    },
    "baseline_eval_only": {
        "config_path": PHYSNAP_ROOT / "configs" / "nap" / "v6.1_diffusion_adapted.yaml",
        "log_dir": "v6.1_diffusion_adapted",
        "source_checkpoint": str(
            PHYSNAP_ROOT / "log" / "v6.1_diffusion_adapted" / "checkpoint" / "5455.pt"
        ),
    },
}

CONDITIONING_EXPORT_MODES = {
    "export_cond_zero_singleview": "zero_singleview",
    "export_cond_zero_multiview": "zero_multiview",
    "export_cond_multistate_singleview": "multistate_singleview",
    "export_cond_multistate_multiview": "multistate_multiview",
}

CONDITIONING_GUIDED_MODES = {
    "run_guided_zero_singleview": "zero_singleview",
    "run_guided_zero_multiview": "zero_multiview",
    "run_guided_multistate_singleview": "multistate_singleview",
    "run_guided_multistate_multiview": "multistate_multiview",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open() as handle:
        return json.load(handle)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def box_conversions_current() -> bool:
    for spec in BOX_DATA_SOURCES:
        meta_path = spec["output_dir"] / "conversion_meta.json"
        split_path = spec["output_dir"] / "infinigen_split.json"
        partkeys_path = spec["output_dir"] / "infinigen_partkeys.json"
        if not meta_path.exists() or not split_path.exists() or not partkeys_path.exists():
            return False
        meta = load_json(meta_path, {})
        if not meta.get("visual_origins_applied"):
            return False
        if int(meta.get("max_K", 0)) < 10:
            return False
    return True


def rebuild_box_graph_datasets(force: bool = False) -> list[dict]:
    summaries = []
    for spec in BOX_DATA_SOURCES:
        output_dir = spec["output_dir"]
        if output_dir.exists() and force:
            remove_tree(output_dir)
        meta_path = output_dir / "conversion_meta.json"
        if output_dir.exists() and not force and meta_path.exists():
            meta = load_json(meta_path, {})
            if meta.get("visual_origins_applied") and int(meta.get("max_K", 0)) >= 10:
                summaries.append({"box_type": spec["key"], "status": "cached", "output_dir": str(output_dir)})
                continue
        run_foreground(
            [
                str(INFINIGEN_PYTHON),
                str(PROJECT_ROOT / "scripts" / "infinigen_to_nap.py"),
                "--input-dir", str(spec["input_dir"]),
                "--output-dir", str(output_dir),
                "--box-type", spec["box_type"],
                "--max-K", "10",
            ],
            cwd=PROJECT_ROOT,
            env=env_with_conda_lib(INFINIGEN_PREFIX),
        )
        summaries.append({"box_type": spec["key"], "status": "reconverted", "output_dir": str(output_dir)})
    return summaries


def invalidate_box_reference_caches() -> list[str]:
    invalidated = []
    log_root = PHYSNAP_ROOT / "log"
    for pattern in BOX_REFERENCE_PATTERNS:
        for path in sorted(log_root.glob(pattern)):
            if path.is_dir():
                remove_tree(path)
                invalidated.append(str(path))
    return invalidated


def env_with_conda_lib(prefix: str) -> dict:
    env = os.environ.copy()
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{prefix}/lib:{existing}" if existing else f"{prefix}/lib"
    return env


def resolve_git() -> str | None:
    for candidate in GIT_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return None


def git_output(args: list[str]) -> str:
    git_bin = resolve_git()
    if git_bin is None:
        return ""
    try:
        return subprocess.check_output(
            [git_bin, *args],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def load_manifest() -> dict:
    manifest_path = CAMPAIGN_DIR / "manifest.yaml"
    if not manifest_path.exists():
        return {}
    with manifest_path.open() as handle:
        return yaml.safe_load(handle) or {}


def conditioning_manifest() -> dict:
    manifest_path = conditioning_campaign_dir() / "manifest.yaml"
    if not manifest_path.exists():
        return {}
    with manifest_path.open() as handle:
        return yaml.safe_load(handle) or {}


def conditioning_campaign_dir() -> Path:
    override = os.environ.get("INFINIGEN_CONDITIONING_CAMPAIGN_DIR")
    if override:
        return Path(override).resolve()
    if CAMPAIGN_DIR.name == "box_conditioning_v2":
        return CAMPAIGN_DIR
    return DEFAULT_PHASE2_CAMPAIGN_DIR


def clear_stale_lock() -> None:
    if not LOCK_PATH.exists():
        return
    payload = load_json(LOCK_PATH, {})
    pid = payload.get("pid")
    if process_alive(pid):
        return
    LOCK_PATH.unlink()


def active_lock() -> dict | None:
    clear_stale_lock()
    if not LOCK_PATH.exists():
        return None
    payload = load_json(LOCK_PATH, {})
    mode = payload.get("mode")
    if mode in EXPERIMENT_SPECS:
        log_dir = EXPERIMENT_SPECS[mode]["log_dir"]
        canonical_meta = canonical_meta_path(log_dir)
        canonical_log = canonical_launcher_log_path(log_dir)
        changed = False
        if canonical_meta.exists() and payload.get("meta_path") != str(canonical_meta):
            payload["meta_path"] = str(canonical_meta)
            changed = True
        if canonical_log.exists() and payload.get("log_path") != str(canonical_log):
            payload["log_path"] = str(canonical_log)
            changed = True
        if changed:
            save_json(LOCK_PATH, payload)
    return payload


def build_run_meta(mode: str, log_dir: str, config_path: Path, *, source_checkpoint: str | None = None):
    return {
        "mode": mode,
        "log_dir": log_dir,
        "config_path": str(config_path),
        "status": "pending",
        "source_checkpoint": source_checkpoint,
        "commit": git_output(["rev-parse", "HEAD"]) or "unknown",
        "branch": git_output(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown",
        "created_at": now_iso(),
    }


def experiment_runtime_dir(log_dir: str) -> Path:
    return RUNTIME_DIR / log_dir


def canonical_meta_path(log_dir: str) -> Path:
    return experiment_runtime_dir(log_dir) / "run_meta.json"


def canonical_resolved_config_path(log_dir: str) -> Path:
    return experiment_runtime_dir(log_dir) / "resolved_config.yaml"


def canonical_launcher_log_path(log_dir: str) -> Path:
    return experiment_runtime_dir(log_dir) / "launcher.log"


def canonical_exit_code_path(log_dir: str) -> Path:
    return experiment_runtime_dir(log_dir) / "exit_code.txt"


def experiment_log_dir(log_dir: str) -> Path:
    return PHYSNAP_ROOT / "log" / log_dir


def mirror_file(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def persist_run_meta(log_dir: str, meta: dict) -> Path:
    meta_path = canonical_meta_path(log_dir)
    save_json(meta_path, meta)
    log_dir_path = experiment_log_dir(log_dir)
    if log_dir_path.exists():
        save_json(log_dir_path / "run_meta.json", meta)
    return meta_path


def mirror_runtime_artifacts(log_dir: str) -> None:
    log_dir_path = experiment_log_dir(log_dir)
    if not log_dir_path.exists():
        return
    mirror_file(canonical_resolved_config_path(log_dir), log_dir_path / "resolved_config.yaml")
    meta_path = canonical_meta_path(log_dir)
    if meta_path.exists():
        save_json(log_dir_path / "run_meta.json", load_json(meta_path, {}))


def launch_background(mode: str, config_path: Path, log_dir: str, log_path: Path) -> dict:
    if active_lock():
        raise RuntimeError(f"Another training job is already active: {LOCK_PATH}")
    meta = build_run_meta(
        mode,
        log_dir,
        config_path,
        source_checkpoint=EXPERIMENT_SPECS.get(mode, {}).get("source_checkpoint"),
    )
    meta["status"] = "running"
    meta["launched_at"] = now_iso()
    meta["launch_command"] = f"python run.py --config {config_path} -f"
    exit_code_path = canonical_exit_code_path(log_dir)
    if exit_code_path.exists():
        exit_code_path.unlink()
    meta["exit_code_path"] = str(exit_code_path)
    meta_path = persist_run_meta(log_dir, meta)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a")
    shell_cmd = (
        f"source {shlex.quote(str(CONDA_SH))} >/dev/null 2>&1 && "
        "conda activate physnap >/dev/null 2>&1 && "
        "export LD_LIBRARY_PATH=\"$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}\" && "
        f"cd {shlex.quote(str(PHYSNAP_ROOT))} && "
        f"{meta['launch_command']}; "
        "rc=$?; "
        f"printf '%s\\n' \"$rc\" > {shlex.quote(str(exit_code_path))}; "
        "exit \"$rc\""
    )
    proc = subprocess.Popen(
        ["bash", "-lc", shell_cmd],
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    meta["pid"] = proc.pid
    persist_run_meta(log_dir, meta)
    save_json(
        LOCK_PATH,
        {
            "pid": proc.pid,
            "mode": mode,
            "meta_path": str(meta_path),
            "log_path": str(log_path),
            "launched_at": meta["launched_at"],
        },
    )
    handle.close()
    for _ in range(20):
        if experiment_log_dir(log_dir).exists():
            mirror_runtime_artifacts(log_dir)
            break
        time.sleep(0.5)
    return meta


def run_foreground(command: list[str], cwd: Path, env: dict | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True, env=env)


def load_yaml(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def write_resolved_config(config_path: Path, log_dir: str) -> Path:
    config = load_yaml(config_path)
    output_dir = experiment_runtime_dir(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "resolved_config.yaml"
    with resolved_path.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    mirror_runtime_artifacts(log_dir)
    return resolved_path


def run_prepare(force: bool = False) -> dict:
    prepare_dir = DEFAULT_PREPARE_DIR
    if prepare_dir.exists() and not force:
        split_path = prepare_dir / "infinigen_split.json"
        codebook_path = prepare_dir / "infinigen_codebook.npz"
        if split_path.exists() and codebook_path.exists() and box_conversions_current():
            return {
                "status": "cached",
                "box_conversions": [
                    {"box_type": spec["key"], "status": "cached", "output_dir": str(spec["output_dir"])}
                    for spec in BOX_DATA_SOURCES
                ],
                "prepared_dir": str(prepare_dir),
                "updated_at": now_iso(),
            }

    input_dirs = [
        PHYSNAP_ROOT / "data" / "infinigen_graph_mailer",
        PHYSNAP_ROOT / "data" / "infinigen_graph_drawer",
        PHYSNAP_ROOT / "data" / "infinigen_graph_sliplid",
        PHYSNAP_ROOT / "data" / "infinigen_graph_tuckend",
    ]
    box_summaries = rebuild_box_graph_datasets(force=force or not box_conversions_current())
    if prepare_dir.exists() and force:
        remove_tree(prepare_dir)
    prepare_dir.mkdir(parents=True, exist_ok=True)
    invalidated_refs = invalidate_box_reference_caches() if force else []
    run_foreground(
        [
            str(INFINIGEN_PYTHON),
            str(PROJECT_ROOT / "scripts" / "merge_infinigen_nap_datasets.py"),
            "--output-dir",
            str(prepare_dir),
            "--input-dirs",
            *[str(path) for path in input_dirs],
        ],
        cwd=PROJECT_ROOT,
        env=env_with_conda_lib(INFINIGEN_PREFIX),
    )
    run_foreground(
        [
            str(PHYSNAP_PYTHON),
            str(PROJECT_ROOT / "scripts" / "encode_infinigen_shapes.py"),
            "--nap-data-dir",
            str(prepare_dir),
            "--ae-checkpoint",
            str(DEFAULT_AE_CHECKPOINT),
            "--physnap-root",
            str(PHYSNAP_ROOT),
        ],
        cwd=PROJECT_ROOT,
        env=env_with_conda_lib(PHYSNAP_PREFIX),
    )
    summary = {
        "status": "prepared",
        "prepared_dir": str(prepare_dir),
        "box_conversions": box_summaries,
        "invalidated_box_refs": invalidated_refs,
        "updated_at": now_iso(),
    }
    save_json(CAMPAIGN_DIR / "runtime" / "prepare_k10_summary.json", summary)
    return summary


def run_prepare_mixed_replay(force: bool = False) -> dict:
    prepare_dir = DEFAULT_MIXED_REPLAY_DIR
    required = [
        prepare_dir / "mixed_split.json",
        prepare_dir / "mixed_partkeys.json",
        prepare_dir / "mixed_codebook.npz",
    ]
    if prepare_dir.exists() and not force and all(path.exists() for path in required):
        return {
            "status": "cached",
            "prepared_dir": str(prepare_dir),
            "updated_at": now_iso(),
        }

    run_foreground(
        [
            str(INFINIGEN_PYTHON),
            str(PROJECT_ROOT / "scripts" / "build_mixed_replay_dataset.py"),
            "--partnet-data-root",
            str(PHYSNAP_ROOT / "data" / "partnet_mobility_graph_v4"),
            "--partnet-split",
            str(PHYSNAP_ROOT / "resource" / "partnet_m_split.json"),
            "--partnet-partkeys",
            str(PHYSNAP_ROOT / "resource" / "partnet_m_partkeys.json"),
            "--partnet-codebook",
            str(PHYSNAP_ROOT / "resource" / "codebook" / "s1.5_partshape_ae_737.npz"),
            "--infinigen-data-root",
            str(DEFAULT_PREPARE_DIR),
            "--infinigen-split",
            str(DEFAULT_PREPARE_DIR / "infinigen_split.json"),
            "--infinigen-partkeys",
            str(DEFAULT_PREPARE_DIR / "infinigen_partkeys.json"),
            "--infinigen-codebook",
            str(DEFAULT_PREPARE_DIR / "infinigen_codebook.npz"),
            "--output-dir",
            str(prepare_dir),
        ],
        cwd=PROJECT_ROOT,
        env=env_with_conda_lib(INFINIGEN_PREFIX),
    )
    summary = {
        "status": "prepared",
        "prepared_dir": str(prepare_dir),
        "updated_at": now_iso(),
    }
    save_json(CAMPAIGN_DIR / "runtime" / "prepare_mixed_replay_summary.json", summary)
    return summary


def run_prepare_conditioning_v2(force: bool = False) -> dict:
    campaign_dir = conditioning_campaign_dir()
    plan_path = campaign_dir / "runtime" / "conditioning_plan.json"
    if plan_path.exists() and not force:
        payload = load_json(plan_path, {})
        payload["status"] = "cached"
        return payload
    run_foreground(
        [
            str(INFINIGEN_PYTHON),
            str(BUILD_CONDITIONING_SCRIPT),
            "--campaign-dir",
            str(campaign_dir),
            "prepare",
        ],
        cwd=PROJECT_ROOT,
        env=env_with_conda_lib(INFINIGEN_PREFIX),
    )
    payload = load_json(plan_path, {})
    payload["status"] = "prepared"
    save_json(campaign_dir / "runtime" / "prepare_conditioning_v2_summary.json", payload)
    return payload


def run_export_conditioning_group(mode: str, force: bool = False) -> dict:
    group_id = CONDITIONING_EXPORT_MODES[mode]
    campaign_dir = conditioning_campaign_dir()
    manifest = conditioning_manifest()
    group = manifest.get("conditioning", {}).get("groups", {}).get(group_id, {})
    cond_dir = Path(group.get("cond_dir", ""))
    if cond_dir and not cond_dir.is_absolute():
        cond_dir = PROJECT_ROOT / cond_dir
    required = [
        cond_dir / "pcs.npy",
        cond_dir / "metadata.json",
        cond_dir / "categories.txt",
        cond_dir / "object_ids.txt",
    ]
    if cond_dir and cond_dir.exists() and not force and all(path.exists() for path in required):
        payload = load_json(campaign_dir / "runtime" / f"{group_id}_export_summary.json", {})
        if not payload:
            payload = {
                "group_id": group_id,
                "cond_dir": str(cond_dir),
                "status": "cached",
                "updated_at": now_iso(),
            }
        else:
            payload["status"] = "cached"
        return payload
    run_foreground(
        [
            str(INFINIGEN_PYTHON),
            str(BUILD_CONDITIONING_SCRIPT),
            "--campaign-dir",
            str(campaign_dir),
            "export",
            "--group-id",
            group_id,
        ],
        cwd=PROJECT_ROOT,
        env=env_with_conda_lib(INFINIGEN_PREFIX),
    )
    payload = load_json(campaign_dir / "runtime" / f"{group_id}_export_summary.json", {})
    payload["status"] = "completed"
    return payload


def run_guided_conditioning_group(mode: str, force: bool = False) -> dict:
    group_id = CONDITIONING_GUIDED_MODES[mode]
    campaign_dir = conditioning_campaign_dir()
    manifest = conditioning_manifest()
    output_name = manifest.get("conditioning", {}).get("groups", {}).get(group_id, {}).get("output_name")
    if output_name:
        runtime_meta = load_json(campaign_dir / "runtime" / output_name / "run_meta.json", {})
        stats_path = PHYSNAP_ROOT / "log" / output_name / "stats.json"
        run_dir = PHYSNAP_ROOT / "log" / output_name
        if runtime_meta.get("status") in {"completed", "cached"} and stats_path.exists() and not force:
            runtime_meta["status"] = "cached"
            return runtime_meta
        if (runtime_meta.get("status") == "failed" or (run_dir.exists() and not stats_path.exists())) and not force:
            force = True
    command = [
        str(INFINIGEN_PYTHON),
        str(RUN_CONDITIONING_GROUP_SCRIPT),
        "--campaign-dir",
        str(campaign_dir),
        "--group-id",
        group_id,
    ]
    if force:
        command.append("--force")
    run_foreground(command, cwd=PROJECT_ROOT, env=env_with_conda_lib(INFINIGEN_PREFIX))
    payload = load_json(campaign_dir / "runtime" / f"box_conditioning_v2__{group_id}" / "run_meta.json", {})
    if not payload:
        manifest = conditioning_manifest()
        output_name = manifest.get("conditioning", {}).get("groups", {}).get(group_id, {}).get("output_name")
        if output_name:
            payload = load_json(campaign_dir / "runtime" / output_name / "run_meta.json", {})
    payload["status"] = payload.get("status", "completed")
    return payload


def write_phase2_claim_memo() -> dict:
    campaign_dir = conditioning_campaign_dir()
    summary = load_json(campaign_dir / "summary.json", {})
    return {
        "status": "completed",
        "updated_at": now_iso(),
        "claim_verdict": summary.get("claim_verdict"),
        "summary_path": str(campaign_dir / "summary.json"),
    }


def ensure_experiment_ready(mode: str) -> tuple[Path, Path, str]:
    spec = EXPERIMENT_SPECS[mode]
    config_path = Path(spec["config_path"])
    log_dir = spec["log_dir"]
    resolved_path = write_resolved_config(config_path, log_dir)
    log_path = canonical_launcher_log_path(log_dir)
    return config_path, resolved_path, str(log_path)


def launch_train(mode: str, background: bool) -> dict:
    config_path, resolved_path, log_path = ensure_experiment_ready(mode)
    log_dir = EXPERIMENT_SPECS[mode]["log_dir"]
    if background:
        return launch_background(mode, resolved_path, log_dir, Path(log_path))

    meta = build_run_meta(
        mode,
        log_dir,
        resolved_path,
        source_checkpoint=EXPERIMENT_SPECS.get(mode, {}).get("source_checkpoint"),
    )
    meta["status"] = "running"
    meta["launched_at"] = now_iso()
    meta["launch_command"] = f"python run.py --config {resolved_path} -f"
    meta_path = persist_run_meta(log_dir, meta)
    try:
        run_foreground(
            [str(PHYSNAP_PYTHON), "run.py", "--config", str(resolved_path), "-f"],
            cwd=PHYSNAP_ROOT,
            env=env_with_conda_lib(PHYSNAP_PREFIX),
        )
    except subprocess.CalledProcessError as exc:
        meta["status"] = "failed"
        meta["finished_at"] = now_iso()
        meta["exit_code"] = exc.returncode
        persist_run_meta(log_dir, meta)
        raise
    meta["status"] = "completed"
    meta["finished_at"] = now_iso()
    meta["exit_code"] = 0
    persist_run_meta(log_dir, meta)
    mirror_runtime_artifacts(log_dir)
    return meta


def baseline_eval_only() -> dict:
    mode = "baseline_eval_only"
    spec = EXPERIMENT_SPECS[mode]
    config_path, resolved_path, _ = ensure_experiment_ready(mode)
    meta = build_run_meta(mode, spec["log_dir"], resolved_path, source_checkpoint=spec["source_checkpoint"])
    meta["status"] = "reference"
    meta["updated_at"] = now_iso()
    meta["checkpoint_path"] = spec["source_checkpoint"]
    persist_run_meta(spec["log_dir"], meta)
    mirror_runtime_artifacts(spec["log_dir"])
    return meta


def read_exit_code(log_dir: str) -> int | None:
    path = canonical_exit_code_path(log_dir)
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def print_status() -> None:
    clear_stale_lock()
    current_lock = active_lock()
    payload = {
        "updated_at": now_iso(),
        "active_train_job": current_lock,
        "experiments": {},
    }
    for mode, spec in EXPERIMENT_SPECS.items():
        meta_path = canonical_meta_path(spec["log_dir"])
        if not meta_path.exists():
            meta_path = experiment_log_dir(spec["log_dir"]) / "run_meta.json"
        meta = load_json(meta_path, {"status": "missing"})
        if meta.get("status") == "missing" and current_lock and current_lock.get("mode") == mode:
            resolved_path = canonical_resolved_config_path(spec["log_dir"])
            config_for_meta = resolved_path if resolved_path.exists() else Path(spec["config_path"])
            meta = build_run_meta(
                mode,
                spec["log_dir"],
                config_for_meta,
                source_checkpoint=spec.get("source_checkpoint"),
            )
            meta["status"] = "running"
            meta["pid"] = current_lock.get("pid")
            meta["launched_at"] = current_lock.get("launched_at")
            meta["launch_command"] = f"python run.py --config {config_for_meta} -f"
            meta["recovered_from_lock"] = True
            persist_run_meta(spec["log_dir"], meta)
        pid = meta.get("pid")
        if meta.get("status") == "running" and (not pid or not process_alive(pid)):
            exit_code = read_exit_code(spec["log_dir"])
            if exit_code == 0:
                meta["status"] = "completed"
            elif exit_code is None:
                meta["status"] = "stopped"
            else:
                meta["status"] = "failed"
                meta["exit_code"] = exit_code
            meta["finished_at"] = now_iso()
            persist_run_meta(spec["log_dir"], meta)
        mirror_runtime_artifacts(spec["log_dir"])
        payload["experiments"][mode] = meta
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch PhysNAP box-prior experiments")
    parser.add_argument(
        "mode",
        choices=[
            "prepare_k10",
            "prepare_mixed_replay",
            "prepare_conditioning_v2",
            "export_cond_zero_singleview",
            "export_cond_zero_multiview",
            "export_cond_multistate_singleview",
            "export_cond_multistate_multiview",
            "run_guided_zero_singleview",
            "run_guided_zero_multiview",
            "run_guided_multistate_singleview",
            "run_guided_multistate_multiview",
            "write_phase2_claim_memo",
            "infinigen_k10",
            "infinigen_k10_fixed",
            "finetune_k10",
            "finetune_k10_fixed",
            "mixed_finetune_k10",
            "partial_finetune_k10",
            "baseline_eval_only",
            "status",
        ],
    )
    parser.add_argument("--background", action="store_true", help="Launch training in background")
    parser.add_argument("--force", action="store_true", help="Force data preparation regeneration")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "status":
        print_status()
        return 0
    if args.mode == "prepare_k10":
        print(json.dumps(run_prepare(force=args.force), indent=2, ensure_ascii=False))
        return 0
    if args.mode == "prepare_mixed_replay":
        print(json.dumps(run_prepare_mixed_replay(force=args.force), indent=2, ensure_ascii=False))
        return 0
    if args.mode == "prepare_conditioning_v2":
        print(json.dumps(run_prepare_conditioning_v2(force=args.force), indent=2, ensure_ascii=False))
        return 0
    if args.mode in CONDITIONING_EXPORT_MODES:
        print(json.dumps(run_export_conditioning_group(args.mode, force=args.force), indent=2, ensure_ascii=False))
        return 0
    if args.mode in CONDITIONING_GUIDED_MODES:
        print(json.dumps(run_guided_conditioning_group(args.mode, force=args.force), indent=2, ensure_ascii=False))
        return 0
    if args.mode == "write_phase2_claim_memo":
        print(json.dumps(write_phase2_claim_memo(), indent=2, ensure_ascii=False))
        return 0
    if args.mode == "baseline_eval_only":
        print(json.dumps(baseline_eval_only(), indent=2, ensure_ascii=False))
        return 0

    if args.mode in {"infinigen_k10", "infinigen_k10_fixed", "finetune_k10", "finetune_k10_fixed", "mixed_finetune_k10", "partial_finetune_k10"}:
        print(json.dumps(launch_train(args.mode, background=args.background), indent=2, ensure_ascii=False))
        return 0

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
