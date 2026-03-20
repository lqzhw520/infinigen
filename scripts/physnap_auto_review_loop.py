#!/usr/bin/env python3
"""Phase-aware auto-review loop for Infinigen x PhysNAP research campaigns."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(os.environ.get("INFINIGEN_PROJECT_ROOT", "/mnt/afs2/zhuhaowu/infinigen")).resolve()
CAMPAIGN_DIR = Path(
    os.environ.get(
        "INFINIGEN_CAMPAIGN_DIR",
        str(PROJECT_ROOT / "experiments" / "physnap" / "box_prior_v1"),
    )
).resolve()
TRAIN_WRAPPER = PROJECT_ROOT / "scripts" / "train_physnap_infinigen.sh"
RECORD_ITERATION = (
    PROJECT_ROOT / ".cursor" / "skills" / "infinigen-project-memory" / "scripts" / "record_iteration.py"
)
UPDATE_STATUS = (
    PROJECT_ROOT / ".cursor" / "skills" / "infinigen-project-memory" / "scripts" / "update_status.py"
)
WRITE_CAMPAIGN_HISTORY = (
    PROJECT_ROOT / ".cursor" / "skills" / "infinigen-project-memory" / "scripts" / "write_campaign_history.py"
)
RENDER_DECISION_MEMO = (
    PROJECT_ROOT / ".cursor" / "skills" / "infinigen-scientific-dev" / "scripts" / "render_decision_memo.py"
)
PHYSNAP_PYTHON = Path("/root/anaconda3/envs/physnap/bin/python")
INFINIGEN_PYTHON = Path("/root/anaconda3/envs/infinigen/bin/python")
RUNTIME_DIR = CAMPAIGN_DIR / "runtime"
EVALUATION_DIR = CAMPAIGN_DIR / "evaluation"
WATCH_PID_PATH = RUNTIME_DIR / "auto_loop_watch.pid"
WATCH_STATUS_PATH = RUNTIME_DIR / "watch_status.json"
EVAL_PROGRESS_PATH = EVALUATION_DIR / "progress.json"
EVALUATE_LOG_PATH = RUNTIME_DIR / "evaluate_all.log"
DECISION_MEMO_PATH = CAMPAIGN_DIR / "decision_memo.md"
ITERATION_MARKER = RUNTIME_DIR / "last_iteration_review.txt"
STATUS_SYNC_MARKER = RUNTIME_DIR / "last_status_sync.txt"

TRAIN_STEP_TO_MODE = {
    "train_infinigen_k10": "infinigen_k10",
    "train_finetune_k10": "finetune_k10",
    "train_mixed_finetune_k10": "mixed_finetune_k10",
    "train_partial_finetune_k10": "partial_finetune_k10",
}
PREPARE_STEPS = {
    "prepare_k10": "prepare_k10",
    "prepare_mixed_replay": "prepare_mixed_replay",
    "prepare_conditioning_v2": "prepare_conditioning_v2",
    "export_cond_zero_singleview": "export_cond_zero_singleview",
    "export_cond_zero_multiview": "export_cond_zero_multiview",
    "export_cond_multistate_singleview": "export_cond_multistate_singleview",
    "export_cond_multistate_multiview": "export_cond_multistate_multiview",
    "run_guided_zero_singleview": "run_guided_zero_singleview",
    "run_guided_zero_multiview": "run_guided_zero_multiview",
    "run_guided_multistate_singleview": "run_guided_multistate_singleview",
    "run_guided_multistate_multiview": "run_guided_multistate_multiview",
    "write_phase2_claim_memo": "write_phase2_claim_memo",
}
PHASE2_EXPORT_STEP_TO_GROUP = {
    "export_cond_zero_singleview": "zero_singleview",
    "export_cond_zero_multiview": "zero_multiview",
    "export_cond_multistate_singleview": "multistate_singleview",
    "export_cond_multistate_multiview": "multistate_multiview",
}
PHASE2_GUIDED_STEP_TO_GROUP = {
    "run_guided_zero_singleview": "zero_singleview",
    "run_guided_zero_multiview": "zero_multiview",
    "run_guided_multistate_singleview": "multistate_singleview",
    "run_guided_multistate_multiview": "multistate_multiview",
}
BASELINE_KEY = "baseline_eval_only"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open() as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        text = path.read_text(errors="ignore")
        payload = extract_json_payload(text)
        if payload is None:
            return default
        return payload


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def extract_json_payload(text: str):
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    candidate_starts = [idx for idx, ch in enumerate(text) if ch in "[{"]
    for start in reversed(candidate_starts):
        candidate = text[start:].strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def process_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def load_manifest() -> dict:
    with (CAMPAIGN_DIR / "manifest.yaml").open() as handle:
        return yaml.safe_load(handle) or {}


def load_acceptance() -> dict:
    return load_json(CAMPAIGN_DIR / "acceptance_criteria.json", {})


def runtime_meta(manifest: dict) -> dict:
    return manifest.get("runtime", {}) or {}


def evaluator_script(manifest: dict) -> Path:
    path_str = runtime_meta(manifest).get("evaluator", "scripts/evaluate_physnap_training.py")
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def evaluator_python(manifest: dict) -> Path:
    env_name = runtime_meta(manifest).get("evaluator_python", "physnap")
    return INFINIGEN_PYTHON if env_name == "infinigen" else PHYSNAP_PYTHON


def evaluation_step_id(state: dict) -> str | None:
    for item in state.get("queue", []):
        if str(item.get("id", "")).startswith("evaluate_") or item.get("id") == "evaluate_all":
            return item["id"]
    return None


def campaign_meta(manifest: dict) -> dict:
    return manifest.get("campaign", {})


def campaign_title(manifest: dict) -> str:
    return campaign_meta(manifest).get("title", CAMPAIGN_DIR.name)


def campaign_claim(manifest: dict) -> str:
    return campaign_meta(manifest).get("main_claim", "")


def campaign_phase(manifest: dict, state: dict | None = None) -> str:
    if state and state.get("phase"):
        return state["phase"]
    return campaign_meta(manifest).get("phase", "phase0_substrate")


def campaign_gate(manifest: dict, state: dict | None = None, review: dict | None = None) -> str:
    if review and review.get("phase_gate"):
        return review["phase_gate"]
    if state and state.get("phase_gate"):
        return state["phase_gate"]
    return campaign_meta(manifest).get("phase_gate", "await_review")


def enabled_experiment_ids(manifest: dict) -> list[str]:
    experiments = manifest.get("experiments", {}) or {}
    output = []
    for experiment_id, spec in experiments.items():
        if spec.get("enabled", True) is False:
            continue
        output.append(experiment_id)
    return output


def launcher_key_for_experiment(experiment_id: str) -> str:
    return BASELINE_KEY if experiment_id == "baseline" else experiment_id


def project_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    prefix = stem.split("_", 1)[0]
    try:
        return int(prefix), stem
    except ValueError:
        return -1, stem


def latest_checkpoint_in_dir(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    latest_named = sorted(checkpoint_dir.glob("*_latest.pt"), key=checkpoint_sort_key)
    if latest_named:
        return latest_named[-1]
    numbered = sorted(checkpoint_dir.glob("*.pt"), key=checkpoint_sort_key)
    return numbered[-1] if numbered else None


def resolve_experiment_artifacts(experiment_id: str, spec: dict | None) -> tuple[str | None, str | None]:
    if not spec:
        return None, None
    config_path = project_path(spec.get("config_path"))
    checkpoint_path = project_path(spec.get("checkpoint_path"))
    if checkpoint_path and checkpoint_path.exists():
        return str(checkpoint_path), str(config_path) if config_path else None

    log_dir = project_path(spec.get("log_dir"))
    if log_dir:
        checkpoint_dir = log_dir / "checkpoint"
        latest_checkpoint = latest_checkpoint_in_dir(checkpoint_dir)
        if latest_checkpoint:
            return str(latest_checkpoint), str(config_path) if config_path else None

    runtime_log_dir = Path(spec.get("log_dir", "")).name
    if runtime_log_dir:
        runtime_meta = CAMPAIGN_DIR / "runtime" / runtime_log_dir / "run_meta.json"
        payload = load_json(runtime_meta, {})
        meta_checkpoint = project_path(payload.get("checkpoint_path"))
        if meta_checkpoint and meta_checkpoint.exists():
            return str(meta_checkpoint), str(config_path) if config_path else None

    return str(checkpoint_path) if checkpoint_path else None, str(config_path) if config_path else None


def load_state() -> dict:
    state = load_json(CAMPAIGN_DIR / "state.json", {})
    if state:
        state.setdefault("retries", {})
        state.setdefault("history", [])
        state.setdefault("force_steps", {})
        state.setdefault("auto_handoff_status", "waiting")
        state.setdefault("phase2_base_checkpoint", None)
        state.setdefault("phase2_base_config", None)
        state.setdefault("phase2_base_experiment", None)
        state.setdefault("active_runtime", None)
        return state
    manifest = load_manifest()
    queue = [{"id": step["id"], "status": "pending", "updated_at": now_iso()} for step in manifest.get("queue", [])]
    return {
        "campaign_id": campaign_meta(manifest).get("id", CAMPAIGN_DIR.name),
        "phase": campaign_phase(manifest),
        "phase_gate": campaign_gate(manifest),
        "auto_handoff_status": "waiting",
        "phase2_base_checkpoint": None,
        "phase2_base_config": None,
        "phase2_base_experiment": None,
        "active_runtime": None,
        "updated_at": now_iso(),
        "queue": queue,
        "active_job": None,
        "history": [],
        "last_review": None,
        "retries": {},
        "force_steps": {},
    }


def save_state(state: dict) -> None:
    state["updated_at"] = now_iso()
    save_json(CAMPAIGN_DIR / "state.json", state)


def queue_item(state: dict, step_id: str) -> dict | None:
    for item in state.get("queue", []):
        if item.get("id") == step_id:
            return item
    return None


def set_active_runtime(
    state: dict,
    *,
    step_id: str,
    worker_kind: str,
    attempt_no: int,
    launched_at: str,
    worker_pid: int | None = None,
) -> None:
    state["active_runtime"] = {
        "step_id": step_id,
        "worker_kind": worker_kind,
        "worker_pid": worker_pid,
        "attempt_no": attempt_no,
        "launched_at": launched_at,
        "heartbeat_at": now_iso(),
    }


def clear_active_runtime(state: dict) -> None:
    state["active_runtime"] = None


def load_watch_status() -> dict:
    return load_json(
        WATCH_STATUS_PATH,
        {
            "supervisor_pid": None,
            "inner_watcher_pid": None,
            "watcher_pid": None,
            "worker_pid": None,
            "cpu_worker_pid": None,
            "mode": None,
            "queue_step": None,
            "stage": None,
            "status": "idle",
            "started_at": None,
            "last_heartbeat": None,
            "latest_artifact": None,
            "active_gpu_step": None,
            "active_cpu_step": None,
            "last_repair_action": None,
            "last_error": None,
        },
    )


def save_watch_status(payload: dict) -> None:
    payload["last_heartbeat"] = now_iso()
    save_json(WATCH_STATUS_PATH, payload)


def update_watch_status(
    *,
    mode: str | None = None,
    queue_step: str | None = None,
    stage: str | None = None,
    status: str | None = None,
    worker_pid: int | None = None,
    clear_worker: bool = False,
    latest_artifact: str | None = None,
    active_gpu_step: str | None = None,
    active_cpu_step: str | None = None,
    last_repair_action: str | None = None,
    last_error: str | None = None,
) -> dict:
    payload = load_watch_status()
    payload["watcher_pid"] = os.getpid()
    payload["inner_watcher_pid"] = os.getpid()
    if payload.get("started_at") is None:
        payload["started_at"] = now_iso()
    if mode is not None:
        payload["mode"] = mode
    if queue_step is not None:
        payload["queue_step"] = queue_step
    if stage is not None:
        payload["stage"] = stage
    if status is not None:
        payload["status"] = status
    if clear_worker:
        payload["worker_pid"] = None
    elif worker_pid is not None or worker_pid == 0:
        payload["worker_pid"] = worker_pid
    if latest_artifact is not None:
        payload["latest_artifact"] = latest_artifact
    if active_gpu_step is not None:
        payload["active_gpu_step"] = active_gpu_step
    if active_cpu_step is not None:
        payload["active_cpu_step"] = active_cpu_step
    if last_repair_action is not None:
        payload["last_repair_action"] = last_repair_action
    if last_error is not None:
        payload["last_error"] = last_error
    elif status in {"running", "completed"}:
        payload["last_error"] = None
    save_watch_status(payload)
    return payload


def load_eval_progress() -> dict:
    return load_json(EVAL_PROGRESS_PATH, {})


def claim_watcher_lock() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    existing_pid = None
    if WATCH_PID_PATH.exists():
        try:
            existing_pid = int(WATCH_PID_PATH.read_text().strip())
        except Exception:
            existing_pid = None
    if existing_pid and existing_pid != os.getpid() and process_alive(existing_pid):
        raise SystemExit(f"Watcher already running under pid {existing_pid}")
    WATCH_PID_PATH.write_text(f"{os.getpid()}\n")
    existing_watch = load_watch_status()
    save_watch_status(
        {
            "supervisor_pid": existing_watch.get("supervisor_pid"),
            "inner_watcher_pid": os.getpid(),
            "watcher_pid": os.getpid(),
            "worker_pid": existing_watch.get("worker_pid"),
            "cpu_worker_pid": existing_watch.get("cpu_worker_pid"),
            "mode": "loop",
            "queue_step": existing_watch.get("queue_step"),
            "stage": existing_watch.get("stage"),
            "status": "starting",
            "started_at": existing_watch.get("started_at") or now_iso(),
            "last_heartbeat": now_iso(),
            "latest_artifact": existing_watch.get("latest_artifact"),
            "active_gpu_step": existing_watch.get("active_gpu_step"),
            "active_cpu_step": existing_watch.get("active_cpu_step"),
            "last_repair_action": existing_watch.get("last_repair_action"),
            "last_error": existing_watch.get("last_error"),
        }
    )


def release_watcher_lock(final_status: str, error: str | None = None) -> None:
    update_watch_status(status=final_status, last_error=error, clear_worker=True)
    try:
        current = int(WATCH_PID_PATH.read_text().strip())
    except Exception:
        current = None
    if current == os.getpid() and WATCH_PID_PATH.exists():
        WATCH_PID_PATH.unlink()


def subprocess_env() -> dict:
    env = os.environ.copy()
    env["INFINIGEN_PROJECT_ROOT"] = str(PROJECT_ROOT)
    env["INFINIGEN_CAMPAIGN_DIR"] = str(CAMPAIGN_DIR)
    return env


def train_status() -> dict:
    output = subprocess.check_output([str(TRAIN_WRAPPER), "status"], cwd=PROJECT_ROOT, env=subprocess_env(), text=True)
    payload = extract_json_payload(output)
    if payload is None:
        raise RuntimeError("Unable to parse launcher status JSON")
    return payload


def launch_train(mode: str) -> dict:
    output = subprocess.check_output(
        [str(TRAIN_WRAPPER), mode, "--background"],
        cwd=PROJECT_ROOT,
        env=subprocess_env(),
        text=True,
    )
    payload = extract_json_payload(output)
    if payload is None:
        raise RuntimeError(f"Unable to parse launcher output for {mode}")
    return payload


def run_prepare(mode: str, force: bool = False, on_start=None, on_poll=None, poll_seconds: int = 15) -> dict:
    command = [str(TRAIN_WRAPPER), mode]
    if force:
        command.append("--force")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if on_start:
        on_start(process.pid)
    while process.poll() is None:
        if on_poll:
            on_poll(process.pid)
        time.sleep(poll_seconds)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command, output=stdout, stderr=stderr)
    payload = extract_json_payload(stdout)
    if payload is None:
        raise RuntimeError(f"Unable to parse prepare output for {mode}")
    return payload


def run_evaluate(state: dict, launcher: dict, manifest: dict, poll_seconds: int = 30) -> None:
    env = subprocess_env()
    existing = env.get("LD_LIBRARY_PATH", "")
    prefixes = [
        "/root/anaconda3/envs/physnap/lib",
        "/root/anaconda3/envs/physnap/lib/python3.9/site-packages/torch/lib",
    ]
    env["LD_LIBRARY_PATH"] = ":".join(prefixes + ([existing] if existing else []))
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    step_id = evaluation_step_id(state) or "evaluate_all"
    eval_script = evaluator_script(manifest)
    eval_python = evaluator_python(manifest)
    eval_log_path = RUNTIME_DIR / f"{step_id}.log"
    with eval_log_path.open("a") as log_handle:
        process = subprocess.Popen(
            [str(eval_python), str(eval_script), "--campaign-dir", str(CAMPAIGN_DIR)],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        update_watch_status(
            mode=step_id,
            queue_step=step_id,
            stage="launching",
            status="running",
            worker_pid=process.pid,
            latest_artifact=str(eval_log_path),
            active_gpu_step=step_id,
            last_error=None,
        )
        while True:
            progress = load_eval_progress()
            return_code = process.poll()
            update_watch_status(
                mode=step_id,
                queue_step=step_id,
                stage=progress.get("current_stage") or "launching",
                status="running" if return_code is None else progress.get("status", "completed"),
                worker_pid=process.pid if return_code is None else None,
                latest_artifact=progress.get("latest_artifact") or str(eval_log_path),
                active_gpu_step=step_id if return_code is None else None,
                last_error=progress.get("message") if progress.get("status") == "failed" else None,
            )
            sync_evaluation_step(state)
            save_state(state)
            if return_code is not None:
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, process.args)
                update_watch_status(
                    mode=step_id,
                    queue_step=step_id,
                    stage="write_report",
                    status="completed",
                    clear_worker=True,
                    latest_artifact=str(EVALUATION_DIR / "comparison_report.md"),
                    active_gpu_step=None,
                    last_error=None,
                )
                return
            time.sleep(poll_seconds)


def summarize_results() -> dict | None:
    summary_path = EVALUATION_DIR / "comparison_summary.json"
    if not summary_path.exists():
        return None
    payload = load_json(summary_path, {})
    save_json(CAMPAIGN_DIR / "summary.json", payload)
    return payload


def get_experiment(summary: dict | None, key: str) -> dict:
    return ((summary or {}).get("experiments") or {}).get(key, {})


def metric(summary: dict | None, experiment: str, section: str, field: str):
    payload = get_experiment(summary, experiment)
    if section == "physics":
        return payload.get("physics_stats", {}).get(field)
    if section == "partnet":
        return payload.get("partnet_sanity", {}).get("multi_state", {}).get(field)
    if section == "box":
        return payload.get("box_metrics", {}).get("multi_state", {}).get(field)
    return None


def all_required_phase1_arms_present(summary: dict | None) -> bool:
    for key in ["baseline", "infinigen_k10", "finetune_k10", "mixed_finetune_k10", "partial_finetune_k10"]:
        if not get_experiment(summary, key):
            return False
    return True


def candidate_is_effective(summary: dict | None, candidate_key: str, tolerance_factor: float, cov_tolerance_drop: float) -> bool:
    finetune_partnet = metric(summary, "finetune_k10", "partnet", "mmd")
    candidate_partnet = metric(summary, candidate_key, "partnet", "mmd")
    finetune_box_mmd = metric(summary, "finetune_k10", "box", "mmd")
    finetune_box_cov = metric(summary, "finetune_k10", "box", "cov")
    candidate_box_mmd = metric(summary, candidate_key, "box", "mmd")
    candidate_box_cov = metric(summary, candidate_key, "box", "cov")
    if None in {finetune_partnet, candidate_partnet, finetune_box_mmd, finetune_box_cov, candidate_box_mmd, candidate_box_cov}:
        return False
    improves_partnet = candidate_partnet < finetune_partnet
    preserves_box = candidate_box_mmd <= finetune_box_mmd * tolerance_factor and candidate_box_cov >= finetune_box_cov - cov_tolerance_drop
    return improves_partnet and preserves_box


def dominant_phase1_diagnosis(summary: dict | None, acceptance: dict) -> str | None:
    if not all_required_phase1_arms_present(summary):
        return None
    scoring = acceptance.get("scoring", {})
    tolerance_factor = scoring.get("candidate_box_mmd_tolerance_factor", 1.05)
    cov_tolerance_drop = scoring.get("candidate_box_cov_tolerance_drop", 0.05)
    mixed_effective = candidate_is_effective(summary, "mixed_finetune_k10", tolerance_factor, cov_tolerance_drop)
    partial_effective = candidate_is_effective(summary, "partial_finetune_k10", tolerance_factor, cov_tolerance_drop)
    if mixed_effective and not partial_effective:
        return "forgetting-dominant"
    if partial_effective and not mixed_effective:
        return "over-update-dominant"
    if mixed_effective and partial_effective:
        mixed_gain = metric(summary, "finetune_k10", "partnet", "mmd") - metric(summary, "mixed_finetune_k10", "partnet", "mmd")
        partial_gain = metric(summary, "finetune_k10", "partnet", "mmd") - metric(summary, "partial_finetune_k10", "partnet", "mmd")
        return "forgetting-dominant" if mixed_gain >= partial_gain else "over-update-dominant"
    return "parameterization-limited"


def select_phase2_base(summary: dict | None, manifest: dict, diagnosis: str | None) -> tuple[str | None, str | None, str | None]:
    experiments = manifest.get("experiments", {}) or {}
    preferred_by_diagnosis = {
        "forgetting-dominant": ["mixed_finetune_k10", "partial_finetune_k10", "finetune_k10", "infinigen_k10", "baseline"],
        "over-update-dominant": ["partial_finetune_k10", "mixed_finetune_k10", "finetune_k10", "infinigen_k10", "baseline"],
        "parameterization-limited": ["finetune_k10", "infinigen_k10", "baseline"],
    }
    order = preferred_by_diagnosis.get(diagnosis or "", ["finetune_k10", "infinigen_k10", "baseline"])
    for key in order:
        spec = experiments.get(key)
        if not spec:
            continue
        checkpoint, config = resolve_experiment_artifacts(key, spec)
        if checkpoint and config:
            return key, checkpoint, config
    return None, None, None


def phase2_group(summary: dict | None, key: str) -> dict:
    return ((summary or {}).get("conditioning_groups") or {}).get(key, {})


def conditioning_metric(summary: dict | None, group: str, field: str):
    return phase2_group(summary, group).get(field)


def has_required_files(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def conditioning_cfg(manifest: dict) -> dict:
    return manifest.get("conditioning", {}) or {}


def phase2_group_spec(manifest: dict, group_id: str) -> dict:
    return (conditioning_cfg(manifest).get("groups") or {}).get(group_id, {})


def phase2_cond_dir(manifest: dict, group_id: str) -> Path | None:
    spec = phase2_group_spec(manifest, group_id)
    cond_dir = spec.get("cond_dir")
    if not cond_dir:
        return None
    path = Path(cond_dir)
    return path if path.is_absolute() else PROJECT_ROOT / path


def phase2_run_output_name(manifest: dict, group_id: str) -> str | None:
    return phase2_group_spec(manifest, group_id).get("output_name")


def phase2_run_dir(manifest: dict, group_id: str) -> Path | None:
    output_name = phase2_run_output_name(manifest, group_id)
    if not output_name:
        return None
    return PROJECT_ROOT / "external" / "physnap" / "log" / output_name


def phase2_runtime_meta_path(manifest: dict, group_id: str) -> Path | None:
    output_name = phase2_run_output_name(manifest, group_id)
    if not output_name:
        return None
    return CAMPAIGN_DIR / "runtime" / output_name / "run_meta.json"


def invalid_metric(value, *, allow_negative: bool = False) -> bool:
    if value is None:
        return True
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return True
    if math.isnan(numeric) or math.isinf(numeric):
        return True
    if not allow_negative and numeric < 0:
        return True
    return False


def phase2_conditioning_complete(manifest: dict, group_id: str) -> bool:
    cond_dir = phase2_cond_dir(manifest, group_id)
    if not cond_dir:
        return False
    required = [
        cond_dir / "pcs.npy",
        cond_dir / "metadata.json",
        cond_dir / "categories.txt",
        cond_dir / "object_ids.txt",
    ]
    return cond_dir.exists() and has_required_files(required)


def phase2_guided_complete(manifest: dict, group_id: str) -> bool:
    run_dir = phase2_run_dir(manifest, group_id)
    meta_path = phase2_runtime_meta_path(manifest, group_id)
    if not run_dir or not meta_path:
        return False
    stats_path = run_dir / "stats.json"
    config_path = run_dir / "config.json"
    meta = load_json(meta_path, {})
    stats = load_json(stats_path, {})
    return (
        run_dir.exists()
        and has_required_files([stats_path, config_path])
        and meta.get("status") == "completed"
        and phase2_group_metrics_valid(stats)
    )


def newest_artifact_under(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    newest_path = None
    newest_mtime = -1.0
    for root, _, files in os.walk(path):
        for name in files:
            candidate = Path(root) / name
            try:
                mtime = candidate.stat().st_mtime
            except OSError:
                continue
            if mtime > newest_mtime:
                newest_mtime = mtime
                newest_path = candidate
    return str(newest_path) if newest_path else None


def sync_phase2_runtime_steps(state: dict, manifest: dict) -> None:
    if campaign_phase(manifest, state) != "phase2_conditioning_design":
        return
    watch = load_watch_status()
    active_runtime = state.get("active_runtime") or {}
    active_step = active_runtime.get("step_id")
    active_pid = active_runtime.get("worker_pid")

    for step_id, group_id in PHASE2_GUIDED_STEP_TO_GROUP.items():
        item = queue_item(state, step_id)
        meta_path = phase2_runtime_meta_path(manifest, group_id)
        meta = load_json(meta_path, {}) if meta_path else {}
        status = meta.get("status")
        run_dir = phase2_run_dir(manifest, group_id)
        if not item:
            continue
        if item.get("status") == "running" and (status == "running" or (active_step == step_id and process_alive(active_pid))):
            item["updated_at"] = now_iso()
            active = state.get("active_runtime") or {}
            if active.get("step_id") == step_id:
                active["heartbeat_at"] = now_iso()
                state["active_runtime"] = active
            latest_artifact = newest_artifact_under(run_dir) or str(meta_path) if meta_path else None
            update_watch_status(
                mode="loop",
                queue_step=step_id,
                stage=step_id,
                status="running",
                worker_pid=active_pid if process_alive(active_pid) else watch.get("worker_pid"),
                latest_artifact=latest_artifact,
                active_gpu_step=step_id,
                active_cpu_step=None,
                last_error=None,
            )
        if status == "completed" and phase2_guided_complete(manifest, group_id):
            item["status"] = "completed"
            item["updated_at"] = now_iso()
            if active_step == step_id and (not active_pid or not process_alive(active_pid)):
                clear_active_runtime(state)
            if watch.get("queue_step") == step_id and watch.get("status") == "running" and not process_alive(watch.get("worker_pid")):
                update_watch_status(
                    mode="prepare",
                    queue_step=step_id,
                    stage=step_id,
                    status="completed",
                    clear_worker=True,
                    latest_artifact=str((run_dir or Path(".")) / "stats.json"),
                    active_gpu_step=None,
                    active_cpu_step=None,
                    last_error=None,
                )
        elif status == "failed" and item.get("status") == "running":
            item["status"] = "failed"
            item["updated_at"] = now_iso()
            if active_step == step_id and (not active_pid or not process_alive(active_pid)):
                clear_active_runtime(state)


def phase2_group_metrics_valid(group_payload: dict) -> bool:
    required_fields = [
        "genfull_per_mmd_mean",
        "genfull_per_cov_mean",
        "genfull_per_1NN-acc_mean",
        "pen_error_mean",
        "mob_error_mean",
        "cond_error_mean",
    ]
    for field in required_fields:
        if invalid_metric(group_payload.get(field)):
            return False
    return True


def phase2_summary_valid(summary: dict | None, manifest: dict) -> bool:
    if not summary:
        return False
    groups = conditioning_cfg(manifest).get("groups", {}) or {}
    payloads = (summary.get("conditioning_groups") or {})
    if not groups:
        return False
    for group_id in groups:
        payload = payloads.get(group_id)
        if not payload or not phase2_group_metrics_valid(payload):
            return False
    return True


def reset_queue_from_step(state: dict, step_id: str) -> None:
    reset = False
    for item in state.get("queue", []):
        if item.get("id") == step_id:
            reset = True
        if not reset:
            continue
        item["status"] = "pending"
        item["updated_at"] = now_iso()


def phase2_source_campaign_dir(manifest: dict) -> Path | None:
    source_campaign = conditioning_cfg(manifest).get("source_campaign")
    if not source_campaign:
        return None
    return PROJECT_ROOT / "experiments" / "physnap" / source_campaign


def repair_phase2_base_selection(state: dict, manifest: dict) -> bool:
    if campaign_phase(manifest, state) != "phase2_conditioning_design":
        return False

    changed = False
    experiment_id = state.get("phase2_base_experiment")
    checkpoint_path = project_path(state.get("phase2_base_checkpoint"))
    config_path = project_path(state.get("phase2_base_config"))
    checkpoint_ok = checkpoint_path.exists() if checkpoint_path else False
    config_ok = config_path.exists() if config_path else False

    if checkpoint_ok and config_ok:
        normalized_checkpoint = str(checkpoint_path)
        normalized_config = str(config_path)
        if state.get("phase2_base_checkpoint") != normalized_checkpoint:
            state["phase2_base_checkpoint"] = normalized_checkpoint
            changed = True
        if state.get("phase2_base_config") != normalized_config:
            state["phase2_base_config"] = normalized_config
            changed = True
        return changed

    source_campaign_dir = phase2_source_campaign_dir(manifest)
    source_manifest = load_json(source_campaign_dir / "state.json", {}) if source_campaign_dir else {}
    source_phase_manifest = None
    if source_campaign_dir and (source_campaign_dir / "manifest.yaml").exists():
        with (source_campaign_dir / "manifest.yaml").open() as handle:
            source_phase_manifest = yaml.safe_load(handle) or {}

    if not experiment_id and source_manifest.get("phase2_base_experiment"):
        experiment_id = source_manifest.get("phase2_base_experiment")
        state["phase2_base_experiment"] = experiment_id
        changed = True

    if experiment_id and source_phase_manifest:
        experiments = source_phase_manifest.get("experiments", {}) or {}
        repaired_checkpoint, repaired_config = resolve_experiment_artifacts(experiment_id, experiments.get(experiment_id))
        if repaired_checkpoint and repaired_config:
            if state.get("phase2_base_checkpoint") != repaired_checkpoint:
                state["phase2_base_checkpoint"] = repaired_checkpoint
                changed = True
            if state.get("phase2_base_config") != repaired_config:
                state["phase2_base_config"] = repaired_config
                changed = True

    if changed:
        for retry_key in list(state.setdefault("retries", {}).keys()):
            if "missing_checkpoint" in retry_key:
                state["retries"].pop(retry_key, None)
        for item in state.get("queue", []):
            if item.get("id") in PHASE2_GUIDED_STEP_TO_GROUP and item.get("status") != "completed":
                reset_queue_from_step(state, item["id"])
                state.setdefault("force_steps", {})[item["id"]] = True
                append_history(
                    state,
                    "phase2_auto_repair",
                    f"Repaired Phase 2 base checkpoint/config for `{state.get('phase2_base_experiment')}` and reset `{item['id']}`.",
                )
                break
    return changed


def repair_phase2_contract(state: dict, manifest: dict) -> bool:
    if campaign_phase(manifest, state) != "phase2_conditioning_design":
        return False
    if repair_phase2_base_selection(state, manifest):
        return True
    changed = False
    force_steps = state.setdefault("force_steps", {})

    for step_id, group_id in PHASE2_EXPORT_STEP_TO_GROUP.items():
        item = queue_item(state, step_id)
        if item and item.get("status") == "completed" and not phase2_conditioning_complete(manifest, group_id):
            reset_queue_from_step(state, step_id)
            force_steps[step_id] = True
            append_history(state, "phase2_auto_repair", f"Reset `{step_id}` because conditioning artifacts for `{group_id}` were incomplete.")
            changed = True
            return True

    for step_id, group_id in PHASE2_GUIDED_STEP_TO_GROUP.items():
        item = queue_item(state, step_id)
        if item and item.get("status") == "completed" and not phase2_guided_complete(manifest, group_id):
            reset_queue_from_step(state, step_id)
            force_steps[step_id] = True
            append_history(state, "phase2_auto_repair", f"Reset `{step_id}` because guided outputs for `{group_id}` were incomplete.")
            changed = True
            return True

    eval_step = evaluation_step_id(state)
    eval_item = queue_item(state, eval_step) if eval_step else None
    summary = summarize_results()
    if eval_item and eval_item.get("status") == "completed" and not phase2_summary_valid(summary, manifest):
        reset_queue_from_step(state, eval_step)
        append_history(state, "phase2_auto_repair", "Reset Phase 2 evaluation because comparison summary was missing or invalid.")
        changed = True

    return changed


def generate_review(summary: dict | None, acceptance: dict, state: dict, manifest: dict) -> dict:
    phase = campaign_phase(manifest, state)
    review = {
        "generated_at": now_iso(),
        "phase": phase,
        "phase_gate": campaign_gate(manifest, state),
        "verdict": "waiting_for_results",
        "decision": "run_experiments",
        "blocker_type": "none",
        "score": 0,
        "workflow_score": 0,
        "evidence_score": 0,
        "strengths": [],
        "weaknesses": [],
        "next_actions": [],
        "claim_assessment": "No evaluation evidence yet.",
    }

    workflow_score = 0
    if summary:
        workflow_score += 2
    if (CAMPAIGN_DIR / "evaluation" / "comparison_report.md").exists():
        workflow_score += 2
    if (CAMPAIGN_DIR / "diagnostics" / "conversion_audit.md").exists() or ((summary or {}).get("diagnostics")):
        workflow_score += 2
    if DECISION_MEMO_PATH.exists():
        workflow_score += 1
    if state.get("queue"):
        workflow_score += 1

    diagnostics = (summary or {}).get("diagnostics", {})
    if diagnostics:
        review["strengths"].append(f"Diagnostics decision: `{diagnostics.get('decision')}`.")
    blocker_reasons = diagnostics.get("blocker_reasons", [])
    if diagnostics.get("decision") == "implementation_blocked" or blocker_reasons:
        review["verdict"] = "implementation_blocked"
        review["decision"] = "stop_for_review"
        review["blocker_type"] = "implementation"
        review["phase_gate"] = "await_phase0_validity"
        review["claim_assessment"] = "A representation or metric blocker is still active, so downstream claim assessment is not scientifically valid."
        review["weaknesses"].extend(blocker_reasons or ["Conversion or GT physics health is blocked."])
        review["next_actions"] = [
            "Fix the implementation blocker before launching new training.",
            "Regenerate the affected references or datasets.",
        ]
        review["workflow_score"] = min(10, workflow_score)
        save_json(CAMPAIGN_DIR / "review.json", review)
        save_json(CAMPAIGN_DIR / "next_actions.json", review["next_actions"])
        return review

    if not summary:
        active_runtime = state.get("active_runtime") or {}
        active_job = state.get("active_job") or {}
        queue_pending = [item["id"] for item in state.get("queue", []) if item.get("status") == "pending"]
        review["weaknesses"].append("No evaluation summary exists yet.")
        failed_items = [item["id"] for item in state.get("queue", []) if item.get("status") in {"failed", "blocked", "stopped"}]
        if failed_items:
            review["next_actions"] = [f"Investigate and repair failed queue step: `{failed_items[0]}`."]
        elif active_runtime.get("step_id"):
            review["next_actions"] = [f"Wait for active queue step `{active_runtime.get('step_id')}` to finish."]
        elif active_job.get("mode"):
            review["next_actions"] = [f"Wait for active training job `{active_job.get('mode')}` to finish."]
        else:
            review["next_actions"] = [f"Run the next pending queue step: `{queue_pending[0]}`."] if queue_pending else ["Wait for the active job to finish and then rerun the loop."]
        review["workflow_score"] = min(10, workflow_score)
        if phase == "phase1_diagnostics":
            review["verdict"] = "phase1_diagnostics_incomplete"
            review["phase_gate"] = "await_phase1_diagnostics"
            review["claim_assessment"] = "Phase 1 cannot be assessed until mixed and partial diagnostic arms are evaluated in the same report."
        save_json(CAMPAIGN_DIR / "review.json", review)
        save_json(CAMPAIGN_DIR / "next_actions.json", review["next_actions"])
        return review

    experiments = (summary or {}).get("experiments", {})
    enabled_ids = enabled_experiment_ids(manifest)
    present_count = sum(1 for key in enabled_ids if experiments.get(key))
    workflow_score += min(2, present_count // 2)

    baseline = get_experiment(summary, "baseline")
    scratch = get_experiment(summary, "infinigen_k10")
    finetune = get_experiment(summary, "finetune_k10")
    mixed = get_experiment(summary, "mixed_finetune_k10")
    partial = get_experiment(summary, "partial_finetune_k10")
    threshold = acceptance.get("scoring", {}).get("partnet_mmd_regression_factor_max", 1.15)
    evidence_score = 0

    if baseline and scratch and finetune:
        review["strengths"].append("Baseline, scratch, and fine-tune arms are present.")
        workflow_score += 1
        ft_box_mmd = metric(summary, "finetune_k10", "box", "mmd")
        ft_box_cov = metric(summary, "finetune_k10", "box", "cov")
        base_box_mmd = metric(summary, "baseline", "box", "mmd")
        scratch_box_cov = metric(summary, "infinigen_k10", "box", "cov")
        ft_pen = metric(summary, "finetune_k10", "physics", "pen_error_mean")
        ft_mob = metric(summary, "finetune_k10", "physics", "mob_error_mean")
        base_pen = metric(summary, "baseline", "physics", "pen_error_mean")
        base_mob = metric(summary, "baseline", "physics", "mob_error_mean")
        ft_partnet = metric(summary, "finetune_k10", "partnet", "mmd")
        base_partnet = metric(summary, "baseline", "partnet", "mmd")

        if ft_box_mmd is not None and base_box_mmd is not None and ft_box_mmd <= base_box_mmd:
            evidence_score += 2
            review["strengths"].append("Fine-tune box-domain multi-state MMD matches or beats baseline.")
        else:
            review["weaknesses"].append("Fine-tune box-domain MMD is worse than baseline.")

        if ft_box_cov is not None and scratch_box_cov is not None and ft_box_cov >= scratch_box_cov:
            evidence_score += 2
            review["strengths"].append("Fine-tune maintains or improves box-domain coverage relative to scratch.")
        else:
            review["weaknesses"].append("Fine-tune loses box-domain coverage relative to scratch.")

        if ft_pen is not None and base_pen is not None and ft_pen <= base_pen:
            evidence_score += 2
            review["strengths"].append("Fine-tune improves or matches baseline penetration error.")
        else:
            review["weaknesses"].append("Penetration error did not improve against baseline.")

        if ft_mob is not None and base_mob is not None and ft_mob <= base_mob:
            evidence_score += 2
            review["strengths"].append("Fine-tune improves or matches baseline mobility error.")
        else:
            review["weaknesses"].append("Mobility error did not improve against baseline.")

        if ft_partnet is not None and base_partnet is not None and ft_partnet <= base_partnet * threshold:
            evidence_score += 2
            review["strengths"].append("PartNet sanity does not show a severe MMD collapse.")
        else:
            review["weaknesses"].append("PartNet sanity shows a notable MMD regression after fine-tuning.")
    else:
        review["weaknesses"].append("Baseline, scratch, and fine-tune comparison arms are incomplete.")

    if phase == "phase1_diagnostics":
        if mixed:
            workflow_score += 1
            review["strengths"].append("Mixed replay diagnostic arm is present.")
        else:
            review["weaknesses"].append("Mixed replay diagnostic arm is missing.")
            review["next_actions"].append("Run `mixed_finetune_k10`.")
        if partial:
            workflow_score += 1
            review["strengths"].append("Partial fine-tune diagnostic arm is present.")
        else:
            review["weaknesses"].append("Partial fine-tune diagnostic arm is missing.")
            review["next_actions"].append("Run `partial_finetune_k10`.")

        diagnosis = dominant_phase1_diagnosis(summary, acceptance)
        if not all_required_phase1_arms_present(summary):
            review["verdict"] = "phase1_diagnostics_incomplete"
            review["decision"] = "run_experiments"
            review["phase_gate"] = "await_phase1_diagnostics"
            review["claim_assessment"] = "Phase 1 remains incomplete until mixed and partial diagnostics are evaluated together."
        else:
            phase2_base_experiment, phase2_base_checkpoint, phase2_base_config = select_phase2_base(summary, manifest, diagnosis)
            review["verdict"] = "phase1_diagnosis_ready"
            review["decision"] = "advance_phase"
            review["phase_gate"] = "auto_handoff_ready"
            review["diagnosis"] = diagnosis
            review["selected_phase2_base_experiment"] = phase2_base_experiment
            review["selected_phase2_base_checkpoint"] = phase2_base_checkpoint
            review["selected_phase2_base_config"] = phase2_base_config
            if diagnosis == "forgetting-dominant":
                review["claim_assessment"] = "The repaired bridge is healthy, and the dominant remaining failure signal is catastrophic forgetting during pure fine-tune."
            elif diagnosis == "over-update-dominant":
                review["claim_assessment"] = "The repaired bridge is healthy, and the dominant remaining failure signal is excessive parameter updates during full fine-tune."
            else:
                review["claim_assessment"] = "The repaired bridge is healthy, but neither mixed replay nor partial fine-tuning resolves the gap; current parameterization appears limiting."
            review["strengths"].append(f"Dominant Phase 1 diagnosis: `{diagnosis}`.")
            if phase2_base_experiment:
                review["strengths"].append(f"Selected Phase 2 base checkpoint: `{phase2_base_experiment}`.")
            review["next_actions"] = [
                "Write the Phase 1 decision memo and selected Phase 2 base checkpoint.",
                "Auto-bootstrap `box_conditioning_v2` immediately after Phase 1 completes.",
            ]
    else:
        anchor = phase2_group(summary, "zero_singleview")
        richer_groups = ["zero_multiview", "multistate_singleview", "multistate_multiview"]
        present_groups = [group for group in ["zero_singleview", *richer_groups] if phase2_group(summary, group)]
        workflow_score += min(3, len(present_groups))
        if not anchor:
            review["verdict"] = "waiting_for_results"
            review["decision"] = "run_experiments"
            review["phase_gate"] = "await_phase2_execution"
            review["weaknesses"].append("Phase 2 anchor group `zero_singleview` is missing.")
            review["claim_assessment"] = "Phase 2 cannot be assessed until the zero-state single-view anchor exists."
        else:
            scoring = acceptance.get("scoring", {})
            phys_factor = scoring.get("physics_tolerance_factor", 1.05)
            best_group = None
            claim_ladder = (summary or {}).get("claim_ladder", [])
            group_outcomes = (summary or {}).get("group_outcomes", {})
            summary_supported_claim = (summary or {}).get("supported_claim_level")
            strongest_true_claim = (summary or {}).get("strongest_true_claim")
            for group in richer_groups:
                payload = phase2_group(summary, group)
                if not payload:
                    review["weaknesses"].append(f"Conditioning group `{group}` is missing.")
                    continue
                workflow_score += 1
                anchor_mmd = conditioning_metric(summary, "zero_singleview", "genfull_per_mmd_mean")
                anchor_cov = conditioning_metric(summary, "zero_singleview", "genfull_per_cov_mean")
                anchor_pen = conditioning_metric(summary, "zero_singleview", "pen_error_mean")
                anchor_mob = conditioning_metric(summary, "zero_singleview", "mob_error_mean")
                cand_mmd = conditioning_metric(summary, group, "genfull_per_mmd_mean")
                cand_cov = conditioning_metric(summary, group, "genfull_per_cov_mean")
                cand_pen = conditioning_metric(summary, group, "pen_error_mean")
                cand_mob = conditioning_metric(summary, group, "mob_error_mean")
                cand_cond = conditioning_metric(summary, group, "cond_error_mean")
                anchor_cond = conditioning_metric(summary, "zero_singleview", "cond_error_mean")
                if None in {anchor_mmd, anchor_cov, anchor_pen, anchor_mob, cand_mmd, cand_cov, cand_pen, cand_mob}:
                    review["weaknesses"].append(f"Conditioning group `{group}` is missing comparable metrics.")
                    continue
                cond_gain = anchor_cond is not None and cand_cond is not None and cand_cond <= anchor_cond
                structural_gain = cand_mmd <= anchor_mmd and cand_cov >= anchor_cov
                physics_preserved = cand_pen <= anchor_pen * phys_factor and cand_mob <= anchor_mob * phys_factor
                if cond_gain:
                    evidence_score += 1
                if structural_gain:
                    evidence_score += 2
                if physics_preserved:
                    evidence_score += 2
                if structural_gain and physics_preserved:
                    review["strengths"].append(f"`{group}` beats the zero-state single-view anchor on structure while preserving physics.")
                    if best_group is None or cand_mmd < conditioning_metric(summary, best_group, "genfull_per_mmd_mean"):
                        best_group = group
                else:
                    review["weaknesses"].append(f"`{group}` does not consistently beat the zero-state single-view anchor.")
            if best_group:
                review["verdict"] = "claim_supported"
                review["decision"] = "stop_for_review"
                review["phase_gate"] = "ready_for_writeup"
                review["supported_claim_level"] = summary_supported_claim or best_group
                review["strongest_true_claim"] = strongest_true_claim or f"`{best_group}` improves the conditioning signal over the original PhysNAP anchor."
                review["claim_assessment"] = f"Phase 2 supports the claim: `{best_group}` improves the conditioning signal over the original PhysNAP anchor."
                review["winning_group"] = best_group
                review["next_actions"] = [
                    "Lock the Phase 2 report and claim memo.",
                    "Prepare the paper-facing narrative around the winning conditioning setting.",
                ]
            elif all(phase2_group(summary, group) for group in ["zero_singleview", *richer_groups]):
                review["verdict"] = "claim_not_supported"
                review["decision"] = "revise_claim"
                review["blocker_type"] = "transfer"
                review["phase_gate"] = "ready_for_writeup"
                phase1_diagnosis = state.get("phase1_diagnosis")
                if phase1_diagnosis == "forgetting-dominant":
                    strongest_true_claim = (
                        "Mixed replay is the strongest validated transfer improvement, but richer conditioning under the current PhysNAP parameterization does not consistently beat the zero-state single-view anchor."
                    )
                else:
                    strongest_true_claim = strongest_true_claim or (
                        "The current PhysNAP conditioning parameterization does not consistently absorb richer procedural observations better than the zero-state single-view anchor."
                    )
                review["strongest_true_claim"] = strongest_true_claim
                review["claim_assessment"] = "All bounded Phase 2 conditioning recipes completed, but none produced a consistent gain over the zero-state single-view anchor."
                if claim_ladder:
                    for entry in claim_ladder:
                        group_id = entry.get("group_id")
                        outcome = group_outcomes.get(group_id, {})
                        review["weaknesses"].append(
                            f"Claim ladder `{entry.get('claim_level')}` via `{group_id}` => {'supported' if outcome.get('supported') else 'not supported'}."
                        )
                review["next_actions"] = [
                    "Write the strongest true claim memo for Phase 2 instead of the original strong claim.",
                    "Record which conditioning groups failed to improve the anchor and why.",
                ]
            else:
                review["verdict"] = "waiting_for_results"
                review["decision"] = "run_experiments"
                review["phase_gate"] = "await_phase2_execution"
                review["claim_assessment"] = "Phase 2 is still running; not all conditioning groups are available yet."
                if not review["next_actions"]:
                    review["next_actions"].append("Run the next pending Phase 2 conditioning step.")

    review["workflow_score"] = min(10, workflow_score)
    review["evidence_score"] = min(10, evidence_score)
    review["score"] = review["evidence_score"]
    save_json(CAMPAIGN_DIR / "review.json", review)
    save_json(CAMPAIGN_DIR / "next_actions.json", review["next_actions"])
    return review


def write_decision_memo(state: dict, review: dict, summary: dict | None, manifest: dict) -> None:
    subprocess.run(
        [str(INFINIGEN_PYTHON), str(RENDER_DECISION_MEMO), "--campaign-dir", str(CAMPAIGN_DIR)],
        check=True,
        cwd=PROJECT_ROOT,
        env=subprocess_env(),
    )


def update_campaign_dashboard(state: dict, summary: dict | None, review: dict | None, launcher: dict, manifest: dict) -> None:
    watch_status = load_watch_status()
    eval_progress = load_eval_progress()
    active_runtime = state.get("active_runtime") or {}
    lines = [
        f"# {campaign_title(manifest)} Campaign Dashboard",
        "",
        f"**Updated**: {now_iso()}",
        f"**Phase**: `{campaign_phase(manifest, state)}`",
        f"**Gate**: `{campaign_gate(manifest, state, review)}`",
    ]
    claim = campaign_claim(manifest)
    if claim:
        lines.append(f"**Claim**: `{claim}`")
    lines.extend(["", "## Queue", ""])
    for item in state.get("queue", []):
        lines.append(f"- `{item['id']}`: {item['status']}")

    lines.extend([
        "",
        "## Runtime",
        "",
        f"- Watcher PID: `{watch_status.get('watcher_pid')}`",
        f"- Supervisor PID: `{watch_status.get('supervisor_pid')}`",
        f"- Worker PID: `{watch_status.get('worker_pid')}`",
        f"- CPU Worker PID: `{watch_status.get('cpu_worker_pid')}`",
        f"- Current Queue Step: `{watch_status.get('queue_step')}`",
        f"- Current Evaluation Stage: `{watch_status.get('stage')}`",
        f"- Watcher Status: `{watch_status.get('status')}`",
        f"- Active GPU Step: `{watch_status.get('active_gpu_step')}`",
        f"- Active CPU Step: `{watch_status.get('active_cpu_step')}`",
        f"- Active Runtime Step: `{active_runtime.get('step_id')}`",
        f"- Active Runtime Worker Kind: `{active_runtime.get('worker_kind')}`",
        f"- Active Runtime Worker PID: `{active_runtime.get('worker_pid')}`",
        f"- Active Runtime Attempt: `{active_runtime.get('attempt_no')}`",
        f"- Last Repair Action: `{watch_status.get('last_repair_action')}`",
        f"- Last Heartbeat: `{watch_status.get('last_heartbeat')}`",
        f"- Latest Artifact Produced: `{watch_status.get('latest_artifact')}`",
        f"- Blocking Error: `{watch_status.get('last_error')}`",
    ])
    if state.get("phase2_base_experiment") or state.get("phase2_base_checkpoint"):
        lines.extend(
            [
                f"- Phase 2 Base Experiment: `{state.get('phase2_base_experiment')}`",
                f"- Phase 2 Base Checkpoint: `{state.get('phase2_base_checkpoint')}`",
                f"- Auto Handoff Status: `{state.get('auto_handoff_status')}`",
            ]
        )

    if eval_progress:
        lines.extend(["", "## Evaluation Progress", ""])
        lines.append(f"- Backend: `{eval_progress.get('distance_backend')}`")
        lines.append(f"- Status: `{eval_progress.get('status')}`")
        lines.append(f"- Current Stage: `{eval_progress.get('current_stage')}`")
        lines.append(f"- Message: {eval_progress.get('message')}")
        for stage_name, stage_state in eval_progress.get("stages", {}).items():
            lines.append(
                f"- `{stage_name}`: `{stage_state.get('status')}` | updated `{stage_state.get('updated_at')}` | {stage_state.get('message')}"
            )

    experiments = launcher.get("experiments", {})
    if enabled_experiment_ids(manifest):
        lines.extend(["", "## Experiment Status", ""])
        for experiment_id in enabled_experiment_ids(manifest):
            key = launcher_key_for_experiment(experiment_id)
            meta = experiments.get(key, {"status": "missing"})
            lines.append(f"### {experiment_id}")
            lines.append("")
            lines.append(f"- Status: `{meta.get('status', 'missing')}`")
            if meta.get("pid"):
                lines.append(f"- PID: `{meta.get('pid')}`")
            if meta.get("launched_at"):
                lines.append(f"- Launched: `{meta.get('launched_at')}`")
            if meta.get("finished_at"):
                lines.append(f"- Finished: `{meta.get('finished_at')}`")
            if meta.get("checkpoint_path"):
                lines.append(f"- Checkpoint: `{meta.get('checkpoint_path')}`")
            if meta.get("log_dir"):
                lines.append(f"- Log dir: `{meta.get('log_dir')}`")
            lines.append("")
    elif summary and summary.get("conditioning_groups"):
        lines.extend(["", "## Conditioning Groups", ""])
        for group_id, payload in summary.get("conditioning_groups", {}).items():
            lines.append(f"- `{group_id}`: output `{payload.get('output_name')}` | mmd `{payload.get('genfull_per_mmd_mean')}` | cov `{payload.get('genfull_per_cov_mean')}` | pen `{payload.get('pen_error_mean')}` | mob `{payload.get('mob_error_mean')}`")

    lines.extend(["## Review", ""])
    if review:
        lines.append(f"- Verdict: `{review.get('verdict')}`")
        lines.append(f"- Decision: `{review.get('decision')}`")
        lines.append(f"- Blocker Type: `{review.get('blocker_type')}`")
        lines.append(f"- Evidence Score: `{review.get('evidence_score', review.get('score'))}`/10")
        lines.append(f"- Workflow Score: `{review.get('workflow_score', 'n/a')}`/10")
        if review.get("diagnosis"):
            lines.append(f"- Diagnosis: `{review.get('diagnosis')}`")
        if review.get("strongest_true_claim"):
            lines.append(f"- Strongest True Claim: {review.get('strongest_true_claim')}")
        if review.get("supported_claim_level"):
            lines.append(f"- Supported Claim Level: `{review.get('supported_claim_level')}`")
        if review.get("claim_assessment"):
            lines.append(f"- Claim assessment: {review.get('claim_assessment')}")
        for weakness in review.get("weaknesses", []):
            lines.append(f"- Weakness: {weakness}")
        for strength in review.get("strengths", []):
            lines.append(f"- Strength: {strength}")
        for action in review.get("next_actions", []):
            lines.append(f"- Next: {action}")
    else:
        lines.append("- Review not generated yet.")

    if state.get("history"):
        lines.extend(["", "## Recent History", ""])
        for event in state["history"][-8:]:
            lines.append(f"- {event.get('at')}: {event.get('event')} | {event.get('details')}")

    if summary:
        lines.extend(["", "## Evaluation Summary", "", "Evaluation artifacts are present."])
        diagnostics = summary.get("diagnostics", {})
        if diagnostics:
            lines.extend([
                "",
                "## Diagnostics",
                "",
                f"- Conversion audit decision: `{diagnostics.get('decision')}`",
                f"- GT pen mean: `{diagnostics.get('gt_self_eval', {}).get('pen_error_mean')}`",
                f"- GT mob mean: `{diagnostics.get('gt_self_eval', {}).get('mob_error_mean')}`",
            ])
    elif eval_progress:
        lines.extend(["", "## Evaluation Summary", "", f"Evaluation is `{eval_progress.get('status')}` at stage `{eval_progress.get('current_stage')}`."])
    else:
        lines.extend(["", "## Evaluation Summary", "", "No current evaluation summary for this campaign phase."])

    (CAMPAIGN_DIR / "campaign_status.md").write_text("\n".join(lines) + "\n")


def refresh_live_campaign_dashboard(state: dict, manifest: dict) -> None:
    launcher = train_status()
    summary = summarize_results()
    acceptance = load_acceptance()
    review = generate_review(summary, acceptance, state, manifest)
    state["last_review"] = review
    update_campaign_dashboard(state, summary, review, launcher, manifest)
    try:
        maybe_sync_project_memory_status(force=False)
    except Exception:
        pass


def update_markdown_files(state: dict, review: dict | None) -> None:
    task_lines = [
        "# Task Plan",
        "",
        f"**Status**: {review.get('verdict') if review else 'RUNNING'}",
        f"**Last Updated**: {now_iso()}",
        "",
        "## Campaign",
        "",
        f"- campaign_dir={CAMPAIGN_DIR}",
        f"- phase={state.get('phase')}",
        f"- phase_gate={state.get('phase_gate')}",
        "",
        "## Queue",
        "",
    ]
    for item in state.get("queue", []):
        task_lines.append(f"- `{item['id']}`: {item['status']}")
    if review:
        task_lines.extend([
            "",
            "## Review",
            "",
            f"- verdict={review.get('verdict')}",
            f"- decision={review.get('decision')}",
            f"- workflow_score={review.get('workflow_score')}",
            f"- evidence_score={review.get('evidence_score', review.get('score'))}",
        ])
    (PROJECT_ROOT / "task_plan.md").write_text("\n".join(task_lines) + "\n")

    findings = [
        "# Findings",
        "",
        f"- campaign={CAMPAIGN_DIR.name}",
        f"- phase={state.get('phase')}",
        f"- gate={state.get('phase_gate')}",
    ]
    if review:
        findings.append(f"- verdict={review.get('verdict')}")
        if review.get("diagnosis"):
            findings.append(f"- diagnosis={review.get('diagnosis')}")
        for weakness in review.get("weaknesses", []):
            findings.append(f"- Weakness: {weakness}")
        for strength in review.get("strengths", []):
            findings.append(f"- Strength: {strength}")
    (PROJECT_ROOT / "findings.md").write_text("\n".join(findings) + "\n")

    progress_path = PROJECT_ROOT / "progress.md"
    existing = progress_path.read_text().rstrip() + "\n" if progress_path.exists() else ""
    note = f"- {now_iso()}: {CAMPAIGN_DIR.name} phase={state.get('phase')} gate={state.get('phase_gate')} verdict={review.get('verdict') if review else 'n/a'} decision={review.get('decision') if review else 'n/a'}"
    progress_path.write_text(existing + note + "\n")


def maybe_sync_project_memory_status(force: bool = False, interval_seconds: int = 300) -> None:
    if not UPDATE_STATUS.exists():
        return
    now_ts = time.time()
    if not force and STATUS_SYNC_MARKER.exists():
        try:
            previous = float(STATUS_SYNC_MARKER.read_text().strip())
        except ValueError:
            previous = 0.0
        if now_ts - previous < interval_seconds:
            return
    try:
        subprocess.run(
            [str(INFINIGEN_PYTHON), str(UPDATE_STATUS), "--project-root", str(PROJECT_ROOT)],
            check=True,
            cwd=PROJECT_ROOT,
            env=subprocess_env(),
        )
    except subprocess.CalledProcessError:
        return
    STATUS_SYNC_MARKER.write_text(f"{now_ts}\n")


def latest_significant_history(state: dict) -> dict | None:
    significant = {"auto_repair", "phase2_auto_repair", "auto_handoff_from_phase1", "phase2_scaffold_created"}
    for event in reversed(state.get("history", [])):
        if event.get("event") in significant:
            return event
    return None


def phase2_completed_guided_count(state: dict) -> int:
    total = 0
    for step_id in PHASE2_GUIDED_STEP_TO_GROUP:
        item = queue_item(state, step_id)
        if item and item.get("status") == "completed":
            total += 1
    return total


def phase2_total_guided_count() -> int:
    return len(PHASE2_GUIDED_STEP_TO_GROUP)


def iteration_title(state: dict, review: dict, summary: dict | None) -> str:
    phase = review.get("phase")
    verdict = review.get("verdict")
    if phase == "phase1_diagnostics" and review.get("diagnosis"):
        return f"{CAMPAIGN_DIR.name} phase1 diagnosis {review.get('diagnosis')}"
    if phase == "phase2_conditioning_design":
        guided_done = phase2_completed_guided_count(state)
        if verdict in {"claim_supported", "claim_not_supported", "implementation_blocked"}:
            return f"{CAMPAIGN_DIR.name} phase2 verdict {verdict}"
        significant = latest_significant_history(state)
        if significant and significant.get("event") in {"auto_repair", "phase2_auto_repair"}:
            return f"{CAMPAIGN_DIR.name} phase2 repair {significant.get('event')}"
        if guided_done:
            return f"{CAMPAIGN_DIR.name} phase2 progress {guided_done}-of-{phase2_total_guided_count()} groups"
    return f"{CAMPAIGN_DIR.name} {phase} verdict {verdict}"


def build_iteration_payload(state: dict, review: dict, summary: dict | None) -> dict:
    phase = review.get("phase")
    verdict = review.get("verdict")
    guided_done = phase2_completed_guided_count(state)
    summary_line = f"{CAMPAIGN_DIR.name} {phase} verdict={verdict} decision={review.get('decision')}"
    if phase == "phase1_diagnostics" and review.get("diagnosis"):
        summary_line = f"{CAMPAIGN_DIR.name} phase1 diagnosis={review.get('diagnosis')}"
    elif phase == "phase2_conditioning_design":
        summary_line = f"{CAMPAIGN_DIR.name} phase2 progress={guided_done}/{phase2_total_guided_count()} verdict={verdict}"

    bugs_fixed = []
    significant = latest_significant_history(state)
    if significant and significant.get("event") in {"auto_repair", "phase2_auto_repair"}:
        bugs_fixed.append(
            {
                "bug": significant.get("details", ""),
                "root_cause": significant.get("event", ""),
                "fix": "Auto-loop detected the issue, reset the affected step, and re-entered the queue with repaired artifacts or launch contract.",
            }
        )

    completed = [f"Queue status: {[item.get('id') + '=' + item.get('status') for item in state.get('queue', [])]}"]
    if phase == "phase2_conditioning_design":
        completed.append(f"Phase 2 guided groups complete: {guided_done}/{phase2_total_guided_count()}")

    lessons = [review.get("claim_assessment", "")]
    if review.get("strongest_true_claim"):
        lessons.append(review.get("strongest_true_claim"))
    if review.get("diagnosis"):
        lessons.append(f"Dominant diagnosis: {review.get('diagnosis')}")

    return {
        "summary": summary_line,
        "completed": completed,
        "bugs_fixed": bugs_fixed,
        "lessons": [item for item in lessons if item],
        "artifacts": {
            "campaign_manifest": str((CAMPAIGN_DIR / "manifest.yaml").relative_to(PROJECT_ROOT)),
            "campaign_status": str((CAMPAIGN_DIR / "campaign_status.md").relative_to(PROJECT_ROOT)),
            "decision_memo": str((CAMPAIGN_DIR / "decision_memo.md").relative_to(PROJECT_ROOT)),
            "campaign_state": str((CAMPAIGN_DIR / "state.json").relative_to(PROJECT_ROOT)),
            "campaign_review": str((CAMPAIGN_DIR / "review.json").relative_to(PROJECT_ROOT)),
        },
        "architecture_changes": [
            f"Phase gate is now {review.get('phase_gate')}",
            f"Campaign phase is {phase}",
        ],
        "next": review.get("next_actions", []),
    }


def maybe_record_iteration(state: dict, review: dict, summary: dict | None) -> None:
    if not review:
        return
    terminal_verdicts = {"implementation_blocked", "phase1_diagnosis_ready", "claim_not_supported", "claim_supported"}
    significant = False
    if review.get("verdict") in terminal_verdicts:
        significant = True
    elif review.get("phase") == "phase1_diagnostics" and review.get("diagnosis"):
        significant = True
    elif review.get("phase") == "phase2_conditioning_design" and phase2_completed_guided_count(state) > 0:
        significant = True
    elif latest_significant_history(state):
        significant = True
    if not significant:
        return
    marker_payload = {
        "phase": review.get("phase"),
        "phase_gate": review.get("phase_gate"),
        "verdict": review.get("verdict"),
        "decision": review.get("decision"),
        "diagnosis": review.get("diagnosis"),
        "strongest_true_claim": review.get("strongest_true_claim"),
        "guided_complete": phase2_completed_guided_count(state),
        "significant_history": latest_significant_history(state),
    }
    marker_value = json.dumps(marker_payload, sort_keys=True, ensure_ascii=False)
    if ITERATION_MARKER.exists() and ITERATION_MARKER.read_text().strip() == marker_value:
        return
    payload_path = CAMPAIGN_DIR / "runtime" / "iteration_payload.json"
    save_json(payload_path, build_iteration_payload(state, review, summary))
    subprocess.run(
        [
            str(INFINIGEN_PYTHON),
            str(RECORD_ITERATION),
            "--project-root",
            str(PROJECT_ROOT),
            "--payload",
            str(payload_path),
            "--update-status",
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=subprocess_env(),
    )
    history_titles = [iteration_title(state, review, summary)]
    if review.get("phase") == "phase1_diagnostics" and review.get("diagnosis"):
        history_titles.append(f"{CAMPAIGN_DIR.name} phase1 final diagnosis {review.get('diagnosis')}")
    if review.get("phase") == "phase2_conditioning_design" and review.get("verdict") in terminal_verdicts:
        history_titles.append(f"{CAMPAIGN_DIR.name} phase2 final verdict {review.get('verdict')}")
    if WRITE_CAMPAIGN_HISTORY.exists():
        seen_titles = set()
        for title in history_titles:
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            subprocess.run(
                [
                    str(INFINIGEN_PYTHON),
                    str(WRITE_CAMPAIGN_HISTORY),
                    "--project-root",
                    str(PROJECT_ROOT),
                    "--campaign-dir",
                    str(CAMPAIGN_DIR),
                    "--title",
                    title,
                ],
                check=True,
                cwd=PROJECT_ROOT,
                env=subprocess_env(),
            )
    ITERATION_MARKER.write_text(marker_value + "\n")

def sync_running_steps(state: dict, launcher: dict) -> None:
    state["active_job"] = launcher.get("active_train_job")
    experiments = launcher.get("experiments", {})
    for step_id, mode in TRAIN_STEP_TO_MODE.items():
        item = queue_item(state, step_id)
        if not item:
            continue
        meta = experiments.get(mode, {})
        status = meta.get("status")
        if status in {"running", "completed", "failed", "stopped"}:
            item["status"] = status
            item["updated_at"] = now_iso()


def sync_evaluation_step(state: dict) -> None:
    eval_step = evaluation_step_id(state)
    if not eval_step:
        return
    item = queue_item(state, eval_step)
    if not item:
        return
    summary_path = EVALUATION_DIR / "comparison_summary.json"
    progress = load_eval_progress()
    watch = load_watch_status()
    if summary_path.exists():
        item["status"] = "completed"
    elif progress.get("status") == "failed":
        item["status"] = "failed"
    elif (
        watch.get("queue_step") == eval_step
        and ((watch.get("worker_pid") and process_alive(watch.get("worker_pid"))) or progress.get("status") == "running")
    ):
        item["status"] = "running"
    elif item.get("status") == "running":
        item["status"] = "pending"
    item["updated_at"] = now_iso()


def read_text_safe(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="ignore")


def launcher_meta_for_step(step_id: str, launcher: dict) -> dict:
    if step_id == "evaluate_all":
        return {}
    mode = TRAIN_STEP_TO_MODE.get(step_id)
    if not mode:
        return {}
    return (launcher.get("experiments") or {}).get(mode, {})


def classify_failure(step_id: str, launcher: dict, manifest: dict) -> str | None:
    meta = launcher_meta_for_step(step_id, launcher)
    log_dir = meta.get("log_dir")
    if not log_dir:
        if step_id in PHASE2_GUIDED_STEP_TO_GROUP:
            group_id = PHASE2_GUIDED_STEP_TO_GROUP[step_id]
            if not phase2_conditioning_complete(manifest, group_id):
                return "conditioning_export_incomplete"
            if not phase2_guided_complete(manifest, group_id):
                return "missing_guided_stats"
            return "invalid_guided_metrics"
        if step_id in PHASE2_EXPORT_STEP_TO_GROUP:
            group_id = PHASE2_EXPORT_STEP_TO_GROUP[step_id]
            if not phase2_conditioning_complete(manifest, group_id):
                return "conditioning_export_incomplete"
            return None
        if step_id == "evaluate_conditioning_v2":
            summary = summarize_results()
            if not phase2_summary_valid(summary, manifest):
                return "invalid_phase2_summary"
            return None
        return None
    log_path = CAMPAIGN_DIR / "runtime" / log_dir / "launcher.log"
    log_text = read_text_safe(log_path)
    if "KeyError: 'all'" in log_text:
        return "optimizer_key_mismatch"
    if "CUDA out of memory" in log_text:
        return "cuda_oom"
    if "No such file or directory" in log_text and "checkpoint" in log_text:
        return "missing_checkpoint"
    return None


def attempt_auto_repair(step_id: str, failure_kind: str | None, state: dict, launcher: dict, manifest: dict) -> bool:
    if not failure_kind:
        return False
    retries = state.setdefault("retries", {})
    repair_key = f"auto_repair::{step_id}::{failure_kind}"
    if retries.get(repair_key, 0) >= 1:
        return False
    retries[repair_key] = retries.get(repair_key, 0) + 1
    item = queue_item(state, step_id)
    if step_id in PHASE2_EXPORT_STEP_TO_GROUP or step_id in PHASE2_GUIDED_STEP_TO_GROUP or step_id == "evaluate_conditioning_v2":
        if step_id in PHASE2_GUIDED_STEP_TO_GROUP and failure_kind == "conditioning_export_incomplete":
            export_step = next(key for key, value in PHASE2_EXPORT_STEP_TO_GROUP.items() if value == PHASE2_GUIDED_STEP_TO_GROUP[step_id])
            reset_queue_from_step(state, export_step)
            state.setdefault("force_steps", {})[export_step] = True
        else:
            reset_queue_from_step(state, step_id)
            state.setdefault("force_steps", {})[step_id] = True
    elif item:
        item["status"] = "pending"
        item["updated_at"] = now_iso()
    eval_item = queue_item(state, evaluation_step_id(state) or "evaluate_all")
    if eval_item and eval_item.get("status") == "running":
        eval_item["status"] = "pending"
        eval_item["updated_at"] = now_iso()

    meta = launcher_meta_for_step(step_id, launcher)
    log_dir = meta.get("log_dir")
    if log_dir:
        meta_path = CAMPAIGN_DIR / "runtime" / log_dir / "run_meta.json"
        exit_code_path = CAMPAIGN_DIR / "runtime" / log_dir / "exit_code.txt"
        if meta_path.exists():
            payload = load_json(meta_path, {})
            payload["status"] = "retry_pending"
            payload.pop("exit_code", None)
            payload.pop("finished_at", None)
            save_json(meta_path, payload)
        if exit_code_path.exists():
            exit_code_path.unlink()

    append_history(state, "auto_repair", f"Queued retry for {step_id} after detected failure `{failure_kind}`.")
    update_watch_status(
        mode="loop",
        queue_step=step_id,
        stage=failure_kind,
        status="retrying",
        clear_worker=True,
        last_repair_action=f"{step_id}:{failure_kind}",
        last_error=failure_kind,
    )
    return True


def append_history(state: dict, event: str, details: str) -> None:
    state.setdefault("history", []).append({"at": now_iso(), "event": event, "details": details})


def apply_review_state_updates(state: dict, review: dict) -> None:
    state["phase_gate"] = review.get("phase_gate", state.get("phase_gate"))
    if review.get("selected_phase2_base_experiment"):
        state["phase2_base_experiment"] = review.get("selected_phase2_base_experiment")
        state["phase2_base_checkpoint"] = review.get("selected_phase2_base_checkpoint")
        state["phase2_base_config"] = review.get("selected_phase2_base_config")
        state["auto_handoff_status"] = "ready"


def step_worker_kind(step_id: str) -> str:
    if step_id in PHASE2_GUIDED_STEP_TO_GROUP:
        return "gpu_guided"
    if step_id in PHASE2_EXPORT_STEP_TO_GROUP or step_id.startswith("prepare_") or step_id == "write_phase2_claim_memo":
        return "cpu_prepare"
    if step_id == evaluation_step_id(load_state()):
        return "evaluation"
    return "unknown"


def run_once() -> dict:
    manifest = load_manifest()
    acceptance = load_acceptance()
    state = load_state()
    launcher = train_status()
    state["phase"] = campaign_phase(manifest, state)
    state["phase_gate"] = campaign_gate(manifest, state)
    sync_running_steps(state, launcher)
    sync_evaluation_step(state)
    sync_phase2_runtime_steps(state, manifest)
    summary = summarize_results()

    if repair_phase2_contract(state, manifest):
        save_state(state)
        launcher = train_status()
        sync_running_steps(state, launcher)
        sync_evaluation_step(state)
        sync_phase2_runtime_steps(state, manifest)
        summary = summarize_results()

    if state.get("active_job"):
        active = state["active_job"]
        update_watch_status(
            mode="train",
            queue_step=f"train_{active['mode']}",
            stage=active["mode"],
            status="running",
            worker_pid=active.get("pid"),
            latest_artifact=active.get("log_path"),
            active_gpu_step=f"train_{active['mode']}",
            last_error=None,
        )
        review = generate_review(summary, acceptance, state, manifest)
        state["last_review"] = review
        apply_review_state_updates(state, review)
        write_decision_memo(state, review, summary, manifest)
        update_markdown_files(state, review)
        save_state(state)
        update_campaign_dashboard(state, summary, review, launcher, manifest)
        maybe_sync_project_memory_status(force=True)
        return {"state": state, "review": review}

    failed_like = next((item for item in state.get("queue", []) if item.get("status") in {"failed", "stopped", "blocked"}), None)
    if failed_like:
        failure_kind = classify_failure(failed_like.get("id"), launcher, manifest)
        if attempt_auto_repair(failed_like.get("id"), failure_kind, state, launcher, manifest):
            save_state(state)
            return run_once()
        update_watch_status(
            mode="loop",
            queue_step=failed_like.get("id"),
            stage=failure_kind or load_eval_progress().get("current_stage"),
            status="failed",
            clear_worker=True,
            last_error=failure_kind,
        )
        review = generate_review(summary, acceptance, state, manifest)
        if failure_kind and failure_kind not in review.get("weaknesses", []):
            review.setdefault("weaknesses", []).append(f"Auto-repair exhausted for failure kind `{failure_kind}` on step `{failed_like.get('id')}`.")
            review.setdefault("next_actions", []).append(f"Investigate and repair `{failed_like.get('id')}` failure `{failure_kind}` before rerunning the queue.")
            save_json(CAMPAIGN_DIR / "review.json", review)
            save_json(CAMPAIGN_DIR / "next_actions.json", review.get("next_actions", []))
        state["last_review"] = review
        apply_review_state_updates(state, review)
        write_decision_memo(state, review, summary, manifest)
        save_state(state)
        update_markdown_files(state, review)
        update_campaign_dashboard(state, summary, review, launcher, manifest)
        maybe_record_iteration(state, review, summary)
        maybe_sync_project_memory_status(force=True)
        return {"state": state, "review": review}

    next_item = next((item for item in state.get("queue", []) if item.get("status") != "completed"), None)
    if next_item and next_item.get("status") == "pending":
        step_id = next_item["id"]
        next_item["updated_at"] = now_iso()
        if step_id in PREPARE_STEPS:
            force = bool(state.setdefault("force_steps", {}).pop(step_id, False))
            attempt_no = state.setdefault("retries", {}).get(step_id, 0) + 1
            next_item["status"] = "running"
            launched_at = now_iso()
            worker_kind = "gpu_guided" if step_id in PHASE2_GUIDED_STEP_TO_GROUP else "cpu_prepare"
            set_active_runtime(
                state,
                step_id=step_id,
                worker_kind=worker_kind,
                attempt_no=attempt_no,
                launched_at=launched_at,
            )
            save_state(state)

            def _on_start(pid: int) -> None:
                active = state.get("active_runtime") or {}
                active["worker_pid"] = pid
                active["heartbeat_at"] = now_iso()
                state["active_runtime"] = active
                save_state(state)
                update_watch_status(
                    mode="prepare",
                    queue_step=step_id,
                    stage=step_id,
                    status="running",
                    worker_pid=pid,
                    active_gpu_step=step_id if worker_kind == "gpu_guided" else None,
                    active_cpu_step=step_id if worker_kind == "cpu_prepare" else None,
                    last_error=None,
                )
                refresh_live_campaign_dashboard(state, manifest)

            def _on_poll(pid: int) -> None:
                active = state.get("active_runtime") or {}
                active["worker_pid"] = pid
                active["heartbeat_at"] = now_iso()
                state["active_runtime"] = active
                save_state(state)
                update_watch_status(
                    mode="prepare",
                    queue_step=step_id,
                    stage=step_id,
                    status="running",
                    worker_pid=pid,
                    active_gpu_step=step_id if worker_kind == "gpu_guided" else None,
                    active_cpu_step=step_id if worker_kind == "cpu_prepare" else None,
                    last_error=None,
                )
                refresh_live_campaign_dashboard(state, manifest)

            try:
                run_prepare(PREPARE_STEPS[step_id], force=force, on_start=_on_start, on_poll=_on_poll)
                next_item["status"] = "completed"
                clear_active_runtime(state)
                append_history(state, step_id, f"Completed {step_id}{' with --force' if force else ''}.")
                update_watch_status(
                    mode="prepare",
                    queue_step=step_id,
                    stage=step_id,
                    status="completed",
                    clear_worker=True,
                    active_gpu_step=None,
                    active_cpu_step=None,
                    last_error=None,
                )
            except subprocess.CalledProcessError as exc:
                next_item["status"] = "failed"
                clear_active_runtime(state)
                update_watch_status(
                    mode="prepare",
                    queue_step=step_id,
                    stage=step_id,
                    status="failed",
                    clear_worker=True,
                    active_gpu_step=None,
                    active_cpu_step=None,
                    last_error=f"{step_id} exited with status {exc.returncode}",
                )
                append_history(state, step_id, f"{step_id} failed with exit code {exc.returncode}.")
        elif step_id in TRAIN_STEP_TO_MODE:
            mode = TRAIN_STEP_TO_MODE[step_id]
            launch_train(mode)
            next_item["status"] = "running"
            append_history(state, step_id, f"Launched {mode} in background.")
        elif step_id == evaluation_step_id(state):
            next_item["status"] = "running"
            save_state(state)
            review = generate_review(summary, acceptance, state, manifest)
            state["last_review"] = review
            write_decision_memo(state, review, summary, manifest)
            update_markdown_files(state, review)
            update_campaign_dashboard(state, summary, review, launcher, manifest)
            try:
                run_evaluate(state, launcher, manifest)
                next_item["status"] = "completed"
                append_history(state, step_id, "Evaluation completed.")
                update_watch_status(
                    mode=step_id,
                    queue_step=step_id,
                    stage="write_report",
                    status="completed",
                    worker_pid=None,
                    latest_artifact=str(EVALUATION_DIR / "comparison_report.md"),
                    active_gpu_step=None,
                    last_error=None,
                )
            except subprocess.CalledProcessError as exc:
                retries = state.setdefault("retries", {})
                retry_count = retries.get(step_id, 0)
                retries[step_id] = retry_count + 1
                next_item["status"] = "pending" if retry_count < 1 else "failed"
                update_watch_status(
                    mode=step_id,
                    queue_step=step_id,
                    stage=load_eval_progress().get("current_stage"),
                    status="retrying" if retry_count < 1 else "failed",
                    clear_worker=True,
                    latest_artifact=str(RUNTIME_DIR / f"{step_id}.log"),
                    active_gpu_step=None,
                    last_error=f"{step_id} exited with status {exc.returncode}",
                )
                append_history(state, step_id, f"Evaluation failed with exit code {exc.returncode}.")
        else:
            next_item["status"] = "blocked"
            append_history(state, step_id, f"No executable handler for queue step `{step_id}`.")

    launcher = train_status()
    sync_running_steps(state, launcher)
    sync_evaluation_step(state)
    summary = summarize_results()
    review = generate_review(summary, acceptance, state, manifest)
    state["last_review"] = review
    apply_review_state_updates(state, review)
    save_state(state)
    write_decision_memo(state, review, summary, manifest)
    update_markdown_files(state, review)
    update_campaign_dashboard(state, summary, review, launcher, manifest)
    maybe_record_iteration(state, review, summary)
    maybe_sync_project_memory_status(force=True)
    return {"state": state, "review": review}


def run_until_blocked() -> dict:
    before = None
    result = {"state": load_state(), "review": None}
    while True:
        result = run_once()
        current = json.dumps(result["state"].get("queue", []), sort_keys=True)
        if current == before or result["state"].get("active_job"):
            return result
        before = current


def run_to_completion(poll_seconds: int) -> dict:
    result = {"state": load_state(), "review": None}
    while True:
        update_watch_status(mode="loop", queue_step=load_watch_status().get("queue_step"), stage=load_eval_progress().get("current_stage"), status="running")
        result = run_once()
        state = result["state"]
        statuses = [item.get("status") for item in state.get("queue", [])]
        if statuses and all(status == "completed" for status in statuses):
            return result
        if not state.get("active_job") and any(status in {"failed", "stopped", "blocked"} for status in statuses):
            return result
        if state.get("active_job"):
            active = state["active_job"]
            update_watch_status(
                mode="train",
                queue_step=f"train_{active['mode']}",
                stage=active["mode"],
                status="running",
                worker_pid=active.get("pid"),
                latest_artifact=active.get("log_path"),
                last_error=None,
            )
            update_campaign_dashboard(state, summarize_results(), result.get("review"), train_status(), load_manifest())
            time.sleep(poll_seconds)
            continue
        time.sleep(min(10, poll_seconds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the phase-aware PhysNAP/Infinigen auto-review loop")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--run-until-blocked", action="store_true")
    parser.add_argument("--run-to-completion", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.status:
        print(json.dumps(load_state(), indent=2, ensure_ascii=False))
        return 0
    claim_watcher_lock()
    try:
        if args.run_once:
            result = run_once()
            release_watcher_lock("completed")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.run_until_blocked:
            result = run_until_blocked()
            final_status = "failed" if any(item.get("status") in {"failed", "stopped", "blocked"} for item in result["state"].get("queue", [])) else "completed"
            release_watcher_lock(final_status)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.run_to_completion:
            result = run_to_completion(args.poll_seconds)
            final_status = "failed" if any(item.get("status") in {"failed", "stopped", "blocked"} for item in result["state"].get("queue", [])) else "completed"
            release_watcher_lock(final_status)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
    except Exception as exc:
        release_watcher_lock("failed", f"{type(exc).__name__}: {exc}")
        raise
    raise SystemExit("Choose one of --status, --run-once, --run-until-blocked, or --run-to-completion")


if __name__ == "__main__":
    raise SystemExit(main())
