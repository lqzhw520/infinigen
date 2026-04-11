#!/usr/bin/env python3
"""P0A: clean official control rerun with atomic bookkeeping."""

from __future__ import annotations

import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

from lerobot.scripts.lerobot_eval import eval_policy_all

from mint_common import now_iso, write_json_atomic, write_text_atomic
from p1_execution_common import (
    ARTIFACT_DIR,
    CONTROL_BASELINE_REF,
    CONTROL_EPISODES,
    CONTROL_N_ACTION_STEPS,
    CONTROL_SEED,
    OUTPUT_DIR,
    atomic_finalize_output_dir,
    bundle_config_hash,
    close_bundle,
    failure_payload,
    make_official_control_bundle,
    upsert_evidence,
    video_integrity_report,
)

ARTIFACT_PATH = ARTIFACT_DIR / "p1j_control_rerun_v2.json"
FINAL_OUTPUT_DIR = OUTPUT_DIR / "p1j_control_rerun_v2"
EVIDENCE_ID = "E052"
EXPERIMENT_ID = "p1j_control_rerun_v2"


def _staging_dir() -> Path:
    return FINAL_OUTPUT_DIR.parent / f"{FINAL_OUTPUT_DIR.name}.staging.{int(time.time())}"


def _write_sidecars(staging_dir: Path, *, stdout_text: str, stderr_text: str, returncode: int, duration_s: float, eval_info: dict[str, Any] | None, variant_summary: dict[str, Any]) -> None:
    write_text_atomic(staging_dir / "stdout.log", stdout_text)
    write_text_atomic(staging_dir / "stderr.log", stderr_text)
    write_text_atomic(staging_dir / "returncode.txt", f"{int(returncode)}\n")
    write_text_atomic(staging_dir / "duration_s.txt", f"{duration_s:.6f}\n")
    write_json_atomic(staging_dir / "variant_summary.json", variant_summary)
    if eval_info is not None:
        write_json_atomic(staging_dir / "eval_info.json", eval_info)


def main() -> int:
    if FINAL_OUTPUT_DIR.exists():
        shutil.rmtree(FINAL_OUTPUT_DIR)
    staging_dir = _staging_dir()
    staging_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    stdout_lines = [
        "[p1j] starting clean official control rerun via direct Python eval path",
        f"[p1j] target output dir: {FINAL_OUTPUT_DIR}",
        f"[p1j] requested n_action_steps={CONTROL_N_ACTION_STEPS}",
    ]
    stderr_text = ""
    returncode = 0
    eval_info: dict[str, Any] | None = None
    bundle = None
    passed = False
    video_integrity = {"passed": False, "video_count": 0, "records": []}
    config_hash = None

    try:
        bundle = make_official_control_bundle(
            n_action_steps=CONTROL_N_ACTION_STEPS,
            n_episodes=CONTROL_EPISODES,
            device="cuda",
            seed=CONTROL_SEED,
            output_dir=FINAL_OUTPUT_DIR,
        )
        config_hash = bundle_config_hash(bundle["cfg"])
        stdout_lines.append(f"[p1j] config_hash={config_hash}")
        eval_info = eval_policy_all(
            envs=bundle["envs"],
            policy=bundle["policy"],
            env_preprocessor=bundle["env_preprocessor"],
            env_postprocessor=bundle["env_postprocessor"],
            preprocessor=bundle["preprocessor"],
            postprocessor=bundle["postprocessor"],
            n_episodes=CONTROL_EPISODES,
            max_episodes_rendered=CONTROL_EPISODES,
            videos_dir=staging_dir / "videos",
            return_episode_data=False,
            start_seed=CONTROL_SEED,
            max_parallel_tasks=1,
        )
        video_paths = [Path(p) for p in eval_info.get("overall", {}).get("video_paths", [])]
        video_integrity = video_integrity_report(video_paths)
        success_reproduced = float(eval_info.get("overall", {}).get("pc_success", 0.0)) >= 100.0
        passed = bool(success_reproduced and video_integrity["passed"])
        stdout_lines.append(f"[p1j] pc_success={eval_info.get('overall', {}).get('pc_success')}")
        stdout_lines.append(f"[p1j] video_count={video_integrity['video_count']}")
        stdout_lines.append(f"[p1j] passed={passed}")
    except Exception as exc:  # noqa: BLE001
        returncode = 1
        failure = failure_payload(exc)
        stderr_text = failure["traceback"]
        stdout_lines.append(f"[p1j] failed with {failure['error']}")
    finally:
        if bundle is not None:
            close_bundle(bundle)

    duration_s = time.time() - started
    variant_summary = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at": now_iso(),
        "control_baseline_ref": CONTROL_BASELINE_REF,
        "n_action_steps": CONTROL_N_ACTION_STEPS,
        "seed": CONTROL_SEED,
        "n_episodes": CONTROL_EPISODES,
        "config_hash": config_hash,
        "passed": passed,
        "returncode": returncode,
    }
    _write_sidecars(
        staging_dir,
        stdout_text="\n".join(stdout_lines).rstrip() + "\n",
        stderr_text=stderr_text,
        returncode=returncode,
        duration_s=duration_s,
        eval_info=eval_info,
        variant_summary=variant_summary,
    )
    atomic_finalize_output_dir(staging_dir, FINAL_OUTPUT_DIR)

    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at": now_iso(),
        "control_baseline_ref": CONTROL_BASELINE_REF,
        "passed": passed,
        "returncode": returncode,
        "duration_s": duration_s,
        "effective_n_action_steps": CONTROL_N_ACTION_STEPS,
        "seed": CONTROL_SEED,
        "n_episodes": CONTROL_EPISODES,
        "config_hash": config_hash,
        "output_dir": str(FINAL_OUTPUT_DIR),
        "log_paths": {
            "stdout": str(FINAL_OUTPUT_DIR / 'stdout.log'),
            "stderr": str(FINAL_OUTPUT_DIR / 'stderr.log'),
            "returncode": str(FINAL_OUTPUT_DIR / 'returncode.txt'),
            "eval_info": str(FINAL_OUTPUT_DIR / 'eval_info.json'),
            "variant_summary": str(FINAL_OUTPUT_DIR / 'variant_summary.json'),
        },
        "video_integrity": video_integrity,
        "video_paths": [item["path"] for item in video_integrity.get("records", [])],
        "eval_info": eval_info,
        "aggregate_metrics": None if eval_info is None else eval_info.get("overall"),
    }
    write_json_atomic(ARTIFACT_PATH, artifact)
    upsert_evidence(
        EVIDENCE_ID,
        EXPERIMENT_ID,
        "control_rerun_atomic",
        ARTIFACT_PATH,
        "Clean direct-Python official control rerun with atomic bookkeeping and explicit video integrity checks.",
        verified=passed,
    )
    print(json.dumps(artifact, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
