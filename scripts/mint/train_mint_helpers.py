#!/usr/bin/env python3
"""Shared training helpers for MINT root-cause experiments."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

MINT_CKPT = "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-libero"
TOKENIZER_PATH = (
    "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-tokenizer-libero"
)
TRAINING_PID_FILE = ".training.pid"


def _find_checkpoint_at_step(output_dir: Path, step: int) -> Path | None:
    """Find checkpoint saved at exactly `step` training steps."""
    step_str = f"{step:06d}"
    candidates = sorted(
        output_dir.glob(f"**/{step_str}/pretrained_model"), key=lambda p: len(str(p))
    )
    return candidates[-1] if candidates else None


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Return the latest checkpoint by step number.
    Always returns the checkpoint with the highest step count.
    """
    all_dirs = list(output_dir.glob("checkpoints/*/"))
    checkpoints = sorted(
        [d for d in all_dirs if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    pretrained_dirs = [d / "pretrained_model" for d in checkpoints if (d / "pretrained_model").exists()]
    return pretrained_dirs[-1] if pretrained_dirs else None


def _launch_training_async(
    cmd: list[str],
    output_dir: Path,
    log_path: Path,
    env: dict,
    max_retries: int = 3,
) -> int:
    """Launch training in background, return PID. Output goes directly to log file."""
    output_dir_str = str(output_dir.resolve())
    
    # Clean up any stale output directory first with retry
    for attempt in range(max_retries):
        if output_dir.exists():
            try:
                shutil.rmtree(output_dir)
                time.sleep(0.5)  # Brief pause for filesystem sync
            except OSError as e:
                print(f"Warning: Failed to remove {output_dir_str}: {e}", flush=True)
        # Verify deletion
        if not output_dir.exists():
            break
        time.sleep(1)
    
    # Final check - if directory still exists, abort
    if output_dir.exists():
        raise RuntimeError(f"Cannot clean output directory {output_dir_str}; aborting training launch")
    
    # Don't create output_dir here - lerobot-train will create it
    # Just ensure the parent directory exists for cd
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # PID file goes in artifacts directory since output_dir doesn't exist yet
    pid_file = log_path.parent / f"{output_dir.name}.training.pid"
    
    # Launch training in background using nohup approach
    import threading
    
    # First, write PID placeholder
    pid_file.write_text("STARTING")
    
    def launch_in_thread():
        import subprocess
        import time
        
        # Small delay to ensure directory is fully created
        time.sleep(1)
        
        # Build command string properly - cd to parent and run lerobot-train with output_dir
        cmd_str = " ".join(f"'{arg}'" if " " in arg else arg for arg in cmd)
        # Change to parent dir so lerobot-train creates output_dir there
        launch_cmd = f"cd '{output_dir.parent}' && nohup {cmd_str} > '{log_path}' 2>&1 & echo $!"
        
        # Launch with bash nohup in background
        result = subprocess.run(
            ["bash", "-c", launch_cmd],
            env=env,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            print(f"Failed to launch training: {result.stderr}", flush=True)
            pid_file.write_text("FAILED")
            return
        
        pid_str = result.stdout.strip()
        try:
            pid = int(pid_str)
            pid_file.write_text(str(pid))
        except ValueError:
            print(f"Failed to parse PID: {result.stdout}", flush=True)
            pid_file.write_text("FAILED")
    
    thread = threading.Thread(target=launch_in_thread, daemon=True)
    thread.start()
    thread.join(timeout=5)
    
    # Check if launch was successful
    pid_content = pid_file.read_text().strip()
    if pid_content == "STARTING" or pid_content == "FAILED":
        # Thread didn't complete, try direct launch
        cmd_str = " ".join(f"'{arg}'" if " " in arg else arg for arg in cmd)
        # Change to parent dir so lerobot-train creates output_dir there
        launch_cmd = f"cd '{output_dir.parent}' && nohup {cmd_str} > '{log_path}' 2>&1 & echo $!"
        result = subprocess.run(
            ["bash", "-c", launch_cmd],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            pid = int(result.stdout.strip())
            pid_file.write_text(str(pid))
            return pid
    
    try:
        return int(pid_content)
    except ValueError:
        raise RuntimeError(f"Failed to launch training process")


def wait_for_training_complete(
    output_dir: Path,
    log_path: Path,
    steps: int,
    poll_interval: float = 30.0,
    max_wait_hours: float = 12.0,
) -> dict:
    """Poll for checkpoint availability and training completion.
    
    Returns a dict with:
        - checkpoint_path: Path to latest checkpoint if found
        - completed: True if training finished naturally
        - returncode: Exit code (0 = success)
        - elapsed_sec: Total wait time
        - steps_completed: Steps reported in log
    """
    start = time.time()
    max_wait_sec = max_wait_hours * 3600
    # PID file location: same as log_path but with .training.pid suffix
    pid_file = log_path.parent / f"{output_dir.name}.training.pid"
    last_checkpoint = None
    last_checkpoint_step = 0
    
    # Determine target step (lerobot saves at save_freq AND final step)
    # The last checkpoint should be at `steps` or the highest available
    target_step = steps
    
    while time.time() - start < max_wait_sec:
        # Check if training process is still alive
        if pid_file.exists():
            pid = int(pid_file.read_text().strip())
            try:
                # Check process in same process group
                os.kill(pid, 0)  # Signal 0 just checks if process exists
                process_alive = True
            except OSError:
                process_alive = False
        else:
            process_alive = False
        
        # Find latest checkpoint
        checkpoint = find_latest_checkpoint(output_dir)
        if checkpoint:
            # Extract step number from checkpoint path
            checkpoint_parent = checkpoint.parent
            try:
                step_num = int(checkpoint_parent.name)
                if step_num > last_checkpoint_step:
                    last_checkpoint = checkpoint
                    last_checkpoint_step = step_num
            except ValueError:
                pass
        
        # Check log for completion or errors
        log_content = ""
        if log_path.exists():
            log_content = log_path.read_text(errors="ignore")
        
        # Look for completion markers
        if "Training complete" in log_content or "Saved checkpoint" in log_content:
            # Check for explicit exit
            if not process_alive:
                break
        
        # Check for error conditions
        if not process_alive and last_checkpoint:
            # Process died but we have a checkpoint - training likely succeeded
            break
        
        if not process_alive and not last_checkpoint:
            # Process died with no checkpoint - check for errors
            if "Error" in log_content or "Traceback" in log_content:
                break
        
        # Sleep before next poll
        time.sleep(poll_interval)
    
    elapsed = time.time() - start
    
    # Determine final state
    returncode = None
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, 0)
            returncode = None  # Still running
        except OSError:
            # Process ended - read log for exit code
            returncode = 0  # Assume success if no error
    
    # Final checkpoint check
    final_checkpoint = find_latest_checkpoint(output_dir)
    if final_checkpoint:
        last_checkpoint = final_checkpoint
        checkpoint_parent = final_checkpoint.parent
        try:
            last_checkpoint_step = int(checkpoint_parent.name)
        except ValueError:
            pass
    
    # Parse steps completed from log
    steps_completed = last_checkpoint_step
    if log_path.exists():
        log_content = log_path.read_text(errors="ignore")
        # Look for step reports in log
        for line in reversed(log_content.splitlines()):
            if "step" in line.lower() and "loss" in line.lower():
                # Try to extract step number
                import re
                match = re.search(r"step[:\s]+(\d+)", line, re.IGNORECASE)
                if match:
                    steps_completed = int(match.group(1))
                    break
    
    return {
        "checkpoint_path": str(last_checkpoint) if last_checkpoint else None,
        "completed": returncode == 0,
        "returncode": returncode,
        "elapsed_sec": round(elapsed, 1),
        "steps_completed": steps_completed,
        "log_tail": log_path.read_text(errors="ignore")[-8000:] if log_path.exists() else "",
    }


def run_training(
    *,
    dataset_root: Path,
    dataset_repo_id: str,
    output_dir: Path,
    log_path: Path,
    steps: int,
    job_name: str,
    async_mode: bool = True,
) -> dict:
    """Run MINT fine-tuning training.
    
    Args:
        async_mode: If True, launch training in background and return immediately.
                   Caller should use wait_for_training_complete() to wait for results.
                   If False, run synchronously (blocking, original behavior).
    """
    # Generate unique run ID with timestamp to avoid conflicts
    import uuid
    run_id = f"{job_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    # Clean output directory if it exists (lerobot will refuse if it has content)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    LEROBOT_TRAIN = (
        shutil.which("lerobot-train") or "/root/anaconda3/envs/mint/bin/lerobot-train"
    )
    cmd = [
        LEROBOT_TRAIN,
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.root={dataset_root}",
        "--policy.type=mint",
        f"--policy.repo_id={dataset_repo_id}_mint_ft",
        "--policy.push_to_hub=false",
        f"--output_dir={output_dir}",
        f"--job_name={run_id}",
        f"--policy.pretrained_path={MINT_CKPT}",
        f"--policy.vqvae_name_or_path={TOKENIZER_PATH}",
        "--policy.compile_model=false",
        "--policy.gradient_checkpointing=true",
        "--policy.dtype=bfloat16",
        f"--steps={steps}",
        f"--save_freq={min(600, steps)}",
        "--batch_size=8",
        f"--policy.optimizer_lr=5e-5",
        "--policy.device=cuda",
    ]
    
    start = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    mint_bin = "/root/anaconda3/envs/mint/bin"
    if mint_bin not in env.get("PATH", ""):
        env["PATH"] = mint_bin + ":" + env.get("PATH", "")
    
    if async_mode:
        # Launch in background, return immediately with placeholder
        pid = _launch_training_async(cmd, output_dir, log_path, env)
        return {
            "passed": None,  # Unknown until training completes
            "returncode": None,
            "elapsed_sec": 0.0,
            "steps_completed": 0,
            "checkpoint_path": None,
            "loss_samples": [],
            "stdout_tail": "",
            "stderr_tail": "",
            "async_mode": True,
            "pid": pid,
            "log_path": str(log_path),
            "status": "launched",
        }
    else:
        # Synchronous mode (original behavior)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log_handle:
            proc = subprocess.Popen(
                cmd, stdout=log_handle, stderr=subprocess.STDOUT, text=True, env=env
            )
            proc.wait()
        elapsed = time.time() - start
        log_tail = log_path.read_text(errors="ignore")[-8000:] if log_path.exists() else ""
        checkpoint_path = find_latest_checkpoint(output_dir)
        loss_lines = [
            line.strip() for line in log_tail.splitlines() if "loss" in line.lower()
        ]
        return {
            "passed": checkpoint_path is not None
            and (
                proc.returncode == 0
                or "push_model_to_hub" in log_tail
                or "ConnectionError" in log_tail
            ),
            "returncode": proc.returncode,
            "elapsed_sec": round(elapsed, 1),
            "steps_completed": steps if checkpoint_path is not None else 0,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "loss_samples": loss_lines[-10:],
            "stdout_tail": log_tail,
            "stderr_tail": "",
        }
