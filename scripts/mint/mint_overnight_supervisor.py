#!/usr/bin/env python3
"""Overnight supervisor: runs launcher, monitors, auto-restarts on failure."""

import json
import os
import subprocess
import time
from pathlib import Path

SCRIPTS = Path("/mnt/afs2/zhuhaowu/infinigen/scripts/mint")
CAMPAIGN = Path("/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1")
CONDA = "source /root/anaconda3/etc/profile.d/conda.sh"
LOG = CAMPAIGN / "runtime" / "supervisor.log"


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    max_restarts = 3
    restart_count = 0

    while restart_count < max_restarts:
        log(f"Starting launcher (attempt {restart_count + 1}/{max_restarts})")
        cmd = f'{CONDA} && conda activate mint && python {SCRIPTS / "mint_experiment_launcher.py"}'
        proc = subprocess.Popen(
            ["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )

        # Monitor
        while proc.poll() is None:
            # Write heartbeat
            heartbeat = {
                "supervisor_pid": os.getpid(),
                "launcher_pid": proc.pid,
                "restart_count": restart_count,
                "timestamp": time.time(),
            }
            with open(CAMPAIGN / "runtime" / "watch_status.json", "w") as f:
                json.dump(heartbeat, f, indent=2)

            # Check state
            state_file = CAMPAIGN / "state.json"
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                if state.get("verdict") in (
                    "sim_eval_complete",
                    "scientific_not_supported",
                ):
                    log(f"Terminal verdict: {state['verdict']}")
                    proc.terminate()
                    return

            time.sleep(30)

        exit_code = proc.returncode
        log(f"Launcher exited with code {exit_code}")

        # Check if terminal
        state_file = CAMPAIGN / "state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            if state.get("verdict") in (
                "sim_eval_complete",
                "scientific_not_supported",
                "implementation_blocked",
            ):
                log(f"Terminal state: {state['verdict']}")
                return

        restart_count += 1
        log("Restarting in 10 seconds...")
        time.sleep(10)

    log("Max restarts reached. Campaign stopped.")
    state = load_state() if (CAMPAIGN / "state.json").exists() else {}
    state["verdict"] = "implementation_blocked"
    with open(CAMPAIGN / "state.json", "w") as f:
        json.dump(state, f, indent=2)


if __name__ == "__main__":
    main()
