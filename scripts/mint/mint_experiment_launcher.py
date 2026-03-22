#!/usr/bin/env python3
"""MINT campaign launcher: runs gates in sequence with auto-repair."""
import json, os, sys, subprocess, time
from pathlib import Path

SCRIPTS = Path("/mnt/afs2/zhuhaowu/infinigen/scripts/mint")
CAMPAIGN = Path("/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1")
CONDA = "source /root/anaconda3/etc/profile.d/conda.sh"

GATE_STEPS = [
    ("g1_asset_load", "run_g1_asset_load.py", "mint"),
    ("g2_obs_contract", "run_g2_obs_contract.py", "mint"),
    ("g3_render_depth", "run_g3_render_depth.py", "mint"),
    ("g3_grasp_plan", "run_g3_grasp_plan.py", "graspnet"),
    ("g4_trajectory_replay", "run_g4_trajectory_replay.py", "mint"),
    ("g5_delta_reconstruction", "run_g5_delta_reconstruction.py", "mint"),
    ("g6_lerobot_pack", "run_g6_lerobot_pack.py", "mint"),
    ("g7_mint_batch_load", "run_g7_mint_batch_load.py", "mint"),
    ("g8_mint_train", "run_g8_mint_train.py", "mint"),
    ("g9_sim_eval", "run_g9_sim_eval.py", "mint"),
]

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(CAMPAIGN / "runtime" / "launcher.log", "a") as f:
        f.write(line + "\n")

def run_step(step_id, script_name, env):
    cmd = f'{CONDA} && conda activate {env} && python {SCRIPTS / script_name}'
    log(f"Running {step_id} in env={env}")
    result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=600)

    artifact_path = CAMPAIGN / "artifacts" / f"{step_id}.json"
    passed = False
    if artifact_path.exists():
        with open(artifact_path) as f:
            artifact = json.load(f)
        passed = artifact.get("passed", False)
    else:
        log(f"  No artifact for {step_id}. stderr: {result.stderr[-200:]}")

    log(f"  {step_id}: {'PASS' if passed else 'FAIL'}")
    return passed

def main():
    log("=" * 40)
    log("MINT Campaign Launcher started")
    os.makedirs(CAMPAIGN / "runtime", exist_ok=True)

    for step_id, script_name, env in GATE_STEPS:
        passed = run_step(step_id, script_name, env)
        if not passed:
            log(f"STOPPED at {step_id}")
            state_path = CAMPAIGN / "state.json"
            if state_path.exists():
                with open(state_path) as f:
                    state = json.load(f)
                state["verdict"] = "implementation_blocked"
                with open(state_path, "w") as f:
                    json.dump(state, f, indent=2)
            return False

    log("All gates passed!")
    return True

if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
