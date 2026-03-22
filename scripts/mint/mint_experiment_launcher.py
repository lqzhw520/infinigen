#!/usr/bin/env python3
"""MINT campaign launcher: runs gates in sequence with auto-repair."""
import json, os, sys, subprocess, time
from pathlib import Path

SCRIPTS = Path("/mnt/afs2/zhuhaowu/infinigen/scripts/mint")
CAMPAIGN = Path("/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1")
CONDA = "source /root/anaconda3/etc/profile.d/conda.sh"

GATE_SCRIPTS = {
    "g1_asset_load": ("run_g1_asset_load.py", "mint"),
    "g2_obs_contract": ("run_g2_obs_contract.py", "mint"),
    "g3_grasp_plan": ("run_g3_grasp_plan.py", "graspnet"),
    "g4_trajectory_replay": ("run_g4_trajectory_replay.py", "mint"),
    "g5_delta_reconstruction": ("run_g5_delta_reconstruction.py", "mint"),
    "g6_lerobot_pack": ("run_g6_lerobot_pack.py", "mint"),
    "g7_mint_batch_load": ("run_g7_mint_batch_load.py", "mint"),
    "g8_mint_train": ("run_g8_mint_train.py", "mint"),
    "g9_sim_eval": ("run_g9_sim_eval.py", "mint"),
}

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(CAMPAIGN / "runtime" / "launcher.log", "a") as f:
        f.write(line + "\n")

def load_state():
    with open(CAMPAIGN / "state.json") as f:
        return json.load(f)

def save_state(state):
    with open(CAMPAIGN / "state.json", "w") as f:
        json.dump(state, f, indent=2)

def run_gate(gate_id):
    script_name, env = GATE_SCRIPTS[gate_id]
    script_path = SCRIPTS / script_name
    cmd = f'{CONDA} && conda activate {env} && python {script_path}'

    # Update state
    state = load_state()
    for step in state["queue"]:
        if step["id"] == gate_id:
            step["status"] = "running"
    state["active_job"] = {
        "gate": gate_id,
        "env": env,
        "started": time.time(),
    }
    save_state(state)

    log(f"Running {gate_id} in env={env}")
    result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=600)

    # Check artifact
    artifact_path = CAMPAIGN / "artifacts" / f"{gate_id}.json"
    if artifact_path.exists():
        with open(artifact_path) as f:
            artifact = json.load(f)
        passed = artifact.get("passed", False)
    else:
        passed = False
        artifact = {"error": "no artifact produced", "stderr": result.stderr[-500:]}

    # Update state
    state = load_state()
    for step in state["queue"]:
        if step["id"] == gate_id:
            step["status"] = "completed" if passed else "failed"
            step["result"] = "pass" if passed else "fail"
    state["active_job"] = None
    state["history"].append({
        "gate": gate_id,
        "result": "pass" if passed else "fail",
        "timestamp": time.time(),
        "env": env,
    })
    save_state(state)

    log(f"  {gate_id}: {'PASS' if passed else 'FAIL'}")
    return passed

def main():
    log("=" * 40)
    log("MINT Campaign Launcher started")

    state = load_state()
    for step in state["queue"]:
        gate_id = step["id"]
        if gate_id == "write_claim_memo":
            continue
        if step["status"] in ("completed",):
            log(f"  {gate_id}: already completed, skipping")
            continue

        passed = run_gate(gate_id)
        if not passed:
            log(f"  STOPPED at {gate_id}")
            state = load_state()
            state["verdict"] = "implementation_blocked"
            save_state(state)
            return False

    # All gates passed - write claim memo
    log("All gates passed. Writing claim memo...")
    state = load_state()
    # Check G9 result
    g9_path = CAMPAIGN / "evaluation" / "comparison_summary.json"
    if g9_path.exists():
        with open(g9_path) as f:
            g9 = json.load(f)
        verdict = "sim_eval_complete" if g9.get("passed") else "scientific_not_supported"
    else:
        verdict = "implementation_blocked"

    state["verdict"] = verdict
    save_state(state)
    log(f"VERDICT: {verdict}")
    return True

if __name__ == "__main__":
    main()
