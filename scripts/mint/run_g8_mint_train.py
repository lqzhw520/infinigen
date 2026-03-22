#!/usr/bin/env python3
"""G8: MINT fine-tune on local LeRobot dataset."""
import os, sys, json, time
os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

CAMPAIGN = "/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1"
ARTIFACT = os.path.join(CAMPAIGN, "artifacts", "g8_train_summary.json")
OUTPUT_DIR = os.path.join(CAMPAIGN, "outputs")
DATASET_ROOT = os.path.join(CAMPAIGN, "dataset")
MINT_CKPT = "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-libero"

def run():
    import subprocess
    cmd = [
        "lerobot-train",
        f"--dataset.repo_id=infinigen_drawer_test",
        f"--dataset.root={DATASET_ROOT}",
        "--policy.type=mint",
        f"--output_dir={OUTPUT_DIR}",
        "--job_name=mint_drawer_ft",
        f"--policy.pretrained_path={MINT_CKPT}",
        "--policy.compile_model=false",
        "--policy.gradient_checkpointing=true",
        "--policy.dtype=bfloat16",
        "--steps=100",
        "--batch_size=2",
        "--policy.device=cuda",
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0

    success = result.returncode == 0
    # Check for loss in output
    loss_lines = [l for l in result.stdout.split("\n") if "loss" in l.lower()]

    output = {
        "gate": "g8_mint_train",
        "passed": success,
        "returncode": result.returncode,
        "elapsed_sec": round(elapsed, 1),
        "loss_samples": loss_lines[-5:] if loss_lines else [],
        "stderr_tail": result.stderr[-500:] if result.stderr else "",
    }
    output["timestamp"] = time.time()
    with open(ARTIFACT, "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))
    return output["passed"]

if __name__ == "__main__":
    sys.exit(0 if run() else 1)
