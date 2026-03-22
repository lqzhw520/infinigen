#!/usr/bin/env python3
"""G9: Evaluate pretrained vs finetuned MINT on held-out drawer objects."""

import json
import os
import sys
import time

os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

CAMPAIGN = "/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1"
ARTIFACT = os.path.join(CAMPAIGN, "evaluation", "comparison_summary.json")
REPORT = os.path.join(CAMPAIGN, "evaluation", "comparison_report.md")
FT_CKPT = os.path.join(CAMPAIGN, "outputs")
PRETRAINED = "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-libero"


def run():
    # Placeholder: full eval requires robosuite env with drawer objects
    # This creates the comparison structure
    comparison = {
        "gate": "g9_sim_eval",
        "passed": False,
        "status": "placeholder",
        "note": "Full G9 eval requires InfinigenDrawerEnv (G1-G5 must pass first)",
        "baselines": {
            "random": {"success_rate": None, "n_episodes": 0},
            "pretrained_mint": {"success_rate": None, "n_episodes": 0},
            "finetuned_mint": {"success_rate": None, "n_episodes": 0},
        },
        "metrics": [
            "success_rate",
            "pull_distance",
            "grasp_success",
            "time_to_completion",
        ],
        "held_out_seeds": [11, 12, 13, 14, 15],
        "timestamp": time.time(),
    }
    with open(ARTIFACT, "w") as f:
        json.dump(comparison, f, indent=2)

    report = "# MINT Sim Eval Comparison\n\nPlaceholder — G1-G5 must pass first.\n"
    with open(REPORT, "w") as f:
        f.write(report)

    print(json.dumps(comparison, indent=2))
    return False  # placeholder, will be implemented after G1-G5 pass


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
