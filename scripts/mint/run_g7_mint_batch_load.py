#!/usr/bin/env python3
"""G7: MINT loads one batch from LeRobot dataset."""
import os, sys, json, time, torch
os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import lerobot_policy_mint

CAMPAIGN = "/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1"
ARTIFACT = os.path.join(CAMPAIGN, "artifacts", "g7_batch_check.json")
DATASET_ROOT = os.path.join(CAMPAIGN, "dataset")

def run():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    try:
        ds = LeRobotDataset(repo_id="infinigen_drawer_test", root=DATASET_ROOT)
        item = ds[0]
        shapes = {k: list(v.shape) if hasattr(v, 'shape') else str(type(v)) for k, v in item.items()}
        has_nan = False
        for k, v in item.items():
            if hasattr(v, 'isnan'):
                if v.isnan().any():
                    has_nan = True

        result = {
            "gate": "g7_mint_batch_load",
            "passed": not has_nan and len(ds) > 0,
            "dataset_size": len(ds),
            "sample_shapes": shapes,
            "has_nan": has_nan,
            "features": list(item.keys()),
        }
    except Exception as e:
        result = {"gate": "g7_mint_batch_load", "passed": False, "error": str(e)}

    result["timestamp"] = time.time()
    with open(ARTIFACT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result["passed"]

if __name__ == "__main__":
    sys.exit(0 if run() else 1)
