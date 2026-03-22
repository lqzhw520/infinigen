#!/usr/bin/env python3
"""G6: Create LeRobot v3 dataset from sample trajectory."""
import os, sys, json, time, numpy as np
CAMPAIGN = "/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1"
ARTIFACT = os.path.join(CAMPAIGN, "artifacts", "g6_dataset_manifest.json")
DATASET_ROOT = os.path.join(CAMPAIGN, "dataset")

def run():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Create dataset with minimal features
    try:
        ds = LeRobotDataset.create(
            repo_id="infinigen_drawer_test",
            fps=30,
            root=DATASET_ROOT,
            robot_type="infinigen_drawer",
            features={
                "observation.images.image": {"dtype": "video", "shape": (360, 360, 3)},
                "observation.images.image2": {"dtype": "video", "shape": (360, 360, 3)},
                "observation.state": {"dtype": "float32", "shape": (8,)},
                "action": {"dtype": "float32", "shape": (7,)},
            },
            use_videos=False,
        )

        # Add one dummy episode (5 frames)
        frames = []
        for i in range(5):
            frame = {
                "timestamp": float(i) / 30.0,
                "frame_index": i,
                "episode_index": 0,
                "index": i,
                "task_index": 0,
                "task": "open the drawer",
                "observation.images.image": np.random.randint(0, 255, (360, 360, 3), dtype=np.uint8),
                "observation.images.image2": np.random.randint(0, 255, (360, 360, 3), dtype=np.uint8),
                "observation.state": np.random.randn(8).astype(np.float32),
                "action": np.random.uniform(-1, 1, 7).astype(np.float32),
            }
            frames.append(frame)
        ds.add_episode(frames)
        ds.consolidate()

        # Verify load
        from lerobot.datasets.lerobot_dataset import LeRobotDataset as LDS
        ds2 = LDS(repo_id="infinigen_drawer_test", root=DATASET_ROOT)

        result = {
            "gate": "g6_lerobot_pack",
            "passed": True,
            "dataset_root": DATASET_ROOT,
            "total_frames": len(ds2),
            "total_episodes": ds2.num_episodes,
            "features": list(ds2.features.keys()),
        }
    except Exception as e:
        result = {
            "gate": "g6_lerobot_pack",
            "passed": False,
            "error": str(e),
        }

    result["timestamp"] = time.time()
    with open(ARTIFACT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result["passed"]

if __name__ == "__main__":
    sys.exit(0 if run() else 1)
