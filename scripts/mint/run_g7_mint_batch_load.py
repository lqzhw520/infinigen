#!/usr/bin/env python3
"""G7: Validate that the LeRobot dataset and MINT preprocessor can load a batch."""

from __future__ import annotations

import json
import time

import torch
from mint_common import ARTIFACT_DIR, DATASET_DIR, DATASET_REPO_ID
from torch.utils.data import DataLoader

ARTIFACT = ARTIFACT_DIR / "g7_batch_check.json"
MINT_CKPT = "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-libero"


def shape_of(value):
    if hasattr(value, "shape"):
        return list(value.shape)
    return str(type(value))


def run() -> bool:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot_policy_mint.modeling_mint import MINTPolicy

    try:
        dataset = LeRobotDataset(
            repo_id=DATASET_REPO_ID, root=DATASET_DIR, revision="main"
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        batch_shapes = {key: shape_of(value) for key, value in batch.items()}

        policy = MINTPolicy.from_pretrained(
            MINT_CKPT, local_files_only=True, dataset_stats=dataset.meta.stats
        )
        preprocessor, _ = make_pre_post_processors(
            policy.config, pretrained_path=MINT_CKPT, dataset_stats=dataset.meta.stats
        )
        processed = preprocessor(batch)
        processed_shapes = {key: shape_of(value) for key, value in processed.items()}
        nan_keys = []
        for key, value in processed.items():
            if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                nan_keys.append(key)

        result = {
            "gate": "g7_mint_batch_load",
            "passed": not nan_keys,
            "dataset_size": len(dataset),
            "batch_shapes": batch_shapes,
            "processed_shapes": processed_shapes,
            "nan_keys": nan_keys,
        }
    except Exception as exc:
        result = {
            "gate": "g7_mint_batch_load",
            "passed": False,
            "error": str(exc),
        }
    result["timestamp"] = time.time()
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return bool(result["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
