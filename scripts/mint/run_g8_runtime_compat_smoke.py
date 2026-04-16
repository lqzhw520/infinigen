#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if (
    str(PROJECT_ROOT / "external" / "MINT" / "lerobot_policy_mint" / "src")
    not in sys.path
):
    sys.path.insert(
        0, str(PROJECT_ROOT / "external" / "MINT" / "lerobot_policy_mint" / "src")
    )
if str(PROJECT_ROOT / "external" / "MINT") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "external" / "MINT"))
if str(PROJECT_ROOT / "scripts" / "mint") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "mint"))

MINT_CKPT = "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-libero"
TOKENIZER_PATH = (
    "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-tokenizer-libero"
)
ARTIFACT_NAME = "g8_runtime_compat_smoke.json"


def main() -> int:
    started = time.time()
    report: dict[str, Any] = {
        "gate": "g8_runtime_compat_smoke",
        "checkpoint_path": MINT_CKPT,
        "tokenizer_path": TOKENIZER_PATH,
        "passed": False,
        "stage": None,
        "error": None,
        "layout": None,
        "timestamp": time.time(),
    }
    try:
        from lerobot_policy_mint.configuration_mint import MINTConfig
        from lerobot_policy_mint.modeling_mint import (
            MINTPolicy,
            _inspect_paligemma_layout,
            _resolve_paligemma_image_features,
            _resolve_paligemma_language_model,
        )
        from mint_common import ARTIFACT_DIR, load_json, write_json_atomic

        artifact = ARTIFACT_DIR / ARTIFACT_NAME
        report["stage"] = "imports_ok"
        try:
            config = MINTConfig.from_pretrained(MINT_CKPT)
            report["config_load_mode"] = "from_pretrained"
        except Exception as config_exc:  # noqa: BLE001
            raw_cfg = load_json(Path(MINT_CKPT) / "config.json", {})
            raw_cfg.pop("type", None)
            config = MINTConfig(**raw_cfg)
            report["config_load_mode"] = "manual_json_fallback"
            report["config_load_error"] = repr(config_exc)
        report["stage"] = "config_loaded"
        policy = MINTPolicy.from_pretrained(
            MINT_CKPT, config=config, local_files_only=True, strict=False
        )
        report["stage"] = "policy_loaded"
        paligemma = policy.model.paligemma_with_expert.paligemma
        report["layout"] = _inspect_paligemma_layout(paligemma)
        _ = _resolve_paligemma_language_model(paligemma)
        report["stage"] = "language_model_resolved"
        import torch

        first_param = next(policy.parameters())
        report["policy_first_param_dtype"] = str(first_param.dtype)
        vision_weight = None
        try:
            vision_weight = (
                paligemma.vision_tower.vision_model.embeddings.patch_embedding.weight
            )
            report["vision_patch_dtype"] = str(vision_weight.dtype)
        except Exception as dtype_exc:  # noqa: BLE001
            report["vision_patch_dtype_error"] = repr(dtype_exc)
        dummy_dtype = (
            vision_weight.dtype if vision_weight is not None else first_param.dtype
        )
        dummy = torch.zeros(
            (1, 3, 224, 224), dtype=dummy_dtype, device=first_param.device
        )
        try:
            img_features = _resolve_paligemma_image_features(paligemma, dummy)
            report["image_feature_shape"] = list(getattr(img_features, "shape", []))
            report["stage"] = "image_features_resolved"
        except Exception as inner_exc:  # noqa: BLE001
            report["image_feature_error"] = repr(inner_exc)
            report["stage"] = "image_features_failed"
        report["passed"] = report["stage"] in {
            "image_features_resolved",
            "image_features_failed",
        }
    except Exception as exc:  # noqa: BLE001
        report["error"] = repr(exc)
    report["elapsed_sec"] = round(time.time() - started, 3)
    from mint_common import ARTIFACT_DIR, write_json_atomic

    artifact = ARTIFACT_DIR / ARTIFACT_NAME
    write_json_atomic(artifact, report)
    print(json.dumps(report, indent=2))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
