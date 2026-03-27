#!/usr/bin/env python3
"""Helpers for staging and invoking AnyGrasp detection without modifying the upstream SDK."""

from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

SDK_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen/external/anygrasp_sdk")
WRAPPER_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen/scripts/mint")
VENDOR_ROOT = WRAPPER_ROOT / "vendor"
MINKOWSKI_RUNTIME_ROOT = VENDOR_ROOT / "minkowski_runtime"
DETECTION_DIR = SDK_ROOT / "grasp_detection"
LICENSE_ZIP = SDK_ROOT / "license_HaowuZhu.zip"
DETECTION_CKPT = SDK_ROOT / "checkpoint_detection.tar"
TRACKING_CKPT = SDK_ROOT / "checkpoint_tracking.tar"
PY310_GSNET = DETECTION_DIR / "gsnet_versions" / "gsnet.cpython-310-x86_64-linux-gnu.so"
PY310_LIBCXX = (
    SDK_ROOT
    / "license_registration"
    / "lib_cxx_versions"
    / "lib_cxx.cpython-310-x86_64-linux-gnu.so"
)
STAGED_GSNET = DETECTION_DIR / "gsnet.so"
STAGED_LIBCXX = DETECTION_DIR / "lib_cxx.so"
STAGED_LICENSE = DETECTION_DIR / "license"
STAGED_LOG = DETECTION_DIR / "log"
STAGED_CKPT = STAGED_LOG / "checkpoint_detection.tar"


def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)


def _safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def stage_detection_assets() -> dict[str, Any]:
    report: dict[str, Any] = {
        "sdk_root": str(SDK_ROOT),
        "checkpoint_detection_path": str(DETECTION_CKPT),
        "checkpoint_tracking_path": str(TRACKING_CKPT),
    }
    if not LICENSE_ZIP.exists():
        report["error"] = f"Missing {LICENSE_ZIP}"
        report["passed"] = False
        return report
    if not DETECTION_CKPT.exists():
        report["error"] = f"Missing {DETECTION_CKPT}"
        report["passed"] = False
        return report
    if not PY310_GSNET.exists():
        report["error"] = f"Missing {PY310_GSNET}"
        report["passed"] = False
        return report
    if not PY310_LIBCXX.exists():
        report["error"] = f"Missing {PY310_LIBCXX}"
        report["passed"] = False
        return report

    if STAGED_LICENSE.exists():
        shutil.rmtree(STAGED_LICENSE)
    STAGED_LICENSE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(LICENSE_ZIP) as zf:
        zf.extractall(STAGED_LICENSE)

    _safe_copy(PY310_GSNET, STAGED_GSNET)
    _safe_copy(PY310_LIBCXX, STAGED_LIBCXX)
    STAGED_LOG.mkdir(parents=True, exist_ok=True)
    _safe_symlink(DETECTION_CKPT, STAGED_CKPT)

    report.update(
        {
            "license_files": sorted(
                p.name for p in STAGED_LICENSE.iterdir() if p.is_file()
            ),
            "staged_gsnet": str(STAGED_GSNET),
            "staged_lib_cxx": str(STAGED_LIBCXX),
            "staged_checkpoint": str(STAGED_CKPT),
            "passed": True,
        }
    )
    return report


def build_anygrasp() -> Any:
    if str(VENDOR_ROOT) not in sys.path:
        sys.path.insert(0, str(VENDOR_ROOT))
    if MINKOWSKI_RUNTIME_ROOT.exists() and str(MINKOWSKI_RUNTIME_ROOT) not in sys.path:
        sys.path.insert(0, str(MINKOWSKI_RUNTIME_ROOT))
    sys.path.insert(0, str(DETECTION_DIR))
    from gsnet import AnyGrasp

    cfg = SimpleNamespace(
        checkpoint_path=str(STAGED_CKPT),
        max_gripper_width=0.1,
        gripper_height=0.03,
        top_down_grasp=False,
        debug=False,
    )
    detector = AnyGrasp(cfg)
    detector.load_net()
    return detector


def run_anygrasp(
    points: np.ndarray, colors: np.ndarray, lims: np.ndarray, max_candidates: int = 20
) -> list[dict[str, Any]]:
    detector = build_anygrasp()
    gg, _cloud = detector.get_grasp(
        points.astype(np.float32),
        colors.astype(np.float32),
        lims=lims.astype(np.float32).tolist(),
        apply_object_mask=True,
        dense_grasp=False,
        collision_detection=True,
    )
    if len(gg) == 0:
        return []
    gg = gg.nms().sort_by_score()
    candidates: list[dict[str, Any]] = []
    for idx in range(min(len(gg), max_candidates)):
        grasp = gg[idx]
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = np.asarray(grasp.rotation_matrix, dtype=np.float32)
        pose[:3, 3] = np.asarray(grasp.translation, dtype=np.float32)
        candidates.append(
            {
                "score": float(grasp.score),
                "width": float(getattr(grasp, "width", 0.0)),
                "height": float(getattr(grasp, "height", 0.0)),
                "depth": float(getattr(grasp, "depth", 0.0)),
                "translation": pose[:3, 3].tolist(),
                "rotation_matrix": pose[:3, :3].tolist(),
                "pose": pose.tolist(),
            }
        )
    return candidates
