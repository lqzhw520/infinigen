from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .schema import CameraIntrinsics, Phase1Sample
from .urdf import parse_urdf


def _read_rgb_png(path: Path) -> np.ndarray:
    import imageio.v2 as imageio

    arr = imageio.imread(path)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape={arr.shape} from {path}")
    return arr.astype(np.uint8)


def load_phase1_sample(sample_dir: Path) -> Phase1Sample:
    """
    Load a single Phase-1 sample folder produced by the Phase-1 exporter script.

    The loader is intentionally strict about shapes/dtypes so downstream training code
    can rely on invariants.
    """
    sample_dir = Path(sample_dir)

    rgb_u8 = _read_rgb_png(sample_dir / "rgb.png")
    depth_m = np.load(sample_dir / "depth.npy").astype(np.float32)
    seg_link_id = np.load(sample_dir / "segmentation.npy").astype(np.int64)
    instance_id = np.load(sample_dir / "instance.npy").astype(np.int64)

    if depth_m.shape != seg_link_id.shape or depth_m.shape != instance_id.shape:
        raise ValueError(
            f"Shape mismatch: depth={depth_m.shape} seg={seg_link_id.shape} instance={instance_id.shape} in {sample_dir}"
        )

    camj = json.loads((sample_dir / "camera_intrinsics.json").read_text())
    K = np.asarray(camj["K"], dtype=np.float32)
    HW = tuple(int(x) for x in camj["HW"])
    camera = CameraIntrinsics(HW=HW, K=K)

    urdf = parse_urdf(sample_dir / "urdf_gt.urdf")

    js = json.loads((sample_dir / "joint_state.json").read_text())
    joint_positions: Dict[str, float] = {str(k): float(v) for k, v in js.get("joint_positions", {}).items()}

    lm = json.loads((sample_dir / "segmentation_label_map.json").read_text())
    link_name_to_id: Dict[str, int] = {str(k): int(v) for k, v in lm.get("link_name_to_id", {}).items()}
    label_records: List[Dict] = lm.get("labels", [])

    return Phase1Sample(
        rgb_u8=rgb_u8,
        depth_m=depth_m,
        seg_link_id=seg_link_id,
        instance_id=instance_id,
        urdf=urdf,
        camera=camera,
        joint_positions=joint_positions,
        link_name_to_id=link_name_to_id,
        label_records=label_records,
    )


def list_phase1_samples(dataset_root: Path, split: str = "train") -> List[Path]:
    """
    List sample directories under:
      <dataset_root>/<split>/<sample_id>/
    """
    dataset_root = Path(dataset_root)
    split_dir = dataset_root / split
    if not split_dir.exists():
        return []
    out: List[Path] = []
    for p in sorted(split_dir.iterdir()):
        if p.is_dir() and (p / "metadata.json").exists():
            out.append(p)
    return out

