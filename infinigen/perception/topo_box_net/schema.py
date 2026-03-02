from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class URDFJoint:
    name: str
    joint_type: str  # fixed | revolute | continuous | prismatic
    parent: str
    child: str
    origin_xyz: np.ndarray  # (3,)
    origin_rpy: np.ndarray  # (3,)
    axis_xyz: np.ndarray  # (3,) (default [1,0,0] if missing)
    limit_lower: Optional[float]
    limit_upper: Optional[float]


@dataclass(frozen=True)
class URDFModel:
    robot_name: str
    links: List[str]
    joints: List[URDFJoint]

    def controlled_joints(self) -> List[URDFJoint]:
        return [j for j in self.joints if j.joint_type in ("revolute", "continuous", "prismatic")]


@dataclass(frozen=True)
class CameraIntrinsics:
    HW: Tuple[int, int]  # (H, W)
    K: np.ndarray  # (3,3)


@dataclass(frozen=True)
class Phase1Sample:
    """
    A single Phase-1 sample exported by `scripts/export_mailerbox_simple_phase1_data_engine.py`.
    """

    # Raster inputs
    rgb_u8: np.ndarray  # (H,W,3) uint8
    depth_m: np.ndarray  # (H,W) float32
    seg_link_id: np.ndarray  # (H,W) int64
    instance_id: np.ndarray  # (H,W) int64

    # Geometry / physics metadata
    urdf: URDFModel
    camera: CameraIntrinsics
    joint_positions: Dict[str, float]  # URDF joint name -> position

    # Optional label map
    link_name_to_id: Dict[str, int]
    label_records: List[Dict]

