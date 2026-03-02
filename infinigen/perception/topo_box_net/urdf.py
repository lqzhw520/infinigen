from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .schema import URDFJoint, URDFModel


def _parse_xyz(text: Optional[str]) -> np.ndarray:
    if text is None:
        return np.zeros(3, dtype=float)
    vals = [float(x) for x in str(text).split()]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 floats, got {vals}")
    return np.asarray(vals, dtype=float)


def _parse_origin(elem: Optional[ET.Element]) -> Tuple[np.ndarray, np.ndarray]:
    if elem is None:
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)
    xyz = _parse_xyz(elem.get("xyz", "0 0 0"))
    rpy = _parse_xyz(elem.get("rpy", "0 0 0"))
    return xyz, rpy


def parse_urdf(urdf_path: Path) -> URDFModel:
    """
    Minimal URDF parser focused on articulated topology and kinematic parameters.

    It extracts:
    - link names
    - joint list with parent/child, origin, axis, limits
    """
    urdf_path = Path(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    robot_name = root.get("name", "robot")
    links = [str(x.get("name")) for x in root.findall("link") if x.get("name") is not None]

    joints = []
    for j in root.findall("joint"):
        name = j.get("name", "joint")
        joint_type = j.get("type", "fixed")

        parent_elem = j.find("parent")
        child_elem = j.find("child")
        parent = parent_elem.get("link") if parent_elem is not None else "world"
        child = child_elem.get("link") if child_elem is not None else "link"

        origin_xyz, origin_rpy = _parse_origin(j.find("origin"))
        axis = _parse_xyz(j.find("axis").get("xyz") if j.find("axis") is not None else "1 0 0")

        limit_lower: Optional[float] = None
        limit_upper: Optional[float] = None
        lim = j.find("limit")
        if lim is not None:
            if lim.get("lower") is not None:
                limit_lower = float(lim.get("lower"))
            if lim.get("upper") is not None:
                limit_upper = float(lim.get("upper"))

        joints.append(
            URDFJoint(
                name=str(name),
                joint_type=str(joint_type),
                parent=str(parent),
                child=str(child),
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                axis_xyz=axis,
                limit_lower=limit_lower,
                limit_upper=limit_upper,
            )
        )

    return URDFModel(robot_name=str(robot_name), links=links, joints=joints)

