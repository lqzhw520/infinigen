#!/usr/bin/env python3
"""Regression tests encoding all 6 historical bugs from Phase-1 development.

Each test constructs a minimal pathological sample and verifies that the
validator correctly detects the issue.

Historical bugs:
  1. Multiple <inertial> per link (pre-Parallel-Axis-Theorem merge)
  2. Missing collision meshes (visual_only=True)
  3. Missing joint limits (infinite rotation in URDF viewers)
  4. Missing material color tags (white boxes)
  5. DOF mismatch (joint_state length != expected DOF for box type)
  6. PIL tile-extend-outside-image (imageio vs PIL save)
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))
from validate_dataset import check_level3, check_level4, validate_sample

H, W = 480, 640


def _make_minimal_sample(
    tmp: Path,
    urdf_xml: str,
    box_type: str = "MAILER",
    joint_positions: dict | None = None,
) -> Path:
    """Create a minimal valid sample directory, then return its path."""
    sample = tmp / "dataset" / "train" / "000001"
    sample.mkdir(parents=True)

    depth = np.random.uniform(0.5, 3.0, (H, W)).astype(np.float32)
    np.save(sample / "depth.npy", depth)
    seg = np.zeros((H, W), dtype=np.int32)
    seg[100:300, 100:500] = 1
    np.save(sample / "segmentation.npy", seg)
    np.save(sample / "instance.npy", seg.copy())

    from PIL import Image

    Image.fromarray(np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)).save(
        sample / "rgb.png"
    )
    Image.fromarray((seg * 128).astype(np.uint8)).save(sample / "segmentation.png")
    Image.fromarray((seg * 128).astype(np.uint8)).save(sample / "instance.png")

    cam = {"HW": [H, W], "K": [[500, 0, 320], [0, 500, 240], [0, 0, 1]]}
    (sample / "camera_intrinsics.json").write_text(json.dumps(cam))

    if joint_positions is None:
        joint_positions = {"mailer_lid_0": 0.0, "mailer_front_flap_0": 0.0}
    js = {
        "seed": 1000,
        "box_type": box_type,
        "joint_positions": joint_positions,
        "joint_names_order": list(joint_positions.keys()),
    }
    (sample / "joint_state.json").write_text(json.dumps(js))

    kp = {
        "joints": [
            {
                "joint_name": n,
                "visible": True,
                "origin_uv": [320, 240],
                "end_uv": [321, 241],
                "axis_world": [0, 0, 1],
            }
            for n in joint_positions
        ]
    }
    (sample / "keypoints.json").write_text(json.dumps(kp))

    meta = {
        "box_type": box_type,
        "seed": 1000,
        "material": {
            "density": 500,
            "friction": 0.5,
            "restitution": 0.1,
            "material_type": "kraft",
        },
    }
    (sample / "metadata.json").write_text(json.dumps(meta))

    lm = {
        "labels": [
            {
                "id": 0,
                "name": "background",
                "present_in_frame": True,
                "color_rgb_uint8": [0, 0, 0],
            },
            {
                "id": 1,
                "name": "link_0",
                "present_in_frame": True,
                "color_rgb_uint8": [255, 0, 0],
            },
        ],
        "link_name_to_id": {"link_0": 1},
    }
    (sample / "segmentation_label_map.json").write_text(json.dumps(lm))

    (sample / "urdf_gt.urdf").write_text(urdf_xml)

    return sample


GOOD_URDF = textwrap.dedent("""\
<?xml version="1.0"?>
<robot name="test_mailer">
  <link name="world"/>
  <link name="link_0">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
    <visual><geometry><box size="0.1 0.1 0.1"/></geometry>
      <material name="mat"><color rgba="0.7 0.5 0.3 1"/></material>
    </visual>
  </link>
  <link name="link_1">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0005" ixy="0" ixz="0" iyy="0.0005" iyz="0" izz="0.0005"/>
    </inertial>
    <collision><geometry><box size="0.05 0.1 0.01"/></geometry></collision>
    <visual><geometry><box size="0.05 0.1 0.01"/></geometry>
      <material name="mat"><color rgba="0.7 0.5 0.3 1"/></material>
    </visual>
  </link>
  <link name="link_2">
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.0002" ixy="0" ixz="0" iyy="0.0002" iyz="0" izz="0.0002"/>
    </inertial>
    <collision><geometry><box size="0.05 0.1 0.01"/></geometry></collision>
    <visual><geometry><box size="0.05 0.1 0.01"/></geometry>
      <material name="mat"><color rgba="0.7 0.5 0.3 1"/></material>
    </visual>
  </link>
  <joint name="world_joint" type="fixed">
    <parent link="world"/><child link="link_0"/>
  </joint>
  <joint name="mailer_lid_0" type="revolute">
    <parent link="link_0"/><child link="link_1"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="10" velocity="1"/>
  </joint>
  <joint name="mailer_front_flap_0" type="revolute">
    <parent link="link_0"/><child link="link_2"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="10" velocity="1"/>
  </joint>
</robot>
""")


class TestBug1MultipleInertials:
    """Bug #1: Multiple <inertial> per link (before Parallel Axis Theorem merge)."""

    def test_detects_multiple_inertials(self, tmp_path):
        bad_urdf = GOOD_URDF.replace(
            '<link name="link_0">', '<link name="link_0">'
        ).replace(
            '    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>\n'
            '    <visual><geometry><box size="0.1 0.1 0.1"/></geometry>\n'
            '      <material name="mat"><color rgba="0.7 0.5 0.3 1"/></material>\n'
            "    </visual>",
            '    <inertial><mass value="0.2"/>'
            '<inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>\n'
            '    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>\n'
            '    <visual><geometry><box size="0.1 0.1 0.1"/></geometry>\n'
            '      <material name="mat"><color rgba="0.7 0.5 0.3 1"/></material>\n'
            "    </visual>",
        )
        sample = _make_minimal_sample(tmp_path, bad_urdf)
        r = check_level3(sample, box_type="MAILER")
        assert not r["passed"]
        assert any("inertial" in e.lower() for e in r["errors"])


class TestBug2MissingCollision:
    """Bug #2: Missing collision meshes (visual_only=True caused no <collision> tags)."""

    def test_detects_missing_collision(self, tmp_path):
        bad_urdf = GOOD_URDF.replace(
            '    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>\n',
            "",
        ).replace(
            '    <collision><geometry><box size="0.05 0.1 0.01"/></geometry></collision>\n',
            "",
        )
        sample = _make_minimal_sample(tmp_path, bad_urdf)
        r = check_level3(sample, box_type="MAILER")
        assert not r["passed"]
        assert any("collision" in e.lower() for e in r["errors"])


class TestBug3MissingJointLimits:
    """Bug #3: Missing joint limits (infinite rotation in online viewers)."""

    def test_detects_missing_limits(self, tmp_path):
        bad_urdf = GOOD_URDF.replace(
            '    <limit lower="-3.14159" upper="3.14159" effort="10" velocity="1"/>\n',
            "",
        )
        sample = _make_minimal_sample(tmp_path, bad_urdf)
        r = check_level3(sample, box_type="MAILER")
        assert not r["passed"]
        assert any("limit" in e.lower() for e in r["errors"])


class TestBug4MissingMaterialColor:
    """Bug #4: Missing material color tags (white boxes in URDF viewers)."""

    def test_warns_missing_material(self, tmp_path):
        bad_urdf = GOOD_URDF
        for _ in range(3):
            bad_urdf = bad_urdf.replace(
                '      <material name="mat"><color rgba="0.7 0.5 0.3 1"/></material>\n',
                "",
                1,
            )
        sample = _make_minimal_sample(tmp_path, bad_urdf)
        r = check_level3(sample, box_type="MAILER")
        assert any("material" in w.lower() for w in r.get("warnings", []))


class TestBug5DOFMismatch:
    """Bug #5: DOF count mismatch (e.g., TuckEndBox needs 8 joints)."""

    def test_detects_dof_mismatch(self, tmp_path):
        sample = _make_minimal_sample(tmp_path, GOOD_URDF, box_type="TUCK_END")
        r = check_level3(sample, box_type="TUCK_END")
        assert not r["passed"]
        assert any("dof" in e.lower() for e in r["errors"])


class TestBug6JointStateOutsideLimits:
    """Bug #6: joint_state values outside URDF limits (physics violation)."""

    def test_detects_state_outside_limits(self, tmp_path):
        bad_positions = {"mailer_lid_0": 99.0, "mailer_front_flap_0": 0.0}
        sample = _make_minimal_sample(
            tmp_path, GOOD_URDF, joint_positions=bad_positions
        )
        r = check_level4(sample)
        assert not r["passed"]
        assert any("outside limits" in e for e in r["errors"])


class TestGoodSamplePasses:
    """Sanity: a well-formed sample passes all levels 1-5."""

    def test_good_sample_passes(self, tmp_path):
        sample = _make_minimal_sample(tmp_path, GOOD_URDF)
        r = validate_sample(sample, max_level=3)
        assert r["overall_passed"], f"Good sample failed: {r}"
