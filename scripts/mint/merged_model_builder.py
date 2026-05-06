"""
Merged Panda + Drawer MuJoCo Model Builder.

Builds a unified MuJoCo scene with:
- Panda robot (LIBERO's robot.xml with 7 actuators + 50 PBR materials)
- Infinigen drawer (with proper rgba colors per semantic mapping)
- Proper lighting, cameras, and scene setup

Usage:
    from merged_model_builder import MergedModelBuilder
    xml, assets = MergedModelBuilder(seed=1).build()
"""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
DRAWER_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen/sim_exports/urdf/drawer")
ROBOT_XML_PATH = "/root/anaconda3/envs/infinigen/lib/python3.11/site-packages/robosuite/models/assets/robots/panda/robot.xml"
GRIPPER_XML_PATH = "/root/anaconda3/envs/infinigen/lib/python3.11/site-packages/robosuite/models/assets/grippers/panda_gripper.xml"

# LIBERO default robot joint configuration (from mounted_panda.py)
LIBERO_INIT_QPOS = np.array(
    [0, -1.61037389e-01, 0.00, -2.44459747e00, 0.00, 2.22675220e00, np.pi / 4]
)

# Reset-clearance placement for the mounted Panda relative to the drawer model.
# This keeps GOC-v3 geom IDs stable while avoiding reset-time robot/cabinet
# interpenetration. It does not alter robot-drawer collision semantics.
ROBOT_BASE_POS = np.array([-0.9, 0.0, 0.0])

# LIBERO-style overhead camera (agentview equivalent)
LIBERO_CAMERA = dict(
    lookat=[0.0, 0.0, 0.45],
    distance=1.6,
    azimuth=90.0,  # Side view: shows both robot arm + cabinet
    elevation=-20.0,
)

# Drawer part colors (PBR-like, matching LIBERO aesthetic)
DRAWER_COLORS = {
    "drawer_base": np.array([0.85, 0.82, 0.80, 1.0]),  # Off-white/light gray
    "drawer_door": np.array([0.90, 0.88, 0.85, 1.0]),  # Lighter drawer front
    "drawer_handle": np.array([0.20, 0.20, 0.22, 1.0]),  # Dark metallic handle
}

# Background colors
BG_COLORS = {
    "neutral_lab": np.array([0.15, 0.15, 0.15]),  # Dark lab background (contrast)
    "neutral_lab_v2": np.array([0.88, 0.87, 0.85]),
}

# GOC-v4 repair: insert Robosuite's real Panda gripper subtree at the
# right_hand marker. The previous GOC-v4 prototype mounted two static pads on
# right_hand, which created exact target geoms but no articulated finger DOF.
# The real gripper adds finger_joint1/finger_joint2 plus finger1/finger2 pad
# collision geoms, so contact dynamics can be controlled by the gripper actuator
# interface instead of a palm-mounted proxy.


# ─────────────────────────────────────────────
#  Helper: Euler → MuJoCo quaternion
# ─────────────────────────────────────────────
def euler_deg_to_quat(rpy_str: str) -> str:
    """Convert RPY euler angles (degrees, space-separated) to MuJoCo quaternion (w,x,y,z)."""
    if not rpy_str:
        return "1 0 0 0"
    rpy = [float(x) for x in rpy_str.split()]
    while len(rpy) < 3:
        rpy.append(0.0)
    roll, pitch, yaw = np.deg2rad(rpy[0]), np.deg2rad(rpy[1]), np.deg2rad(rpy[2])
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return f"{w:.6f} {x:.6f} {y:.6f} {z:.6f}"


# ─────────────────────────────────────────────
#  Helper: Geom / Inertial extraction from URDF link
# ─────────────────────────────────────────────
def get_origin(origin_elem: ET.Element | None) -> tuple[str, str]:
    """Return (xyz, quat) for a URDF origin element."""
    if origin_elem is None:
        return "0 0 0", "1 0 0 0"
    xyz = origin_elem.get("xyz", "0 0 0")
    rpy = origin_elem.get("rpy")
    quat = euler_deg_to_quat(rpy) if rpy else "1 0 0 0"
    return xyz, quat


def geoms_to_xml(
    link_elem: ET.Element,
    color: np.ndarray,
    group_vis: int = 1,
    group_col: int = 0,
    prefix: str = "",
) -> str:
    """Convert URDF link visual+collision geoms to MuJoCo XML string with fixed RGBA."""
    lines = []
    rgba_str = f"{color[0]:.4f} {color[1]:.4f} {color[2]:.4f} {color[3]:.4f}"

    for visual in link_elem.findall("visual"):
        mesh = visual.find(".//mesh")
        origin = visual.find("origin")
        xyz, quat = get_origin(origin)
        fname = ""
        if mesh is not None:
            # Strip 'assets/' prefix → flat asset key
            fname = mesh.get("filename", "").replace("assets/", "", 1)
            parts = [
                f'  <geom mesh="{fname}" type="mesh" group="{group_vis}"',
                f'pos="{xyz}" quat="{quat}"',
                'contype="0" conaffinity="0"',
                f'rgba="{rgba_str}"',
                "/>",
            ]
        else:
            parts = [
                f'  <geom type="box" size="0.05 0.05 0.05" group="{group_vis}"',
                f'pos="{xyz}" quat="{quat}"',
                'contype="0" conaffinity="0"',
                f'rgba="{rgba_str}"',
                "/>",
            ]
        lines.append(" ".join(parts))

    for collision in link_elem.findall("collision"):
        mesh = collision.find(".//mesh")
        origin = collision.find("origin")
        xyz, quat = get_origin(origin)
        fname = ""
        if mesh is not None:
            fname = mesh.get("filename", "").replace("assets/", "", 1)
            parts = [
                f'  <geom mesh="{fname}" type="mesh" group="{group_col}"',
                f'pos="{xyz}" quat="{quat}"',
                'contype="1" conaffinity="1"',
                f'rgba="{rgba_str}"',
                "/>",
            ]
        else:
            parts = [
                f'  <geom type="box" size="0.05 0.05 0.05" group="{group_col}"',
                f'pos="{xyz}" quat="{quat}"',
                'contype="1" conaffinity="1"',
                f'rgba="{rgba_str}"',
                "/>",
            ]
        lines.append(" ".join(parts))

    return "\n".join(lines)


def inertial_to_xml(link_elem: ET.Element) -> str:
    """Convert URDF link inertial to MuJoCo XML string."""
    inertial = link_elem.find("inertial")
    if inertial is None:
        return ""
    parts = ["  <inertial"]
    origin = inertial.find("origin")
    xyz, quat = get_origin(origin)
    if xyz != "0 0 0":
        parts.append(f'pos="{xyz}"')
    if quat != "1 0 0 0":
        parts.append(f'quat="{quat}"')
    mass = inertial.find("mass")
    if mass is not None:
        parts.append(f'mass="{mass.get("value", "1")}"')
    inertia = inertial.find("inertia")
    if inertia is not None:
        ixx = inertia.get("ixx", "0")
        iyy = inertia.get("iyy", "0")
        izz = inertia.get("izz", "0")
        parts.append(f'diaginertia="{ixx} {iyy} {izz}"')
    parts[-1] += "/>"
    return "\n".join(parts) + "\n"


# ─────────────────────────────────────────────
#  Core Builder
# ─────────────────────────────────────────────
class MergedModelBuilder:
    """Builds a merged Panda + Drawer MuJoCo model."""

    def __init__(
        self,
        seed: int = 1,
        drawer_color: np.ndarray | None = None,
        handle_color: np.ndarray | None = None,
        bg_color: str = "neutral_lab",
        robot_init_qpos: np.ndarray | None = None,
        camera_config: dict | None = None,
        robot_base_pos: np.ndarray | None = None,
    ):
        self.seed = seed
        self.drawer_color = drawer_color or DRAWER_COLORS["drawer_door"]
        self.handle_color = handle_color or DRAWER_COLORS["drawer_handle"]
        self.bg_color = BG_COLORS.get(bg_color, BG_COLORS["neutral_lab"])
        self.robot_init_qpos = (
            robot_init_qpos if robot_init_qpos is not None else LIBERO_INIT_QPOS
        )
        self.robot_base_pos = np.asarray(
            robot_base_pos if robot_base_pos is not None else ROBOT_BASE_POS,
            dtype=float,
        )
        self.camera_config = camera_config or LIBERO_CAMERA

    # ── public API ────────────────────────────────

    def build(self) -> tuple[str, dict[str, bytes], dict[str, Any], str]:
        """
        Returns:
            xml (str):              Merged MuJoCo XML string
            assets (dict):           {filename: bytes} for all mesh files
            metadata (dict):          Extracted drawer metadata
            semantic_hash (str):      SHA256 of semantic_mapping
        """
        # 1. Load assets (drawer + robot)
        assets, metadata, semantic_hash = self._load_assets()
        robot_assets = self.load_robot_assets()
        assets.update(robot_assets)  # robot assets take precedence if name collision

        # 2. Get drawer structure
        drawer_tree, all_links, all_joints = self._parse_drawer()

        # 3. Build drawer body XML
        drawer_body_xml = self._build_drawer_bodies(drawer_tree, all_links, all_joints)

        # 4. Build merged XML
        merged_xml = self._build_merged_xml(
            drawer_body_xml,
            self._get_robot_xml_inner(),
        )

        return merged_xml, assets, metadata, semantic_hash

    # ── private: asset loading ──────────────────

    def _load_assets(self) -> tuple[dict[str, bytes], dict[str, Any], str]:
        """Load drawer URDF + all assets (recursive, fixes texture loading bug)."""
        seed_dir = DRAWER_ROOT / str(self.seed)
        urdf_path = seed_dir / "drawer.urdf"
        metadata_path = seed_dir / "metadata.json"
        semantic_path = seed_dir / "semantic_mapping.json"

        # Assets: recursive glob (FIX for missing textures)
        assets: dict[str, bytes] = {}
        assets_dir = seed_dir / "assets"
        if assets_dir.exists():
            for p in assets_dir.rglob("*"):
                if p.is_file():
                    rel = str(p.relative_to(assets_dir))
                    assets[rel] = p.read_bytes()

        # Metadata
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
            except json.JSONDecodeError:
                pass

        # Semantic mapping
        semantic_hash = ""
        semantic_mapping: dict[str, Any] = {}
        if semantic_path.exists():
            semantic_hash = hashlib.sha256(semantic_path.read_bytes()).hexdigest()
            try:
                semantic_mapping = json.loads(semantic_path.read_text())
            except json.JSONDecodeError:
                pass

        return assets, metadata, semantic_hash

    @staticmethod
    def _fmt_vec(values: list[float] | tuple[float, ...]) -> str:
        return " ".join(f"{float(v):.6g}" for v in values)

    @staticmethod
    def _patch_float_list(
        patch: dict[str, Any], key: str, default: tuple[float, ...]
    ) -> tuple[float, ...]:
        raw = patch.get(key, default)
        if not isinstance(raw, (list, tuple)):
            return default
        try:
            vals = tuple(float(v) for v in raw)
        except Exception:
            return default
        return vals if len(vals) == len(default) else default

    def _apply_finger_pad_contact_patch(self, gripper_tree: ET.Element) -> None:
        """Apply an explicitly requested finite fingertip contact patch model.

        The default robot/gripper XML is unchanged. Generated candidates can opt
        in through ``model_builder_parameters.finger_pad_contact_patch``. Added
        geoms remain small, named finger pad surfaces; they are not arm, wrist,
        hand shell, or hidden broad-catcher target geometry.
        """

        variant = getattr(self, "variant", {}) or {}
        patch = variant.get("finger_pad_contact_patch") or {}
        if not isinstance(patch, dict) or not patch.get("kind"):
            return
        if patch.get("enabled", True) is False:
            return

        kind = str(patch.get("kind", "")).lower()
        if kind in {"baseline", "current_single_pad_baseline"}:
            return

        base_size = self._patch_float_list(patch, "base_size", (0.008, 0.004, 0.008))
        if "enlarged" in kind or "subpatch" in kind or "rounded" in kind:
            base_size = self._patch_float_list(
                patch, "base_size", (0.010, 0.0055, 0.010)
            )
        base_size = tuple(
            min(max(v, lo), hi)
            for v, lo, hi in zip(
                base_size, (0.004, 0.0025, 0.004), (0.014, 0.007, 0.014)
            )
        )
        margin = min(max(float(patch.get("margin", 0.0) or 0.0), 0.0), 0.003)
        friction = self._patch_float_list(patch, "friction", (3.0, 0.06, 0.0002))
        solref = self._patch_float_list(patch, "solref", (0.008, 0.45))
        solimp = self._patch_float_list(patch, "solimp", (0.90, 0.95, 0.001))
        condim = str(int(patch.get("condim", 4) or 4))

        base_attrs = {
            "size": self._fmt_vec(base_size),
            "friction": self._fmt_vec(friction),
            "solref": self._fmt_vec(solref),
            "solimp": self._fmt_vec(solimp),
            "condim": condim,
            "priority": str(int(patch.get("priority", 2) or 2)),
        }
        if margin > 0.0:
            base_attrs["margin"] = f"{margin:.6g}"
        for name in ("finger1_pad_collision", "finger2_pad_collision"):
            geom = gripper_tree.find(f".//geom[@name='{name}']")
            if geom is not None:
                for attr, value in base_attrs.items():
                    geom.set(attr, value)

        if "subpatch" not in kind and int(patch.get("subpatch_count", 0) or 0) <= 0:
            return

        subpatch_size = self._patch_float_list(
            patch, "subpatch_size", (0.0065, 0.0035, 0.0065)
        )
        subpatch_size = tuple(
            min(max(v, lo), hi)
            for v, lo, hi in zip(
                subpatch_size, (0.003, 0.002, 0.003), (0.010, 0.005, 0.010)
            )
        )
        offsets = patch.get("subpatch_offsets") or [
            [0.0, 0.0, -0.006],
            [0.0, 0.0, 0.006],
        ]
        if not isinstance(offsets, list):
            offsets = []
        finger_specs = [
            ("finger_joint1_tip", "finger1", [0.0, -0.005, -0.015]),
            ("finger_joint2_tip", "finger2", [0.0, 0.005, -0.015]),
        ]
        for body_name, prefix, base_pos in finger_specs:
            body = gripper_tree.find(f".//body[@name='{body_name}']")
            if body is None:
                continue
            for idx, raw_offset in enumerate(offsets[:3]):
                if not isinstance(raw_offset, (list, tuple)) or len(raw_offset) != 3:
                    continue
                pos = [base_pos[i] + float(raw_offset[i]) for i in range(3)]
                attrs = {
                    "name": f"{prefix}_pad_collision_patch_{idx}",
                    "type": "box",
                    "group": "0",
                    "contype": "1",
                    "conaffinity": "1",
                    "size": self._fmt_vec(subpatch_size),
                    "pos": self._fmt_vec(pos),
                    "quat": "0 0 0 1",
                    "friction": self._fmt_vec(friction),
                    "solref": self._fmt_vec(solref),
                    "solimp": self._fmt_vec(solimp),
                    "condim": condim,
                    "priority": str(int(patch.get("priority", 2) or 2)),
                    "rgba": str(patch.get("rgba", "0.04 0.04 0.04 0.55")),
                }
                if margin > 0.0:
                    attrs["margin"] = f"{margin:.6g}"
                ET.SubElement(body, "geom", attrs)

    def _get_robot_xml_inner(self) -> tuple[str, str, str]:
        """Extract robot XML and insert the real Panda gripper subtree."""
        robot_xml = Path(ROBOT_XML_PATH).read_text()

        def inner(elem: ET.Element) -> str:
            return "".join(ET.tostring(c, encoding="unicode") for c in elem)

        # Simplify mesh paths in robot XML (meshes/link0.stl -> link0.stl)
        simplified = re.sub(r'file="meshes/([^"]+)"', r'file="\1"', robot_xml)

        gripper_xml = Path(GRIPPER_XML_PATH).read_text()
        # Broad finger mesh collisions reach the drawer before the dedicated
        # pads and create non-target contact. Preserve the visual finger meshes
        # and the small finger*_pad_collision boxes as the only active finger
        # contact patches for GOC-v4.
        for collision_name in ("finger1_collision", "finger2_collision"):
            pattern = (
                rf'(<geom[^>]*name="{collision_name}"[^>]*conaffinity=")1("[^>]*/>)'
            )
            gripper_xml = re.sub(pattern, r"\g<1>0\2", gripper_xml)
        # Simplify gripper mesh paths (meshes/panda_gripper/finger.stl -> finger.stl).
        gripper_xml = re.sub(
            r'file="meshes/panda_gripper/([^"]+)"', r'file="\1"', gripper_xml
        )
        gripper_tree = ET.fromstring(gripper_xml)
        for geom in gripper_tree.findall(".//geom"):
            if geom.get("name") in {"finger1_collision", "finger2_collision"}:
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
        self._apply_finger_pad_contact_patch(gripper_tree)
        gripper_worldbody_inner = inner(gripper_tree.find("worldbody"))
        gripper_actuator_inner = inner(gripper_tree.find("actuator"))
        gripper_asset_inner = inner(gripper_tree.find("asset"))

        gripper_marker = "<!-- to add gripper -->"
        if "finger_joint1" not in simplified:
            if gripper_marker not in simplified:
                raise RuntimeError(
                    "Panda right_hand gripper insertion marker not found"
                )
            simplified = simplified.replace(
                gripper_marker,
                gripper_worldbody_inner
                + "\n                                                "
                + gripper_marker,
                1,
            )

        robot_tree2 = ET.fromstring(simplified)

        return (
            inner(robot_tree2.find("actuator")) + gripper_actuator_inner,
            inner(robot_tree2.find("asset")) + gripper_asset_inner,
            inner(robot_tree2.find("worldbody")),
        )

    def load_robot_assets(self) -> dict[str, bytes]:
        """Load robot and gripper mesh files as assets dict."""
        robot_base = Path(ROBOT_XML_PATH).parent
        gripper_base = Path(GRIPPER_XML_PATH).parent
        assets = {}
        for base, subdirs in [
            (robot_base, ["meshes", "obj_meshes"]),
            (gripper_base, ["meshes/panda_gripper"]),
        ]:
            for subdir in subdirs:
                full_dir = base / subdir
                if full_dir.exists():
                    for fp in full_dir.rglob("*"):
                        if fp.is_file():
                            fname = fp.name  # Use basename only
                            assets[fname] = fp.read_bytes()
        return assets

    # ── private: drawer ───────────────────────

    def _parse_drawer(self) -> tuple[ET.Element, dict, dict]:
        """Parse drawer URDF and return tree + link/joint maps."""
        seed_dir = DRAWER_ROOT / str(self.seed)
        with open(seed_dir / "drawer.urdf") as f:
            drawer_urdf = f.read()
        tree = ET.fromstring(drawer_urdf)
        links = {l.get("name"): l for l in tree.findall(".//link")}
        joints = {j.get("name"): j for j in tree.findall(".//joint")}
        return tree, links, joints

    def _build_drawer_bodies(
        self,
        drawer_tree: ET.Element,
        all_links: dict,
        all_joints: dict,
    ) -> str:
        """Build MuJoCo body XML for drawer (base + movable doors)."""
        parts = []

        # ── Drawer base (link_0) ──
        link0 = all_links["link_0"]
        base_color = DRAWER_COLORS["drawer_base"]
        parts.append(
            f'<body name="drawer_base" pos="0 0 0">\n'
            f"{geoms_to_xml(link0, base_color)}\n"
            f"{inertial_to_xml(link0)}"
        )

        # ── Active drawer movable links ──
        # Keep the task focused on the same active drawer bodies as the previous
        # GOC-v4 repair (link_1/link_2), but make the model robust to single-door
        # assets where link_2 / drawer_slider_1 do not exist.
        self._drawer_slide_joints = []
        self._drawer_movable_bodies = []
        for link_name, joint_name in [
            ("link_1", "drawer_slider_0"),
            ("link_2", "drawer_slider_1"),
        ]:
            if link_name not in all_links or joint_name not in all_joints:
                continue
            link = all_links[link_name]
            joint = all_joints[joint_name]
            self._drawer_slide_joints.append(joint_name)
            self._drawer_movable_bodies.append(link_name)

            # Joint origin in parent frame
            j_origin = joint.find("origin")
            j_xyz = j_origin.get("xyz", "0 0 0") if j_origin is not None else "0 0 0"
            axis_elem = joint.find("axis")
            axis = axis_elem.get("xyz", "1 0 0") if axis_elem is not None else "1 0 0"
            limit = joint.find("limit")
            range_str = (
                f"{limit.get('lower', '0')} {limit.get('upper', '1')}"
                if limit is not None
                else "0 1"
            )
            dynamics = joint.find("dynamics")
            damping = dynamics.get("damping", "0") if dynamics is not None else "0"

            body_xml = (
                f'<body name="{link_name}" pos="{j_xyz}">\n'
                f'  <joint name="{joint_name}" type="slide" axis="{axis}" '
                f'range="{range_str}" damping="{damping}"/>\n'
                f"{geoms_to_xml(link, self.drawer_color)}\n"
                f"{inertial_to_xml(link)}"
            )

            # Handle highlight: apply darker color to handle geoms
            # Detect handle geoms by name
            handle_xml = ""
            for visual in link.findall("visual"):
                mesh = visual.find(".//mesh")
                if mesh is None:
                    continue
                origin = visual.find("origin")
                xyz, quat = get_origin(origin)
                fname = mesh.get("filename", "").replace("assets/", "", 1)
                if "handle" in fname.lower():
                    handle_xml += (
                        f'  <geom mesh="{fname}" type="mesh" group="1" '
                        f'pos="{xyz}" quat="{quat}" '
                        f'rgba="{self.handle_color[0]:.4f} {self.handle_color[1]:.4f} '
                        f'{self.handle_color[2]:.4f} {self.handle_color[3]:.4f}"/>\n'
                    )

            parts.append(body_xml)
            if handle_xml:
                parts.append(handle_xml)
            parts.append("</body>")

        parts.append("</body>")
        return "\n".join(parts)

    # ── private: full merged XML ───────────────

    def _build_merged_xml(
        self, drawer_body_xml: str, robot_inner: tuple[str, str, str]
    ) -> str:
        """Build the complete merged MuJoCo XML document."""
        actuator_inner, asset_inner, worldbody_inner = robot_inner

        # Add drawer meshes to asset section
        drawer_mesh_xml = self._drawer_mesh_asset_xml()
        # Add drawer actuators only for slider joints that exist in this instance.
        drawer_actuator_xml = "\n".join(
            f'  <motor ctrllimited="true" ctrlrange="-10 10" joint="{joint}" name="drawer{i}"/>'
            for i, joint in enumerate(
                getattr(self, "_drawer_slide_joints", ["drawer_slider_0"])
            )
        )

        # Lighting: LIBERO-style three-point lighting
        lighting_xml = self._build_lighting()

        # Keep the drawer at the scene origin and mount the robot beside it.
        # Wrapping robot bodies in a transform body preserves robot geom order/IDs.
        robot_base_pos = " ".join(f"{float(v):.6f}" for v in self.robot_base_pos)
        robot_worldbody = (
            f'<body name="robot_mount" pos="{robot_base_pos}">\n'
            f"{worldbody_inner}\n"
            "</body>"
        )
        new_worldbody = drawer_body_xml + "\n" + robot_worldbody

        # Suppress only internal closed-drawer self-collision. Robot-drawer and
        # robot-handle contacts remain enabled for GOC-v3 contact validation.
        contact_exclude_xml = "\n".join(
            f'  <exclude body1="drawer_base" body2="{body}"/>'
            for body in getattr(self, "_drawer_movable_bodies", ["link_1"])
        )

        xml = f"""<mujoco model="panda_drawer">
  <compiler angle="radian" inertiafromgeom="auto"/>
  <option timestep="0.002"/>

  <asset>
{asset_inner}
{drawer_mesh_xml}
  </asset>

  <actuator>
{actuator_inner}
  {drawer_actuator_xml}
  </actuator>

  <contact>
{contact_exclude_xml}
  </contact>

  <worldbody>
{lighting_xml}
{new_worldbody}
  </worldbody>

  <visual>
    <rgba haze="{self.bg_color[0]:.3f} {self.bg_color[1]:.3f} {self.bg_color[2]:.3f} 1"/>
    <headlight ambient="0.50 0.50 0.50" diffuse="0.95 0.93 0.90" specular="0.50 0.50 0.50"/>
  </visual>
</mujoco>"""

        return xml

    def _drawer_mesh_asset_xml(self) -> str:
        """Add drawer mesh declarations to <asset> section."""
        seed_dir = DRAWER_ROOT / str(self.seed)
        assets_dir = seed_dir / "assets"
        lines = []
        if assets_dir.exists():
            for p in sorted(assets_dir.rglob("*")):
                if p.is_file() and (p.suffix in {".obj", ".stl"}):
                    rel = str(p.relative_to(assets_dir))
                    fname = rel.replace("assets/", "", 1)
                    lines.append(f'  <mesh name="{fname}" file="{fname}"/>')
        return "\n".join(lines)

    def _build_lighting(self) -> str:
        """LIBERO-style three-point lighting setup."""
        return f"""    <light name="top" pos="0 0 3" dir="0 0 -1" diffuse="{self.bg_color[0]:.3f} {self.bg_color[1]:.3f} {self.bg_color[2]:.3f}"/>
    <light name="key" pos="1 1 2" dir="-0.5 -0.5 -1" diffuse="0.9 0.88 0.84"/>
    <light name="fill" pos="-1 0.5 1.5" dir="0.5 -0.25 -0.75" diffuse="0.5 0.5 0.5"/>
    <light name="rim" pos="-1 -1 2" dir="0.5 0.5 -1" diffuse="0.3 0.3 0.35"/>"""
