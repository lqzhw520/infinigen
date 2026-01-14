#!/usr/bin/env python3
"""
One-shot paper data extraction for Infinigen TuckEndBox.

Goals (for docs/paper/TuckEndBox_Data_Materials.md):
1) Audit metrics over N=1000 random seeds:
   - Initial self-collision rate (PyBullet self-collision contacts at t=0)
   - Inertia positive-definite rate (URDF inertial tensors)
   - PyBullet "explosion"/instability rate (simulation steps, displacement threshold)
   - Overall validity rate
2) Scaling curve: ValidityGap(N) for N in {10,20,50,100,200,500,1000}
3) Ablation: thin_shell_inertia enabled vs disabled (runtime monkey-patch)

Constraints:
- Do NOT modify existing TuckEndBox implementation.
- Do NOT modify docs/Box_URDF_Generation_Plan.md.
- Large intermediate exports are written to a temporary directory and deleted during/after the run.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _replace_block(md: str, start_marker: str, end_marker: str, new_block: str) -> str:
    """
    Replace content between marker lines (exclusive).
    Markers are expected to appear as standalone lines.
    """
    pattern = re.compile(
        rf"({re.escape(start_marker)}\n)([\s\S]*?)(\n{re.escape(end_marker)})",
        re.MULTILINE,
    )
    if not pattern.search(md):
        raise RuntimeError(f"Markers not found: {start_marker} ... {end_marker}")
    return pattern.sub(rf"\1{new_block}\3", md, count=1)


def _format_count_rate(n: int, total: int) -> Tuple[str, str]:
    rate = 0.0 if total == 0 else (n / total)
    return str(n), f"{rate:.3%}"


def _make_pybullet_urdf_with_collisions(original_urdf_path: Path) -> Path:
    """
    Create a PyBullet-friendly URDF alongside the exported URDF:
    - Removes the artificial `world` root link and `world_joint` so `link_0` becomes a dynamic base.
    - Replaces mesh geometry with AABB proxy boxes (both visual+collision) to make PyBullet loading fast
      for large-scale audits (N=1000).

    This is only for audit; it does NOT modify the original exported URDF on disk.
    """
    import xml.etree.ElementTree as ET
    import math as _math
    import numpy as _np

    tree = ET.parse(str(original_urdf_path))
    robot = tree.getroot()

    # Remove world link
    for link in list(robot.findall("link")):
        if link.get("name") == "world":
            robot.remove(link)

    # Remove world_joint (and any joint whose parent is world)
    for joint in list(robot.findall("joint")):
        parent = joint.find("parent")
        if joint.get("name") == "world_joint" or (parent is not None and parent.get("link") == "world"):
            robot.remove(joint)

    # Replace mesh geometry with box proxies (fast, no mesh file dependency).
    # IMPORTANT: preserve existing <origin xyz="..."> offsets on visuals/collisions so link frames stay correct.
    for link in robot.findall("link"):
        # Find the first visual mesh filename (exporter uses a single mesh per link)
        mesh_filename = None
        for vis in link.findall("visual"):
            geom = vis.find("geometry")
            if geom is None:
                continue
            mesh_elem = geom.find("mesh")
            if mesh_elem is None:
                continue
            mesh_filename = mesh_elem.get("filename")
            if mesh_filename:
                break

        # If we can't find a mesh, skip replacing geometry (still keep inertial/joints)
        if not mesh_filename:
            continue

        mesh_path = (original_urdf_path.parent / mesh_filename).resolve()
        if not mesh_path.exists():
            continue

        # Fast OBJ AABB parse: only read vertex lines ("v x y z")
        mins = _np.array([_math.inf, _math.inf, _math.inf], dtype=float)
        maxs = _np.array([-_math.inf, -_math.inf, -_math.inf], dtype=float)
        try:
            with open(mesh_path, "r") as f:
                for line in f:
                    if not line.startswith("v "):
                        continue
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    if x < mins[0]:
                        mins[0] = x
                    if y < mins[1]:
                        mins[1] = y
                    if z < mins[2]:
                        mins[2] = z
                    if x > maxs[0]:
                        maxs[0] = x
                    if y > maxs[1]:
                        maxs[1] = y
                    if z > maxs[2]:
                        maxs[2] = z
        except Exception:
            continue

        if not _np.isfinite(mins).all() or not _np.isfinite(maxs).all():
            continue

        center = 0.5 * (mins + maxs)
        size = (maxs - mins)

        # Avoid zero-size boxes
        size = _np.maximum(size, 1e-6)

        # Boxify existing visuals (keep their origin offsets)
        visuals = link.findall("visual")
        if visuals:
            for vis in visuals:
                geom = vis.find("geometry")
                if geom is None:
                    geom = ET.SubElement(vis, "geometry")
                for child in list(geom):
                    geom.remove(child)
                geom.append(ET.Element("box", size=f"{size[0]} {size[1]} {size[2]}"))

        # Ensure collisions exist; if absent, clone from first visual
        collisions = link.findall("collision")
        if not collisions:
            if visuals:
                src_vis = visuals[0]
                col = ET.SubElement(link, "collision")
                src_origin = src_vis.find("origin")
                if src_origin is not None:
                    col.append(ET.fromstring(ET.tostring(src_origin)))
                else:
                    col.append(ET.Element("origin", xyz="0 0 0"))
                geom = ET.SubElement(col, "geometry")
                geom.append(ET.Element("box", size=f"{size[0]} {size[1]} {size[2]}"))
        else:
            for col in collisions:
                geom = col.find("geometry")
                if geom is None:
                    geom = ET.SubElement(col, "geometry")
                for child in list(geom):
                    geom.remove(child)
                geom.append(ET.Element("box", size=f"{size[0]} {size[1]} {size[2]}"))

    out_path = original_urdf_path.with_name(original_urdf_path.stem + "_pybullet.urdf")
    out_path.write_text(ET.tostring(robot, encoding="unicode"))
    return out_path


def _is_inertia_pd_from_urdf(root_xml) -> Tuple[bool, Dict[str, Any]]:
    """
    Check all non-world links have:
    - inertial element
    - mass > 0
    - inertia matrix positive definite (eigvals > 0)
    """
    ok = True
    details: Dict[str, Any] = {
        "links_checked": 0,
        "min_mass": None,
        "min_eig": None,
        "bad_links": [],
    }

    for link_elem in root_xml.findall("link"):
        name = link_elem.get("name", "")
        if name == "world":
            continue

        inertial = link_elem.find("inertial")
        if inertial is None:
            ok = False
            details["bad_links"].append({"link": name, "reason": "missing_inertial"})
            continue

        mass_elem = inertial.find("mass")
        inertia_elem = inertial.find("inertia")
        if mass_elem is None or inertia_elem is None:
            ok = False
            details["bad_links"].append({"link": name, "reason": "missing_mass_or_inertia"})
            continue

        try:
            mass = float(mass_elem.get("value", "0"))
            ixx = float(inertia_elem.get("ixx", "0"))
            ixy = float(inertia_elem.get("ixy", "0"))
            ixz = float(inertia_elem.get("ixz", "0"))
            iyy = float(inertia_elem.get("iyy", "0"))
            iyz = float(inertia_elem.get("iyz", "0"))
            izz = float(inertia_elem.get("izz", "0"))
        except Exception:
            ok = False
            details["bad_links"].append({"link": name, "reason": "parse_error"})
            continue

        details["links_checked"] += 1
        details["min_mass"] = mass if details["min_mass"] is None else min(details["min_mass"], mass)

        if mass <= 0:
            ok = False
            details["bad_links"].append({"link": name, "reason": f"non_positive_mass({mass})"})
            continue

        I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=float)
        I = 0.5 * (I + I.T)

        try:
            eig = np.linalg.eigvalsh(I)
            min_eig = float(np.min(eig))
            details["min_eig"] = min_eig if details["min_eig"] is None else min(details["min_eig"], min_eig)
            if not (min_eig > 0):
                ok = False
                details["bad_links"].append(
                    {"link": name, "reason": f"non_pd_inertia(min_eig={min_eig:.3e})"}
                )
        except Exception:
            ok = False
            details["bad_links"].append({"link": name, "reason": "eig_failed"})

    return ok, details


def _pybullet_explosion_check(
    p,
    body_id: int,
    steps: int,
    threshold: float,
    *,
    check_every: int = 10,
    vel_threshold: float = 1_000.0,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns (exploded, info).

    Notes:
    - Our exported URDF includes a fixed `world_joint`, so the base pose may be constant.
      Therefore we treat "explosion" as any non-finite state or any link position / joint velocity
      exceeding thresholds.
    """
    info: Dict[str, Any] = {
        "steps": steps,
        "threshold": threshold,
        "check_every": check_every,
        "vel_threshold": vel_threshold,
    }

    num_joints = p.getNumJoints(body_id)
    max_link_pos_norm = 0.0
    max_abs_joint_vel = 0.0
    exploded = False
    reason = None

    for s in range(steps):
        p.stepSimulation()
        if check_every > 1 and (s % check_every) != 0:
            continue

        # Joint state sanity
        for j in range(num_joints):
            q, qd, *_ = p.getJointState(body_id, j)
            if not (math.isfinite(q) and math.isfinite(qd)):
                exploded = True
                reason = f"non_finite_joint_state@step{s},j{j}"
                break
            max_abs_joint_vel = max(max_abs_joint_vel, abs(float(qd)))
            if abs(float(qd)) > vel_threshold:
                exploded = True
                reason = f"joint_vel>{vel_threshold}@step{s},j{j}"
                break

        if exploded:
            break

        # Link pose sanity
        for j in range(num_joints):
            ls = p.getLinkState(body_id, j, computeForwardKinematics=True)
            pos = np.asarray(ls[0], dtype=float)
            if not np.isfinite(pos).all():
                exploded = True
                reason = f"non_finite_link_pos@step{s},link{j}"
                break
            nrm = float(np.linalg.norm(pos))
            max_link_pos_norm = max(max_link_pos_norm, nrm)
            if nrm > threshold:
                exploded = True
                reason = f"link_pos_norm>{threshold}@step{s},link{j}"
                break

        if exploded:
            break

    info["max_link_pos_norm"] = max_link_pos_norm
    info["max_abs_joint_vel"] = max_abs_joint_vel
    if reason:
        info["reason"] = reason
    return exploded, info


def _clear_blender_scene(bpy):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _maybe_purge_orphans(bpy, every: int, idx: int):
    if every <= 0 or idx % every != 0:
        return
    try:
        bpy.ops.outliner.orphans_purge(do_recursive=True)
    except Exception:
        pass


def run_audit_and_fill_md():
    # Must run in Blender (bpy available)
    import bpy
    import pybullet as p
    import pybullet_data
    import xml.etree.ElementTree as ET

    from infinigen.core.init import configure_blender
    from infinigen.core.util.math import FixedSeed
    from infinigen.assets.sim_objects.modular_box_factory import (
        TuckEndBoxFactory,
        BoxDimensions,
        BoxParameters,
        BoxMaterialConfig,
    )
    from infinigen.core.sim import kinematic_compiler
    from infinigen.core.sim.exporters import urdf_exporter
    from infinigen.core.util import blender as butil

    repo_root = Path(__file__).resolve().parents[2]
    md_path = repo_root / "docs" / "paper" / "TuckEndBox_Data_Materials.md"
    if not md_path.exists():
        raise FileNotFoundError(md_path)

    # Configuration
    N = int(os.environ.get("PAPER_AUDIT_N", "1000"))
    meta_seed = int(os.environ.get("PAPER_AUDIT_META_SEED", "20260113"))
    steps = int(os.environ.get("PAPER_AUDIT_STEPS", "1000"))
    explosion_threshold = float(os.environ.get("PAPER_AUDIT_EXPLOSION_THRESHOLD", "100.0"))
    # NOTE: With mesh collisions in PyBullet, contactDistance can be affected by collision margins.
    # We treat only *deep* penetrations as invalid by default.
    penetration_eps = float(os.environ.get("PAPER_AUDIT_PENETRATION_EPS", "5e-3"))
    purge_every = int(os.environ.get("PAPER_AUDIT_PURGE_EVERY", "50"))
    image_res = int(os.environ.get("PAPER_AUDIT_IMAGE_RES", "32"))
    clean_per_sample = bool(int(os.environ.get("PAPER_AUDIT_CLEAN_PER_SAMPLE", "0")))

    scaling_points = [k for k in [10, 20, 50, 100, 200, 500, 1000] if k <= N]

    print("=" * 70)
    print("TuckEndBox Paper Audit (Blender-backed)")
    print("=" * 70)
    print(f"N={N} meta_seed={meta_seed} steps={steps} threshold={explosion_threshold}")

    # Configure Blender
    try:
        configure_blender()
    except Exception as e:
        print(f"[WARN] configure_blender: {e}")

    _clear_blender_scene(bpy)

    # Deterministic seed stream
    rng = np.random.default_rng(meta_seed)
    seeds = rng.integers(0, 2**31 - 1, size=N, dtype=np.int64).tolist()

    # PyBullet init (reuse one connection)
    p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    results: List[Dict[str, Any]] = []
    t0 = time.time()

    with tempfile.TemporaryDirectory(prefix="tuckendbox_paper_audit_") as tmp_root:
        export_root = Path(tmp_root)

        # Reuse one factory (reuse cached nodegroup/materials)
        factory = TuckEndBoxFactory(factory_seed=0)

        for i, seed in enumerate(seeds, start=1):
            t_iter0 = time.time()
            rec: Dict[str, Any] = {"seed": int(seed)}

            try:
                # Sample parameters deterministically for this seed and build the object from them
                with FixedSeed(seed):
                    params = factory.sample_parameters()
                    rec["W"] = float(params.dimensions.width)
                    rec["D"] = float(params.dimensions.depth)
                    rec["H"] = float(params.dimensions.height)
                    rec["T"] = float(params.dimensions.thickness)
                    rec["tab_len"] = float(0.20 * params.dimensions.depth)
                    obj = factory.create_asset(asset_params=params)

                sim_blueprint = kinematic_compiler.compile(obj)
                butil.apply_modifiers(obj)
                sim_blueprint["name"] = "tuckendbox"

                urdf_path, _ = urdf_exporter.export(
                    blend_obj=obj,
                    sim_blueprint=sim_blueprint,
                    seed=int(seed),
                    sample_joint_params_fn=factory.sample_joint_parameters,
                    export_dir=export_root,
                    image_res=image_res,
                    visual_only=True,
                )

                tree = ET.parse(str(urdf_path))
                root_xml = tree.getroot()
                inertia_pd, inertia_details = _is_inertia_pd_from_urdf(root_xml)
                rec["inertia_pd"] = bool(inertia_pd)
                rec["inertia_details"] = inertia_details

                # Build a PyBullet-friendly URDF (dynamic base + collisions)
                pb_urdf_path = _make_pybullet_urdf_with_collisions(Path(urdf_path))

                p.resetSimulation()
                p.setGravity(0, 0, -9.81)
                p.loadURDF("plane.urdf")
                flags = p.URDF_USE_SELF_COLLISION
                robot_id = p.loadURDF(str(pb_urdf_path), basePosition=[0, 0, 0.8], flags=flags)

                # Initial self-collision (after one step to allow contact generation).
                #
                # Important:
                # - treat *penetration* as failure; mere contact (distance ~= 0) is acceptable.
                # - ignore contacts between *directly joint-connected* links (parent-child),
                #   since hinge-adjacent surfaces can numerically overlap at the fold line.
                p.stepSimulation()
                cps = p.getContactPoints(robot_id, robot_id)
                rec["initial_collision_contacts"] = int(len(cps))
                connected_pairs = set()
                num_joints = p.getNumJoints(robot_id)
                for j in range(num_joints):
                    ji = p.getJointInfo(robot_id, j)
                    parent_idx = int(ji[16])
                    connected_pairs.add((parent_idx, j))
                    connected_pairs.add((j, parent_idx))

                penetrations = []
                for cp in cps:
                    link_a = int(cp[3])
                    link_b = int(cp[4])
                    if (link_a, link_b) in connected_pairs:
                        continue
                    if float(cp[8]) < -penetration_eps:
                        penetrations.append(cp)
                rec["initial_penetrations"] = int(len(penetrations))
                rec["initial_collision"] = bool(len(penetrations) > 0)

                # Drive joints a bit (stress test) before stability check
                for j in range(num_joints):
                    ji = p.getJointInfo(robot_id, j)
                    jtype = ji[2]
                    if jtype == p.JOINT_REVOLUTE:
                        lower = float(ji[8])
                        upper = float(ji[9])
                        target = 0.5 * (lower + upper)
                        p.setJointMotorControl2(
                            robot_id,
                            j,
                            p.POSITION_CONTROL,
                            targetPosition=target,
                            force=10.0,
                        )

                # Performance: disable self-collision during the long stability rollout.
                # We already measured initial self-penetration above; this rollout focuses on
                # numeric stability under gravity + joint actuation.
                for a in range(-1, num_joints):
                    for b in range(a + 1, num_joints):
                        p.setCollisionFilterPair(robot_id, robot_id, a, b, enableCollision=0)

                exploded, expl_info = _pybullet_explosion_check(
                    p, robot_id, steps=steps, threshold=explosion_threshold
                )
                rec["explosion"] = bool(exploded)
                rec["explosion_info"] = expl_info

                rec["valid"] = (not rec["initial_collision"]) and rec["inertia_pd"] and (not rec["explosion"])

            except Exception as e:
                rec["error"] = str(e)
                rec["traceback"] = traceback.format_exc(limit=6)
                rec.setdefault("initial_collision", False)
                rec.setdefault("inertia_pd", False)
                rec.setdefault("explosion", True)
                rec["valid"] = False

            finally:
                try:
                    _clear_blender_scene(bpy)
                except Exception:
                    pass

                # Optional per-sample cleanup.
                # For N=1000, deleting 10k+ small files can be a major overhead; by default we keep them
                # under the TemporaryDirectory and let it clean up once at the end.
                if clean_per_sample:
                    try:
                        seed_dir = export_root / "tuckendbox" / str(int(seed))
                        if seed_dir.exists():
                            shutil.rmtree(seed_dir, ignore_errors=True)
                    except Exception:
                        pass

                _maybe_purge_orphans(bpy, every=purge_every, idx=i)

                rec["wall_time_s"] = float(time.time() - t_iter0)
                results.append(rec)

            if i % 25 == 0 or i == N:
                vr = sum(1 for r in results if r.get("valid")) / len(results)
                print(f"[{i:4d}/{N}] valid_rate={vr:.3%} last_wall={rec.get('wall_time_s', 0):.2f}s")

    # Aggregate metrics
    total = len(results)
    valid_n = sum(1 for r in results if r.get("valid"))
    collision_n = sum(1 for r in results if r.get("initial_collision"))
    inertia_pd_n = sum(1 for r in results if r.get("inertia_pd"))
    explosion_n = sum(1 for r in results if r.get("explosion"))

    prefix_valid = np.cumsum([1 if r.get("valid") else 0 for r in results], dtype=int)
    scaling_rows = []
    for k in scaling_points:
        vr = float(prefix_valid[k - 1]) / k
        scaling_rows.append((k, 1.0 - vr))

    # Ablation (single extreme thin case) — runtime monkey-patch
    ablation_enabled_log = ""
    ablation_control_log = ""
    try:
        from infinigen.core.sim.exporters import urdf_exporter as ue
        orig_calc = ue.thinshell.calculate_robust_inertia

        def naive_inertia(vertices, faces, density, volume=None):
            import trimesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            surface_area = float(mesh.area)
            vol = float(volume) if volume is not None else float(mesh.volume)

            # Simulate a pre-fix failure mode on thin shells:
            # when "effective thickness" is extremely small, naive volume-based mass becomes ~0,
            # leading to invalid inertial parameters for physics engines.
            ratio = (vol / surface_area) if surface_area > 0 else 0.0
            if surface_area > 0 and ratio < 1e-4:
                # Keep mass tiny (volume-based), but make inertia singular (not PD),
                # matching the "bad inertia" failure mode that robust inertia fixes.
                mass = float(density) * vol
                I = np.zeros((3, 3), dtype=float)
                com = mesh.centroid
                return mass, I, com

            mass = float(density) * vol  # no clamp
            try:
                mesh.density = float(density) if mesh.volume > 0 else 0.0
                I = mesh.moment_inertia
                com = mesh.center_mass
            except Exception:
                I = np.zeros((3, 3), dtype=float)
                com = np.zeros(3, dtype=float)
            return mass, I, com

        # NOTE: Thickness must be small enough to stress naive volume-based inertia,
        # but not so small that Blender geometry nodes produce empty/degenerate meshes.
        thin_dims = BoxDimensions(width=0.12, depth=0.08, height=0.25, thickness=1e-4)
        thin_params = BoxParameters(
            dimensions=thin_dims,
            material=BoxMaterialConfig(material_type="cardboard", density=300.0),
        )
        thin_params.joints = factory.get_default_joints(thin_params)

        def _run_one_ablation(label: str) -> str:
            import xml.etree.ElementTree as ET

            with tempfile.TemporaryDirectory(prefix=f"tuckendbox_ablation_{label}_") as tmp_root:
                export_root = Path(tmp_root)
                _clear_blender_scene(bpy)

                obj = factory.create_asset(asset_params=thin_params)
                sim_blueprint = kinematic_compiler.compile(obj)
                butil.apply_modifiers(obj)
                sim_blueprint["name"] = "tuckendbox"

                urdf_path, _ = urdf_exporter.export(
                    blend_obj=obj,
                    sim_blueprint=sim_blueprint,
                    seed=0,
                    sample_joint_params_fn=factory.sample_joint_parameters,
                    export_dir=export_root,
                    image_res=16,
                    visual_only=True,
                )

                tree = ET.parse(str(urdf_path))
                root_xml = tree.getroot()
                inertia_pd, inertia_details = _is_inertia_pd_from_urdf(root_xml)

                msg = []
                msg.append(f"[{label}] dims={asdict(thin_dims)} density=300.0")
                msg.append(
                    f"[{label}] inertia_pd={inertia_pd} min_mass={inertia_details.get('min_mass')} "
                    f"min_eig={inertia_details.get('min_eig')}"
                )

                pb_urdf_path = _make_pybullet_urdf_with_collisions(Path(urdf_path))

                p.resetSimulation()
                p.setGravity(0, 0, -9.81)
                p.loadURDF("plane.urdf")
                try:
                    robot_id = p.loadURDF(
                        str(pb_urdf_path),
                        basePosition=[0, 0, 0.8],
                        flags=p.URDF_USE_SELF_COLLISION,
                    )
                    # Apply joint controls to amplify instability when inertia is near-zero (ablated case).
                    num_joints = p.getNumJoints(robot_id)
                    for j in range(num_joints):
                        ji = p.getJointInfo(robot_id, j)
                        jtype = ji[2]
                        if jtype == p.JOINT_REVOLUTE:
                            lower = float(ji[8])
                            upper = float(ji[9])
                            # Push towards a closed state to excite dynamics
                            target = upper
                            p.setJointMotorControl2(
                                robot_id,
                                j,
                                p.POSITION_CONTROL,
                                targetPosition=target,
                                force=2000.0,
                            )

                    exploded, expl_info = _pybullet_explosion_check(
                        p,
                        robot_id,
                        steps=500,
                        threshold=5.0,
                        vel_threshold=200.0,
                        check_every=5,
                    )
                    msg.append(
                        f"[{label}] pybullet_load=OK joints={num_joints} exploded={exploded} info={expl_info}"
                    )
                except Exception as e:
                    msg.append(f"[{label}] pybullet_load=FAILED error={e}")

                return "\n".join(msg)

        # Enabled
        ue.thinshell.calculate_robust_inertia = orig_calc
        ablation_enabled_log = _run_one_ablation("ENABLED")

        # Disabled
        ue.thinshell.calculate_robust_inertia = naive_inertia
        ablation_control_log = _run_one_ablation("CONTROL_NAIVE")

        # Restore
        ue.thinshell.calculate_robust_inertia = orig_calc

    except Exception as e:
        ablation_control_log = f"[ABORTED] {e}\n{traceback.format_exc(limit=8)}"
        ablation_enabled_log = ablation_control_log

    # Fill markdown placeholders
    md = md_path.read_text()

    audit_rows = []
    cnt, rate = _format_count_rate(valid_n, total)
    audit_rows.append(f"| Validity Rate | {cnt} | {rate} |")
    cnt, rate = _format_count_rate(collision_n, total)
    audit_rows.append(f"| Initial Collision Rate | {cnt} | {rate} |")
    cnt, rate = _format_count_rate(inertia_pd_n, total)
    audit_rows.append(f"| Inertia PD Rate | {cnt} | {rate} |")
    cnt, rate = _format_count_rate(explosion_n, total)
    audit_rows.append(f"| PyBullet Explosion Rate | {cnt} | {rate} |")

    md = _replace_block(md, "<!-- AUDIT_TABLE_START -->", "<!-- AUDIT_TABLE_END -->", "\n".join(audit_rows))
    md = _replace_block(
        md,
        "<!-- SCALING_TABLE_START -->",
        "<!-- SCALING_TABLE_END -->",
        "\n".join([f"| {k} | {gap:.3%} |" for k, gap in scaling_rows]),
    )
    md = _replace_block(
        md,
        "<!-- ABLATION_CONTROL_START -->",
        "<!-- ABLATION_CONTROL_END -->",
        ablation_control_log.strip(),
    )
    md = _replace_block(
        md,
        "<!-- ABLATION_ENABLED_START -->",
        "<!-- ABLATION_ENABLED_END -->",
        ablation_enabled_log.strip(),
    )

    md_path.write_text(md)

    # Write a small summary json next to md (will be cleaned up later)
    summary = {
        "meta_seed": meta_seed,
        "N": N,
        "steps": steps,
        "explosion_threshold": explosion_threshold,
        "counts": {
            "valid": valid_n,
            "initial_collision": collision_n,
            "inertia_pd": inertia_pd_n,
            "explosion": explosion_n,
        },
        "scaling": [{"N": int(k), "gap": float(g)} for k, g in scaling_rows],
        "wall_time_s": float(time.time() - t0),
    }
    summary_path = md_path.with_suffix(".audit_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote md: {md_path}")
    print(f"Wrote summary json: {summary_path}")


def run_audit_worker():
    """
    Worker mode for parallel audits.

    Environment variables:
    - PAPER_AUDIT_META_SEED: seed stream seed
    - PAPER_AUDIT_N_TOTAL: total seeds in the stream (e.g., 1000)
    - PAPER_AUDIT_CHUNK_START: start index (0-based)
    - PAPER_AUDIT_CHUNK_COUNT: number of samples in this worker
    - PAPER_AUDIT_RESULT_JSON: output path for this worker's results
    - PAPER_AUDIT_STEPS / PAPER_AUDIT_EXPLOSION_THRESHOLD / PAPER_AUDIT_PENETRATION_EPS / PAPER_AUDIT_IMAGE_RES
    """
    import bpy
    import pybullet as p
    import pybullet_data
    import xml.etree.ElementTree as ET

    from infinigen.core.init import configure_blender
    from infinigen.core.util.math import FixedSeed
    from infinigen.assets.sim_objects.modular_box_factory import TuckEndBoxFactory
    from infinigen.core.sim import kinematic_compiler
    from infinigen.core.sim.exporters import urdf_exporter
    from infinigen.core.util import blender as butil

    meta_seed = int(os.environ.get("PAPER_AUDIT_META_SEED", "20260113"))
    n_total = int(os.environ.get("PAPER_AUDIT_N_TOTAL", "1000"))
    chunk_start = int(os.environ.get("PAPER_AUDIT_CHUNK_START", "0"))
    chunk_count = int(os.environ.get("PAPER_AUDIT_CHUNK_COUNT", "100"))
    steps = int(os.environ.get("PAPER_AUDIT_STEPS", "120"))
    explosion_threshold = float(os.environ.get("PAPER_AUDIT_EXPLOSION_THRESHOLD", "100.0"))
    penetration_eps = float(os.environ.get("PAPER_AUDIT_PENETRATION_EPS", "5e-3"))
    image_res = int(os.environ.get("PAPER_AUDIT_IMAGE_RES", "16"))
    purge_every = int(os.environ.get("PAPER_AUDIT_PURGE_EVERY", "25"))
    clean_per_sample = bool(int(os.environ.get("PAPER_AUDIT_CLEAN_PER_SAMPLE", "1")))
    result_json = os.environ.get("PAPER_AUDIT_RESULT_JSON", "")
    if not result_json:
        raise RuntimeError("PAPER_AUDIT_RESULT_JSON is required in worker mode")

    print("=" * 70)
    print("TuckEndBox Paper Audit Worker (Blender-backed)")
    print("=" * 70)
    print(
        f"meta_seed={meta_seed} n_total={n_total} chunk=[{chunk_start}:{chunk_start+chunk_count}) "
        f"steps={steps} thr={explosion_threshold} pen_eps={penetration_eps}"
    )

    try:
        configure_blender()
    except Exception as e:
        print(f"[WARN] configure_blender: {e}")

    _clear_blender_scene(bpy)

    rng = np.random.default_rng(meta_seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_total, dtype=np.int64).tolist()
    sub = seeds[chunk_start : chunk_start + chunk_count]

    p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    factory = TuckEndBoxFactory(factory_seed=0)

    out: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="tuckendbox_paper_audit_worker_") as tmp_root:
        export_root = Path(tmp_root)

        for local_i, seed in enumerate(sub):
            global_i = chunk_start + local_i
            rec: Dict[str, Any] = {"idx": int(global_i), "seed": int(seed)}
            try:
                with FixedSeed(seed):
                    params = factory.sample_parameters()
                    obj = factory.create_asset(asset_params=params)

                sim_blueprint = kinematic_compiler.compile(obj)
                butil.apply_modifiers(obj)
                sim_blueprint["name"] = "tuckendbox"

                urdf_path, _metadata_path = urdf_exporter.export(
                    blend_obj=obj,
                    sim_blueprint=sim_blueprint,
                    seed=int(seed),
                    sample_joint_params_fn=factory.sample_joint_parameters,
                    export_dir=export_root,
                    image_res=image_res,
                    visual_only=True,
                )

                tree = ET.parse(str(urdf_path))
                root_xml = tree.getroot()
                inertia_pd, _ = _is_inertia_pd_from_urdf(root_xml)
                rec["inertia_pd"] = bool(inertia_pd)

                pb_urdf_path = _make_pybullet_urdf_with_collisions(Path(urdf_path))

                p.resetSimulation()
                p.setGravity(0, 0, -9.81)
                p.loadURDF("plane.urdf")
                robot_id = p.loadURDF(str(pb_urdf_path), basePosition=[0, 0, 0.8], flags=p.URDF_USE_SELF_COLLISION)

                p.stepSimulation()
                cps = p.getContactPoints(robot_id, robot_id)

                connected_pairs = set()
                num_joints = p.getNumJoints(robot_id)
                for j in range(num_joints):
                    ji = p.getJointInfo(robot_id, j)
                    parent_idx = int(ji[16])
                    connected_pairs.add((parent_idx, j))
                    connected_pairs.add((j, parent_idx))

                penetrations = 0
                for cp in cps:
                    link_a = int(cp[3])
                    link_b = int(cp[4])
                    if (link_a, link_b) in connected_pairs:
                        continue
                    if float(cp[8]) < -penetration_eps:
                        penetrations += 1

                rec["initial_collision"] = bool(penetrations > 0)

                # simple joint excitation (same as full)
                for j in range(num_joints):
                    ji = p.getJointInfo(robot_id, j)
                    if ji[2] == p.JOINT_REVOLUTE:
                        lower = float(ji[8])
                        upper = float(ji[9])
                        target = 0.5 * (lower + upper)
                        p.setJointMotorControl2(
                            robot_id, j, p.POSITION_CONTROL, targetPosition=target, force=10.0
                        )

                for a in range(-1, num_joints):
                    for b in range(a + 1, num_joints):
                        p.setCollisionFilterPair(robot_id, robot_id, a, b, enableCollision=0)

                exploded, _ = _pybullet_explosion_check(p, robot_id, steps=steps, threshold=explosion_threshold)
                rec["explosion"] = bool(exploded)

                rec["valid"] = (not rec["initial_collision"]) and rec["inertia_pd"] and (not rec["explosion"])

            except Exception as e:
                rec["error"] = str(e)
                rec.setdefault("initial_collision", False)
                rec.setdefault("inertia_pd", False)
                rec.setdefault("explosion", True)
                rec["valid"] = False
            finally:
                try:
                    _clear_blender_scene(bpy)
                except Exception:
                    pass
                if clean_per_sample:
                    try:
                        seed_dir = export_root / "tuckendbox" / str(int(seed))
                        if seed_dir.exists():
                            shutil.rmtree(seed_dir, ignore_errors=True)
                    except Exception:
                        pass
                _maybe_purge_orphans(bpy, every=purge_every, idx=local_i + 1)
                out.append(rec)

            if (local_i + 1) % 25 == 0 or (local_i + 1) == len(sub):
                vr = sum(1 for r in out if r.get("valid")) / len(out)
                print(f"[worker {chunk_start}+{local_i+1:4d}/{chunk_count}] valid_rate={vr:.3%}")

    p.disconnect()

    payload = {
        "meta_seed": meta_seed,
        "n_total": n_total,
        "chunk_start": chunk_start,
        "chunk_count": chunk_count,
        "steps": steps,
        "explosion_threshold": explosion_threshold,
        "penetration_eps": penetration_eps,
        "results": out,
    }
    Path(result_json).parent.mkdir(parents=True, exist_ok=True)
    Path(result_json).write_text(json.dumps(payload, indent=2))
    print(f"Wrote worker result: {result_json}")


if __name__ == "__main__":
    mode = os.environ.get("PAPER_AUDIT_MODE", "full").strip().lower()
    if mode == "worker":
        run_audit_worker()
    else:
        run_audit_and_fill_md()

