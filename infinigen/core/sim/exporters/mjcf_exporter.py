# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: primary author

import json
import re
import xml.dom.minidom
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional
from xml.dom.minidom import parseString

import bmesh
import bpy
import mujoco
import numpy as np

import infinigen.core.sim.exporters.utils as exputils
from infinigen.core import surface
from infinigen.core.sim.exporters.base import (
    JointType,
    PathItem,
    RigidBody,
    SimBuilder,
)
from infinigen.core.sim.kinematic_node import (
    KinematicNode,
)
from infinigen.core.sim.physics import joint_dynamics as jointdyna
from infinigen.core.sim.physics import material_physics as mtlphysics
from infinigen.tools.export import export_sim_ready, skipBake


def create_element(tag: str, **kwargs) -> ET.Element:
    return ET.Element(tag, attrib=kwargs)


class MJCFBuilder(SimBuilder):
    def __init__(self, assets_dir):
        super().__init__(assets_dir)

        self.mujoco = self._initialize_mjcf()

        self.asset_freq = defaultdict(int)
        self.joint_freq = defaultdict(int)
        self.joint_map = dict()
        self.joint_to_coord_frame = dict()

        self.post_processing_collision_info = defaultdict(dict)
        self.exclude_links = set()

        self.link_count = 0

    @property
    def xml(self):
        """Returns the MJCF as a string."""
        rough_string = ET.tostring(self.mujoco, "utf-8")
        reparsed = xml.dom.minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def _initialize_semantic_mapping(self, exporter: str) -> None:
        semantic_entities = list(self.metadata.get("semantic_entities", []))
        self.semantic_mapping = {
            "schema_version": 1,
            "asset_name": str(self.metadata.get("asset_name", "unknown")),
            "exporter": exporter,
            "entities": [],
        }
        self._semantic_entities_by_label = {}
        for entity in semantic_entities:
            record = {
                "entity_uid": str(entity.get("entity_uid", "")).strip(),
                "part_label": str(entity.get("part_label", "")).strip(),
                "entity_kind": str(entity.get("entity_kind", "generic_part")).strip(),
                "resolution_policy": str(entity.get("resolution_policy", "exact_geom_name_set")).strip(),
                "exported_visual_geom_names": [],
                "exported_collision_geom_names": [],
                "exported_visual_mesh_files": [],
                "exported_collision_mesh_files": [],
            }
            if record["entity_uid"]:
                self.semantic_mapping["entities"].append(record)
            if record["part_label"]:
                self._semantic_entities_by_label[record["part_label"]] = record

    def _primary_part_label(self, geometry: bpy.types.Object) -> str:
        labels = self._get_labels(geometry)
        if len(labels) == 0:
            return "geom"
        labels_w_counts = []
        for lab in sorted(labels):
            if lab not in set([n.name for n in geometry.data.attributes]):
                continue
            vert = int(sum(surface.read_attr_data(geometry, lab)))
            labels_w_counts.append((vert, lab))
        if not labels_w_counts:
            return sorted(labels)[0]
        min_count = min(count for count, _ in labels_w_counts)
        min_labels = sorted(label for count, label in labels_w_counts if count == min_count)
        return min_labels[0]

    def _next_export_names(self, primary_part_label: str, collision_count: int) -> tuple[str, list[str]]:
        asset_idx = self.asset_freq[primary_part_label]
        self.asset_freq[primary_part_label] += 1
        visual_name = f"{primary_part_label}_visual_{asset_idx}"
        collision_names = [f"{primary_part_label}_collision_{asset_idx}_{col_idx}" for col_idx in range(collision_count)]
        return visual_name, collision_names

    def _rename_exported_asset(self, asset_path: Path, desired_stem: str) -> Path:
        desired_path = asset_path.with_name(f"{desired_stem}{asset_path.suffix}")
        if asset_path == desired_path:
            return asset_path
        if desired_path.exists():
            desired_path.unlink()
        asset_path.replace(desired_path)
        return desired_path

    def _register_semantic_mapping(self, primary_part_label: str, visual_name: str, collision_names: list[str], visasset_path: Path, colasset_paths: list[Path]) -> None:
        record = self._semantic_entities_by_label.get(primary_part_label)
        if not record:
            return
        record["exported_visual_geom_names"].append(visual_name)
        record["exported_visual_mesh_files"].append(f"assets/{visasset_path.name}")
        for collision_name, colasset_path in zip(collision_names, colasset_paths):
            record["exported_collision_geom_names"].append(collision_name)
            record["exported_collision_mesh_files"].append(f"assets/{colasset_path.name}")

    def _validate_semantic_mapping(self) -> None:
        if str(self.metadata.get("asset_name", "")) != "drawer":
            return
        handle_entities = [e for e in self.semantic_mapping.get("entities", []) if e.get("entity_uid") == "drawer_handle"]
        if len(handle_entities) != 1:
            raise ValueError("drawer_handle entity missing or nonunique in semantic_mapping")
        handle = handle_entities[0]
        visual_names = list(handle.get("exported_visual_geom_names", []))
        collision_names = list(handle.get("exported_collision_geom_names", []))
        all_names = visual_names + collision_names
        if not visual_names:
            raise ValueError("drawer_handle visual geom names are empty")
        if any(not str(name).strip() for name in all_names):
            raise ValueError("drawer_handle exported geom names contain empty strings")
        if len(all_names) != len(set(all_names)):
            raise ValueError("drawer_handle exported geom names are duplicate")

    def _initialize_mjcf(self) -> ET.Element:
        """
        Initializes an MJCF file required to construct an asset.
        """
        mujoco = create_element("mujoco")

        # adding compiler attributes
        self.compiler = create_element("compiler", angle="radian", meshdir="assets")
        mujoco.append(self.compiler)

        # creating general defaults
        default = create_element("default")
        geom_default = create_element("geom", rgba="1 1 1 1")
        default.append(geom_default)
        mujoco.append(default)

        # create root level for the worldbody and the asset
        self.asset = create_element("asset")
        self.worldbody = create_element("worldbody")
        self.main_body = create_element("body", name="object")
        self.worldbody.append(self.main_body)
        self.contact = create_element("contact")
        mujoco.append(self.asset)
        mujoco.append(self.worldbody)
        mujoco.append(self.contact)

        return mujoco

    def build(
        self,
        blend_obj: bpy.types.Object,
        kinematic_root: KinematicNode,
        sample_joint_params_fn: Callable,
        metadata: Dict,
        visual_only: bool = False,
        image_res: int = 512,
    ):
        super().build(blend_obj, metadata)
        self._initialize_semantic_mapping("mjcf")

        if not visual_only:
            self.compiler.set("inertiagrouprange", "0 0")

        # construct a skeleton for the rigid body
        root, _ = self._construct_rigid_body_skeleton(kinematic_root)
        root = self._wrap_in_body(root)
        self._simplify_skeleton(root)

        asset_body, _ = self._populate_mjcf(
            root, visual_only=visual_only, image_res=image_res
        )
        self._populate_joints(asset_body, sample_joint_params_fn)
        self.main_body.append(asset_body)

        self._sort_asset_elements(self.asset)

    def _populate_mjcf(
        self,
        root: RigidBody,
        joint_nodes: List[KinematicNode] = [],
        pos_offset: np.array = np.zeros(3),
        visual_only: bool = False,
        image_res: int = 512,
    ):
        """Populates the mjcf with assets and joints."""

        # create the link element
        link_name = f"link_{self.link_count}"
        link = create_element("body", name=link_name)
        self.link_count += 1

        # add all the assets for the body
        visgeom_refs = []
        colgeom_refs = []
        assets = []
        all_attribs_root = [at.attribs for at in root.assets]
        for asset in root.assets:
            visgeom, colgeoms, asset = self._add_mesh(
                asset.attribs, link, visual_only, image_res
            )
            if asset:
                visgeom_refs.append(visgeom)
                colgeom_refs.append(colgeoms)
                assets.append(asset)

        aabb_center = exputils.get_aabb_center(assets)

        # set the link pos relative to the parent line
        link.set("pos", exputils.array_to_string(aabb_center - pos_offset))

        # set the position offsets for geometries
        for visgeom, colgeoms, asset in zip(visgeom_refs, colgeom_refs, assets):
            geom_center = exputils.get_aabb_center(asset)
            offset = geom_center - aabb_center
            visgeom.set("pos", exputils.array_to_string(offset))
            for colgeom in colgeoms:
                colgeom.set("pos", exputils.array_to_string(offset))

                # add information for post processing
                self.post_processing_collision_info[link_name][colgeom.get("name")] = {
                    "offset": geom_center,
                    "path": self.asset.find(f"mesh[@name='{colgeom.get('mesh')}']").get(
                        "file"
                    ),
                }

        # add joints to the body if they exist
        if joint_nodes:
            for node in joint_nodes:
                joint_name = self.metadata[node.idn]["joint label"]
                unique_joint_idx = self.joint_freq[joint_name]
                unique_joint_name = f"{joint_name}_{unique_joint_idx}"
                self.joint_freq[joint_name] += 1
                joint_type = "hinge" if node.joint_type == JointType.HINGE else "slide"

                coord_frame = exputils.get_coord_frame(
                    self.blend_obj, node.idn, unique_joint_idx, aabb_center
                )

                # store the coordinate frame for the unique joint
                self.joint_to_coord_frame[unique_joint_name] = coord_frame

                joint = create_element(
                    "joint",
                    name=unique_joint_name,
                    type=joint_type,
                    pos="0 0 0",
                    axis="0 0 0",
                )
                self.joint_map[unique_joint_name] = node.idn
                link.append(joint)

        # add all children bodies
        for child, joints in root.children.items():
            child_link, child_name = self._populate_mjcf(
                child,
                joint_nodes=joints,
                pos_offset=aabb_center,
                visual_only=visual_only,
            )
            link.append(child_link)

            # exclude contacts between parent and child bodies
            self.contact.append(
                create_element("exclude", body1=link_name, body2=child_name)
            )
            self.exclude_links.add((link_name, child_name))

        return link, link_name

    def _add_mesh(
        self,
        attribs: List[PathItem],
        body: ET.Element,
        visual_only: bool,
        image_res: int,
        zaxis: np.array = np.array([0, 0, 1]),
    ):
        attribs_tuple = exputils.attribs_to_tuples(attribs)
        extra_attribs = [("axis_group", 0)]
        asset = exputils.get_geometry_given_attribs(
            self.blend_obj, attribs_tuple, extra_attribs=extra_attribs
        )
        if exputils.is_2d(asset):
            return None, None, None
        primary_part_label = self._primary_part_label(asset)

        # export the asset
        geometry_center = exputils.get_aabb_center(asset)
        preview_name = f"{primary_part_label}_visual_{self.asset_freq[primary_part_label]}"
        export_paths = export_sim_ready(
            asset,
            output_folder=self.assets_dir,
            image_res=image_res,
            translation=-geometry_center,
            name=preview_name,
            visual_only=visual_only,
            zaxis=zaxis,
        )
        visual_name, collision_names = self._next_export_names(primary_part_label, 0 if visual_only else len(export_paths["collision"]))
        visasset_path = export_paths["visual"][0]
        if visasset_path.stem != visual_name:
            visasset_path = self._rename_exported_asset(visasset_path, visual_name)

        mesh_temp = asset.to_mesh()
        bm = bmesh.new()
        bm.from_mesh(mesh_temp)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.transform(asset.matrix_world)
        vol = bm.calc_volume(signed=False)
        bm.free()

        # add the visual asset to the list of assets in the scene
        self._add_asset(
            asset_name=visual_name,
            asset_path=visasset_path,
            asset_type="visual",
            has_material=not skipBake(asset),
            intertia="legacy" if not np.isclose(vol, mujoco.mjMINVAL) else "shell",
        )

        # getting material physical properties
        mat_physics = mtlphysics.get_material_properties(asset)

        # create and link a geom for the asset
        visgeom = create_element(
            "geom",
            name=visual_name,
            type="mesh",
            mesh=visual_name,
            group="1",
            contype="0",
            conaffinity="0",
            friction=f"{mat_physics['friction']} 0.005 0.0001",
            density=f"{mat_physics['density']}",
        )
        if not skipBake(asset):
            visgeom.set("material", f"{visual_name}_mat")
        body.append(visgeom)

        colgeoms = []
        final_collision_paths = []
        if not visual_only:
            # add the collision asset to the list of assets in the scene
            for col_idx, colasset_path in enumerate(export_paths["collision"]):
                colasset_name = collision_names[col_idx]
                renamed_colasset_path = self._rename_exported_asset(colasset_path, colasset_name)
                self._add_asset(
                    asset_name=colasset_name,
                    asset_path=renamed_colasset_path,
                    asset_type="collision",
                    has_material=False,
                )

                # create and link a geom for the asset
                colgeom = create_element(
                    "geom",
                    name=colasset_name,
                    type="mesh",
                    mesh=colasset_name,
                    group="0",
                    contype="1",
                    conaffinity="1",
                    friction=f"{mat_physics['friction']} 0.005 0.0001",
                    density=f"{mat_physics['density']}",
                )
                body.append(colgeom)
                colgeoms.append(colgeom)
                final_collision_paths.append(renamed_colasset_path)

        self._register_semantic_mapping(primary_part_label, visual_name, collision_names, visasset_path, final_collision_paths)
        return visgeom, colgeoms, asset

    def _add_asset(
        self,
        asset_name: str,
        asset_path: Path,
        asset_type: str,
        has_material: bool,
        intertia: str = "convex",
    ):
        """Adds a mesh along with its materials and texture to the mjcf."""
        mesh_element = create_element(
            "mesh",
            name=asset_name,
            file=str(f"{asset_type}/{asset_path.name}"),
            inertia=intertia,
        )
        self.asset.append(mesh_element)

        # add a material if it exists for the part
        if has_material:
            texture_element = create_element(
                "texture",
                name=f"{asset_name}_tex",
                type="2d",
                file=f"assets/textures/{asset_name}_DIFFUSE.png",
            )
            material_element = create_element(
                "material", name=f"{asset_name}_mat", texture=f"{asset_name}_tex"
            )
            self.asset.append(texture_element)
            self.asset.append(material_element)

    def _populate_joints(self, body: ET.Element, sample_joint_params_fn: Callable):
        """
        Populates all the joints with the true value given the object.
        """
        # sample the physics distribution for the joints
        joint_params = sample_joint_params_fn()

        for joint in body.findall(".//joint"):
            joint_name = joint.get("name")
            prefix = self.joint_map[joint_name]

            poschild, axis, range_min, range_max = exputils.get_joint_properties(
                self.blend_obj, prefix
            )

            # get the current coordinate frame
            R = self.joint_to_coord_frame[joint_name]

            # set the position of the joint relative to the child
            joint.set("pos", exputils.array_to_string(R @ poschild))

            # set the axis of the joint
            joint.set("axis", exputils.array_to_string(R @ axis))

            # set the min and max range for the joint values
            # Fix: if range is zero, set reasonable defaults based on joint type
            if np.isclose(range_max, 0.0) and np.isclose(range_min, 0.0):
                joint_type = joint.get("type")
                if joint_type == "hinge":
                    # Default hinge range: 0 to π (180 degrees)
                    range_min = 0.0
                    range_max = np.pi
                    print(
                        f"Warning: Joint {joint_name} has zero range, setting default [0, π] for hinge"
                    )
                elif joint_type == "slide":
                    # Default slide range: -0.5 to 0.5 meters
                    range_min = -0.5
                    range_max = 0.5
                    print(
                        f"Warning: Joint {joint_name} has zero range, setting default [-0.5, 0.5] for slide"
                    )

            if not (np.isclose(range_max, 0.0) and np.isclose(range_min, 0.0)):
                joint.set("limited", "true")
                joint.set("range", f"{range_min} {range_max}")

            ref = 0.0
            if 0 < range_min:
                ref = range_min
            if range_max < 0:
                ref = range_max
            joint.set("ref", f"{ref}")

            # set joint physics parameters
            nonunique_joint_name = re.sub(r"_\d+$", "", joint_name)
            joint_properties = jointdyna.get_joint_properties(
                nonunique_joint_name, joint_params
            )

            if nonunique_joint_name in joint_params:
                joint.set("stiffness", str(joint_properties["stiffness"]))
                joint.set("damping", str(joint_properties["damping"]))
                joint.set("frictionloss", str(joint_properties["friction"]))

    def _sort_asset_elements(self, asset: ET.Element):
        mesh_elements = []
        material_elements = []
        texture_elements = []
        other_elements = []

        for child in list(asset):
            if child.tag == "mesh":
                mesh_elements.append(child)
            elif child.tag == "material":
                material_elements.append(child)
            elif child.tag == "texture":
                texture_elements.append(child)
            else:
                other_elements.append(child)

        # Clear the current children
        asset.clear()

        # Re-add in the desired order
        for elem in (
            mesh_elements + material_elements + texture_elements + other_elements
        ):
            asset.append(elem)


def export(
    blend_obj: bpy.types.Object,
    sim_blueprint: Dict,
    seed: int,
    sample_joint_params_fn: Callable,
    export_dir: Path = Path("./sim_exports/mjcf"),
    image_res: int = 512,
    visual_only: bool = True,
    get_raw_output: bool = False,
    options: Optional[Dict] = None,
    extra_exclude: Optional[set] = None,
    **kwargs,
):
    """Export function for the MJCF file format."""
    # parse the provided blueprint and set the object export directory
    asset_name, kinematic_root, metadata = exputils.parse_sim_blueprint(sim_blueprint)
    metadata["asset_name"] = asset_name
    metadata["semantic_entities"] = list(sim_blueprint.get("semantic_entities", []))

    # create export directories
    obj_export_dir = export_dir / asset_name / str(seed)
    obj_assets_dir = obj_export_dir / "assets"
    obj_export_dir.mkdir(parents=True, exist_ok=True)
    obj_assets_dir.mkdir(parents=True, exist_ok=True)

    # build asset
    builder = MJCFBuilder(obj_assets_dir)
    builder.build(
        blend_obj=blend_obj,
        kinematic_root=kinematic_root,
        sample_joint_params_fn=sample_joint_params_fn,
        metadata=metadata,
        visual_only=visual_only,
        image_res=image_res,
    )

    metadata.update(builder.get_bounding_box_info())

    # post process collision geometries and exclude links
    links = builder.post_processing_collision_info.keys()
    if extra_exclude is not None:
        for e in extra_exclude:
            if e[0] in links and e[1] in links:
                builder.contact.append(
                    create_element("exclude", body1=e[0], body2=e[1])
                )
                builder.exclude_links.add(e)

    exputils.post_process_collisions(
        builder.post_processing_collision_info, obj_assets_dir, builder.exclude_links
    )

    # additional options
    if options is not None:
        builder.mujoco.append(create_element("option", **options))

    if get_raw_output:
        return builder.mujoco, metadata

    builder._validate_semantic_mapping()

    # save the mjcf
    mjcf_path, semantic_mapping_path = save(
        fname=asset_name,
        export_dir=obj_export_dir,
        contents=builder.mujoco,
        metadata=metadata,
        semantic_mapping=builder.semantic_mapping,
    )

    return mjcf_path, semantic_mapping_path


def save(fname: str, export_dir: Path, contents: ET.Element, metadata: Dict, semantic_mapping: Dict) -> None:
    """Save the MJCF contents."""
    mjcf_path = export_dir / f"{fname}.xml"
    with open(mjcf_path, "w") as f:
        raw_xml = ET.tostring(contents, encoding="unicode")
        formatted_xml = parseString(raw_xml).toprettyxml(indent="  ")
        lines = [line for line in formatted_xml.splitlines() if line.strip()]
        f.write("\n".join(lines))

    metadata_path = export_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    semantic_mapping_path = export_dir / "semantic_mapping.json"
    with open(semantic_mapping_path, "w") as f:
        json.dump(semantic_mapping, f, indent=4)

    return mjcf_path, semantic_mapping_path
