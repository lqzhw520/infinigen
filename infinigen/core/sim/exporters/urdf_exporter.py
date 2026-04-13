# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: primary author

import json
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional
from xml.dom.minidom import parseString

import bmesh
import bpy
import numpy as np
import trimesh

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
from infinigen.core.sim.physics import thin_shell_inertia as thinshell
from infinigen.tools.export import export_sim_ready


def create_element(tag: str, **kwargs) -> ET.Element:
    return ET.Element(tag, attrib=kwargs)


class URDFBuilder(SimBuilder):
    def __init__(self, assets_dir):
        super().__init__(assets_dir)

        self.urdf = self._initialize_urdf()
        self.debug = os.getenv("INFINIGEN_URDF_DEBUG", "").strip() not in ("", "0", "false", "False")

        # create a joint that links the top most link to the world
        self._create_joint(
            name="world_joint",
            joint_type=JointType.WELD,
            origin=np.array([0.0, 0.0, 0.0]),
            parent_link="world",
            child_link="link_0",
        )

        self.asset_freq = defaultdict(int)
        self.joint_freq = defaultdict(int)
        self.joint_map = dict()

        self.post_processing_collision_info = defaultdict(dict)
        self.exclude_links = set()

        self.link_count = 0

    @property
    def xml(self):
        """Returns the URDF as a string."""
        rough_string = ET.tostring(self.urdf, "utf-8")
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

    def _initialize_urdf(self) -> ET.Element:
        """
        Initializes an URDF file required to construct an asset.
        """
        robot = create_element("robot", name="object")
        world = create_element("link", name="world")
        robot.append(world)

        return robot

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
        self._initialize_semantic_mapping("urdf")

        # construct a skeleton for the rigid body
        root, _ = self._construct_rigid_body_skeleton(kinematic_root)
        root = self._wrap_in_body(root)
        self._simplify_skeleton(root)

        joint_params = sample_joint_params_fn()
        self._populate_links(
            root,
            visual_only=visual_only,
            image_res=image_res,
            joint_params=joint_params,
        )

    def _populate_links(
        self,
        root: RigidBody,
        joint_params: Dict,
        parent_link: str = "world",
        joint_nodes: List[KinematicNode] = [],
        pos_offset: np.array = np.zeros(3),
        parent_abs_pos: np.array = np.zeros(3),
        visual_only: bool = False,
        image_res: int = 512,
    ):
        """Populates the urdf with links and joints.
        
        重要修复 (URDF 规范合规):
        ========================
        URDF 规范规定一个 <link> 只能有一个 <inertial> 元素。
        当一个 link 包含多个几何体 (assets) 时，必须使用平行轴定理
        将所有几何体的惯性合并为一个。
        
        参考: http://wiki.ros.org/urdf/XML/link
        """

        # create a link for the body
        link_name = f"link_{self.link_count}"
        link = create_element("link", name=link_name)
        self.link_count += 1

        if self.debug:
            try:
                joint_ids = [getattr(jn, "idn", str(jn)) for jn in (joint_nodes or [])]
            except Exception:
                joint_ids = ["<unprintable>"]
            print(
                f"[URDF_DEBUG] begin link={link_name} parent={parent_link} "
                f"pos_offset_in={pos_offset} joint_nodes={joint_ids}"
            )

        self.exclude_links.add((parent_link, link_name))

        vis_origin_refs = []
        col_origin_refs = []
        col_paths = []
        assets = []
        
        # ============================================================
        # 惯性数据收集器 (用于后续合并)
        # URDF 规范要求: 一个 link 只能有一个 <inertial> 元素
        # 因此必须使用平行轴定理将多个几何体的惯性合并
        # ============================================================
        inertia_masses = []        # 各几何体的质量
        inertia_coms = []          # 各几何体的质心位置
        inertia_tensors = []       # 各几何体的惯性张量
        
        for asset in root.assets:
            # export the mesh and set the filename
            visasset_path, colasset_paths, mesh, visual_name, collision_names = self._get_mesh(
                asset.attribs, visual_only=visual_only, image_res=image_res
            )
            if not mesh:
                continue

            # add all the assets for the given link
            visual = create_element("visual", name=visual_name)
            visual_origin = create_element("origin", xyz="0.0 0.0 0.0")
            geometry = create_element("geometry")
            mesh_element = create_element("mesh")

            mesh_element.set("filename", f"assets/{visasset_path.name}")
            geometry.append(mesh_element)
            visual.append(geometry)
            visual.append(visual_origin)

            link.append(visual)

            mat_physics = mtlphysics.get_material_properties(mesh)
            # 回退: 如果未识别材质或密度异常，尝试从对象自定义属性读取
            if (
                ("density" not in mat_physics)
                or mat_physics.get("density", 0) <= 0
                or mat_physics.get("density", 0) >= 2000  # 卡纸/瓦楞/木/塑料常用范围
            ):
                try:
                    override_density = float(self.blend_obj.get("physics_density", 0))
                    if override_density > 0:
                        mat_physics["density"] = override_density
                except Exception:
                    pass

            # Estimate the mass of the object given the density
            mesh_temp = mesh.to_mesh()
            bm = bmesh.new()
            bm.from_mesh(mesh_temp)
            bmesh.ops.triangulate(bm, faces=bm.faces)
            bm.transform(mesh.matrix_world)
            vol = bm.calc_volume(signed=False)
            bm.free()

            # R-Deep-1 修复: 使用稳健惯性计算 (薄壳修正)
            vertices = np.array([list(vertex.co) for vertex in mesh.data.vertices])
            faces = np.array([
                list(triangle.vertices) for triangle in mesh.data.loop_triangles
            ])
            
            robust_mass, I_tensor, com = thinshell.calculate_robust_inertia(
                vertices=vertices,
                faces=faces,
                density=mat_physics["density"],
                volume=vol,
            )
            
            # ============================================================
            # 收集惯性数据，稍后使用平行轴定理合并
            # (不再在循环中创建 inertial 元素)
            # ============================================================
            inertia_masses.append(robust_mass)
            inertia_coms.append(com)
            inertia_tensors.append(I_tensor)

            collision_refs = []
            collision_paths = []
            if not visual_only:
                for col_idx, colasset_path in enumerate(colasset_paths):
                    collision = create_element("collision", name=collision_names[col_idx])
                    collision_origin = create_element("origin", xyz="0.0 0.0 0.0")
                    geometry = create_element("geometry")
                    mesh_element = create_element("mesh")

                    mesh_element.set("filename", f"assets/{colasset_path.name}")
                    geometry.append(mesh_element)
                    collision.append(geometry)
                    collision.append(collision_origin)
                    link.append(collision)
                    collision_refs.append(collision_origin)
                    collision_paths.append(colasset_path.name)

            vis_origin_refs.append(visual_origin)
            col_origin_refs.append(collision_refs)
            col_paths.append(collision_paths)
            assets.append(mesh)
        
        # ============================================================
        # 使用平行轴定理合并所有几何体的惯性
        # 创建单个 <inertial> 元素 (符合 URDF 规范)
        # ============================================================
        if len(inertia_masses) > 0:
            # 使用 thin_shell_inertia.combine_multiple_inertias 函数
            # 该函数实现了严格的平行轴定理:
            #   I_new = I_old + m × [(r·r)×E - r⊗r]
            combined_mass, combined_com, combined_inertia = thinshell.combine_multiple_inertias(
                masses=inertia_masses,
                centers_of_mass=inertia_coms,
                inertia_tensors=inertia_tensors,
            )
            
            if self.debug:
                print(
                    f"[URDF_DEBUG] link={link_name} 惯性合并: "
                    f"n_assets={len(inertia_masses)}, "
                    f"individual_masses={inertia_masses}, "
                    f"combined_mass={combined_mass:.6f} kg, "
                    f"combined_com={combined_com}"
                )
            
            # 创建单个 inertial 元素
            inertial = create_element("inertial")
            mass_element = create_element("mass", value=str(combined_mass))
            inertial.append(mass_element)
            
            # 提取惯性张量分量
            ixx, ixy, ixz = combined_inertia[0]
            _, iyy, iyz = combined_inertia[1]
            _, _, izz = combined_inertia[2]

            inertia = create_element(
                "inertia",
                ixx=str(ixx),
                ixy=str(ixy),
                ixz=str(ixz),
                iyy=str(iyy),
                iyz=str(iyz),
                izz=str(izz),
            )
            inertial.append(inertia)

            origin = create_element("origin", xyz=exputils.array_to_string(combined_com))
            inertial.append(origin)

            # 只添加一个 inertial 元素 (符合 URDF 规范)
            link.append(inertial)

        aabb_center = exputils.get_aabb_center(assets)

        # calculate the absolute joint position
        # R6 修复: 支持多关节连接 (创建中间 link)
        if len(joint_nodes) > 0:
            current_parent_link = parent_link
            current_pos_offset = pos_offset
            
            for joint_idx, joint_node in enumerate(joint_nodes):
                # 确定当前关节的子 link
                is_last_joint = (joint_idx == len(joint_nodes) - 1)
                
                if is_last_joint:
                    # 最后一个关节连接到实际的 child link
                    current_child_link = link_name
                else:
                    # 中间关节连接到一个无质量的中间 link
                    intermediate_link_name = f"link_{self.link_count}_intermediate_{joint_idx}"
                    self.link_count += 1
                    
                    # 创建无质量中间 link
                    intermediate_link = create_element("link", name=intermediate_link_name)
                    # 添加最小惯性以保持 URDF 有效性
                    inertial = create_element("inertial")
                    inertial.append(create_element("mass", value="0.001"))  # 1g 虚拟质量
                    inertial.append(create_element(
                        "inertia",
                        ixx="1e-9", ixy="0", ixz="0",
                        iyy="1e-9", iyz="0", izz="1e-9"
                    ))
                    inertial.append(create_element("origin", xyz="0 0 0"))
                    intermediate_link.append(inertial)
                    self.urdf.append(intermediate_link)
                    
                    current_child_link = intermediate_link_name
                    self.exclude_links.add((current_parent_link, current_child_link))
                
                # 获取关节属性
                joint_name = self.metadata[joint_node.idn]["joint label"]
                unique_joint_idx = self.joint_freq[joint_name]
                unique_joint_name = f"{joint_name}_{self.joint_freq[joint_name]}"
                self.joint_freq[joint_name] += 1

                coord_frame = R = exputils.get_coord_frame(
                    self.blend_obj, joint_node.idn, unique_joint_idx, aabb_center
                )

                poschild, axis, range_min, range_max = exputils.get_joint_properties(
                    self.blend_obj, joint_node.idn
                )
                abs_joint_pos = aabb_center + R @ poschild

                joint_properties = jointdyna.get_joint_properties(joint_name, joint_params)

                self._create_joint(
                    name=unique_joint_name,
                    joint_type=joint_node.joint_type,
                    origin=abs_joint_pos - current_pos_offset,
                    parent_link=current_parent_link,
                    child_link=current_child_link,
                    min_range=range_min,
                    max_range=range_max,
                    axis=coord_frame @ axis,
                    damping=joint_properties["damping"],
                    friction=joint_properties["friction"],
                )

                if self.debug:
                    print(
                        f"[URDF_DEBUG] joint name={unique_joint_name} parent={current_parent_link} "
                        f"child={current_child_link} abs_joint_pos={abs_joint_pos} "
                        f"origin={abs_joint_pos - current_pos_offset}"
                    )
                
                # 更新下一个关节的 parent 和 offset
                current_parent_link = current_child_link
                current_pos_offset = abs_joint_pos
            
            pos_offset = current_pos_offset

        if self.debug:
            print(f"[URDF_DEBUG] link={link_name} pos_offset_link={pos_offset} aabb_center={aabb_center}")

        # set the position of the links geometries relative to the joint
        # TODO (ajoshi): Clean this up.
        for vis_origin, col_origins, cpaths, asset in zip(
            vis_origin_refs, col_origin_refs, col_paths, assets
        ):
            geom_center = exputils.get_aabb_center(asset)
            offset = geom_center - pos_offset
            if self.debug:
                print(
                    f"[URDF_DEBUG]   asset={getattr(asset, 'name', '<unnamed>')} "
                    f"geom_center={geom_center} offset={offset}"
                )
            vis_origin.set("xyz", exputils.array_to_string(offset))
            for col_origin, path in zip(col_origins, cpaths):
                col_origin.set("xyz", exputils.array_to_string(offset))

                # add information for post processing
                self.post_processing_collision_info[link_name][path] = {
                    "offset": geom_center,
                    "path": path,
                }

        for child, joints in root.children.items():
            self._populate_links(
                child,
                joint_params,
                parent_link=link_name,
                joint_nodes=joints,
                pos_offset=pos_offset,
                parent_abs_pos=aabb_center,
                visual_only=visual_only,
            )

        self.urdf.append(link)

    def _create_joint(
        self,
        name: str,
        joint_type: JointType,
        origin: np.ndarray,
        parent_link: str,
        child_link: str,
        damping: float = 0.0,
        friction: float = 0.0,
        min_range: Optional[float] = -np.pi,
        max_range: Optional[float] = np.pi,
        axis: Optional[np.ndarray] = None,
    ):
        if joint_type == JointType.HINGE:
            jt = "revolute"
        elif joint_type == JointType.SLIDING:
            jt = "prismatic"
        elif joint_type == JointType.WELD or joint_type == JointType.NONE:
            jt = "fixed"
        else:
            raise ValueError("Joint is not valid")

        joint = create_element("joint", name=name, type=jt)
        joint.append(create_element("origin", xyz=exputils.array_to_string(origin)))
        joint.append(create_element("parent", link=parent_link))
        joint.append(create_element("child", link=child_link))
        joint.append(
            create_element("dynamics", damping=str(damping), friction=str(friction))
        )

        if joint_type != JointType.WELD:
            joint.append(create_element("axis", xyz=exputils.array_to_string(axis)))

            if min_range == max_range == 0:
                # default ranges when range undefined
                if joint_type == JointType.HINGE:
                    min_range = -np.pi
                    max_range = np.pi
                elif joint_type == JointType.SLIDING:
                    min_range = -100
                    max_range = 100
            joint.append(
                ET.Element(
                    "limit", attrib={"lower": str(min_range), "upper": str(max_range)}
                )
            )

        self.urdf.append(joint)
        return joint

    def _get_mesh(self, attribs: List[PathItem], visual_only: bool, image_res: int):
        attribs = exputils.attribs_to_tuples(attribs)
        extra_attribs = [("axis_group", 0)]
        mesh = exputils.get_geometry_given_attribs(
            self.blend_obj, attribs, extra_attribs=extra_attribs
        )

        # return None is the asset it not a proper volume
        if exputils.is_2d(mesh):
            return None, None, None, None, None

        primary_part_label = self._primary_part_label(mesh)

        # export the asset
        geometry_center = exputils.get_aabb_center(mesh)
        preview_name = f"{primary_part_label}_visual_{self.asset_freq[primary_part_label]}"
        export_paths = export_sim_ready(
            mesh,
            output_folder=self.assets_dir,
            image_res=image_res,
            translation=-geometry_center,
            separate_asset_dirs=False,
            name=preview_name,
            visual_only=visual_only,
        )

        visual_name, collision_names = self._next_export_names(primary_part_label, 0 if visual_only else len(export_paths["collision"]))
        visasset_path = export_paths["visual"][0]
        if visasset_path.stem != visual_name:
            visasset_path = self._rename_exported_asset(visasset_path, visual_name)
        colasset_paths = []
        for col_idx, colasset_path in enumerate(export_paths["collision"]):
            renamed_colasset_path = self._rename_exported_asset(colasset_path, collision_names[col_idx])
            colasset_paths.append(renamed_colasset_path)

        self._register_semantic_mapping(primary_part_label, visual_name, collision_names, visasset_path, colasset_paths)
        return visasset_path, colasset_paths, mesh, visual_name, collision_names


def export(
    blend_obj: bpy.types.Object,
    sim_blueprint: Dict,
    seed: int,
    sample_joint_params_fn: Callable,
    export_dir: Path = Path("./sim_exports/urdf"),
    image_res: int = 512,
    visual_only: bool = True,
    get_raw_output: bool = False,
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
    builder = URDFBuilder(obj_assets_dir)
    builder.build(
        blend_obj=blend_obj,
        kinematic_root=kinematic_root,
        sample_joint_params_fn=sample_joint_params_fn,
        metadata=metadata,
        visual_only=visual_only,
        image_res=image_res,
    )

    metadata.update(builder.get_bounding_box_info())

    exputils.post_process_collisions(
        builder.post_processing_collision_info, obj_assets_dir, builder.exclude_links
    )

    if get_raw_output:
        return builder.urdf, metadata

    builder._validate_semantic_mapping()

    # save the urdf
    urdf_path, semantic_mapping_path = save(
        fname=asset_name,
        export_dir=obj_export_dir,
        contents=builder.urdf,
        metadata=metadata,
        semantic_mapping=builder.semantic_mapping,
    )

    return urdf_path, semantic_mapping_path


def save(fname: str, export_dir: Path, contents: ET.Element, metadata: Dict, semantic_mapping: Dict) -> None:
    """Save the URDF contents."""
    urdf_path = export_dir / f"{fname}.urdf"
    with open(urdf_path, "w") as f:
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

    return urdf_path, semantic_mapping_path
