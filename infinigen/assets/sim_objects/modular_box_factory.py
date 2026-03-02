# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""
ModularBoxFactory - 模块化盒子生成工厂

这是 Phase 1 的核心架构组件，解决了 R-Deep-2 (单一巨型节点树) 问题。

特点:
1. 动态加载盒型特定的几何节点生成函数
2. 统一的参数采样接口
3. 支持 16 种盒型的可扩展架构
4. 集成关节注入和材质系统
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Type

import bpy
import numpy as np
from numpy.random import randint, uniform

from infinigen.assets.utils.joints import nodegroup_hinge_joint, nodegroup_sliding_joint
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil

logger = logging.getLogger(__name__)


class BoxType(Enum):
    """16 种盒型枚举"""

    TUCK_END = 1  # 双插盒
    LOCK_BOTTOM = 2  # 锁底盒
    TUCK_END_SAFETY = 3  # 带保险双插盒
    LOCK_BOTTOM_SAFETY = 4  # 带保险锁底盒
    HOOK_LOCK_BOTTOM = 5  # 带挂钩锁底盒
    HOOK_TUCK_END = 6  # 带挂钩双插盒
    MAILER = 7  # 飞机盒
    DRAWER = 8  # 抽屉盒
    SLIP_LID = 9  # 天地盒
    GIFT_WITH_HANDLE = 10  # 自带手提礼盒
    TUCK_HANDLE = 11  # 对插手提礼盒
    GABLE_TOP = 12  # 屋脊手提礼盒
    DOUBLE_LID_GIFT = 13  # 双盖手提礼盒
    PLASTIC_HANDLE = 14  # 塑料手提礼盒
    RSC = 15  # 平口箱 (Regular Slotted Container)
    CUSTOM = 16  # 自定义盒型


@dataclass
class BoxDimensions:
    """盒子尺寸参数"""

    width: float = 0.2  # X 方向 (米)
    depth: float = 0.2  # Y 方向 (米)
    height: float = 0.15  # Z 方向 (米)
    thickness: float = 0.002  # 壁厚 (米)

    def validate(self) -> Tuple[bool, str]:
        """验证尺寸是否合理"""
        issues = []

        if self.width <= 0 or self.depth <= 0 or self.height <= 0:
            issues.append("尺寸必须为正数")

        if self.thickness <= 0:
            issues.append("壁厚必须为正数")

        if self.thickness > min(self.width, self.depth, self.height) * 0.1:
            issues.append("壁厚过大 (> 最小尺寸的 10%)")

        if self.width > 2 or self.depth > 2 or self.height > 2:
            issues.append("尺寸过大 (> 2m)")

        if issues:
            return False, "; ".join(issues)
        return True, "尺寸有效"


@dataclass
class BoxJointConfig:
    """盒子关节配置"""

    joint_type: str = "hinge"  # "hinge" 或 "slide"
    label: str = ""
    position: Tuple[float, float, float] = (0, 0, 0)
    axis: Tuple[float, float, float] = (0, 1, 0)
    min_angle: float = 0.0
    max_angle: float = np.pi
    damping: float = 0.5
    friction: float = 0.1


@dataclass
class BoxMaterialConfig:
    """盒子材质配置

    密度参考 material_definitions.py:
    - Cardboard: 200-400 kg/m³
    - Corrugated: 100-300 kg/m³
    """

    # ----------------------------
    # Physics parameters
    # ----------------------------
    # NOTE: keep material_type stable to allow `material_physics.py` to infer defaults.
    # Recommended box material keys:
    # - cardboard / cardboard_kraft / cardboard_white / corrugated / plastic_box / wood_box
    material_type: str = "cardboard"
    density: float = 300.0  # kg/m³
    friction: float = 0.5
    restitution: float = 0.1

    # ----------------------------
    # Visual shader parameters (Principled BSDF)
    # ----------------------------
    color: Tuple[float, float, float] = (0.6, 0.5, 0.4)  # RGB
    roughness: float = 0.75
    specular: float = 0.5
    metallic: float = 0.0
    transmission: float = 0.0
    ior: float = 1.45
    clearcoat: float = 0.0
    clearcoat_roughness: float = 0.03

    # Normal / bump (procedural)
    bump_strength: float = 0.0
    bump_scale: float = 30.0

    # Optional: create per-asset unique shader instances while preserving physics lookup.
    # If provided, `_apply_box_material` will use `shader_<material_type>_deepcopy_<variant_id>`.
    variant_id: Optional[int] = None


@dataclass
class BoxParameters:
    """完整的盒子参数"""

    box_type: BoxType = BoxType.TUCK_END
    dimensions: BoxDimensions = field(default_factory=BoxDimensions)
    material: BoxMaterialConfig = field(default_factory=BoxMaterialConfig)
    joints: List[BoxJointConfig] = field(default_factory=list)

    # 盒型特定参数
    extra_params: Dict = field(default_factory=dict)


class ModularBoxFactory(AssetFactory, ABC):
    """
    模块化盒子生成工厂基类

    子类需要实现:
    - get_box_type(): 返回盒型枚举
    - create_geometry_nodegroup(): 创建几何节点组
    - get_default_joints(): 返回默认关节配置
    """

    # 注册的盒型工厂
    _registry: Dict[BoxType, Type["ModularBoxFactory"]] = {}

    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=coarse)
        self._cached_nodegroup = None

    @classmethod
    def register(cls, box_type: BoxType):
        """装饰器: 注册盒型工厂"""

        def decorator(factory_cls):
            cls._registry[box_type] = factory_cls
            logger.debug(f"注册盒型工厂: {box_type.name} -> {factory_cls.__name__}")
            return factory_cls

        return decorator

    @classmethod
    def get_factory(cls, box_type: BoxType) -> Type["ModularBoxFactory"]:
        """获取指定盒型的工厂类"""
        if box_type not in cls._registry:
            raise ValueError(f"盒型 {box_type.name} 未注册")
        return cls._registry[box_type]

    @classmethod
    def list_registered_types(cls) -> List[BoxType]:
        """列出所有已注册的盒型"""
        return list(cls._registry.keys())

    # --- 抽象方法: 子类必须实现 ---

    @abstractmethod
    def get_box_type(self) -> BoxType:
        """返回此工厂生成的盒型"""
        raise NotImplementedError

    @abstractmethod
    def create_geometry_nodegroup(self, nw: NodeWrangler, params: BoxParameters):
        """
        创建盒子的几何节点组

        Args:
            nw: NodeWrangler 实例
            params: 盒子参数
        """
        raise NotImplementedError

    @abstractmethod
    def get_default_joints(self, params: BoxParameters) -> List[BoxJointConfig]:
        """
        返回此盒型的默认关节配置

        Args:
            params: 盒子参数

        Returns:
            关节配置列表
        """
        raise NotImplementedError

    # --- 可选覆盖的方法 ---

    def sample_dimensions(self) -> BoxDimensions:
        """采样盒子尺寸 (可覆盖以自定义)"""
        return BoxDimensions(
            width=uniform(0.1, 0.5),
            depth=uniform(0.1, 0.5),
            height=uniform(0.08, 0.4),
            thickness=uniform(0.001, 0.005),
        )

    def sample_material(self) -> BoxMaterialConfig:
        """采样材质配置 (可覆盖以自定义)

        密度参考 material_definitions.py 中的定义:
        - Cardboard: 200-400 kg/m³
        - Corrugated: 100-300 kg/m³
        - Plastic: 900-1400 kg/m³
        - Wood: 500-900 kg/m³
        """
        # Prefer sampling from the physics registry (material_definitions.py) when possible.
        from infinigen.core.sim.physics import material_definitions as matdefs

        # NOTE: This default sampler is intentionally conservative.
        # Domain-randomization for research datasets should use an explicit RNG in the dataset exporter.
        material_types = [
            "cardboard_kraft",
            "corrugated",
            "cardboard_white",
            "plastic_box",
        ]
        selected = material_types[randint(0, len(material_types))]

        # Sample physics
        mat_cls = matdefs.MATERIALS.get(selected, None)
        phys = mat_cls().sample_parameters() if mat_cls is not None else {}
        density = float(phys.get("density", 300.0))
        friction = float(phys.get("friction", 0.5))
        restitution = float(phys.get("restitution", 0.1))

        # Sample visuals (lightweight defaults)
        if selected in ("cardboard_kraft", "cardboard"):
            color = (
                float(uniform(0.50, 0.70)),
                float(uniform(0.40, 0.60)),
                float(uniform(0.25, 0.45)),
            )
            roughness = float(uniform(0.70, 0.92))
            bump_strength = float(uniform(0.05, 0.20))
        elif selected.startswith("corrugated"):
            color = (
                float(uniform(0.50, 0.70)),
                float(uniform(0.40, 0.60)),
                float(uniform(0.25, 0.45)),
            )
            roughness = float(uniform(0.75, 0.95))
            bump_strength = float(uniform(0.10, 0.35))
        elif selected == "cardboard_white":
            color = (
                float(uniform(0.75, 0.92)),
                float(uniform(0.75, 0.92)),
                float(uniform(0.75, 0.92)),
            )
            roughness = float(uniform(0.25, 0.55))
            bump_strength = float(uniform(0.00, 0.08))
        elif "plastic" in selected:
            color = (
                float(uniform(0.05, 0.95)),
                float(uniform(0.05, 0.95)),
                float(uniform(0.05, 0.95)),
            )
            roughness = float(uniform(0.10, 0.45))
            bump_strength = float(uniform(0.00, 0.05))
        else:
            color = (0.6, 0.5, 0.4)
            roughness = 0.75
            bump_strength = 0.0

        # Additional shader tuning
        clearcoat = 0.0
        transmission = 0.0
        if selected == "cardboard_white":
            # A reasonable proxy for glossy laminated paper.
            clearcoat = float(uniform(0.2, 0.8))
        if "plastic" in selected:
            transmission = float(uniform(0.0, 0.35))

        return BoxMaterialConfig(
            material_type=selected,
            density=density,
            friction=friction,
            restitution=restitution,
            color=color,
            roughness=roughness,
            transmission=transmission,
            clearcoat=clearcoat,
            bump_strength=bump_strength,
            bump_scale=float(uniform(15.0, 60.0)),
        )

    def sample_parameters(self) -> BoxParameters:
        """采样完整的盒子参数"""
        dimensions = self.sample_dimensions()
        material = self.sample_material()

        params = BoxParameters(
            box_type=self.get_box_type(),
            dimensions=dimensions,
            material=material,
        )

        # 获取默认关节配置
        params.joints = self.get_default_joints(params)

        return params

    def sample_joint_parameters(self) -> Dict:
        """
        采样关节动力学参数

        返回与现有 BoxFactory.sample_joint_parameters 兼容的格式
        """
        params = self.sample_parameters()
        result = {}

        for joint in params.joints:
            result[joint.label] = {
                "stiffness": 0.0,
                "damping": joint.damping,
                "friction": joint.friction,
            }

        return result

    # --- 核心生成方法 ---

    def create_asset(self, asset_params=None, **kwargs) -> bpy.types.Object:
        """
        创建盒子资产

        Args:
            asset_params: 可选的预设参数

        Returns:
            Blender 对象
        """
        # 采样参数
        if asset_params is None:
            params = self.sample_parameters()
        else:
            params = asset_params

        # 验证尺寸
        valid, msg = params.dimensions.validate()
        if not valid:
            logger.warning(f"盒子尺寸验证失败: {msg}")

        # 创建基础对象
        obj = butil.spawn_vert()

        # 获取或创建几何节点组
        nodegroup = self._get_or_create_nodegroup(params)

        # 应用几何节点
        ng_inputs = self._params_to_ng_inputs(params)
        butil.modify_mesh(
            obj,
            "NODES",
            apply=False,
            node_group=nodegroup,
            ng_inputs=ng_inputs,
        )

        # P2.1-T2 修复: 为对象添加命名材质以便 URDF 导出器识别
        self._apply_box_material(obj, params.material)

        return obj

    def _apply_box_material(self, obj, material_config: BoxMaterialConfig):
        """为对象应用正确命名的材质，以便 URDF 导出器使用正确的物理属性"""
        # Create or retrieve a (potentially per-asset unique) shader material.
        # material_physics.py will strip `shader_` and truncate at `_deepcopy`.
        if material_config.variant_id is not None:
            mat_name = f"shader_{material_config.material_type}_deepcopy_{int(material_config.variant_id)}"
        else:
            mat_name = f"shader_{material_config.material_type}"

        if mat_name in bpy.data.materials:
            mat = bpy.data.materials[mat_name]
        else:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True

            # Configure principled shader
            if mat.node_tree:
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links
                bsdf = nodes.get("Principled BSDF")
                out = nodes.get("Material Output")
                if bsdf is not None:
                    # Blender versions may rename Principled BSDF sockets (e.g., Specular/Coat/Transmission).
                    if "Base Color" in bsdf.inputs:
                        bsdf.inputs["Base Color"].default_value = (
                            *material_config.color,
                            1.0,
                        )
                    if "Roughness" in bsdf.inputs:
                        bsdf.inputs["Roughness"].default_value = float(
                            material_config.roughness
                        )

                    # Specular naming changed in Blender 4.x.
                    if "Specular" in bsdf.inputs:
                        bsdf.inputs["Specular"].default_value = float(
                            material_config.specular
                        )
                    elif "Specular IOR Level" in bsdf.inputs:
                        bsdf.inputs["Specular IOR Level"].default_value = float(
                            material_config.specular
                        )

                    if "Metallic" in bsdf.inputs:
                        bsdf.inputs["Metallic"].default_value = float(
                            material_config.metallic
                        )

                    # Transmission naming changed in Blender 4.x.
                    if "Transmission" in bsdf.inputs:
                        bsdf.inputs["Transmission"].default_value = float(
                            material_config.transmission
                        )
                    elif "Transmission Weight" in bsdf.inputs:
                        bsdf.inputs["Transmission Weight"].default_value = float(
                            material_config.transmission
                        )

                    if "IOR" in bsdf.inputs:
                        bsdf.inputs["IOR"].default_value = float(material_config.ior)

                    # Clearcoat naming changed in Blender 4.x.
                    for coat_name in ("Clearcoat", "Coat Weight"):
                        if coat_name in bsdf.inputs:
                            bsdf.inputs[coat_name].default_value = float(
                                material_config.clearcoat
                            )
                            break
                    for coat_r_name in ("Clearcoat Roughness", "Coat Roughness"):
                        if coat_r_name in bsdf.inputs:
                            bsdf.inputs[coat_r_name].default_value = float(
                                material_config.clearcoat_roughness
                            )
                            break

                # Optional procedural bump (corrugated / kraft etc)
                # Only build once (material is either new or uniquely named).
                if (
                    mat.node_tree
                    and bsdf is not None
                    and out is not None
                    and float(material_config.bump_strength) > 0
                    and "Normal" in bsdf.inputs
                ):
                    bump = nodes.new("ShaderNodeBump")
                    bump.inputs["Strength"].default_value = float(
                        material_config.bump_strength
                    )

                    # Choose texture primitive by material family
                    mname = str(material_config.material_type).lower()
                    if "corrugated" in mname:
                        tex = nodes.new("ShaderNodeTexWave")
                        tex.inputs["Scale"].default_value = float(
                            material_config.bump_scale
                        )
                        # Prefer banded ridges; orient along X by default.
                        if hasattr(tex, "bands_direction"):
                            tex.bands_direction = "X"
                        height_socket = tex.outputs.get("Fac", None) or tex.outputs[0]
                    else:
                        tex = nodes.new("ShaderNodeTexNoise")
                        tex.inputs["Scale"].default_value = float(
                            material_config.bump_scale
                        )
                        height_socket = tex.outputs.get("Fac", None) or tex.outputs[0]

                    links.new(height_socket, bump.inputs["Height"])
                    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

        # 确保对象有 mesh 数据后再应用材质
        # 注意: 在几何节点应用前，对象可能只是顶点
        # 材质会在 URDF 导出时传递给生成的 mesh
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        # Write physics parameters to object custom properties (URDF exporter can use density override).
        try:
            obj["physics_density"] = float(material_config.density)
        except Exception:
            pass
        try:
            obj["physics_friction"] = float(material_config.friction)
        except Exception:
            pass
        try:
            obj["physics_restitution"] = float(material_config.restitution)
        except Exception:
            pass

    def _get_or_create_nodegroup(self, params: BoxParameters):
        """获取或创建几何节点组"""
        # 创建节点组名称
        ng_name = f"nodegroup_{self.get_box_type().name.lower()}_box"

        # 检查是否已存在
        if ng_name in bpy.data.node_groups:
            return bpy.data.node_groups[ng_name]

        # 创建新的节点组
        @node_utils.to_nodegroup(ng_name, singleton=False, type="GeometryNodeTree")
        def _create_ng(nw: NodeWrangler):
            self.create_geometry_nodegroup(nw, params)

        return _create_ng()

    def _params_to_ng_inputs(self, params: BoxParameters) -> Dict:
        """将参数转换为几何节点输入"""
        return {
            "Width": params.dimensions.width,
            "Depth": params.dimensions.depth,
            "Height": params.dimensions.height,
            "Thickness": params.dimensions.thickness,
            **params.extra_params,
        }


# ============================================================
# 基础盒型实现示例: TuckEndBox (双插盒)
# ============================================================


@ModularBoxFactory.register(BoxType.TUCK_END)
class TuckEndBoxFactory(ModularBoxFactory):
    """
    双插盒 (Tuck End Box) 工厂

    特点: 上下两个翻盖
    关节: 2-4 个铰链 (上盖、下盖、可选的侧翼)
    复杂度: ⭐⭐
    """

    def get_box_type(self) -> BoxType:
        return BoxType.TUCK_END

    def get_default_joints(self, params: BoxParameters) -> List[BoxJointConfig]:
        """双插盒的默认关节配置"""
        dims = params.dimensions

        # 目标：boxes-style.jpg #1 双插盒
        # 默认动作：向内折入（closing），即 joint value 从 0 -> max
        max_fold = np.pi / 2  # 90° 作为第一版闭合动作（后续用碰撞验证收缩）
        return [
            # 顶部主翻盖（背面顶边，轴=X）
            BoxJointConfig(
                joint_type="hinge",
                label="top_back_flap",
                position=(0, dims.depth / 2, dims.height / 2),
                axis=(1, 0, 0),
                min_angle=0.0,
                max_angle=max_fold,
                damping=0.4,
            ),
            # 顶部插舌二段折（插入盒口）：位于前侧盒口边缘，轴=X
            BoxJointConfig(
                joint_type="hinge",
                label="top_tuck_tab",
                position=(0, -dims.depth / 2, dims.height / 2),
                axis=(1, 0, 0),
                min_angle=0.0,
                max_angle=max_fold,
                damping=0.25,
            ),
            # 顶部左 dust flap（左侧顶边，轴=Y，向内为 +X）
            BoxJointConfig(
                joint_type="hinge",
                label="top_dust_left",
                position=(-dims.width / 2, 0.0, dims.height / 2),
                axis=(0, 1, 0),
                min_angle=0.0,
                max_angle=max_fold,
                damping=0.35,
            ),
            # 顶部右 dust flap（右侧顶边，轴=-Y，向内为 -X）
            BoxJointConfig(
                joint_type="hinge",
                label="top_dust_right",
                position=(dims.width / 2, 0.0, dims.height / 2),
                axis=(0, -1, 0),
                min_angle=0.0,
                max_angle=max_fold,
                damping=0.35,
            ),
            # 底部主翻盖（正面底边，轴=X）
            BoxJointConfig(
                joint_type="hinge",
                label="bottom_front_flap",
                position=(0, -dims.depth / 2, -dims.height / 2),
                axis=(1, 0, 0),
                min_angle=0.0,
                max_angle=max_fold,
                damping=0.4,
            ),
            # 底部插舌二段折：位于后侧盒口边缘，轴=X
            BoxJointConfig(
                joint_type="hinge",
                label="bottom_tuck_tab",
                position=(0, dims.depth / 2, -dims.height / 2),
                axis=(1, 0, 0),
                min_angle=0.0,
                max_angle=max_fold,
                damping=0.25,
            ),
            # 底部左 dust flap（左侧底边，轴=Y，向内为 +X）
            BoxJointConfig(
                joint_type="hinge",
                label="bottom_dust_left",
                position=(-dims.width / 2, 0.0, -dims.height / 2),
                # 注意：底部 dust flap 初始朝下（Z<0），要“向内”折入需要反转轴方向
                axis=(0, -1, 0),
                min_angle=0.0,
                max_angle=max_fold,
                damping=0.35,
            ),
            # 底部右 dust flap（右侧底边，向内为 -X）
            BoxJointConfig(
                joint_type="hinge",
                label="bottom_dust_right",
                position=(dims.width / 2, 0.0, -dims.height / 2),
                axis=(0, 1, 0),
                min_angle=0.0,
                max_angle=max_fold,
                damping=0.35,
            ),
        ]

    def sample_dimensions(self) -> BoxDimensions:
        """
        双插盒（纸盒直筒）更接近 boxes-style.jpg #1：
        - 高度通常明显大于宽/深
        - 厚度更薄（卡纸）
        """
        return BoxDimensions(
            width=uniform(0.06, 0.18),
            depth=uniform(0.05, 0.14),
            height=uniform(0.18, 0.45),
            thickness=uniform(0.0008, 0.002),
        )

    def create_geometry_nodegroup(self, nw: NodeWrangler, params: BoxParameters):
        """创建双插盒的几何节点组 (直筒薄壁 + 上下折页 + 插舌/锁扣占位)"""
        dims = params.dimensions

        # 创建输入节点（仅暴露尺寸；折页比例后续可做成可配置参数）
        group_input = nw.new_node(
            Nodes.GroupInput,
            expose_input=[
                ("NodeSocketFloat", "Width", dims.width),
                ("NodeSocketFloat", "Depth", dims.depth),
                ("NodeSocketFloat", "Height", dims.height),
                ("NodeSocketFloat", "Thickness", dims.thickness),
            ],
        )

        # ========== 通用量 ==========
        half_width = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Width"], 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )
        half_depth = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Depth"], 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )
        half_height = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Height"], 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )
        half_thickness = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )
        # NOTE: 这里保持与导出器/关节定义一致的“厚度外偏”策略：
        # - lid/front_flap 在其局部坐标系沿 +Y 偏移 half_thickness，使得在常用闭合方向上更贴近盒体外侧
        neg_half_thickness = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_thickness, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )
        neg_half_thickness = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_thickness, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )

        neg_half_width = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_width, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )
        neg_half_depth = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_depth, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )
        neg_half_height = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_height, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )

        # ========= 盒体：四面薄壁直筒（无上下盖） =========
        # 外轮廓对齐：墙体中心 = half_dim - half_thickness
        y_back = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_depth, 1: half_thickness},
            attrs={"operation": "SUBTRACT"},
        )
        y_front = nw.new_node(
            Nodes.Math,
            input_kwargs={0: y_back, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )
        x_right = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_width, 1: half_thickness},
            attrs={"operation": "SUBTRACT"},
        )
        x_left = nw.new_node(
            Nodes.Math,
            input_kwargs={0: x_right, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )

        # 前后墙：宽 x 厚 x 高
        wall_fb = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": group_input.outputs["Width"],
                        "Y": group_input.outputs["Thickness"],
                        "Z": group_input.outputs["Height"],
                    },
                ),
            },
        )
        front_wall = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": wall_fb,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"Y": y_front}
                ),
            },
        )
        back_wall = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": wall_fb,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"Y": y_back}
                ),
            },
        )

        # 左右墙：厚 x 深 x 高
        wall_lr = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": group_input.outputs["Thickness"],
                        "Y": group_input.outputs["Depth"],
                        "Z": group_input.outputs["Height"],
                    },
                ),
            },
        )
        left_wall = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": wall_lr,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"X": x_left}
                ),
            },
        )
        right_wall = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": wall_lr,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"X": x_right}
                ),
            },
        )

        body = nw.new_node(
            Nodes.JoinGeometry,
            input_kwargs={"Geometry": [front_wall, back_wall, left_wall, right_wall]},
        )

        # ========== 折页尺寸（对齐双插盒插舌插入逻辑） ==========
        # 主翻盖“面板”长度：需要基本覆盖整个盒口（从一侧折痕到另一侧盒口边缘）
        # 经验值：Depth - Thickness（留出少量厚度余量，避免穿墙）
        major_flap_len = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: group_input.outputs["Depth"],
                1: group_input.outputs["Thickness"],
            },
            attrs={"operation": "SUBTRACT"},
        )
        major_flap_half = nw.new_node(
            Nodes.Math,
            input_kwargs={0: major_flap_len, 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )

        # dust flap 长度（沿 Z）：取 width 的 35%
        # 说明：dust flaps 折入后会占据盒口两侧空间；过长会阻挡插舌下插并产生穿插。
        dust_flap_len = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Width"], 1: 0.35},
            attrs={"operation": "MULTIPLY"},
        )
        dust_flap_half = nw.new_node(
            Nodes.Math,
            input_kwargs={0: dust_flap_len, 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )

        # 插舌长度：与 Depth 挂钩（而非随面板长度线性放大）
        # 经验值：约为 Depth 的 20%（插入够深但不至于太长造成干涉）
        tab_len = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Depth"], 1: 0.20},
            attrs={"operation": "MULTIPLY"},
        )
        tab_half = nw.new_node(
            Nodes.Math,
            input_kwargs={0: tab_len, 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )
        tab_width = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Width"], 1: 0.28},
            attrs={"operation": "MULTIPLY"},
        )

        # ========== 顶部主翻盖（背面顶边，初始为竖直“展开态”） ==========
        # 注意：nodegroup_hinge_joint 会以输入的 Position 作为枢轴原点（并把 Child 放到该位置）
        # 因此 Child 的几何应在“关节局部坐标系”下建模（相对 hinge 位置），避免重复平移导致折页悬浮
        top_back_panel = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": group_input.outputs["Width"],
                        "Y": group_input.outputs["Thickness"],
                        "Z": major_flap_len,
                    },
                ),
            },
        )
        top_back_panel = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": top_back_panel,
                "Translation": nw.new_node(
                    # 关节位于 (Y=+half_depth, Z=+half_height)
                    # flap 中心应位于 (Y=y_back, Z=half_height+major_flap_half)
                    # => 局部偏移:
                    #   - Y: 选择 +half_thickness 让“主翻盖”在闭合时位于 dust flaps 之上（减少穿插）
                    #   - Z: major_flap_half
                    Nodes.CombineXYZ,
                    input_kwargs={"Y": half_thickness, "Z": major_flap_half},
                ),
            },
        )

        # ---------- 顶部插舌：作为独立刚体 + 独立铰链（模拟“插入盒口”需要的二段折叠） ----------
        # tab 关节位于面板末端。
        # 经验：由于 hinge_joint 元数据的坐标系以 body transform / bbox 为基准，
        # 这里使用 major_flap_half 来让导出的 URDF 关节落在面板末端（避免额外 +major_flap_half 的偏移）。
        top_tab_hinge_pos = nw.new_node(
            Nodes.CombineXYZ,
            input_kwargs={"Y": half_thickness, "Z": major_flap_half},
        )

        top_tuck_tab = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": tab_width,
                        "Y": group_input.outputs["Thickness"],
                        "Z": tab_len,
                    },
                )
            },
        )
        top_tuck_tab = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": top_tuck_tab,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ,
                    # tab 自身的局部：以 tab hinge 为原点，向 +Z 伸出 tab_len
                    # 同时将厚度整体偏向“盒内侧”，让插舌落在盒壁内侧（而不是穿过盒壁厚度）
                    input_kwargs={
                        # 注意：在最终闭合态 tab 的 local +Y 指向 world -Y（盒外侧），
                        # 因此这里用 -half_thickness 将厚度推向盒内侧（world +Y）
                        "Y": neg_half_thickness,
                        "Z": tab_half,
                    },
                ),
            },
        )

        # tab 先与面板通过铰链连接，形成 flap 子结构：panel -> tab
        j_top_tab = nw.new_node(
            nodegroup_hinge_joint().name,
            input_kwargs={
                "Joint Label": "top_tuck_tab",
                "Parent": top_back_panel,
                "Child": top_tuck_tab,
                "Position": top_tab_hinge_pos,
                "Axis": (1, 0, 0),
                "Min": 0.0,
                "Max": np.pi / 2,
            },
        )
        top_back_flap = j_top_tab.outputs["Geometry"]

        # ========== 底部主翻盖（正面底边，初始为竖直“展开态”） ==========
        bottom_front_panel = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": group_input.outputs["Width"],
                        "Y": group_input.outputs["Thickness"],
                        "Z": major_flap_len,
                    },
                ),
            },
        )
        neg_major_flap_half = nw.new_node(
            Nodes.Math,
            input_kwargs={0: major_flap_half, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )
        bottom_front_panel = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": bottom_front_panel,
                "Translation": nw.new_node(
                    # 关节位于 (Y=-half_depth, Z=-half_height)
                    # flap 中心应位于 (Y=y_front, Z=-half_height-major_flap_half)
                    # => 局部偏移:
                    #   - Y: 选择 -half_thickness 让“底部主翻盖”在闭合时位于 dust flaps 之下（减少穿插）
                    #   - Z: -major_flap_half
                    Nodes.CombineXYZ,
                    input_kwargs={"Y": neg_half_thickness, "Z": neg_major_flap_half},
                ),
            },
        )

        # ---------- 底部插舌（独立刚体+铰链） ----------
        bottom_tab_hinge_pos = nw.new_node(
            Nodes.CombineXYZ,
            input_kwargs={"Y": neg_half_thickness, "Z": neg_major_flap_half},
        )

        bottom_tuck_tab = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": tab_width,
                        "Y": group_input.outputs["Thickness"],
                        "Z": tab_len,
                    },
                )
            },
        )
        bottom_tuck_tab = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": bottom_tuck_tab,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ,
                    # 底部插舌同理：偏向盒内侧（背面内侧方向为 -Y）
                    input_kwargs={
                        # 底部插舌最终闭合态 local +Y 同样指向 world -Y（盒内侧），
                        # 因此使用 +half_thickness 将厚度推向盒内侧
                        "Y": half_thickness,
                        "Z": nw.new_node(
                            Nodes.Math,
                            input_kwargs={0: tab_half, 1: -1.0},
                            attrs={"operation": "MULTIPLY"},
                        ),
                    },
                ),
            },
        )

        j_bottom_tab = nw.new_node(
            nodegroup_hinge_joint().name,
            input_kwargs={
                "Joint Label": "bottom_tuck_tab",
                "Parent": bottom_front_panel,
                "Child": bottom_tuck_tab,
                "Position": bottom_tab_hinge_pos,
                "Axis": (1, 0, 0),
                "Min": 0.0,
                "Max": np.pi / 2,
            },
        )
        bottom_front_flap = j_bottom_tab.outputs["Geometry"]

        # ========== dust flaps（左右侧壁，上/下各一片） ==========
        dust_flap_geom = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": group_input.outputs["Thickness"],
                        "Y": group_input.outputs["Depth"],
                        "Z": dust_flap_len,
                    },
                )
            },
        )

        neg_dust_flap_half = nw.new_node(
            Nodes.Math,
            input_kwargs={0: dust_flap_half, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )
        top_dust_left = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": dust_flap_geom,
                "Translation": nw.new_node(
                    # hinge: (X=-half_width, Z=+half_height)
                    # dust center: (X=x_left, Z=half_height+dust_flap_half)
                    # => 局部偏移: (X=+half_thickness, Z=+dust_flap_half)
                    Nodes.CombineXYZ,
                    input_kwargs={"X": half_thickness, "Z": dust_flap_half},
                ),
            },
        )
        top_dust_right = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": dust_flap_geom,
                "Translation": nw.new_node(
                    # hinge: (X=+half_width, Z=+half_height)
                    # => 局部偏移: (X=-half_thickness, Z=+dust_flap_half)
                    Nodes.CombineXYZ,
                    input_kwargs={"X": neg_half_thickness, "Z": dust_flap_half},
                ),
            },
        )

        bottom_dust_left = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": dust_flap_geom,
                "Translation": nw.new_node(
                    # hinge: (X=-half_width, Z=-half_height)
                    # => 局部偏移: (X=+half_thickness, Z=-dust_flap_half)
                    Nodes.CombineXYZ,
                    input_kwargs={"X": half_thickness, "Z": neg_dust_flap_half},
                ),
            },
        )
        bottom_dust_right = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": dust_flap_geom,
                "Translation": nw.new_node(
                    # hinge: (X=+half_width, Z=-half_height)
                    # => 局部偏移: (X=-half_thickness, Z=-dust_flap_half)
                    Nodes.CombineXYZ,
                    input_kwargs={"X": neg_half_thickness, "Z": neg_dust_flap_half},
                ),
            },
        )

        # ========== 铰链位置（折痕） ==========
        hinge_top_back_pos = nw.new_node(
            Nodes.CombineXYZ, input_kwargs={"Y": half_depth, "Z": half_height}
        )
        hinge_bottom_front_pos = nw.new_node(
            Nodes.CombineXYZ, input_kwargs={"Y": neg_half_depth, "Z": neg_half_height}
        )
        hinge_top_left_pos = nw.new_node(
            Nodes.CombineXYZ, input_kwargs={"X": neg_half_width, "Z": half_height}
        )
        hinge_top_right_pos = nw.new_node(
            Nodes.CombineXYZ, input_kwargs={"X": half_width, "Z": half_height}
        )
        hinge_bottom_left_pos = nw.new_node(
            Nodes.CombineXYZ, input_kwargs={"X": neg_half_width, "Z": neg_half_height}
        )
        hinge_bottom_right_pos = nw.new_node(
            Nodes.CombineXYZ, input_kwargs={"X": half_width, "Z": neg_half_height}
        )

        # ========== 关节（默认 closing: 0 -> π/2） ==========
        geo = body

        j_top_back = nw.new_node(
            nodegroup_hinge_joint().name,
            input_kwargs={
                "Joint Label": "top_back_flap",
                "Parent": geo,
                "Child": top_back_flap,
                "Position": hinge_top_back_pos,
                "Axis": (1, 0, 0),
                "Min": 0.0,
                "Max": np.pi / 2,
            },
        )
        geo = j_top_back.outputs["Geometry"]

        j_top_left = nw.new_node(
            nodegroup_hinge_joint().name,
            input_kwargs={
                "Joint Label": "top_dust_left",
                "Parent": geo,
                "Child": top_dust_left,
                "Position": hinge_top_left_pos,
                "Axis": (0, 1, 0),
                "Min": 0.0,
                "Max": np.pi / 2,
            },
        )
        geo = j_top_left.outputs["Geometry"]

        j_top_right = nw.new_node(
            nodegroup_hinge_joint().name,
            input_kwargs={
                "Joint Label": "top_dust_right",
                "Parent": geo,
                "Child": top_dust_right,
                "Position": hinge_top_right_pos,
                "Axis": (0, -1, 0),
                "Min": 0.0,
                "Max": np.pi / 2,
            },
        )
        geo = j_top_right.outputs["Geometry"]

        j_bottom_front = nw.new_node(
            nodegroup_hinge_joint().name,
            input_kwargs={
                "Joint Label": "bottom_front_flap",
                "Parent": geo,
                "Child": bottom_front_flap,
                "Position": hinge_bottom_front_pos,
                "Axis": (1, 0, 0),
                "Min": 0.0,
                "Max": np.pi / 2,
            },
        )
        geo = j_bottom_front.outputs["Geometry"]

        j_bottom_left = nw.new_node(
            nodegroup_hinge_joint().name,
            input_kwargs={
                "Joint Label": "bottom_dust_left",
                "Parent": geo,
                "Child": bottom_dust_left,
                "Position": hinge_bottom_left_pos,
                # 底部 dust flap 初始朝下，轴方向与顶部相反才能向内折入
                "Axis": (0, -1, 0),
                "Min": 0.0,
                "Max": np.pi / 2,
            },
        )
        geo = j_bottom_left.outputs["Geometry"]

        j_bottom_right = nw.new_node(
            nodegroup_hinge_joint().name,
            input_kwargs={
                "Joint Label": "bottom_dust_right",
                "Parent": geo,
                "Child": bottom_dust_right,
                "Position": hinge_bottom_right_pos,
                "Axis": (0, 1, 0),
                "Min": 0.0,
                "Max": np.pi / 2,
            },
        )
        geo = j_bottom_right.outputs["Geometry"]

        # 输出
        nw.new_node(
            Nodes.GroupOutput,
            input_kwargs={"Geometry": geo},
            attrs={"is_active_output": True},
        )

    def _create_lid(self, nw: NodeWrangler, group_input, is_top: bool, y_offset=0.0):
        """创建盖子几何体"""
        # 盖子是一个薄板
        multiply_thickness = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: group_input.outputs["Thickness"],
                1: 1.0,
            },
            attrs={"operation": "MULTIPLY"},
        )

        lid = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": group_input.outputs["Width"],
                        "Y": group_input.outputs["Depth"],
                        "Z": multiply_thickness,
                    },
                ),
            },
        )

        # 移动到正确位置:
        # - Z: 顶盖放到上表面之上 (center 在 height/2 + thickness/2)
        #      底盖放到下表面之下 (center 在 -height/2 - thickness/2)
        # - Y: 仍以中心对齐，铰链轴在 y=depth/2 处
        offset_z = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: group_input.outputs["Height"],
                1: 2.0,
            },
            attrs={"operation": "DIVIDE"},
        )
        half_height = offset_z  # height / 2

        half_thickness = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: group_input.outputs["Thickness"],
                1: 2.0,
            },
            attrs={"operation": "DIVIDE"},
        )

        # 计算 Z 平移
        # 顶盖: z = height/2 + thickness/2
        # 底盖: z = -(height/2 + thickness/2)
        z_base = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: half_height,
                1: half_thickness,
            },
            attrs={"operation": "ADD"},
        )

        if is_top:
            z_translation = z_base
        else:
            # 底盖需要取负: z = -z_base
            z_translation = nw.new_node(
                Nodes.Math,
                input_kwargs={
                    0: z_base,
                    1: -1.0,
                },
                attrs={"operation": "MULTIPLY"},
            )

        translated = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": lid,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={"Y": y_offset, "Z": z_translation},
                ),
            },
        )

        return translated


# ============================================================
# 基础盒型实现: MailerBox-simple（飞机盒简化版，2-DOF）
# ============================================================


@ModularBoxFactory.register(BoxType.MAILER)
class MailerBoxFactory(ModularBoxFactory):
    """
    飞机盒 / MailerBox（简化版）

    目标样式: docs/images/MailerBox-simple.jpg / MailerBox-simple-1.jpg

    结构（仅做论文/演示所需的最小“2 折页铰链”版本，不包含复杂插舌/侧翼）:
    - 固定盒体: 底板 + 四周墙
    - 可动大折页（Lid）: 通过背面上沿铰链连接盒体
    - 可动小折页（Front flap）: 通过大折页前沿铰链连接大折页

    关节:
    - mailer_lid: body -> lid (hinge, axis=X)
    - mailer_front_flap: lid -> front_flap (hinge, axis=X)
    """

    def get_box_type(self) -> BoxType:
        return BoxType.MAILER

    def __init__(
        self, factory_seed=None, coarse=False, randomize_front_flap_len: bool = False
    ):
        """
        Args:
            randomize_front_flap_len:
                True: 第二折页（front flap）的长度（沿 Z）按 Height 比例随机，不保证与盒体高度完全对齐（用于需求 5）
                False: front flap 长度固定等于 Height（用于需求 4）
        """
        super().__init__(factory_seed=factory_seed, coarse=coarse)
        self.randomize_front_flap_len = randomize_front_flap_len

    def sample_dimensions(self) -> BoxDimensions:
        """
        飞机盒通常更“矮胖”，高度显著小于宽/深；厚度仍为薄纸板量级。
        """
        return BoxDimensions(
            width=uniform(0.10, 0.30),
            depth=uniform(0.10, 0.30),
            height=uniform(0.04, 0.14),
            thickness=uniform(0.0008, 0.0025),
        )

    def sample_parameters(self) -> BoxParameters:
        """
        采样完整参数，并注入本盒型的额外几何参数：
        - EdgeExtension: 盒体底部三侧凸出长度，同时用于 lid/front_flap 的横向（X）凸出，对齐且同步随机
        - FrontFlapLen: 第二折页（front flap）长度（沿 Z），可选随机
        """
        params = super().sample_parameters()
        dims = params.dimensions

        # 凸出长度（EdgeExtension）：
        # 需求口径：0.5cm~4cm（0.005~0.04m），并且不应超过 min(W,D) 的 15%（避免小盒子被夸张外扩）。
        min_wd = float(min(dims.width, dims.depth))
        max_ext = float(min(0.04, 0.15 * min_wd))
        min_ext = float(min(0.005, max_ext))
        edge_ext = float(uniform(min_ext, max_ext))

        if self.randomize_front_flap_len:
            # 第二折页长度：默认不超过 Height（允许“不到底”），范围 60%~100%
            front_flap_len = float(uniform(0.60, 1.00) * dims.height)
        else:
            front_flap_len = float(dims.height)

        params.extra_params["EdgeExtension"] = edge_ext
        params.extra_params["FrontFlapLen"] = front_flap_len
        return params

    def get_default_joints(self, params: BoxParameters) -> List[BoxJointConfig]:
        """简化飞机盒的默认关节配置（2 个铰链）"""
        dims = params.dimensions
        # 需求：支持“向内 + 向外”折页，因此 range 需要包含负角度。
        # 这里给两个铰链都开放到 [-π, +π]（±180°），在线查看器可自由调节。
        min_fold = -np.pi
        max_fold = np.pi

        # 说明：
        # - `mailer_lid` 的铰链位于背面上沿 (Y=+D/2, Z=+H/2)
        # - `mailer_front_flap` 的铰链位于 lid 的前沿；在“展开态”(joint=0) 时 lid 竖直上翻，
        #   因此前沿位于 (Y=+D/2, Z=+H/2 + D)
        return [
            BoxJointConfig(
                joint_type="hinge",
                label="mailer_lid",
                position=(0.0, dims.depth / 2, dims.height / 2),
                axis=(1, 0, 0),
                min_angle=min_fold,
                max_angle=max_fold,
                damping=0.45,
                friction=0.12,
            ),
            BoxJointConfig(
                joint_type="hinge",
                label="mailer_front_flap",
                position=(0.0, dims.depth / 2, dims.height / 2 + dims.depth),
                axis=(1, 0, 0),
                min_angle=min_fold,
                max_angle=max_fold,
                damping=0.30,
                friction=0.10,
            ),
        ]

    def create_geometry_nodegroup(self, nw: NodeWrangler, params: BoxParameters):
        """
        创建简化飞机盒几何节点组：
        - body: 底板 + 4 面墙（固定）
        - lid: 大折页（竖直展开态）
        - front_flap: 小折页（竖直展开态，连接在 lid 前沿）
        """
        dims = params.dimensions

        default_edge_ext = float(
            params.extra_params.get("EdgeExtension", 0.10 * min(dims.width, dims.depth))
        )
        default_front_flap_len = float(
            params.extra_params.get("FrontFlapLen", dims.height)
        )

        group_input = nw.new_node(
            Nodes.GroupInput,
            expose_input=[
                ("NodeSocketFloat", "Width", dims.width),
                ("NodeSocketFloat", "Depth", dims.depth),
                ("NodeSocketFloat", "Height", dims.height),
                ("NodeSocketFloat", "Thickness", dims.thickness),
                ("NodeSocketFloat", "EdgeExtension", default_edge_ext),
                ("NodeSocketFloat", "FrontFlapLen", default_front_flap_len),
            ],
        )

        # --- 常用量 ---
        half_width = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Width"], 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )
        half_depth = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Depth"], 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )
        half_height = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Height"], 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )
        half_thickness = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )
        quarter_thickness = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_thickness, 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )
        neg_half_thickness = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_thickness, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )

        neg_half_width = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_width, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )
        neg_half_depth = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_depth, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )
        neg_half_height = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_height, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )

        # --- EdgeExtension 相关量 ---
        edge_ext_x2 = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["EdgeExtension"], 1: 2.0},
            attrs={"operation": "MULTIPLY"},
        )
        half_edge_ext = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["EdgeExtension"], 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )
        neg_half_edge_ext = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_edge_ext, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )

        # --- 盒体（固定） ---
        # 前墙：宽 x 厚 x 高
        wall_front = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": group_input.outputs["Width"],
                        "Y": group_input.outputs["Thickness"],
                        "Z": group_input.outputs["Height"],
                    },
                ),
            },
        )
        # 背墙（与 lid 铰接的固定页）：(宽 + 2E) x 厚 x 高
        # 目的：与 “底板凸出 + 两折页凸出” 在侧边形成连续外轮廓，避免出现“盖子变宽但背墙不变宽”的侧向缝隙。
        wall_back = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": nw.new_node(
                            Nodes.Math,
                            input_kwargs={
                                0: group_input.outputs["Width"],
                                1: edge_ext_x2,
                            },
                            attrs={"operation": "ADD"},
                        ),
                        "Y": group_input.outputs["Thickness"],
                        "Z": group_input.outputs["Height"],
                    },
                ),
            },
        )

        front_wall = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": wall_front,
                # 将盒体整体向 +Y 平移 E/2，使得底板在 -Y 方向形成“前侧凸出”
                "Translation": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "Y": nw.new_node(
                            Nodes.Math,
                            input_kwargs={0: neg_half_depth, 1: half_edge_ext},
                            attrs={"operation": "ADD"},
                        )
                    },
                ),
            },
        )
        back_wall = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": wall_back,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "Y": nw.new_node(
                            Nodes.Math,
                            input_kwargs={0: half_depth, 1: half_edge_ext},
                            attrs={"operation": "ADD"},
                        )
                    },
                ),
            },
        )

        # 左右墙：厚 x 深 x 高
        wall_lr = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": group_input.outputs["Thickness"],
                        "Y": group_input.outputs["Depth"],
                        "Z": group_input.outputs["Height"],
                    },
                ),
            },
        )
        left_wall = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": wall_lr,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={"X": neg_half_width, "Y": half_edge_ext},
                ),
            },
        )
        right_wall = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": wall_lr,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={"X": half_width, "Y": half_edge_ext},
                ),
            },
        )

        # 底板：宽 x 深 x 厚（放在盒体内部底面，保持 bbox center 仍为 0，便于关节位置对齐）
        bottom = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        # 底板三侧凸出：
                        # - X 左右各 +EdgeExtension（总宽 W + 2E）
                        # - Y 仅“前侧” +EdgeExtension（总深 D + E，并整体向 -Y 平移 E/2，使背面边缘仍对齐 y=+D/2）
                        "X": nw.new_node(
                            Nodes.Math,
                            input_kwargs={
                                0: group_input.outputs["Width"],
                                1: edge_ext_x2,
                            },
                            attrs={"operation": "ADD"},
                        ),
                        "Y": nw.new_node(
                            Nodes.Math,
                            input_kwargs={
                                0: group_input.outputs["Depth"],
                                1: group_input.outputs["EdgeExtension"],
                            },
                            attrs={"operation": "ADD"},
                        ),
                        "Z": group_input.outputs["Thickness"],
                    },
                )
            },
        )
        bottom_z = nw.new_node(
            Nodes.Math,
            input_kwargs={0: neg_half_height, 1: half_thickness},
            attrs={"operation": "ADD"},
        )
        bottom = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": bottom,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        # 底板保持居中（不平移），配合墙体整体 +E/2 形成“前侧凸出”
                        "Z": bottom_z,
                    },
                ),
            },
        )

        body = nw.new_node(
            Nodes.JoinGeometry,
            input_kwargs={
                "Geometry": [front_wall, back_wall, left_wall, right_wall, bottom]
            },
        )

        # --- 大折页 lid（竖直展开态；绕背面上沿铰链折下盖住盒口） ---
        # lid 的建模规则：将 hinge 放在局部原点，面板沿 +Z 伸出
        lid_panel = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        # lid 左右凸出，与底板凸出长度同步：W + 2E
                        "X": nw.new_node(
                            Nodes.Math,
                            input_kwargs={
                                0: group_input.outputs["Width"],
                                1: edge_ext_x2,
                            },
                            attrs={"operation": "ADD"},
                        ),
                        "Y": group_input.outputs["Thickness"],
                        "Z": group_input.outputs["Depth"],
                    },
                )
            },
        )
        lid_panel = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": lid_panel,
                # Y=+half_thickness: 厚度向盒体外侧偏置，减少闭合态穿插（并保持打开态的视觉连续性）
                # Z=+half_depth: 让 lid 从 hinge (Z=0) 延伸到 Z=Depth
                "Translation": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={"Y": half_thickness, "Z": half_depth},
                ),
            },
        )

        # --- 小折页 front_flap（连接在 lid 前沿；折下后覆盖盒体高度） ---
        front_flap = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        # 第二折页与 lid 在 X 向完全对齐（同样的左右凸出）
                        "X": nw.new_node(
                            Nodes.Math,
                            input_kwargs={
                                0: group_input.outputs["Width"],
                                1: edge_ext_x2,
                            },
                            attrs={"operation": "ADD"},
                        ),
                        "Y": group_input.outputs["Thickness"],
                        # 第二折页长度（沿 Z）：默认等于 Height；需求 5 可随机缩短
                        "Z": group_input.outputs["FrontFlapLen"],
                    },
                )
            },
        )
        front_flap_half = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["FrontFlapLen"], 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )
        front_flap = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": front_flap,
                # Y=+half_thickness: 厚度向盒体外侧偏置，减少闭合态穿插
                # Z=+front_flap_half: flap 从 hinge (Z=0) 延伸到 Z=FrontFlapLen
                "Translation": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={"Y": half_thickness, "Z": front_flap_half},
                ),
            },
        )

        # 先把 front_flap 铰接到 lid（形成 lid 子结构：lid -> front_flap）
        # NOTE: 参考 TuckEndBox 的 “二段折叠”写法，这里用 half_depth 作为局部位置输入，
        # 让 hinge 落在 lid 前沿（Depth 末端）并保持稳定的导出坐标系。
        front_hinge_pos = nw.new_node(
            Nodes.CombineXYZ,
            input_kwargs={"Y": half_thickness, "Z": half_depth},
        )
        j_front = nw.new_node(
            nodegroup_hinge_joint().name,
            input_kwargs={
                "Joint Label": "mailer_front_flap",
                "Parent": lid_panel,
                "Child": front_flap,
                "Position": front_hinge_pos,
                "Axis": (1, 0, 0),
                "Min": -np.pi,
                "Max": np.pi,
            },
        )
        lid_with_flap = j_front.outputs["Geometry"]

        # 再把 lid 结构铰接到 body（body -> lid）
        hinge_lid_pos = nw.new_node(
            Nodes.CombineXYZ,
            input_kwargs={
                # hinge 放在“背面外侧折痕”位置，确保盖子与盒体无缝连接、不出现可见缝隙
                # 说明：`nodegroup_hinge_joint` / exporter 的 joint-origin 计算会对 Position 存在一个
                # 约 ~T/4 的有效偏移（与 child AABB center/poschild 的定义有关）。
                # 为使最终 joint origin 落在 back wall 的外侧面（center + T/2），这里使用：
                #   back wall center + (T/4)
                "Y": nw.new_node(
                    Nodes.Math,
                    input_kwargs={
                        0: nw.new_node(
                            Nodes.Math,
                            input_kwargs={0: half_depth, 1: half_edge_ext},
                            attrs={"operation": "ADD"},
                        ),
                        1: nw.new_node(
                            Nodes.Math,
                            input_kwargs={0: half_thickness, 1: 2.0},
                            attrs={"operation": "DIVIDE"},
                        ),
                    },
                    attrs={"operation": "ADD"},
                ),
                "Z": half_height,
            },
        )
        j_lid = nw.new_node(
            nodegroup_hinge_joint().name,
            input_kwargs={
                "Joint Label": "mailer_lid",
                "Parent": body,
                "Child": lid_with_flap,
                "Position": hinge_lid_pos,
                "Axis": (1, 0, 0),
                "Min": -np.pi,
                "Max": np.pi,
            },
        )

        nw.new_node(
            Nodes.GroupOutput,
            input_kwargs={"Geometry": j_lid.outputs["Geometry"]},
            attrs={"is_active_output": True},
        )


# ============================================================
# 盒型实现: DrawerBox（抽屉盒，1-DOF prismatic/slide）
# ============================================================


@ModularBoxFactory.register(BoxType.DRAWER)
class DrawerBoxFactory(ModularBoxFactory):
    """
    抽屉盒 (DrawerBox) 工厂（Phase 1.1 最小实现）

    目标：
    - outer_box (link_0) 作为固定外壳
    - drawer (link_1) 通过滑轨关节沿 -Y 方向拉出

    Notes (科学/工程折衷)：
    - Phase-1 重点在于 **URDF/惯性/碰撞/标注闭环**，几何采用“薄壁面板拼接”的简化版本，
      保留抽屉的 prismatic DOF 以服务下游感知/规划研究。
    - 后续可逐步细化面板拓扑与锁扣结构（不影响 URDF 导出与标注协议）。
    """

    def get_box_type(self) -> BoxType:
        return BoxType.DRAWER

    def sample_dimensions(self) -> BoxDimensions:
        # 抽屉盒通常更“扁平”，高度相对较小
        return BoxDimensions(
            width=uniform(0.12, 0.35),
            depth=uniform(0.12, 0.35),
            height=uniform(0.06, 0.18),
            thickness=uniform(0.0008, 0.0025),
        )

    def get_default_joints(self, params: BoxParameters) -> List[BoxJointConfig]:
        dims = params.dimensions
        travel = float(0.8 * dims.depth)
        return [
            BoxJointConfig(
                joint_type="slide",
                label="drawer_slide",
                position=(0.0, 0.0, 0.0),
                axis=(0.0, -1.0, 0.0),  # pull forward
                min_angle=0.0,
                max_angle=travel,  # meters
                damping=0.6,
                friction=0.25,
            )
        ]

    def create_geometry_nodegroup(self, nw: NodeWrangler, params: BoxParameters):
        dims = params.dimensions

        # Clearance is to avoid initial overlaps between drawer and outer shell.
        default_clearance = float(
            params.extra_params.get("Clearance", 3.0 * dims.thickness)
        )

        group_input = nw.new_node(
            Nodes.GroupInput,
            expose_input=[
                ("NodeSocketFloat", "Width", dims.width),
                ("NodeSocketFloat", "Depth", dims.depth),
                ("NodeSocketFloat", "Height", dims.height),
                ("NodeSocketFloat", "Thickness", dims.thickness),
                ("NodeSocketFloat", "Clearance", default_clearance),
            ],
        )

        half_w = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Width"], 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )
        half_d = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Depth"], 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )
        half_h = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Height"], 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )
        half_t = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )

        neg_half_h = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_h, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )
        neg_half_d = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_d, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )
        neg_half_w = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_w, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )

        # ----------------------------
        # Outer shell: bottom + 3 walls (no front wall, open top)
        # ----------------------------
        bottom = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": group_input.outputs["Width"],
                        "Y": group_input.outputs["Depth"],
                        "Z": group_input.outputs["Thickness"],
                    },
                )
            },
        )
        bottom_z = nw.new_node(
            Nodes.Math,
            input_kwargs={0: neg_half_h, 1: half_t},
            attrs={"operation": "ADD"},
        )
        bottom = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": bottom,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"Z": bottom_z}
                ),
            },
        )

        wall_y = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_d, 1: half_t},
            attrs={"operation": "SUBTRACT"},
        )
        wall_x = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_w, 1: half_t},
            attrs={"operation": "SUBTRACT"},
        )
        neg_wall_x = nw.new_node(
            Nodes.Math,
            input_kwargs={0: wall_x, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )

        back = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": group_input.outputs["Width"],
                        "Y": group_input.outputs["Thickness"],
                        "Z": group_input.outputs["Height"],
                    },
                )
            },
        )
        back = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": back,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"Y": wall_y}
                ),
            },
        )

        left = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": group_input.outputs["Thickness"],
                        "Y": group_input.outputs["Depth"],
                        "Z": group_input.outputs["Height"],
                    },
                )
            },
        )
        left = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": left,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"X": neg_wall_x}
                ),
            },
        )

        right = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": group_input.outputs["Thickness"],
                        "Y": group_input.outputs["Depth"],
                        "Z": group_input.outputs["Height"],
                    },
                )
            },
        )
        right = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": right,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"X": wall_x}
                ),
            },
        )

        outer_shell = nw.new_node(
            Nodes.JoinGeometry, input_kwargs={"Geometry": [bottom, back, left, right]}
        )

        # ----------------------------
        # Drawer: thin-walled panels (open top)
        #   - keeps mass/inertia in the correct "thin-shell" regime
        #   - avoids unrealistic "solid block" inertia for cardboard boxes
        # ----------------------------
        drawer_w = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: group_input.outputs["Width"],
                1: nw.new_node(
                    Nodes.Math,
                    input_kwargs={0: group_input.outputs["Clearance"], 1: 2.0},
                    attrs={"operation": "MULTIPLY"},
                ),
            },
            attrs={"operation": "SUBTRACT"},
        )
        drawer_h = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: group_input.outputs["Height"],
                1: group_input.outputs["Clearance"],
            },
            attrs={"operation": "SUBTRACT"},
        )
        drawer_d = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Depth"], 1: 0.80},
            attrs={"operation": "MULTIPLY"},
        )

        drawer_half_w = nw.new_node(
            Nodes.Math,
            input_kwargs={0: drawer_w, 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )
        drawer_half_d = nw.new_node(
            Nodes.Math,
            input_kwargs={0: drawer_d, 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )
        drawer_half_h = nw.new_node(
            Nodes.Math,
            input_kwargs={0: drawer_h, 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )
        drawer_neg_half_h = nw.new_node(
            Nodes.Math,
            input_kwargs={0: drawer_half_h, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )

        drawer_bottom = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": drawer_w,
                        "Y": drawer_d,
                        "Z": group_input.outputs["Thickness"],
                    },
                )
            },
        )
        drawer_bottom_z = nw.new_node(
            Nodes.Math,
            input_kwargs={0: drawer_neg_half_h, 1: half_t},
            attrs={"operation": "ADD"},
        )
        drawer_bottom = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": drawer_bottom,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"Z": drawer_bottom_z}
                ),
            },
        )

        drawer_wall_y = nw.new_node(
            Nodes.Math,
            input_kwargs={0: drawer_half_d, 1: half_t},
            attrs={"operation": "SUBTRACT"},
        )
        drawer_neg_wall_y = nw.new_node(
            Nodes.Math,
            input_kwargs={0: drawer_wall_y, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )
        drawer_wall_x = nw.new_node(
            Nodes.Math,
            input_kwargs={0: drawer_half_w, 1: half_t},
            attrs={"operation": "SUBTRACT"},
        )
        drawer_neg_wall_x = nw.new_node(
            Nodes.Math,
            input_kwargs={0: drawer_wall_x, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )

        drawer_back = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": drawer_w,
                        "Y": group_input.outputs["Thickness"],
                        "Z": drawer_h,
                    },
                )
            },
        )
        drawer_back = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": drawer_back,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"Y": drawer_wall_y}
                ),
            },
        )

        drawer_front = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": drawer_w,
                        "Y": group_input.outputs["Thickness"],
                        "Z": drawer_h,
                    },
                )
            },
        )
        drawer_front = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": drawer_front,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"Y": drawer_neg_wall_y}
                ),
            },
        )

        drawer_left = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": group_input.outputs["Thickness"],
                        "Y": drawer_d,
                        "Z": drawer_h,
                    },
                )
            },
        )
        drawer_left = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": drawer_left,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"X": drawer_neg_wall_x}
                ),
            },
        )

        drawer_right = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": group_input.outputs["Thickness"],
                        "Y": drawer_d,
                        "Z": drawer_h,
                    },
                )
            },
        )
        drawer_right = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": drawer_right,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"X": drawer_wall_x}
                ),
            },
        )

        drawer = nw.new_node(
            Nodes.JoinGeometry,
            input_kwargs={
                "Geometry": [
                    drawer_bottom,
                    drawer_back,
                    drawer_front,
                    drawer_left,
                    drawer_right,
                ]
            },
        )
        # Put drawer near the opening side (-Y) at value=0; slide axis will move it further outward.
        drawer_y0 = nw.new_node(
            Nodes.Math,
            input_kwargs={0: neg_half_d, 1: drawer_half_d},
            attrs={"operation": "ADD"},
        )
        drawer = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": drawer,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"Y": drawer_y0}
                ),
            },
        )

        # Sliding joint: outer_shell -> drawer
        travel = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Depth"], 1: 0.80},
            attrs={"operation": "MULTIPLY"},
        )
        j_slide = nw.new_node(
            nodegroup_sliding_joint().name,
            input_kwargs={
                "Joint Label": "drawer_slide",
                "Parent": outer_shell,
                "Child": drawer,
                "Position": (0.0, 0.0, 0.0),
                "Axis": (0.0, -1.0, 0.0),
                "Min": 0.0,
                "Max": travel,
            },
        )

        nw.new_node(
            Nodes.GroupOutput,
            input_kwargs={"Geometry": j_slide.outputs["Geometry"]},
            attrs={"is_active_output": True},
        )


# ============================================================
# 盒型实现: SlipLidBox（天地盒，0-1 DOF prismatic/slide）
# ============================================================


@ModularBoxFactory.register(BoxType.SLIP_LID)
class SlipLidBoxFactory(ModularBoxFactory):
    """
    天地盒 (SlipLidBox) 工厂（Phase 1.1 最小实现）

    结构：
    - base (link_0): 下盒
    - lid (link_1): 上盖（可沿 +Z 方向滑动抬起）
    """

    def get_box_type(self) -> BoxType:
        return BoxType.SLIP_LID

    def sample_dimensions(self) -> BoxDimensions:
        return BoxDimensions(
            width=uniform(0.10, 0.35),
            depth=uniform(0.10, 0.35),
            height=uniform(0.06, 0.20),
            thickness=uniform(0.0008, 0.0025),
        )

    def get_default_joints(self, params: BoxParameters) -> List[BoxJointConfig]:
        dims = params.dimensions
        travel = float(0.6 * dims.height)
        return [
            BoxJointConfig(
                joint_type="slide",
                label="slip_lid_slide",
                position=(0.0, 0.0, 0.0),
                axis=(0.0, 0.0, 1.0),  # lift the lid
                min_angle=0.0,
                max_angle=travel,
                damping=0.4,
                friction=0.2,
            )
        ]

    def create_geometry_nodegroup(self, nw: NodeWrangler, params: BoxParameters):
        dims = params.dimensions
        default_clearance = float(
            params.extra_params.get("Clearance", 3.0 * dims.thickness)
        )

        group_input = nw.new_node(
            Nodes.GroupInput,
            expose_input=[
                ("NodeSocketFloat", "Width", dims.width),
                ("NodeSocketFloat", "Depth", dims.depth),
                ("NodeSocketFloat", "Height", dims.height),
                ("NodeSocketFloat", "Thickness", dims.thickness),
                ("NodeSocketFloat", "Clearance", default_clearance),
            ],
        )

        half_w = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Width"], 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )
        half_d = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Depth"], 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )
        half_h = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Height"], 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )
        half_t = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0},
            attrs={"operation": "DIVIDE"},
        )

        neg_half_h = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_h, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )
        neg_half_d = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_d, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )
        neg_half_w = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_w, 1: -1.0},
            attrs={"operation": "MULTIPLY"},
        )

        # ----------------------------
        # Base: bottom + 4 walls (open top)
        # ----------------------------
        bottom = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": group_input.outputs["Width"],
                        "Y": group_input.outputs["Depth"],
                        "Z": group_input.outputs["Thickness"],
                    },
                )
            },
        )
        bottom_z = nw.new_node(
            Nodes.Math,
            input_kwargs={0: neg_half_h, 1: half_t},
            attrs={"operation": "ADD"},
        )
        bottom = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": bottom,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"Z": bottom_z}
                ),
            },
        )

        wall_y = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_d, 1: half_t},
            attrs={"operation": "SUBTRACT"},
        )
        wall_x = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_w, 1: half_t},
            attrs={"operation": "SUBTRACT"},
        )

        front = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": group_input.outputs["Width"],
                        "Y": group_input.outputs["Thickness"],
                        "Z": group_input.outputs["Height"],
                    },
                )
            },
        )
        front = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": front,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"Y": neg_half_d}
                ),
            },
        )

        back = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": group_input.outputs["Width"],
                        "Y": group_input.outputs["Thickness"],
                        "Z": group_input.outputs["Height"],
                    },
                )
            },
        )
        back = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": back,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"Y": wall_y}
                ),
            },
        )

        left = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": group_input.outputs["Thickness"],
                        "Y": group_input.outputs["Depth"],
                        "Z": group_input.outputs["Height"],
                    },
                )
            },
        )
        left = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": left,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"X": neg_half_w}
                ),
            },
        )

        right = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": group_input.outputs["Thickness"],
                        "Y": group_input.outputs["Depth"],
                        "Z": group_input.outputs["Height"],
                    },
                )
            },
        )
        right = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": right,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"X": wall_x}
                ),
            },
        )

        base = nw.new_node(
            Nodes.JoinGeometry,
            input_kwargs={"Geometry": [bottom, front, back, left, right]},
        )

        # ----------------------------
        # Lid: top + 4 walls (no bottom), slightly larger in XY
        # ----------------------------
        lid_w = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: group_input.outputs["Width"],
                1: group_input.outputs["Clearance"],
            },
            attrs={"operation": "ADD"},
        )
        lid_d = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: group_input.outputs["Depth"],
                1: group_input.outputs["Clearance"],
            },
            attrs={"operation": "ADD"},
        )
        lid_h = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Height"], 1: 0.60},
            attrs={"operation": "MULTIPLY"},
        )

        top = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": lid_w,
                        "Y": lid_d,
                        "Z": group_input.outputs["Thickness"],
                    },
                )
            },
        )
        # place lid "closed" on top of base at value=0: top panel near +half_h
        top_z = nw.new_node(
            Nodes.Math,
            input_kwargs={0: half_h, 1: neg_half_h},
            attrs={"operation": "ADD"},
        )
        top_z = nw.new_node(
            Nodes.Math, input_kwargs={0: half_h, 1: half_t}, attrs={"operation": "ADD"}
        )
        top = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": top,
                "Translation": nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": top_z}),
            },
        )

        lid_wall_y = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: nw.new_node(
                    Nodes.Math,
                    input_kwargs={0: lid_d, 1: 2.0},
                    attrs={"operation": "DIVIDE"},
                ),
                1: half_t,
            },
            attrs={"operation": "SUBTRACT"},
        )
        lid_wall_x = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: nw.new_node(
                    Nodes.Math,
                    input_kwargs={0: lid_w, 1: 2.0},
                    attrs={"operation": "DIVIDE"},
                ),
                1: half_t,
            },
            attrs={"operation": "SUBTRACT"},
        )
        lid_neg_half_d = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: nw.new_node(
                    Nodes.Math,
                    input_kwargs={0: lid_d, 1: 2.0},
                    attrs={"operation": "DIVIDE"},
                ),
                1: -1.0,
            },
            attrs={"operation": "MULTIPLY"},
        )
        lid_neg_half_w = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: nw.new_node(
                    Nodes.Math,
                    input_kwargs={0: lid_w, 1: 2.0},
                    attrs={"operation": "DIVIDE"},
                ),
                1: -1.0,
            },
            attrs={"operation": "MULTIPLY"},
        )

        lid_front = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": lid_w,
                        "Y": group_input.outputs["Thickness"],
                        "Z": lid_h,
                    },
                )
            },
        )
        lid_front = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": lid_front,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"Y": lid_neg_half_d, "Z": top_z}
                ),
            },
        )

        lid_back = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": lid_w,
                        "Y": group_input.outputs["Thickness"],
                        "Z": lid_h,
                    },
                )
            },
        )
        lid_back = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": lid_back,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"Y": lid_wall_y, "Z": top_z}
                ),
            },
        )

        lid_left = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": group_input.outputs["Thickness"],
                        "Y": lid_d,
                        "Z": lid_h,
                    },
                )
            },
        )
        lid_left = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": lid_left,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"X": lid_neg_half_w, "Z": top_z}
                ),
            },
        )

        lid_right = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": group_input.outputs["Thickness"],
                        "Y": lid_d,
                        "Z": lid_h,
                    },
                )
            },
        )
        lid_right = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": lid_right,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ, input_kwargs={"X": lid_wall_x, "Z": top_z}
                ),
            },
        )

        lid = nw.new_node(
            Nodes.JoinGeometry,
            input_kwargs={"Geometry": [top, lid_front, lid_back, lid_left, lid_right]},
        )

        travel = nw.new_node(
            Nodes.Math,
            input_kwargs={0: group_input.outputs["Height"], 1: 0.60},
            attrs={"operation": "MULTIPLY"},
        )
        j_slide = nw.new_node(
            nodegroup_sliding_joint().name,
            input_kwargs={
                "Joint Label": "slip_lid_slide",
                "Parent": base,
                "Child": lid,
                "Position": (0.0, 0.0, 0.0),
                "Axis": (0.0, 0.0, 1.0),
                "Min": 0.0,
                "Max": travel,
            },
        )

        nw.new_node(
            Nodes.GroupOutput,
            input_kwargs={"Geometry": j_slide.outputs["Geometry"]},
            attrs={"is_active_output": True},
        )


# ============================================================
# 工具函数
# ============================================================


def create_box(box_type: BoxType, seed: int = None, **kwargs) -> bpy.types.Object:
    """
    便捷函数: 创建指定类型的盒子

    Args:
        box_type: 盒型枚举
        seed: 随机种子
        **kwargs: 传递给 create_asset 的参数

    Returns:
        Blender 对象
    """
    factory_cls = ModularBoxFactory.get_factory(box_type)
    factory = factory_cls(factory_seed=seed)
    return factory.create_asset(**kwargs)


def list_available_box_types() -> List[str]:
    """列出所有可用的盒型"""
    return [bt.name for bt in ModularBoxFactory.list_registered_types()]
