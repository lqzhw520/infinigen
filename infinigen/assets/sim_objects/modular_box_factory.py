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
from typing import Callable, Dict, List, Optional, Tuple, Type
from enum import Enum

import bpy
import gin
import numpy as np
from numpy.random import uniform, randint

from infinigen.assets.utils.joints import nodegroup_hinge_joint, nodegroup_sliding_joint
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil

logger = logging.getLogger(__name__)


class BoxType(Enum):
    """16 种盒型枚举"""
    TUCK_END = 1               # 双插盒
    LOCK_BOTTOM = 2            # 锁底盒
    TUCK_END_SAFETY = 3        # 带保险双插盒
    LOCK_BOTTOM_SAFETY = 4     # 带保险锁底盒
    HOOK_LOCK_BOTTOM = 5       # 带挂钩锁底盒
    HOOK_TUCK_END = 6          # 带挂钩双插盒
    MAILER = 7                 # 飞机盒
    DRAWER = 8                 # 抽屉盒
    SLIP_LID = 9               # 天地盒
    GIFT_WITH_HANDLE = 10      # 自带手提礼盒
    TUCK_HANDLE = 11           # 对插手提礼盒
    GABLE_TOP = 12             # 屋脊手提礼盒
    DOUBLE_LID_GIFT = 13       # 双盖手提礼盒
    PLASTIC_HANDLE = 14        # 塑料手提礼盒
    RSC = 15                   # 平口箱 (Regular Slotted Container)
    CUSTOM = 16                # 自定义盒型


@dataclass
class BoxDimensions:
    """盒子尺寸参数"""
    width: float = 0.2       # X 方向 (米)
    depth: float = 0.2       # Y 方向 (米)
    height: float = 0.15     # Z 方向 (米)
    thickness: float = 0.002 # 壁厚 (米)
    
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
    material_type: str = "cardboard"  # cardboard, corrugated, plastic, wood
    density: float = 300.0  # kg/m³ (P2.1-T2: 修正为 cardboard 中值)
    color: Tuple[float, float, float] = (0.6, 0.5, 0.4)  # RGB


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
        material_types = ["cardboard", "corrugated"]
        selected = material_types[randint(0, len(material_types))]
        
        # P2.1-T2 修复: 密度值对齐 material_definitions.py
        density_map = {
            "cardboard": uniform(200, 400),    # 卡纸: 200-400 kg/m³
            "corrugated": uniform(100, 300),   # 瓦楞纸: 100-300 kg/m³
            "plastic": uniform(900, 1400),     # 塑料: 900-1400 kg/m³
            "wood": uniform(500, 900),         # 木材: 500-900 kg/m³
        }
        
        return BoxMaterialConfig(
            material_type=selected,
            density=density_map.get(selected, 300.0),  # 默认使用 cardboard 中值
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
        # 创建或获取材质
        mat_name = f"shader_{material_config.material_type}"
        
        if mat_name in bpy.data.materials:
            mat = bpy.data.materials[mat_name]
        else:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            
            # 设置基础颜色
            if mat.node_tree:
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf:
                    bsdf.inputs["Base Color"].default_value = (*material_config.color, 1.0)
        
        # 确保对象有 mesh 数据后再应用材质
        # 注意: 在几何节点应用前，对象可能只是顶点
        # 材质会在 URDF 导出时传递给生成的 mesh
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
    
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
        
        return [
            BoxJointConfig(
                joint_type="hinge",
                label="top_lid",
                position=(0, dims.depth / 2, dims.height / 2),
                axis=(1, 0, 0),
                min_angle=0,
                max_angle=np.pi * 0.9,  # 最大 162°
                damping=0.5,
            ),
            BoxJointConfig(
                joint_type="hinge",
                label="bottom_lid",
                position=(0, -dims.depth / 2, -dims.height / 2),
                axis=(1, 0, 0),
                min_angle=-np.pi * 0.9,
                max_angle=0,
                damping=0.5,
            ),
        ]
    
    def create_geometry_nodegroup(self, nw: NodeWrangler, params: BoxParameters):
        """创建双插盒的几何节点组"""
        dims = params.dimensions
        
        # 创建输入节点
        group_input = nw.new_node(
            Nodes.GroupInput,
            expose_input=[
                ("NodeSocketFloat", "Width", dims.width),
                ("NodeSocketFloat", "Depth", dims.depth),
                ("NodeSocketFloat", "Height", dims.height),
                ("NodeSocketFloat", "Thickness", dims.thickness),
            ],
        )
        
        # 创建主体盒子 (使用 Cube 作为基础)
        cube = nw.new_node(
            Nodes.MeshCube,
            input_kwargs={
                "Size": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={
                        "X": group_input.outputs["Width"],
                        "Y": group_input.outputs["Depth"],
                        "Z": group_input.outputs["Height"],
                    },
                ),
            },
        )
        
        # 创建顶盖
        top_lid = self._create_lid(nw, group_input, is_top=True)
        
        # 创建底盖
        bottom_lid = self._create_lid(nw, group_input, is_top=False)
        
        # 创建铰链关节 (顶盖)
        top_hinge = nw.new_node(
            nodegroup_hinge_joint().name,
            input_kwargs={
                "Joint Label": "top_lid",
                "Parent": cube,
                "Child": top_lid,
                "Position": (0, dims.depth / 2, dims.height / 2),
                "Axis": (1, 0, 0),
                "Min": 0,
                "Max": np.pi * 0.9,
            },
        )
        
        # 创建铰链关节 (底盖)
        bottom_hinge = nw.new_node(
            nodegroup_hinge_joint().name,
            input_kwargs={
                "Joint Label": "bottom_lid",
                "Parent": top_hinge,  # 链接到上一个关节输出
                "Child": bottom_lid,
                "Position": (0, -dims.depth / 2, -dims.height / 2),
                "Axis": (1, 0, 0),
                "Min": -np.pi * 0.9,
                "Max": 0,
            },
        )
        
        # 输出
        nw.new_node(
            Nodes.GroupOutput,
            input_kwargs={"Geometry": bottom_hinge},
            attrs={"is_active_output": True},
        )
    
    def _create_lid(self, nw: NodeWrangler, group_input, is_top: bool):
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
        
        # 移动到正确位置
        offset_z = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: group_input.outputs["Height"],
                1: 2.0 if is_top else -2.0,
            },
            attrs={"operation": "DIVIDE"},
        )
        
        translated = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": lid,
                "Translation": nw.new_node(
                    Nodes.CombineXYZ,
                    input_kwargs={"Z": offset_z},
                ),
            },
        )
        
        return translated


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
