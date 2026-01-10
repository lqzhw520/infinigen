# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""
P1-T3: 关节注入系统 (joint_injector.py)

将铰链/滑轨关节附加到面板边缘，用于创建可动盒子资产。
集成 Infinigen 现有的 nodegroup_hinge_joint 和 nodegroup_sliding_joint。

主要功能:
- add_hinge_at_edge(): 在边缘添加铰链
- add_slide_joint(): 添加滑轨关节
- validate_joint_range(): 验证关节范围
- calculate_safe_joint_range(): 计算安全关节范围

依赖:
- P1-T2: box_geometry_modules (EdgeInfo, PanelGeometry)
- infinigen.assets.utils.joints
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np


class JointType(Enum):
    """关节类型枚举"""
    HINGE = "hinge"      # 铰链 (旋转)
    SLIDE = "slide"      # 滑轨 (平移)
    FIXED = "fixed"      # 固定 (无运动)


class MaterialType(Enum):
    """材料类型枚举 (用于关节动力学参数)"""
    CARDBOARD = "cardboard"
    CORRUGATED = "corrugated"
    PLASTIC = "plastic"
    WOOD = "wood"
    METAL = "metal"


@dataclass
class JointConfig:
    """
    关节配置数据类
    
    Attributes:
        label: 关节标识符
        joint_type: 关节类型
        position: 关节位置 (x, y, z)
        axis: 旋转轴或滑动方向 (ax, ay, az)
        min_value: 最小值 (角度弧度或位移米)
        max_value: 最大值
        initial_value: 初始值
        parent_link: 父链接名称
        child_link: 子链接名称
    """
    label: str
    joint_type: JointType
    position: Tuple[float, float, float]
    axis: Tuple[float, float, float]
    min_value: float = 0.0
    max_value: float = np.pi
    initial_value: float = 0.0
    parent_link: str = "base"
    child_link: str = ""
    
    def validate(self) -> Tuple[bool, str]:
        """验证关节配置有效性"""
        errors = []
        
        if not self.label:
            errors.append("关节标签不能为空")
        
        if self.min_value > self.max_value:
            errors.append(f"最小值 ({self.min_value}) > 最大值 ({self.max_value})")
        
        axis_norm = np.linalg.norm(self.axis)
        if axis_norm < 0.9 or axis_norm > 1.1:
            errors.append(f"轴向量应为单位向量，当前长度: {axis_norm}")
        
        if self.initial_value < self.min_value or self.initial_value > self.max_value:
            errors.append("初始值超出范围")
        
        if errors:
            return False, "; ".join(errors)
        return True, "配置有效"
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "label": self.label,
            "joint_type": self.joint_type.value,
            "position": list(self.position),
            "axis": list(self.axis),
            "min_value": self.min_value,
            "max_value": self.max_value,
            "initial_value": self.initial_value,
            "parent_link": self.parent_link,
            "child_link": self.child_link,
        }


@dataclass
class JointDynamics:
    """
    关节动力学参数
    
    Attributes:
        damping: 阻尼系数 (Ns/m 或 Nms/rad)
        friction: 摩擦系数
        stiffness: 刚度系数 (N/m 或 Nm/rad)
        velocity_limit: 速度限制
        effort_limit: 力矩/力限制
    """
    damping: float = 0.5
    friction: float = 0.1
    stiffness: float = 0.0
    velocity_limit: float = 10.0
    effort_limit: float = 100.0
    
    def to_dict(self) -> Dict:
        return {
            "damping": self.damping,
            "friction": self.friction,
            "stiffness": self.stiffness,
            "velocity_limit": self.velocity_limit,
            "effort_limit": self.effort_limit,
        }


# ============================================================
# 材料相关关节动力学预设 (R8)
# ============================================================

JOINT_DYNAMICS_PRESETS: Dict[MaterialType, Dict[JointType, JointDynamics]] = {
    MaterialType.CARDBOARD: {
        JointType.HINGE: JointDynamics(
            damping=0.3,
            friction=0.15,
            stiffness=0.0,  # 卡纸无弹性恢复
            velocity_limit=5.0,
            effort_limit=2.0,  # 卡纸较弱
        ),
        JointType.SLIDE: JointDynamics(
            damping=0.5,
            friction=0.3,  # 滑动摩擦较大
            stiffness=0.0,
            velocity_limit=2.0,
            effort_limit=1.0,
        ),
    },
    MaterialType.CORRUGATED: {
        JointType.HINGE: JointDynamics(
            damping=0.5,
            friction=0.2,
            stiffness=0.1,  # 瓦楞纸有轻微弹性
            velocity_limit=4.0,
            effort_limit=5.0,  # 瓦楞纸较强
        ),
        JointType.SLIDE: JointDynamics(
            damping=0.6,
            friction=0.25,
            stiffness=0.0,
            velocity_limit=1.5,
            effort_limit=3.0,
        ),
    },
    MaterialType.PLASTIC: {
        JointType.HINGE: JointDynamics(
            damping=0.2,
            friction=0.1,
            stiffness=0.5,  # 塑料有弹性
            velocity_limit=8.0,
            effort_limit=10.0,
        ),
        JointType.SLIDE: JointDynamics(
            damping=0.3,
            friction=0.1,  # 塑料滑动摩擦小
            stiffness=0.0,
            velocity_limit=5.0,
            effort_limit=8.0,
        ),
    },
    MaterialType.WOOD: {
        JointType.HINGE: JointDynamics(
            damping=0.8,
            friction=0.3,
            stiffness=0.0,  # 木质无弹性
            velocity_limit=3.0,
            effort_limit=50.0,  # 木质较强
        ),
        JointType.SLIDE: JointDynamics(
            damping=1.0,
            friction=0.4,
            stiffness=0.0,
            velocity_limit=2.0,
            effort_limit=40.0,
        ),
    },
    MaterialType.METAL: {
        JointType.HINGE: JointDynamics(
            damping=0.1,
            friction=0.05,
            stiffness=1.0,
            velocity_limit=15.0,
            effort_limit=200.0,
        ),
        JointType.SLIDE: JointDynamics(
            damping=0.2,
            friction=0.05,
            stiffness=0.0,
            velocity_limit=10.0,
            effort_limit=150.0,
        ),
    },
}


def get_joint_dynamics(
    material: MaterialType,
    joint_type: JointType,
) -> JointDynamics:
    """
    获取材料和关节类型对应的动力学参数
    
    Args:
        material: 材料类型
        joint_type: 关节类型
    
    Returns:
        JointDynamics: 动力学参数
    """
    if material in JOINT_DYNAMICS_PRESETS:
        presets = JOINT_DYNAMICS_PRESETS[material]
        if joint_type in presets:
            return presets[joint_type]
    
    # 默认返回卡纸铰链参数
    return JOINT_DYNAMICS_PRESETS[MaterialType.CARDBOARD][JointType.HINGE]


# ============================================================
# 关节注入函数
# ============================================================

def add_hinge_at_edge(
    parent_geometry: "PanelGeometry",
    child_geometry: "PanelGeometry",
    edge_info: "EdgeInfo",
    label: str,
    min_angle: float = 0.0,
    max_angle: float = np.pi,
    initial_angle: float = 0.0,
    material: MaterialType = MaterialType.CARDBOARD,
) -> JointConfig:
    """
    在边缘添加铰链关节
    
    Args:
        parent_geometry: 父面板几何
        child_geometry: 子面板几何
        edge_info: 边缘信息
        label: 关节标签
        min_angle: 最小旋转角度 (弧度)
        max_angle: 最大旋转角度 (弧度)
        initial_angle: 初始角度
        material: 材料类型
    
    Returns:
        JointConfig: 铰链关节配置
    """
    position = edge_info.position
    axis = edge_info.get_hinge_axis()
    
    # 验证角度范围
    min_angle = max(0.0, min_angle)
    max_angle = min(np.pi * 2, max_angle)
    
    if max_angle <= min_angle:
        max_angle = min_angle + np.pi / 2  # 默认 90 度范围
    
    # 获取动力学参数 (用于后续 URDF 生成)
    dynamics = get_joint_dynamics(material, JointType.HINGE)
    
    return JointConfig(
        label=label,
        joint_type=JointType.HINGE,
        position=position,
        axis=axis,
        min_value=min_angle,
        max_value=max_angle,
        initial_value=initial_angle,
        parent_link=f"{parent_geometry.panel_type.value}_link",
        child_link=f"{child_geometry.panel_type.value}_link",
    )


def add_slide_joint(
    parent_geometry: "PanelGeometry",
    child_geometry: "PanelGeometry",
    slide_direction: Tuple[float, float, float],
    label: str,
    min_distance: float = 0.0,
    max_distance: float = 0.1,
    initial_distance: float = 0.0,
    material: MaterialType = MaterialType.CARDBOARD,
) -> JointConfig:
    """
    添加滑轨关节
    
    Args:
        parent_geometry: 父面板几何
        child_geometry: 子面板几何
        slide_direction: 滑动方向 (单位向量)
        label: 关节标签
        min_distance: 最小滑动距离 (米)
        max_distance: 最大滑动距离 (米)
        initial_distance: 初始位置
        material: 材料类型
    
    Returns:
        JointConfig: 滑轨关节配置
    """
    # 归一化滑动方向
    direction = np.array(slide_direction)
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    else:
        direction = np.array([1, 0, 0])
    
    # 关节位置在子几何中心
    position = child_geometry.center
    
    return JointConfig(
        label=label,
        joint_type=JointType.SLIDE,
        position=position,
        axis=tuple(direction.tolist()),
        min_value=min_distance,
        max_value=max_distance,
        initial_value=initial_distance,
        parent_link=f"{parent_geometry.panel_type.value}_link",
        child_link=f"{child_geometry.panel_type.value}_link",
    )


def validate_joint_range(
    joint_config: JointConfig,
    parent_vertices: np.ndarray,
    child_vertices: np.ndarray,
    check_collision: bool = True,
    step_count: int = 20,
) -> Tuple[bool, str, Optional[float]]:
    """
    验证关节范围是否有效 (无碰撞)
    
    Args:
        joint_config: 关节配置
        parent_vertices: 父几何顶点
        child_vertices: 子几何顶点
        check_collision: 是否检查碰撞
        step_count: 碰撞检测步数
    
    Returns:
        (is_valid, message, safe_max_value): 验证结果
    """
    # 基本验证
    is_valid, msg = joint_config.validate()
    if not is_valid:
        return False, msg, None
    
    if not check_collision:
        return True, "范围有效（未检查碰撞）", joint_config.max_value
    
    # 碰撞检测
    position = np.array(joint_config.position)
    axis = np.array(joint_config.axis)
    
    safe_max = joint_config.max_value
    
    for step in range(step_count):
        # 计算当前角度/距离
        t = step / (step_count - 1)
        value = joint_config.min_value + t * (joint_config.max_value - joint_config.min_value)
        
        # 变换子几何
        if joint_config.joint_type == JointType.HINGE:
            transformed = _rotate_vertices(child_vertices, position, axis, value)
        else:
            transformed = child_vertices + axis * value
        
        # 检查碰撞
        if _check_collision_simple(parent_vertices, transformed):
            safe_max = value - (joint_config.max_value - joint_config.min_value) / step_count
            return False, f"碰撞发生在值 {value:.3f}", max(joint_config.min_value, safe_max)
    
    return True, "范围有效（无碰撞）", joint_config.max_value


def calculate_safe_joint_range(
    parent_geometry: "PanelGeometry",
    child_geometry: "PanelGeometry",
    edge_info: "EdgeInfo",
    joint_type: JointType = JointType.HINGE,
    safety_margin: float = 0.01,
) -> Tuple[float, float]:
    """
    计算安全的关节范围
    
    使用二分搜索找到无碰撞的最大角度/距离。
    
    Args:
        parent_geometry: 父几何
        child_geometry: 子几何
        edge_info: 边缘信息
        joint_type: 关节类型
        safety_margin: 安全余量 (米或弧度)
    
    Returns:
        (min_value, max_value): 安全范围
    """
    position = np.array(edge_info.position)
    axis = np.array(edge_info.get_hinge_axis())
    
    parent_vertices = parent_geometry.vertices
    child_vertices = child_geometry.vertices
    
    if joint_type == JointType.HINGE:
        # 铰链: 搜索最大安全角度
        min_angle = 0.0
        max_angle = np.pi
        
        # 二分搜索
        low, high = 0.0, np.pi
        while high - low > 0.01:  # 精度约 0.5 度
            mid = (low + high) / 2
            transformed = _rotate_vertices(child_vertices, position, axis, mid)
            
            if _check_collision_simple(parent_vertices, transformed):
                high = mid
            else:
                low = mid
        
        max_angle = low - safety_margin
        return (0.0, max(0.0, max_angle))
    
    else:
        # 滑轨: 基于几何尺寸估算
        # 最大滑动距离 = 子几何在滑动方向上的尺寸
        child_extent = np.max(child_vertices, axis=0) - np.min(child_vertices, axis=0)
        max_distance = np.dot(child_extent, axis) * 0.8  # 80% 作为安全范围
        
        return (0.0, max(0.0, max_distance - safety_margin))


def _rotate_vertices(
    vertices: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    angle: float,
) -> np.ndarray:
    """
    绕轴旋转顶点
    
    使用 Rodrigues 旋转公式。
    """
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # 平移到原点
    v = vertices - origin
    
    # Rodrigues 公式
    v_rot = (
        v * cos_a +
        np.cross(axis, v) * sin_a +
        axis * np.dot(v, axis).reshape(-1, 1) * (1 - cos_a)
    )
    
    # 平移回去
    return v_rot + origin


def _check_collision_simple(
    vertices_a: np.ndarray,
    vertices_b: np.ndarray,
    tolerance: float = 0.001,
) -> bool:
    """
    简单碰撞检测 (AABB 穿透检测)
    
    注意: 这是简化检测，用于快速筛选。
    精确检测需要使用 trimesh 或专门的碰撞库。
    
    使用体积穿透检测而非简单的 AABB 重叠，
    避免相邻面板被误判为碰撞。
    """
    # 计算 AABB
    min_a, max_a = np.min(vertices_a, axis=0), np.max(vertices_a, axis=0)
    min_b, max_b = np.min(vertices_b, axis=0), np.max(vertices_b, axis=0)
    
    # 计算 AABB 交叉体积
    overlap_min = np.maximum(min_a, min_b)
    overlap_max = np.minimum(max_a, max_b)
    overlap_size = overlap_max - overlap_min
    
    # 只有在所有维度都有显著重叠时才判定为碰撞
    # 使用 tolerance 作为最小穿透深度
    penetration_threshold = tolerance * 10  # 需要至少 1cm 的穿透
    
    if np.all(overlap_size > penetration_threshold):
        # 计算穿透体积
        penetration_volume = np.prod(np.maximum(overlap_size, 0))
        
        # 计算两个物体的最小体积
        volume_a = np.prod(max_a - min_a)
        volume_b = np.prod(max_b - min_b)
        min_volume = min(volume_a, volume_b)
        
        # 只有穿透体积超过最小物体体积的一定比例时才判定为碰撞
        if min_volume > 0 and penetration_volume > min_volume * 0.1:
            return True
    
    return False


# ============================================================
# Blender Geometry Nodes 接口
# ============================================================

def create_hinge_joint_node_inputs(joint_config: JointConfig) -> Dict:
    """
    创建 nodegroup_hinge_joint 的输入参数
    
    用于在 Blender Geometry Nodes 中创建铰链关节。
    
    Args:
        joint_config: 关节配置
    
    Returns:
        Dict: 节点组输入参数
    """
    return {
        "Joint Label": joint_config.label,
        "Position": joint_config.position,
        "Axis": joint_config.axis,
        "Value": joint_config.initial_value,
        "Min": joint_config.min_value,
        "Max": joint_config.max_value,
        "Show Joint": False,
    }


def create_sliding_joint_node_inputs(joint_config: JointConfig) -> Dict:
    """
    创建 nodegroup_sliding_joint 的输入参数
    
    Args:
        joint_config: 关节配置
    
    Returns:
        Dict: 节点组输入参数
    """
    return {
        "Joint Label": joint_config.label,
        "Position": joint_config.position,
        "Axis": joint_config.axis,
        "Value": joint_config.initial_value,
        "Min": joint_config.min_value,
        "Max": joint_config.max_value,
        "Show Joint": False,
    }


# ============================================================
# 高级关节配置功能
# ============================================================

@dataclass
class BoxJointSet:
    """
    盒子关节集合
    
    管理一个盒子的所有关节配置。
    """
    box_type: str
    joints: List[JointConfig] = field(default_factory=list)
    material: MaterialType = MaterialType.CARDBOARD
    
    def add_joint(self, joint: JointConfig):
        """添加关节"""
        self.joints.append(joint)
    
    def get_joint(self, label: str) -> Optional[JointConfig]:
        """按标签获取关节"""
        for joint in self.joints:
            if joint.label == label:
                return joint
        return None
    
    def validate_all(self) -> Tuple[bool, List[str]]:
        """验证所有关节"""
        errors = []
        for joint in self.joints:
            is_valid, msg = joint.validate()
            if not is_valid:
                errors.append(f"{joint.label}: {msg}")
        
        return len(errors) == 0, errors
    
    def get_dynamics(self) -> Dict[str, JointDynamics]:
        """获取所有关节的动力学参数"""
        result = {}
        for joint in self.joints:
            dynamics = get_joint_dynamics(self.material, joint.joint_type)
            result[joint.label] = dynamics
        return result
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "box_type": self.box_type,
            "material": self.material.value,
            "joints": [j.to_dict() for j in self.joints],
        }


def create_tuck_end_box_joints(
    base_dimensions: Tuple[float, float, float],
    height: float,
    thickness: float,
) -> BoxJointSet:
    """
    创建双插盒的关节配置
    
    双插盒有顶盖和底盖两个铰链关节。
    
    Args:
        base_dimensions: 底面尺寸 (width, depth, thickness)
        height: 盒子高度
        thickness: 材料厚度
    
    Returns:
        BoxJointSet: 关节集合
    """
    width, depth, _ = base_dimensions
    
    joint_set = BoxJointSet(box_type="TuckEndBox", material=MaterialType.CARDBOARD)
    
    # 顶盖铰链 (后边缘)
    top_hinge = JointConfig(
        label="top_lid_hinge",
        joint_type=JointType.HINGE,
        position=(0, depth / 2, height),
        axis=(1, 0, 0),
        min_value=0.0,
        max_value=np.pi * 0.8,  # 约 144 度
        initial_value=0.0,
        parent_link="side_back_link",
        child_link="lid_top_link",
    )
    joint_set.add_joint(top_hinge)
    
    # 底盖铰链 (前边缘)
    bottom_hinge = JointConfig(
        label="bottom_lid_hinge",
        joint_type=JointType.HINGE,
        position=(0, -depth / 2, 0),
        axis=(1, 0, 0),
        min_value=-np.pi * 0.8,
        max_value=0.0,
        initial_value=0.0,
        parent_link="side_front_link",
        child_link="lid_bottom_link",
    )
    joint_set.add_joint(bottom_hinge)
    
    return joint_set


def create_drawer_box_joints(
    box_dimensions: Tuple[float, float, float],
    drawer_depth: float,
) -> BoxJointSet:
    """
    创建抽屉盒的关节配置
    
    Args:
        box_dimensions: 盒子尺寸 (width, depth, height)
        drawer_depth: 抽屉可拉出深度
    
    Returns:
        BoxJointSet: 关节集合
    """
    width, depth, height = box_dimensions
    
    joint_set = BoxJointSet(box_type="DrawerBox", material=MaterialType.CARDBOARD)
    
    # 抽屉滑轨
    drawer_slide = JointConfig(
        label="drawer_slide",
        joint_type=JointType.SLIDE,
        position=(0, 0, height / 2),
        axis=(0, -1, 0),  # 向前拉出
        min_value=0.0,
        max_value=drawer_depth,
        initial_value=0.0,
        parent_link="outer_box_link",
        child_link="drawer_link",
    )
    joint_set.add_joint(drawer_slide)
    
    return joint_set


def create_mailer_box_joints(
    base_dimensions: Tuple[float, float, float],
    height: float,
) -> BoxJointSet:
    """
    创建飞机盒的关节配置
    
    飞机盒有一个大盖子从后边缘铰接。
    
    Args:
        base_dimensions: 底面尺寸 (width, depth, thickness)
        height: 盒子高度
    
    Returns:
        BoxJointSet: 关节集合
    """
    width, depth, _ = base_dimensions
    
    joint_set = BoxJointSet(box_type="MailerBox", material=MaterialType.CORRUGATED)
    
    # 盖子铰链
    lid_hinge = JointConfig(
        label="lid_hinge",
        joint_type=JointType.HINGE,
        position=(0, depth / 2, height),
        axis=(1, 0, 0),
        min_value=0.0,
        max_value=np.pi,  # 完全打开
        initial_value=0.0,
        parent_link="back_panel_link",
        child_link="lid_link",
    )
    joint_set.add_joint(lid_hinge)
    
    return joint_set
