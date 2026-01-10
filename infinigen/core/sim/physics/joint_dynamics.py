# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: primary author

"""
关节动力学模块

提供关节的动力学参数配置，包括：
- 材质相关的预设 (P1-T5)
- 盒型相关的默认值
- URDF/MJCF 导出所需的参数

P1-T5 扩展: 添加材质相关的关节动力学预设
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
import numpy as np


class JointDynamicsType(Enum):
    """关节动力学类型"""
    HINGE = "hinge"    # 铰链/旋转关节
    SLIDE = "slide"    # 滑轨/平移关节
    BALL = "ball"      # 球窝关节
    FIXED = "fixed"    # 固定关节


class MaterialCategory(Enum):
    """材质类别"""
    CARDBOARD = "cardboard"
    CORRUGATED = "corrugated"
    PLASTIC = "plastic"
    WOOD = "wood"
    METAL = "metal"
    FABRIC = "fabric"


@dataclass
class JointDynamicsParams:
    """
    关节动力学参数
    
    用于 URDF/MJCF 导出和仿真。
    
    Attributes:
        damping: 阻尼系数 (Ns/m 或 Nms/rad)
        friction: 摩擦系数 (静摩擦)
        stiffness: 刚度系数 (N/m 或 Nm/rad)
        velocity_limit: 速度限制 (m/s 或 rad/s)
        effort_limit: 力/力矩限制 (N 或 Nm)
        armature: 转动惯量等效 (用于 MJCF)
    """
    damping: float = 0.5
    friction: float = 0.1
    stiffness: float = 0.0
    velocity_limit: float = 10.0
    effort_limit: float = 100.0
    armature: float = 0.0
    
    def to_urdf_dict(self) -> Dict[str, float]:
        """转换为 URDF 格式的字典"""
        return {
            "damping": self.damping,
            "friction": self.friction,
        }
    
    def to_mjcf_dict(self) -> Dict[str, float]:
        """转换为 MJCF 格式的字典"""
        return {
            "damping": self.damping,
            "frictionloss": self.friction,
            "stiffness": self.stiffness,
            "armature": self.armature,
        }
    
    def scale(self, factor: float) -> "JointDynamicsParams":
        """按比例缩放所有参数"""
        return JointDynamicsParams(
            damping=self.damping * factor,
            friction=self.friction * factor,
            stiffness=self.stiffness * factor,
            velocity_limit=self.velocity_limit,
            effort_limit=self.effort_limit * factor,
            armature=self.armature * factor,
        )


# ============================================================
# 材质相关关节动力学预设 (P1-T5)
# ============================================================

MATERIAL_JOINT_PRESETS: Dict[MaterialCategory, Dict[JointDynamicsType, JointDynamicsParams]] = {
    MaterialCategory.CARDBOARD: {
        JointDynamicsType.HINGE: JointDynamicsParams(
            damping=0.3,
            friction=0.15,
            stiffness=0.0,      # 卡纸无弹性恢复
            velocity_limit=5.0,
            effort_limit=2.0,   # 卡纸较弱
            armature=0.001,
        ),
        JointDynamicsType.SLIDE: JointDynamicsParams(
            damping=0.5,
            friction=0.3,       # 滑动摩擦较大
            stiffness=0.0,
            velocity_limit=2.0,
            effort_limit=1.0,
            armature=0.001,
        ),
    },
    MaterialCategory.CORRUGATED: {
        JointDynamicsType.HINGE: JointDynamicsParams(
            damping=0.5,
            friction=0.2,
            stiffness=0.1,      # 瓦楞纸有轻微弹性
            velocity_limit=4.0,
            effort_limit=5.0,   # 瓦楞纸较强
            armature=0.002,
        ),
        JointDynamicsType.SLIDE: JointDynamicsParams(
            damping=0.6,
            friction=0.25,
            stiffness=0.0,
            velocity_limit=1.5,
            effort_limit=3.0,
            armature=0.002,
        ),
    },
    MaterialCategory.PLASTIC: {
        JointDynamicsType.HINGE: JointDynamicsParams(
            damping=0.2,
            friction=0.1,
            stiffness=0.5,      # 塑料有弹性
            velocity_limit=8.0,
            effort_limit=10.0,
            armature=0.003,
        ),
        JointDynamicsType.SLIDE: JointDynamicsParams(
            damping=0.3,
            friction=0.1,       # 塑料滑动摩擦小
            stiffness=0.0,
            velocity_limit=5.0,
            effort_limit=8.0,
            armature=0.002,
        ),
    },
    MaterialCategory.WOOD: {
        JointDynamicsType.HINGE: JointDynamicsParams(
            damping=0.8,
            friction=0.3,
            stiffness=0.0,      # 木质无弹性
            velocity_limit=3.0,
            effort_limit=50.0,  # 木质较强
            armature=0.01,
        ),
        JointDynamicsType.SLIDE: JointDynamicsParams(
            damping=1.0,
            friction=0.4,
            stiffness=0.0,
            velocity_limit=2.0,
            effort_limit=40.0,
            armature=0.01,
        ),
    },
    MaterialCategory.METAL: {
        JointDynamicsType.HINGE: JointDynamicsParams(
            damping=0.1,
            friction=0.05,
            stiffness=1.0,
            velocity_limit=15.0,
            effort_limit=200.0,
            armature=0.05,
        ),
        JointDynamicsType.SLIDE: JointDynamicsParams(
            damping=0.2,
            friction=0.05,
            stiffness=0.0,
            velocity_limit=10.0,
            effort_limit=150.0,
            armature=0.03,
        ),
    },
    MaterialCategory.FABRIC: {
        JointDynamicsType.HINGE: JointDynamicsParams(
            damping=0.1,
            friction=0.4,
            stiffness=0.0,
            velocity_limit=10.0,
            effort_limit=0.5,   # 织物很弱
            armature=0.0005,
        ),
        JointDynamicsType.SLIDE: JointDynamicsParams(
            damping=0.2,
            friction=0.5,
            stiffness=0.0,
            velocity_limit=5.0,
            effort_limit=0.3,
            armature=0.0005,
        ),
    },
}


# ============================================================
# 盒型相关关节预设
# ============================================================

BOX_TYPE_JOINT_PRESETS: Dict[str, Dict[str, JointDynamicsParams]] = {
    "TuckEndBox": {
        "top_lid": JointDynamicsParams(damping=0.3, friction=0.15, stiffness=0.0),
        "bottom_lid": JointDynamicsParams(damping=0.3, friction=0.15, stiffness=0.0),
        "tuck_flap": JointDynamicsParams(damping=0.2, friction=0.1, stiffness=0.05),
    },
    "MailerBox": {
        "lid": JointDynamicsParams(damping=0.5, friction=0.2, stiffness=0.1),
    },
    "DrawerBox": {
        "drawer": JointDynamicsParams(damping=0.6, friction=0.25, stiffness=0.0),
    },
    "RSCBox": {
        "top_flap_front": JointDynamicsParams(damping=0.4, friction=0.2, stiffness=0.05),
        "top_flap_back": JointDynamicsParams(damping=0.4, friction=0.2, stiffness=0.05),
        "top_flap_left": JointDynamicsParams(damping=0.4, friction=0.2, stiffness=0.05),
        "top_flap_right": JointDynamicsParams(damping=0.4, friction=0.2, stiffness=0.05),
        "bottom_flap_front": JointDynamicsParams(damping=0.4, friction=0.2, stiffness=0.05),
        "bottom_flap_back": JointDynamicsParams(damping=0.4, friction=0.2, stiffness=0.05),
        "bottom_flap_left": JointDynamicsParams(damping=0.4, friction=0.2, stiffness=0.05),
        "bottom_flap_right": JointDynamicsParams(damping=0.4, friction=0.2, stiffness=0.05),
    },
    "GiftBoxWithHandle": {
        "lid": JointDynamicsParams(damping=0.3, friction=0.15, stiffness=0.0),
    },
    "SlipLidBox": {
        # 天地盒无关节或简单滑动
        "lid_slide": JointDynamicsParams(damping=0.5, friction=0.3, stiffness=0.0),
    },
}


# ============================================================
# 关节动力学查询函数
# ============================================================

def get_joint_properties(joint_name: str, joint_params: Dict) -> Dict:
    """
    获取关节属性 (原有接口，保持兼容)
    
    Args:
        joint_name: 关节名称
        joint_params: 关节参数字典
    
    Returns:
        包含 stiffness, damping, friction 的字典
    """
    if joint_name not in joint_params:
        return {"stiffness": 0, "damping": 0, "friction": 0}
    res = {
        "stiffness": joint_params[joint_name].get("stiffness", 0.0),
        "damping": joint_params[joint_name].get("damping", 0.0),
        "friction": joint_params[joint_name].get("friction", 0.0),
    }

    return res


def get_material_joint_dynamics(
    material: MaterialCategory,
    joint_type: JointDynamicsType = JointDynamicsType.HINGE,
) -> JointDynamicsParams:
    """
    获取材质相关的关节动力学参数
    
    Args:
        material: 材质类别
        joint_type: 关节类型
    
    Returns:
        JointDynamicsParams 实例
    """
    if material in MATERIAL_JOINT_PRESETS:
        presets = MATERIAL_JOINT_PRESETS[material]
        if joint_type in presets:
            return presets[joint_type]
    
    # 默认返回卡纸铰链参数
    return MATERIAL_JOINT_PRESETS[MaterialCategory.CARDBOARD][JointDynamicsType.HINGE]


def get_box_type_joint_dynamics(
    box_type: str,
    joint_label: str,
    material: Optional[MaterialCategory] = None,
) -> JointDynamicsParams:
    """
    获取盒型和关节标签对应的动力学参数
    
    优先使用盒型特定预设，如果没有则使用材质预设。
    
    Args:
        box_type: 盒型名称
        joint_label: 关节标签
        material: 材质类别 (可选)
    
    Returns:
        JointDynamicsParams 实例
    """
    # 首先检查盒型特定预设
    if box_type in BOX_TYPE_JOINT_PRESETS:
        box_presets = BOX_TYPE_JOINT_PRESETS[box_type]
        if joint_label in box_presets:
            return box_presets[joint_label]
    
    # 如果有材质信息，使用材质预设
    if material:
        return get_material_joint_dynamics(material, JointDynamicsType.HINGE)
    
    # 默认返回卡纸铰链
    return MATERIAL_JOINT_PRESETS[MaterialCategory.CARDBOARD][JointDynamicsType.HINGE]


def material_name_to_category(material_name: str) -> MaterialCategory:
    """
    将材质名称转换为材质类别
    
    Args:
        material_name: 材质名称 (如 "cardboard", "corrugated_a", "plastic_pp")
    
    Returns:
        MaterialCategory 枚举值
    """
    name_lower = material_name.lower()
    
    if "cardboard" in name_lower:
        return MaterialCategory.CARDBOARD
    elif "corrugated" in name_lower:
        return MaterialCategory.CORRUGATED
    elif "plastic" in name_lower:
        return MaterialCategory.PLASTIC
    elif "wood" in name_lower:
        return MaterialCategory.WOOD
    elif "metal" in name_lower:
        return MaterialCategory.METAL
    elif "fabric" in name_lower:
        return MaterialCategory.FABRIC
    else:
        return MaterialCategory.CARDBOARD  # 默认


def list_available_materials() -> list:
    """列出所有可用的材质类别"""
    return [m.value for m in MaterialCategory]


def list_box_type_presets() -> list:
    """列出所有盒型预设"""
    return list(BOX_TYPE_JOINT_PRESETS.keys())


# ============================================================
# 关节动力学计算辅助函数
# ============================================================

def calculate_hinge_damping(
    mass: float,
    arm_length: float,
    settling_time: float = 1.0,
) -> float:
    """
    根据物理参数计算铰链阻尼
    
    基于临界阻尼公式计算合适的阻尼值。
    
    Args:
        mass: 质量 (kg)
        arm_length: 力臂长度 (m)
        settling_time: 期望的稳定时间 (s)
    
    Returns:
        阻尼系数 (Nms/rad)
    """
    # 简化的临界阻尼计算
    # damping ≈ 2 * sqrt(I * k)，这里 I = m * r^2
    inertia = mass * arm_length ** 2
    natural_freq = 2 * np.pi / settling_time
    critical_damping = 2 * np.sqrt(inertia * natural_freq ** 2)
    
    # 返回略低于临界阻尼的值（欠阻尼，有轻微振荡）
    return critical_damping * 0.7


def calculate_slide_friction(
    normal_force: float,
    coefficient: float = 0.3,
) -> float:
    """
    计算滑动摩擦力
    
    Args:
        normal_force: 法向力 (N)
        coefficient: 摩擦系数
    
    Returns:
        摩擦力 (N)
    """
    return normal_force * coefficient


def estimate_joint_effort_limit(
    mass: float,
    arm_length: float,
    safety_factor: float = 2.0,
) -> float:
    """
    估算关节力矩限制
    
    Args:
        mass: 质量 (kg)
        arm_length: 力臂长度 (m)
        safety_factor: 安全系数
    
    Returns:
        力矩限制 (Nm)
    """
    gravity_torque = mass * 9.81 * arm_length
    return gravity_torque * safety_factor
