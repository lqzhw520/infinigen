# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Max Gonzalez Saez-Diez: primary author

from dataclasses import dataclass
from typing import Dict, Type

import numpy as np


@dataclass
class BaseMaterial:
    """Base material class with physics properties"""

    min_friction: float = 0.8
    max_friction: float = 1.2
    min_density: float = 500  # kg/m³
    max_density: float = 1200  # kg/m³

    def sample_parameters(self) -> Dict[str, float]:
        """Sample random parameters within the material's ranges"""
        return {
            "friction": np.random.uniform(self.min_friction, self.max_friction),
            "density": np.random.uniform(self.min_density, self.max_density),
        }


@dataclass
class Metal(BaseMaterial):
    min_friction: float = 0.8
    max_friction: float = 1.2
    min_density: float = 2500
    max_density: float = 5000


@dataclass
class Wood(BaseMaterial):
    min_friction: float = 0.8
    max_friction: float = 1.2
    min_density: float = 600
    max_density: float = 1000


@dataclass
class Plastic(BaseMaterial):
    min_friction: float = 0.8
    max_friction: float = 1.2
    min_density: float = 850
    max_density: float = 1400


@dataclass
class Ceramic(BaseMaterial):
    min_friction: float = 0.9
    max_friction: float = 1.1
    min_density: float = 2300
    max_density: float = 2500


@dataclass
class Glass(BaseMaterial):
    min_friction: float = 0.9
    max_friction: float = 1.1
    min_density: float = 4900
    max_density: float = 5100


@dataclass
class Marble(BaseMaterial):
    min_friction: float = 0.9
    max_friction: float = 1.1
    min_density: float = 2550
    max_density: float = 2750


@dataclass
class Granite(BaseMaterial):
    min_friction: float = 0.7
    max_friction: float = 0.9
    min_density: float = 2550
    max_density: float = 2750


@dataclass
class Brick(BaseMaterial):
    min_friction: float = 0.9
    max_friction: float = 1.1
    min_density: float = 1700
    max_density: float = 1900


@dataclass
class Plaster(BaseMaterial):
    min_friction: float = 0.9
    max_friction: float = 1.1
    min_density: float = 600
    max_density: float = 800


@dataclass
class Fabric(BaseMaterial):
    min_friction: float = 0.9
    max_friction: float = 1.1
    min_density: float = 50
    max_density: float = 250


@dataclass
class Rubber(BaseMaterial):
    min_friction: float = 0.9
    max_friction: float = 1.1
    min_density: float = 1200
    max_density: float = 1400


# ===================== BOX MATERIALS (P1-T4) =====================
# 盒子资产专用材质，包含额外的物理属性

@dataclass
class BoxMaterial(BaseMaterial):
    """盒子材质基类，扩展了恢复系数和厚度范围"""
    
    min_restitution: float = 0.0  # 恢复系数 (弹性)
    max_restitution: float = 0.1
    min_thickness: float = 0.001  # 材料厚度范围 (m)
    max_thickness: float = 0.005
    
    def sample_parameters(self) -> Dict[str, float]:
        """采样包含恢复系数和厚度的参数"""
        base_params = super().sample_parameters()
        base_params["restitution"] = np.random.uniform(
            self.min_restitution, self.max_restitution
        )
        base_params["thickness"] = np.random.uniform(
            self.min_thickness, self.max_thickness
        )
        return base_params


@dataclass
class Cardboard(BoxMaterial):
    """
    卡纸材质
    
    物理属性基于常见瓦楞纸箱卡纸：
    - 密度: 200-400 kg/m³ (较轻)
    - 摩擦系数: 0.4-0.6 (中等)
    - 恢复系数: 0.05-0.15 (低弹性)
    - 厚度: 0.3-1.5 mm
    """
    min_friction: float = 0.4
    max_friction: float = 0.6
    min_density: float = 200
    max_density: float = 400
    min_restitution: float = 0.05
    max_restitution: float = 0.15
    min_thickness: float = 0.0003  # 0.3 mm
    max_thickness: float = 0.0015  # 1.5 mm


@dataclass
class CardboardWhite(Cardboard):
    """白卡纸 - 较硬，用于礼品盒"""
    min_density: float = 300
    max_density: float = 500
    min_thickness: float = 0.0005
    max_thickness: float = 0.002


@dataclass
class CardboardKraft(Cardboard):
    """牛皮卡纸 - 常见瓦楞纸色"""
    min_density: float = 180
    max_density: float = 350


@dataclass
class Corrugated(BoxMaterial):
    """
    瓦楞纸板材质
    
    包含多种瓦楞类型 (A, B, C, E 型)，
    每种类型的楞高和物理特性不同。
    
    物理属性:
    - 密度: 100-300 kg/m³ (因为有空心结构)
    - 摩擦系数: 0.5-0.7 (略粗糙)
    - 恢复系数: 0.1-0.2 (有一定缓冲性)
    - 厚度: 2-8 mm (取决于瓦楞类型)
    """
    min_friction: float = 0.5
    max_friction: float = 0.7
    min_density: float = 100  # 有效密度 (考虑空心)
    max_density: float = 300
    min_restitution: float = 0.1
    max_restitution: float = 0.2
    min_thickness: float = 0.002  # 2 mm
    max_thickness: float = 0.008  # 8 mm
    
    # 瓦楞类型特定属性
    flute_type: str = "B"  # A, B, C, E 型
    
    @classmethod
    def type_a(cls) -> "Corrugated":
        """A 型瓦楞 (粗瓦楞) - 楞高 4.5-5.0 mm"""
        return cls(
            min_thickness=0.004,
            max_thickness=0.006,
            min_density=80,
            max_density=200,
            flute_type="A",
        )
    
    @classmethod
    def type_b(cls) -> "Corrugated":
        """B 型瓦楞 (细瓦楞) - 楞高 2.5-3.0 mm"""
        return cls(
            min_thickness=0.002,
            max_thickness=0.004,
            min_density=120,
            max_density=280,
            flute_type="B",
        )
    
    @classmethod
    def type_c(cls) -> "Corrugated":
        """C 型瓦楞 (中瓦楞) - 楞高 3.5-4.0 mm"""
        return cls(
            min_thickness=0.003,
            max_thickness=0.005,
            min_density=100,
            max_density=250,
            flute_type="C",
        )
    
    @classmethod
    def type_e(cls) -> "Corrugated":
        """E 型瓦楞 (微瓦楞) - 楞高 1.1-1.5 mm"""
        return cls(
            min_thickness=0.001,
            max_thickness=0.002,
            min_density=150,
            max_density=350,
            flute_type="E",
        )


@dataclass
class PlasticBox(BoxMaterial):
    """
    塑料盒材质 (PP, PET, HDPE)
    
    物理属性:
    - 密度: 900-1400 kg/m³
    - 摩擦系数: 0.3-0.5 (光滑)
    - 恢复系数: 0.3-0.5 (有弹性)
    - 厚度: 0.5-3 mm
    """
    min_friction: float = 0.3
    max_friction: float = 0.5
    min_density: float = 900
    max_density: float = 1400
    min_restitution: float = 0.3
    max_restitution: float = 0.5
    min_thickness: float = 0.0005  # 0.5 mm
    max_thickness: float = 0.003   # 3 mm
    
    plastic_type: str = "PP"  # PP, PET, HDPE
    
    @classmethod
    def pp(cls) -> "PlasticBox":
        """聚丙烯 (PP)"""
        return cls(
            min_density=900,
            max_density=910,
            plastic_type="PP",
        )
    
    @classmethod
    def pet(cls) -> "PlasticBox":
        """聚对苯二甲酸乙二醇酯 (PET)"""
        return cls(
            min_density=1300,
            max_density=1400,
            plastic_type="PET",
        )
    
    @classmethod
    def hdpe(cls) -> "PlasticBox":
        """高密度聚乙烯 (HDPE)"""
        return cls(
            min_density=940,
            max_density=970,
            plastic_type="HDPE",
        )


@dataclass
class WoodBox(BoxMaterial):
    """
    木质盒材质 (MDF, 胶合板)
    
    物理属性:
    - 密度: 500-900 kg/m³
    - 摩擦系数: 0.4-0.6
    - 恢复系数: 0.1-0.2
    - 厚度: 3-15 mm
    """
    min_friction: float = 0.4
    max_friction: float = 0.6
    min_density: float = 500
    max_density: float = 900
    min_restitution: float = 0.1
    max_restitution: float = 0.2
    min_thickness: float = 0.003   # 3 mm
    max_thickness: float = 0.015   # 15 mm
    
    wood_type: str = "MDF"
    
    @classmethod
    def mdf(cls) -> "WoodBox":
        """中密度纤维板 (MDF)"""
        return cls(
            min_density=600,
            max_density=800,
            wood_type="MDF",
        )
    
    @classmethod
    def plywood(cls) -> "WoodBox":
        """胶合板"""
        return cls(
            min_density=500,
            max_density=700,
            wood_type="Plywood",
        )


# ===================== MATERIAL REGISTRY =====================
MATERIALS: Dict[str, Type[BaseMaterial]] = {
    "base": BaseMaterial,
    "metal": Metal,
    "wood": Wood,
    "plastic": Plastic,
    "granite": Granite,
    "brick": Brick,
    "ceramic": Ceramic,
    "glass": Glass,
    "marble": Marble,
    "plaster": Plaster,
    "fabric": Fabric,
    "rubber": Rubber,
    # Box materials (P1-T4)
    "cardboard": Cardboard,
    "cardboard_white": CardboardWhite,
    "cardboard_kraft": CardboardKraft,
    "corrugated": Corrugated,
    "plastic_box": PlasticBox,
    "wood_box": WoodBox,
}


# ===================== BOX MATERIAL HELPERS =====================

def get_box_material(material_name: str) -> BoxMaterial:
    """
    获取盒子材质实例
    
    Args:
        material_name: 材质名称
    
    Returns:
        BoxMaterial 实例
    
    Raises:
        ValueError: 如果材质名称无效
    """
    material_map = {
        "cardboard": Cardboard,
        "cardboard_white": CardboardWhite,
        "cardboard_kraft": CardboardKraft,
        "corrugated": Corrugated,
        "corrugated_a": lambda: Corrugated.type_a(),
        "corrugated_b": lambda: Corrugated.type_b(),
        "corrugated_c": lambda: Corrugated.type_c(),
        "corrugated_e": lambda: Corrugated.type_e(),
        "plastic_pp": lambda: PlasticBox.pp(),
        "plastic_pet": lambda: PlasticBox.pet(),
        "plastic_hdpe": lambda: PlasticBox.hdpe(),
        "wood_mdf": lambda: WoodBox.mdf(),
        "wood_plywood": lambda: WoodBox.plywood(),
    }
    
    if material_name not in material_map:
        raise ValueError(
            f"Unknown box material: {material_name}. "
            f"Available: {list(material_map.keys())}"
        )
    
    material = material_map[material_name]
    if callable(material) and not isinstance(material, type):
        return material()
    return material()


def list_box_materials() -> list:
    """列出所有可用的盒子材质"""
    return [
        "cardboard",
        "cardboard_white",
        "cardboard_kraft",
        "corrugated",
        "corrugated_a",
        "corrugated_b",
        "corrugated_c",
        "corrugated_e",
        "plastic_pp",
        "plastic_pet",
        "plastic_hdpe",
        "wood_mdf",
        "wood_plywood",
    ]
