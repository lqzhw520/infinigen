# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""
P1-T2: 基础几何模块 (box_geometry_modules.py)

提供盒子资产的基础几何构建块，用于各种盒型工厂。
所有函数返回 Blender Geometry Nodes 节点组或几何数据。

主要功能:
- create_base_panel(): 底面板生成
- create_side_panel(): 侧面板生成
- create_lid_panel(): 盖板生成
- create_flap(): 翻盖/插舌生成
- create_fold_line(): 折痕线生成

依赖:
- P1-T1: ModularBoxFactory (BoxDimensions, BoxMaterialConfig)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class PanelType(Enum):
    """面板类型枚举"""
    BASE = "base"          # 底面板
    TOP = "top"            # 顶面板
    SIDE_LEFT = "side_left"    # 左侧面板
    SIDE_RIGHT = "side_right"  # 右侧面板
    SIDE_FRONT = "side_front"  # 前侧面板
    SIDE_BACK = "side_back"    # 后侧面板
    LID = "lid"            # 盖板
    FLAP = "flap"          # 翻盖


class FlapType(Enum):
    """翻盖类型枚举"""
    TUCK = "tuck"          # 插舌
    LOCK = "lock"          # 锁扣
    DUST_FLAP = "dust_flap"  # 防尘翻盖
    SAFETY = "safety"      # 保险扣
    HOOK = "hook"          # 挂钩


class EdgePosition(Enum):
    """边缘位置枚举"""
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    FRONT = "front"
    BACK = "back"


@dataclass
class EdgeInfo:
    """
    边缘信息，用于关节附着
    
    Attributes:
        position: 边缘中心位置 (x, y, z)
        direction: 边缘方向向量 (单位向量)
        length: 边缘长度 (米)
        edge_type: 边缘位置类型
    """
    position: Tuple[float, float, float]
    direction: Tuple[float, float, float]
    length: float
    edge_type: EdgePosition
    
    def get_hinge_axis(self) -> Tuple[float, float, float]:
        """获取适合铰链的旋转轴"""
        return self.direction
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "position": list(self.position),
            "direction": list(self.direction),
            "length": self.length,
            "edge_type": self.edge_type.value,
        }


@dataclass
class PanelGeometry:
    """
    面板几何数据
    
    Attributes:
        panel_type: 面板类型
        vertices: 顶点坐标 (N x 3)
        faces: 面索引 (M x 4 for quads)
        edges: 边缘信息字典
        center: 面板中心位置
        normal: 面板法线方向
        dimensions: (宽度, 高度, 厚度)
    """
    panel_type: PanelType
    vertices: np.ndarray
    faces: np.ndarray
    edges: Dict[EdgePosition, EdgeInfo]
    center: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    dimensions: Tuple[float, float, float]  # width, height, thickness
    
    def get_edge(self, position: EdgePosition) -> Optional[EdgeInfo]:
        """获取指定位置的边缘信息"""
        return self.edges.get(position)
    
    def to_trimesh(self):
        """转换为 trimesh 对象"""
        import trimesh
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
    
    def to_dict(self) -> Dict:
        """转换为可序列化的字典"""
        return {
            "panel_type": self.panel_type.value,
            "vertices": self.vertices.tolist(),
            "faces": self.faces.tolist(),
            "edges": {k.value: v.to_dict() for k, v in self.edges.items()},
            "center": list(self.center),
            "normal": list(self.normal),
            "dimensions": list(self.dimensions),
        }


@dataclass
class FlapGeometry:
    """
    翻盖几何数据
    
    Attributes:
        flap_type: 翻盖类型
        vertices: 顶点坐标
        faces: 面索引
        hinge_edge: 铰链边缘信息
        tip_edge: 翻盖末端边缘
        lock_feature: 锁定特征信息 (如果有)
    """
    flap_type: FlapType
    vertices: np.ndarray
    faces: np.ndarray
    hinge_edge: EdgeInfo
    tip_edge: EdgeInfo
    lock_feature: Optional[Dict] = None
    
    def to_trimesh(self):
        """转换为 trimesh 对象"""
        import trimesh
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces)


def create_base_panel(
    width: float,
    depth: float,
    thickness: float,
    center: Tuple[float, float, float] = (0, 0, 0),
    corner_radius: float = 0.0,
) -> PanelGeometry:
    """
    创建底面板几何
    
    底面板位于 XY 平面，法线朝向 +Z。
    
    Args:
        width: 面板宽度 (X 方向, 米)
        depth: 面板深度 (Y 方向, 米)
        thickness: 面板厚度 (Z 方向, 米)
        center: 面板中心位置
        corner_radius: 角部圆角半径 (0 表示直角)
    
    Returns:
        PanelGeometry: 底面板几何数据
    """
    cx, cy, cz = center
    hw = width / 2  # half width
    hd = depth / 2  # half depth
    ht = thickness / 2  # half thickness
    
    if corner_radius <= 0:
        # 简单矩形面板 (8 顶点的长方体)
        vertices = np.array([
            # 底面 (z = cz - ht)
            [cx - hw, cy - hd, cz - ht],  # 0
            [cx + hw, cy - hd, cz - ht],  # 1
            [cx + hw, cy + hd, cz - ht],  # 2
            [cx - hw, cy + hd, cz - ht],  # 3
            # 顶面 (z = cz + ht)
            [cx - hw, cy - hd, cz + ht],  # 4
            [cx + hw, cy - hd, cz + ht],  # 5
            [cx + hw, cy + hd, cz + ht],  # 6
            [cx - hw, cy + hd, cz + ht],  # 7
        ])
        
        # 六个面 (每个面 4 个顶点索引)
        faces = np.array([
            [0, 3, 2, 1],  # 底面 (-Z)
            [4, 5, 6, 7],  # 顶面 (+Z)
            [0, 1, 5, 4],  # 前面 (-Y)
            [2, 3, 7, 6],  # 后面 (+Y)
            [0, 4, 7, 3],  # 左面 (-X)
            [1, 2, 6, 5],  # 右面 (+X)
        ])
    else:
        # 带圆角的面板 (使用细分)
        vertices, faces = _create_rounded_box(
            width, depth, thickness, center, corner_radius
        )
    
    # 构建边缘信息
    edges = {
        EdgePosition.FRONT: EdgeInfo(
            position=(cx, cy - hd, cz + ht),
            direction=(1, 0, 0),
            length=width,
            edge_type=EdgePosition.FRONT,
        ),
        EdgePosition.BACK: EdgeInfo(
            position=(cx, cy + hd, cz + ht),
            direction=(1, 0, 0),
            length=width,
            edge_type=EdgePosition.BACK,
        ),
        EdgePosition.LEFT: EdgeInfo(
            position=(cx - hw, cy, cz + ht),
            direction=(0, 1, 0),
            length=depth,
            edge_type=EdgePosition.LEFT,
        ),
        EdgePosition.RIGHT: EdgeInfo(
            position=(cx + hw, cy, cz + ht),
            direction=(0, 1, 0),
            length=depth,
            edge_type=EdgePosition.RIGHT,
        ),
    }
    
    return PanelGeometry(
        panel_type=PanelType.BASE,
        vertices=vertices,
        faces=faces,
        edges=edges,
        center=center,
        normal=(0, 0, 1),
        dimensions=(width, depth, thickness),
    )


def create_side_panel(
    width: float,
    height: float,
    thickness: float,
    side: EdgePosition,
    parent_dimensions: Tuple[float, float, float],
    parent_center: Tuple[float, float, float] = (0, 0, 0),
    corner_radius: float = 0.0,
) -> PanelGeometry:
    """
    创建侧面板几何
    
    侧面板从底面板边缘向上延伸。
    
    Args:
        width: 侧面板宽度 (沿边缘方向)
        height: 侧面板高度 (垂直方向)
        thickness: 侧面板厚度
        side: 侧面板位置 (FRONT, BACK, LEFT, RIGHT)
        parent_dimensions: 父面板尺寸 (width, depth, thickness)
        parent_center: 父面板中心位置
        corner_radius: 角部圆角半径
    
    Returns:
        PanelGeometry: 侧面板几何数据
    """
    pcx, pcy, pcz = parent_center
    pw, pd, pt = parent_dimensions
    hw = width / 2
    hh = height / 2
    ht = thickness / 2
    
    # 根据侧面位置计算中心和法线
    if side == EdgePosition.FRONT:
        center = (pcx, pcy - pd/2 - ht, pcz + pt/2 + hh)
        normal = (0, -1, 0)
        panel_type = PanelType.SIDE_FRONT
    elif side == EdgePosition.BACK:
        center = (pcx, pcy + pd/2 + ht, pcz + pt/2 + hh)
        normal = (0, 1, 0)
        panel_type = PanelType.SIDE_BACK
    elif side == EdgePosition.LEFT:
        center = (pcx - pw/2 - ht, pcy, pcz + pt/2 + hh)
        normal = (-1, 0, 0)
        panel_type = PanelType.SIDE_LEFT
    elif side == EdgePosition.RIGHT:
        center = (pcx + pw/2 + ht, pcy, pcz + pt/2 + hh)
        normal = (1, 0, 0)
        panel_type = PanelType.SIDE_RIGHT
    else:
        raise ValueError(f"Invalid side position: {side}")
    
    cx, cy, cz = center
    
    # 根据朝向创建顶点
    if side in [EdgePosition.FRONT, EdgePosition.BACK]:
        # 面朝 Y 方向的面板
        sign = -1 if side == EdgePosition.FRONT else 1
        vertices = np.array([
            # 内侧面 (靠近盒子内部)
            [cx - hw, cy - sign * ht, cz - hh],
            [cx + hw, cy - sign * ht, cz - hh],
            [cx + hw, cy - sign * ht, cz + hh],
            [cx - hw, cy - sign * ht, cz + hh],
            # 外侧面
            [cx - hw, cy + sign * ht, cz - hh],
            [cx + hw, cy + sign * ht, cz - hh],
            [cx + hw, cy + sign * ht, cz + hh],
            [cx - hw, cy + sign * ht, cz + hh],
        ])
        # 边缘方向
        edge_dir_horizontal = (1, 0, 0)
        edge_dir_vertical = (0, 0, 1)
    else:
        # 面朝 X 方向的面板
        sign = -1 if side == EdgePosition.LEFT else 1
        vertices = np.array([
            # 内侧面
            [cx - sign * ht, cy - hw, cz - hh],
            [cx - sign * ht, cy + hw, cz - hh],
            [cx - sign * ht, cy + hw, cz + hh],
            [cx - sign * ht, cy - hw, cz + hh],
            # 外侧面
            [cx + sign * ht, cy - hw, cz - hh],
            [cx + sign * ht, cy + hw, cz - hh],
            [cx + sign * ht, cy + hw, cz + hh],
            [cx + sign * ht, cy - hw, cz + hh],
        ])
        edge_dir_horizontal = (0, 1, 0)
        edge_dir_vertical = (0, 0, 1)
    
    faces = np.array([
        [0, 1, 2, 3],  # 内侧面
        [4, 7, 6, 5],  # 外侧面
        [0, 3, 7, 4],  # 左边
        [1, 5, 6, 2],  # 右边
        [0, 4, 5, 1],  # 底边
        [3, 2, 6, 7],  # 顶边
    ])
    
    # 构建边缘信息
    edges = {
        EdgePosition.TOP: EdgeInfo(
            position=(cx, cy, cz + hh),
            direction=edge_dir_horizontal,
            length=width,
            edge_type=EdgePosition.TOP,
        ),
        EdgePosition.BOTTOM: EdgeInfo(
            position=(cx, cy, cz - hh),
            direction=edge_dir_horizontal,
            length=width,
            edge_type=EdgePosition.BOTTOM,
        ),
    }
    
    return PanelGeometry(
        panel_type=panel_type,
        vertices=vertices,
        faces=faces,
        edges=edges,
        center=center,
        normal=normal,
        dimensions=(width, height, thickness),
    )


def create_lid_panel(
    width: float,
    depth: float,
    thickness: float,
    attach_edge: EdgePosition,
    parent_dimensions: Tuple[float, float, float],
    parent_center: Tuple[float, float, float] = (0, 0, 0),
    parent_height: float = 0.1,
    corner_radius: float = 0.0,
) -> PanelGeometry:
    """
    创建盖板几何
    
    盖板附着在盒子顶部边缘，可以翻转打开。
    
    Args:
        width: 盖板宽度
        depth: 盖板深度
        thickness: 盖板厚度
        attach_edge: 附着边缘位置
        parent_dimensions: 父面板(底)尺寸
        parent_center: 父面板中心位置
        parent_height: 盒子高度
        corner_radius: 角部圆角半径
    
    Returns:
        PanelGeometry: 盖板几何数据
    """
    pcx, pcy, pcz = parent_center
    pw, pd, pt = parent_dimensions
    hw = width / 2
    hd = depth / 2
    ht = thickness / 2
    
    # 盖板位于盒子顶部，初始状态是闭合的
    top_z = pcz + pt/2 + parent_height
    
    # 根据附着边缘确定盖板位置
    if attach_edge == EdgePosition.BACK:
        # 盖板铰链在后边缘，向前延伸
        center = (pcx, pcy + pd/2 - hd, top_z + ht)
        hinge_pos = (pcx, pcy + pd/2, top_z)
        hinge_dir = (1, 0, 0)
    elif attach_edge == EdgePosition.FRONT:
        # 盖板铰链在前边缘，向后延伸
        center = (pcx, pcy - pd/2 + hd, top_z + ht)
        hinge_pos = (pcx, pcy - pd/2, top_z)
        hinge_dir = (1, 0, 0)
    elif attach_edge == EdgePosition.LEFT:
        center = (pcx - pw/2 + hw, pcy, top_z + ht)
        hinge_pos = (pcx - pw/2, pcy, top_z)
        hinge_dir = (0, 1, 0)
    elif attach_edge == EdgePosition.RIGHT:
        center = (pcx + pw/2 - hw, pcy, top_z + ht)
        hinge_pos = (pcx + pw/2, pcy, top_z)
        hinge_dir = (0, 1, 0)
    else:
        raise ValueError(f"Invalid attach edge: {attach_edge}")
    
    cx, cy, cz = center
    
    # 创建简单矩形盖板
    vertices = np.array([
        [cx - hw, cy - hd, cz - ht],
        [cx + hw, cy - hd, cz - ht],
        [cx + hw, cy + hd, cz - ht],
        [cx - hw, cy + hd, cz - ht],
        [cx - hw, cy - hd, cz + ht],
        [cx + hw, cy - hd, cz + ht],
        [cx + hw, cy + hd, cz + ht],
        [cx - hw, cy + hd, cz + ht],
    ])
    
    faces = np.array([
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 4, 7, 3],
        [1, 2, 6, 5],
    ])
    
    # 构建边缘信息，包含铰链边缘
    edges = {
        EdgePosition.TOP: EdgeInfo(
            position=hinge_pos,
            direction=hinge_dir,
            length=width,
            edge_type=attach_edge,
        ),
    }
    
    return PanelGeometry(
        panel_type=PanelType.LID,
        vertices=vertices,
        faces=faces,
        edges=edges,
        center=center,
        normal=(0, 0, 1),
        dimensions=(width, depth, thickness),
    )


def create_flap(
    width: float,
    length: float,
    thickness: float,
    flap_type: FlapType,
    attach_edge: EdgeInfo,
    tuck_depth: float = 0.0,
) -> FlapGeometry:
    """
    创建翻盖/插舌几何
    
    Args:
        width: 翻盖宽度 (沿铰链方向)
        length: 翻盖长度 (延伸方向)
        thickness: 翻盖厚度
        flap_type: 翻盖类型
        attach_edge: 附着边缘信息
        tuck_depth: 插舌深度 (仅对 TUCK 类型有效)
    
    Returns:
        FlapGeometry: 翻盖几何数据
    """
    ex, ey, ez = attach_edge.position
    dx, dy, dz = attach_edge.direction
    hw = width / 2
    hl = length / 2
    ht = thickness / 2
    
    # 计算翻盖的延伸方向 (垂直于铰链边缘)
    # 假设翻盖向下延伸 (初始闭合状态)
    if abs(dx) > 0.5:  # 铰链沿 X 轴
        extend_dir = np.array([0, 0, -1])
    else:  # 铰链沿 Y 轴
        extend_dir = np.array([0, 0, -1])
    
    # 翻盖中心位置
    center = np.array([ex, ey, ez]) + extend_dir * hl
    cx, cy, cz = center
    
    if flap_type == FlapType.TUCK:
        # 插舌形状: 顶部矩形 + 下部锥形
        taper_start = length * 0.6  # 锥形开始位置
        tip_width = width * 0.6  # 锥形末端宽度
        
        vertices = np.array([
            # 顶部矩形部分
            [cx - hw, cy - ht, ez],
            [cx + hw, cy - ht, ez],
            [cx + hw, cy + ht, ez],
            [cx - hw, cy + ht, ez],
            [cx - hw, cy - ht, ez - taper_start],
            [cx + hw, cy - ht, ez - taper_start],
            [cx + hw, cy + ht, ez - taper_start],
            [cx - hw, cy + ht, ez - taper_start],
            # 锥形末端
            [cx - tip_width/2, cy - ht, ez - length],
            [cx + tip_width/2, cy - ht, ez - length],
            [cx + tip_width/2, cy + ht, ez - length],
            [cx - tip_width/2, cy + ht, ez - length],
        ])
        
        faces = np.array([
            [0, 3, 2, 1],  # 顶面
            [4, 5, 6, 7],  # 中间面
            [8, 11, 10, 9],  # 底面
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
            [4, 8, 9, 5],
            [5, 9, 10, 6],
            [6, 10, 11, 7],
            [7, 11, 8, 4],
        ])
    elif flap_type == FlapType.LOCK:
        # 锁扣形状: 带凸起的矩形
        lock_height = length * 0.2
        lock_width = width * 0.4
        
        vertices = np.array([
            # 主体
            [cx - hw, cy - ht, ez],
            [cx + hw, cy - ht, ez],
            [cx + hw, cy + ht, ez],
            [cx - hw, cy + ht, ez],
            [cx - hw, cy - ht, ez - length],
            [cx + hw, cy - ht, ez - length],
            [cx + hw, cy + ht, ez - length],
            [cx - hw, cy + ht, ez - length],
            # 锁扣凸起
            [cx - lock_width/2, cy - ht - lock_height, ez - length * 0.3],
            [cx + lock_width/2, cy - ht - lock_height, ez - length * 0.3],
            [cx + lock_width/2, cy - ht - lock_height, ez - length * 0.7],
            [cx - lock_width/2, cy - ht - lock_height, ez - length * 0.7],
        ])
        
        faces = np.array([
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
            [1, 2, 6, 5],
            # 锁扣面
            [8, 9, 10, 11],
        ])
        
    else:
        # 简单矩形翻盖
        vertices = np.array([
            [cx - hw, cy - ht, ez],
            [cx + hw, cy - ht, ez],
            [cx + hw, cy + ht, ez],
            [cx - hw, cy + ht, ez],
            [cx - hw, cy - ht, ez - length],
            [cx + hw, cy - ht, ez - length],
            [cx + hw, cy + ht, ez - length],
            [cx - hw, cy + ht, ez - length],
        ])
        
        faces = np.array([
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
            [1, 2, 6, 5],
        ])
    
    # 铰链边缘信息
    hinge_edge = EdgeInfo(
        position=attach_edge.position,
        direction=attach_edge.direction,
        length=width,
        edge_type=EdgePosition.TOP,
    )
    
    # 末端边缘
    tip_edge = EdgeInfo(
        position=(cx, cy, ez - length),
        direction=attach_edge.direction,
        length=width if flap_type != FlapType.TUCK else tip_width,
        edge_type=EdgePosition.BOTTOM,
    )
    
    # 锁定特征 (仅锁扣类型)
    lock_feature = None
    if flap_type == FlapType.LOCK:
        lock_feature = {
            "type": "protrusion",
            "position": (cx, cy - ht - lock_height/2, ez - length * 0.5),
            "size": (lock_width, lock_height, length * 0.4),
        }
    
    return FlapGeometry(
        flap_type=flap_type,
        vertices=vertices,
        faces=faces,
        hinge_edge=hinge_edge,
        tip_edge=tip_edge,
        lock_feature=lock_feature,
    )


def create_fold_line(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    fold_angle: float = np.pi / 2,
    fold_direction: Tuple[float, float, float] = (0, 0, 1),
) -> Dict:
    """
    创建折痕线信息
    
    折痕线用于标识盒子折叠位置，是铰链的附着点。
    
    Args:
        start: 折痕线起点
        end: 折痕线终点
        fold_angle: 折叠角度 (弧度)
        fold_direction: 折叠方向 (法线方向)
    
    Returns:
        Dict: 折痕线信息
    """
    start = np.array(start)
    end = np.array(end)
    
    direction = end - start
    length = np.linalg.norm(direction)
    direction_normalized = direction / length if length > 0 else np.array([1, 0, 0])
    
    return {
        "start": start.tolist(),
        "end": end.tolist(),
        "center": ((start + end) / 2).tolist(),
        "direction": direction_normalized.tolist(),
        "length": float(length),
        "fold_angle": float(fold_angle),
        "fold_direction": list(fold_direction),
    }


def _create_rounded_box(
    width: float,
    depth: float,
    height: float,
    center: Tuple[float, float, float],
    corner_radius: float,
    segments: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建带圆角的长方体几何
    
    Args:
        width, depth, height: 尺寸
        center: 中心位置
        corner_radius: 圆角半径
        segments: 圆角分段数
    
    Returns:
        vertices, faces: 顶点和面数据
    """
    # 简化实现：如果圆角半径过小，返回普通长方体
    if corner_radius < 0.001:
        cx, cy, cz = center
        hw, hd, hh = width/2, depth/2, height/2
        vertices = np.array([
            [cx - hw, cy - hd, cz - hh],
            [cx + hw, cy - hd, cz - hh],
            [cx + hw, cy + hd, cz - hh],
            [cx - hw, cy + hd, cz - hh],
            [cx - hw, cy - hd, cz + hh],
            [cx + hw, cy - hd, cz + hh],
            [cx + hw, cy + hd, cz + hh],
            [cx - hw, cy + hd, cz + hh],
        ])
        faces = np.array([
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 4, 7, 3],
            [1, 2, 6, 5],
        ])
        return vertices, faces
    
    # 对于真正的圆角，使用 trimesh 或更复杂的算法
    # 这里仅返回简单长方体作为占位符
    cx, cy, cz = center
    hw, hd, hh = width/2, depth/2, height/2
    vertices = np.array([
        [cx - hw, cy - hd, cz - hh],
        [cx + hw, cy - hd, cz - hh],
        [cx + hw, cy + hd, cz - hh],
        [cx - hw, cy + hd, cz - hh],
        [cx - hw, cy - hd, cz + hh],
        [cx + hw, cy - hd, cz + hh],
        [cx + hw, cy + hd, cz + hh],
        [cx - hw, cy + hd, cz + hh],
    ])
    faces = np.array([
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 4, 7, 3],
        [1, 2, 6, 5],
    ])
    return vertices, faces


# ============================================================
# Blender Geometry Nodes 接口函数
# ============================================================

def create_panel_nodegroup_params(panel_geometry: PanelGeometry) -> Dict:
    """
    从 PanelGeometry 创建 Geometry Nodes 输入参数
    
    用于将程序化生成的几何转换为 Blender 节点组输入。
    
    Args:
        panel_geometry: 面板几何数据
    
    Returns:
        Dict: 节点组输入参数
    """
    w, h, t = panel_geometry.dimensions
    cx, cy, cz = panel_geometry.center
    
    return {
        "Width": w,
        "Height": h,
        "Thickness": t,
        "Position X": cx,
        "Position Y": cy,
        "Position Z": cz,
        "Panel Type": panel_geometry.panel_type.value,
    }


def calculate_hinge_position(
    parent_panel: PanelGeometry,
    child_panel: PanelGeometry,
    edge_type: EdgePosition,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    计算铰链精确位置
    
    基于两个面板的边缘信息计算铰链的位置和轴向。
    
    Args:
        parent_panel: 父面板几何
        child_panel: 子面板几何
        edge_type: 边缘类型
    
    Returns:
        position: 铰链中心位置 (x, y, z)
        axis: 旋转轴方向 (ax, ay, az)
    """
    parent_edge = parent_panel.get_edge(edge_type)
    
    if parent_edge is None:
        # 如果父面板没有对应边缘，使用子面板边缘
        child_edge = child_panel.get_edge(EdgePosition.BOTTOM)
        if child_edge:
            return child_edge.position, child_edge.direction
        else:
            # 默认返回父面板中心和 X 轴
            return parent_panel.center, (1, 0, 0)
    
    return parent_edge.position, parent_edge.get_hinge_axis()


def validate_panel_geometry(panel: PanelGeometry) -> Tuple[bool, str]:
    """
    验证面板几何的有效性
    
    Args:
        panel: 面板几何数据
    
    Returns:
        (is_valid, message): 验证结果和消息
    """
    errors = []
    
    # 检查顶点数量
    if len(panel.vertices) < 4:
        errors.append("顶点数量不足")
    
    # 检查面数量
    if len(panel.faces) < 1:
        errors.append("面数量不足")
    
    # 检查尺寸
    w, h, t = panel.dimensions
    if w <= 0 or h <= 0 or t <= 0:
        errors.append("尺寸必须为正数")
    
    # 检查法线
    normal = np.array(panel.normal)
    if np.linalg.norm(normal) < 0.9:
        errors.append("法线向量无效")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, "面板几何有效"


def merge_panels_to_mesh(panels: List[PanelGeometry]) -> Tuple[np.ndarray, np.ndarray]:
    """
    合并多个面板为单个网格
    
    Args:
        panels: 面板列表
    
    Returns:
        vertices, faces: 合并后的顶点和面
    """
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    for panel in panels:
        all_vertices.append(panel.vertices)
        all_faces.append(panel.faces + vertex_offset)
        vertex_offset += len(panel.vertices)
    
    if not all_vertices:
        return np.array([]), np.array([])
    
    return np.vstack(all_vertices), np.vstack(all_faces)
