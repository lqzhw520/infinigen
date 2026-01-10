# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""
碰撞网格简化模块 (R7)

功能:
1. 简化碰撞网格以提高仿真性能
2. 生成盒子基元碰撞体作为替代
3. 凸分解用于复杂非凸几何体
"""

import logging
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

# 默认碰撞网格简化参数
DEFAULT_TARGET_FACE_COUNT = 500  # 目标面数
DEFAULT_DECIMATION_RATIO = 0.1   # 简化比例 (保留 10% 的面)
MAX_COLLISION_FACES = 1000       # 碰撞网格最大面数


def simplify_collision_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: Optional[int] = None,
    decimation_ratio: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    简化碰撞网格
    
    Args:
        vertices: 顶点坐标 (N, 3)
        faces: 面索引 (M, 3)
        target_faces: 目标面数 (优先于 decimation_ratio)
        decimation_ratio: 简化比例 (0-1, 保留的面比例)
        
    Returns:
        simplified_vertices, simplified_faces
    """
    import trimesh
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    original_face_count = len(mesh.faces)
    
    # 如果已经足够简单，直接返回
    if original_face_count <= DEFAULT_TARGET_FACE_COUNT:
        logger.debug(f"网格已经足够简单 ({original_face_count} faces), 跳过简化")
        return vertices, faces
    
    # 确定目标面数
    if target_faces is not None:
        target = target_faces
    elif decimation_ratio is not None:
        target = int(original_face_count * decimation_ratio)
    else:
        target = DEFAULT_TARGET_FACE_COUNT
    
    # 确保目标合理
    target = max(min(target, original_face_count), 4)  # 至少保留 4 个面
    
    # 使用 trimesh 的简化功能
    # 尝试多种简化方法
    
    # 方法 1: 尝试 quadric decimation (需要 open3d 或特定后端)
    try:
        simplified = mesh.simplify_quadric_decimation(target)
        if len(simplified.faces) < original_face_count:
            logger.debug(f"网格简化 (quadric): {original_face_count} -> {len(simplified.faces)} faces")
            return np.array(simplified.vertices), np.array(simplified.faces)
    except Exception as e:
        logger.debug(f"Quadric decimation 不可用: {e}")
    
    # 方法 2: 使用凸包作为简化替代 (对于大多数盒子部件有效)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(vertices)
        hull_vertices = vertices[hull.vertices]
        # 重建面索引
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(hull.vertices)}
        hull_faces = np.array([
            [vertex_map[v] for v in simplex]
            for simplex in hull.simplices
        ])
        if len(hull_faces) < original_face_count:
            logger.debug(f"网格简化 (convex hull): {original_face_count} -> {len(hull_faces)} faces")
            return hull_vertices, hull_faces
    except Exception as e:
        logger.debug(f"凸包简化失败: {e}")
    
    # 方法 3: 顶点采样 + 凸包
    try:
        from scipy.spatial import ConvexHull
        # 简单的顶点采样
        step = max(1, len(vertices) // min(target * 3, 600))
        sampled_indices = np.arange(0, len(vertices), step)
        if len(sampled_indices) >= 4:
            sampled_verts = vertices[sampled_indices]
            hull = ConvexHull(sampled_verts)
            result_faces = len(hull.simplices)
            if result_faces < original_face_count:
                logger.debug(f"网格简化 (sampling + hull): {original_face_count} -> {result_faces} faces")
                return sampled_verts[hull.vertices], np.array(hull.simplices)
    except Exception as e:
        logger.debug(f"采样简化失败: {e}")
    
    # 如果所有方法都失败，返回原始网格
    logger.warning(f"所有简化方法都失败或无效, 返回原始网格")
    return vertices, faces


def generate_box_primitive_collider(
    vertices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成盒子基元碰撞体 (AABB)
    
    这是最简单的碰撞体，适用于大致呈盒子形状的几何体
    
    Args:
        vertices: 顶点坐标 (N, 3)
        
    Returns:
        center: 盒子中心 (3,)
        extents: 盒子尺寸 (3,) [width, height, depth]
        box_vertices: 盒子顶点 (8, 3)
    """
    # 计算 AABB
    min_corner = np.min(vertices, axis=0)
    max_corner = np.max(vertices, axis=0)
    
    center = (min_corner + max_corner) / 2
    extents = max_corner - min_corner
    
    # 生成盒子顶点
    import trimesh
    box = trimesh.creation.box(extents=extents)
    box_vertices = np.array(box.vertices) + center
    
    return center, extents, box_vertices


def generate_convex_hull_collider(
    vertices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成凸包碰撞体
    
    Args:
        vertices: 顶点坐标 (N, 3)
        
    Returns:
        hull_vertices, hull_faces
    """
    import trimesh
    from scipy.spatial import ConvexHull
    
    try:
        hull = ConvexHull(vertices)
        hull_vertices = vertices[hull.vertices]
        
        # 重建面索引
        # ConvexHull 的 simplices 是相对于原始顶点的
        # 需要映射到新的顶点索引
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(hull.vertices)}
        hull_faces = np.array([
            [vertex_map[v] for v in simplex]
            for simplex in hull.simplices
        ])
        
        logger.debug(f"凸包生成: {len(vertices)} -> {len(hull_vertices)} vertices, {len(hull_faces)} faces")
        return hull_vertices, hull_faces
        
    except Exception as e:
        logger.warning(f"凸包生成失败: {e}, 返回简化网格")
        return simplify_collision_mesh(vertices, np.array([]))


def convex_decomposition(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_parts: int = 8,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    凸分解: 将非凸几何体分解为多个凸部件
    
    注意: 这是一个简化实现，使用 trimesh 的 convex_decomposition
          完整实现需要 VHACD 或类似算法
    
    Args:
        vertices: 顶点坐标 (N, 3)
        faces: 面索引 (M, 3)
        max_parts: 最大分解部件数
        
    Returns:
        List of (vertices, faces) tuples for each convex part
    """
    import trimesh
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # 检查是否已经是凸的
    if mesh.is_convex:
        logger.debug("网格已经是凸的，无需分解")
        return [(vertices, faces)]
    
    try:
        # 尝试使用 trimesh 的凸分解
        # 注意: 这需要安装 VHACD
        parts = mesh.convex_decomposition(maxhulls=max_parts)
        
        result = []
        for part in parts:
            result.append((np.array(part.vertices), np.array(part.faces)))
        
        logger.debug(f"凸分解: 分解为 {len(result)} 个部件")
        return result
        
    except Exception as e:
        logger.warning(f"凸分解失败 ({e}), 使用凸包代替")
        hull_v, hull_f = generate_convex_hull_collider(vertices)
        return [(hull_v, hull_f)]


def select_optimal_collider(
    vertices: np.ndarray,
    faces: np.ndarray,
    prefer_primitive: bool = True,
) -> dict:
    """
    选择最优碰撞体类型
    
    Args:
        vertices: 顶点坐标
        faces: 面索引
        prefer_primitive: 是否优先使用基元碰撞体
        
    Returns:
        dict with keys:
            - type: "box", "convex", "mesh"
            - data: 对应的碰撞体数据
    """
    import trimesh
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    
    # 计算体积比 (实际体积 / 包围盒体积)
    box_volume = np.prod(extents)
    mesh_volume = abs(mesh.volume) if mesh.is_watertight else box_volume * 0.5
    
    volume_ratio = mesh_volume / box_volume if box_volume > 0 else 0
    
    # 决策逻辑
    if prefer_primitive and volume_ratio > 0.7:
        # 体积比 > 70% 说明几何体接近盒子形状
        center, ext, box_verts = generate_box_primitive_collider(vertices)
        return {
            "type": "box",
            "center": center,
            "extents": ext,
            "volume_ratio": volume_ratio,
        }
    elif mesh.is_convex:
        return {
            "type": "convex",
            "vertices": vertices,
            "faces": faces,
            "volume_ratio": volume_ratio,
        }
    else:
        # 简化网格用于碰撞
        simp_v, simp_f = simplify_collision_mesh(vertices, faces)
        return {
            "type": "mesh",
            "vertices": simp_v,
            "faces": simp_f,
            "original_faces": len(faces),
            "simplified_faces": len(simp_f),
        }


def export_collision_mesh(
    collider_info: dict,
    output_path: Path,
    format: str = "obj",
) -> Path:
    """
    导出碰撞网格到文件
    
    Args:
        collider_info: select_optimal_collider 的返回值
        output_path: 输出路径
        format: 文件格式 ("obj", "stl")
        
    Returns:
        导出的文件路径
    """
    import trimesh
    
    collider_type = collider_info["type"]
    
    if collider_type == "box":
        # 生成盒子网格
        box = trimesh.creation.box(extents=collider_info["extents"])
        box.apply_translation(collider_info["center"])
        mesh = box
    elif collider_type in ["convex", "mesh"]:
        mesh = trimesh.Trimesh(
            vertices=collider_info["vertices"],
            faces=collider_info["faces"]
        )
    else:
        raise ValueError(f"Unknown collider type: {collider_type}")
    
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(f".{format}")
    
    mesh.export(str(output_path))
    logger.debug(f"碰撞网格导出: {output_path}")
    
    return output_path
