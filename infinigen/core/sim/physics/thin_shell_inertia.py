# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""
薄壳惯性修正模块 (R-Deep-1)

问题: 薄壁物体（如盒子盖子、卡纸面板）使用体积法计算惯性时，
     由于体积接近 0，导致惯性张量过小，在 PyBullet/MuJoCo 中引发数值不稳定。

解决方案: 使用基于表面积的惯性估算，假设最小有效厚度
"""

import logging
import numpy as np
from typing import Tuple, Optional

# 最小有效厚度 (米) - 用于防止惯性张量过小
MIN_EFFECTIVE_THICKNESS = 0.002  # 2mm

# 最小惯性值 (kg·m²) - 防止数值不稳定
MIN_INERTIA_VALUE = 1e-8

# 薄壳检测阈值: 体积/表面积比
# 对于厚度为 t 的正方形薄板 (边长 L >> t):
#   体积 ≈ L² × t
#   表面积 ≈ 2L²
#   比值 ≈ t/2
# 所以阈值 0.002 对应约 4mm 厚的薄板
THIN_SHELL_THRESHOLD = 0.002  # 对应约 4mm 厚度

# 最小尺寸阈值: 避免小物体误判
MIN_SIZE_FOR_THIN_SHELL_CHECK = 0.01  # 最小边长 1cm


def is_thin_shell(volume: float, surface_area: float) -> bool:
    """
    判断几何体是否为薄壳
    
    薄壳定义: 
    1. 体积/表面积 比值小于阈值
    2. 表面积足够大 (避免小物体误判)
    
    对于厚度为 t 的薄板，该比值约等于 t/2
    
    Args:
        volume: 几何体体积 (m³)
        surface_area: 几何体表面积 (m²)
        
    Returns:
        True 如果是薄壳几何体
    """
    if surface_area <= 0:
        return False
    
    # 避免小物体误判: 表面积至少需要相当于 2cm x 2cm 的面积
    if surface_area < 0.001:  # 约 3cm x 3cm 正方形的一面
        return False
    
    ratio = volume / surface_area
    return ratio < THIN_SHELL_THRESHOLD


def calculate_robust_inertia(
    vertices: np.ndarray,
    faces: np.ndarray,
    density: float,
    volume: Optional[float] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    计算稳健的质量和惯性张量，对薄壳物体进行特殊处理
    
    Args:
        vertices: 顶点坐标 (N, 3)
        faces: 面索引 (M, 3)
        density: 材料密度 (kg/m³)
        volume: 预计算的体积 (可选)
        
    Returns:
        mass: 质量 (kg)
        inertia_tensor: 惯性张量 (3, 3) (kg·m²)
        center_of_mass: 质心坐标 (3,)
    """
    import trimesh
    
    # 创建 trimesh 对象
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # 获取几何属性
    if volume is None:
        volume = mesh.volume
    surface_area = mesh.area
    
    # 检查是否为薄壳
    if is_thin_shell(volume, surface_area):
        logging.debug(f"检测到薄壳几何体: volume={volume:.6e}, area={surface_area:.4f}")
        mass, inertia_tensor, com = _calculate_thin_shell_inertia(
            mesh, density, surface_area
        )
    else:
        # 标准体积法
        mass = density * volume
        mesh.density = density
        inertia_tensor = mesh.moment_inertia
        com = mesh.center_mass
    
    # 确保惯性张量正定 (数值稳定性)
    inertia_tensor = _ensure_positive_definite(inertia_tensor)
    
    # 确保最小质量
    mass = max(mass, 0.001)  # 最小 1g
    
    return mass, inertia_tensor, com


def _calculate_thin_shell_inertia(
    mesh, density: float, surface_area: float
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    计算薄壳物体的惯性 (基于表面积)
    
    使用等效薄板模型:
    - 质量 = 密度 × 表面积 × 有效厚度
    - 惯性 = 基于边界框的薄板惯性公式
    """
    # 使用最小有效厚度计算质量
    effective_thickness = MIN_EFFECTIVE_THICKNESS
    mass = density * surface_area * effective_thickness
    
    # 获取边界框尺寸
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]  # [width, height, depth]
    a, b, c = extents
    
    # 对薄壳使用薄板惯性公式 (假设质量均匀分布在表面)
    # 对于一个薄板，主惯性矩:
    # Ixx = (1/12) * m * (b² + c²)
    # Iyy = (1/12) * m * (a² + c²)
    # Izz = (1/12) * m * (a² + b²)
    
    Ixx = (1.0 / 12.0) * mass * (b**2 + c**2)
    Iyy = (1.0 / 12.0) * mass * (a**2 + c**2)
    Izz = (1.0 / 12.0) * mass * (a**2 + b**2)
    
    # 确保最小惯性值
    Ixx = max(Ixx, MIN_INERTIA_VALUE)
    Iyy = max(Iyy, MIN_INERTIA_VALUE)
    Izz = max(Izz, MIN_INERTIA_VALUE)
    
    inertia_tensor = np.diag([Ixx, Iyy, Izz])
    
    # 质心为几何中心
    com = mesh.centroid
    
    logging.debug(f"薄壳惯性: mass={mass:.4f} kg, I_diag={np.diag(inertia_tensor)}")
    
    return mass, inertia_tensor, com


def _ensure_positive_definite(inertia: np.ndarray) -> np.ndarray:
    """
    确保惯性张量是正定矩阵
    
    通过特征值分解，将任何负或过小的特征值替换为最小值
    """
    # 对角元素应该为正
    inertia = np.clip(inertia, a_min=0, a_max=None)
    
    # 确保对角线最小值
    for i in range(3):
        if inertia[i, i] < MIN_INERTIA_VALUE:
            inertia[i, i] = MIN_INERTIA_VALUE
    
    # 检查正定性 (所有特征值 > 0)
    try:
        eigenvalues = np.linalg.eigvalsh(inertia)
        if np.any(eigenvalues <= 0):
            # 修正: 将非正定矩阵转换为正定
            eigenvalues = np.maximum(eigenvalues, MIN_INERTIA_VALUE)
            # 重构对角惯性张量 (忽略非对角项)
            inertia = np.diag([max(inertia[i, i], MIN_INERTIA_VALUE) for i in range(3)])
    except np.linalg.LinAlgError:
        # 如果特征值分解失败，使用安全的对角矩阵
        inertia = np.diag([max(inertia[i, i], MIN_INERTIA_VALUE) for i in range(3)])
    
    return inertia


def validate_inertia_for_simulation(
    mass: float, inertia_tensor: np.ndarray
) -> Tuple[bool, str]:
    """
    验证质量和惯性张量是否适合物理仿真
    
    Returns:
        (is_valid, message)
    """
    issues = []
    
    # 检查质量
    if mass <= 0:
        issues.append(f"质量无效: {mass}")
    elif mass < 0.001:
        issues.append(f"质量过小 ({mass:.6f} kg)，可能导致数值不稳定")
    
    # 检查惯性张量对角元素
    diag = np.diag(inertia_tensor)
    for i, val in enumerate(diag):
        if val <= 0:
            issues.append(f"惯性张量 I[{i},{i}] 非正: {val}")
        elif val < MIN_INERTIA_VALUE:
            issues.append(f"惯性张量 I[{i},{i}] 过小: {val:.2e}")
    
    # 检查正定性
    try:
        eigenvalues = np.linalg.eigvalsh(inertia_tensor)
        if np.any(eigenvalues <= 0):
            issues.append(f"惯性张量非正定: eigenvalues={eigenvalues}")
    except Exception:
        issues.append("无法检查惯性张量正定性")
    
    # 检查物理合理性: 最大惯性不应超过最小惯性的 1000 倍
    if diag.min() > 0:
        ratio = diag.max() / diag.min()
        if ratio > 1000:
            issues.append(f"惯性比例过大 ({ratio:.0f}:1)，可能是几何问题")
    
    if issues:
        return False, "; ".join(issues)
    return True, "惯性参数有效"


# 便捷函数：直接从 Blender mesh 计算
def calculate_robust_inertia_from_bpy_mesh(
    bpy_mesh,
    density: float,
    volume: Optional[float] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    从 Blender mesh 对象计算稳健惯性
    
    Args:
        bpy_mesh: Blender mesh 对象 (bpy.types.Object)
        density: 材料密度 (kg/m³)
        volume: 预计算的体积 (可选)
        
    Returns:
        mass, inertia_tensor, center_of_mass
    """
    # 提取顶点和面
    vertices = np.array([list(v.co) for v in bpy_mesh.data.vertices])
    faces = np.array([list(f.vertices) for f in bpy_mesh.data.loop_triangles])
    
    return calculate_robust_inertia(vertices, faces, density, volume)
