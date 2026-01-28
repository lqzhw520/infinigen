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


def combine_multiple_inertias(
    masses: list,
    centers_of_mass: list,
    inertia_tensors: list,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    使用平行轴定理 (Parallel Axis Theorem) 合并多个刚体的惯性属性。
    
    这是解决 URDF 中一个 link 只能有一个 <inertial> 元素的标准方法。
    
    数学原理:
    =========
    
    1. 总质量:
       M = Σ m_i
    
    2. 合并质心 (质量加权平均):
       C = (1/M) × Σ (m_i × c_i)
    
    3. 平行轴定理 (Parallel Axis Theorem):
       当惯性张量的参考点从质心 c_i 移动到合并质心 C 时:
       
       I_new = I_old + m × [(r·r)×E - r⊗r]
       
       其中:
       - r = c_i - C (质心偏移向量)
       - E 是 3×3 单位矩阵
       - ⊗ 表示外积 (outer product)
       - (r·r) 是 r 与自身的点积
    
    展开为分量形式:
       I_xx^(new) = I_xx^(old) + m × (r_y² + r_z²)
       I_yy^(new) = I_yy^(old) + m × (r_x² + r_z²)
       I_zz^(new) = I_zz^(old) + m × (r_x² + r_y²)
       I_xy^(new) = I_xy^(old) - m × r_x × r_y
       I_xz^(new) = I_xz^(old) - m × r_x × r_z
       I_yz^(new) = I_yz^(old) - m × r_y × r_z
    
    Args:
        masses: 各刚体的质量列表 (长度为 n)
        centers_of_mass: 各刚体的质心位置列表 (每个为 3D 向量, shape=(n,3))
        inertia_tensors: 各刚体的惯性张量列表 (每个为 3x3 矩阵)
        
    Returns:
        total_mass: 合并后的总质量 (kg)
        combined_com: 合并后的质心位置 (3D 向量)
        combined_inertia: 合并后的惯性张量 (3x3 矩阵, kg·m²)
        
    References:
        [1] Goldstein, H. (1980). Classical Mechanics (2nd ed.). Addison-Wesley.
            Chapter 5: The Rigid Body Equations of Motion.
        [2] https://en.wikipedia.org/wiki/Parallel_axis_theorem
    """
    n = len(masses)
    
    # 边界检查
    if n == 0:
        logging.warning("combine_multiple_inertias: 空输入，返回最小惯性")
        return 0.001, np.zeros(3), np.eye(3) * MIN_INERTIA_VALUE
    
    if n == 1:
        # 单个刚体，无需合并
        return masses[0], np.asarray(centers_of_mass[0]), np.asarray(inertia_tensors[0])
    
    assert len(centers_of_mass) == n, f"质心数量不匹配: {len(centers_of_mass)} != {n}"
    assert len(inertia_tensors) == n, f"惯性张量数量不匹配: {len(inertia_tensors)} != {n}"
    
    # 转换为 numpy 数组
    masses_arr = np.array(masses, dtype=np.float64)
    coms_arr = np.array(centers_of_mass, dtype=np.float64)  # shape: (n, 3)
    
    # ============================================================
    # Step 1: 计算总质量
    # ============================================================
    total_mass = np.sum(masses_arr)
    
    if total_mass <= 0:
        logging.warning("combine_multiple_inertias: 总质量 <= 0，返回最小惯性")
        return 0.001, np.zeros(3), np.eye(3) * MIN_INERTIA_VALUE
    
    # ============================================================
    # Step 2: 计算合并后的质心 (质量加权平均)
    #         C = (1/M) × Σ (m_i × c_i)
    # ============================================================
    combined_com = np.sum(masses_arr[:, np.newaxis] * coms_arr, axis=0) / total_mass
    
    # ============================================================
    # Step 3: 使用平行轴定理合并惯性张量
    # ============================================================
    combined_inertia = np.zeros((3, 3), dtype=np.float64)
    
    for i in range(n):
        m_i = masses_arr[i]
        I_i = np.asarray(inertia_tensors[i], dtype=np.float64)
        c_i = coms_arr[i]
        
        # 计算质心偏移向量: r = c_i - C
        r = c_i - combined_com
        
        # 平行轴定理: I_new = I_old + m × [(r·r)×E - r⊗r]
        # 其中:
        #   (r·r) = r_x² + r_y² + r_z² (标量)
        #   E = 3×3 单位矩阵
        #   r⊗r = 外积矩阵 (3×3)
        
        r_dot_r = np.dot(r, r)  # 标量: r_x² + r_y² + r_z²
        r_outer_r = np.outer(r, r)  # 3×3 矩阵: r ⊗ r
        
        # 平行轴偏移贡献: m × [(r·r)×E - r⊗r]
        parallel_axis_term = m_i * (r_dot_r * np.eye(3) - r_outer_r)
        
        # 累加: I_total = Σ (I_i + parallel_axis_term_i)
        combined_inertia += I_i + parallel_axis_term
    
    # ============================================================
    # Step 4: 确保惯性张量正定 (数值稳定性)
    # ============================================================
    combined_inertia = _ensure_positive_definite(combined_inertia)
    
    # 确保最小质量
    total_mass = max(total_mass, 0.001)
    
    logging.debug(
        f"combine_multiple_inertias: n={n}, total_mass={total_mass:.4f} kg, "
        f"combined_com={combined_com}, I_diag={np.diag(combined_inertia)}"
    )
    
    return total_mass, combined_com, combined_inertia


def verify_combined_inertia(
    masses: list,
    centers_of_mass: list,
    inertia_tensors: list,
    combined_mass: float,
    combined_com: np.ndarray,
    combined_inertia: np.ndarray,
    tolerance: float = 1e-6,
) -> Tuple[bool, str]:
    """
    验证惯性合并的正确性。
    
    检查项:
    1. 总质量应等于各部分质量之和
    2. 合并质心应等于质量加权平均
    3. 惯性张量应正定
    4. 惯性张量的迹应满足物理约束
    
    Args:
        masses: 原始质量列表
        centers_of_mass: 原始质心列表
        inertia_tensors: 原始惯性张量列表
        combined_mass: 合并后的质量
        combined_com: 合并后的质心
        combined_inertia: 合并后的惯性张量
        tolerance: 数值容差
        
    Returns:
        (is_valid, message): 验证结果和消息
    """
    issues = []
    
    # 1. 验证总质量
    expected_mass = sum(masses)
    if abs(combined_mass - expected_mass) > tolerance and expected_mass > tolerance:
        issues.append(f"质量不匹配: {combined_mass:.6f} vs expected {expected_mass:.6f}")
    
    # 2. 验证质心
    if expected_mass > tolerance:
        expected_com = np.sum(
            [m * np.array(c) for m, c in zip(masses, centers_of_mass)], axis=0
        ) / expected_mass
        com_error = np.linalg.norm(combined_com - expected_com)
        if com_error > tolerance:
            issues.append(f"质心不匹配: 误差={com_error:.6e}")
    
    # 3. 验证惯性张量正定性
    try:
        eigenvalues = np.linalg.eigvalsh(combined_inertia)
        if np.any(eigenvalues <= 0):
            issues.append(f"惯性张量非正定: eigenvalues={eigenvalues}")
    except np.linalg.LinAlgError:
        issues.append("无法计算惯性张量特征值")
    
    # 4. 验证惯性张量对角元素为正
    diag = np.diag(combined_inertia)
    if np.any(diag <= 0):
        issues.append(f"惯性张量对角元素非正: {diag}")
    
    # 5. 验证惯性张量满足三角不等式 (物理约束)
    # 对于任意刚体: I_xx + I_yy >= I_zz (及其循环)
    if len(diag) == 3:
        if diag[0] + diag[1] < diag[2] - tolerance:
            issues.append(f"惯性张量违反三角不等式: I_xx+I_yy < I_zz")
        if diag[0] + diag[2] < diag[1] - tolerance:
            issues.append(f"惯性张量违反三角不等式: I_xx+I_zz < I_yy")
        if diag[1] + diag[2] < diag[0] - tolerance:
            issues.append(f"惯性张量违反三角不等式: I_yy+I_zz < I_xx")
    
    if issues:
        return False, "; ".join(issues)
    return True, "惯性合并验证通过"
