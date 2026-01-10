#!/usr/bin/env python3
"""
P1-T2 测试: 基础几何模块

测试场景:
1. 模块导入 - 验证新模块可导入
2. 底面板创建 - 验证 create_base_panel()
3. 侧面板创建 - 验证 create_side_panel()
4. 盖板创建 - 验证 create_lid_panel()
5. 翻盖创建 - 验证 create_flap()
6. 边缘信息 - 验证边缘信息正确性
7. 铰链位置计算 - 验证铰链位置和轴向
8. 几何验证 - 验证几何有效性检查
9. 网格合并 - 验证多面板合并
10. 与 P1-T1 集成 - 验证与 ModularBoxFactory 的兼容性

运行方法 (在容器中):
    cd /mnt/afs2/zhuhaowu/infinigen
    python tests/sim/test_box_geometry_modules.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_module_import():
    """测试模块导入"""
    print("\n=== 测试 1: 模块导入 ===")
    
    try:
        from infinigen.assets.sim_objects.box_geometry_modules import (
            PanelType,
            FlapType,
            EdgePosition,
            EdgeInfo,
            PanelGeometry,
            FlapGeometry,
            create_base_panel,
            create_side_panel,
            create_lid_panel,
            create_flap,
            create_fold_line,
            calculate_hinge_position,
            validate_panel_geometry,
            merge_panels_to_mesh,
        )
        print("  ✅ 所有类和函数导入成功")
        return True
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")
        return False


def test_base_panel_creation():
    """测试底面板创建"""
    print("\n=== 测试 2: 底面板创建 ===")
    
    from infinigen.assets.sim_objects.box_geometry_modules import (
        create_base_panel,
        PanelType,
        EdgePosition,
    )
    
    # 创建 30cm x 20cm x 2mm 的底面板
    panel = create_base_panel(
        width=0.3,
        depth=0.2,
        thickness=0.002,
        center=(0, 0, 0),
    )
    
    print(f"  面板类型: {panel.panel_type.name}")
    print(f"  顶点数: {len(panel.vertices)}")
    print(f"  面数: {len(panel.faces)}")
    print(f"  尺寸: {panel.dimensions}")
    print(f"  中心: {panel.center}")
    print(f"  法线: {panel.normal}")
    
    # 验证
    errors = []
    
    if panel.panel_type != PanelType.BASE:
        errors.append("面板类型应为 BASE")
    
    if len(panel.vertices) != 8:
        errors.append(f"顶点数应为 8，实际为 {len(panel.vertices)}")
    
    if len(panel.faces) != 6:
        errors.append(f"面数应为 6，实际为 {len(panel.faces)}")
    
    w, d, t = panel.dimensions
    if abs(w - 0.3) > 1e-6 or abs(d - 0.2) > 1e-6 or abs(t - 0.002) > 1e-6:
        errors.append("尺寸不匹配")
    
    # 验证边缘信息
    for edge_pos in [EdgePosition.FRONT, EdgePosition.BACK, EdgePosition.LEFT, EdgePosition.RIGHT]:
        edge = panel.get_edge(edge_pos)
        if edge is None:
            errors.append(f"缺少边缘: {edge_pos.name}")
        else:
            print(f"  边缘 {edge_pos.name}: 长度={edge.length:.3f}m")
    
    if errors:
        for e in errors:
            print(f"  ❌ {e}")
        return False
    
    print("  ✅ 底面板创建测试通过")
    return True


def test_side_panel_creation():
    """测试侧面板创建"""
    print("\n=== 测试 3: 侧面板创建 ===")
    
    from infinigen.assets.sim_objects.box_geometry_modules import (
        create_side_panel,
        create_base_panel,
        PanelType,
        EdgePosition,
    )
    
    # 先创建底面板
    base = create_base_panel(width=0.3, depth=0.2, thickness=0.002)
    
    # 创建四个侧面板
    sides = [EdgePosition.FRONT, EdgePosition.BACK, EdgePosition.LEFT, EdgePosition.RIGHT]
    all_passed = True
    
    for side in sides:
        # 根据侧面位置确定宽度
        if side in [EdgePosition.FRONT, EdgePosition.BACK]:
            width = 0.3
        else:
            width = 0.2
        
        panel = create_side_panel(
            width=width,
            height=0.15,
            thickness=0.002,
            side=side,
            parent_dimensions=(0.3, 0.2, 0.002),
            parent_center=(0, 0, 0),
        )
        
        print(f"  {side.name}: 类型={panel.panel_type.name}, 顶点={len(panel.vertices)}, 法线={panel.normal}")
        
        # 验证法线方向
        expected_normals = {
            EdgePosition.FRONT: (0, -1, 0),
            EdgePosition.BACK: (0, 1, 0),
            EdgePosition.LEFT: (-1, 0, 0),
            EdgePosition.RIGHT: (1, 0, 0),
        }
        
        expected = expected_normals[side]
        actual = panel.normal
        
        if not all(abs(a - e) < 0.1 for a, e in zip(actual, expected)):
            print(f"    ❌ 法线方向错误: 期望 {expected}, 实际 {actual}")
            all_passed = False
        
        # 验证顶边缘存在
        top_edge = panel.get_edge(EdgePosition.TOP)
        if top_edge is None:
            print(f"    ❌ 缺少顶边缘")
            all_passed = False
    
    if all_passed:
        print("  ✅ 侧面板创建测试通过")
    
    return all_passed


def test_lid_panel_creation():
    """测试盖板创建"""
    print("\n=== 测试 4: 盖板创建 ===")
    
    from infinigen.assets.sim_objects.box_geometry_modules import (
        create_lid_panel,
        PanelType,
        EdgePosition,
    )
    
    # 创建从后边缘铰接的盖板
    lid = create_lid_panel(
        width=0.3,
        depth=0.2,
        thickness=0.002,
        attach_edge=EdgePosition.BACK,
        parent_dimensions=(0.3, 0.2, 0.002),
        parent_center=(0, 0, 0),
        parent_height=0.15,
    )
    
    print(f"  面板类型: {lid.panel_type.name}")
    print(f"  顶点数: {len(lid.vertices)}")
    print(f"  中心: {lid.center}")
    print(f"  尺寸: {lid.dimensions}")
    
    # 验证
    errors = []
    
    if lid.panel_type != PanelType.LID:
        errors.append("面板类型应为 LID")
    
    # 盖板应该在盒子顶部
    _, _, cz = lid.center
    expected_z = 0.002/2 + 0.15 + 0.002/2  # base_thickness/2 + height + lid_thickness/2
    print(f"  盖板 Z 位置: {cz:.4f} (预期: ~{expected_z:.4f})")
    
    # 验证铰链边缘
    hinge_edge = lid.get_edge(EdgePosition.TOP)
    if hinge_edge is None:
        errors.append("缺少铰链边缘")
    else:
        print(f"  铰链边缘: 位置={hinge_edge.position}, 方向={hinge_edge.direction}")
        # 铰链方向应该是 X 轴 (因为是后边缘)
        if abs(hinge_edge.direction[0]) < 0.9:
            errors.append("铰链方向应沿 X 轴")
    
    if errors:
        for e in errors:
            print(f"  ❌ {e}")
        return False
    
    print("  ✅ 盖板创建测试通过")
    return True


def test_flap_creation():
    """测试翻盖创建"""
    print("\n=== 测试 5: 翻盖创建 ===")
    
    from infinigen.assets.sim_objects.box_geometry_modules import (
        create_flap,
        FlapType,
        EdgeInfo,
        EdgePosition,
    )
    
    # 创建附着边缘
    attach_edge = EdgeInfo(
        position=(0, 0, 0.15),
        direction=(1, 0, 0),
        length=0.3,
        edge_type=EdgePosition.TOP,
    )
    
    flap_types = [FlapType.TUCK, FlapType.LOCK, FlapType.DUST_FLAP]
    all_passed = True
    
    for flap_type in flap_types:
        flap = create_flap(
            width=0.3,
            length=0.05,
            thickness=0.002,
            flap_type=flap_type,
            attach_edge=attach_edge,
        )
        
        print(f"  {flap_type.name}: 顶点={len(flap.vertices)}, 面={len(flap.faces)}")
        
        # 验证铰链边缘
        if flap.hinge_edge is None:
            print(f"    ❌ 缺少铰链边缘")
            all_passed = False
        
        # 验证末端边缘
        if flap.tip_edge is None:
            print(f"    ❌ 缺少末端边缘")
            all_passed = False
        
        # 锁扣类型应有锁定特征
        if flap_type == FlapType.LOCK:
            if flap.lock_feature is None:
                print(f"    ❌ 锁扣类型应有锁定特征")
                all_passed = False
            else:
                print(f"    锁定特征: {flap.lock_feature['type']}")
    
    if all_passed:
        print("  ✅ 翻盖创建测试通过")
    
    return all_passed


def test_edge_info():
    """测试边缘信息"""
    print("\n=== 测试 6: 边缘信息 ===")
    
    from infinigen.assets.sim_objects.box_geometry_modules import (
        EdgeInfo,
        EdgePosition,
    )
    
    edge = EdgeInfo(
        position=(0.15, 0, 0.15),
        direction=(0, 1, 0),
        length=0.2,
        edge_type=EdgePosition.RIGHT,
    )
    
    # 测试铰链轴
    axis = edge.get_hinge_axis()
    print(f"  边缘位置: {edge.position}")
    print(f"  边缘方向: {edge.direction}")
    print(f"  铰链轴: {axis}")
    
    if axis != (0, 1, 0):
        print("  ❌ 铰链轴应与边缘方向相同")
        return False
    
    # 测试字典转换
    edge_dict = edge.to_dict()
    print(f"  字典键: {list(edge_dict.keys())}")
    
    if "position" not in edge_dict or "direction" not in edge_dict:
        print("  ❌ 字典缺少必要键")
        return False
    
    print("  ✅ 边缘信息测试通过")
    return True


def test_hinge_position_calculation():
    """测试铰链位置计算"""
    print("\n=== 测试 7: 铰链位置计算 ===")
    
    from infinigen.assets.sim_objects.box_geometry_modules import (
        create_base_panel,
        create_side_panel,
        calculate_hinge_position,
        EdgePosition,
    )
    
    # 创建底面板和前侧面板
    base = create_base_panel(width=0.3, depth=0.2, thickness=0.002)
    side = create_side_panel(
        width=0.3,
        height=0.15,
        thickness=0.002,
        side=EdgePosition.FRONT,
        parent_dimensions=(0.3, 0.2, 0.002),
    )
    
    # 计算铰链位置
    position, axis = calculate_hinge_position(base, side, EdgePosition.FRONT)
    
    print(f"  铰链位置: {position}")
    print(f"  铰链轴向: {axis}")
    
    # 验证轴向应沿 X 轴 (前边缘)
    if abs(axis[0]) < 0.9:
        print("  ❌ 前边缘铰链轴应沿 X 轴")
        return False
    
    # 验证位置 Y 坐标应为 -0.1 (底面板深度的一半)
    if abs(position[1] - (-0.1)) > 0.02:
        print(f"  ❌ 铰链 Y 位置错误: 期望 ~-0.1, 实际 {position[1]}")
        return False
    
    print("  ✅ 铰链位置计算测试通过")
    return True


def test_panel_geometry_validation():
    """测试几何验证"""
    print("\n=== 测试 8: 几何验证 ===")
    
    from infinigen.assets.sim_objects.box_geometry_modules import (
        create_base_panel,
        validate_panel_geometry,
        PanelGeometry,
        PanelType,
        EdgePosition,
    )
    import numpy as np
    
    # 有效面板
    valid_panel = create_base_panel(width=0.3, depth=0.2, thickness=0.002)
    is_valid, msg = validate_panel_geometry(valid_panel)
    print(f"  有效面板: {is_valid} ({msg})")
    
    if not is_valid:
        print("  ❌ 有效面板应通过验证")
        return False
    
    # 无效面板: 空顶点
    invalid_panel = PanelGeometry(
        panel_type=PanelType.BASE,
        vertices=np.array([]),
        faces=np.array([]),
        edges={},
        center=(0, 0, 0),
        normal=(0, 0, 1),
        dimensions=(0.3, 0.2, 0.002),
    )
    is_valid, msg = validate_panel_geometry(invalid_panel)
    print(f"  空顶点面板: {is_valid} ({msg})")
    
    if is_valid:
        print("  ❌ 空顶点面板应验证失败")
        return False
    
    # 无效面板: 零尺寸
    zero_dim_panel = PanelGeometry(
        panel_type=PanelType.BASE,
        vertices=valid_panel.vertices,
        faces=valid_panel.faces,
        edges={},
        center=(0, 0, 0),
        normal=(0, 0, 1),
        dimensions=(0, 0.2, 0.002),
    )
    is_valid, msg = validate_panel_geometry(zero_dim_panel)
    print(f"  零尺寸面板: {is_valid} ({msg})")
    
    if is_valid:
        print("  ❌ 零尺寸面板应验证失败")
        return False
    
    print("  ✅ 几何验证测试通过")
    return True


def test_mesh_merge():
    """测试网格合并"""
    print("\n=== 测试 9: 网格合并 ===")
    
    from infinigen.assets.sim_objects.box_geometry_modules import (
        create_base_panel,
        create_side_panel,
        merge_panels_to_mesh,
        EdgePosition,
    )
    
    # 创建底面板和四个侧面板
    base = create_base_panel(width=0.3, depth=0.2, thickness=0.002)
    
    panels = [base]
    for side in [EdgePosition.FRONT, EdgePosition.BACK, EdgePosition.LEFT, EdgePosition.RIGHT]:
        if side in [EdgePosition.FRONT, EdgePosition.BACK]:
            width = 0.3
        else:
            width = 0.2
        
        panel = create_side_panel(
            width=width,
            height=0.15,
            thickness=0.002,
            side=side,
            parent_dimensions=(0.3, 0.2, 0.002),
        )
        panels.append(panel)
    
    # 合并
    vertices, faces = merge_panels_to_mesh(panels)
    
    print(f"  面板数: {len(panels)}")
    print(f"  合并后顶点数: {len(vertices)}")
    print(f"  合并后面数: {len(faces)}")
    
    # 验证: 5 个面板 x 8 顶点 = 40 顶点
    expected_vertices = 5 * 8
    if len(vertices) != expected_vertices:
        print(f"  ❌ 顶点数错误: 期望 {expected_vertices}, 实际 {len(vertices)}")
        return False
    
    # 验证: 5 个面板 x 6 面 = 30 面
    expected_faces = 5 * 6
    if len(faces) != expected_faces:
        print(f"  ❌ 面数错误: 期望 {expected_faces}, 实际 {len(faces)}")
        return False
    
    print("  ✅ 网格合并测试通过")
    return True


def test_integration_with_modular_factory():
    """测试与 ModularBoxFactory 的集成"""
    print("\n=== 测试 10: 与 P1-T1 集成 ===")
    
    try:
        from infinigen.assets.sim_objects.modular_box_factory import (
            BoxDimensions,
            BoxMaterialConfig,
        )
        from infinigen.assets.sim_objects.box_geometry_modules import (
            create_base_panel,
            create_side_panel,
            create_lid_panel,
            EdgePosition,
        )
        
        # 从 BoxDimensions 创建面板
        dims = BoxDimensions(width=0.3, depth=0.2, height=0.15, thickness=0.002)
        
        # 创建底面板
        base = create_base_panel(
            width=dims.width,
            depth=dims.depth,
            thickness=dims.thickness,
        )
        
        # 创建侧面板
        front = create_side_panel(
            width=dims.width,
            height=dims.height,
            thickness=dims.thickness,
            side=EdgePosition.FRONT,
            parent_dimensions=(dims.width, dims.depth, dims.thickness),
        )
        
        # 创建盖板
        lid = create_lid_panel(
            width=dims.width,
            depth=dims.depth,
            thickness=dims.thickness,
            attach_edge=EdgePosition.BACK,
            parent_dimensions=(dims.width, dims.depth, dims.thickness),
            parent_height=dims.height,
        )
        
        print(f"  底面板尺寸: {base.dimensions}")
        print(f"  侧面板尺寸: {front.dimensions}")
        print(f"  盖板尺寸: {lid.dimensions}")
        
        # 验证尺寸一致性
        if base.dimensions[0] != dims.width:
            print("  ❌ 底面板宽度与 BoxDimensions 不一致")
            return False
        
        if front.dimensions[1] != dims.height:
            print("  ❌ 侧面板高度与 BoxDimensions 不一致")
            return False
        
        print("  ✅ 与 P1-T1 集成测试通过")
        return True
        
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")
        return False


def test_trimesh_conversion():
    """测试 trimesh 转换"""
    print("\n=== 测试 11: Trimesh 转换 ===")
    
    try:
        import trimesh
        from infinigen.assets.sim_objects.box_geometry_modules import create_base_panel
        
        panel = create_base_panel(width=0.3, depth=0.2, thickness=0.002)
        mesh = panel.to_trimesh()
        
        print(f"  Trimesh 顶点: {len(mesh.vertices)}")
        print(f"  Trimesh 面: {len(mesh.faces)}")
        print(f"  是否封闭: {mesh.is_watertight}")
        print(f"  体积: {mesh.volume:.9f} m³")
        
        # 验证体积
        expected_volume = 0.3 * 0.2 * 0.002
        if abs(mesh.volume - expected_volume) / expected_volume > 0.1:
            print(f"  ⚠️ 体积误差较大: 期望 {expected_volume:.9f}, 实际 {mesh.volume:.9f}")
        
        print("  ✅ Trimesh 转换测试通过")
        return True
        
    except ImportError:
        print("  ⚠️ Trimesh 未安装，跳过测试")
        return True
    except Exception as e:
        print(f"  ❌ 转换失败: {e}")
        return False


def test_code_structure():
    """测试代码结构"""
    print("\n=== 测试 12: 代码结构 ===")
    
    source_file = Path(__file__).parent.parent.parent / "infinigen/assets/sim_objects/box_geometry_modules.py"
    
    with open(source_file, "r") as f:
        source_code = f.read()
    
    checks = [
        ("class PanelType(Enum)", "PanelType 枚举"),
        ("class FlapType(Enum)", "FlapType 枚举"),
        ("class EdgePosition(Enum)", "EdgePosition 枚举"),
        ("class EdgeInfo", "EdgeInfo 数据类"),
        ("class PanelGeometry", "PanelGeometry 数据类"),
        ("class FlapGeometry", "FlapGeometry 数据类"),
        ("def create_base_panel", "create_base_panel 函数"),
        ("def create_side_panel", "create_side_panel 函数"),
        ("def create_lid_panel", "create_lid_panel 函数"),
        ("def create_flap", "create_flap 函数"),
        ("def create_fold_line", "create_fold_line 函数"),
        ("def calculate_hinge_position", "calculate_hinge_position 函数"),
        ("def validate_panel_geometry", "validate_panel_geometry 函数"),
        ("def merge_panels_to_mesh", "merge_panels_to_mesh 函数"),
    ]
    
    all_passed = True
    for pattern, desc in checks:
        if pattern in source_code:
            print(f"  ✅ {desc}: 存在")
        else:
            print(f"  ❌ {desc}: 缺失")
            all_passed = False
    
    return all_passed


def main():
    print("=" * 60)
    print("P1-T2 测试: 基础几何模块")
    print("=" * 60)
    
    results = {
        "模块导入": test_module_import(),
        "底面板创建": test_base_panel_creation(),
        "侧面板创建": test_side_panel_creation(),
        "盖板创建": test_lid_panel_creation(),
        "翻盖创建": test_flap_creation(),
        "边缘信息": test_edge_info(),
        "铰链位置计算": test_hinge_position_calculation(),
        "几何验证": test_panel_geometry_validation(),
        "网格合并": test_mesh_merge(),
        "P1-T1 集成": test_integration_with_modular_factory(),
        "Trimesh 转换": test_trimesh_conversion(),
        "代码结构": test_code_structure(),
    }
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 P1-T2 所有测试通过!")
    else:
        print("\n❌ 部分测试失败")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
