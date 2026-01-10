#!/usr/bin/env python3
"""
P1-T3 测试: 关节注入系统

测试场景:
1. 模块导入 - 验证新模块可导入
2. JointConfig 验证 - 验证关节配置数据类
3. 关节动力学预设 - 验证 JOINT_DYNAMICS_PRESETS
4. 铰链关节添加 - 验证 add_hinge_at_edge()
5. 滑轨关节添加 - 验证 add_slide_joint()
6. 关节范围验证 - 验证 validate_joint_range()
7. 安全范围计算 - 验证 calculate_safe_joint_range()
8. 盒型关节集合 - 验证 BoxJointSet
9. 预定义盒型关节 - 验证 create_*_box_joints()
10. 与 P1-T2 集成 - 验证与几何模块的兼容性
11. Blender 节点输入 - 验证节点组输入生成

运行方法 (在容器中):
    cd /mnt/afs2/zhuhaowu/infinigen
    python tests/sim/test_joint_injector.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_module_import():
    """测试模块导入"""
    print("\n=== 测试 1: 模块导入 ===")
    
    try:
        from infinigen.assets.sim_objects.joint_injector import (
            JointType,
            MaterialType,
            JointConfig,
            JointDynamics,
            JOINT_DYNAMICS_PRESETS,
            get_joint_dynamics,
            add_hinge_at_edge,
            add_slide_joint,
            validate_joint_range,
            calculate_safe_joint_range,
            BoxJointSet,
            create_tuck_end_box_joints,
            create_drawer_box_joints,
            create_mailer_box_joints,
            create_hinge_joint_node_inputs,
            create_sliding_joint_node_inputs,
        )
        print("  ✅ 所有类和函数导入成功")
        return True
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")
        return False


def test_joint_config():
    """测试 JointConfig 验证"""
    print("\n=== 测试 2: JointConfig 验证 ===")
    
    from infinigen.assets.sim_objects.joint_injector import JointConfig, JointType
    
    # 有效配置
    valid_config = JointConfig(
        label="test_hinge",
        joint_type=JointType.HINGE,
        position=(0, 0, 0.1),
        axis=(1, 0, 0),
        min_value=0.0,
        max_value=np.pi,
        initial_value=0.0,
    )
    is_valid, msg = valid_config.validate()
    print(f"  有效配置: {is_valid} ({msg})")
    
    if not is_valid:
        print("  ❌ 有效配置应通过验证")
        return False
    
    # 无效配置: 空标签
    invalid_label = JointConfig(
        label="",
        joint_type=JointType.HINGE,
        position=(0, 0, 0.1),
        axis=(1, 0, 0),
    )
    is_valid, msg = invalid_label.validate()
    print(f"  空标签: {is_valid} ({msg})")
    
    if is_valid:
        print("  ❌ 空标签应验证失败")
        return False
    
    # 无效配置: min > max
    invalid_range = JointConfig(
        label="test",
        joint_type=JointType.HINGE,
        position=(0, 0, 0.1),
        axis=(1, 0, 0),
        min_value=2.0,
        max_value=1.0,
    )
    is_valid, msg = invalid_range.validate()
    print(f"  范围错误: {is_valid} ({msg})")
    
    if is_valid:
        print("  ❌ min > max 应验证失败")
        return False
    
    # 测试字典转换
    config_dict = valid_config.to_dict()
    print(f"  字典键: {list(config_dict.keys())}")
    
    if "label" not in config_dict or "joint_type" not in config_dict:
        print("  ❌ 字典缺少必要键")
        return False
    
    print("  ✅ JointConfig 验证测试通过")
    return True


def test_joint_dynamics_presets():
    """测试关节动力学预设"""
    print("\n=== 测试 3: 关节动力学预设 ===")
    
    from infinigen.assets.sim_objects.joint_injector import (
        JOINT_DYNAMICS_PRESETS,
        MaterialType,
        JointType,
        get_joint_dynamics,
    )
    
    # 验证预设存在
    materials = [MaterialType.CARDBOARD, MaterialType.CORRUGATED, MaterialType.PLASTIC]
    all_passed = True
    
    for material in materials:
        if material not in JOINT_DYNAMICS_PRESETS:
            print(f"  ❌ 缺少材料预设: {material.name}")
            all_passed = False
            continue
        
        presets = JOINT_DYNAMICS_PRESETS[material]
        
        for joint_type in [JointType.HINGE, JointType.SLIDE]:
            if joint_type not in presets:
                print(f"  ❌ 缺少 {material.name} 的 {joint_type.name} 预设")
                all_passed = False
            else:
                dynamics = presets[joint_type]
                print(f"  {material.name}/{joint_type.name}: damping={dynamics.damping}, friction={dynamics.friction}")
    
    # 测试 get_joint_dynamics
    dynamics = get_joint_dynamics(MaterialType.CARDBOARD, JointType.HINGE)
    print(f"  get_joint_dynamics: damping={dynamics.damping}")
    
    if dynamics.damping <= 0:
        print("  ❌ 阻尼系数应为正数")
        all_passed = False
    
    if all_passed:
        print("  ✅ 关节动力学预设测试通过")
    
    return all_passed


def test_add_hinge_at_edge():
    """测试铰链关节添加"""
    print("\n=== 测试 4: 铰链关节添加 ===")
    
    from infinigen.assets.sim_objects.joint_injector import add_hinge_at_edge, JointType
    from infinigen.assets.sim_objects.box_geometry_modules import (
        create_base_panel,
        create_lid_panel,
        EdgePosition,
    )
    
    # 创建几何
    base = create_base_panel(width=0.3, depth=0.2, thickness=0.002)
    lid = create_lid_panel(
        width=0.3,
        depth=0.2,
        thickness=0.002,
        attach_edge=EdgePosition.BACK,
        parent_dimensions=(0.3, 0.2, 0.002),
        parent_height=0.15,
    )
    
    # 获取边缘
    edge = base.get_edge(EdgePosition.BACK)
    
    # 添加铰链
    joint_config = add_hinge_at_edge(
        parent_geometry=base,
        child_geometry=lid,
        edge_info=edge,
        label="lid_hinge",
        min_angle=0.0,
        max_angle=np.pi * 0.75,
    )
    
    print(f"  关节标签: {joint_config.label}")
    print(f"  关节类型: {joint_config.joint_type.name}")
    print(f"  位置: {joint_config.position}")
    print(f"  轴向: {joint_config.axis}")
    print(f"  范围: [{joint_config.min_value:.2f}, {joint_config.max_value:.2f}] rad")
    
    # 验证
    if joint_config.joint_type != JointType.HINGE:
        print("  ❌ 关节类型应为 HINGE")
        return False
    
    is_valid, msg = joint_config.validate()
    if not is_valid:
        print(f"  ❌ 配置验证失败: {msg}")
        return False
    
    print("  ✅ 铰链关节添加测试通过")
    return True


def test_add_slide_joint():
    """测试滑轨关节添加"""
    print("\n=== 测试 5: 滑轨关节添加 ===")
    
    from infinigen.assets.sim_objects.joint_injector import add_slide_joint, JointType
    from infinigen.assets.sim_objects.box_geometry_modules import create_base_panel
    
    # 创建几何 (外盒和内抽屉)
    outer_box = create_base_panel(width=0.3, depth=0.25, thickness=0.002)
    inner_drawer = create_base_panel(width=0.28, depth=0.22, thickness=0.002)
    
    # 添加滑轨
    joint_config = add_slide_joint(
        parent_geometry=outer_box,
        child_geometry=inner_drawer,
        slide_direction=(0, -1, 0),
        label="drawer_slide",
        min_distance=0.0,
        max_distance=0.2,
    )
    
    print(f"  关节标签: {joint_config.label}")
    print(f"  关节类型: {joint_config.joint_type.name}")
    print(f"  滑动方向: {joint_config.axis}")
    print(f"  范围: [{joint_config.min_value:.2f}, {joint_config.max_value:.2f}] m")
    
    # 验证
    if joint_config.joint_type != JointType.SLIDE:
        print("  ❌ 关节类型应为 SLIDE")
        return False
    
    # 验证滑动方向归一化
    axis_norm = np.linalg.norm(joint_config.axis)
    if abs(axis_norm - 1.0) > 0.01:
        print(f"  ❌ 轴向量应为单位向量，当前长度: {axis_norm}")
        return False
    
    print("  ✅ 滑轨关节添加测试通过")
    return True


def test_validate_joint_range():
    """测试关节范围验证"""
    print("\n=== 测试 6: 关节范围验证 ===")
    
    from infinigen.assets.sim_objects.joint_injector import (
        JointConfig,
        JointType,
        validate_joint_range,
    )
    
    # 创建简单测试几何
    parent_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 0.1], [1, 0, 0.1], [1, 1, 0.1], [0, 1, 0.1],
    ])
    
    child_vertices = np.array([
        [0, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0],
        [0, 1, 0.1], [1, 1, 0.1], [1, 2, 0.1], [0, 2, 0.1],
    ])
    
    # 有效关节配置
    valid_joint = JointConfig(
        label="test_hinge",
        joint_type=JointType.HINGE,
        position=(0.5, 1, 0.05),
        axis=(1, 0, 0),
        min_value=0.0,
        max_value=np.pi / 2,
    )
    
    is_valid, msg, safe_max = validate_joint_range(
        valid_joint, parent_vertices, child_vertices, check_collision=False
    )
    print(f"  无碰撞检测: {is_valid} ({msg}), safe_max={safe_max:.2f}")
    
    if not is_valid:
        print("  ❌ 无碰撞检测应通过")
        return False
    
    # 带碰撞检测
    is_valid, msg, safe_max = validate_joint_range(
        valid_joint, parent_vertices, child_vertices, check_collision=True
    )
    safe_max_str = f"{safe_max:.2f}" if safe_max is not None else "N/A"
    print(f"  有碰撞检测: {is_valid} ({msg}), safe_max={safe_max_str}")
    
    print("  ✅ 关节范围验证测试通过")
    return True


def test_calculate_safe_joint_range():
    """测试安全范围计算"""
    print("\n=== 测试 7: 安全范围计算 ===")
    
    from infinigen.assets.sim_objects.joint_injector import calculate_safe_joint_range, JointType
    from infinigen.assets.sim_objects.box_geometry_modules import (
        create_base_panel,
        create_side_panel,
        EdgePosition,
    )
    
    # 创建底面板和侧面板
    base = create_base_panel(width=0.3, depth=0.2, thickness=0.002)
    side = create_side_panel(
        width=0.3,
        height=0.15,
        thickness=0.002,
        side=EdgePosition.FRONT,
        parent_dimensions=(0.3, 0.2, 0.002),
    )
    
    edge = base.get_edge(EdgePosition.FRONT)
    
    # 计算安全范围
    min_val, max_val = calculate_safe_joint_range(
        parent_geometry=base,
        child_geometry=side,
        edge_info=edge,
        joint_type=JointType.HINGE,
    )
    
    print(f"  铰链安全范围: [{min_val:.2f}, {max_val:.2f}] rad")
    print(f"  角度范围: [{np.degrees(min_val):.1f}°, {np.degrees(max_val):.1f}°]")
    
    if max_val <= min_val:
        print("  ❌ 最大值应大于最小值")
        return False
    
    # 滑轨安全范围
    min_val_s, max_val_s = calculate_safe_joint_range(
        parent_geometry=base,
        child_geometry=side,
        edge_info=edge,
        joint_type=JointType.SLIDE,
    )
    
    print(f"  滑轨安全范围: [{min_val_s:.3f}, {max_val_s:.3f}] m")
    
    print("  ✅ 安全范围计算测试通过")
    return True


def test_box_joint_set():
    """测试盒型关节集合"""
    print("\n=== 测试 8: BoxJointSet ===")
    
    from infinigen.assets.sim_objects.joint_injector import (
        BoxJointSet,
        JointConfig,
        JointType,
        MaterialType,
    )
    
    # 创建关节集合
    joint_set = BoxJointSet(
        box_type="TestBox",
        material=MaterialType.CARDBOARD,
    )
    
    # 添加关节
    joint1 = JointConfig(
        label="hinge_1",
        joint_type=JointType.HINGE,
        position=(0, 0, 0.1),
        axis=(1, 0, 0),
    )
    joint2 = JointConfig(
        label="hinge_2",
        joint_type=JointType.HINGE,
        position=(0, 0, 0.2),
        axis=(1, 0, 0),
    )
    
    joint_set.add_joint(joint1)
    joint_set.add_joint(joint2)
    
    print(f"  盒型: {joint_set.box_type}")
    print(f"  材料: {joint_set.material.name}")
    print(f"  关节数: {len(joint_set.joints)}")
    
    # 获取关节
    retrieved = joint_set.get_joint("hinge_1")
    if retrieved is None:
        print("  ❌ 无法获取关节")
        return False
    print(f"  获取关节: {retrieved.label}")
    
    # 验证所有
    is_valid, errors = joint_set.validate_all()
    print(f"  验证结果: {is_valid}, 错误数: {len(errors)}")
    
    if not is_valid:
        print(f"  ❌ 验证失败: {errors}")
        return False
    
    # 获取动力学
    dynamics = joint_set.get_dynamics()
    print(f"  动力学参数键: {list(dynamics.keys())}")
    
    # 字典转换
    set_dict = joint_set.to_dict()
    print(f"  字典键: {list(set_dict.keys())}")
    
    print("  ✅ BoxJointSet 测试通过")
    return True


def test_predefined_box_joints():
    """测试预定义盒型关节"""
    print("\n=== 测试 9: 预定义盒型关节 ===")
    
    from infinigen.assets.sim_objects.joint_injector import (
        create_tuck_end_box_joints,
        create_drawer_box_joints,
        create_mailer_box_joints,
        JointType,
    )
    
    all_passed = True
    
    # 双插盒
    tuck_end_joints = create_tuck_end_box_joints(
        base_dimensions=(0.3, 0.2, 0.002),
        height=0.15,
        thickness=0.002,
    )
    print(f"  双插盒: {len(tuck_end_joints.joints)} 个关节")
    for joint in tuck_end_joints.joints:
        print(f"    - {joint.label}: {joint.joint_type.name}")
    
    if len(tuck_end_joints.joints) < 2:
        print("  ❌ 双插盒应至少有 2 个关节")
        all_passed = False
    
    # 抽屉盒
    drawer_joints = create_drawer_box_joints(
        box_dimensions=(0.3, 0.25, 0.15),
        drawer_depth=0.2,
    )
    print(f"  抽屉盒: {len(drawer_joints.joints)} 个关节")
    for joint in drawer_joints.joints:
        print(f"    - {joint.label}: {joint.joint_type.name}")
        if joint.joint_type != JointType.SLIDE:
            print(f"  ❌ 抽屉关节应为滑轨类型")
            all_passed = False
    
    # 飞机盒
    mailer_joints = create_mailer_box_joints(
        base_dimensions=(0.35, 0.25, 0.002),
        height=0.1,
    )
    print(f"  飞机盒: {len(mailer_joints.joints)} 个关节")
    for joint in mailer_joints.joints:
        print(f"    - {joint.label}: {joint.joint_type.name}")
    
    if all_passed:
        print("  ✅ 预定义盒型关节测试通过")
    
    return all_passed


def test_integration_with_geometry():
    """测试与几何模块的集成"""
    print("\n=== 测试 10: 与 P1-T2 集成 ===")
    
    from infinigen.assets.sim_objects.joint_injector import (
        add_hinge_at_edge,
        validate_joint_range,
    )
    from infinigen.assets.sim_objects.box_geometry_modules import (
        create_base_panel,
        create_side_panel,
        create_lid_panel,
        EdgePosition,
    )
    
    # 创建完整的盒子几何
    base = create_base_panel(width=0.3, depth=0.2, thickness=0.002)
    
    sides = []
    for side_pos in [EdgePosition.FRONT, EdgePosition.BACK, EdgePosition.LEFT, EdgePosition.RIGHT]:
        width = 0.3 if side_pos in [EdgePosition.FRONT, EdgePosition.BACK] else 0.2
        side = create_side_panel(
            width=width,
            height=0.15,
            thickness=0.002,
            side=side_pos,
            parent_dimensions=(0.3, 0.2, 0.002),
        )
        sides.append((side_pos, side))
    
    lid = create_lid_panel(
        width=0.3,
        depth=0.2,
        thickness=0.002,
        attach_edge=EdgePosition.BACK,
        parent_dimensions=(0.3, 0.2, 0.002),
        parent_height=0.15,
    )
    
    # 获取后侧面板
    back_panel = next(s for pos, s in sides if pos == EdgePosition.BACK)
    
    # 添加盖子铰链
    edge = back_panel.get_edge(EdgePosition.TOP)
    if edge is None:
        print("  ❌ 无法获取侧面板顶边缘")
        return False
    
    joint = add_hinge_at_edge(
        parent_geometry=back_panel,
        child_geometry=lid,
        edge_info=edge,
        label="lid_hinge",
    )
    
    print(f"  盖子铰链: {joint.label}")
    print(f"  铰链位置: {joint.position}")
    print(f"  铰链轴向: {joint.axis}")
    
    # 验证范围
    is_valid, msg, safe_max = validate_joint_range(
        joint,
        back_panel.vertices,
        lid.vertices,
        check_collision=True,
    )
    print(f"  范围验证: {is_valid} ({msg})")
    
    print("  ✅ 与 P1-T2 集成测试通过")
    return True


def test_blender_node_inputs():
    """测试 Blender 节点输入生成"""
    print("\n=== 测试 11: Blender 节点输入 ===")
    
    from infinigen.assets.sim_objects.joint_injector import (
        JointConfig,
        JointType,
        create_hinge_joint_node_inputs,
        create_sliding_joint_node_inputs,
    )
    
    # 铰链配置
    hinge_config = JointConfig(
        label="test_hinge",
        joint_type=JointType.HINGE,
        position=(0.1, 0.1, 0.15),
        axis=(1, 0, 0),
        min_value=0.0,
        max_value=np.pi,
        initial_value=0.5,
    )
    
    hinge_inputs = create_hinge_joint_node_inputs(hinge_config)
    print(f"  铰链节点输入键: {list(hinge_inputs.keys())}")
    print(f"    Joint Label: {hinge_inputs['Joint Label']}")
    print(f"    Position: {hinge_inputs['Position']}")
    print(f"    Axis: {hinge_inputs['Axis']}")
    
    required_keys = ["Joint Label", "Position", "Axis", "Value", "Min", "Max"]
    for key in required_keys:
        if key not in hinge_inputs:
            print(f"  ❌ 缺少必要键: {key}")
            return False
    
    # 滑轨配置
    slide_config = JointConfig(
        label="test_slide",
        joint_type=JointType.SLIDE,
        position=(0, 0, 0),
        axis=(0, 1, 0),
        min_value=0.0,
        max_value=0.2,
    )
    
    slide_inputs = create_sliding_joint_node_inputs(slide_config)
    print(f"  滑轨节点输入键: {list(slide_inputs.keys())}")
    
    print("  ✅ Blender 节点输入测试通过")
    return True


def test_code_structure():
    """测试代码结构"""
    print("\n=== 测试 12: 代码结构 ===")
    
    source_file = Path(__file__).parent.parent.parent / "infinigen/assets/sim_objects/joint_injector.py"
    
    with open(source_file, "r") as f:
        source_code = f.read()
    
    checks = [
        ("class JointType(Enum)", "JointType 枚举"),
        ("class MaterialType(Enum)", "MaterialType 枚举"),
        ("class JointConfig", "JointConfig 数据类"),
        ("class JointDynamics", "JointDynamics 数据类"),
        ("JOINT_DYNAMICS_PRESETS", "关节动力学预设"),
        ("def get_joint_dynamics", "get_joint_dynamics 函数"),
        ("def add_hinge_at_edge", "add_hinge_at_edge 函数"),
        ("def add_slide_joint", "add_slide_joint 函数"),
        ("def validate_joint_range", "validate_joint_range 函数"),
        ("def calculate_safe_joint_range", "calculate_safe_joint_range 函数"),
        ("class BoxJointSet", "BoxJointSet 类"),
        ("def create_tuck_end_box_joints", "create_tuck_end_box_joints 函数"),
        ("def create_drawer_box_joints", "create_drawer_box_joints 函数"),
        ("def create_hinge_joint_node_inputs", "create_hinge_joint_node_inputs 函数"),
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
    print("P1-T3 测试: 关节注入系统")
    print("=" * 60)
    
    results = {
        "模块导入": test_module_import(),
        "JointConfig 验证": test_joint_config(),
        "关节动力学预设": test_joint_dynamics_presets(),
        "铰链关节添加": test_add_hinge_at_edge(),
        "滑轨关节添加": test_add_slide_joint(),
        "关节范围验证": test_validate_joint_range(),
        "安全范围计算": test_calculate_safe_joint_range(),
        "BoxJointSet": test_box_joint_set(),
        "预定义盒型关节": test_predefined_box_joints(),
        "P1-T2 集成": test_integration_with_geometry(),
        "Blender 节点输入": test_blender_node_inputs(),
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
        print("\n🎉 P1-T3 所有测试通过!")
    else:
        print("\n❌ 部分测试失败")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
