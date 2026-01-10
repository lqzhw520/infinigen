#!/usr/bin/env python3
"""
P1-T5 测试: 关节动力学预设

测试场景:
1. 模块导入 - 验证新增类和函数可导入
2. 材质预设 - 验证 MATERIAL_JOINT_PRESETS
3. 盒型预设 - 验证 BOX_TYPE_JOINT_PRESETS
4. 参数转换 - 验证 URDF/MJCF 格式转换
5. 材质关节查询 - 验证 get_material_joint_dynamics
6. 盒型关节查询 - 验证 get_box_type_joint_dynamics
7. 材质名称转换 - 验证 material_name_to_category
8. 阻尼计算 - 验证 calculate_hinge_damping
9. 与 P1-T4 集成 - 验证与材质系统的兼容性
10. 原有接口兼容 - 验证 get_joint_properties

运行方法 (在容器中):
    cd /mnt/afs2/zhuhaowu/infinigen
    python tests/sim/test_joint_dynamics.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_module_import():
    """测试模块导入"""
    print("\n=== 测试 1: 模块导入 ===")
    
    try:
        from infinigen.core.sim.physics.joint_dynamics import (
            JointDynamicsType,
            MaterialCategory,
            JointDynamicsParams,
            MATERIAL_JOINT_PRESETS,
            BOX_TYPE_JOINT_PRESETS,
            get_joint_properties,
            get_material_joint_dynamics,
            get_box_type_joint_dynamics,
            material_name_to_category,
            list_available_materials,
            list_box_type_presets,
            calculate_hinge_damping,
            calculate_slide_friction,
            estimate_joint_effort_limit,
        )
        print("  ✅ 所有类和函数导入成功")
        return True
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")
        return False


def test_material_presets():
    """测试材质预设"""
    print("\n=== 测试 2: 材质预设 ===")
    
    from infinigen.core.sim.physics.joint_dynamics import (
        MATERIAL_JOINT_PRESETS,
        MaterialCategory,
        JointDynamicsType,
    )
    
    expected_materials = [
        MaterialCategory.CARDBOARD,
        MaterialCategory.CORRUGATED,
        MaterialCategory.PLASTIC,
        MaterialCategory.WOOD,
        MaterialCategory.METAL,
    ]
    
    all_passed = True
    
    for material in expected_materials:
        if material not in MATERIAL_JOINT_PRESETS:
            print(f"  ❌ 缺少材质: {material.name}")
            all_passed = False
            continue
        
        presets = MATERIAL_JOINT_PRESETS[material]
        
        for joint_type in [JointDynamicsType.HINGE, JointDynamicsType.SLIDE]:
            if joint_type not in presets:
                print(f"  ❌ 缺少 {material.name} 的 {joint_type.name} 预设")
                all_passed = False
            else:
                params = presets[joint_type]
                print(f"  {material.name}/{joint_type.name}: damping={params.damping}, friction={params.friction}")
    
    if all_passed:
        print("  ✅ 材质预设测试通过")
    
    return all_passed


def test_box_type_presets():
    """测试盒型预设"""
    print("\n=== 测试 3: 盒型预设 ===")
    
    from infinigen.core.sim.physics.joint_dynamics import (
        BOX_TYPE_JOINT_PRESETS,
        list_box_type_presets,
    )
    
    expected_box_types = ["TuckEndBox", "MailerBox", "DrawerBox", "RSCBox"]
    
    all_passed = True
    
    for box_type in expected_box_types:
        if box_type not in BOX_TYPE_JOINT_PRESETS:
            print(f"  ❌ 缺少盒型: {box_type}")
            all_passed = False
        else:
            joints = list(BOX_TYPE_JOINT_PRESETS[box_type].keys())
            print(f"  {box_type}: {len(joints)} 个关节预设 - {joints}")
    
    # 测试列表函数
    available = list_box_type_presets()
    print(f"  可用盒型预设: {len(available)} 种")
    
    if all_passed:
        print("  ✅ 盒型预设测试通过")
    
    return all_passed


def test_params_conversion():
    """测试参数转换"""
    print("\n=== 测试 4: 参数转换 ===")
    
    from infinigen.core.sim.physics.joint_dynamics import JointDynamicsParams
    
    params = JointDynamicsParams(
        damping=0.5,
        friction=0.2,
        stiffness=0.1,
        velocity_limit=10.0,
        effort_limit=100.0,
        armature=0.01,
    )
    
    # URDF 格式
    urdf_dict = params.to_urdf_dict()
    print(f"  URDF: {urdf_dict}")
    
    if "damping" not in urdf_dict or "friction" not in urdf_dict:
        print("  ❌ URDF 字典缺少必要键")
        return False
    
    # MJCF 格式
    mjcf_dict = params.to_mjcf_dict()
    print(f"  MJCF: {mjcf_dict}")
    
    if "damping" not in mjcf_dict or "frictionloss" not in mjcf_dict:
        print("  ❌ MJCF 字典缺少必要键")
        return False
    
    # 缩放
    scaled = params.scale(2.0)
    print(f"  缩放后 damping: {scaled.damping} (原: {params.damping})")
    
    if abs(scaled.damping - params.damping * 2) > 0.01:
        print("  ❌ 缩放功能错误")
        return False
    
    print("  ✅ 参数转换测试通过")
    return True


def test_get_material_joint_dynamics():
    """测试材质关节查询"""
    print("\n=== 测试 5: 材质关节查询 ===")
    
    from infinigen.core.sim.physics.joint_dynamics import (
        get_material_joint_dynamics,
        MaterialCategory,
        JointDynamicsType,
    )
    
    # 测试各种材质
    materials = [
        (MaterialCategory.CARDBOARD, JointDynamicsType.HINGE),
        (MaterialCategory.CORRUGATED, JointDynamicsType.SLIDE),
        (MaterialCategory.PLASTIC, JointDynamicsType.HINGE),
    ]
    
    for material, joint_type in materials:
        params = get_material_joint_dynamics(material, joint_type)
        print(f"  {material.name}/{joint_type.name}: damping={params.damping}")
        
        if params.damping <= 0:
            print("  ❌ 阻尼应为正数")
            return False
    
    print("  ✅ 材质关节查询测试通过")
    return True


def test_get_box_type_joint_dynamics():
    """测试盒型关节查询"""
    print("\n=== 测试 6: 盒型关节查询 ===")
    
    from infinigen.core.sim.physics.joint_dynamics import (
        get_box_type_joint_dynamics,
        MaterialCategory,
    )
    
    # 测试盒型特定预设
    params = get_box_type_joint_dynamics("TuckEndBox", "top_lid")
    print(f"  TuckEndBox/top_lid: damping={params.damping}")
    
    # 测试不存在的关节标签（应使用材质默认值）
    params = get_box_type_joint_dynamics("TuckEndBox", "unknown_joint", MaterialCategory.CORRUGATED)
    print(f"  TuckEndBox/unknown (瓦楞纸): damping={params.damping}")
    
    # 测试不存在的盒型
    params = get_box_type_joint_dynamics("UnknownBox", "lid")
    print(f"  UnknownBox/lid (默认): damping={params.damping}")
    
    print("  ✅ 盒型关节查询测试通过")
    return True


def test_material_name_conversion():
    """测试材质名称转换"""
    print("\n=== 测试 7: 材质名称转换 ===")
    
    from infinigen.core.sim.physics.joint_dynamics import (
        material_name_to_category,
        MaterialCategory,
    )
    
    test_cases = [
        ("cardboard", MaterialCategory.CARDBOARD),
        ("cardboard_white", MaterialCategory.CARDBOARD),
        ("corrugated", MaterialCategory.CORRUGATED),
        ("corrugated_a", MaterialCategory.CORRUGATED),
        ("plastic_pp", MaterialCategory.PLASTIC),
        ("wood_mdf", MaterialCategory.WOOD),
        ("unknown", MaterialCategory.CARDBOARD),  # 默认
    ]
    
    all_passed = True
    
    for name, expected in test_cases:
        result = material_name_to_category(name)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {name} -> {result.name} (预期: {expected.name})")
        
        if result != expected:
            all_passed = False
    
    if all_passed:
        print("  ✅ 材质名称转换测试通过")
    
    return all_passed


def test_damping_calculation():
    """测试阻尼计算"""
    print("\n=== 测试 8: 阻尼计算 ===")
    
    from infinigen.core.sim.physics.joint_dynamics import (
        calculate_hinge_damping,
        calculate_slide_friction,
        estimate_joint_effort_limit,
    )
    
    # 铰链阻尼
    mass = 0.5  # 500g 盖子
    arm_length = 0.1  # 10cm 力臂
    damping = calculate_hinge_damping(mass, arm_length, settling_time=1.0)
    print(f"  铰链阻尼 (500g, 10cm): {damping:.4f} Nms/rad")
    
    if damping <= 0:
        print("  ❌ 阻尼应为正数")
        return False
    
    # 滑动摩擦
    friction = calculate_slide_friction(5.0, 0.3)  # 5N 法向力
    print(f"  滑动摩擦力 (5N, μ=0.3): {friction:.2f} N")
    
    if abs(friction - 1.5) > 0.01:
        print("  ❌ 摩擦力计算错误")
        return False
    
    # 力矩限制
    effort = estimate_joint_effort_limit(0.5, 0.1)
    print(f"  力矩限制 (500g, 10cm): {effort:.4f} Nm")
    
    if effort <= 0:
        print("  ❌ 力矩限制应为正数")
        return False
    
    print("  ✅ 阻尼计算测试通过")
    return True


def test_integration_with_materials():
    """测试与材质系统的集成"""
    print("\n=== 测试 9: 与 P1-T4 集成 ===")
    
    from infinigen.core.sim.physics.material_definitions import (
        list_box_materials,
    )
    from infinigen.core.sim.physics.joint_dynamics import (
        material_name_to_category,
        get_material_joint_dynamics,
        JointDynamicsType,
    )
    
    # 对所有盒子材质测试关节动力学获取
    box_materials = list_box_materials()
    
    for material_name in box_materials[:5]:  # 测试前 5 个
        category = material_name_to_category(material_name)
        dynamics = get_material_joint_dynamics(category, JointDynamicsType.HINGE)
        print(f"  {material_name} -> {category.name}: damping={dynamics.damping}")
    
    print("  ✅ 与 P1-T4 集成测试通过")
    return True


def test_original_interface():
    """测试原有接口兼容性"""
    print("\n=== 测试 10: 原有接口兼容 ===")
    
    from infinigen.core.sim.physics.joint_dynamics import get_joint_properties
    
    # 测试原有接口
    joint_params = {
        "hinge_1": {"stiffness": 0.5, "damping": 0.3, "friction": 0.1},
        "hinge_2": {"damping": 0.4},
    }
    
    result1 = get_joint_properties("hinge_1", joint_params)
    print(f"  hinge_1: {result1}")
    
    if result1["stiffness"] != 0.5 or result1["damping"] != 0.3:
        print("  ❌ 原有接口返回错误")
        return False
    
    # 测试缺失的关节
    result2 = get_joint_properties("unknown", joint_params)
    print(f"  unknown: {result2}")
    
    if result2["stiffness"] != 0 or result2["damping"] != 0:
        print("  ❌ 缺失关节应返回默认值")
        return False
    
    print("  ✅ 原有接口兼容性测试通过")
    return True


def test_code_structure():
    """测试代码结构"""
    print("\n=== 测试 11: 代码结构 ===")
    
    source_file = Path(__file__).parent.parent.parent / "infinigen/core/sim/physics/joint_dynamics.py"
    
    with open(source_file, "r") as f:
        source_code = f.read()
    
    checks = [
        ("class JointDynamicsType(Enum)", "JointDynamicsType 枚举"),
        ("class MaterialCategory(Enum)", "MaterialCategory 枚举"),
        ("class JointDynamicsParams", "JointDynamicsParams 数据类"),
        ("MATERIAL_JOINT_PRESETS", "材质预设字典"),
        ("BOX_TYPE_JOINT_PRESETS", "盒型预设字典"),
        ("def get_material_joint_dynamics", "get_material_joint_dynamics 函数"),
        ("def get_box_type_joint_dynamics", "get_box_type_joint_dynamics 函数"),
        ("def material_name_to_category", "material_name_to_category 函数"),
        ("def to_urdf_dict", "URDF 转换方法"),
        ("def to_mjcf_dict", "MJCF 转换方法"),
        ("def calculate_hinge_damping", "calculate_hinge_damping 函数"),
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
    print("P1-T5 测试: 关节动力学预设")
    print("=" * 60)
    
    results = {
        "模块导入": test_module_import(),
        "材质预设": test_material_presets(),
        "盒型预设": test_box_type_presets(),
        "参数转换": test_params_conversion(),
        "材质关节查询": test_get_material_joint_dynamics(),
        "盒型关节查询": test_get_box_type_joint_dynamics(),
        "材质名称转换": test_material_name_conversion(),
        "阻尼计算": test_damping_calculation(),
        "P1-T4 集成": test_integration_with_materials(),
        "原有接口兼容": test_original_interface(),
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
        print("\n🎉 P1-T5 所有测试通过!")
    else:
        print("\n❌ 部分测试失败")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
