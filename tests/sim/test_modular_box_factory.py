#!/usr/bin/env python3
"""
P1-T1 测试: ModularBoxFactory 基类

测试场景:
1. 模块导入 - 验证新模块可导入
2. 数据类验证 - 验证 BoxDimensions 等数据类
3. 工厂注册 - 验证盒型工厂注册机制
4. 参数采样 - 验证参数采样功能
5. 代码结构 - 验证关键代码元素

运行方法 (在容器中):
    cd /mnt/afs2/zhuhaowu/infinigen
    python tests/sim/test_modular_box_factory.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_module_import():
    """测试模块导入"""
    print("\n=== 测试 1: 模块导入 ===")
    
    try:
        from infinigen.assets.sim_objects.modular_box_factory import (
            ModularBoxFactory,
            BoxType,
            BoxDimensions,
            BoxParameters,
            BoxJointConfig,
            BoxMaterialConfig,
            TuckEndBoxFactory,
            create_box,
            list_available_box_types,
        )
        print("  ✅ 所有类和函数导入成功")
        return True
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")
        return False


def test_box_dimensions():
    """测试 BoxDimensions 数据类"""
    print("\n=== 测试 2: BoxDimensions 验证 ===")
    
    from infinigen.assets.sim_objects.modular_box_factory import BoxDimensions
    
    # 有效尺寸
    dims = BoxDimensions(width=0.3, depth=0.2, height=0.15, thickness=0.002)
    valid, msg = dims.validate()
    print(f"  有效尺寸: {valid} ({msg})")
    
    if not valid:
        print("  ❌ 有效尺寸应该通过验证")
        return False
    
    # 无效尺寸: 负值
    dims_invalid = BoxDimensions(width=-0.1, depth=0.2, height=0.15, thickness=0.002)
    valid, msg = dims_invalid.validate()
    print(f"  负值尺寸: {valid} ({msg})")
    
    if valid:
        print("  ❌ 负值尺寸应该验证失败")
        return False
    
    # 无效尺寸: 壁厚过大
    dims_thick = BoxDimensions(width=0.1, depth=0.1, height=0.1, thickness=0.05)
    valid, msg = dims_thick.validate()
    print(f"  壁厚过大: {valid} ({msg})")
    
    if valid:
        print("  ❌ 壁厚过大应该验证失败")
        return False
    
    print("  ✅ BoxDimensions 验证测试通过")
    return True


def test_box_type_enum():
    """测试 BoxType 枚举"""
    print("\n=== 测试 3: BoxType 枚举 ===")
    
    from infinigen.assets.sim_objects.modular_box_factory import BoxType
    
    # 验证 16 种盒型都存在
    expected_types = [
        "TUCK_END", "LOCK_BOTTOM", "TUCK_END_SAFETY", "LOCK_BOTTOM_SAFETY",
        "HOOK_LOCK_BOTTOM", "HOOK_TUCK_END", "MAILER", "DRAWER",
        "SLIP_LID", "GIFT_WITH_HANDLE", "TUCK_HANDLE", "GABLE_TOP",
        "DOUBLE_LID_GIFT", "PLASTIC_HANDLE", "RSC", "CUSTOM",
    ]
    
    actual_types = [bt.name for bt in BoxType]
    
    missing = set(expected_types) - set(actual_types)
    extra = set(actual_types) - set(expected_types)
    
    if missing:
        print(f"  ❌ 缺少的盒型: {missing}")
        return False
    
    if extra:
        print(f"  ⚠️ 额外的盒型: {extra}")
    
    print(f"  ✅ 所有 {len(expected_types)} 种盒型都已定义")
    return True


def test_factory_registration():
    """测试工厂注册机制"""
    print("\n=== 测试 4: 工厂注册 ===")
    
    from infinigen.assets.sim_objects.modular_box_factory import (
        ModularBoxFactory,
        BoxType,
        TuckEndBoxFactory,
    )
    
    # 验证 TuckEndBox 已注册
    registered = ModularBoxFactory.list_registered_types()
    print(f"  已注册的盒型: {[bt.name for bt in registered]}")
    
    if BoxType.TUCK_END in registered:
        print("  ✅ TuckEndBox 已注册")
    else:
        print("  ❌ TuckEndBox 未注册")
        return False
    
    # 验证可以获取工厂
    try:
        factory_cls = ModularBoxFactory.get_factory(BoxType.TUCK_END)
        if factory_cls == TuckEndBoxFactory:
            print("  ✅ 可以正确获取工厂类")
        else:
            print("  ❌ 获取的工厂类不正确")
            return False
    except Exception as e:
        print(f"  ❌ 获取工厂失败: {e}")
        return False
    
    return True


def test_parameter_sampling():
    """测试参数采样"""
    print("\n=== 测试 5: 参数采样 ===")
    
    from infinigen.assets.sim_objects.modular_box_factory import (
        TuckEndBoxFactory,
        BoxType,
    )
    
    factory = TuckEndBoxFactory(factory_seed=42)
    
    # 采样参数
    params = factory.sample_parameters()
    
    print(f"  盒型: {params.box_type.name}")
    print(f"  尺寸: {params.dimensions.width:.3f} x {params.dimensions.depth:.3f} x {params.dimensions.height:.3f}")
    print(f"  材质: {params.material.material_type}")
    print(f"  关节数: {len(params.joints)}")
    
    # 验证
    if params.box_type != BoxType.TUCK_END:
        print("  ❌ 盒型不正确")
        return False
    
    if len(params.joints) < 2:
        print("  ❌ 双插盒应该至少有 2 个关节")
        return False
    
    # 验证关节配置
    for joint in params.joints:
        print(f"    关节: {joint.label}, 类型: {joint.joint_type}, 轴: {joint.axis}")
    
    print("  ✅ 参数采样测试通过")
    return True


def test_joint_parameters():
    """测试关节参数采样"""
    print("\n=== 测试 6: 关节参数采样 ===")
    
    from infinigen.assets.sim_objects.modular_box_factory import TuckEndBoxFactory
    
    factory = TuckEndBoxFactory(factory_seed=42)
    joint_params = factory.sample_joint_parameters()
    
    print(f"  关节参数: {list(joint_params.keys())}")
    
    for label, params in joint_params.items():
        print(f"    {label}: stiffness={params['stiffness']}, damping={params['damping']}")
    
    if len(joint_params) >= 2:
        print("  ✅ 关节参数采样测试通过")
        return True
    else:
        print("  ❌ 关节参数不足")
        return False


def test_code_structure():
    """测试代码结构"""
    print("\n=== 测试 7: 代码结构 ===")
    
    source_file = Path(__file__).parent.parent.parent / "infinigen/assets/sim_objects/modular_box_factory.py"
    
    with open(source_file, "r") as f:
        source_code = f.read()
    
    checks = [
        ("class ModularBoxFactory", "基类定义"),
        ("class BoxType(Enum)", "盒型枚举"),
        ("class BoxDimensions", "尺寸数据类"),
        ("class BoxParameters", "参数数据类"),
        ("def create_asset", "create_asset 方法"),
        ("def get_box_type", "get_box_type 抽象方法"),
        ("def create_geometry_nodegroup", "create_geometry_nodegroup 抽象方法"),
        ("def get_default_joints", "get_default_joints 抽象方法"),
        ("@ModularBoxFactory.register", "工厂注册装饰器"),
        ("class TuckEndBoxFactory", "TuckEndBox 实现"),
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
    print("P1-T1 测试: ModularBoxFactory 基类")
    print("=" * 60)
    
    results = {
        "模块导入": test_module_import(),
        "BoxDimensions 验证": test_box_dimensions(),
        "BoxType 枚举": test_box_type_enum(),
        "工厂注册": test_factory_registration(),
        "参数采样": test_parameter_sampling(),
        "关节参数": test_joint_parameters(),
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
        print("\n🎉 P1-T1 所有测试通过!")
    else:
        print("\n❌ 部分测试失败")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
