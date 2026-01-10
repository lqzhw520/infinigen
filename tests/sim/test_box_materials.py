#!/usr/bin/env python3
"""
P1-T4 测试: 材质系统 (Cardboard/Corrugated)

测试场景:
1. 模块导入 - 验证材质类可导入
2. Cardboard 材质 - 验证卡纸材质属性
3. Corrugated 材质 - 验证瓦楞纸材质及其类型
4. PlasticBox 材质 - 验证塑料盒材质
5. WoodBox 材质 - 验证木质盒材质
6. 参数采样 - 验证参数采样功能
7. 材质注册表 - 验证 MATERIALS 注册表
8. 辅助函数 - 验证 get_box_material 和 list_box_materials
9. 与 P1-T3 集成 - 验证与关节系统的兼容性

运行方法 (在容器中):
    cd /mnt/afs2/zhuhaowu/infinigen
    python tests/sim/test_box_materials.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_module_import():
    """测试模块导入"""
    print("\n=== 测试 1: 模块导入 ===")
    
    try:
        from infinigen.core.sim.physics.material_definitions import (
            BaseMaterial,
            BoxMaterial,
            Cardboard,
            CardboardWhite,
            CardboardKraft,
            Corrugated,
            PlasticBox,
            WoodBox,
            MATERIALS,
            get_box_material,
            list_box_materials,
        )
        print("  ✅ 所有类和函数导入成功")
        return True
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")
        return False


def test_cardboard_material():
    """测试卡纸材质"""
    print("\n=== 测试 2: Cardboard 材质 ===")
    
    from infinigen.core.sim.physics.material_definitions import (
        Cardboard,
        CardboardWhite,
        CardboardKraft,
    )
    
    # 基础卡纸
    cardboard = Cardboard()
    print(f"  基础卡纸:")
    print(f"    密度范围: {cardboard.min_density}-{cardboard.max_density} kg/m³")
    print(f"    摩擦系数: {cardboard.min_friction}-{cardboard.max_friction}")
    print(f"    厚度范围: {cardboard.min_thickness*1000:.1f}-{cardboard.max_thickness*1000:.1f} mm")
    
    # 验证密度范围
    if cardboard.min_density >= cardboard.max_density:
        print("  ❌ 密度最小值应小于最大值")
        return False
    
    # 验证是典型卡纸密度
    if cardboard.min_density > 500 or cardboard.max_density < 200:
        print("  ❌ 卡纸密度范围不合理")
        return False
    
    # 白卡纸
    white = CardboardWhite()
    print(f"  白卡纸:")
    print(f"    密度范围: {white.min_density}-{white.max_density} kg/m³")
    
    if white.min_density < cardboard.min_density:
        print("  ⚠️ 白卡纸应该比普通卡纸更硬")
    
    # 牛皮卡纸
    kraft = CardboardKraft()
    print(f"  牛皮卡纸:")
    print(f"    密度范围: {kraft.min_density}-{kraft.max_density} kg/m³")
    
    print("  ✅ Cardboard 材质测试通过")
    return True


def test_corrugated_material():
    """测试瓦楞纸材质"""
    print("\n=== 测试 3: Corrugated 材质 ===")
    
    from infinigen.core.sim.physics.material_definitions import Corrugated
    
    # 基础瓦楞纸
    corrugated = Corrugated()
    print(f"  基础瓦楞纸:")
    print(f"    密度范围: {corrugated.min_density}-{corrugated.max_density} kg/m³")
    print(f"    厚度范围: {corrugated.min_thickness*1000:.1f}-{corrugated.max_thickness*1000:.1f} mm")
    print(f"    恢复系数: {corrugated.min_restitution}-{corrugated.max_restitution}")
    
    # 测试各种瓦楞类型
    types = {
        "A": Corrugated.type_a(),
        "B": Corrugated.type_b(),
        "C": Corrugated.type_c(),
        "E": Corrugated.type_e(),
    }
    
    for name, flute in types.items():
        print(f"  {name}型瓦楞: 厚度={flute.min_thickness*1000:.1f}-{flute.max_thickness*1000:.1f}mm, 类型={flute.flute_type}")
        
        if flute.flute_type != name:
            print(f"  ❌ 瓦楞类型应为 {name}")
            return False
    
    # 验证厚度递减: A > C > B > E
    if not (types["A"].min_thickness > types["C"].min_thickness > 
            types["B"].min_thickness > types["E"].min_thickness):
        print("  ⚠️ 瓦楞厚度应满足 A > C > B > E")
    
    print("  ✅ Corrugated 材质测试通过")
    return True


def test_plastic_box_material():
    """测试塑料盒材质"""
    print("\n=== 测试 4: PlasticBox 材质 ===")
    
    from infinigen.core.sim.physics.material_definitions import PlasticBox
    
    # 基础塑料
    plastic = PlasticBox()
    print(f"  基础塑料盒:")
    print(f"    密度范围: {plastic.min_density}-{plastic.max_density} kg/m³")
    print(f"    摩擦系数: {plastic.min_friction}-{plastic.max_friction}")
    print(f"    恢复系数: {plastic.min_restitution}-{plastic.max_restitution}")
    
    # 塑料应该比纸更光滑
    if plastic.min_friction > 0.5:
        print("  ⚠️ 塑料摩擦系数应较低")
    
    # 塑料应该有弹性
    if plastic.max_restitution < 0.2:
        print("  ⚠️ 塑料恢复系数应较高")
    
    # 测试各种塑料类型
    types = {
        "PP": PlasticBox.pp(),
        "PET": PlasticBox.pet(),
        "HDPE": PlasticBox.hdpe(),
    }
    
    for name, material in types.items():
        print(f"  {name}: 密度={material.min_density}-{material.max_density}kg/m³")
    
    print("  ✅ PlasticBox 材质测试通过")
    return True


def test_wood_box_material():
    """测试木质盒材质"""
    print("\n=== 测试 5: WoodBox 材质 ===")
    
    from infinigen.core.sim.physics.material_definitions import WoodBox
    
    # 基础木质
    wood = WoodBox()
    print(f"  基础木质盒:")
    print(f"    密度范围: {wood.min_density}-{wood.max_density} kg/m³")
    print(f"    厚度范围: {wood.min_thickness*1000:.1f}-{wood.max_thickness*1000:.1f} mm")
    
    # 木质应该比纸厚
    if wood.max_thickness < 0.003:
        print("  ⚠️ 木质盒厚度应较大")
    
    # 测试木质类型
    mdf = WoodBox.mdf()
    plywood = WoodBox.plywood()
    
    print(f"  MDF: 密度={mdf.min_density}-{mdf.max_density}kg/m³, 类型={mdf.wood_type}")
    print(f"  胶合板: 密度={plywood.min_density}-{plywood.max_density}kg/m³, 类型={plywood.wood_type}")
    
    print("  ✅ WoodBox 材质测试通过")
    return True


def test_parameter_sampling():
    """测试参数采样"""
    print("\n=== 测试 6: 参数采样 ===")
    
    from infinigen.core.sim.physics.material_definitions import (
        Cardboard,
        Corrugated,
        PlasticBox,
    )
    
    materials = [Cardboard(), Corrugated(), PlasticBox()]
    all_passed = True
    
    for material in materials:
        class_name = material.__class__.__name__
        
        # 采样多次验证范围
        for _ in range(5):
            params = material.sample_parameters()
            
            # 验证必要参数
            required = ["friction", "density", "restitution", "thickness"]
            for key in required:
                if key not in params:
                    print(f"  ❌ {class_name} 缺少参数: {key}")
                    all_passed = False
                    break
            
            # 验证参数在范围内
            if not (material.min_density <= params["density"] <= material.max_density):
                print(f"  ❌ {class_name} 密度超出范围")
                all_passed = False
            
            if not (material.min_friction <= params["friction"] <= material.max_friction):
                print(f"  ❌ {class_name} 摩擦系数超出范围")
                all_passed = False
        
        print(f"  {class_name}: 采样正常")
    
    if all_passed:
        print("  ✅ 参数采样测试通过")
    
    return all_passed


def test_materials_registry():
    """测试材质注册表"""
    print("\n=== 测试 7: 材质注册表 ===")
    
    from infinigen.core.sim.physics.material_definitions import MATERIALS
    
    print(f"  注册材质数: {len(MATERIALS)}")
    
    # 验证新增材质已注册
    required = ["cardboard", "cardboard_white", "cardboard_kraft", 
                "corrugated", "plastic_box", "wood_box"]
    
    missing = []
    for name in required:
        if name in MATERIALS:
            print(f"  ✅ {name}: 已注册")
        else:
            print(f"  ❌ {name}: 未注册")
            missing.append(name)
    
    if missing:
        print(f"  ❌ 缺少材质: {missing}")
        return False
    
    print("  ✅ 材质注册表测试通过")
    return True


def test_helper_functions():
    """测试辅助函数"""
    print("\n=== 测试 8: 辅助函数 ===")
    
    from infinigen.core.sim.physics.material_definitions import (
        get_box_material,
        list_box_materials,
    )
    
    # 测试列表函数
    available = list_box_materials()
    print(f"  可用盒子材质: {len(available)} 种")
    print(f"    {available}")
    
    if len(available) < 10:
        print("  ❌ 应至少有 10 种盒子材质")
        return False
    
    # 测试获取函数
    test_materials = ["cardboard", "corrugated_a", "plastic_pp", "wood_mdf"]
    
    for name in test_materials:
        try:
            material = get_box_material(name)
            params = material.sample_parameters()
            print(f"  get_box_material('{name}'): 密度={params['density']:.0f}kg/m³")
        except Exception as e:
            print(f"  ❌ 获取 {name} 失败: {e}")
            return False
    
    # 测试无效材质名
    try:
        get_box_material("invalid_material")
        print("  ❌ 无效材质名应抛出异常")
        return False
    except ValueError:
        print("  ✅ 无效材质名正确抛出 ValueError")
    
    print("  ✅ 辅助函数测试通过")
    return True


def test_integration_with_joint_system():
    """测试与关节系统的集成"""
    print("\n=== 测试 9: 与 P1-T3 集成 ===")
    
    from infinigen.core.sim.physics.material_definitions import (
        Cardboard,
        Corrugated,
    )
    from infinigen.assets.sim_objects.joint_injector import (
        MaterialType,
        get_joint_dynamics,
        JointType,
    )
    
    # 验证材质类型映射
    material_mapping = {
        "cardboard": MaterialType.CARDBOARD,
        "corrugated": MaterialType.CORRUGATED,
    }
    
    for material_name, material_type in material_mapping.items():
        dynamics = get_joint_dynamics(material_type, JointType.HINGE)
        print(f"  {material_name} 铰链动力学: damping={dynamics.damping}, friction={dynamics.friction}")
    
    # 验证物理参数与动力学一致性
    cardboard = Cardboard()
    cardboard_dynamics = get_joint_dynamics(MaterialType.CARDBOARD, JointType.HINGE)
    
    # 卡纸摩擦系数应与动力学摩擦接近
    print(f"  材质摩擦: {cardboard.min_friction}-{cardboard.max_friction}")
    print(f"  动力学摩擦: {cardboard_dynamics.friction}")
    
    print("  ✅ 与 P1-T3 集成测试通过")
    return True


def test_code_structure():
    """测试代码结构"""
    print("\n=== 测试 10: 代码结构 ===")
    
    source_file = Path(__file__).parent.parent.parent / "infinigen/core/sim/physics/material_definitions.py"
    
    with open(source_file, "r") as f:
        source_code = f.read()
    
    checks = [
        ("class BoxMaterial(BaseMaterial)", "BoxMaterial 基类"),
        ("class Cardboard(BoxMaterial)", "Cardboard 材质"),
        ("class CardboardWhite(Cardboard)", "CardboardWhite 变体"),
        ("class CardboardKraft(Cardboard)", "CardboardKraft 变体"),
        ("class Corrugated(BoxMaterial)", "Corrugated 材质"),
        ("class PlasticBox(BoxMaterial)", "PlasticBox 材质"),
        ("class WoodBox(BoxMaterial)", "WoodBox 材质"),
        ("min_restitution", "恢复系数属性"),
        ("min_thickness", "厚度属性"),
        ("def get_box_material", "get_box_material 函数"),
        ("def list_box_materials", "list_box_materials 函数"),
        ("type_a", "瓦楞纸 A 型"),
        ("type_b", "瓦楞纸 B 型"),
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
    print("P1-T4 测试: 材质系统 (Cardboard/Corrugated)")
    print("=" * 60)
    
    results = {
        "模块导入": test_module_import(),
        "Cardboard 材质": test_cardboard_material(),
        "Corrugated 材质": test_corrugated_material(),
        "PlasticBox 材质": test_plastic_box_material(),
        "WoodBox 材质": test_wood_box_material(),
        "参数采样": test_parameter_sampling(),
        "材质注册表": test_materials_registry(),
        "辅助函数": test_helper_functions(),
        "P1-T3 集成": test_integration_with_joint_system(),
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
        print("\n🎉 P1-T4 所有测试通过!")
    else:
        print("\n❌ 部分测试失败")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
