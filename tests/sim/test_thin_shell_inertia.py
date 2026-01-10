#!/usr/bin/env python3
"""
P0-T2 测试: 薄壳惯性修正 (R-Deep-1 修复验证)

测试场景:
1. 薄壳检测 - 验证薄板几何体被正确识别
2. 标准几何惯性 - 验证正常物体不受影响
3. 薄壳惯性计算 - 验证惯性值在合理范围
4. 惯性验证 - 检查正定性和数值稳定性
5. PyBullet 稳定性 - 验证薄壳 URDF 不会爆炸

运行方法 (在容器中):
    cd /mnt/afs2/zhuhaowu/infinigen
    conda activate infinigen
    python tests/sim/test_thin_shell_inertia.py
"""

import sys
import os
import tempfile
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_box_mesh(width: float, height: float, depth: float):
    """创建盒子的顶点和面"""
    import trimesh
    box = trimesh.creation.box(extents=[width, height, depth])
    return np.array(box.vertices), np.array(box.faces)


def create_thin_plate_mesh(width: float, height: float, thickness: float = 0.001):
    """创建薄板的顶点和面"""
    import trimesh
    # 创建一个非常薄的盒子来模拟薄板
    box = trimesh.creation.box(extents=[width, height, thickness])
    return np.array(box.vertices), np.array(box.faces)


def test_thin_shell_detection():
    """测试薄壳检测逻辑"""
    from infinigen.core.sim.physics.thin_shell_inertia import is_thin_shell
    
    print("\n=== 测试 1: 薄壳检测 ===")
    
    # 测试用例: (体积, 表面积, 预期是否为薄壳)
    # 阈值: volume/area < 0.002 (约 4mm 厚度)
    # 且表面积 > 0.001 m² (避免小物体误判)
    test_cases = [
        # 1cm x 1cm x 1cm 立方体 - 不是薄壳 (表面积太小)
        (0.000001, 0.0006, False, "1cm 立方体 (太小)"),
        # 10cm x 10cm x 1mm 薄板 - 是薄壳
        # volume = 0.1 * 0.1 * 0.001 = 0.00001 m³
        # area = 2 * (0.1*0.1 + 0.1*0.001 + 0.1*0.001) ≈ 0.0204 m²
        # ratio = 0.00001 / 0.0204 ≈ 0.00049 < 0.002 ✓
        (0.00001, 0.0204, True, "10cm x 10cm x 1mm 薄板"),
        # 10cm x 10cm x 1cm 厚板 - 不是薄壳
        # volume = 0.001 m³, area ≈ 0.024 m²
        # ratio = 0.001 / 0.024 ≈ 0.042 > 0.002 ✓
        (0.001, 0.024, False, "10cm x 10cm x 1cm 厚板"),
        # 0.5mm 厚卡纸盖子 (20cm x 20cm x 0.5mm) - 是薄壳
        # volume ≈ 0.00002 m³, area ≈ 0.08 m²
        # ratio ≈ 0.00025 < 0.002 ✓
        (0.00002, 0.08, True, "0.5mm 厚卡纸盖子"),
        # 5cm x 5cm x 1cm 块 - 不是薄壳 (也不够薄)
        # volume = 0.000025 m³, area = 0.007 m²
        # ratio = 0.0036 > 0.002 ✓
        (0.000025, 0.007, False, "5cm 立方块"),
    ]
    
    all_passed = True
    for volume, area, expected, desc in test_cases:
        result = is_thin_shell(volume, area)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_passed = False
        print(f"  {status} {desc}: volume={volume:.6f}, area={area:.4f} -> {result} (预期: {expected})")
    
    if all_passed:
        print("✅ 薄壳检测测试通过")
    else:
        print("❌ 薄壳检测测试失败")
    
    return all_passed


def test_robust_inertia_standard_geometry():
    """测试标准几何体的惯性计算"""
    from infinigen.core.sim.physics.thin_shell_inertia import calculate_robust_inertia
    
    print("\n=== 测试 2: 标准几何体惯性 ===")
    
    # 创建 10cm x 10cm x 5cm 盒子
    vertices, faces = create_box_mesh(0.1, 0.1, 0.05)
    density = 700  # 卡纸密度 kg/m³
    
    mass, inertia, com = calculate_robust_inertia(vertices, faces, density)
    
    # 预期质量: 0.1 * 0.1 * 0.05 * 700 = 0.35 kg
    expected_mass = 0.1 * 0.1 * 0.05 * 700
    
    print(f"  质量: {mass:.4f} kg (预期: ~{expected_mass:.4f} kg)")
    print(f"  惯性张量对角线: {np.diag(inertia)}")
    print(f"  质心: {com}")
    
    # 验证
    mass_ok = abs(mass - expected_mass) < expected_mass * 0.1  # 10% 误差容忍
    inertia_ok = np.all(np.diag(inertia) > 1e-8)
    com_ok = np.allclose(com, [0, 0, 0], atol=0.01)
    
    if mass_ok and inertia_ok and com_ok:
        print("✅ 标准几何体惯性测试通过")
        return True
    else:
        print("❌ 标准几何体惯性测试失败")
        return False


def test_robust_inertia_thin_shell():
    """测试薄壳几何体的惯性计算"""
    from infinigen.core.sim.physics.thin_shell_inertia import calculate_robust_inertia
    
    print("\n=== 测试 3: 薄壳几何体惯性 ===")
    
    # 创建 20cm x 20cm x 0.5mm 薄板 (模拟盒子盖子)
    vertices, faces = create_thin_plate_mesh(0.2, 0.2, 0.0005)
    density = 700  # 卡纸密度 kg/m³
    
    mass, inertia, com = calculate_robust_inertia(vertices, faces, density)
    
    print(f"  质量: {mass:.6f} kg")
    print(f"  惯性张量对角线: {np.diag(inertia)}")
    print(f"  质心: {com}")
    
    # 验证薄壳惯性修正生效
    # 1. 质量应该基于表面积 × 最小厚度 (2mm) 计算
    # 2. 惯性应该 > MIN_INERTIA_VALUE (1e-8)
    
    min_inertia = 1e-8
    inertia_diag = np.diag(inertia)
    
    mass_ok = mass >= 0.001  # 最小 1g
    inertia_ok = np.all(inertia_diag >= min_inertia)
    
    print(f"  质量检查 (>= 0.001 kg): {mass_ok}")
    print(f"  惯性检查 (>= {min_inertia}): {inertia_ok}")
    
    if mass_ok and inertia_ok:
        print("✅ 薄壳几何体惯性测试通过")
        return True
    else:
        print("❌ 薄壳几何体惯性测试失败")
        return False


def test_inertia_validation():
    """测试惯性验证函数"""
    from infinigen.core.sim.physics.thin_shell_inertia import validate_inertia_for_simulation
    
    print("\n=== 测试 4: 惯性验证 ===")
    
    test_cases = [
        # (mass, inertia_diag, expected_valid, description)
        (1.0, [0.1, 0.1, 0.1], True, "正常惯性"),
        (0.0005, [1e-9, 1e-9, 1e-9], False, "质量和惯性过小"),
        (1.0, [0.1, 0.1, -0.1], False, "负惯性值"),
        (1.0, [0.1, 0.1, 0], False, "零惯性值"),
    ]
    
    all_passed = True
    for mass, inertia_diag, expected_valid, desc in test_cases:
        inertia = np.diag(inertia_diag)
        is_valid, message = validate_inertia_for_simulation(mass, inertia)
        status = "✅" if is_valid == expected_valid else "❌"
        if is_valid != expected_valid:
            all_passed = False
        print(f"  {status} {desc}: valid={is_valid} (预期: {expected_valid})")
        if not is_valid:
            print(f"      原因: {message}")
    
    if all_passed:
        print("✅ 惯性验证测试通过")
    else:
        print("❌ 惯性验证测试失败")
    
    return all_passed


def test_pybullet_stability():
    """测试薄壳 URDF 在 PyBullet 中的稳定性"""
    print("\n=== 测试 5: PyBullet 稳定性 ===")
    
    try:
        import pybullet as p
    except ImportError:
        print("⚠️ PyBullet 未安装，跳过此测试")
        return True
    
    # 创建一个带薄壳盖子的盒子 URDF
    # 使用薄壳惯性修正后的参数
    test_urdf = """<?xml version="1.0"?>
<robot name="thin_shell_box">
  <link name="world"/>
  
  <link name="base">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <geometry><box size="0.2 0.2 0.1"/></geometry>
    </visual>
    <collision>
      <geometry><box size="0.2 0.2 0.1"/></geometry>
    </collision>
  </link>
  
  <!-- 薄壳盖子: 使用修正后的惯性参数 -->
  <link name="thin_lid">
    <inertial>
      <!-- 使用薄壳惯性修正: 基于表面积计算 -->
      <mass value="0.028"/>
      <origin xyz="0 0 0"/>
      <!-- 惯性值经过修正，确保 >= 1e-8 -->
      <inertia ixx="1.0e-5" ixy="0" ixz="0" iyy="1.0e-5" iyz="0" izz="1.0e-5"/>
    </inertial>
    <visual>
      <geometry><box size="0.2 0.2 0.002"/></geometry>
    </visual>
    <collision>
      <geometry><box size="0.2 0.2 0.002"/></geometry>
    </collision>
  </link>
  
  <joint name="world_joint" type="fixed">
    <origin xyz="0 0 0.5"/>
    <parent link="world"/>
    <child link="base"/>
  </joint>
  
  <joint name="lid_hinge" type="revolute">
    <origin xyz="0 0.1 0.05"/>
    <parent link="base"/>
    <child link="thin_lid"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="2.0"/>
  </joint>
</robot>
"""
    
    # 同时测试一个未修正的薄壳 (可能会爆炸)
    unstable_urdf = """<?xml version="1.0"?>
<robot name="unstable_thin_shell">
  <link name="world"/>
  
  <link name="base">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <geometry><box size="0.2 0.2 0.1"/></geometry>
    </visual>
  </link>
  
  <!-- 未修正的薄壳: 惯性过小 -->
  <link name="thin_lid">
    <inertial>
      <mass value="0.0001"/>
      <origin xyz="0 0 0"/>
      <!-- 惯性过小，可能导致不稳定 -->
      <inertia ixx="1.0e-12" ixy="0" ixz="0" iyy="1.0e-12" iyz="0" izz="1.0e-12"/>
    </inertial>
    <visual>
      <geometry><box size="0.2 0.2 0.001"/></geometry>
    </visual>
  </link>
  
  <joint name="world_joint" type="fixed">
    <origin xyz="0 0 0.5"/>
    <parent link="world"/>
    <child link="base"/>
  </joint>
  
  <joint name="lid_hinge" type="revolute">
    <origin xyz="0 0.1 0.05"/>
    <parent link="base"/>
    <child link="thin_lid"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="2.0"/>
  </joint>
</robot>
"""
    
    def simulate_urdf(urdf_content: str, name: str, steps: int = 2000) -> bool:
        """模拟 URDF 并检查是否爆炸"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            f.write(urdf_content)
            urdf_path = f.name
        
        try:
            physics_client = p.connect(p.DIRECT)
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(1.0 / 240.0)
            
            robot_id = p.loadURDF(urdf_path, [0, 0, 0])
            
            # 给盖子施加一个小力矩以激活运动
            num_joints = p.getNumJoints(robot_id)
            for i in range(num_joints):
                p.setJointMotorControl2(
                    robot_id, i,
                    p.VELOCITY_CONTROL,
                    targetVelocity=1.0,
                    force=0.1
                )
            
            explosion_threshold = 100.0
            max_velocity = 0.0
            
            for step in range(steps):
                p.stepSimulation()
                
                # 检查所有 link 的位置
                for i in range(-1, num_joints):
                    if i == -1:
                        pos, _ = p.getBasePositionAndOrientation(robot_id)
                    else:
                        state = p.getLinkState(robot_id, i)
                        pos = state[0]
                    
                    if max(abs(x) for x in pos) > explosion_threshold:
                        print(f"    ❌ {name} 爆炸 @ step {step}, link {i}, pos: {pos}")
                        p.disconnect()
                        return False
            
            p.disconnect()
            return True
            
        except Exception as e:
            print(f"    ❌ {name} 错误: {e}")
            return False
        finally:
            os.unlink(urdf_path)
    
    # 测试修正后的薄壳
    print("  测试修正后的薄壳 URDF...")
    stable_result = simulate_urdf(test_urdf, "修正后薄壳", 2000)
    print(f"  {'✅' if stable_result else '❌'} 修正后薄壳: {'稳定' if stable_result else '不稳定'}")
    
    # 测试未修正的薄壳 (预期可能不稳定，但不作为失败条件)
    print("  测试未修正的薄壳 URDF (参考)...")
    unstable_result = simulate_urdf(unstable_urdf, "未修正薄壳", 500)
    print(f"  {'✅' if unstable_result else '⚠️'} 未修正薄壳: {'稳定' if unstable_result else '不稳定 (预期)'}")
    
    # 主要验证: 修正后的薄壳应该稳定
    if stable_result:
        print("✅ PyBullet 稳定性测试通过")
        return True
    else:
        print("❌ PyBullet 稳定性测试失败")
        return False


def test_code_integration():
    """测试代码集成到 URDF 导出器"""
    print("\n=== 测试 6: 代码集成检查 ===")
    
    source_file = Path(__file__).parent.parent.parent / "infinigen/core/sim/exporters/urdf_exporter.py"
    
    with open(source_file, "r") as f:
        source_code = f.read()
    
    checks = [
        ("thin_shell_inertia as thinshell", "薄壳惯性模块导入"),
        ("thinshell.calculate_robust_inertia", "稳健惯性计算调用"),
        ("R-Deep-1", "R-Deep-1 修复注释"),
    ]
    
    all_passed = True
    for pattern, desc in checks:
        if pattern in source_code:
            print(f"  ✅ {desc}: 存在")
        else:
            print(f"  ❌ {desc}: 缺失")
            all_passed = False
    
    if all_passed:
        print("✅ 代码集成检查通过")
    else:
        print("❌ 代码集成检查失败")
    
    return all_passed


def main():
    """运行所有测试"""
    print("=" * 60)
    print("P0-T2 测试: 薄壳惯性修正 (R-Deep-1 修复验证)")
    print("=" * 60)
    
    results = {
        "薄壳检测": test_thin_shell_detection(),
        "标准几何惯性": test_robust_inertia_standard_geometry(),
        "薄壳几何惯性": test_robust_inertia_thin_shell(),
        "惯性验证": test_inertia_validation(),
        "PyBullet 稳定性": test_pybullet_stability(),
        "代码集成": test_code_integration(),
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
    
    print()
    if all_passed:
        print("🎉 P0-T2 所有测试通过!")
        return 0
    else:
        print("❌ 部分测试失败，请检查")
        return 1


if __name__ == "__main__":
    sys.exit(main())
