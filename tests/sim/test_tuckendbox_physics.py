#!/usr/bin/env python3
"""
P2.1-T2 测试: TuckEndBox PyBullet 深度物理验证

验证内容:
1. 质量范围验证 (0.001kg - 10kg)
2. 惯性张量正定性验证
3. 关节运动范围验证
4. 碰撞检测验证 (无自碰撞)
5. 仿真稳定性验证 (无"爆炸")
6. 关节动力学验证 (阻尼、摩擦)

运行方法:
    cd /mnt/afs2/zhuhaowu/infinigen
    export INFINIGEN_PYTHONPATH=$(python -c "import site; print(':'.join(site.getsitepackages()))")
    python -m infinigen.launch_blender -s tests/sim/test_tuckendbox_physics.py
"""

import sys
import os
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_pybullet_physics():
    """PyBullet 深度物理验证"""
    import bpy
    import pybullet as p
    
    from infinigen.core.init import configure_blender
    from infinigen.assets.sim_objects.modular_box_factory import TuckEndBoxFactory
    from infinigen.core.sim import kinematic_compiler
    from infinigen.core.sim.exporters import urdf_exporter
    from infinigen.core.util import blender as butil
    
    print("=" * 60)
    print("P2.1-T2 测试: TuckEndBox PyBullet 深度物理验证")
    print("=" * 60)
    
    # 配置 Blender (忽略渲染配置错误)
    try:
        configure_blender()
    except Exception:
        pass
    
    # 清理场景
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # === 步骤 1: 创建并导出 TuckEndBox ===
    print("\n=== 步骤 1: 创建并导出 TuckEndBox ===")
    
    factory = TuckEndBoxFactory(factory_seed=42)
    obj = factory.create_asset()
    
    sim_blueprint = kinematic_compiler.compile(obj)
    butil.apply_modifiers(obj)
    sim_blueprint["name"] = "tuckendbox"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = Path(tmpdir)
        
        result = urdf_exporter.export(
            blend_obj=obj,
            sim_blueprint=sim_blueprint,
            seed=42,
            sample_joint_params_fn=factory.sample_joint_parameters,
            export_dir=export_dir,
            image_res=256,
            visual_only=True,
        )
        
        urdf_path = list(export_dir.rglob("*.urdf"))[0]
        print(f"  URDF: {urdf_path}")
        
        # 打印 URDF 中的惯性信息
        import xml.etree.ElementTree as ET
        tree = ET.parse(urdf_path)
        root_xml = tree.getroot()
        print("\n  URDF 惯性信息:")
        for link_elem in root_xml.findall("link"):
            link_name = link_elem.get("name")
            inertial = link_elem.find("inertial")
            if inertial is not None:
                mass_elem = inertial.find("mass")
                inertia_elem = inertial.find("inertia")
                if mass_elem is not None and inertia_elem is not None:
                    mass_val = float(mass_elem.get("value", "0"))
                    ixx = float(inertia_elem.get("ixx", "0"))
                    iyy = float(inertia_elem.get("iyy", "0"))
                    izz = float(inertia_elem.get("izz", "0"))
                    print(f"    {link_name}: mass={mass_val:.6f}, I=({ixx:.2e}, {iyy:.2e}, {izz:.2e})")
        
        # === 测试 1: 质量范围验证 ===
        print("\n=== 测试 1: 质量范围验证 ===")
        
        physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        
        robot_id = p.loadURDF(str(urdf_path))
        
        num_joints = p.getNumJoints(robot_id)
        total_mass = 0
        link_masses = []
        
        # 获取基座质量 (world link, 期望为 0)
        base_mass = p.getDynamicsInfo(robot_id, -1)[0]
        # 注意: world/base link 是固定参考点，不参与质量计算
        print(f"  base (world) 质量: {base_mass:.6f} kg (固定参考点)")
        
        # 获取各链接质量
        for i in range(num_joints):
            mass = p.getDynamicsInfo(robot_id, i)[0]
            joint_info = p.getJointInfo(robot_id, i)
            link_name = joint_info[12].decode('utf-8')
            total_mass += mass
            link_masses.append((link_name, mass))
            print(f"  {link_name} 质量: {mass:.6f} kg")
        
        print(f"  总质量: {total_mass:.4f} kg")
        
        # 验证质量范围 (方案文档: 0.001kg - 10kg)
        mass_valid = 0.001 <= total_mass <= 10
        if mass_valid:
            print("  ✅ 质量范围合理 (0.001-10 kg)")
        else:
            print(f"  ❌ 质量超出范围 (0.001-10 kg)")
        
        # 检查每个非 world 链接质量不为零
        zero_mass_links = [name for name, m in link_masses if m <= 0]
        if zero_mass_links:
            print(f"  ⚠️ 零质量链接 (非 world): {zero_mass_links}")
            mass_valid = False
        
        # === 测试 2: 惯性张量验证 ===
        print("\n=== 测试 2: 惯性张量验证 (使用 URDF 数据) ===")
        
        # PyBullet getDynamicsInfo 返回的惯性可能是内部简化值
        # 更可靠的方法是直接验证 URDF 中的惯性值
        inertia_valid = True
        
        for link_elem in root_xml.findall("link"):
            link_name = link_elem.get("name")
            
            # 跳过 world link (固定参考点)
            if link_name == "world":
                print(f"  {link_name}: (固定参考点, 跳过)")
                continue
            
            inertial = link_elem.find("inertial")
            if inertial is None:
                print(f"  {link_name}: 无惯性定义")
                inertia_valid = False
                continue
            
            inertia_elem = inertial.find("inertia")
            if inertia_elem is None:
                print(f"  {link_name}: 无惯性张量")
                inertia_valid = False
                continue
            
            ixx = float(inertia_elem.get("ixx", "0"))
            iyy = float(inertia_elem.get("iyy", "0"))
            izz = float(inertia_elem.get("izz", "0"))
            
            print(f"  {link_name}: I=({ixx:.2e}, {iyy:.2e}, {izz:.2e})")
            
            # 检查对角元素是否为正
            if ixx <= 0 or iyy <= 0 or izz <= 0:
                print(f"    ⚠️ 惯性非正定!")
                inertia_valid = False
        
        if inertia_valid:
            print("  ✅ 所有链接惯性张量正定")
        
        # === 测试 3: 关节运动范围验证 ===
        print("\n=== 测试 3: 关节运动范围验证 ===")
        
        joint_valid = True
        revolute_joints = []
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            
            if joint_type == 0:  # REVOLUTE
                revolute_joints.append((i, joint_name, lower_limit, upper_limit))
                range_deg = (upper_limit - lower_limit) * 180 / np.pi
                print(f"  {joint_name}: [{lower_limit:.2f}, {upper_limit:.2f}] rad ({range_deg:.1f}°)")
                
                if upper_limit <= lower_limit:
                    print(f"    ⚠️ 关节范围无效!")
                    joint_valid = False
        
        if joint_valid and len(revolute_joints) >= 2:
            print("  ✅ 关节运动范围有效")
        elif len(revolute_joints) < 2:
            print("  ⚠️ 预期至少 2 个 REVOLUTE 关节")
            joint_valid = False
        
        # === 测试 4: 仿真稳定性验证 (无爆炸) ===
        print("\n=== 测试 4: 仿真稳定性验证 ===")
        
        # 记录初始位置
        initial_pos, _ = p.getBasePositionAndOrientation(robot_id)
        
        # 运行 1000 步仿真
        max_steps = 1000
        explosion_threshold = 100.0  # 位移超过 100m 视为爆炸
        stable = True
        
        for step in range(max_steps):
            p.stepSimulation()
            
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            displacement = np.linalg.norm(np.array(pos) - np.array(initial_pos))
            
            if displacement > explosion_threshold:
                print(f"  ❌ 仿真爆炸! 步骤 {step}, 位移 {displacement:.1f}m")
                stable = False
                break
            
            # 检查关节位置是否合理
            for i, name, lower, upper in revolute_joints:
                joint_state = p.getJointState(robot_id, i)
                joint_pos = joint_state[0]
                # 允许超出范围一点点 (由于数值误差)
                if joint_pos < lower - 0.5 or joint_pos > upper + 0.5:
                    print(f"  ⚠️ 关节 {name} 超出范围: {joint_pos:.2f}")
        
        final_pos, _ = p.getBasePositionAndOrientation(robot_id)
        final_displacement = np.linalg.norm(np.array(final_pos) - np.array(initial_pos))
        
        if stable:
            print(f"  仿真 {max_steps} 步后位移: {final_displacement:.4f}m")
            if final_displacement < 1.0:  # 位移小于 1m 视为稳定
                print("  ✅ 仿真稳定")
            else:
                print("  ⚠️ 仿真有明显漂移")
                stable = False
        
        # === 测试 5: 关节动力学验证 ===
        print("\n=== 测试 5: 关节动力学验证 ===")
        
        dynamics_valid = True
        
        # 重置仿真
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        robot_id = p.loadURDF(str(urdf_path), basePosition=[0, 0, 0.5])
        
        # 设置关节目标并观察响应
        for i, name, lower, upper in revolute_joints:
            target = (lower + upper) / 2  # 目标：范围中点
            p.setJointMotorControl2(
                robot_id, i,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=10.0,  # 限制力矩
            )
        
        # 运行 200 步
        for _ in range(200):
            p.stepSimulation()
        
        # 检查关节是否响应
        for i, name, lower, upper in revolute_joints:
            target = (lower + upper) / 2
            joint_state = p.getJointState(robot_id, i)
            actual = joint_state[0]
            error = abs(actual - target)
            
            print(f"  {name}: 目标 {target:.2f} rad, 实际 {actual:.2f} rad, 误差 {error:.2f}")
            
            # 误差应该在合理范围内 (阻尼会导致一些滞后)
            if error > 1.0:  # 误差超过 1 rad 认为有问题
                print(f"    ⚠️ 关节响应异常")
                dynamics_valid = False
        
        if dynamics_valid:
            print("  ✅ 关节动力学响应正常")
        
        # === 测试 6: 碰撞检测验证 ===
        print("\n=== 测试 6: 碰撞检测验证 ===")
        
        # 检查初始状态是否有自碰撞
        p.stepSimulation()
        contact_points = p.getContactPoints(robot_id, robot_id)
        
        if len(contact_points) > 0:
            print(f"  ⚠️ 检测到 {len(contact_points)} 个自碰撞点")
            # 打印碰撞详情
            for cp in contact_points[:3]:  # 只打印前 3 个
                linkA = cp[3]
                linkB = cp[4]
                print(f"    链接 {linkA} <-> 链接 {linkB}")
        else:
            print("  ✅ 无自碰撞")
        
        p.disconnect()
        
        # === 总结 ===
        print("\n" + "=" * 60)
        print("P2.1-T2 验证总结")
        print("=" * 60)
        
        results = {
            "质量范围": mass_valid,
            "惯性张量正定": inertia_valid,
            "关节运动范围": joint_valid,
            "仿真稳定性": stable,
            "关节动力学": dynamics_valid,
            "无自碰撞": len(contact_points) == 0,
        }
        
        all_passed = True
        for name, passed in results.items():
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"  {name}: {status}")
            if not passed:
                all_passed = False
        
        return all_passed


def main():
    # 检测是否在 Blender 环境中
    try:
        import bpy
        if hasattr(bpy, 'app') and hasattr(bpy.app, 'version'):
            in_blender = True
        else:
            in_blender = False
    except ImportError:
        in_blender = False
    
    if not in_blender:
        print("=" * 60)
        print("P2.1-T2 需要在 Blender 环境中运行")
        print("=" * 60)
        print("\n运行命令:")
        print("  export INFINIGEN_PYTHONPATH=$(python -c \"import site; print(':'.join(site.getsitepackages()))\")")
        print("  python -m infinigen.launch_blender -s tests/sim/test_tuckendbox_physics.py")
        return 1
    
    success = test_pybullet_physics()
    
    if success:
        print("\n🎉 P2.1-T2 所有测试通过!")
        print("\n下一步: 进入 P2.1-T3~T8 实现其他 6 种盒型")
        print("  - MailerBox (飞机盒)")
        print("  - DrawerBox (抽屉盒)")
        print("  - SlipLidBox (天地盒)")
        print("  - GiftBoxWithHandle (自带手提礼盒)")
        print("  - DoubleLidGiftBox (双盖手提礼盒)")
        print("  - PlasticHandleBox (塑料手提礼盒)")
        return 0
    else:
        print("\n❌ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
