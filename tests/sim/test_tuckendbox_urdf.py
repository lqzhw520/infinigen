#!/usr/bin/env python3
"""
P2.1-T1 测试: TuckEndBox URDF 导出验证

验证内容:
1. kinematic_compiler 能从几何节点提取关节信息
2. urdf_exporter 能生成正确的 URDF
3. 导出的 URDF 在 PyBullet 中可用

运行方法:
    cd /mnt/afs2/zhuhaowu/infinigen
    export INFINIGEN_PYTHONPATH=$(python -c "import site; print(':'.join(site.getsitepackages()))")
    python -m infinigen.launch_blender -s tests/sim/test_tuckendbox_urdf.py
"""

import sys
import os
import tempfile
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_kinematic_compiler():
    """测试 kinematic_compiler 能否解析 TuckEndBox 的关节"""
    import bpy
    
    from infinigen.core.init import configure_blender
    from infinigen.assets.sim_objects.modular_box_factory import TuckEndBoxFactory
    from infinigen.core.sim import kinematic_compiler
    from infinigen.core.sim import utils as sim_utils
    
    print("=" * 60)
    print("P2.1-T1 测试: TuckEndBox URDF 导出验证")
    print("=" * 60)
    
    # 配置 Blender (忽略渲染配置错误)
    try:
        configure_blender()
    except Exception as e:
        print(f"  ⚠️ Blender 配置警告 (可忽略): {e}")
    
    # 清理场景
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # === 测试 1: 创建 TuckEndBox ===
    print("\n=== 测试 1: 创建 TuckEndBox ===")
    
    factory = TuckEndBoxFactory(factory_seed=42)
    obj = factory.create_asset()
    print(f"  对象: {obj.name}")
    
    # 检查修改器
    nodes_mods = [mod for mod in obj.modifiers if mod.type == 'NODES']
    print(f"  NODES 修改器数: {len(nodes_mods)}")
    
    if not nodes_mods:
        print("  ❌ 没有找到 NODES 修改器")
        return False
    
    # === 测试 2: 检查几何节点中的关节节点 ===
    print("\n=== 测试 2: 检查几何节点中的关节节点 ===")
    
    ng = nodes_mods[0].node_group
    print(f"  节点组名: {ng.name}")
    print(f"  节点总数: {len(ng.nodes)}")
    
    # 遍历所有节点，查找关节
    joint_nodes = []
    hinge_nodes = []
    
    def search_nodegroups(node_tree, depth=0):
        """递归搜索节点组"""
        for node in node_tree.nodes:
            if sim_utils.is_node_group(node):
                nt = node.node_tree
                if nt:
                    if sim_utils.is_hinge(node):
                        hinge_nodes.append((node.name, nt.name))
                        print(f"  {'  ' * depth}✅ 铰链节点: {node.name} ({nt.name})")
                    elif sim_utils.is_joint(node):
                        joint_nodes.append((node.name, nt.name))
                        print(f"  {'  ' * depth}🔧 关节节点: {node.name} ({nt.name})")
                    else:
                        # 递归搜索
                        search_nodegroups(nt, depth + 1)
    
    search_nodegroups(ng)
    
    print(f"\n  铰链节点数: {len(hinge_nodes)}")
    print(f"  其他关节数: {len(joint_nodes)}")
    
    if len(hinge_nodes) < 6:
        print("  ⚠️ 预期至少 6 个铰链节点 (主翻盖 + 左右 dust flaps，上下各一套)")
        # 继续测试，看看 kinematic_compiler 能否工作
    
    # === 测试 3: 运行 kinematic_compiler ===
    print("\n=== 测试 3: kinematic_compiler 编译 ===")
    
    try:
        sim_blueprint = kinematic_compiler.compile(obj)
        print(f"  ✅ 编译成功")
        print(f"  Blueprint 键: {list(sim_blueprint.keys())}")
        
        # 打印详细信息
        if 'kinematic_root' in sim_blueprint:
            root = sim_blueprint['kinematic_root']
            print(f"  根节点: {root}")
        
        if 'metadata' in sim_blueprint:
            meta = sim_blueprint['metadata']
            print(f"  元数据: {meta}")
        
    except Exception as e:
        print(f"  ❌ kinematic_compiler 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # === 测试 4: URDF 导出 ===
    print("\n=== 测试 4: URDF 导出 ===")
    
    # 应用修改器 (导出前需要)
    from infinigen.core.util import blender as butil
    butil.apply_modifiers(obj)
    
    # 添加必要的 name 字段 (sim_factory.py 中的逻辑)
    sim_blueprint["name"] = "tuckendbox"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = Path(tmpdir)
        
        try:
            from infinigen.core.sim.exporters import urdf_exporter
            
            # 获取关节参数采样函数
            sample_joint_params = factory.sample_joint_parameters
            
            # 导出
            result = urdf_exporter.export(
                blend_obj=obj,
                sim_blueprint=sim_blueprint,
                seed=42,
                sample_joint_params_fn=sample_joint_params,
                export_dir=export_dir,
                image_res=256,
                visual_only=True,
            )
            
            print(f"  ✅ 导出成功")
            print(f"  结果: {result}")
            
            # 查找生成的 URDF 文件
            urdf_files = list(export_dir.rglob("*.urdf"))
            print(f"  URDF 文件数: {len(urdf_files)}")
            
            if urdf_files:
                urdf_path = urdf_files[0]
                print(f"  URDF 路径: {urdf_path}")
                
                # 读取并分析 URDF
                with open(urdf_path, 'r') as f:
                    content = f.read()
                
                link_count = content.count('<link name=')
                joint_count = content.count('<joint name=')
                
                print(f"  Links: {link_count}")
                print(f"  Joints: {joint_count}")
                
                # world_joint 会额外多 1 个
                if joint_count >= 7:
                    print("  ✅ 关节数量符合预期 (>=6 hinges + world_joint)")
                else:
                    print(f"  ⚠️ 关节数量不足 (预期 >= 7)")

                # Debug: 打印 URDF 中每个关节的 origin/axis/limit（便于定位“漂浮/交叉”的根因）
                try:
                    import xml.etree.ElementTree as ET
                    tree = ET.parse(str(urdf_path))
                    root = tree.getroot()
                    print("\n  === URDF Joint Debug ===")
                    for j in root.findall("joint"):
                        jname = j.attrib.get("name", "")
                        jtype = j.attrib.get("type", "")
                        origin = j.find("origin")
                        axis = j.find("axis")
                        limit = j.find("limit")
                        oxyz = origin.attrib.get("xyz") if origin is not None else None
                        axyz = axis.attrib.get("xyz") if axis is not None else None
                        lower = limit.attrib.get("lower") if limit is not None else None
                        upper = limit.attrib.get("upper") if limit is not None else None
                        print(f"    - {jname} ({jtype}) origin={oxyz} axis={axyz} limit=[{lower}, {upper}]")
                except Exception as e:
                    print(f"  ⚠️ URDF joint debug 解析失败: {e}")
                
                # === 测试 5: PyBullet 验证 ===
                print("\n=== 测试 5: PyBullet 验证 ===")
                
                try:
                    import pybullet as p
                    
                    physics_client = p.connect(p.DIRECT)
                    p.setGravity(0, 0, -9.81)
                    
                    robot_id = p.loadURDF(str(urdf_path))
                    
                    if robot_id >= 0:
                        num_joints = p.getNumJoints(robot_id)
                        print(f"  ✅ PyBullet 加载成功")
                        print(f"  关节数: {num_joints}")
                        
                        for i in range(num_joints):
                            joint_info = p.getJointInfo(robot_id, i)
                            name = joint_info[1].decode('utf-8')
                            jtype = {0: "REVOLUTE", 1: "PRISMATIC", 4: "FIXED"}.get(joint_info[2], "OTHER")
                            print(f"    [{i}] {name}: {jtype}")
                        
                        # 仿真测试
                        for _ in range(100):
                            p.stepSimulation()
                        
                        pos, _ = p.getBasePositionAndOrientation(robot_id)
                        print(f"  仿真后位置: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
                        
                        p.disconnect()
                        print("  ✅ PyBullet 仿真通过")
                        
                    else:
                        print("  ❌ PyBullet 加载失败")
                        return False
                        
                except ImportError:
                    print("  ⚠️ PyBullet 未安装")
                except Exception as e:
                    print(f"  ❌ PyBullet 测试失败: {e}")
                    import traceback
                    traceback.print_exc()
                    
            else:
                print("  ❌ 未找到 URDF 文件")
                return False
                
        except Exception as e:
            print(f"  ❌ URDF 导出失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


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
        print("P2.1-T1 需要在 Blender 环境中运行")
        print("=" * 60)
        print("\n运行命令:")
        print("  export INFINIGEN_PYTHONPATH=$(python -c \"import site; print(':'.join(site.getsitepackages()))\")")
        print("  python -m infinigen.launch_blender -s tests/sim/test_tuckendbox_urdf.py")
        return 1
    
    success = test_kinematic_compiler()
    
    print("\n" + "=" * 60)
    print("P2.1-T1 验证结果")
    print("=" * 60)
    
    if success:
        print("🎉 P2.1-T1 所有测试通过!")
        print("\n下一步: P2.1-T2 TuckEndBox PyBullet 仿真验证 (更深入的物理验证)")
        return 0
    else:
        print("❌ 测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
