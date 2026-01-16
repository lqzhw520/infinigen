#!/usr/bin/env python3
"""
导出 MailerBox-simple（2-DOF）URDF
用于网页 URDF 查看器验证

运行方法:
    cd /mnt/afs2/zhuhaowu/infinigen
    export INFINIGEN_PYTHONPATH=$(python -c "import site; print(':'.join(site.getsitepackages()))")
    python -m infinigen.launch_blender -s scripts/export_mailerbox_simple_urdf.py

导出结果:
    sim_exports/urdf/mailerbox_simple/42/
    ├── mailerbox_simple.urdf
    ├── assets/
    │   ├── geom_*.obj
    └── metadata.json
"""

import bpy
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def export_mailerbox_simple(seed: int = 42, output_dir: str = "./sim_exports/urdf"):
    """导出 MailerBox-simple 到指定目录"""
    from infinigen.core.init import configure_blender
    from infinigen.assets.sim_objects.modular_box_factory import MailerBoxFactory
    from infinigen.core.sim import kinematic_compiler
    from infinigen.core.sim.exporters import urdf_exporter
    from infinigen.core.util import blender as butil

    print("=" * 60)
    print("MailerBox-simple (2-DOF) URDF 导出")
    print("=" * 60)

    # 配置 Blender
    try:
        configure_blender()
    except Exception as e:
        print(f"配置 Blender 时警告: {e}")

    # 清理场景
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # 创建 MailerBox-simple
    print("\n1. 创建 MailerBox-simple...")
    factory = MailerBoxFactory(factory_seed=seed)
    obj = factory.create_asset()
    print(f"   对象: {obj.name}")

    # 编译运动学图
    print("\n2. 编译运动学图...")
    sim_blueprint = kinematic_compiler.compile(obj)
    butil.apply_modifiers(obj)
    sim_blueprint["name"] = "mailerbox_simple"

    # 关节数量统计（用于 sanity check）
    if "kinematic_root" in sim_blueprint:
        root = sim_blueprint["kinematic_root"]
        joint_count = 0

        def count_joints(node):
            nonlocal joint_count
            if (
                hasattr(node, "joint_type")
                and hasattr(node.joint_type, "value")
                and node.joint_type.value != 0
            ):
                joint_count += 1
            if hasattr(node, "children"):
                for children in node.children.values():
                    for child_node in children:
                        count_joints(child_node)

        count_joints(root)
        print(f"   关节数量: {joint_count} (期望: 2)")
    else:
        print(f"   sim_blueprint keys: {list(sim_blueprint.keys())}")

    # 导出 URDF
    print("\n3. 导出 URDF...")
    export_path = Path(output_dir)
    urdf_exporter.export(
        blend_obj=obj,
        sim_blueprint=sim_blueprint,
        seed=seed,
        sample_joint_params_fn=factory.sample_joint_parameters,
        export_dir=export_path,
        image_res=256,
        visual_only=False,
    )

    # 打印结果
    urdf_dir = export_path / "mailerbox_simple" / str(seed)
    urdf_file = urdf_dir / "mailerbox_simple.urdf"
    assets_dir = urdf_dir / "assets"

    print("\n" + "=" * 60)
    print("✅ 导出完成!")
    print("=" * 60)
    print(f"\n📁 导出目录: {urdf_dir.absolute()}")
    print(f"\n📄 URDF 文件: {urdf_file}")
    print(f"\n📦 Mesh 资产:")
    if assets_dir.exists():
        for f in sorted(assets_dir.glob("*.obj")):
            print(f"   - {f.name}")

    # 打印 URDF 内容预览
    print("\n📝 URDF 内容预览:")
    print("-" * 40)
    with open(urdf_file, "r") as f:
        lines = f.read().split("\n")[:80]
        print("\n".join(lines))
        if len(lines) >= 80:
            print("... (更多内容省略)")

    print("\n" + "=" * 60)
    print("使用说明")
    print("=" * 60)
    print(
        f"""
1. 下载整个导出目录:
   scp -r user@host:{urdf_dir.absolute()} ./

2. 使用在线 URDF 查看器:
   - https://gkjohnson.github.io/urdf-loaders/javascript/example/bundle/
   - 上传 URDF 文件和 assets/ 目录中的 OBJ 文件

3. 建议验证的闭合姿态:
   - mailer_lid = π/2
   - mailer_front_flap = π/2

4. 或使用 PyBullet GUI:
   python -c "import pybullet as p; p.connect(p.GUI); p.loadURDF('{urdf_file.absolute()}')"
"""
    )

    return urdf_dir


if __name__ == "__main__":
    # 检测是否在 Blender 环境
    try:
        import bpy  # noqa: F401

        if hasattr(bpy, "app") and hasattr(bpy.app, "version"):
            export_mailerbox_simple()
        else:
            print("请在 Blender 环境中运行此脚本")
    except ImportError:
        print("请使用以下命令运行:")
        print("  python -m infinigen.launch_blender -s scripts/export_mailerbox_simple_urdf.py")

