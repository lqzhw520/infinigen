#!/usr/bin/env python3
"""
Export SlipLidBox (0-1 DOF prismatic) URDF for verification.

Run:
  cd /mnt/afs2/zhuhaowu/infinigen
  python -m infinigen.launch_blender -s scripts/export_sliplidbox_urdf.py

Output:
  sim_exports/urdf/sliplidbox/<seed>/
"""

import sys
from pathlib import Path

import bpy

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def export_sliplidbox(seed: int = 42, output_dir: str = "./sim_exports/urdf"):
    from infinigen.assets.sim_objects.modular_box_factory import SlipLidBoxFactory
    from infinigen.core.init import configure_blender
    from infinigen.core.sim import kinematic_compiler
    from infinigen.core.sim.exporters import urdf_exporter
    from infinigen.core.util import blender as butil

    print("=" * 60)
    print("SlipLidBox (1-DOF) URDF export")
    print("=" * 60)

    try:
        configure_blender()
    except Exception as e:
        print(f"configure_blender warning: {e}")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    factory = SlipLidBoxFactory(factory_seed=seed)
    obj = factory.create_asset()

    sim_blueprint = kinematic_compiler.compile(obj)
    butil.apply_modifiers(obj)
    sim_blueprint["name"] = "sliplidbox"

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

    out_dir = export_path / "sliplidbox" / str(seed)
    print(f"✅ Exported: {out_dir}")
    return out_dir


if __name__ == "__main__":
    export_sliplidbox()

