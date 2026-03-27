#!/usr/bin/env python3
"""Batch-export drawerbox URDF variants for MINT held-out simulation splits."""

from __future__ import annotations

import json
from pathlib import Path

import bpy

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
EXPORT_ROOT = PROJECT_ROOT / "sim_exports" / "urdf"
SEEDS = list(range(1, 16))


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def export_one(seed: int) -> dict:
    from infinigen.assets.sim_objects.modular_box_factory import DrawerBoxFactory
    from infinigen.core.sim import kinematic_compiler
    from infinigen.core.sim.exporters import urdf_exporter
    from infinigen.core.util import blender as butil

    clear_scene()
    factory = DrawerBoxFactory(factory_seed=seed)
    obj = factory.create_asset()
    sim_blueprint = kinematic_compiler.compile(obj)
    butil.apply_modifiers(obj)
    sim_blueprint["name"] = "drawerbox"
    urdf_exporter.export(
        blend_obj=obj,
        sim_blueprint=sim_blueprint,
        seed=seed,
        sample_joint_params_fn=factory.sample_joint_parameters,
        export_dir=EXPORT_ROOT,
        image_res=256,
        visual_only=False,
    )
    out_dir = EXPORT_ROOT / "drawerbox" / str(seed)
    return {
        "seed": seed,
        "export_dir": str(out_dir),
        "urdf": str(out_dir / "drawerbox.urdf"),
        "exists": (out_dir / "drawerbox.urdf").exists(),
    }


def main() -> None:
    from infinigen.core.init import configure_blender

    try:
        configure_blender()
    except Exception as exc:
        print(f"configure_blender warning: {exc}")

    manifest = {"asset_name": "drawerbox", "variants": []}
    for seed in SEEDS:
        target = EXPORT_ROOT / "drawerbox" / str(seed) / "drawerbox.urdf"
        if target.exists():
            record = {
                "seed": seed,
                "export_dir": str(target.parent),
                "urdf": str(target),
                "exists": True,
                "reused": True,
            }
        else:
            record = export_one(seed)
            record["reused"] = False
        manifest["variants"].append(record)
        print(f"seed={seed} reused={record['reused']} exists={record['exists']}")

    manifest_path = EXPORT_ROOT / "drawerbox" / "variants_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
