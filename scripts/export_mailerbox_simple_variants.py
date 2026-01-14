#!/usr/bin/env python3
"""
批量导出 MailerBox-simple 变体（用于在线 URDF viewer 检查）

需求对应：
- 5 个：仅 EdgeExtension 不同（第二折页长度固定等于 Height）
- 5 个：EdgeExtension 不同 + 第二折页长度 FrontFlapLen 按 Height 比例随机（可不到底）

运行方法:
    cd /mnt/afs2/zhuhaowu/infinigen
    export INFINIGEN_PYTHONPATH=$(python -c "import site; print(':'.join(site.getsitepackages()))")
    python -m infinigen.launch_blender -s scripts/export_mailerbox_simple_variants.py

输出目录:
    sim_exports/urdf/mailerbox_simple/<seed>/
并在:
    sim_exports/urdf/mailerbox_simple/variants_manifest.json
写出每个 seed 的参数（W,D,H,T,E,FrontFlapLen 等）。
"""

from __future__ import annotations

import json
from pathlib import Path

import bpy


PROJECT_ROOT = Path(__file__).parent.parent


def _clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _export_one(
    *,
    seed: int,
    export_dir: Path,
    dims: "BoxDimensions",
    edge_ext: float,
    front_flap_len: float,
    image_res: int = 256,
):
    """
    导出单个变体。

    重要：为满足用户“只随机凸出长度”的对比需求，这里 **固定 W/D/H/T**，仅修改：
    - EdgeExtension（底板三侧凸出 + lid/front_flap 横向凸出，同步）
    - FrontFlapLen（仅在第二组 5 个中随机）
    """
    from infinigen.assets.sim_objects.modular_box_factory import (
        BoxMaterialConfig,
        BoxParameters,
        BoxType,
        MailerBoxFactory,
    )
    from infinigen.core.sim import kinematic_compiler
    from infinigen.core.sim.exporters import urdf_exporter
    from infinigen.core.util import blender as butil

    _clear_scene()

    factory = MailerBoxFactory(factory_seed=seed, randomize_front_flap_len=False)

    params = BoxParameters(
        box_type=BoxType.MAILER,
        dimensions=dims,
        material=BoxMaterialConfig(material_type="cardboard", density=300.0),
        joints=[],
        extra_params={
            "EdgeExtension": float(edge_ext),
            "FrontFlapLen": float(front_flap_len),
        },
    )
    params.joints = factory.get_default_joints(params)

    obj = factory.create_asset(asset_params=params)

    sim_blueprint = kinematic_compiler.compile(obj)
    butil.apply_modifiers(obj)
    sim_blueprint["name"] = "mailerbox_simple"

    urdf_exporter.export(
        blend_obj=obj,
        sim_blueprint=sim_blueprint,
        seed=seed,
        sample_joint_params_fn=factory.sample_joint_parameters,
        export_dir=export_dir,
        image_res=image_res,
        visual_only=True,
    )

    out_dir = export_dir / "mailerbox_simple" / str(seed)
    urdf_path = out_dir / "mailerbox_simple.urdf"

    return {
        "seed": seed,
        "dimensions": {
            "width": float(dims.width),
            "depth": float(dims.depth),
            "height": float(dims.height),
            "thickness": float(dims.thickness),
        },
        "edge_extension": float(edge_ext),
        "front_flap_len": float(front_flap_len),
        "export_dir": str(out_dir),
        "urdf": str(urdf_path),
    }


def main():
    from infinigen.core.init import configure_blender
    from infinigen.assets.sim_objects.modular_box_factory import BoxDimensions

    try:
        configure_blender()
    except Exception as e:
        # 非渲染任务允许忽略 cycles 配置缺参
        print(f"configure_blender warning: {e}")

    export_dir = PROJECT_ROOT / "sim_exports" / "urdf"
    export_dir.mkdir(parents=True, exist_ok=True)

    # 固定盒体尺寸（用户需求：前 5 个只随机凸出长度，不要混入盒体长宽高变化）
    base_dims = BoxDimensions(width=0.24, depth=0.18, height=0.09, thickness=0.0015)

    # 5 个（仅凸出长度 E 不同；FrontFlapLen 固定 = Height）
    seeds_a = [101, 102, 103, 104, 105]
    # 5 个（凸出长度 E 不同 + 第二折页长度 Lf 随机）
    seeds_b = [201, 202, 203, 204, 205]

    manifest = {
        "asset_name": "mailerbox_simple",
        "base_dimensions_fixed": {
            "width": float(base_dims.width),
            "depth": float(base_dims.depth),
            "height": float(base_dims.height),
            "thickness": float(base_dims.thickness),
        },
        "variants": [],
    }

    print("=" * 80)
    print("Exporting MailerBox-simple variants")
    print("=" * 80)

    # 需求：凸出长度要“明显有大有小”，用户示例为 1cm~15cm。
    # 这里直接用 5 个固定值（米），便于在线对比、可复现：
    edge_ext_values_a = [0.01, 0.04, 0.07, 0.11, 0.15]  # 1cm, 4cm, 7cm, 11cm, 15cm

    for s, edge_ext in zip(seeds_a, edge_ext_values_a):
        front_len = float(base_dims.height)
        rec = _export_one(seed=s, export_dir=export_dir, dims=base_dims, edge_ext=edge_ext, front_flap_len=front_len)
        manifest["variants"].append(rec)
        print(f"[A] seed={s} edge_ext={rec['edge_extension']:.6f} front_flap_len={rec['front_flap_len']:.6f}")

    # 第二组：E 仍取 5 个不同值；Lf 取 60%~100% 的 5 个比例点（第二折页可不到底）
    edge_ext_values_b = [0.02, 0.05, 0.08, 0.12, 0.15]  # 2cm, 5cm, 8cm, 12cm, 15cm
    flap_ratios = [0.60, 0.70, 0.80, 0.90, 1.00]

    for s, edge_ext, fr in zip(seeds_b, edge_ext_values_b, flap_ratios):
        front_len = float(fr * base_dims.height)
        rec = _export_one(seed=s, export_dir=export_dir, dims=base_dims, edge_ext=edge_ext, front_flap_len=front_len)
        manifest["variants"].append(rec)
        print(f"[B] seed={s} edge_ext={rec['edge_extension']:.6f} front_flap_len={rec['front_flap_len']:.6f}")

    manifest_path = export_dir / "mailerbox_simple" / "variants_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("=" * 80)
    print(f"✅ Wrote manifest: {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

