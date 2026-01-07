#!/usr/bin/env python
# verify_sim_assets.py
#
# 通用可动资产验证脚本，支持所有 Infinigen 可动资产类型
# 适用于 A800 无桌面环境（Headless）
#
# 支持的资产类型：
#   door, toaster, dishwasher, lamp, cabinet, drawer, refrigerator,
#   oven, microwave, soap_dispenser, faucet, plier, window, box,
#   pepper_grinder, trash, door_handle, stovetop

# ⚠️ 必须在 import mujoco 之前设置环境变量！
import os
import sys

# 解析命令行参数以提前获取 gl_backend 和 render 标志
render_mode = "--render" in sys.argv
gl_backend = "osmesa"  # 默认使用 OSMesa（最稳定）

for i, arg in enumerate(sys.argv):
    if arg == "--gl_backend" and i + 1 < len(sys.argv):
        gl_backend = sys.argv[i + 1]

# 只有在渲染模式下才设置 OpenGL 环境变量
if render_mode:
    os.environ["MUJOCO_GL"] = gl_backend
    print(f"[Init] MUJOCO_GL={gl_backend}")

# 现在才 import mujoco
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: opencv-python not installed. Video rendering disabled.")
    print("Install with: pip install opencv-python")

# 支持的资产类型列表
SUPPORTED_ASSETS = [
    "door",
    "toaster",
    "dishwasher",
    "lamp",
    "cabinet",
    "drawer",
    "refrigerator",
    "oven",
    "microwave",
    "soap_dispenser",
    "faucet",
    "plier",
    "window",
    "box",
    "pepper_grinder",
    "trash",
    "door_handle",
    "stovetop",
]


def find_xml_file(asset_dir: Path, seed: int, asset_name: str) -> Path:
    """
    查找资产的 XML 文件
    尝试多种可能的文件名
    """
    seed_dir = asset_dir / str(seed)

    # 尝试的文件名列表
    possible_names = [
        f"{asset_name}.xml",
        f"{asset_name.replace('_', '')}.xml",  # soap_dispenser -> soapdispenser
        "asset.xml",
        "model.xml",
    ]

    for name in possible_names:
        xml_path = seed_dir / name
        if xml_path.exists():
            return xml_path

    # 如果都没找到，尝试查找目录下的任何 .xml 文件
    if seed_dir.exists():
        xml_files = list(seed_dir.glob("*.xml"))
        if xml_files:
            return xml_files[0]

    return None


def verify_single_asset(xml_path: Path):
    """验证单个资产（不需要渲染）"""
    print(f"\n{'='*60}")
    print(f"Verifying: {xml_path}")
    print("=" * 60)

    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    print("\n[Basic Info]")
    print(f"  Bodies:  {model.nbody}")
    print(f"  Joints:  {model.njnt}")
    print(f"  Geoms:   {model.ngeom}")
    print(f"  Meshes:  {model.nmesh}")

    print("\n[Joint Details]")
    if model.njnt == 0:
        print("  ⚠️  No joints found in this asset")

    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jtype_id = model.jnt_type[i]
        jtype_names = ["FREE", "BALL", "SLIDE", "HINGE"]
        jtype = jtype_names[jtype_id] if jtype_id < len(jtype_names) else "UNKNOWN"

        range_min = model.jnt_range[i, 0]
        range_max = model.jnt_range[i, 1]
        axis = model.jnt_axis[i]

        print(f"  Joint {i}: {name}")
        print(f"    Type:  {jtype}")
        print(f"    Range: [{range_min:.3f}, {range_max:.3f}] rad")
        print(
            f"           [{np.degrees(range_min):.1f}°, {np.degrees(range_max):.1f}°]"
        )
        print(f"    Axis:  [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")

    print("\n[Motion Simulation Test]")
    if model.njnt == 0:
        print("  ⚠️  Skipped (no joints)")
        return True

    all_valid = True
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        range_min = model.jnt_range[i, 0]
        range_max = model.jnt_range[i, 1]

        test_positions = np.linspace(range_min, range_max, 20)
        errors = []

        for pos in test_positions:
            data.qpos[model.jnt_qposadr[i]] = pos
            mujoco.mj_forward(model, data)

            if np.any(np.isnan(data.xpos)):
                errors.append(f"NaN at {pos:.2f}")
            elif np.any(np.isinf(data.xpos)):
                errors.append(f"Inf at {pos:.2f}")

        if errors:
            print(f"  ❌ {name}: {len(errors)} errors")
            all_valid = False
        else:
            print(f"  ✅ {name}: All 20 positions valid")

    return all_valid


def modify_xml_for_offscreen(
    xml_path: Path,
    width: int,
    height: int,
    camera_distance: float = 2.0,
    center: np.ndarray = None,
) -> str:
    """修改 XML 以支持更大的 offscreen buffer 并添加光照和相机"""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # 默认中心点
    if center is None:
        center = np.array([0.0, 0.0, 0.0])

    # 1. 配置 visual/global (offscreen buffer尺寸)
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")

    global_elem = visual.find("global")
    if global_elem is None:
        global_elem = ET.SubElement(visual, "global")

    global_elem.set("offwidth", str(width))
    global_elem.set("offheight", str(height))

    # 2. 添加headlight用于基础照明
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("ambient", "0.3 0.3 0.3")
    headlight.set("diffuse", "0.6 0.6 0.6")
    headlight.set("specular", "0.2 0.2 0.2")

    # 3. 添加光照到 worldbody
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    # 移除已存在的同名光源（避免重复）
    for light in worldbody.findall("light"):
        if light.get("name") in [
            "render_main_light",
            "render_fill_light",
            "render_back_light",
        ]:
            worldbody.remove(light)

    # 根据资产中心位置调整光源位置
    cx, cy, cz = center

    # 添加主方向光（从上方，提供主要照明）
    light1 = ET.SubElement(worldbody, "light")
    light1.set("name", "render_main_light")
    light1.set("directional", "true")
    light1.set("diffuse", "1.0 1.0 1.0")
    light1.set("specular", "0.3 0.3 0.3")
    light1.set("pos", f"{cx} {cy} {cz + camera_distance * 2}")
    light1.set("dir", "0 0 -1")
    light1.set("castshadow", "false")

    # 添加辅助光源（从侧前方，填充阴影）
    light2 = ET.SubElement(worldbody, "light")
    light2.set("name", "render_fill_light")
    light2.set("directional", "true")
    light2.set("diffuse", "0.7 0.7 0.7")
    light2.set("specular", "0.1 0.1 0.1")
    light2.set(
        "pos", f"{cx + camera_distance} {cy + camera_distance} {cz + camera_distance}"
    )
    light2.set("dir", "-0.5 -0.5 -0.7")
    light2.set("castshadow", "false")

    # 添加背光（从后方，提供轮廓光）
    light3 = ET.SubElement(worldbody, "light")
    light3.set("name", "render_back_light")
    light3.set("directional", "true")
    light3.set("diffuse", "0.5 0.5 0.5")
    light3.set("specular", "0.0 0.0 0.0")
    light3.set(
        "pos", f"{cx - camera_distance} {cy - camera_distance} {cz + camera_distance}"
    )
    light3.set("dir", "0.5 0.5 -0.7")
    light3.set("castshadow", "false")

    # 添加灰色地面作为参考背景
    existing_ground = worldbody.find(".//geom[@name='render_ground']")
    if existing_ground is None:
        ground = ET.SubElement(worldbody, "geom")
        ground.set("name", "render_ground")
        ground.set("type", "plane")
        ground.set("size", f"{camera_distance * 3} {camera_distance * 3} 0.1")
        ground.set("pos", f"{cx} {cy} -0.01")  # 稍微在原点下方
        ground.set("rgba", "0.6 0.6 0.6 1")  # 浅灰色地面

    # 4. 移除已存在的同名相机并添加新相机
    for cam in worldbody.findall("camera"):
        if cam.get("name") == "render_camera":
            worldbody.remove(cam)

    # 计算相机位置 (45度俯视角)
    angle = np.radians(45)  # 方位角
    elev = np.radians(30)  # 仰角
    cam_x = cx + camera_distance * np.cos(elev) * np.cos(angle)
    cam_y = cy + camera_distance * np.cos(elev) * np.sin(angle)
    cam_z = cz + camera_distance * np.sin(elev)

    camera = ET.SubElement(worldbody, "camera")
    camera.set("name", "render_camera")
    camera.set("pos", f"{cam_x:.4f} {cam_y:.4f} {cam_z:.4f}")
    camera.set("mode", "targetbody")
    # 找到第一个body作为目标
    first_body = worldbody.find("body")
    if first_body is not None and first_body.get("name"):
        camera.set("target", first_body.get("name"))
    else:
        # 如果没有目标body，使用lookat方式
        camera.attrib.pop("mode", None)
        camera.attrib.pop("target", None)
        # 使用euler角度指向中心
        dx, dy, dz = cx - cam_x, cy - cam_y, cz - cam_z
        camera.set("xyaxes", "-0.707 0.707 0 -0.408 -0.408 0.816")

    temp_xml_path = xml_path.parent / "_temp_render.xml"
    tree.write(str(temp_xml_path), encoding="unicode")
    return str(temp_xml_path)


def render_joint_motion(
    xml_path: Path, output_dir: Path, width: int = 256, height: int = 256
):
    """渲染关节运动序列"""
    if not HAS_CV2:
        print("  ⚠️  Skipping render: opencv-python not installed")
        return False

    # 步骤1: 先加载原始模型计算边界框
    try:
        temp_model = mujoco.MjModel.from_xml_path(str(xml_path))
        temp_data = mujoco.MjData(temp_model)
        mujoco.mj_forward(temp_model, temp_data)

        # 计算所有geom的边界框
        min_pos = np.full(3, np.inf)
        max_pos = np.full(3, -np.inf)
        for i in range(temp_model.ngeom):
            pos = temp_data.geom_xpos[i]
            min_pos = np.minimum(min_pos, pos)
            max_pos = np.maximum(max_pos, pos)

        # 也考虑body位置
        for i in range(temp_model.nbody):
            pos = temp_data.xpos[i]
            min_pos = np.minimum(min_pos, pos)
            max_pos = np.maximum(max_pos, pos)

        center = (min_pos + max_pos) / 2
        bbox_diagonal = np.linalg.norm(max_pos - min_pos)
        # 使用更紧凑的相机距离，让物体更大更清晰
        camera_distance = max(bbox_diagonal * 1.5, 0.3)  # 更近的距离

        print(
            f"  [Camera] Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), Distance: {camera_distance:.2f}"
        )

    except Exception as e:
        print(f"  ⚠️  Failed to compute bounding box: {e}, using defaults")
        center = np.array([0.0, 0.0, 0.0])
        camera_distance = 2.0

    # 步骤2: 用计算出的相机参数修改XML
    modified_xml = modify_xml_for_offscreen(
        xml_path, width, height, camera_distance, center
    )

    try:
        model = mujoco.MjModel.from_xml_path(modified_xml)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
    except Exception as e:
        print(f"  ❌ Failed to create renderer: {e}")
        if os.path.exists(modified_xml):
            os.unlink(modified_xml)
        return False

    if model.njnt == 0:
        print("  ⚠️  No joints to render")
        renderer.close()
        if os.path.exists(modified_xml):
            os.unlink(modified_xml)
        return True

    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找我们添加的相机ID
    cam_id = -1
    for i in range(model.ncam):
        cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if cam_name == "render_camera":
            cam_id = i
            break

    for j in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        # 处理可能的空名称
        if jname is None:
            jname = f"joint_{j}"
        # 清理文件名中的特殊字符
        jname_safe = jname.replace("/", "_").replace("\\", "_").replace(" ", "_")

        range_min = model.jnt_range[j, 0]
        range_max = model.jnt_range[j, 1]
        range_delta = abs(range_max - range_min)

        # 警告小范围关节（动作可能不明显）
        if range_delta < 0.1:  # 小于 ~5.7 度
            print(
                f"  ⚠️ {jname}: Very small range ({np.degrees(range_delta):.1f}°), motion may be subtle"
            )

        frames = []
        positions = np.linspace(range_min, range_max, 30)
        for t, pos in enumerate(positions):
            # 重置所有关节到中间位置（更自然的姿态）
            for k in range(model.njnt):
                mid_pos = (model.jnt_range[k, 0] + model.jnt_range[k, 1]) / 2
                data.qpos[model.jnt_qposadr[k]] = mid_pos
            # 设置当前关节到目标位置
            data.qpos[model.jnt_qposadr[j]] = pos
            mujoco.mj_forward(model, data)

            # 使用我们添加的相机，如果失败则使用默认
            if cam_id >= 0:
                renderer.update_scene(data, camera=cam_id)
            else:
                renderer.update_scene(data)

            pixels = renderer.render()

            # 在图像上添加进度指示（帮助确认动画在播放）
            frame_with_text = pixels.copy()
            progress = int((t / 29) * 100)
            # 在左上角添加白色文字背景
            cv2.rectangle(frame_with_text, (5, 5), (120, 30), (255, 255, 255), -1)
            cv2.putText(
                frame_with_text,
                f"{progress}%",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
            frames.append(frame_with_text)

        video_path = output_dir / f"{jname_safe}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, 10, (width, height))
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()

        print(f"  ✅ Saved: {video_path}")

        for idx, percent in [(0, 0), (14, 50), (29, 100)]:
            img_path = output_dir / f"{jname_safe}_{percent}pct.png"
            cv2.imwrite(str(img_path), cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR))

    renderer.close()
    if os.path.exists(modified_xml):
        os.unlink(modified_xml)

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify simulation-ready articulated assets (no Blender required)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported asset types:
  {', '.join(SUPPORTED_ASSETS)}

Examples:
  # Verify cabinet assets
  python verify_sim_assets.py --asset_name cabinet --seeds 1001 1002 1003

  # Verify drawer assets with rendering
  python verify_sim_assets.py --asset_name drawer --seeds 1001 1002 --render

  # Verify all 10 door assets
  python verify_sim_assets.py --asset_name door --seeds $(seq 1001 1010)
""",
    )
    parser.add_argument(
        "--asset_name",
        required=True,
        choices=SUPPORTED_ASSETS,
        help="Asset type to verify",
    )
    parser.add_argument(
        "--asset_dir",
        default=None,
        help="Directory containing assets (default: sim_exports/mjcf/<asset_name>)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, required=True, help="Seeds to verify"
    )
    parser.add_argument("--render", action="store_true", help="Render motion videos")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for renders (default: <asset_name>_renders)",
    )
    parser.add_argument(
        "--resolution", type=int, default=256, help="Render resolution (default: 256)"
    )
    parser.add_argument(
        "--gl_backend",
        default="osmesa",
        choices=["egl", "osmesa", "glfw"],
        help="OpenGL backend (osmesa recommended)",
    )
    args = parser.parse_args()

    # 设置默认目录
    if args.asset_dir is None:
        args.asset_dir = f"sim_exports/mjcf/{args.asset_name}"
    if args.output_dir is None:
        args.output_dir = f"{args.asset_name}_renders"

    asset_dir = Path(args.asset_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Simulation Asset Verification Tool")
    print("=" * 60)
    print(f"Asset Type:     {args.asset_name}")
    print(f"Asset Dir:      {asset_dir}")
    print(f"Seeds:          {args.seeds}")
    if args.render:
        print(f"Render:         enabled, {args.resolution}x{args.resolution}")
        print(f"Output Dir:     {output_dir}")
        print(f"Backend:        {args.gl_backend}")

    results = {}
    for seed in args.seeds:
        xml_path = find_xml_file(asset_dir, seed, args.asset_name)

        if xml_path is None:
            print(f"\n⚠️  Seed {seed}: No XML file found in {asset_dir / str(seed)}")
            results[seed] = None
            continue

        valid = verify_single_asset(xml_path)
        results[seed] = valid

        if args.render:
            print(f"\n[Rendering Joint Motion for seed {seed}]")
            render_dir = output_dir / f"seed_{seed}"
            try:
                success = render_joint_motion(
                    xml_path, render_dir, args.resolution, args.resolution
                )
                if not success:
                    print(f"  ⚠️  Render skipped for seed {seed}")
            except Exception as e:
                print(f"  ❌ Render failed: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for seed, valid in results.items():
        if valid is None:
            status = "⚠️  NOT FOUND"
        elif valid:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        print(f"  Seed {seed}: {status}")

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    missing = sum(1 for v in results.values() if v is None)
    print(f"\nTotal: {len(results)} assets")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Missing: {missing}")


if __name__ == "__main__":
    main()
