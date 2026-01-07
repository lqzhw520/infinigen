#!/usr/bin/env python
"""
测试 MuJoCo 渲染环境
运行此脚本来检查你的环境支持哪种渲染后端
"""

import os
import sys


def test_backend(backend):
    """测试特定后端是否可用"""
    print(f"\n{'='*40}")
    print(f"Testing MUJOCO_GL={backend}")
    print("=" * 40)

    # 设置环境变量 - 必须在 import mujoco 之前！
    os.environ["MUJOCO_GL"] = backend

    # EGL 需要同时设置 PYOPENGL_PLATFORM
    if backend == "egl":
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        print("  Also setting PYOPENGL_PLATFORM=egl")
    else:
        if "PYOPENGL_PLATFORM" in os.environ:
            del os.environ["PYOPENGL_PLATFORM"]

    # 清除已导入的 mujoco 和 OpenGL 模块（强制重新导入）
    mods_to_remove = [
        k
        for k in list(sys.modules.keys())
        if "mujoco" in k.lower() or "opengl" in k.lower() or "gl" in k.lower()
    ]
    for mod in mods_to_remove:
        try:
            del sys.modules[mod]
        except:
            pass

    try:
        import mujoco

        print("  ✅ mujoco imported successfully")

        # 创建简单模型
        xml = """
        <mujoco>
            <visual>
                <global offwidth="256" offheight="256"/>
            </visual>
            <worldbody>
                <body>
                    <geom type="box" size="1 1 1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        print("  ✅ Model created successfully")

        # 尝试创建渲染器
        renderer = mujoco.Renderer(model, height=256, width=256)
        print("  ✅ Renderer created successfully")

        # 尝试渲染
        renderer.update_scene(data)
        pixels = renderer.render()
        print(f"  ✅ Rendered successfully, shape: {pixels.shape}")

        renderer.close()
        print(f"\n  🎉 Backend '{backend}' WORKS!")
        return True

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


def main():
    print("MuJoCo Render Backend Test")
    print("=" * 40)

    # 检查系统依赖
    print("\n[System Check]")

    # 检查 DISPLAY
    display = os.environ.get("DISPLAY", "")
    print(f"  DISPLAY: {display if display else '(not set - headless mode)'}")

    # 检查当前 PYOPENGL_PLATFORM
    pyopengl = os.environ.get("PYOPENGL_PLATFORM", "")
    print(f"  PYOPENGL_PLATFORM: {pyopengl if pyopengl else '(not set)'}")

    # 检查 GPU
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            print(f"  GPU: {result.stdout.strip()}")
        else:
            print("  GPU: (nvidia-smi failed)")
    except:
        print("  GPU: (nvidia-smi not available)")

    # 检查库文件
    libs_to_check = [
        ("/usr/lib/x86_64-linux-gnu/libOSMesa.so", "OSMesa"),
        ("/usr/lib/x86_64-linux-gnu/libOSMesa.so.8", "OSMesa.8"),
        ("/usr/lib/x86_64-linux-gnu/libEGL.so", "EGL"),
        ("/usr/lib/x86_64-linux-gnu/libEGL.so.1", "EGL.1"),
        ("/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0", "EGL_nvidia"),
    ]
    print("\n[Library Check]")
    for lib_path, name in libs_to_check:
        if os.path.exists(lib_path):
            print(f"  ✅ {name}: found at {lib_path}")
        else:
            print(f"  ❌ {name}: NOT found")

    # 测试 EGL（因为 EGL 库存在，且有 NVIDIA GPU）
    print("\n" + "=" * 40)
    print("Testing EGL with PYOPENGL_PLATFORM=egl")
    print("=" * 40)

    # 关键：在新进程中测试以避免 import 问题
    import subprocess

    test_script = '''
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import mujoco
xml = """
<mujoco>
    <visual><global offwidth="256" offheight="256"/></visual>
    <worldbody><body><geom type="box" size="1 1 1"/></body></worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=256, width=256)
renderer.update_scene(data)
pixels = renderer.render()
print(f"SUCCESS: Rendered image shape {pixels.shape}")
renderer.close()
'''

    result = subprocess.run(
        [sys.executable, "-c", test_script], capture_output=True, text=True, timeout=30
    )

    if result.returncode == 0:
        print("  ✅ EGL works!")
        print(f"  Output: {result.stdout.strip()}")
        print("\n推荐命令:")
        print(
            "  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python verify_cabinet_assets.py ..."
        )
    else:
        print("  ❌ EGL failed")
        print(f"  Error: {result.stderr.strip()}")

        # 测试 OSMesa
        print("\n" + "=" * 40)
        print("Testing OSMesa")
        print("=" * 40)

        test_script_osmesa = '''
import os
os.environ['MUJOCO_GL'] = 'osmesa'

import mujoco
xml = """
<mujoco>
    <visual><global offwidth="256" offheight="256"/></visual>
    <worldbody><body><geom type="box" size="1 1 1"/></body></worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=256, width=256)
renderer.update_scene(data)
pixels = renderer.render()
print(f"SUCCESS: Rendered image shape {pixels.shape}")
renderer.close()
'''
        result = subprocess.run(
            [sys.executable, "-c", test_script_osmesa],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("  ✅ OSMesa works!")
            print(f"  Output: {result.stdout.strip()}")
        else:
            print("  ❌ OSMesa failed")
            print(f"  Error: {result.stderr.strip()}")
            print("\n安装 OSMesa:")
            print("  sudo apt install libosmesa6-dev")


if __name__ == "__main__":
    main()
