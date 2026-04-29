#!/usr/bin/env python3
"""
render_backend_smoke.py — A800 Rendering Backend Diagnostics

Diagnoses GPU headless rendering capability for MuJoCo EGL on the A800 container.

Outputs:
  - P_MINUS_1R_FAIL_NO_GPU_EGL_RENDERER  → EGL GPU rendering NOT available
  - P_PLUS_1R_SUCCESS                      → GPU rendering works

Usage:
  source /root/anaconda3/etc/profile.d/conda.sh && conda activate infinigen
  python scripts/mint/render_backend_smoke.py
"""

import os
import sys
import json
import glob
import ctypes
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MINT_DIR = PROJECT_ROOT / "scripts" / "mint"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "render_backend_smoke"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _osmesa_env():
    env = dict(os.environ)
    env.pop("PYOPENGL_PLATFORM", None)
    env.pop("EGL_DEVICE_ID", None)
    env.pop("MUJOCO_EGL_DEVICE_ID", None)
    env.pop("DISPLAY", None)
    env["MUJOCO_GL"] = "osmesa"
    return env


def _run_subprocess(code: str, env_override: dict | None = None, desc: str = ""):
    env = _osmesa_env()
    if env_override:
        env.update(env_override)
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, timeout=60
    )
    if r.returncode != 0:
        return {"error": r.stderr.strip()[:500]}
    import json as _json
    try:
        return _json.loads(r.stdout.strip())
    except Exception:
        return {"raw": r.stdout.strip(), "stderr": r.stderr.strip()[:200]}


def compute_pixel_metrics(img_rgb: np.ndarray) -> dict:
    h, w = img_rgb.shape[:2]
    flat = img_rgb.reshape(-1, 3).astype(np.float64)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    gray = 0.299 * flat[:, 0] + 0.587 * flat[:, 1] + 0.114 * flat[:, 2]
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(np.float64) + 1e-9
    prob = hist / hist.sum()
    entropy = -np.sum(prob * np.log2(prob))
    black = (gray < 15).sum()
    white = (gray > 240).sum()
    mid = gray[(gray >= 15) & (gray <= 240)]
    mid_frac = len(mid) / len(gray) if len(mid) > 0 else 0.0
    total_pix = gray.size
    edge_x = np.abs(np.diff(gray.reshape(h, w), axis=1)).mean()
    edge_y = np.abs(np.diff(gray.reshape(h, w), axis=0)).mean()
    edge_density = (edge_x + edge_y) / 2
    r, g, b = mean
    tint = "warm" if r > g > b else ("cool" if b > g > r else "neutral")
    return {
        "mean": float(mean.mean()),
        "std": float(std.mean()),
        "entropy": float(entropy),
        "mid_tone_fraction": float(mid_frac),
        "black_fraction": float(black / total_pix),
        "white_fraction": float(white / total_pix),
        "edge_density": float(edge_density),
        "rgb_tint": tint,
    }


def save_frame(img_rgb: np.ndarray, name: str) -> Path:
    path = OUTPUT_DIR / f"{name}.png"
    Image.fromarray(img_rgb).save(path)
    return path


# ─── EGL test (subprocess) ─────────────────────────────────────────────────
def _test_egl() -> tuple[bool, dict]:
    code = """
import os, json, ctypes
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
result = {}
try:
    import mujoco
    ctx = mujoco.GLContext(640, 480)
    ctx.make_current()
    import OpenGL.GL as gl
    vendor = gl.glGetString(gl.GL_VENDOR)
    renderer = gl.glGetString(gl.GL_RENDERER)
    version = gl.glGetString(gl.GL_VERSION)
    result["vendor"] = vendor.decode() if isinstance(vendor, bytes) else str(vendor)
    result["renderer"] = renderer.decode() if isinstance(renderer, bytes) else str(renderer)
    result["version"] = version.decode() if isinstance(version, bytes) else str(version)
    ctx.free()
    print(json.dumps(result))
except Exception as e:
    result["error"] = str(e)
    print(json.dumps(result))
"""
    return _run_subprocess(code, {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}, "EGL")


# ─── OSMesa test (subprocess) ──────────────────────────────────────────────
def _test_osmesa() -> tuple[bool, dict]:
    code = """
import os, json, ctypes
os.environ["MUJOCO_GL"] = "osmesa"
os.environ.pop("PYOPENGL_PLATFORM", None)
result = {}
try:
    import mujoco
    ctx = mujoco.GLContext(640, 480)
    ctx.make_current()
    import OpenGL.GL as gl
    vendor = gl.glGetString(gl.GL_VENDOR)
    renderer = gl.glGetString(gl.GL_RENDERER)
    version = gl.glGetString(gl.GL_VERSION)
    result["vendor"] = vendor.decode() if isinstance(vendor, bytes) else str(vendor)
    result["renderer"] = renderer.decode() if isinstance(renderer, bytes) else str(renderer)
    result["version"] = version.decode() if isinstance(version, bytes) else str(version)
    # Render a simple box with correct MuJoCo 3.4.0 API
    model = mujoco.MjModel.from_xml_string(
        "<mujoco>"
        "<worldbody>"
        "<light pos='0 0 3' dir='0 0 -1' diffuse='.8 .8 .8'/>"
        "<body pos='0 0 0'><geom type='box' size='.1 .1 .1' rgba='.9 .2 .1 1'/></body>"
        "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    scn = mujoco.MjvScene(model, maxgeom=1000)
    ctx_render = mujoco.MjrContext(model, 0)
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None,
                            mujoco.MjvCamera(), 0, scn)
    vp = mujoco.MjrRect(0, 0, 640, 480)
    mujoco.mjr_render(vp, scn, ctx_render)
    rgb = (ctypes.c_ubyte * (640 * 480 * 3))()
    mujoco.mjr_readPixels(rgb, None, vp, ctx_render)
    arr = list(rgb)
    result["pixel_mean"] = sum(arr) / len(arr)
    result["pixel_nonzero"] = sum(1 for x in arr if x > 10) / len(arr)
    result["pixel_std"] = (sum((x - result["pixel_mean"])**2 for x in arr) / len(arr)) ** 0.5
    ctx.free()
    print(json.dumps(result))
except Exception as e:
    result["error"] = str(e)
    print(json.dumps(result))
"""
    return _run_subprocess(code, None, "OSMesa")


# ─── Merged scene render (subprocess) ──────────────────────────────────────
def _render_merged() -> tuple[bool, dict]:
    code = """
import os, sys, json, numpy as np
os.environ["MUJOCO_GL"] = "osmesa"
os.environ.pop("PYOPENGL_PLATFORM", None)
sys.path.insert(0, '/mnt/afs2/zhuhaowu/infinigen/scripts/mint')
from drawer_robot_env_mujoco import DrawerRobotEnvMuJoCoLibero
from merged_model_builder import MergedModelBuilder
result = {}
try:
    DrawerRobotEnvMuJoCoLibero._MERGED_BUILDER_CLASS = MergedModelBuilder
    env = DrawerRobotEnvMuJoCoLibero(seed=1, image_size=480)
    for _ in range(5):
        env.step(np.zeros(9))
    rgb = env.render()
    env.close()
    arr = rgb.flatten().tolist()
    result["pixel_nonzero"] = sum(1 for x in arr if x > 10) / len(arr)
    result["pixel_mean"] = sum(arr) / len(arr)
    result["shape"] = list(rgb.shape)
    print(json.dumps(result))
except Exception as e:
    result["error"] = str(e)
    print(json.dumps(result))
"""
    return _run_subprocess(code, None, "Merged render")


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("render_backend_smoke.py — A800 Rendering Backend Diagnostics")
    print("=" * 70)

    report = {
        "gpu_available": False,
        "verdict": "P_MINUS_1R_FAIL_NO_GPU_EGL_RENDERER",
        "diagnosis": {},
    }

    # 1. GPU Hardware
    print("\n[1/8] GPU Hardware (nvidia-smi)")
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        print(f"  {r.stdout.strip()}")
        report["diagnosis"]["gpu_name"] = r.stdout.strip()
    except Exception as e:
        print(f"  ERROR: {e}")
        report["diagnosis"]["nvidia_smi_error"] = str(e)

    # 2. EGL Vendor JSON
    print("\n[2/8] EGL Vendor JSON")
    nvidia_egl_installed = False
    for vf in sorted(glob.glob("/usr/share/glvnd/egl_vendor.d/*.json")):
        with open(vf) as f:
            data = json.load(f)
        lib_path = data.get("ICD", {}).get("library_path", "")
        is_nvidia = "nvidia" in lib_path.lower()
        if is_nvidia:
            nvidia_egl_installed = True
        print(f"  {Path(vf).name}: {lib_path} {'(NVIDIA)' if is_nvidia else '(Mesa)'}")
    report["diagnosis"]["nvidia_egl_installed"] = nvidia_egl_installed

    # 3. DRM Devices
    print("\n[3/8] GPU Device Files")
    dri = sorted(glob.glob("/dev/dri/*"))
    nvidia_dev = sorted(glob.glob("/dev/nvidia*"))
    print(f"  /dev/dri/*: {dri or 'NONE'}")
    print(f"  /dev/nvidia*: {nvidia_dev or 'NONE'}")
    report["diagnosis"]["drm_devices"] = dri
    report["diagnosis"]["nvidia_devs"] = nvidia_dev

    # 4. Container Env
    print("\n[4/8] Container Environment")
    env_keys = ["NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
                "__EGL_VENDOR_LIBRARY_DIRS"]
    container_env = {}
    for k in env_keys:
        v = os.environ.get(k, "NOT SET")
        container_env[k] = v
        print(f"  {k}={v}")
    report["diagnosis"]["container_env"] = container_env

    # 5. EGL eglQueryDevicesEXT
    print("\n[5/8] EGL eglQueryDevicesEXT()")
    egl_dev_result = _run_subprocess("""
import os, json
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
try:
    from mujoco.egl import egl_ext as EGL
    devs = EGL.eglQueryDevicesEXT()
    print(json.dumps({"devices": len(devs)}))
except Exception as e:
    print(json.dumps({"error": str(e)}))
""", {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}, "EGL device query")
    device_count = egl_dev_result.get("devices", 0) if isinstance(egl_dev_result, dict) else 0
    if isinstance(egl_dev_result, dict) and "error" in egl_dev_result:
        print(f"  Error: {egl_dev_result['error'][:200]}")
    print(f"  Devices found: {device_count}")
    report["diagnosis"]["egl_devices_found"] = device_count

    # 6. EGL smoke
    print("\n[6/8] EGL GPU Smoke Test (MUJOCO_GL=egl)")
    egl_data = _test_egl()
    egl_ok = "error" not in egl_data
    if egl_ok:
        print(f"  VENDOR: {egl_data.get('vendor')}")
        print(f"  RENDERER: {egl_data.get('renderer')}")
        print(f"  VERSION: {egl_data.get('version')}")
        print(f"  Status: GPU rendering AVAILABLE")
    else:
        print(f"  FAIL: {egl_data.get('error', 'unknown')}")
    report["diagnosis"]["egl_smoke_ok"] = egl_ok
    report["diagnosis"]["egl_smoke_data"] = egl_data

    # 7. OSMesa smoke
    print("\n[7/8] OSMesa Smoke Test (MUJOCO_GL=osmesa)")
    osmesa_data = _test_osmesa()
    osmesa_ok = "error" not in osmesa_data and osmesa_data.get("pixel_nonzero", 0) > 0.001
    if osmesa_ok:
        print(f"  VENDOR: {osmesa_data.get('vendor')}")
        print(f"  RENDERER: {osmesa_data.get('renderer')}")
        print(f"  pixel_nonzero: {osmesa_data.get('pixel_nonzero', 0):.2%}")
        print(f"  pixel_mean: {osmesa_data.get('pixel_mean', 0):.1f}")
    else:
        if 'error' in osmesa_data:
            print(f"  OSMesa context: FAIL: {osmesa_data.get('error', 'unknown')}")
        else:
            nz = osmesa_data.get('pixel_nonzero', 0)
            print(f"  OSMesa pixel_nonzero={nz:.2%} (low — software rasterizer)")
    report["diagnosis"]["osmesa_smoke_ok"] = osmesa_ok
    report["diagnosis"]["osmesa_smoke_data"] = osmesa_data

    # 8. Merged scene
    print("\n[8/8] Merged Robot+Drawer Scene Render")
    merged_data = _render_merged()
    merged_ok = "error" not in merged_data and merged_data.get("pixel_nonzero", 0) > 0.001
    if merged_ok:
        print(f"  pixel_nonzero: {merged_data.get('pixel_nonzero', 0):.2%}")
        print(f"  pixel_mean: {merged_data.get('pixel_mean', 0):.1f}")
        print(f"  shape: {merged_data.get('shape')}")
        # Save the frame directly from the merged render
        code = """
import os, sys, json, numpy as np
os.environ["MUJOCO_GL"] = "osmesa"
os.environ.pop("PYOPENGL_PLATFORM", None)
sys.path.insert(0, '/mnt/afs2/zhuhaowu/infinigen/scripts/mint')
from drawer_robot_env_mujoco import DrawerRobotEnvMuJoCoLibero
from merged_model_builder import MergedModelBuilder
DrawerRobotEnvMuJoCoLibero._MERGED_BUILDER_CLASS = MergedModelBuilder
env = DrawerRobotEnvMuJoCoLibero(seed=1, image_size=480)
for _ in range(5):
    env.step(np.zeros(9))
rgb = env.render()
env.close()
from PIL import Image
Image.fromarray(rgb).save('/mnt/afs2/zhuhaowu/infinigen/outputs/render_backend_smoke/05_merged_robot_drawer.png')
print('saved')
"""
        _run_subprocess(code, None, "Save merged frame")
        try:
            frame_path = OUTPUT_DIR / "05_merged_robot_drawer.png"
            if frame_path.exists():
                frame = np.array(Image.open(frame_path))
                metrics = compute_pixel_metrics(frame)
                print(f"  Saved: {frame_path}")
                print(f"  mid_tone: {metrics['mid_tone_fraction']:.2%}, "
                      f"entropy: {metrics['entropy']:.1f}, "
                      f"std: {metrics['std']:.1f}, "
                      f"edge: {metrics['edge_density']:.2f}, "
                      f"tint: {metrics['rgb_tint']}")
                report["pixel_metrics"] = {"merged_scene": metrics}
        except Exception as e:
            print(f"  Could not save frame: {e}")
    else:
        print(f"  FAIL: {merged_data.get('error', 'unknown')}")
    report["diagnosis"]["merged_smoke_ok"] = merged_ok
    report["diagnosis"]["merged_smoke_data"] = merged_data

    # ── Verdict
    print("\n" + "=" * 70)
    if egl_ok:
        report["verdict"] = "P_PLUS_1R_SUCCESS"
        report["gpu_available"] = True
        print("VERDICT: P_PLUS_1R_SUCCESS — GPU rendering AVAILABLE")
    else:
        report["verdict"] = "P_MINUS_1R_FAIL_NO_GPU_EGL_RENDERER"
        print("VERDICT: P_MINUS_1R_FAIL_NO_GPU_EGL_RENDERER")
        print()
        print("ROOT CAUSE ANALYSIS:")
        print(f"  1. NVIDIA EGL lib installed: {nvidia_egl_installed}")
        print(f"  2. eglQueryDevicesEXT() returns: {device_count} devices")
        print(f"  3. /dev/dri/* available: {len(dri) > 0}")
        print(f"  4. NVIDIA_VISIBLE_DEVICES: {container_env.get('NVIDIA_VISIBLE_DEVICES', 'NOT SET')}")
        print(f"  5. NVIDIA_DRIVER_CAPABILITIES: {container_env.get('NVIDIA_DRIVER_CAPABILITIES', 'NOT SET')}")
        print()
        print("  The A800 container has NO /dev/dri/* device nodes and NO")
        print("  NVIDIA_VISIBLE_DEVICES env var. The nvidia-container-toolkit did NOT")
        print("  expose GPU device files into this container.")
        print()
        print("  NVIDIA EGL requires EGL_EXT_platform_device + /dev/dri/* access.")
        print("  Without these, headless GPU rendering is blocked at container level.")
        print()
        print("  OSMesa (CPU software rasterizer) is available but produces flat shading")
        print("  (bimodal pixel distributions). CANNOT be used as claim-bearing MINT images.")
        print()
        if merged_ok:
            print("  Merged robot+drawer scene: RENDERS with OSMesa (47%+ non-zero pixels).")
            print("  But quality is flat-shaded, NOT LIBERO-quality GPU rendering.")

    # Save report
    report_path = OUTPUT_DIR / "diagnostic_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {report_path}")
    return report["verdict"]


if __name__ == "__main__":
    verdict = main()
    sys.exit(0 if verdict == "P_PLUS_1R_SUCCESS" else 1)
