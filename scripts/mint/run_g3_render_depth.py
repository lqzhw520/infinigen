#!/usr/bin/env python3
"""G3a: Render depth from drawer scene (mint env). Produces artifact for graspnet env."""
import os, sys, json, time, numpy as np
os.environ["MUJOCO_GL"] = "egl"
CAMPAIGN = "/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1"
URDF_DIR = "/mnt/afs2/zhuhaowu/infinigen/sim_exports/urdf/drawerbox/42"
DEPTH_DIR = os.path.join(CAMPAIGN, "artifacts")

def run():
    import mujoco
    with open(os.path.join(URDF_DIR, "drawerbox.urdf")) as f:
        urdf = f.read()
    assets = {}
    for fname in os.listdir(os.path.join(URDF_DIR, "assets")):
        with open(os.path.join(URDF_DIR, "assets", fname), "rb") as f:
            assets[fname] = f.read()
    m = mujoco.MjModel.from_xml_string(urdf, assets)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    renderer = mujoco.Renderer(m, height=360, width=360)

    # Render RGB
    renderer.update_scene(d)
    rgb = renderer.render()

    # Render depth
    renderer.enable_depth_rendering()
    renderer.update_scene(d)
    depth = renderer.render()
    renderer.disable_depth_rendering()

    # Convert to point cloud
    z = depth.flatten().astype(np.float32)
    y, x = np.mgrid[0:360, 0:360].reshape(2, -1).astype(np.float32)
    fx = fy = 360.0
    cx, cy = 180.0, 180.0
    pc_x = (x - cx) * z / fx
    pc_y = (y - cy) * z / fy
    pc = np.stack([pc_x, pc_y, z], axis=-1)

    # Subsample
    valid = z > 0.01
    pc_valid = pc[valid]
    if len(pc_valid) < 2048:
        idx = np.arange(len(pc_valid))
    else:
        idx = np.random.choice(len(pc_valid), 2048, replace=False)
    pc_sampled = pc_valid[idx].astype(np.float32)

    # Save
    os.makedirs(DEPTH_DIR, exist_ok=True)
    np.savez(os.path.join(DEPTH_DIR, "drawer_depth.npz"),
             depth=depth, pc=pc_sampled, rgb=rgb)
    np.save(os.path.join(DEPTH_DIR, "drawer_intrinsics.npy"),
            np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32))

    result = {
        "depth_shape": list(depth.shape),
        "pc_shape": list(pc_sampled.shape),
        "depth_range": [float(depth.min()), float(depth.max())],
        "pc_range": [float(pc_sampled.min(axis=0).min()), float(pc_sampled.max(axis=0).max())],
        "saved": os.path.join(DEPTH_DIR, "drawer_depth.npz"),
    }
    print(json.dumps(result, indent=2))
    renderer.close()
    return True

if __name__ == "__main__":
    sys.exit(0 if run() else 1)
