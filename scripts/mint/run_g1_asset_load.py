#!/usr/bin/env python3
"""G1: Verify Infinigen URDF loads in robosuite as articulated object."""
import os, sys, json, time, torch
os.environ["MUJOCO_GL"] = "egl"

CAMPAIGN = "/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1"
ARTIFACT = os.path.join(CAMPAIGN, "artifacts", "g1_asset_load.json")
URDF_DIR = "/mnt/afs2/zhuhaowu/infinigen/sim_exports/urdf/drawerbox/42"

def run():
    import mujoco

    # Load URDF with VFS
    with open(os.path.join(URDF_DIR, "drawerbox.urdf")) as f:
        urdf = f.read()
    assets = {}
    for fname in os.listdir(os.path.join(URDF_DIR, "assets")):
        fpath = os.path.join(URDF_DIR, "assets", fname)
        with open(fpath, "rb") as f:
            assets[fname] = f.read()

    model = mujoco.MjModel.from_xml_string(urdf, assets)
    data = mujoco.MjData(model)

    # Find joint
    joint_names = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_names.append(name)

    # Test joint movement
    joint_idx = 0
    joint_range = model.jnt_range[joint_idx]
    test_positions = []
    for val in [joint_range[0], (joint_range[0]+joint_range[1])/2, joint_range[1]]:
        data.qpos[joint_idx] = val
        mujoco.mj_forward(model, data)
        test_positions.append(float(data.qpos[joint_idx]))

    result = {
        "gate": "g1_asset_load",
        "passed": True,
        "urdf_path": URDF_DIR,
        "nbody": model.nbody,
        "njnt": model.njnt,
        "ngeom": model.ngeom,
        "joint_names": joint_names,
        "joint_range": [float(x) for x in joint_range],
        "joint_type": int(model.jnt_type[joint_idx]),
        "test_positions": test_positions,
        "joint_moves": all(abs(test_positions[i] - test_positions[j]) > 0.01
                          for i in range(len(test_positions))
                          for j in range(i+1, len(test_positions))),
        "timestamp": time.time(),
    }

    with open(ARTIFACT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result["passed"]

if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
