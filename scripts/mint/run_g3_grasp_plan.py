#!/usr/bin/env python3
"""G3b: Contact-GraspNet predicts grasps from pre-rendered depth (graspnet env)."""
import os, sys, json, time, numpy as np, torch
CAMPAIGN = "/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1"
ARTIFACT = os.path.join(CAMPAIGN, "artifacts", "g3_grasp_plan.json")
BASE_DIR = "/mnt/afs2/zhuhaowu/infinigen/external/contact_graspnet_pytorch"
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints/contact_graspnet")

def run():
    sys.path.insert(0, os.path.join(BASE_DIR, "Pointnet_Pointnet2_pytorch"))
    import yaml
    with open(os.path.join(CKPT_DIR, "config.yaml")) as f:
        config = yaml.safe_load(f)
    from contact_graspnet_pytorch.contact_graspnet import ContactGraspnet

    model = ContactGraspnet(config, device="cuda:0")
    ckpt = torch.load(os.path.join(CKPT_DIR, "checkpoints/model.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load pre-rendered point cloud
    data = np.load(os.path.join(CAMPAIGN, "artifacts", "drawer_depth.npz"))
    pc = data["pc"]
    if len(pc) > 2048:
        idx = np.random.choice(len(pc), 2048, replace=False)
        pc = pc[idx]

    pc_torch = torch.from_numpy(pc.astype(np.float32)).unsqueeze(0).to("cuda:0")
    with torch.no_grad():
        pred = model(pc_torch)

    grasps = pred["pred_grasps_cam"][0].cpu().numpy()
    scores = pred["pred_scores"][0, :, 0].cpu().numpy()
    top_idx = int(np.argmax(scores))

    result = {
        "gate": "g3_grasp_plan",
        "passed": float(scores[top_idx]) > 0.001,
        "n_grasps": len(grasps),
        "top_score": float(scores[top_idx]),
        "top_grasp_pose": grasps[top_idx].tolist(),
        "pc_source": os.path.join(CAMPAIGN, "artifacts", "drawer_depth.npz"),
        "timestamp": time.time(),
    }
    with open(ARTIFACT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result["passed"]

if __name__ == "__main__":
    sys.exit(0 if run() else 1)
