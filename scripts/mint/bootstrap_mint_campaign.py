#!/usr/bin/env python3
"""Bootstrap MINT drawer campaign structure. Run once before overnight loop."""
import json, os, sys
from datetime import datetime
from pathlib import Path

CAMPAIGN = Path("/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1")
def w(name, data):
    p = CAMPAIGN / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        if isinstance(data, str):
            f.write(data)
        else:
            json.dump(data, f, indent=2)
    print(f"  created {p}")

print("Bootstrapping mint_drawer_v1 campaign...")

w("manifest.yaml", """campaign: mint_drawer_v1
phase: mint_sim_transfer_pipeline
phase_gate: await_g9_eval
main_claim: >
  Infinigen-generated articulated manipulation data improves MINT success
  on held-out drawer variants in simulation relative to pretrained MINT.
queue:
  - g1_asset_load
  - g2_obs_contract
  - g3_grasp_plan
  - g4_trajectory_replay
  - g5_delta_reconstruction
  - g6_lerobot_pack
  - g7_mint_batch_load
  - g8_mint_train
  - g9_sim_eval
  - write_claim_memo
env_routing:
  g1_asset_load: mint
  g2_obs_contract: mint
  g3_grasp_plan: graspnet
  g4_trajectory_replay: mint
  g5_delta_reconstruction: mint
  g6_lerobot_pack: mint
  g7_mint_batch_load: mint
  g8_mint_train: mint
  g9_sim_eval: mint
object_split:
  train: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  held_out_sim: [11, 12, 13, 14, 15]
task: open the drawer
baselines:
  - random
  - pretrained_mint
  - finetuned_mint
primary_metric: held_out_sim_success_rate
""")

w("state.json", {
    "phase": "mint_sim_transfer_pipeline",
    "phase_gate": "await_g9_eval",
    "queue": [
        {"id": "g1_asset_load", "status": "pending"},
        {"id": "g2_obs_contract", "status": "pending"},
        {"id": "g3_grasp_plan", "status": "pending"},
        {"id": "g4_trajectory_replay", "status": "pending"},
        {"id": "g5_delta_reconstruction", "status": "pending"},
        {"id": "g6_lerobot_pack", "status": "pending"},
        {"id": "g7_mint_batch_load", "status": "pending"},
        {"id": "g8_mint_train", "status": "pending"},
        {"id": "g9_sim_eval", "status": "pending"},
        {"id": "write_claim_memo", "status": "pending"},
    ],
    "active_job": None,
    "history": [],
    "last_review": None,
    "retries": {},
    "verdict": None,
})

w("acceptance_criteria.json", {
    "g1_asset_load": {"pass": "robosuite env loads drawerbox, joint moves"},
    "g2_obs_contract": {"pass": "obs keys/shapes/dtypes match MINT spec"},
    "g3_grasp_plan": {"pass": ">=1 grasp with score>0.5 on drawer handle"},
    "g4_trajectory_replay": {"pass": "scripted trajectory achieves task success in sim"},
    "g5_delta_reconstruction": {"pass": "reconstruction error < 1mm pos, < 1 deg rot"},
    "g6_lerobot_pack": {"pass": "LeRobotDataset loads without error"},
    "g7_mint_batch_load": {"pass": "dataloader batch correct shapes, no NaN"},
    "g8_mint_train": {"pass": "loss decreases over 1000 steps"},
    "g9_sim_eval": {"pass": "finetuned > pretrained MINT on held-out success rate"},
})

for d in ["runtime", "artifacts", "evaluation"]:
    (CAMPAIGN / d).mkdir(exist_ok=True)

w("campaign_status.md", f"""# Campaign: mint_drawer_v1
**Status**: initialized
**Phase**: mint_sim_transfer_pipeline
**Gate**: await_g9_eval
**Verdict**: null
**Initialized**: {datetime.now().isoformat()}
""")

w("decision_memo.md", "# Pending — no gate completed yet\n")
w("review.json", {})
w("summary.json", {"campaign": "mint_drawer_v1", "status": "initialized"})

print("Bootstrap complete.")
