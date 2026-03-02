#!/usr/bin/env python3
"""
Generate a reproducible Sim2Real ablation run matrix (Phase 3.1).

This script does NOT run training. It writes a JSON plan that enumerates:
  - dataset generation configs
  - model training configs (placeholders)
  - evaluation configs (placeholders)

Usage:
  cd /mnt/afs2/zhuhaowu/infinigen
  python scripts/run_sim2real_ablation_matrix.py --out sim_exports/experiments/sim2real/ablation_runs.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class AblationRun:
    run_id: str
    box_types: List[str]
    data_quantity: str  # small | medium | large
    physical_alignment: str  # aligned | degraded
    dr_level: str  # dr0 | dr1 | dr2
    seeds: List[int]
    n_views: int
    joint_state_specs: List[str]
    image_hw: Tuple[int, int]

    # Human-readable command templates (kept as strings for reproducibility)
    cmd_generate_dataset: str
    cmd_train_model: str
    cmd_eval_planner: str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-out-root", type=str, default="sim_exports/experiments/sim2real")
    args = ap.parse_args()

    # Minimal 2x2x2 matrix (recommended starter)
    box_types = ["MAILER", "DRAWER"]
    quantities = {
        "small": {"seeds": list(range(100, 120)), "n_views": 1, "joint_states": [""]},
        "medium": {"seeds": list(range(200, 260)), "n_views": 2, "joint_states": [""]},
    }
    physical_levels = ["aligned", "degraded"]
    dr_levels = ["dr0", "dr2"]

    runs: List[AblationRun] = []
    for q_name, q_cfg in quantities.items():
        for phys in physical_levels:
            for dr in dr_levels:
                run_id = f"{'+'.join(box_types)}__Q={q_name}__PHYS={phys}__DR={dr}"
                out_root = f"{args.base_out_root}/{run_id}"

                # Dataset generation: single entry-point exporter; users can expand into more elaborate pipelines.
                cmd_generate = (
                    "python -m infinigen.launch_blender -s scripts/export_mailerbox_simple_phase1_data_engine.py -- "
                    f"--box_type {{BOX_TYPE}} --seeds {' '.join(str(s) for s in q_cfg['seeds'])} "
                    f"--n_views {q_cfg['n_views']} --joint_states \"{';'.join(q_cfg['joint_states'])}\" "
                    f"--width 320 --height 240 --out_root {out_root}/dataset"
                )

                # Training/eval commands are placeholders; plug in your trainer/planner runner.
                cmd_train = f"python -m YOUR_TRAIN_ENTRY --dataset {out_root}/dataset --out {out_root}/train_logs"
                cmd_eval = f"python -m YOUR_EVAL_ENTRY --urdf_root {out_root}/dataset --out {out_root}/eval_logs"

                runs.append(
                    AblationRun(
                        run_id=run_id,
                        box_types=box_types,
                        data_quantity=q_name,
                        physical_alignment=phys,
                        dr_level=dr,
                        seeds=q_cfg["seeds"],
                        n_views=q_cfg["n_views"],
                        joint_state_specs=q_cfg["joint_states"],
                        image_hw=(240, 320),
                        cmd_generate_dataset=cmd_generate,
                        cmd_train_model=cmd_train,
                        cmd_eval_planner=cmd_eval,
                    )
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict = {
        "schema": "infinigen.sim2real.ablation_matrix.v1",
        "base_out_root": args.base_out_root,
        "runs": [asdict(r) for r in runs],
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"✅ Wrote ablation plan: {args.out}")
    print(f"Runs: {len(runs)}")


if __name__ == "__main__":
    main()

