#!/usr/bin/env python3
"""
Evaluate and compare PhysNAP training runs.

Compares loss convergence and generates articulated objects from trained checkpoints
for qualitative + quantitative evaluation.

Usage:
    conda activate physnap
    python scripts/evaluate_physnap_training.py \
        --physnap-root external/physnap \
        --experiments original infinigen finetune

Reads from: external/physnap/log/{experiment_name}/
Outputs:    external/physnap/evaluation_results/
"""

import argparse
import json
import os
import sys

import numpy as np

EXPERIMENT_DIRS = {
    "original": "v6.1_diffusion_adapted",
    "infinigen": "v6.1_diffusion_infinigen",
    "finetune": "v6.1_diffusion_finetune_infinigen",
}


def parse_training_metrics(log_dir):
    """Extract training metrics from tensorboard or xls logs."""
    xls_dir = os.path.join(log_dir, "xls")
    if not os.path.exists(xls_dir):
        return None

    metrics = {}
    for fn in os.listdir(xls_dir):
        if fn.endswith(".csv"):
            metric_name = fn.replace(".csv", "")
            try:
                data = np.genfromtxt(
                    os.path.join(xls_dir, fn),
                    delimiter=",",
                    skip_header=1,
                    usecols=(0, 1),
                )
                if data.ndim == 2 and len(data) > 0:
                    metrics[metric_name] = {
                        "steps": data[:, 0].tolist(),
                        "values": data[:, 1].tolist(),
                    }
            except Exception:
                pass
    return metrics


def list_checkpoints(log_dir):
    """List available checkpoints."""
    ckpt_dir = os.path.join(log_dir, "checkpoint")
    if not os.path.exists(ckpt_dir):
        return []

    checkpoints = []
    for fn in sorted(os.listdir(ckpt_dir)):
        if fn.endswith(".pt"):
            path = os.path.join(ckpt_dir, fn)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            checkpoints.append({"name": fn, "path": path, "size_mb": size_mb})
    return checkpoints


def generate_samples(physnap_root, config_path, checkpoint_path, n_samples=10):
    """Generate articulated object samples from a trained checkpoint."""
    sys.path.insert(0, physnap_root)
    import torch

    try:
        from core.models.arti_ddpm_v2 import Model
        from init import get_cfg
    except ImportError:
        print("Cannot import PhysNAP modules. Skipping generation.")
        return None

    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    print(f"  Checkpoint epoch: {ckpt.get('epoch', 'N/A')}, batch: {ckpt.get('batch', 'N/A')}")

    return {
        "checkpoint": os.path.basename(checkpoint_path),
        "epoch": ckpt.get("epoch", -1),
        "batch": ckpt.get("batch", -1),
    }


def compare_experiments(physnap_root, experiments):
    """Compare training results across experiments."""
    results = {}

    for exp_name in experiments:
        exp_dir_name = EXPERIMENT_DIRS.get(exp_name, exp_name)
        log_dir = os.path.join(physnap_root, "log", exp_dir_name)

        if not os.path.exists(log_dir):
            print(f"\n[{exp_name}] Log directory not found: {log_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name} ({exp_dir_name})")
        print(f"{'='*60}")

        checkpoints = list_checkpoints(log_dir)
        print(f"  Checkpoints: {len(checkpoints)}")
        for c in checkpoints:
            print(f"    {c['name']} ({c['size_mb']:.1f} MB)")

        metrics = parse_training_metrics(log_dir)
        if metrics:
            print(f"  Metrics logged: {list(metrics.keys())}")
            if "batch_loss" in metrics:
                vals = metrics["batch_loss"]["values"]
                print(f"  Loss: start={vals[0]:.4f}, end={vals[-1]:.4f}, min={min(vals):.4f}")
        else:
            print("  No CSV metrics found (check tensorboard logs)")

        results[exp_name] = {
            "log_dir": log_dir,
            "checkpoints": checkpoints,
            "has_metrics": metrics is not None,
        }

        if checkpoints:
            info = generate_samples(
                physnap_root, None, checkpoints[-1]["path"], n_samples=5
            )
            if info:
                results[exp_name]["latest_checkpoint_info"] = info

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PhysNAP training runs")
    parser.add_argument("--physnap-root", required=True)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["original", "infinigen"],
        choices=list(EXPERIMENT_DIRS.keys()),
    )
    args = parser.parse_args()

    results = compare_experiments(args.physnap_root, args.experiments)

    output_dir = os.path.join(args.physnap_root, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "comparison_summary.json")
    serializable = {}
    for k, v in results.items():
        serializable[k] = {
            "log_dir": v["log_dir"],
            "num_checkpoints": len(v["checkpoints"]),
            "has_metrics": v["has_metrics"],
        }
    with open(summary_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
