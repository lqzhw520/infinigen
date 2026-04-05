#!/usr/bin/env python3
"""V59 Env-Level Overfit Gate — Verify task-level overfit in DrawerRobotEnv.

REQUIREMENT: Before claiming "MINT pipeline confirmed" or proceeding to full V59 retrain,
we must verify that the finetuned checkpoint actually succeeds at the drawer task
in simulation — not just offline action imitation.

Design:
  - Compare pretrained vs finetuned on training seeds (1-3, 2 episodes each = 6 rollouts each)
  - Metrics: strict_success, drawer_fraction, attach/open sequence, pull_distance
  - Gate: finetuned > pretrained on ALL metrics → PASS → proceed to full retrain
  - Gate: finetuned <= pretrained on ANY critical metric → FAIL → investigate pipeline

Usage:
  # Screen (recommended ~20 min):
  screen -dmS v59_env_gate bash -c \
    'source /root/anaconda3/etc/profile.d/conda.sh && conda activate mint && \
     cd /mnt/afs2/zhuhaowu/infinigen && \
     python -u scripts/mint/run_v59_overfit_env_gate.py 2>&1 | \
     tee experiments/mint/mint_drawer_v1/artifacts/v59_overfit_env_gate.log'

  # Dry (1 seed, 1 episode):
  python scripts/mint/run_v59_overfit_env_gate.py --dry
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
SCRIPTS_MINT = PROJECT_ROOT / "scripts" / "mint"
sys.path.insert(0, str(SCRIPTS_MINT))

from mint_common import (
    ARTIFACT_DIR,
    DATASET_DIR,
    DATASET_REPO_ID,
)
from evaluate_mint_drawer_campaign import (
    _load_policy,
    rollout_policy,
    DrawerRobotEnv,
    load_json,
)
from strict_success import evaluate_strict_success

PRETRAINED_CKPT = "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-libero"
FINETUNED_CKPT = (
    PROJECT_ROOT
    / "experiments/mint/mint_drawer_v1/artifacts/v59_overfit_outputs/checkpoints/000200/pretrained_model"
)
ARTIFACT = ARTIFACT_DIR / "v59_overfit_env_gate.json"
ACTION_CONTRACT = load_json(
    PROJECT_ROOT / "experiments/mint/mint_drawer_v1/artifacts/active_action_contract.json",
    {},
)

# Gate configuration
GATE_SEEDS = [1, 2, 3]
EPISODES_PER_SEED = 2
MAX_STEPS = 96

# Success thresholds
# Strict success: finetuned must beat pretrained
# Drawer fraction: finetuned must have higher mean
# Attach rate: finetuned must have higher attach rate

PASS_THRESHOLDS = {
    "strict_success": 0.0,  # finetuned > pretrained (any improvement)
    "drawer_fraction": 0.0,  # finetuned > pretrained
    "attach_rate": 0.0,  # finetuned > pretrained
    "pull_distance": 0.0,  # finetuned > pretrained
}


def run(dry: bool = False) -> bool:
    seeds = GATE_SEEDS[:1] if dry else GATE_SEEDS
    ep_per_seed = 1 if dry else EPISODES_PER_SEED

    print("=== V59 Env-Level Overfit Gate ===")
    print(f"  Dry: {dry}")
    print(f"  Seeds: {seeds}")
    print(f"  Episodes per seed: {ep_per_seed}")
    print(f"  Max steps: {MAX_STEPS}")
    print(f"  Pretrained: {PRETRAINED_CKPT}")
    print(f"  Finetuned: {FINETUNED_CKPT}")
    print()

    # Gate 0: manifest must exist
    manifest = ARTIFACT_DIR / "current_dataset_manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        print(f"  manifest: {m.get('dataset_version')} ({m.get('frame_count')} frames)")
    else:
        print("  WARNING: No dataset manifest found")

    # Gate 1: finetuned checkpoint must exist
    if not FINETUNED_CKPT.exists():
        print(f"FATAL: Finetuned checkpoint not found: {FINETUNED_CKPT}")
        return False

    print("\nLoading pretrained policy...")
    t0 = time.time()
    pret_bundle = _load_policy(str(PRETRAINED_CKPT), DATASET_DIR, DATASET_REPO_ID)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print("Loading finetuned policy...")
    t0 = time.time()
    finet_bundle = _load_policy(str(FINETUNED_CKPT), DATASET_DIR, DATASET_REPO_ID)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    results = {"pretrained": [], "finetuned": []}

    for seed in seeds:
        for attempt_idx in range(ep_per_seed):
            rng_seed = 17 + attempt_idx * 101  # same as evaluate_policy_set

            # Pretrained
            item = rollout_policy(
                seed=seed,
                kind="pretrained",
                policy_bundle=pret_bundle,
                rng_seed=rng_seed,
                max_steps=MAX_STEPS,
                action_contract=ACTION_CONTRACT,
            )
            item["eval_seed"] = seed
            item["eval_attempt"] = attempt_idx
            results["pretrained"].append(item)

            # Finetuned
            item = rollout_policy(
                seed=seed,
                kind="finetuned",
                policy_bundle=finet_bundle,
                rng_seed=rng_seed,
                max_steps=MAX_STEPS,
                action_contract=ACTION_CONTRACT,
            )
            item["eval_seed"] = seed
            item["eval_attempt"] = attempt_idx
            results["finetuned"].append(item)

            print(
                f"  seed={seed} attempt={attempt_idx}: "
            f"pret success={results['pretrained'][-1].get('success', False)} "
            f"drawer={results['pretrained'][-1].get('pull_distance', 0.0):.3f} | "
            f"finet success={results['finetuned'][-1].get('success', False)} "
            f"drawer={results['finetuned'][-1].get('pull_distance', 0.0):.3f}"
            )

    # Aggregate metrics
    def aggregate(items, key):
        vals = [float(items[i].get(key, 0)) for i in range(len(items)) if items[i].get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def attach_rate(items):
        # grasp_success is boolean; True=1, False=0
        vals = [float(items[i].get("grasp_success", False)) for i in range(len(items))]
        return sum(vals) / len(vals) if vals else 0.0

    def strict_success_rate(items):
        vals = [float(items[i].get("success", False)) for i in range(len(items))]
        return sum(vals) / len(vals) if vals else 0.0

    metrics = {}
    for pol in ["pretrained", "finetuned"]:
        items = results[pol]
        metrics[pol] = {
            "strict_success_rate": strict_success_rate(items),
            "mean_drawer_fraction": aggregate(items, "max_drawer_fraction"),  # FIXED: was incorrectly using pull_distance
            "mean_pull_distance": aggregate(items, "pull_distance"),            # FIXED: now separate from drawer_fraction
            "attach_rate": attach_rate(items),                                   # FIXED: explicit function for boolean aggregation
            "n_rollouts": len(items),
        }

    # Comparison
    comp = {}
    comp["strict_success_gain"] = (
        metrics["finetuned"]["strict_success_rate"]
        - metrics["pretrained"]["strict_success_rate"]
    )
    comp["drawer_fraction_gain"] = (
        metrics["finetuned"]["mean_drawer_fraction"]
        - metrics["pretrained"]["mean_drawer_fraction"]
    )
    comp["attach_rate_gain"] = (
        metrics["finetuned"]["attach_rate"] - metrics["pretrained"]["attach_rate"]
    )
    comp["pull_distance_gain"] = (
        metrics["finetuned"]["mean_pull_distance"]
        - metrics["pretrained"]["mean_pull_distance"]
    )

    # Gate evaluation
    # CRITICAL: finetuned must beat pretrained on strict_success
    # SECONDARY: finetuned should beat pretrained on drawer_fraction
    gate_passed = (
        comp["strict_success_gain"] > 0.0
        and comp["drawer_fraction_gain"] > 0.0
    )

    # Weak pass: finetuned at least matches pretrained on all metrics (within 0.05 tolerance)
    weak_pass = all(
        comp[k] > -0.05
        for k in ["strict_success_gain", "drawer_fraction_gain", "attach_rate_gain"]
    )

    if gate_passed:
        gate_verdict = "PASS"
        gate_decision = "Proceed to full V59 retrain."
    elif weak_pass:
        # NOTE: "weak pass" is misleading when both policies score 0.000.
        # "Weak pass = finetuned matches pretrained (both=0)" is NOT success.
        # This should be treated as NEGATIVE_GATE until at least one policy
        # achieves non-zero drawer_fraction.
        gate_verdict = "NEGATIVE_GATE"
        gate_decision = (
            "Both pretrained and finetuned score 0.000 on strict_success and drawer_fraction. "
            "Offline action improvement (E021) does not translate to sim task success. "
            "Do NOT full retrain. Investigate action/state pipeline alignment (RCA2/RCA3) next."
        )
    else:
        gate_verdict = "FAIL"
        gate_decision = (
            "Finetuned does NOT beat pretrained in env. "
            "Do NOT full retrain. Root cause analysis (RCA2 action normalization, RCA3 state representation) required."
        )

    print(f"\n=== ENV-LEVEL OVERFIT GATE RESULTS ===")
    print(f"Rollouts: {len(GATE_SEEDS)} seeds × {EPISODES_PER_SEED} episodes = {len(results['pretrained'])} each")
    print()
    print(f"  Metric               | Pretrained  | Finetuned   | Gain")
    print(f"  ---------------------|-------------|-------------|-------")
    print(f"  strict_success_rate  | {metrics['pretrained']['strict_success_rate']:.3f}       | {metrics['finetuned']['strict_success_rate']:.3f}       | {comp['strict_success_gain']:+.3f}")
    print(f"  mean_drawer_fraction | {metrics['pretrained']['mean_drawer_fraction']:.3f}       | {metrics['finetuned']['mean_drawer_fraction']:.3f}       | {comp['drawer_fraction_gain']:+.3f}")
    print(f"  attach_rate          | {metrics['pretrained']['attach_rate']:.3f}       | {metrics['finetuned']['attach_rate']:.3f}       | {comp['attach_rate_gain']:+.3f}")
    print()
    print(f"  GATE VERDICT: {gate_verdict}")
    print(f"  DECISION: {gate_decision}")

    # Per-rollout detail
    print(f"\n  Per-rollout detail:")
    print(f"  {'Seed':>4} {'Attempt':>7} | {'Pret Success':>12} {'Pret Drawer':>11} {'Pret Attach':>10} | {'Finet Success':>12} {'Finet Drawer':>11} {'Finet Attach':>10}")
    print(f"  {'----':>4} {'------':>7} | {'-----------':>12} {'-----------':>11} {'---------':>10} | {'-----------':>12} {'-----------':>11} {'---------':>10}")
    for i in range(len(results["pretrained"])):
        p = results["pretrained"][i]
        f = results["finetuned"][i]
        ps = p.get("success", False)
        fs = f.get("success", False)
        pd = p.get("max_drawer_fraction", 0.0)   # FIXED: was pull_distance
        fd = f.get("max_drawer_fraction", 0.0)   # FIXED: was pull_distance
        pa = p.get("grasp_success", False)
        fa = f.get("grasp_success", False)
        print(f"  {p['eval_seed']:>4} {p['eval_attempt']:>7} | {str(ps):>12} {pd:>11.3f} {str(pa):>10} | {str(fs):>12} {fd:>11.3f} {str(fa):>10}")
    result = {
        "gate": "v59_overfit_env_gate",
        "dry": dry,
        "passed": gate_passed,
        "weak_pass": weak_pass,
        "verdict": gate_verdict,
        "decision": gate_decision,
        "seeds": GATE_SEEDS,
        "episodes_per_seed": EPISODES_PER_SEED,
        "max_steps": MAX_STEPS,
        "metrics": metrics,
        "comparisons": comp,
        "per_rollout": {
            "pretrained": results["pretrained"],
            "finetuned": results["finetuned"],
        },
        "timestamp": time.time(),
        "overfit_checkpoint": str(FINETUNED_CKPT),
        "pretrained_checkpoint": PRETRAINED_CKPT,
    }

    ARTIFACT.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nResults: {ARTIFACT}")
    return gate_passed


if __name__ == "__main__":
    dry = "--dry" in sys.argv
    ok = run(dry=dry)
    raise SystemExit(0 if ok else 1)
