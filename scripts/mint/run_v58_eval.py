#!/usr/bin/env python3
"""V58 End-to-End Evaluation: Train probe (D3-style) + Held-out eval (E1-style).

V58 checkpoint: experiments/mint/mint_drawer_v1/artifacts/v58_training_outputs/
V58 dataset:    experiments/mint/mint_drawer_v1/dataset/
  -> 229 episodes, 18701 frames, seeds 1-15 (physics-legal rollouts from seeds 1-6 P4 + 7-15 recovery)

Pipeline:
  Stage 1 — Train Probe: finetuned vs pretrained on seeds 1-10 (train seeds)
  Stage 2 — Held-out Eval: finetuned vs pretrained on seeds 11-15 (held-out seeds)

Usage:
  # Run with screen (night runner):
  screen -dmS v58_eval bash -c 'source /root/anaconda3/etc/profile.d/conda.sh && conda activate mint && cd /mnt/afs2/zhuhaowu/infinigen && python -u scripts/mint/run_v58_eval.py 2>&1 | tee experiments/mint/mint_drawer_v1/outputs/v58_eval.log'

  # Run dry (1 seed):
  python scripts/mint/run_v58_eval.py --dry
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import sys

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
SCRIPTS_MINT = PROJECT_ROOT / "scripts" / "mint"
sys.path.insert(0, str(SCRIPTS_MINT))

from mint_common import (
    ACTIVE_ACTION_CONTRACT_PATH,
    ARTIFACT_DIR,
    CAMPAIGN_DIR,
    DATASET_DIR,
    DATASET_REPO_ID,
    load_json,
)
from evaluate_mint_drawer_campaign import (
    evaluate_policy_set,
    evaluate_train_probe,
    render_report,
)

V58_CKPT_DIR = ARTIFACT_DIR / "v58_training_outputs"
V58_SUMMARY = ARTIFACT_DIR / "v58_train_summary.json"
EVAL_DIR = CAMPAIGN_DIR / "evaluation"
ARTIFACT_OUT = ARTIFACT_DIR / "v58_eval_rollouts.json"
TRAIN_PROBE_OUT = EVAL_DIR / "v58_train_seed_probe.json"
EVAL_REPORT = EVAL_DIR / "v58_eval_report.md"
EVAL_SUMMARY = EVAL_DIR / "v58_eval_summary.json"

TRAIN_SEEDS = list(range(1, 11))     # seeds 1-10
HELDOUT_SEEDS = list(range(11, 16)) # seeds 11-15
TRAIN_EPISODES = 3
HELDOUT_EPISODES = 5


def _load_finetuned_ckpt() -> Path | None:
    """Find the latest V58 checkpoint."""
    # Check last symlink first
    last = V58_CKPT_DIR / "last"
    if last.is_symlink() or last.exists():
        pretrained = last / "pretrained_model"
        if pretrained.exists():
            return pretrained
    # Fall back to step 1000
    ckpt = V58_CKPT_DIR / "checkpoints" / "001000" / "pretrained_model"
    if ckpt.exists():
        return ckpt
    # Scan for highest step
    cp_dir = V58_CKPT_DIR / "checkpoints"
    if not cp_dir.exists():
        return None
    steps = []
    for d in cp_dir.iterdir():
        if d.is_dir() and d.name.startswith("00"):
            try:
                steps.append(int(d.name))
            except ValueError:
                pass
    if not steps:
        return None
    best = max(steps)
    ckpt = cp_dir / f"{best:06d}" / "pretrained_model"
    return ckpt if ckpt.exists() else None


def _stage1_train_probe(ckpt_path: Path) -> tuple[dict, list, bool]:
    """Stage 1: finetuned vs pretrained on train seeds 1-10."""
    print(f"\n[Stage 1] Train Probe — seeds {TRAIN_SEEDS}")
    print(f"[Stage 1] Finetuned ckpt: {ckpt_path}")

    action_contract = load_json(ACTIVE_ACTION_CONTRACT_PATH, {})

    comparison, records = evaluate_policy_set(
        finetuned_path=str(ckpt_path),
        seeds=TRAIN_SEEDS,
        dataset_root=DATASET_DIR,
        repo_id=DATASET_REPO_ID,
        max_steps=96,
        include_random=False,
        action_contract=action_contract,
        episodes_per_seed=TRAIN_EPISODES,
    )

    ft = comparison["finetuned_mint"]
    pt = comparison["pretrained_mint"]
    success_gain = float(ft["success_rate"] - pt["success_rate"])
    trend_passed = bool(
        ft["success_rate"] > pt["success_rate"] and ft["successful_seed_count"] >= 2
    )

    probe_summary = {
        "stage": "stage1_train_probe",
        "seeds": TRAIN_SEEDS,
        "episodes_per_seed": TRAIN_EPISODES,
        "finetuned_success_rate": ft["success_rate"],
        "pretrained_success_rate": pt["success_rate"],
        "success_gain": success_gain,
        "finetuned_successful_seeds": ft["successful_seed_count"],
        "trend_passed": trend_passed,
        "finetuned_eef_motion": ft.get("total_eef_motion_mean"),
        "pretrained_eef_motion": pt.get("total_eef_motion_mean"),
    }
    print(f"[Stage 1] finetuned={ft['success_rate']:.3f} pretrained={pt['success_rate']:.3f} gain={success_gain:+.3f} trend_passed={trend_passed}")

    return probe_summary, records, trend_passed


def _stage2_heldout_eval(ckpt_path: Path) -> tuple[dict, list, str]:
    """Stage 2: finetuned vs pretrained on held-out seeds 11-15."""
    print(f"\n[Stage 2] Held-out Eval — seeds {HELDOUT_SEEDS}")

    action_contract = load_json(ACTIVE_ACTION_CONTRACT_PATH, {})

    comparison, records = evaluate_policy_set(
        finetuned_path=str(ckpt_path),
        seeds=HELDOUT_SEEDS,
        dataset_root=DATASET_DIR,
        repo_id=DATASET_REPO_ID,
        max_steps=96,
        include_random=True,
        action_contract=action_contract,
        episodes_per_seed=HELDOUT_EPISODES,
    )

    ft = comparison["finetuned_mint"]
    pt = comparison["pretrained_mint"]
    rd = comparison.get("random", None)
    success_gain = float(ft["success_rate"] - pt["success_rate"])

    # Verdict: does finetuned beat pretrained on held-out?
    verdict = (
        "claim_supported"
        if ft["success_rate"] > pt["success_rate"]
        else "scientific_not_supported"
    )

    heldout_summary = {
        "stage": "stage2_heldout_eval",
        "seeds": HELDOUT_SEEDS,
        "episodes_per_seed": HELDOUT_EPISODES,
        "finetuned_success_rate": ft["success_rate"],
        "pretrained_success_rate": pt["success_rate"],
        "random_success_rate": rd["success_rate"] if rd else None,
        "success_gain": success_gain,
        "finetuned_successful_seeds": ft["successful_seed_count"],
        "finetuned_eef_motion": ft.get("total_eef_motion_mean"),
        "pretrained_eef_motion": pt.get("total_eef_motion_mean"),
        "verdict": verdict,
    }
    print(f"[Stage 2] finetuned={ft['success_rate']:.3f} pretrained={pt['success_rate']:.3f} random={rd['success_rate'] if rd else 'N/A'} verdict={verdict}")

    return heldout_summary, records, verdict


def run(dry: bool = False) -> bool:
    t0 = time.time()

    # ── Gate: must have V58 training summary ──────────────────────────────
    if not V58_SUMMARY.exists():
        result = {
            "gate": "v58_endtoend_eval",
            "passed": False,
            "error": f"V58 training summary not found: {V58_SUMMARY}",
            "timestamp": time.time(),
        }
        ARTIFACT_OUT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT_OUT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return False

    v58_meta = json.loads(V58_SUMMARY.read_text())
    if not v58_meta.get("passed"):
        result = {
            "gate": "v58_endtoend_eval",
            "passed": False,
            "error": "V58 training did not pass",
            "v58_summary": v58_meta,
            "timestamp": time.time(),
        }
        ARTIFACT_OUT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return False

    ckpt_path = _load_finetuned_ckpt()
    if ckpt_path is None:
        result = {
            "gate": "v58_endtoend_eval",
            "passed": False,
            "error": "No V58 checkpoint found",
            "checkpoint_dir": str(V58_CKPT_DIR),
            "timestamp": time.time(),
        }
        ARTIFACT_OUT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return False

    print(f"[V58 Eval] Using checkpoint: {ckpt_path}")
    print(f"[V58 Eval] Dataset: {DATASET_DIR}")
    print(f"[V58 Eval] Repo ID: {DATASET_REPO_ID}")

    # ── Dry run: just Stage 1 on seed 2 ─────────────────────────────────
    if dry:
        print("\n[DRY RUN] Running Stage 1 on seed=2 only")
        action_contract = load_json(ACTIVE_ACTION_CONTRACT_PATH, {})
        comparison, records = evaluate_policy_set(
            finetuned_path=str(ckpt_path),
            seeds=[2],
            dataset_root=DATASET_DIR,
            repo_id=DATASET_REPO_ID,
            max_steps=96,
            include_random=False,
            action_contract=action_contract,
            episodes_per_seed=1,
        )
        result = {
            "gate": "v58_endtoend_eval",
            "dry": True,
            "passed": True,
            "checkpoint_path": str(ckpt_path),
            "dry_comparison": {k: dict(v) for k, v in comparison.items()},
            "elapsed_sec": round(time.time() - t0, 1),
            "timestamp": time.time(),
        }
        ARTIFACT_OUT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return True

    # ── Stage 1: Train Probe ───────────────────────────────────────────────
    probe_summary, probe_records, trend_passed = _stage1_train_probe(ckpt_path)

    # ── Stage 2: Held-out Eval ─────────────────────────────────────────────
    heldout_summary, heldout_records, verdict = _stage2_heldout_eval(ckpt_path)

    # ── Compose all records ────────────────────────────────────────────────
    all_records = {**probe_records, **heldout_records}

    # ── Write artifacts ────────────────────────────────────────────────────
    ARTIFACT_OUT.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_PROBE_OUT.parent.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    ARTIFACT_OUT.write_text(json.dumps(all_records, indent=2))
    TRAIN_PROBE_OUT.write_text(json.dumps(probe_summary, indent=2))

    eval_report_body = render_report(
        {
            "verdict": verdict,
            "held_out_seeds": HELDOUT_SEEDS,
            "eval_max_steps": 96,
            "episodes_per_seed": HELDOUT_EPISODES,
            "comparison": {},
            "strongest_true_claim": (
                "Fine-tuned MINT (V58, physics-legal Infinigen data) improves held-out drawer success over pretrained MINT."
                if verdict == "claim_supported"
                else "Fine-tuned MINT (V58, physics-legal Infinigen data) does NOT beat pretrained MINT on held-out drawer variants."
            ),
        }
    )

    report_lines = [
        "# V58 End-to-End Evaluation",
        "",
        f"**Checkpoint**: `{ckpt_path}`",
        f"**V58 Train Summary**: `{V58_SUMMARY}`",
        f"**Dataset**: `{DATASET_DIR}` ({DATASET_REPO_ID})",
        f"**Episodes**: {v58_meta.get('steps_completed', 'N/A')} steps",
        "",
        "## Stage 1 — Train Seed Probe",
        f"| Metric | Finetuned | Pretrained |",
        f"| --- | ---: | ---: |",
        f"| Success Rate | {probe_summary['finetuned_success_rate']:.3f} | {probe_summary['pretrained_success_rate']:.3f} |",
        f"| EEF Motion (m) | {probe_summary.get('finetuned_eef_motion', 'N/A')} | {probe_summary.get('pretrained_eef_motion', 'N/A')} |",
        f"| Trend Passed | {probe_summary['trend_passed']} |",
        "",
        "## Stage 2 — Held-Out Eval (Seeds 11-15)",
        f"| Policy | Success Rate | EEF Motion (m) |",
        f"| --- | ---: | ---: |",
        f"| Finetuned (V58) | {heldout_summary['finetuned_success_rate']:.3f} | {heldout_summary.get('finetuned_eef_motion', 'N/A')} |",
        f"| Pretrained (LIBERO) | {heldout_summary['pretrained_success_rate']:.3f} | {heldout_summary.get('pretrained_eef_motion', 'N/A')} |",
        f"| Random | {heldout_summary.get('random_success_rate', 'N/A')} |",
        "",
        "## Verdict",
        f"**`{verdict}`**",
        "",
        "## Raw Summaries",
        "",
        "### Train Probe",
        "",
        "```json",
        json.dumps(probe_summary, indent=2),
        "```",
        "",
        "### Held-out Eval",
        "",
        "```json",
        json.dumps(heldout_summary, indent=2),
        "```",
        "",
    ]
    EVAL_REPORT.write_text("\n".join(report_lines))

    # Also write machine-readable summary
    summary_doc = {
        "gate": "v58_endtoend_eval",
        "passed": verdict == "claim_supported",
        "verdict": verdict,
        "checkpoint_path": str(ckpt_path),
        "v58_train_summary": v58_meta,
        "stage1_train_probe": probe_summary,
        "stage2_heldout_eval": heldout_summary,
        "all_record_count": sum(len(v) for v in all_records.values()),
        "elapsed_sec": round(time.time() - t0, 1),
        "timestamp": time.time(),
    }
    EVAL_SUMMARY.write_text(json.dumps(summary_doc, indent=2))

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"[V58 Eval] {'PASS' if verdict == 'claim_supported' else 'FAIL'}")
    print(f"  Train Probe: finetuned={probe_summary['finetuned_success_rate']:.3f} pretrained={probe_summary['pretrained_success_rate']:.3f} trend={probe_summary['trend_passed']}")
    print(f"  Held-out:    finetuned={heldout_summary['finetuned_success_rate']:.3f} pretrained={heldout_summary['pretrained_success_rate']:.3f} verdict={verdict}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print(f"{'='*60}")

    return verdict == "claim_supported"


if __name__ == "__main__":
    dry = "--dry" in sys.argv
    ok = run(dry=dry)
    raise SystemExit(0 if ok else 1)
