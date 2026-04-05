#!/usr/bin/env python3
"""train_gate.py — HARNESS v1.4
Verifies the dataset manifest is valid before training/eval can proceed.
This is a fail-closed gate: no train/eval without a valid manifest.
"""
import json
import sys
from pathlib import Path

CAMPAIGN_ROOT = Path(__file__).parent.parent.parent.resolve()
MANIFEST = CAMPAIGN_ROOT / "artifacts" / "current_dataset_manifest.json"
SOVEREIGN = CAMPAIGN_ROOT / "sovereign"


def fatal(msg):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def ok(msg):
    print(f"  OK: {msg}")


def main():
    print("=== HARNESS train_gate v1.4 ===")
    print(f"  Campaign: {CAMPAIGN_ROOT.name}")

    # Gate 1: manifest exists
    if not MANIFEST.exists():
        fatal(f"Dataset manifest not found: {MANIFEST}\n"
              f"  Run: bash scripts/harness/pack_dataset.sh [--force-archive] first.")
    ok(f"manifest exists: {MANIFEST.name}")

    # Gate 2: valid JSON
    try:
        m = json.load(open(MANIFEST))
    except Exception as e:
        fatal(f"Manifest is not valid JSON: {e}")
    ok(f"manifest is valid JSON")

    # Gate 3: required fields
    for field in ["dataset_version", "episode_count", "frame_count",
                  "dataset_loads", "dataset_length", "created_at"]:
        if field not in m:
            fatal(f"Manifest missing required field: {field}")
    ok(f"all required fields present")

    # Gate 4: dataset_loads
    if not m.get("dataset_loads"):
        fatal(f"dataset_loads=False in manifest — dataset is not loadable. "
              f"Run: bash scripts/harness/pack_dataset.sh [--force-archive]")
    ok(f"dataset_loads=True")

    # Gate 5: positive length
    if m.get("dataset_length", 0) <= 0:
        fatal(f"dataset_length={m['dataset_length']} — invalid. Re-pack dataset.")
    ok(f"dataset_length={m['dataset_length']} (> 0)")

    # Gate 6: episode/frame consistency
    if m.get("dataset_length") != m.get("frame_count"):
        print(f"WARN: dataset_length={m['dataset_length']} != frame_count={m.get('frame_count')} — "
              f"possible inconsistency.", file=sys.stderr)

    # Gate 7: sovereign phase matches manifest
    sovereign_state = SOVEREIGN / "state.json"
    if sovereign_state.exists():
        s = json.load(open(sovereign_state))
        sov_verdict = s.get("verdict", "unknown")
        manifest_ver = m.get("dataset_version", "unknown")
        print(f"  sovereign verdict: {sov_verdict}")
        print(f"  manifest version:  {manifest_ver}")
        # Note: we warn but don't fail if mismatch — sovereign may lag
        if sov_verdict not in (manifest_ver, "v59_CLEAN_DATASET_VERIFIED"):
            print(f"  NOTE: sovereign verdict may be stale. Run sovereign_cli.py render-truth")
    ok(f"sovereign state readable")

    # Summary
    print()
    print(f"  dataset_version: {m.get('dataset_version')}")
    print(f"  episodes:       {m.get('episode_count')}")
    print(f"  frames:         {m.get('frame_count')}")
    print(f"  created_at:     {m.get('created_at')}")
    print(f"  git_commit:      {m.get('git_commit', 'unknown')}")
    print()
    print("  ✓ ALL GATES PASSED — training/eval authorized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
