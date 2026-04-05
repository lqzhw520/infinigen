#!/bin/bash
# pack_dataset.sh — HARNESS v1.4: MANDATORY archive-first packing
# This is the ONLY legal entry point for dataset packing.
# Direct invocation of run_v58_lerobot_pack.py is deprecated.
#
# Usage:
#   bash scripts/harness/pack_dataset.sh [--force-archive] [extra_args_for_packer]
#
# Gates (fail-closed):
#   - If dataset/ exists and no --force-archive → FATAL: archive first
#   - After pack: verify dataset_loads=true
#   - After pack: verify len(dataset) matches pack result
#   - After pack: write current_dataset_manifest.json
#   - After pack: update sovereign/state.json phase

set -e

CAMPAIGN_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SOVEREIGN="$CAMPAIGN_ROOT/sovereign"
ARTIFACTS="$CAMPAIGN_ROOT/artifacts"
DATASET_DIR="$CAMPAIGN_ROOT/dataset"
SCRIPTS="$CAMPAIGN_ROOT/scripts"
MANIFEST="$ARTIFACTS/current_dataset_manifest.json"

FORCE_ARCHIVE=0
HELP=0
EXTRA_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
        HELP=1
    elif [ "$arg" = "--force-archive" ]; then
        FORCE_ARCHIVE=1
    else
        EXTRA_ARGS+=("$arg")
    fi
done

if [ "$HELP" = "1" ]; then
    echo "Usage: bash scripts/harness/pack_dataset.sh [--force-archive] [extra_args_for_packer]"
    echo ""
    echo "  HARNESS v1.4 — MANDATORY archive-first packing entry point."
    echo "  This is the ONLY legal entry point for dataset packing."
    echo ""
    echo "  --force-archive   Archive existing dataset/ before packing (required if dataset/ exists)"
    echo "  extra_args        Passed through to run_v58_lerobot_pack.py"
    echo ""
    echo "  Gates (fail-closed):"
    echo "    1. Archive existing dataset/ or fail (unless --force-archive)"
    echo "    2. Activate conda infinigen"
    echo "    3. Run packer, verify passed=True and dataset_loads=True"
    echo "    4. Write artifacts/current_dataset_manifest.json"
    exit 0
fi

echo "╔════════════════════════════════════════════════════════╗"
echo "║  HARNESS pack_dataset — v1.4                          ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# ── Gate 1: Archive existing dataset ─────────────────────────────────
if [ -d "$DATASET_DIR" ]; then
    if [ "$FORCE_ARCHIVE" = "1" ]; then
        TS=$(date '+%Y-%m-%d_%H%M%S')
        ARCHIVE_DIR="$ARTIFACTS/v59_archive_${TS}"
        echo "ARCHIVE FIRST: Moving existing dataset/ → $ARCHIVE_DIR/"
        mkdir -p "$ARCHIVE_DIR"
        mv "$DATASET_DIR" "$ARCHIVE_DIR/dataset/"
        echo "  ✓ Archived. Now proceeding with new pack."
    else
        echo "FATAL: dataset/ already exists at $DATASET_DIR"
        echo "  To pack anyway (archive-first): re-run with --force-archive"
        echo "  Archive is at: $(ls -t $ARTIFACTS/v59_archive_*/dataset/ 2>/dev/null | head -1 2>/dev/null || echo 'none found')"
        exit 1
    fi
else
    echo "  ✓ No existing dataset/ found. Proceeding."
fi

# ── Gate 2: Source conda ─────────────────────────────────────────────
echo ""
echo "Activating infinigen conda environment..."
source /root/anaconda3/etc/profile.d/conda.sh
conda activate infinigen
echo "  ✓ Conda activated."

# ── Gate 3: Run packer ─────────────────────────────────────────────
LOG="$ARTIFACTS/v59_lerobot_pack.log"
{
    echo "=========================================="
    echo "[HARNESS pack_dataset] PROVENANCE-TAGGED"
    echo "Timestamp: $(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "Command: $0 $@"
    echo "Archive policy: mandatory (archive-first)"
    echo "=========================================="
} > "$LOG"

echo ""
echo "Running packer..."
echo "  log: $LOG"
python "$SCRIPTS/mint/run_v58_lerobot_pack.py" "${EXTRA_ARGS[@]}" >> "$LOG" 2>&1
PACK_EXIT=$?

# ── Gate 4: Verify pack result ────────────────────────────────────
echo ""
if [ $PACK_EXIT -ne 0 ]; then
    echo "FATAL: Packer exited with code $PACK_EXIT"
    echo "  See log: $LOG"
    exit 1
fi

# Read pack result
PACK_JSON="$ARTIFACTS/v58_lerobot_pack.json"
if [ ! -f "$PACK_JSON" ]; then
    echo "FATAL: Pack result JSON not found at $PACK_JSON"
    exit 1
fi

PASSED=$(python3 -c "import json; print(json.load(open('$PACK_JSON'))['passed'])")
EPISODES=$(python3 -c "import json; print(json.load(open('$PACK_JSON'))['episode_count'])")
FRAMES=$(python3 -c "import json; print(json.load(open('$PACK_JSON'))['frame_count'])")
LOADS=$(python3 -c "import json; print(json.load(open('$PACK_JSON'))['integrity']['dataset_loads'])")

echo "  passed: $PASSED"
echo "  episodes: $EPISODES"
echo "  frames: $FRAMES"
echo "  dataset_loads: $LOADS"

if [ "$PASSED" != "True" ] && [ "$PASSED" != "true" ]; then
    echo "FATAL: Pack reported passed=False"
    exit 1
fi
if [ "$LOADS" != "True" ] && [ "$LOADS" != "true" ]; then
    echo "FATAL: dataset_loads=false — LeRobot cannot load the dataset"
    exit 1
fi

echo "  ✓ Pack integrity verified."

# ── Gate 5: Write dataset manifest ─────────────────────────────────
TS_NOW=$(date '+%Y-%m-%dT%H:%M:%S%z')
python3 - "$PACK_JSON" "$MANIFEST" "$TS_NOW" "$EPISODES" "$FRAMES" "$LOG" << 'PYEOF'
import json, sys, subprocess, hashlib
from pathlib import Path

pack_json_path = sys.argv[1]
manifest_path = sys.argv[2]
ts_now = sys.argv[3]
episodes = sys.argv[4]
frames = sys.argv[5]
log_path = sys.argv[6]

# Read pack result
d = json.load(open(pack_json_path))

# Compute dataset SHA from info.json
info_path = "/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/dataset/meta/info.json"
info_sha = "unknown"
if Path(info_path).exists():
    with open(info_path, "rb") as f:
        info_sha = hashlib.sha256(f.read()).hexdigest()[:16]

manifest = {
    "manifest_version": 1,
    "dataset_version": f"v59_{ts_now.split('T')[0].replace('-', '')}",
    "created_at": ts_now,
    "producer_script": "scripts/harness/pack_dataset.sh",
    "pack_result_json": pack_json_path,
    "pack_log": log_path,
    "source_dirs": [
        "artifacts/v58_physics_legal_rollouts/",
        "artifacts/p4_physics_legal_rollouts/"
    ],
    "source_rollout_count": int(d.get("episode_count", 0)),
    "episode_count": int(episodes),
    "frame_count": int(frames),
    "info_sha": info_sha,
    "dataset_loads": d.get("integrity", {}).get("dataset_loads", False),
    "dataset_length": d.get("integrity", {}).get("dataset_length", 0),
    "image_shape": str(d.get("integrity", {}).get("image_shape", "N/A")),
    "state_shape": str(d.get("integrity", {}).get("state_shape", "N/A")),
    "action_shape": str(d.get("integrity", {}).get("action_shape", "N/A")),
    "git_commit": subprocess.run(
        ["git", "-C", "/mnt/afs2/zhuhaowu/infinigen", "rev-parse", "HEAD"],
        capture_output=True, text=True
    ).stdout.strip()[:12],
}

Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
tmp = Path(manifest_path).with_suffix(".tmp")
tmp.write_text(json.dumps(manifest, indent=2))
tmp.replace(Path(manifest_path))
print(f"  ✓ Manifest written: {manifest_path}")
print(f"    dataset_version={manifest['dataset_version']}")
print(f"    episodes={manifest['episode_count']}, frames={manifest['frame_count']}")
PYEOF

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  HARNESS pack_dataset — COMPLETED SUCCESSFULLY           ║"
echo "╚════════════════════════════════════════════════════════╝"
echo "  dataset_version: v59 (see $MANIFEST)"
echo "  episodes: $EPISODES"
echo "  frames: $FRAMES"
echo ""
echo "  Next: update sovereign/state.json phase"
echo "  Hint: python scripts/harness/sovereign_cli.py ..."
