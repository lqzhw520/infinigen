#!/usr/bin/env python3
"""Patch gripper_joint in existing V58 LeRobot dataset parquet files.

Root cause: dataset_builder._build_libero_state() incorrectly called
_infinigen_to_libero_gripper(state[7]) on ALREADY-continuous gripper_joint values,
freezing all frames to -0.042.

Fix: 
1. Load NPZ files in same order as build_dataset_from_rollouts (alphabetical)
2. Map episode_index -> NPZ path
3. Extract per-frame gripper_joint from raw NPZ states, clip to LIBERO range
4. Patch parquet with corrected gripper_joint values

Usage:
    screen -dmS v58_patch bash -c 'source /root/anaconda3/etc/profile.d/conda.sh && conda activate infinigen && cd /mnt/afs2/zhuhaowu/infinigen && python scripts/mint/patch_gripper_joint_dataset.py 2>&1 | tee experiments/mint/mint_drawer_v1/outputs/v58_gripper_patch.log'
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
SCRIPTS_MINT = PROJECT_ROOT / "scripts" / "mint"
DATASET_ROOT = PROJECT_ROOT / "experiments/mint/mint_drawer_v1/dataset"
LIBERO_MIN = -0.042
LIBERO_MAX = +0.001


def _clip_gripper(v: float) -> float:
    return float(max(LIBERO_MIN, min(LIBERO_MAX, float(v))))


def _load_npz_gripper_joint_series(npz_path: Path) -> np.ndarray | None:
    """Load per-frame gripper_joint from NPZ, clipped to LIBERO range.
    
    Returns None if NPZ is invalid or has no states.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        states = data.get("states", np.zeros((0, 8)))
        if len(states) == 0:
            return None
        
        raw_gripper = states[:, 7].astype(np.float32)
        
        # Check if values are already continuous (not binary frozen)
        unique_vals = np.unique(raw_gripper)
        
        if len(unique_vals) <= 2:
            # Frozen/binary gripper in NPZ - use gripper_values to reconstruct
            gripper_binary = data.get("gripper_values", np.zeros(len(states)))
            if len(gripper_binary) != len(states):
                return None
            result = np.where(
                gripper_binary > 0.5,
                np.full(len(states), LIBERO_MAX, dtype=np.float32),
                np.full(len(states), LIBERO_MIN, dtype=np.float32),
            )
            return result
        else:
            # Already continuous - just clip to LIBERO range
            return np.array([_clip_gripper(v) for v in raw_gripper], dtype=np.float32)
    except Exception as e:
        print(f"    [error] loading {npz_path.name}: {e}")
        return None


def _build_episode_npz_map(
    npz_paths: list[Path],
    n_total_episodes: int
) -> list[Path | None]:
    """Build episode_index -> NPZ path mapping.
    
    Episodes are written in order of sorted NPZ filenames.
    Each episode has 1-N frames, and episodes are sequential (0, 1, 2, ...).
    """
    # sort alphabetically (same as build_dataset_from_rollouts)
    sorted_npz = sorted(npz_paths)
    
    # Build: episode_idx -> npz_path
    episode_map: list[Path | None] = [None] * n_total_episodes
    for ep_idx, npz_path in enumerate(sorted_npz):
        if ep_idx < n_total_episodes:
            episode_map[ep_idx] = npz_path
        else:
            break
    
    return episode_map


def patch_dataset() -> dict:
    """Patch all parquet files in the LeRobot dataset."""
    def find_legal_npz(source_dirs):
        paths = []
        for d in source_dirs:
            if not d.exists():
                continue
            for pf in sorted(d.glob("*.npz")):
                try:
                    data = dict(np.load(pf, allow_pickle=True))
                    if data.get("success"):
                        paths.append(pf)
                except Exception:
                    pass
        return paths

    from scripts.mint.mint_common import ARTIFACT_DIR
    v58_dir = ARTIFACT_DIR / "v58_physics_legal_rollouts"
    p4_dir = ARTIFACT_DIR / "p4_physics_legal_rollouts"
    npz_paths = find_legal_npz([v58_dir, p4_dir])
    
    print(f"[patch] Found {len(npz_paths)} NPZ files")
    print(f"[patch] NPZ order: {[p.name for p in sorted(npz_paths)[:5]]}...")

    chunk_dirs = sorted(DATASET_ROOT.glob("data/chunk-*"))
    if not chunk_dirs:
        return {"passed": False, "error": "No chunks found", "files_patched": 0}
    
    # Count total episodes from all parquet files
    total_episodes = 0
    parquet_files = []
    for chunk_dir in chunk_dirs:
        for pf in sorted(chunk_dir.glob("file-*.parquet")):
            pf_data = pq.read_table(str(pf))
            df = pf_data.to_pandas()
            n_eps = int(df["episode_index"].max()) + 1 if len(df) > 0 else 0
            parquet_files.append((pf, df, n_eps))
            total_episodes = max(total_episodes, int(df["episode_index"].max()) + 1)
    
    print(f"[patch] Total episodes in dataset: {total_episodes}")
    
    # Build episode -> NPZ mapping
    episode_npz_map = _build_episode_npz_map(npz_paths, total_episodes)
    
    # Pre-load all gripper series into a dict {ep_idx: np.ndarray}
    print("[patch] Pre-loading gripper_joint series from NPZs...")
    ep_gripper_cache: dict[int, np.ndarray] = {}
    for ep_idx, npz_path in enumerate(episode_npz_map):
        if npz_path is not None:
            gripper_series = _load_npz_gripper_joint_series(npz_path)
            if gripper_series is not None:
                ep_gripper_cache[ep_idx] = gripper_series
                print(f"  ep{ep_idx}: {len(gripper_series)} frames, "
                      f"gripper range [{gripper_series.min():.4f}, {gripper_series.max():.4f}]")
            else:
                print(f"  ep{ep_idx}: {npz_path.name} - FAILED to load")
    
    # Now patch each parquet file
    total_patched = 0
    total_frames = 0
    total_changed = 0
    total_errors = 0
    
    for pf_path, df_orig, n_eps in parquet_files:
        print(f"[patch] Processing {pf_path.name}...")
        try:
            df = df_orig.copy()
            state_col = "observation.state"
            
            ep_indices = df["episode_index"].values
            frame_indices = df["frame_index"].values
            
            new_states = []
            changed_count = 0
            
            for i in range(len(df)):
                ep_idx = int(ep_indices[i])
                frame_idx = int(frame_indices[i])
                
                state = df.iloc[i][state_col]
                if isinstance(state, np.ndarray):
                    state = state.copy()
                else:
                    state = np.array(state, dtype=np.float32).copy()
                
                old_g = float(state[7])
                
                gripper_series = ep_gripper_cache.get(ep_idx)
                if gripper_series is not None and frame_idx < len(gripper_series):
                    new_g = float(gripper_series[frame_idx])
                else:
                    # Fallback: clip old value to LIBERO range
                    new_g = _clip_gripper(old_g)
                
                state[7] = new_g
                new_states.append(state)
                
                if abs(new_g - old_g) > 1e-6:
                    changed_count += 1
            
            # Replace state column with patched values
            df[state_col] = new_states
            
            # Write back with same compression
            pq.write_table(
                pa.table(df),
                str(pf_path),
                compression="snappy",
            )
            
            total_patched += 1
            total_frames += len(df)
            total_changed += changed_count
            
            print(f"  [ok] {pf_path.name}: {changed_count}/{len(df)} frames changed")
        
        except Exception as exc:
            print(f"  [error] {pf_path.name}: {exc}")
            total_errors += 1
    
    # Verify: compute gripper_joint stats from patched parquet
    print("\n[patch] Verifying patched dataset...")
    gripper_vals = []
    for pf, _, _ in parquet_files:
        df = pq.read_table(str(pf)).to_pandas()
        for row in df["observation.state"]:
            if isinstance(row, np.ndarray):
                gripper_vals.append(float(row[7]))
            else:
                gripper_vals.append(float(np.array(row)[7]))
    
    gripper_arr = np.array(gripper_vals, dtype=np.float32)
    n_unique = len(np.unique(gripper_arr.round(decimals=5)))
    n_frozen = np.sum(np.abs(gripper_arr - LIBERO_MIN) < 1e-6)
    pct_frozen = n_frozen / len(gripper_arr) * 100
    
    print(f"  Total frames: {len(gripper_arr)}")
    print(f"  Gripper range: [{gripper_arr.min():.4f}, {gripper_arr.max():.4f}]")
    print(f"  Gripper std: {gripper_arr.std():.6f}")
    print(f"  Unique values: {n_unique}")
    print(f"  Frames at LIBERO_MIN (-0.042): {n_frozen} ({pct_frozen:.1f}%)")
    
    gripper_ok = (
        gripper_arr.std() > 0.001 and
        n_unique > 10 and
        gripper_arr.min() >= LIBERO_MIN - 1e-6 and
        gripper_arr.max() <= LIBERO_MAX + 1e-6
    )
    print(f"  Gripper FIXED: {gripper_ok}")
    
    return {
        "passed": gripper_ok and total_errors == 0,
        "files_patched": total_patched,
        "files_errors": total_errors,
        "frames_patched": total_frames,
        "frames_changed": total_changed,
        "gripper_stats": {
            "min": float(gripper_arr.min()),
            "max": float(gripper_arr.max()),
            "std": float(gripper_arr.std()),
            "n_unique": int(n_unique),
            "n_frozen_at_min": int(n_frozen),
            "pct_frozen": float(pct_frozen),
            "gripper_ok": gripper_ok,
        },
    }


def main():
    t0 = time.time()
    print(f"[patch] Starting gripper_joint patch")
    print(f"[patch] Dataset: {DATASET_ROOT}")
    
    result = patch_dataset()
    elapsed = time.time() - t0
    
    print(f"\n[patch] Done in {elapsed:.1f}s")
    print(f"[patch] Files patched: {result.get('files_patched', 0)}")
    print(f"[patch] Frames changed: {result.get('frames_changed', 0)}")
    print(f"[patch] Gripper std: {result.get('gripper_stats', {}).get('std', 'N/A')}")
    print(f"[patch] PASSED: {result.get('passed', False)}")
    
    # Save result
    result_path = DATASET_ROOT.parent / "v58_gripper_patch_result.json"
    result["timestamp"] = time.time()
    result["elapsed_sec"] = elapsed
    result_path.write_text(json.dumps(result, indent=2))
    print(f"[patch] Result saved to: {result_path}")
    
    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(SCRIPTS_MINT))
    result = main()
    raise SystemExit(0 if result.get("passed") else 1)
