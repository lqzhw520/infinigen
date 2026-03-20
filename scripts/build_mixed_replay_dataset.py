#!/usr/bin/env python3
"""Build a symlinked PartNet + Infinigen mixed replay dataset for PhysNAP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def dataset_index_map(partkeys: dict) -> tuple[list[str], dict[str, int]]:
    ordered = partkeys["train"] + partkeys["val"] + partkeys["test"]
    return ordered, {key: idx for idx, key in enumerate(ordered)}


def symlink_npzs(src_root: Path, dst_root: Path) -> int:
    created = 0
    for npz in sorted(src_root.glob("*.npz")):
        if npz.name.endswith("codebook.npz"):
            continue
        dst = dst_root / npz.name
        if dst.exists() or dst.is_symlink():
            continue
        dst.symlink_to(npz)
        created += 1
    return created


def merge_splits(partnet_split: dict, infinigen_split: dict) -> dict:
    merged = {}
    merged.update(partnet_split)
    merged.update(infinigen_split)
    return merged


def build_codebook(
    output_path: Path,
    combined_partkeys: dict,
    partnet_partkeys: dict,
    partnet_codebook: np.lib.npyio.NpzFile,
    infinigen_partkeys: dict,
    infinigen_codebook: np.lib.npyio.NpzFile,
) -> None:
    ordered_all = combined_partkeys["train"] + combined_partkeys["val"] + combined_partkeys["test"]
    partnet_ordered, partnet_lookup = dataset_index_map(partnet_partkeys)
    infinigen_ordered, infinigen_lookup = dataset_index_map(infinigen_partkeys)

    dim = int(partnet_codebook["embedding"].shape[1])
    embedding = np.zeros((len(ordered_all), dim), dtype=np.float32)
    valid_mask = np.zeros(len(ordered_all), dtype=bool)

    for row_idx, key in enumerate(ordered_all):
        if key in partnet_lookup:
            src_idx = partnet_lookup[key]
            embedding[row_idx] = partnet_codebook["embedding"][src_idx]
            valid_mask[row_idx] = bool(partnet_codebook["valid_mask"][src_idx])
        elif key in infinigen_lookup:
            src_idx = infinigen_lookup[key]
            embedding[row_idx] = infinigen_codebook["embedding"][src_idx]
            valid_mask[row_idx] = bool(infinigen_codebook["valid_mask"][src_idx])
        else:
            raise KeyError(f"Missing code for part key {key}")

    std = embedding[valid_mask].std(axis=0) if valid_mask.any() else np.ones(dim, dtype=np.float32)
    np.savez(output_path, embedding=embedding, valid_mask=valid_mask, std=std.astype(np.float32))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build mixed replay dataset for PartNet + Infinigen")
    parser.add_argument("--partnet-data-root", required=True)
    parser.add_argument("--partnet-split", required=True)
    parser.add_argument("--partnet-partkeys", required=True)
    parser.add_argument("--partnet-codebook", required=True)
    parser.add_argument("--infinigen-data-root", required=True)
    parser.add_argument("--infinigen-split", required=True)
    parser.add_argument("--infinigen-partkeys", required=True)
    parser.add_argument("--infinigen-codebook", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    partnet_data_root = Path(args.partnet_data_root)
    infinigen_data_root = Path(args.infinigen_data_root)
    partnet_split = load_json(Path(args.partnet_split))
    infinigen_split = load_json(Path(args.infinigen_split))
    partnet_partkeys = load_json(Path(args.partnet_partkeys))
    infinigen_partkeys = load_json(Path(args.infinigen_partkeys))

    combined_split = merge_splits(partnet_split, infinigen_split)
    combined_partkeys = {
        phase: partnet_partkeys[phase] + infinigen_partkeys[phase]
        for phase in ["train", "val", "test"]
    }

    created = 0
    created += symlink_npzs(partnet_data_root, output_dir)
    created += symlink_npzs(infinigen_data_root, output_dir)

    save_json(output_dir / "mixed_split.json", combined_split)
    save_json(output_dir / "mixed_partkeys.json", combined_partkeys)

    with np.load(args.partnet_codebook, allow_pickle=True) as partnet_codebook:
        with np.load(args.infinigen_codebook, allow_pickle=True) as infinigen_codebook:
            build_codebook(
                output_dir / "mixed_codebook.npz",
                combined_partkeys,
                partnet_partkeys,
                partnet_codebook,
                infinigen_partkeys,
                infinigen_codebook,
            )

    summary = {
        "partnet_npz_root": str(partnet_data_root),
        "infinigen_npz_root": str(infinigen_data_root),
        "output_dir": str(output_dir),
        "symlinked_npz": created,
        "split_counts": {
            phase: len(combined_partkeys[phase]) for phase in ["train", "val", "test"]
        },
    }
    save_json(output_dir / "mixed_dataset_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
