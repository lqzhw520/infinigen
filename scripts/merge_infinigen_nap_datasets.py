#!/usr/bin/env python3
"""
Merge multiple per-box-type NAP graph datasets into a single combined dataset.

Usage:
    python scripts/merge_infinigen_nap_datasets.py \
        --input-dirs external/physnap/data/infinigen_graph_mailer \
                     external/physnap/data/infinigen_graph_drawer \
                     external/physnap/data/infinigen_graph_sliplid \
                     external/physnap/data/infinigen_graph_tuckend \
        --output-dir external/physnap/data/infinigen_graph_combined_k10
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def merge_datasets(input_dirs, output_dir, exclude_types=None):
    os.makedirs(output_dir, exist_ok=True)
    exclude_types = set(t.lower() for t in (exclude_types or []))

    combined_split = {}
    combined_partkeys = {"train": [], "val": [], "test": []}
    total_copied = 0

    for input_dir in input_dirs:
        input_dir = Path(input_dir)
        meta_path = input_dir / "conversion_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            box_type = meta.get("box_type", "unknown").lower()
        else:
            box_type = input_dir.name.replace("infinigen_graph_", "")

        if box_type in exclude_types:
            print(f"Skipping {box_type} (excluded)")
            continue

        split_path = input_dir / "infinigen_split.json"
        partkeys_path = input_dir / "infinigen_partkeys.json"

        if not split_path.exists() or not partkeys_path.exists():
            print(f"Warning: Missing split/partkeys in {input_dir}, skipping")
            continue

        with open(split_path) as f:
            split = json.load(f)
        with open(partkeys_path) as f:
            partkeys = json.load(f)

        for cat_name, cat_split in split.items():
            combined_split[cat_name] = cat_split

        for phase in ["train", "val", "test"]:
            combined_partkeys[phase].extend(partkeys.get(phase, []))

        npz_files = list(input_dir.glob("*.npz"))
        for npz in npz_files:
            dst = Path(output_dir) / npz.name
            if not dst.exists():
                shutil.copy2(str(npz), str(dst))
            total_copied += 1

        print(f"  {box_type}: {len(npz_files)} samples copied")

    with open(os.path.join(output_dir, "infinigen_split.json"), "w") as f:
        json.dump(combined_split, f, indent=2)

    with open(os.path.join(output_dir, "infinigen_partkeys.json"), "w") as f:
        json.dump(combined_partkeys, f, indent=2)

    print("\nMerged dataset:")
    print(f"  Total samples: {total_copied}")
    print(f"  Categories: {list(combined_split.keys())}")
    for phase in ["train", "val", "test"]:
        count = sum(len(s[phase]) for s in combined_split.values())
        print(f"  {phase}: {count} objects, {len(combined_partkeys[phase])} parts")


def main():
    parser = argparse.ArgumentParser(description="Merge per-box-type NAP datasets")
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude-types", nargs="*", default=[])
    args = parser.parse_args()

    merge_datasets(args.input_dirs, args.output_dir, args.exclude_types)


if __name__ == "__main__":
    main()
