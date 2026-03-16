#!/usr/bin/env python3
"""
Encode Infinigen part meshes into 128-dim shape latent codes using NAP's
pretrained PointNet shape autoencoder.

Usage:
    conda activate physnap
    python scripts/encode_infinigen_shapes.py \
        --nap-data-dir external/physnap/data/infinigen_graph_combined \
        --ae-checkpoint external/physnap/log/s1.5_partshape_ae/checkpoint/737.pt \
        --physnap-root external/physnap

Produces: {nap-data-dir}/infinigen_codebook.npz
"""

import argparse
import json
import os
import sys

import numpy as np

PHYSNAP_ROOT = None


def build_encoder(ae_checkpoint_path):
    """Load the pretrained PointNet encoder from NAP's shape AE checkpoint."""
    sys.path.insert(0, PHYSNAP_ROOT)
    import torch
    from core.lib.point_encoder.pointnet import ResnetPointnet

    encoder = ResnetPointnet(c_dim=128, dim=3, hidden_dim=512)

    ckpt = torch.load(ae_checkpoint_path, map_location="cpu")
    state = ckpt["model_state_dict"]

    encoder_state = {}
    for k, v in state.items():
        if k.startswith("network_dict.encoder."):
            new_k = k.replace("network_dict.encoder.", "")
            encoder_state[new_k] = v

    encoder.load_state_dict(encoder_state)
    encoder.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    print(f"Encoder loaded on {device}")
    return encoder, device


def mesh_to_surface_points(vertices, faces, n_points=1024):
    """Sample points uniformly on mesh surface via area-weighted face sampling."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total_area = areas.sum()
    if total_area < 1e-12:
        idx = np.random.choice(len(vertices), n_points, replace=True)
        return vertices[idx]

    probs = areas / total_area
    face_idx = np.random.choice(len(faces), n_points, p=probs)

    r1 = np.sqrt(np.random.rand(n_points, 1))
    r2 = np.random.rand(n_points, 1)
    a = 1 - r1
    b = r1 * (1 - r2)
    c = r1 * r2

    points = a * v0[face_idx] + b * v1[face_idx] + c * v2[face_idx]
    return points.astype(np.float32)


def normalize_points(points):
    """Normalize point cloud to unit sphere (center + scale)."""
    center = (points.max(axis=0) + points.min(axis=0)) / 2.0
    points = points - center
    max_dist = np.linalg.norm(points, axis=1).max()
    if max_dist > 1e-8:
        points = points / max_dist
    return points


def encode_parts(encoder, device, nap_data_dir, partkeys_path):
    """Encode all Infinigen parts using the pretrained encoder."""
    import torch

    with open(partkeys_path) as f:
        partkeys = json.load(f)

    all_keys = partkeys["train"] + partkeys["val"] + partkeys["test"]
    n_parts = len(all_keys)
    print(f"Encoding {n_parts} parts...")

    embeddings = np.zeros((n_parts, 128), dtype=np.float32)
    valid_mask = np.zeros(n_parts, dtype=bool)

    obj_cache = {}

    for idx, key in enumerate(all_keys):
        parts = key.rsplit("_", 1)
        obj_id = parts[0]
        part_idx = int(parts[1])

        if obj_id not in obj_cache:
            npz_path = os.path.join(nap_data_dir, f"{obj_id}.npz")
            if not os.path.exists(npz_path):
                continue
            data = np.load(npz_path, allow_pickle=True)
            obj_cache[obj_id] = data["V"].tolist()

        V = obj_cache[obj_id]
        if part_idx >= len(V):
            continue

        node = V[part_idx]
        mesh_data = node.get("agg_mesh")
        if mesh_data is None:
            continue

        vertices, faces = mesh_data
        if len(vertices) < 3 or len(faces) < 1:
            continue

        surface_pts = mesh_to_surface_points(vertices, faces, n_points=1024)
        surface_pts = normalize_points(surface_pts)

        pts_tensor = torch.from_numpy(surface_pts).float().unsqueeze(0).to(device)

        with torch.no_grad():
            latent = encoder(pts_tensor)

        embeddings[idx] = latent.cpu().numpy().squeeze()
        valid_mask[idx] = True

        if (idx + 1) % 100 == 0 or idx == n_parts - 1:
            print(f"  Encoded {idx + 1}/{n_parts} parts ({valid_mask.sum()} valid)")

    obj_cache.clear()
    return embeddings, valid_mask, all_keys


def main():
    global PHYSNAP_ROOT

    parser = argparse.ArgumentParser(description="Encode Infinigen shapes with NAP AE")
    parser.add_argument("--nap-data-dir", required=True)
    parser.add_argument("--ae-checkpoint", required=True)
    parser.add_argument("--physnap-root", required=True)
    args = parser.parse_args()

    PHYSNAP_ROOT = os.path.abspath(args.physnap_root)

    partkeys_path = os.path.join(args.nap_data_dir, "infinigen_partkeys.json")
    if not os.path.exists(partkeys_path):
        print(f"ERROR: {partkeys_path} not found. Run infinigen_to_nap.py first.")
        sys.exit(1)

    encoder, device = build_encoder(os.path.abspath(args.ae_checkpoint))
    embeddings, valid_mask, all_keys = encode_parts(
        encoder, device, args.nap_data_dir, partkeys_path
    )

    std = embeddings[valid_mask].std(axis=0) if valid_mask.sum() > 1 else np.ones(128)

    codebook_path = os.path.join(args.nap_data_dir, "infinigen_codebook.npz")
    np.savez(codebook_path, embedding=embeddings, valid_mask=valid_mask, std=std)

    print(f"\nCodebook saved: {codebook_path}")
    print(f"  Total parts: {len(all_keys)}")
    print(f"  Valid: {valid_mask.sum()}")
    print(f"  Embedding range: [{embeddings[valid_mask].min():.4f}, {embeddings[valid_mask].max():.4f}]")
    print(f"  Std range: [{std.min():.4f}, {std.max():.4f}]")


if __name__ == "__main__":
    main()
