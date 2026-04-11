#!/usr/bin/env python3
"""P1A: audit visual semantics with canonical and diagnostic-only lanes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mint_common import now_iso, write_json_atomic, write_text_atomic
from p1_execution_common import ARTIFACT_DIR, OUTPUT_DIR, upsert_evidence

CONTROL_TRACE_ROOT = ARTIFACT_DIR / "p1k_control_traces"
M4_ARTIFACT = ARTIFACT_DIR / "m4_robot_rollout_gate.json"
CANONICAL_ARTIFACT = ARTIFACT_DIR / "p1n_visual_semantics_canonical.json"
DIAGNOSTIC_ARTIFACT = ARTIFACT_DIR / "p1n_visual_semantics_ablation.json"
REPORT_PATH = OUTPUT_DIR / "p1n_visual_semantics.md"
EVIDENCE_ID = "E056"
EXPERIMENT_ID = "p1n_visual_semantics_canonical"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-diagnostic", action="store_true")
    return parser.parse_args()


def image_metrics(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    arr = rgb.astype(np.float32) / 255.0
    flat = arr.reshape(-1, 3)
    mean_rgb = flat.mean(axis=0)
    std_rgb = flat.std(axis=0)
    gray = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean() / 255.0)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float((edges > 0).mean())
    hist = np.bincount(gray.flatten(), minlength=256).astype(np.float64)
    hist /= max(hist.sum(), 1.0)
    nonzero = hist[hist > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())
    return mean_rgb, std_rgb, brightness, edge_density, entropy


def summarize_visual(samples: list[np.ndarray]) -> dict[str, Any]:
    means, stds, brightness, edges, entropies = [], [], [], [], []
    for rgb in samples:
        m, s, b, e, h = image_metrics(rgb)
        means.append(m)
        stds.append(s)
        brightness.append(b)
        edges.append(e)
        entropies.append(h)
    return {
        "sample_count": len(samples),
        "mean_rgb": np.mean(np.stack(means), axis=0).round(6).tolist(),
        "std_rgb": np.mean(np.stack(stds), axis=0).round(6).tolist(),
        "brightness_mean": float(np.mean(brightness)),
        "edge_density_mean": float(np.mean(edges)),
        "entropy_mean": float(np.mean(entropies)),
    }


def gap(control: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    return {
        "brightness_gap": round(abs(control["brightness_mean"] - target["brightness_mean"]), 6),
        "std_rgb_abs_gap": [round(abs(a - b), 6) for a, b in zip(control["std_rgb"], target["std_rgb"])],
        "edge_density_gap": round(abs(control["edge_density_mean"] - target["edge_density_mean"]), 6),
        "entropy_gap": round(abs(control["entropy_mean"] - target["entropy_mean"]), 6),
    }


def _load_control_images(limit: int = 60) -> list[np.ndarray]:
    frames = []
    for trace_path in sorted(CONTROL_TRACE_ROOT.glob("episode_*/trace.npz")):
        data = np.load(trace_path)
        for frame in data["image"]:
            img = np.asarray(frame, dtype=np.float32)
            if img.max() <= 1.0:
                img = img * 255.0
            frames.append(np.clip(img, 0, 255).astype(np.uint8))
            if len(frames) >= limit:
                return frames
    return frames


def _load_target_images(limit: int = 60) -> list[np.ndarray]:
    m4 = json.loads(M4_ARTIFACT.read_text())
    learning_dir = Path(m4["learning_rollout_dir"])
    frames = []
    for path in sorted(learning_dir.glob("*.npz")):
        data = np.load(path)
        for frame in data["images"]:
            frames.append(np.asarray(frame, dtype=np.uint8))
            if len(frames) >= limit:
                return frames
    return frames


def _diagnostic_texture_enhancement(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    texture = gray.astype(np.float32) - blur.astype(np.float32)
    enhanced = rgb.astype(np.float32) + texture[..., None] * 0.35
    return np.clip(enhanced, 0.0, 255.0).astype(np.uint8)


def main() -> int:
    args = parse_args()
    control_samples = _load_control_images()
    canonical_samples = _load_target_images()
    control_stats = summarize_visual(control_samples)
    canonical_stats = summarize_visual(canonical_samples)
    canonical_gap = gap(control_stats, canonical_stats)
    canonical_pass = (
        canonical_gap["edge_density_gap"] <= 0.03
        and canonical_gap["entropy_gap"] <= 3.5
        and float(np.mean(canonical_gap["std_rgb_abs_gap"])) <= 0.08
    )

    canonical_payload = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at": now_iso(),
        "passed": canonical_pass,
        "lane": "canonical",
        "control_stats": control_stats,
        "target_stats": canonical_stats,
        "gap": canonical_gap,
        "gate": {
            "edge_density_gap_le_0_03": canonical_gap["edge_density_gap"] <= 0.03,
            "entropy_gap_le_3_5": canonical_gap["entropy_gap"] <= 3.5,
            "mean_std_rgb_gap_le_0_08": float(np.mean(canonical_gap["std_rgb_abs_gap"])) <= 0.08,
        },
        "note": "Canonical lane uses current render-level images only. No post-render texture injection is applied here.",
    }
    write_json_atomic(CANONICAL_ARTIFACT, canonical_payload)

    diagnostic_payload = None
    if args.run_diagnostic:
        diagnostic_samples = [_diagnostic_texture_enhancement(frame) for frame in canonical_samples]
        diagnostic_stats = summarize_visual(diagnostic_samples)
        diagnostic_gap = gap(control_stats, diagnostic_stats)
        diagnostic_payload = {
            "experiment_id": "p1n_visual_semantics_ablation",
            "generated_at": now_iso(),
            "passed": False,
            "lane": "diagnostic_only",
            "control_stats": control_stats,
            "target_stats": diagnostic_stats,
            "gap": diagnostic_gap,
            "note": "Diagnostic-only unsharp texture enhancement applied after render. This lane must not become canonical training data without later encoder-level justification.",
        }
        write_json_atomic(DIAGNOSTIC_ARTIFACT, diagnostic_payload)

    lines = [
        "# p1n Visual Semantics",
        "",
        f"Generated: {canonical_payload['generated_at']}",
        "",
        "## Canonical Lane",
        f"- passed: {canonical_pass}",
        f"- brightness_gap: {canonical_gap['brightness_gap']}",
        f"- edge_density_gap: {canonical_gap['edge_density_gap']}",
        f"- entropy_gap: {canonical_gap['entropy_gap']}",
        f"- mean std_rgb gap: {float(np.mean(canonical_gap['std_rgb_abs_gap'])):.6f}",
    ]
    if diagnostic_payload is not None:
        lines.extend([
            "",
            "## Diagnostic-Only Lane",
            f"- brightness_gap: {diagnostic_payload['gap']['brightness_gap']}",
            f"- edge_density_gap: {diagnostic_payload['gap']['edge_density_gap']}",
            f"- entropy_gap: {diagnostic_payload['gap']['entropy_gap']}",
            f"- mean std_rgb gap: {float(np.mean(diagnostic_payload['gap']['std_rgb_abs_gap'])):.6f}",
        ])
    write_text_atomic(REPORT_PATH, "\n".join(lines).rstrip() + "\n")

    upsert_evidence(
        EVIDENCE_ID,
        EXPERIMENT_ID,
        "visual_semantics_canonical",
        CANONICAL_ARTIFACT,
        "Canonical visual semantics audit comparing control traces to current render-level target images, with any post-render texture injection explicitly isolated to diagnostic-only ablations.",
        verified=canonical_pass,
    )
    print(json.dumps(canonical_payload, indent=2, ensure_ascii=False))
    return 0 if canonical_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
