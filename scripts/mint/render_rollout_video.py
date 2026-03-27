#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def _load_meta(path: Path) -> dict:
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def _maybe_array(data: np.lib.npyio.NpzFile, key: str):
    if key not in data.files:
        return None
    return data[key]


def _to_bool(value) -> bool:
    if isinstance(value, np.bool_):
        return bool(value)
    return bool(value)


def render_rollout_video(
    rollout_path: Path,
    output_path: Path,
    *,
    stage: str,
    policy: str,
    attempt: str | None,
    variant: str | None,
    fps: int,
    view_mode: str = "stereo",
    report_safe: bool = False,
) -> Path:
    meta = _load_meta(rollout_path)
    data = np.load(rollout_path, allow_pickle=True)
    images = _maybe_array(data, "images")
    images2 = _maybe_array(data, "images2")
    if images is None:
        raise ValueError(f"{rollout_path} is missing `images`")
    if images2 is None:
        images2 = images

    phase_labels = _maybe_array(data, "phase_labels")
    drawer_trace = _maybe_array(data, "absolute_drawer_fraction")
    attached_trace = _maybe_array(data, "attached_trace")
    if phase_labels is None:
        phase_labels = np.array(["unknown"] * len(images), dtype=object)
    if drawer_trace is None:
        drawer_trace = np.zeros(len(images), dtype=np.float32)
    if attached_trace is None:
        attached_trace = np.zeros(len(images), dtype=np.bool_)

    seed = meta.get("seed", "unknown")
    episode = meta.get("episode_index", meta.get("episode", "unknown"))
    strict_success = meta.get("strict_metrics", {}).get("passed")
    if strict_success is None:
        strict_success = meta.get("success")

    frames = []
    for idx in range(len(images)):
        left = Image.fromarray(images[idx])
        right = Image.fromarray(images2[idx])
        if report_safe:
            left = ImageOps.autocontrast(left, cutoff=1)
            right = ImageOps.autocontrast(right, cutoff=1)
        if view_mode == "primary_only":
            primary = left.resize(
                (left.width * 2, left.height * 2), Image.Resampling.NEAREST
            )
            canvas = Image.new(
                "RGB", (primary.width, primary.height + 68), (18, 18, 18)
            )
            canvas.paste(primary, (0, 68))
        else:
            canvas = Image.new(
                "RGB",
                (left.width + right.width, max(left.height, right.height) + 68),
                (18, 18, 18),
            )
            canvas.paste(left, (0, 68))
            canvas.paste(right, (left.width, 68))
        draw = ImageDraw.Draw(canvas)

        header = f"stage={stage} policy={policy} seed={seed} episode={episode}"
        line2 = (
            f"attempt={attempt or '-'} variant={variant or '-'} "
            f"strict_success={strict_success} view={view_mode}"
        )
        line3 = (
            f"frame={idx:03d} phase={phase_labels[idx]} "
            f"attached={_to_bool(attached_trace[idx])} drawer={float(drawer_trace[idx]):.3f}"
        )
        line4 = (
            f"branch={meta.get('branch_id', meta.get('source_branch_id', 'unknown'))} "
            f"teacher_mode={meta.get('teacher_mode', 'unknown')}"
        )
        draw.text((8, 8), header, fill=(255, 255, 255))
        draw.text((8, 24), line2, fill=(180, 220, 255))
        draw.text(
            (8, 40),
            line3,
            fill=(180, 255, 180) if _to_bool(attached_trace[idx]) else (255, 220, 180),
        )
        draw.text((8, 56), line4, fill=(255, 255, 180))
        frames.append(np.array(canvas))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    return output_path


def render_trace_video(
    *,
    images: list[np.ndarray],
    images2: list[np.ndarray] | None,
    output_path: Path,
    stage: str,
    policy: str,
    seed: int | str,
    attempt: str | None,
    variant: str | None,
    drawer_trace: list[float],
    attached_trace: list[bool],
    strict_success: bool | None,
    phase_labels: list[str] | None = None,
    extra_line: str | None = None,
    fps: int = 10,
    view_mode: str = "stereo",
    report_safe: bool = False,
) -> Path:
    if not images:
        raise ValueError("trace video requires at least one frame")
    if images2 is None:
        images2 = images
    if phase_labels is None:
        phase_labels = ["unknown"] * len(images)

    frames = []
    for idx in range(len(images)):
        left = Image.fromarray(images[idx])
        right = Image.fromarray(images2[idx])
        if report_safe:
            left = ImageOps.autocontrast(left, cutoff=1)
            right = ImageOps.autocontrast(right, cutoff=1)
        if view_mode == "primary_only":
            primary = left.resize(
                (left.width * 2, left.height * 2), Image.Resampling.NEAREST
            )
            canvas = Image.new(
                "RGB", (primary.width, primary.height + 68), (18, 18, 18)
            )
            canvas.paste(primary, (0, 68))
        else:
            canvas = Image.new(
                "RGB",
                (left.width + right.width, max(left.height, right.height) + 68),
                (18, 18, 18),
            )
            canvas.paste(left, (0, 68))
            canvas.paste(right, (left.width, 68))
        draw = ImageDraw.Draw(canvas)
        header = f"stage={stage} policy={policy} seed={seed}"
        line2 = (
            f"attempt={attempt or '-'} variant={variant or '-'} "
            f"strict_success={strict_success} view={view_mode}"
        )
        line3 = (
            f"frame={idx:03d} phase={phase_labels[idx]} "
            f"attached={bool(attached_trace[idx])} drawer={float(drawer_trace[idx]):.3f}"
        )
        draw.text((8, 8), header, fill=(255, 255, 255))
        draw.text((8, 24), line2, fill=(180, 220, 255))
        draw.text(
            (8, 40),
            line3,
            fill=(180, 255, 180) if bool(attached_trace[idx]) else (255, 220, 180),
        )
        if extra_line:
            draw.text((8, 56), extra_line, fill=(255, 255, 180))
        frames.append(np.array(canvas))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a rollout NPZ to MP4 with MINT stage overlays."
    )
    parser.add_argument("rollout", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stage", default="unknown")
    parser.add_argument("--policy", default="unknown")
    parser.add_argument("--attempt", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--view-mode", choices=["stereo", "primary_only"], default="stereo"
    )
    parser.add_argument("--report-safe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = render_rollout_video(
        args.rollout,
        args.output,
        stage=args.stage,
        policy=args.policy,
        attempt=args.attempt,
        variant=args.variant,
        fps=args.fps,
        view_mode=args.view_mode,
        report_safe=args.report_safe,
    )
    print(path)


if __name__ == "__main__":
    main()
