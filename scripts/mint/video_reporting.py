from __future__ import annotations

from pathlib import Path

from evaluate_mint_drawer_campaign import record_policy_video
from render_rollout_video import render_rollout_video


def _ensure_stage_dir(campaign_dir: Path, stage: str) -> Path:
    path = campaign_dir / "artifacts" / "videos" / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def choose_best_success_rollout(summary: dict, key: str) -> str | None:
    payload = summary.get(key) or {}
    records = payload.get("records") or []
    best = None
    best_score = None
    for seed_record in records:
        for attempt in seed_record.get("attempts", []):
            if not attempt.get("rollout_path"):
                continue
            if not attempt.get("success"):
                continue
            score = float(attempt.get("drawer_fraction_final", 0.0))
            if best is None or score > best_score:
                best = attempt["rollout_path"]
                best_score = score
    return best


def safe_render_teacher_rollout(
    *,
    campaign_dir: Path,
    rollout_path: str | Path | None,
    stage: str,
    policy: str,
    attempt: str | None,
    variant: str | None,
) -> str | None:
    if not rollout_path:
        return None
    path = Path(rollout_path)
    if not path.exists():
        return None
    stage_dir = _ensure_stage_dir(campaign_dir, stage)
    output = stage_dir / f"{path.stem}_{policy}_{variant or 'default'}.mp4"
    render_rollout_video(
        path,
        output,
        stage=stage,
        policy=policy,
        attempt=attempt,
        variant=variant,
        fps=10,
    )
    return str(output)


def safe_render_teacher_rollout_report(
    *,
    campaign_dir: Path,
    rollout_path: str | Path | None,
    stage: str,
    policy: str,
    attempt: str | None,
    variant: str | None,
) -> str | None:
    if not rollout_path:
        return None
    path = Path(rollout_path)
    if not path.exists():
        return None
    stage_dir = _ensure_stage_dir(campaign_dir, stage)
    output = stage_dir / f"{path.stem}_{policy}_{variant or 'default'}_report_safe.mp4"
    render_rollout_video(
        path,
        output,
        stage=stage,
        policy=policy,
        attempt=attempt,
        variant=variant,
        fps=10,
        view_mode="primary_only",
        report_safe=True,
    )
    return str(output)


def safe_render_policy_pair(
    *,
    campaign_dir: Path,
    stage: str,
    seed: int,
    finetuned_path: str,
    dataset_root: str | Path,
    repo_id: str,
    variant: str | None,
    max_steps: int = 96,
    attempt_idx: int = 0,
) -> dict:
    stage_dir = _ensure_stage_dir(campaign_dir, stage)
    outputs: dict[str, str] = {}
    for policy in ("pretrained_mint", "finetuned_mint"):
        output = (
            stage_dir
            / f"seed_{seed:03d}_{policy}_{variant or 'default'}_attempt_{attempt_idx}.mp4"
        )
        result = record_policy_video(
            seed=seed,
            kind=policy,
            output_path=output,
            finetuned_path=finetuned_path,
            dataset_root=Path(dataset_root),
            repo_id=repo_id,
            max_steps=max_steps,
            attempt_idx=attempt_idx,
            stage=stage,
            variant=variant,
        )
        outputs[policy] = (
            str(result["video_path"]) if result.get("video_path") else str(output)
        )
    return outputs


def choose_representative_seed(
    comparison: dict,
    *,
    preferred_policy: str = "finetuned_mint",
    fallback_policy: str = "pretrained_mint",
) -> int | None:
    preferred = comparison.get(preferred_policy, {}).get("per_seed", {})
    ranked = sorted(
        preferred.items(),
        key=lambda item: (
            float(item[1].get("success_rate", 0.0)),
            float(item[1].get("grasp_success_rate", 0.0)),
            float(item[1].get("pull_distance_mean", 0.0)),
        ),
        reverse=True,
    )
    for seed, metrics in ranked:
        if (
            float(metrics.get("success_rate", 0.0)) > 0.0
            or float(metrics.get("grasp_success_rate", 0.0)) > 0.0
        ):
            return int(seed)
    fallback = comparison.get(fallback_policy, {}).get("per_seed", {})
    ranked = sorted(
        fallback.items(),
        key=lambda item: (
            float(item[1].get("success_rate", 0.0)),
            float(item[1].get("grasp_success_rate", 0.0)),
            float(item[1].get("pull_distance_mean", 0.0)),
        ),
        reverse=True,
    )
    if ranked:
        return int(ranked[0][0])
    return None
