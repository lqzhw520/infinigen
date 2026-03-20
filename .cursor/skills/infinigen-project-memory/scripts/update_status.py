#!/usr/bin/env python3
"""Regenerate .project-memory/STATUS.md from evolution.json + live campaign state."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

GIT_CANDIDATES = [
    Path("/root/anaconda3/envs/infinigen/bin/git"),
    Path("/usr/bin/git"),
    Path("/bin/git"),
]
CAMPAIGN_NAMES = ["box_prior_v1", "box_conditioning_v2"]


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def git_output(cwd: Path, *args: str) -> str:
    for candidate in GIT_CANDIDATES:
        if not candidate.exists():
            continue
        try:
            return subprocess.check_output(
                [str(candidate), *args],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            continue
    return ""


def parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    normalized = value.replace("Z", "+00:00")
    for candidate in (normalized, normalized.replace("+0800", "+08:00")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    return datetime.min.replace(tzinfo=timezone.utc)


def unique_display_iterations(iterations: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[tuple] = set()
    for item in iterations:
        key = (
            item.get("summary"),
            item.get("commit"),
            tuple(sorted((item.get("artifacts") or {}).items())),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def collect_lessons(iterations: list[dict]) -> list[str]:
    lessons: list[str] = []
    seen: set[str] = set()
    for item in iterations:
        for lesson in item.get("lessons", []):
            if not lesson or lesson in seen:
                continue
            seen.add(lesson)
            lessons.append(lesson)
    return lessons


def build_milestone_table(iterations: list[dict]) -> str:
    rows = []
    for item in iterations:
        completed_str = ", ".join(item.get("completed", []))
        artifacts = item.get("artifacts", {})
        output_str = ", ".join(str(value) for value in artifacts.values()) if artifacts else completed_str[:60]
        rows.append(
            f"| {item['id']} | {item['date']} | {item['summary']} | {output_str} | {item['commit']} |"
        )
    return "\n".join(rows) if rows else "| - | - | No milestones yet | - | - |"


def build_bugs_table(iterations: list[dict]) -> str:
    rows = []
    bug_id = 0
    for item in iterations:
        for bug in item.get("bugs_fixed", []):
            bug_id += 1
            if isinstance(bug, str):
                bug = {"bug": bug, "root_cause": "", "fix": ""}
            rows.append(
                f"| {bug_id} | {item['date']} | {bug.get('bug', '')} | "
                f"{bug.get('root_cause', '')} | {bug.get('fix', '')} | {item['commit']} |"
            )
    return "\n".join(rows) if rows else "| - | - | No bugs recorded | - | - | - |"


def build_artifacts_table(iterations: list[dict]) -> str:
    if not iterations:
        return "| - | - | - | - |"
    latest = iterations[-1]
    rows = []
    for name, path in latest.get("artifacts", {}).items():
        rows.append(f"| {name} | `{path}` | (run verification) | - |")
    return "\n".join(rows) if rows else "| - | - | - | - |"


def build_next_steps(iterations: list[dict], active_campaign: dict | None) -> str:
    if active_campaign and active_campaign.get("next_actions"):
        return "\n".join(f"- {item}" for item in active_campaign["next_actions"])
    if not iterations:
        return "- No planned next steps"
    latest = iterations[-1]
    lines = [f"- {item}" for item in latest.get("next", [])]
    return "\n".join(lines) if lines else "- No planned next steps"


def summarize_queue(queue: list[dict]) -> str:
    parts = [f"{item.get('id')}={item.get('status')}" for item in queue]
    return ", ".join(parts[:8]) + (" ..." if len(parts) > 8 else "")


def campaign_snapshot(root: Path, campaign_name: str) -> dict | None:
    campaign_dir = root / "experiments" / "physnap" / campaign_name
    state = load_json(campaign_dir / "state.json", {})
    review = load_json(campaign_dir / "review.json", {})
    if not state and not review:
        return None
    queue = state.get("queue", [])
    active_runtime = state.get("active_runtime") or {}
    next_incomplete = next((item.get("id") for item in queue if item.get("status") != "completed"), None)
    running = any(item.get("status") == "running" for item in queue) or bool(active_runtime.get("step_id"))
    completed_steps = sum(1 for item in queue if item.get("status") == "completed")
    updated_at = max(
        parse_timestamp(state.get("updated_at")),
        parse_timestamp(review.get("generated_at")),
        parse_timestamp(active_runtime.get("heartbeat_at")),
    )
    return {
        "name": campaign_name,
        "dir": campaign_dir,
        "phase": review.get("phase") or state.get("phase") or "unknown",
        "gate": review.get("phase_gate") or state.get("phase_gate") or "unknown",
        "verdict": review.get("verdict"),
        "decision": review.get("decision"),
        "claim_assessment": review.get("claim_assessment"),
        "next_actions": review.get("next_actions") or [],
        "queue_summary": summarize_queue(queue),
        "next_incomplete": next_incomplete,
        "active_step": active_runtime.get("step_id"),
        "active_worker_pid": active_runtime.get("worker_pid"),
        "running": running,
        "completed_steps": completed_steps,
        "total_steps": len(queue),
        "updated_at": updated_at,
    }


def collect_campaign_snapshots(root: Path) -> list[dict]:
    snapshots = []
    for campaign_name in CAMPAIGN_NAMES:
        snapshot = campaign_snapshot(root, campaign_name)
        if snapshot:
            snapshots.append(snapshot)
    snapshots.sort(key=lambda item: (item["running"], item["updated_at"]), reverse=True)
    return snapshots


def select_active_campaign(campaigns: list[dict]) -> dict | None:
    for campaign in campaigns:
        if campaign["running"]:
            return campaign
    return campaigns[0] if campaigns else None


def campaign_section(campaigns: list[dict]) -> str:
    if not campaigns:
        return ""
    lines = ["", "## Live Campaign Snapshot", ""]
    for campaign in campaigns:
        lines.append(f"### {campaign['name']}")
        lines.append("")
        lines.append(f"- Phase: `{campaign['phase']}`")
        lines.append(f"- Gate: `{campaign['gate']}`")
        lines.append(f"- Verdict: `{campaign.get('verdict')}`")
        lines.append(f"- Decision: `{campaign.get('decision')}`")
        lines.append(f"- Active Step: `{campaign.get('active_step')}`")
        lines.append(f"- Next Incomplete Step: `{campaign.get('next_incomplete')}`")
        lines.append(f"- Queue: {campaign.get('queue_summary')}")
        lines.append(
            f"- Progress: `{campaign.get('completed_steps')}` / `{campaign.get('total_steps')}` steps complete"
        )
        unknown_ts = datetime.min.replace(tzinfo=timezone.utc)
        lines.append(f"- Last Updated: `{campaign.get('updated_at').isoformat() if campaign.get('updated_at') != unknown_ts else 'unknown'}`")
        if campaign.get("claim_assessment"):
            lines.append(f"- Claim Assessment: {campaign['claim_assessment']}")
        lines.append("")
    return "\n".join(lines)


def key_files_section(iterations: list[dict], active_campaign: dict | None, root: Path) -> str:
    lines = []
    latest = iterations[-1] if iterations else {}
    for name, path in (latest.get("artifacts") or {}).items():
        lines.append(f"- `{path}` -- {name}")
    if active_campaign:
        campaign_dir = active_campaign["dir"]
        lines.extend(
            [
                f"- `{(campaign_dir / 'state.json').relative_to(root)}` -- active campaign state",
                f"- `{(campaign_dir / 'review.json').relative_to(root)}` -- active campaign review",
                f"- `{(campaign_dir / 'decision_memo.md').relative_to(root)}` -- active decision memo",
                f"- `{(campaign_dir / 'campaign_status.md').relative_to(root)}` -- active dashboard",
            ]
        )
    return "\n".join(lines) if lines else "- (see evolution.json)"


def sync_checkpoint(root: Path, branch: str, commit_line: str, active_campaign: dict | None, latest: dict):
    session_dir = root / ".session"
    session_dir.mkdir(exist_ok=True)
    cp_path = session_dir / "checkpoint.md"

    lines = [
        "# Session Checkpoint",
        "<!-- Auto-synced from .project-memory/STATUS.md by infinigen-project-memory skill -->",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d')}",
        f"**Branch**: {branch}",
        f"**Last Commit**: {commit_line}",
        "",
        "## Status",
        "",
        "See `.project-memory/STATUS.md` for the full project status.",
        "",
    ]

    if active_campaign:
        lines.extend(
            [
                "## Live Campaign",
                f"- name={active_campaign['name']}",
                f"- phase={active_campaign['phase']}",
                f"- gate={active_campaign['gate']}",
                f"- active_step={active_campaign.get('active_step')}",
                f"- next_incomplete={active_campaign.get('next_incomplete')}",
                "",
            ]
        )

    if latest:
        lines.append("## Completed")
        for item in latest.get("completed", []):
            lines.append(f"- {item}")
        lines.append("")
        lines.append("## Next")
        for item in latest.get("next", []):
            lines.append(f"- {item}")
        lines.append("")
        if latest.get("lessons"):
            lines.append("## Lessons")
            for item in latest["lessons"]:
                lines.append(f"- {item}")
            lines.append("")

    cp_path.write_text("\n".join(lines))
    print(f"checkpoint.md synced: {cp_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Project root directory")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    pm_dir = root / ".project-memory"
    evo_path = pm_dir / "evolution.json"
    if not evo_path.exists():
        print(f"ERROR: {evo_path} not found. Run initial setup first.")
        return 1

    evolution = load_json(evo_path, {})
    iterations = evolution.get("iterations", [])
    display_iterations = unique_display_iterations(iterations)
    campaigns = collect_campaign_snapshots(root)
    active_campaign = select_active_campaign(campaigns)

    branch = git_output(root, "rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    commit_line = git_output(root, "log", "--oneline", "-1") or "unknown"
    commit_hash = commit_line.split()[0] if commit_line else "unknown"
    commit_msg = " ".join(commit_line.split()[1:]) if commit_line else ""

    latest = iterations[-1] if iterations else {}
    current_phase = "Unknown"
    active_work = "Unknown"
    if active_campaign:
        current_phase = f"{active_campaign['name']}::{active_campaign['phase']} gate={active_campaign['gate']}"
        active_work = active_campaign.get("active_step") or active_campaign.get("next_incomplete") or active_work
    elif latest:
        completed = latest.get("completed", [])
        nexts = latest.get("next", [])
        if completed:
            current_phase = completed[-1]
        if nexts:
            active_work = nexts[0]

    template_path = root / ".cursor" / "skills" / "infinigen-project-memory" / "references" / "status-template.md"
    content = template_path.read_text() if template_path.exists() else "# Infinigen-AnyBox Project Status\n"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    content = content.replace("{date}", now)
    content = content.replace("{branch}", branch)
    content = content.replace("{commit_hash}", commit_hash)
    content = content.replace("{commit_message}", commit_msg)
    content = content.replace("{current_phase}", current_phase)
    content = content.replace("{active_work}", active_work)
    content = content.replace("{milestones_rows}", build_milestone_table(display_iterations))
    content = content.replace("{bugs_rows}", build_bugs_table(display_iterations))
    content = content.replace("{artifacts_rows}", build_artifacts_table(display_iterations))
    content = content.replace("{next_steps}", build_next_steps(display_iterations, active_campaign))
    content = content.replace("{key_files}", key_files_section(display_iterations, active_campaign, root))

    lessons = collect_lessons(display_iterations)
    if lessons:
        content += "\n## Lessons Learned (cumulative)\n\n"
        for lesson in lessons:
            content += f"- {lesson}\n"

    content += campaign_section(campaigns)

    status_path = pm_dir / "STATUS.md"
    status_path.write_text(content)
    print(f"STATUS.md updated: {status_path}")

    sync_checkpoint(root, branch, commit_line, active_campaign, latest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
