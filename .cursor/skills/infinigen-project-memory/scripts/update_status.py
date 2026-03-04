#!/usr/bin/env python3
"""Regenerate .project-memory/STATUS.md from evolution.json + git state.

Usage:
    python .cursor/skills/infinigen-project-memory/scripts/update_status.py [--project-root .]
"""
import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


def _git(cmd: str, cwd: Path) -> str:
    try:
        return subprocess.check_output(
            cmd, shell=True, cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def _build_milestone_table(iterations: list) -> str:
    rows = []
    for it in iterations:
        completed_str = ", ".join(it.get("completed", []))
        artifacts = it.get("artifacts", {})
        output_str = ", ".join(f"{v}" for v in artifacts.values()) if artifacts else completed_str[:60]
        rows.append(
            f"| {it['id']} | {it['date']} | {it['summary']} | {output_str} | {it['commit']} |"
        )
    return "\n".join(rows) if rows else "| - | - | No milestones yet | - | - |"


def _build_bugs_table(iterations: list) -> str:
    rows = []
    bug_id = 0
    for it in iterations:
        for bug in it.get("bugs_fixed", []):
            bug_id += 1
            rows.append(
                f"| {bug_id} | {it['date']} | {bug.get('bug','')} | "
                f"{bug.get('root_cause','')} | {bug.get('fix','')} | {it['commit']} |"
            )
    return "\n".join(rows) if rows else "| - | - | No bugs recorded | - | - | - |"


def _build_artifacts_table(iterations: list) -> str:
    if not iterations:
        return "| - | - | - | - |"
    latest = iterations[-1]
    rows = []
    for name, path in latest.get("artifacts", {}).items():
        rows.append(f"| {name} | `{path}` | (run verification) | - |")
    return "\n".join(rows) if rows else "| - | - | - | - |"


def _build_next_steps(iterations: list) -> str:
    if not iterations:
        return "- No planned next steps"
    latest = iterations[-1]
    lines = []
    for item in latest.get("next", []):
        lines.append(f"- {item}")
    return "\n".join(lines) if lines else "- No planned next steps"


def _collect_lessons(iterations: list) -> list:
    all_lessons = []
    for it in iterations:
        all_lessons.extend(it.get("lessons", []))
    return all_lessons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Project root directory")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    pm_dir = root / ".project-memory"
    evo_path = pm_dir / "evolution.json"

    if not evo_path.exists():
        print(f"ERROR: {evo_path} not found. Run initial setup first.")
        return 1

    evo = json.loads(evo_path.read_text())
    iterations = evo.get("iterations", [])

    branch = _git("git rev-parse --abbrev-ref HEAD", root) or "unknown"
    commit_line = _git("git log --oneline -1", root) or "unknown"
    commit_hash = commit_line.split()[0] if commit_line else "unknown"
    commit_msg = " ".join(commit_line.split()[1:]) if commit_line else ""

    latest = iterations[-1] if iterations else {}
    current_phase = "Unknown"
    active_work = "Unknown"
    if latest:
        completed = latest.get("completed", [])
        nexts = latest.get("next", [])
        if completed:
            current_phase = completed[-1]
        if nexts:
            active_work = nexts[0]

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    template = root / ".cursor/skills/infinigen-project-memory/references/status-template.md"
    if template.exists():
        content = template.read_text()
    else:
        content = "# Infinigen-AnyBox Project Status\n"

    content = content.replace("{date}", now)
    content = content.replace("{branch}", branch)
    content = content.replace("{commit_hash}", commit_hash)
    content = content.replace("{commit_message}", commit_msg)
    content = content.replace("{current_phase}", current_phase)
    content = content.replace("{active_work}", active_work)
    content = content.replace("{milestones_rows}", _build_milestone_table(iterations))
    content = content.replace("{bugs_rows}", _build_bugs_table(iterations))
    content = content.replace("{artifacts_rows}", _build_artifacts_table(iterations))
    content = content.replace("{next_steps}", _build_next_steps(iterations))

    key_files_section = ""
    if latest and latest.get("artifacts"):
        for name, path in latest["artifacts"].items():
            key_files_section += f"- `{path}` -- {name}\n"
    lessons = _collect_lessons(iterations)
    if lessons:
        content += "\n## Lessons Learned (cumulative)\n\n"
        for lesson in lessons:
            content += f"- {lesson}\n"

    content = content.replace("{key_files}", key_files_section or "- (see evolution.json)")

    status_path = pm_dir / "STATUS.md"
    status_path.write_text(content)
    print(f"STATUS.md updated: {status_path}")

    sync_checkpoint(root, content, branch, commit_line, latest)
    return 0


def sync_checkpoint(root: Path, status_content: str, branch: str, commit_line: str, latest: dict):
    """Keep .session/checkpoint.md in sync as a lightweight view."""
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


if __name__ == "__main__":
    raise SystemExit(main())
