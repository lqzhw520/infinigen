#!/usr/bin/env python3
"""Archive the current iteration: snapshot task_plan + findings into history/.

After archiving, optionally resets task_plan.md and findings.md to blank state
(--clean flag, also triggered automatically by post-commit hook).

Usage:
    python .cursor/skills/infinigen-project-memory/scripts/archive_iteration.py \
        --title "Phase-1 1K dataset" [--project-root .] [--clean]
"""
import argparse
import json
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True, help="Iteration title for the archive filename")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--clean", action="store_true",
                        help="Reset task_plan.md and findings.md after archiving")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    pm_dir = root / ".project-memory"
    history_dir = pm_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = args.title.lower().replace(" ", "-").replace("/", "-")[:50]
    archive_name = f"{date_str}_{slug}.md"
    archive_path = history_dir / archive_name

    sections = []
    sections.append(f"# Iteration Archive: {args.title}")
    sections.append(f"\n**Archived**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    task_plan = root / "task_plan.md"
    if task_plan.exists():
        sections.append("## Task Plan\n")
        sections.append(task_plan.read_text())
        sections.append("")

    findings = root / "findings.md"
    if findings.exists():
        sections.append("## Findings\n")
        sections.append(findings.read_text())
        sections.append("")

    checkpoint = root / ".session" / "checkpoint.md"
    if checkpoint.exists():
        sections.append("## Checkpoint Snapshot\n")
        sections.append(checkpoint.read_text())
        sections.append("")

    status = pm_dir / "STATUS.md"
    if status.exists():
        sections.append("## STATUS.md Snapshot\n")
        sections.append(status.read_text())
        sections.append("")

    archive_path.write_text("\n".join(sections))
    print(f"Archived iteration to: {archive_path}")

    if args.clean:
        if task_plan.exists():
            task_plan.write_text(
                "# Task Plan\n\n"
                f"**Status**: AWAITING NEW TASK\n"
                f"**Last Updated**: {date_str}\n"
            )
            print(f"Reset: {task_plan}")
        if findings.exists():
            findings.write_text(
                "# Findings\n\n"
                "(No active findings)\n"
            )
            print(f"Reset: {findings}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
