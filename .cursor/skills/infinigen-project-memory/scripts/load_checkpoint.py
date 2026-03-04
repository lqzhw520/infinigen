#!/usr/bin/env python3
"""Load the last checkpoint from evolution.json.

Usage as CLI:
    python .cursor/skills/infinigen-project-memory/scripts/load_checkpoint.py [--project-root .]

Usage as library:
    from scripts.load_checkpoint import load_last_checkpoint
    cp = load_last_checkpoint("/path/to/project")
"""
import argparse
import json
from pathlib import Path


def load_last_checkpoint(project_root: str = ".") -> dict | None:
    """Return the most recent iteration record, or None if no iterations exist."""
    evo_path = Path(project_root) / ".project-memory" / "evolution.json"
    if not evo_path.exists():
        return None
    evo = json.loads(evo_path.read_text())
    iterations = evo.get("iterations", [])
    return iterations[-1] if iterations else None


def load_all_iterations(project_root: str = ".") -> list:
    """Return all iteration records."""
    evo_path = Path(project_root) / ".project-memory" / "evolution.json"
    if not evo_path.exists():
        return []
    evo = json.loads(evo_path.read_text())
    return evo.get("iterations", [])


def load_recent_commits(project_root: str = ".", n: int = 10) -> list:
    """Return the last n commits logged by the post-commit hook."""
    evo_path = Path(project_root) / ".project-memory" / "evolution.json"
    if not evo_path.exists():
        return []
    evo = json.loads(evo_path.read_text())
    return evo.get("commits", [])[-n:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--all", action="store_true", help="Show all iterations")
    args = parser.parse_args()

    if args.all:
        iterations = load_all_iterations(args.project_root)
        for it in iterations:
            print(f"[{it['id']}] {it['date']} - {it['summary']} (commit: {it['commit']})")
    else:
        cp = load_last_checkpoint(args.project_root)
        if cp is None:
            print("No checkpoint found.")
            return 1
        print(json.dumps(cp, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
