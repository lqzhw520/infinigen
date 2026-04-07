#!/usr/bin/env python3
"""
Harness v2 night runner.

Collector/publisher split:
  - The collector command is defined by experiment spec
  - Publishing is delegated to sovereign_cli.py publish-experiment
  - No free-form sovereign verdict edits happen here
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN_ROOT = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1"
SOVEREIGN = CAMPAIGN_ROOT / "sovereign"
SPECS_DIR = SOVEREIGN / "experiment_specs"
CURRENT_TRUTH = SOVEREIGN / "current_truth.json"
NEXT_ACTIONS = SOVEREIGN / "next_actions.json"
NIGHT_DIR = SOVEREIGN / "night"
CLI = CAMPAIGN_ROOT / "scripts" / "harness" / "sovereign_cli.py"


def now_ts() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_specs() -> dict[str, dict]:
    specs = {}
    for path in sorted(SPECS_DIR.glob("*.yaml")):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        specs[data.get("spec_id", path.stem)] = data
    return specs


def shell(command: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-lc", command], capture_output=True, text=True)


def canonical_next_action(next_actions: dict) -> dict | None:
    actions = next_actions.get("actions", [])
    in_progress = [item for item in actions if item.get("status") == "in_progress"]
    pending = [item for item in actions if item.get("status") == "pending"]
    if in_progress:
        return in_progress[0]
    if pending:
        return pending[0]
    return None


def find_runnable_spec(next_action: dict | None, specs: dict[str, dict]) -> tuple[str | None, dict | None]:
    if not next_action:
        return None, None
    action_type = next_action.get("type")
    action_id = next_action.get("id")
    for spec_id, spec in specs.items():
        bindings = spec.get("bindings", {})
        if action_type in bindings.get("action_types", []) or action_id in bindings.get("action_ids", []):
            collector = spec.get("collector", {})
            if collector.get("command") and collector.get("artifact"):
                return spec_id, spec
    return None, None


def render_status() -> dict:
    truth = load_json(CURRENT_TRUTH) if CURRENT_TRUTH.exists() else {}
    next_actions = load_json(NEXT_ACTIONS) if NEXT_ACTIONS.exists() else {}
    specs = load_specs()
    next_action = canonical_next_action(next_actions)
    spec_id, spec = find_runnable_spec(next_action, specs)
    return {
        "generated_at": now_ts(),
        "phase": truth.get("current", {}).get("phase"),
        "verdict": truth.get("current", {}).get("verdict"),
        "next_action": next_action,
        "runnable_spec_id": spec_id,
        "auto_publish": bool(spec and spec.get("auto_publish")),
    }


def write_report(data: dict) -> Path:
    NIGHT_DIR.mkdir(parents=True, exist_ok=True)
    path = NIGHT_DIR / "v59_night_runner_last.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Harness v2 night runner")
    parser.add_argument("--status", action="store_true", help="Show current runnable nightly action")
    parser.add_argument("--dry-run", action="store_true", help="Show collector/publisher plan without executing")
    parser.add_argument("--run", action="store_true", help="Run current spec-backed collector and publisher")
    parser.add_argument("--collector-only", action="store_true", help="Run collector only, never publish")
    parser.add_argument("--max-hours", type=float, default=8.0)
    args = parser.parse_args()

    if not CURRENT_TRUTH.exists():
        print("current_truth.json missing. Run: python scripts/harness/sovereign_cli.py go", file=sys.stderr)
        sys.exit(1)

    status = render_status()
    if args.status or args.dry_run:
        print(json.dumps(status, indent=2, ensure_ascii=False))
        if args.status:
            return

    next_action = status["next_action"]
    spec_id = status["runnable_spec_id"]
    specs = load_specs()
    spec = specs.get(spec_id) if spec_id else None

    if not spec:
        report = {
            "generated_at": now_ts(),
            "status": "no_runnable_spec",
            "message": "Current canonical next action has no auto-runnable collector spec.",
            "next_action": next_action,
        }
        path = write_report(report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"report: {path}")
        if args.run:
            sys.exit(2)
        return

    collector = spec["collector"]
    command = collector["command"].format(max_hours=args.max_hours)
    artifact = collector["artifact"]
    auto_publish = bool(spec.get("auto_publish")) and not args.collector_only

    if args.dry_run and not args.run:
        print(json.dumps({
            "next_action": next_action,
            "spec_id": spec_id,
            "collector_command": command,
            "artifact": artifact,
            "auto_publish": auto_publish,
        }, indent=2, ensure_ascii=False))
        return

    if not args.run:
        parser.error("Specify one of --status, --dry-run, or --run")

    collector_result = shell(command)
    report = {
        "generated_at": now_ts(),
        "status": "collector_complete" if collector_result.returncode == 0 else "collector_failed",
        "spec_id": spec_id,
        "collector_command": command,
        "collector_returncode": collector_result.returncode,
        "collector_stdout": collector_result.stdout[-4000:],
        "collector_stderr": collector_result.stderr[-4000:],
        "artifact": artifact,
        "auto_publish_attempted": False,
    }

    if collector_result.returncode != 0:
        path = write_report(report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"report: {path}")
        sys.exit(collector_result.returncode)

    if auto_publish:
        publish_cmd = (
            f"cd {PROJECT_ROOT} && python {CLI} publish-experiment "
            f"--spec {spec_id} --from {artifact}"
        )
        publish_result = shell(publish_cmd)
        report["auto_publish_attempted"] = True
        report["publish_returncode"] = publish_result.returncode
        report["publish_stdout"] = publish_result.stdout[-4000:]
        report["publish_stderr"] = publish_result.stderr[-4000:]
        if publish_result.returncode != 0:
            report["status"] = "publisher_failed"
            path = write_report(report)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            print(f"report: {path}")
            sys.exit(publish_result.returncode)

    path = write_report(report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report: {path}")


if __name__ == "__main__":
    main()
