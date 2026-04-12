#!/usr/bin/env python3
"""CLI entrypoint for the unified visual-fidelity root-cause controller."""

from __future__ import annotations

import argparse
import json
import os

from root_cause_contracts import CLAIM_BOUNDARY_PATH, load_claim_boundary
from root_cause_controller import run_controller
from root_cause_registry import archive_controller_state

ENV_BOOL_TRUE = {"1", "true", "yes", "on"}
ENV_BOOL_FALSE = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    normalized = value.strip().lower()
    if normalized in ENV_BOOL_TRUE:
        return True
    if normalized in ENV_BOOL_FALSE:
        return False
    return bool(default)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return int(default)
    return int(value)


def _controller_defaults() -> dict[str, object]:
    boundary = load_claim_boundary(CLAIM_BOUNDARY_PATH)
    defaults = boundary.get("controller_defaults") or {}
    return {
        "auto_promote_sovereign": bool(defaults.get("auto_promote_sovereign", False)),
        "allow_full_retrain": bool(defaults.get("allow_full_retrain", False)),
        "allow_new_claim": bool(defaults.get("allow_new_claim", False)),
        "run_mode": str(
            defaults.get(
                "default_run_mode",
                "autonomous_cycle_phase" if defaults.get("autonomous_cycle_phase", True) else "manual_build_phase",
            )
        ),
    }


def parse_args() -> argparse.Namespace:
    defaults = _controller_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-family", choices=["VR", "RCA"], default=os.getenv("EXPERIMENT_FAMILY", "RCA"))
    parser.add_argument("--cap-strongest-negative", dest="cap_strongest_negative", action="store_true", default=_env_bool("CAP_STRONGEST_NEGATIVE", True))
    parser.add_argument("--no-cap-strongest-negative", dest="cap_strongest_negative", action="store_false")
    parser.add_argument("--max-cycles", type=int, default=_env_int("MAX_CYCLES", 4))
    parser.add_argument("--max-experiments-per-cycle", type=int, default=_env_int("MAX_EXPERIMENTS_PER_CYCLE", 3))
    parser.add_argument("--sleep-seconds", type=int, default=_env_int("SLEEP_SECONDS", 30))
    parser.add_argument("--max-rollouts-per-experiment", type=int, default=_env_int("MAX_ROLLOUTS_PER_EXPERIMENT", 3))
    parser.add_argument("--max-disk-growth-mb", type=int, default=_env_int("MAX_DISK_GROWTH_MB", 4096))
    parser.add_argument("--retry-backoff-seconds", type=int, default=_env_int("RETRY_BACKOFF_SECONDS", 5))
    parser.add_argument("--max-retries", type=int, default=_env_int("MAX_RETRIES", 2))
    parser.add_argument("--max-cycles-per-run", type=int, default=_env_int("MAX_CYCLES_PER_RUN", 4))
    parser.add_argument("--max-tiny-retrain-budget", type=int, default=_env_int("MAX_TINY_RETRAIN_BUDGET", 1))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cycle-mode", choices=["dry_run", "preflight", "unattended_cycle", "soak_rehearsal"], default=None)
    parser.add_argument("--auto-promote-sovereign", dest="auto_promote_sovereign", action="store_true", default=_env_bool("AUTO_PROMOTE_SOVEREIGN", bool(defaults["auto_promote_sovereign"])))
    parser.add_argument("--no-auto-promote-sovereign", dest="auto_promote_sovereign", action="store_false")
    parser.add_argument("--allow-full-retrain", dest="allow_full_retrain", action="store_true", default=_env_bool("ALLOW_FULL_RETRAIN", bool(defaults["allow_full_retrain"])))
    parser.add_argument("--disallow-full-retrain", dest="allow_full_retrain", action="store_false")
    parser.add_argument("--allow-new-claim", dest="allow_new_claim", action="store_true", default=_env_bool("ALLOW_NEW_CLAIM", bool(defaults["allow_new_claim"])))
    parser.add_argument("--disallow-new-claim", dest="allow_new_claim", action="store_false")
    parser.add_argument("--run-mode", choices=["manual_build_phase", "autonomous_cycle_phase"], default=os.getenv("RUN_MODE", str(defaults["run_mode"])))
    return parser.parse_args()


def _resolve_cycle_mode(args: argparse.Namespace) -> str:
    if args.cycle_mode:
        return args.cycle_mode
    if args.dry_run:
        return "dry_run"
    if args.run_mode == "manual_build_phase":
        return "preflight"
    return "unattended_cycle"


def main() -> int:
    args = parse_args()
    reset_archive = archive_controller_state(reason="cli_reset_state") if args.reset_state else None
    cycle_mode = _resolve_cycle_mode(args)
    payload = run_controller(
        max_cycles=args.max_cycles,
        max_experiments_per_cycle=args.max_experiments_per_cycle,
        sleep_seconds=args.sleep_seconds,
        dry_run=args.dry_run,
        auto_promote_sovereign=args.auto_promote_sovereign,
        allow_full_retrain=args.allow_full_retrain,
        allow_new_claim=args.allow_new_claim,
        run_mode=args.run_mode,
        cycle_mode=cycle_mode,
        experiment_family=args.experiment_family,
        cap_strongest_negative=args.cap_strongest_negative,
        resume=args.resume,
        max_rollouts_per_experiment=args.max_rollouts_per_experiment,
        max_disk_growth_mb=args.max_disk_growth_mb,
        retry_backoff_seconds=args.retry_backoff_seconds,
        max_retries=args.max_retries,
        max_cycles_per_run=args.max_cycles_per_run,
        max_tiny_retrain_budget=args.max_tiny_retrain_budget,
    )
    if reset_archive is not None:
        payload["reset_archive"] = reset_archive
    payload["cycle_mode"] = cycle_mode
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
