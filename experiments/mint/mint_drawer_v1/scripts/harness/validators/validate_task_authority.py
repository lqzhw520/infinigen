#!/usr/bin/env python3
"""
validate_task_authority.py - campaign-native

Validates that runtime geometry/contact authority matches the task spec's GOC
authority. Supports both the legacy GOC-v2 count contract and the GOC-v3 exact
geom-id contract.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

CAMPAIGN_ROOT = Path(
    os.environ.get(
        "MINT_TASK_ROOT", str(Path(__file__).resolve().parent.parent.parent.parent)
    )
)
REPO_ROOT = Path(
    os.environ.get("MINT_REPO_ROOT", str(CAMPAIGN_ROOT.parent.parent.parent))
)
YAML_AVAILABLE = False
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    pass

V3_SET_KEYS = (
    "legal_gripper_surface_geom_ids",
    "forbidden_robot_surface_geom_ids",
    "drawer_handle_geom_ids",
)


def die(msg: str, code: int = 1) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(code)


def load_yaml(path: Path) -> dict[str, Any]:
    if not YAML_AVAILABLE:
        try:
            import ruamel.yaml as ruamel_yaml

            with path.open() as fh:
                return dict(ruamel_yaml.YAML().load(fh))
        except ImportError:
            pass
        die(
            f"yaml not available; install pyyaml or ruamel.yaml to parse {path}", code=2
        )
    with path.open() as fh:
        data = yaml.safe_load(fh)
        return dict(data or {})


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def resolve_path(base: Path, rel: str) -> Path:
    if rel.startswith("/"):
        return Path(rel)
    repo_resolved = (REPO_ROOT / rel).resolve()
    if repo_resolved.exists():
        return repo_resolved
    return (base / rel).resolve()


def normalize_ids(values: Any) -> list[int]:
    if values is None:
        return []
    return sorted(int(v) for v in values)


def normalize_exact_sets(data: dict[str, Any]) -> dict[str, list[int]]:
    if "exact_id_sets" in data:
        data = data["exact_id_sets"]
    elif "contact_authority" in data:
        ca = data["contact_authority"]
        allowed = ca.get("allowed_target_contact", {})
        forbidden = ca.get("forbidden_contact", {})
        data = {
            "legal_gripper_surface_geom_ids": allowed.get(
                "legal_gripper_surface_geom_ids", []
            ),
            "forbidden_robot_surface_geom_ids": forbidden.get(
                "forbidden_robot_surface_geom_ids", []
            ),
            "drawer_handle_geom_ids": allowed.get(
                "drawer_handle_geom_ids", forbidden.get("drawer_handle_geom_ids", [])
            ),
        }
    return {key: normalize_ids(data.get(key, [])) for key in V3_SET_KEYS}


def is_contact_report_smoke(data: dict[str, Any]) -> bool:
    markers = (
        "contact_report_schema_version",
        "emitted_exact_sets",
        "observed_target_contact_pairs",
        "observed_forbidden_contact_pairs",
        "raw_runtime_contact_report",
    )
    return any(marker in data for marker in markers)


def normalize_pairs(values: Any) -> list[list[int]]:
    if values is None:
        return []
    pairs: list[list[int]] = []
    for item in values:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            die(
                f"Invalid contact pair, expected [geom_id_a, geom_id_b]: {item}", code=2
            )
        pairs.append([int(item[0]), int(item[1])])
    return pairs


def pair_crosses(pair: list[int], left: set[int], right: set[int]) -> bool:
    a, b = pair
    return (a in left and b in right) or (a in right and b in left)


def validate_v3_contact_report_smoke(
    expected: dict[str, list[int]], report: dict[str, Any]
) -> tuple[bool, str]:
    if report.get("uses_body_based_31_27_authority"):
        return (
            False,
            "AUTHORITY_MISMATCH\nBODY_BASED_31_27_REJECTED: contact report attempted to use body-based authority.",
        )
    if report.get("uses_count_only_authority"):
        return (
            False,
            "AUTHORITY_MISMATCH\nCOUNT_ONLY_AUTHORITY_REJECTED: contact report attempted to use count-only authority.",
        )
    if not report.get("runtime_exact_geom_ids_emitted"):
        return (
            False,
            "AUTHORITY_MISMATCH\nEXACT_GEOM_IDS_REQUIRED: contact report must emit exact geom IDs.",
        )

    emitted_sets = normalize_exact_sets(report.get("emitted_exact_sets", {}))
    mismatches = []
    for key in V3_SET_KEYS:
        if emitted_sets.get(key) != expected.get(key):
            mismatches.append(
                f"{key}: GOC_v3={expected.get(key)} vs contact_report={emitted_sets.get(key)}"
            )
    if mismatches:
        return (
            False,
            "AUTHORITY_MISMATCH\nContact report emitted exact sets do not match GOC-v3 authority.\n"
            + "\n".join(f"  - {m}" for m in mismatches),
        )

    legal = set(expected["legal_gripper_surface_geom_ids"])
    handle = set(expected["drawer_handle_geom_ids"])
    forbidden = set(expected["forbidden_robot_surface_geom_ids"])
    drawer_or_handle = set(report.get("drawer_body_or_cabinet_geom_ids", [])) | handle

    target_pairs = normalize_pairs(report.get("observed_target_contact_pairs", []))
    if not target_pairs:
        return (
            False,
            "AUTHORITY_MISMATCH\nNO_TARGET_EXACT_CONTACT_PAIRS: smoke must observe at least one exact target pair.",
        )
    bad_target = [
        pair for pair in target_pairs if not pair_crosses(pair, legal, handle)
    ]
    if bad_target:
        return (
            False,
            "AUTHORITY_MISMATCH\nTarget contact pairs must be legal_gripper_surface_geom_ids <-> drawer_handle_geom_ids. "
            f"bad_pairs={bad_target}",
        )

    forbidden_pairs = normalize_pairs(
        report.get("observed_forbidden_contact_pairs", [])
    )
    bad_forbidden = [
        pair
        for pair in forbidden_pairs
        if not pair_crosses(pair, forbidden, drawer_or_handle)
    ]
    if bad_forbidden:
        return (
            False,
            "AUTHORITY_MISMATCH\nForbidden contact pairs must be forbidden_robot_surface_geom_ids <-> drawer/handle exact IDs. "
            f"bad_pairs={bad_forbidden}",
        )

    raw_pairs = normalize_pairs(report.get("observed_all_contact_pairs", []))
    non_exact = [pair for pair in raw_pairs if any(gid < 0 for gid in pair)]
    if non_exact:
        return (
            False,
            f"AUTHORITY_MISMATCH\nNon-exact contact pair IDs observed: {non_exact}",
        )

    return (
        True,
        "PASS: GOC-v3 contact report smoke exact-ID authority match. "
        f"target_pairs={target_pairs} forbidden_pairs={forbidden_pairs} "
        f"legacy_body_based_fields_present={bool(report.get('legacy_body_based_fields_observed_not_used'))}.",
    )


def load_goc_v3_contract(task_auth: dict[str, Any]) -> dict[str, Any]:
    rel = task_auth.get("goc_artifact_path") or task_auth.get("goc_v3_contract_path")
    if not rel:
        die("GOC-v3 task_authority requires goc_artifact_path", code=2)
    path = resolve_path(CAMPAIGN_ROOT, str(rel))
    if not path.exists():
        die(f"GOC-v3 contract not found: {path}", code=2)
    contract = load_json(path)
    if contract.get("contract_id") != "GOC_V3_EXACT_ID_GEOMETRY_OWNERSHIP_CONTRACT":
        die(
            f"Unexpected GOC-v3 contract_id in {path}: {contract.get('contract_id')}",
            code=2,
        )
    return contract


def v2_runtime_counts_from_fixture(fixture: str) -> dict[str, int]:
    if fixture == "invalid_authority_drift":
        return {"legal_pad": 31, "forbidden": 27, "handle": 9}
    if fixture == "valid_authority_match":
        return {"legal_pad": 29, "forbidden": 26, "handle": 9}
    die(
        f"Unknown fixture: {fixture}. Available: invalid_authority_drift, valid_authority_match",
        code=2,
    )


def load_runtime_authority(
    args: argparse.Namespace, task_spec: dict[str, Any]
) -> dict[str, Any] | None:
    task_auth = task_spec.get("task_authority", {})
    is_v3 = task_auth.get("goc_version") == "v3"

    if args.fixture:
        fixture_data = task_spec.get("fixtures", {}).get(f"fixture_{args.fixture}", {})
        if is_v3:
            if "runtime_exact_ids" in fixture_data:
                return {
                    "mode": "exact_sets",
                    "sets": normalize_exact_sets(fixture_data["runtime_exact_ids"]),
                }
            if args.fixture == "valid_authority_match":
                return {
                    "mode": "exact_sets",
                    "sets": normalize_exact_sets(load_goc_v3_contract(task_auth)),
                }
            die(f"GOC-v3 fixture lacks runtime_exact_ids: {args.fixture}", code=2)
        return {
            "mode": "counts",
            "counts": v2_runtime_counts_from_fixture(args.fixture),
        }

    if args.runtime_json:
        path = resolve_path(CAMPAIGN_ROOT, args.runtime_json)
        data = load_json(path)
        if is_v3:
            if is_contact_report_smoke(data):
                return {"mode": "contact_report_smoke", "report": data}
            if (
                any(key in data for key in V3_SET_KEYS)
                or "exact_id_sets" in data
                or "contact_authority" in data
            ):
                return {"mode": "exact_sets", "sets": normalize_exact_sets(data)}
            return {
                "mode": "counts_only",
                "counts": {
                    "legal_pad": data.get(
                        "legal_pad", data.get("gripper_contact_geom_count", 0)
                    ),
                    "forbidden": data.get(
                        "forbidden", data.get("forbidden_robot_geom_count", 0)
                    ),
                    "handle": data.get("handle", data.get("handle_geom_count", 0)),
                },
            }
        return {
            "mode": "counts",
            "counts": {
                "legal_pad": data.get(
                    "legal_pad", data.get("gripper_contact_geom_count", 0)
                ),
                "forbidden": data.get(
                    "forbidden", data.get("forbidden_robot_geom_count", 0)
                ),
                "handle": data.get("handle", data.get("handle_geom_count", 0)),
            },
        }

    if args.runtime_counts:
        parts = args.runtime_counts.split(",")
        counts: dict[str, int] = {}
        for part in parts:
            if ":" not in part:
                die(
                    f"Invalid --runtime-counts format: {part}. Expected 'l:<int>,f:<int>,h:<int>'",
                    code=2,
                )
            key, val = part.split(":", 1)
            counts[key.strip()] = int(val.strip())
        if is_v3:
            return {"mode": "counts_only", "counts": counts}
        return {"mode": "counts", "counts": counts}

    return None


def validate_v2_authority(
    task_spec: dict[str, Any], runtime: dict[str, Any] | None, dry_run: bool
) -> tuple[bool, str]:
    task_auth = task_spec.get("task_authority", {})
    artifact_counts = {
        "legal_pad": task_auth.get("legal_pad_count", 0),
        "forbidden": task_auth.get("forbidden_count", 0),
        "handle": task_auth.get("handle_count", 0),
    }
    goc_version = task_auth.get("goc_version", "unknown")
    goc_artifact_commit = task_auth.get("goc_artifact_commit", "unknown")

    if dry_run:
        return (
            True,
            f"[DRY-RUN] task_authority={artifact_counts} goc_version={goc_version} "
            f"goc_artifact_commit={goc_artifact_commit}. "
            "Would validate against runtime counts if provided.",
        )

    if runtime is None:
        return (
            True,
            f"task_authority={artifact_counts} goc_version={goc_version}. "
            "No runtime counts provided. Validator will run again during execution.",
        )

    if runtime.get("mode") != "counts":
        return False, "AUTHORITY_MISMATCH\nExpected GOC-v2 runtime count authority."
    runtime_counts = runtime["counts"]
    mismatches = []
    for key in ("legal_pad", "forbidden", "handle"):
        if artifact_counts[key] != runtime_counts[key]:
            mismatches.append(
                f"{key}: GOC_artifact={artifact_counts[key]} vs runtime={runtime_counts[key]}"
            )

    if mismatches:
        report = [
            "AUTHORITY_MISMATCH",
            "=" * 60,
            "ERROR: Runtime geometry classification disagrees with GOC artifact authority.",
            f"GOC version:      {goc_version}",
            f"GOC artifact:     {task_auth.get('goc_artifact_path', 'unknown')}",
            f"GOC commit:       {goc_artifact_commit}",
            "",
            "Mismatches detected:",
        ]
        report.extend(f"  - {m}" for m in mismatches)
        report.extend(
            [
                "",
                "CRITICAL: The agent MUST NOT continue if runtime and GOC disagree.",
                "Next gate: GOC_V3_EXACT_ID_CONTRACT_REBUILD",
            ]
        )
        return False, "\n".join(report)

    return (
        True,
        f"PASS: Authority match. GOC={artifact_counts} runtime={runtime_counts}",
    )


def validate_v3_authority(
    task_spec: dict[str, Any], runtime: dict[str, Any] | None, dry_run: bool
) -> tuple[bool, str]:
    task_auth = task_spec.get("task_authority", {})
    contract = load_goc_v3_contract(task_auth)
    expected = normalize_exact_sets(contract)
    task_sets = normalize_exact_sets(task_auth)
    if task_sets != expected:
        return (
            False,
            "AUTHORITY_MISMATCH\nTask spec exact sets do not match the GOC-v3 contract."
            f"\nexpected={expected}\ntask_spec={task_sets}",
        )
    if (
        contract.get("invariant_results", {})
        .get("summary_counts", {})
        .get("unknown_contact_relevant_geom_count")
        != 0
    ):
        return False, "AUTHORITY_MISMATCH\nGOC-v3 has unknown contact-relevant geoms."

    if dry_run:
        return (
            True,
            "[DRY-RUN] GOC-v3 exact-ID authority loaded. "
            f"legal={expected['legal_gripper_surface_geom_ids']} "
            f"forbidden={expected['forbidden_robot_surface_geom_ids']} "
            f"handle={expected['drawer_handle_geom_ids']}. "
            "Would compare runtime exact geom IDs if provided.",
        )

    if runtime is None:
        return (
            True,
            "GOC-v3 exact-ID authority initialized. No runtime exact IDs provided. "
            "Validator will run again during contact-report smoke.",
        )

    if runtime.get("mode") == "contact_report_smoke":
        return validate_v3_contact_report_smoke(expected, runtime["report"])

    if runtime.get("mode") == "counts_only":
        return (
            False,
            "AUTHORITY_MISMATCH\nCOUNT_ONLY_AUTHORITY_REJECTED: GOC-v3 requires exact geom-id sets; "
            f"received counts_only={runtime.get('counts')}",
        )
    if runtime.get("mode") != "exact_sets":
        return False, "AUTHORITY_MISMATCH\nGOC-v3 requires runtime exact geom-id sets."

    actual = runtime["sets"]
    mismatches = []
    for key in V3_SET_KEYS:
        if normalize_ids(expected[key]) != normalize_ids(actual.get(key, [])):
            mismatches.append(
                f"{key}: GOC_v3={expected[key]} vs runtime={actual.get(key, [])}"
            )

    if mismatches:
        report = [
            "AUTHORITY_MISMATCH",
            "=" * 60,
            "ERROR: Runtime/contact-report exact geom IDs disagree with GOC-v3 authority.",
            f"GOC artifact: {task_auth.get('goc_artifact_path', 'unknown')}",
            f"GOC commit:   {task_auth.get('goc_artifact_commit', 'unknown')}",
            "",
            "Mismatches detected:",
        ]
        report.extend(f"  - {m}" for m in mismatches)
        report.extend(
            [
                "",
                "CRITICAL: Body-based 31/27 and GOC-v2 count-only authority are prohibited.",
                "Next gate: V11_G4_GOC_V3_CONTACT_REPORT_SMOKE",
            ]
        )
        return False, "\n".join(report)

    return True, f"PASS: GOC-v3 exact authority match. exact_sets={actual}"


def validate_authority(
    task_spec: dict[str, Any], runtime: dict[str, Any] | None, dry_run: bool = False
) -> tuple[bool, str]:
    task_auth = task_spec.get("task_authority", {})
    if task_auth.get("goc_version") == "v3":
        return validate_v3_authority(task_spec, runtime, dry_run)
    return validate_v2_authority(task_spec, runtime, dry_run)


def run_fixture_test(fixture_name: str, task_spec: dict[str, Any]) -> tuple[bool, str]:
    runtime = load_runtime_authority(
        argparse.Namespace(
            fixture=fixture_name, runtime_json=None, runtime_counts=None
        ),
        task_spec,
    )
    passed, msg = validate_authority(task_spec, runtime, dry_run=False)

    fixture_data = task_spec.get("fixtures", {}).get(f"fixture_{fixture_name}", {})
    expected = fixture_data.get("expected", "PASS")

    if expected == "PASS" and passed:
        return True, f"fixture={fixture_name} expected=PASS actual=PASS - OK"
    if expected == "FAIL" and not passed:
        expected_error = fixture_data.get("expected_error", "FAIL")
        if expected_error in msg:
            return (
                True,
                f"fixture={fixture_name} expected=FAIL({expected_error}) actual=FAIL - validator correctly rejected drift",
            )
        return (
            False,
            f"fixture={fixture_name} expected=FAIL({expected_error}) but FAIL reason differs",
        )
    if expected == "PASS" and not passed:
        return (
            False,
            f"fixture={fixture_name} expected=PASS actual=FAIL - validator broke",
        )
    if expected == "FAIL" and passed:
        return (
            False,
            f"fixture={fixture_name} expected=FAIL actual=PASS - validator should have caught mismatch",
        )
    return False, f"fixture={fixture_name} unknown state"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate runtime geometry/contact authority against task GOC authority.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "task_spec",
        nargs="?",
        default="sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml",
        help="Path to task YAML spec (relative to CAMPAIGN_ROOT/repo root or absolute)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fixture", help="Run a named test fixture instead of real runtime data"
    )
    parser.add_argument("--runtime-json", metavar="PATH")
    parser.add_argument("--runtime-counts", metavar="l:<int>,f:<int>,h:<int>")

    args = parser.parse_args()

    task_spec_path = resolve_path(CAMPAIGN_ROOT, args.task_spec)
    if not task_spec_path.exists():
        die(f"Task spec not found: {task_spec_path}", code=2)

    task_spec = load_yaml(task_spec_path)
    if not task_spec.get("task_authority"):
        die("task_authority block not found in task spec", code=2)

    if args.fixture:
        passed, msg = run_fixture_test(args.fixture, task_spec)
        print(msg)
        if not passed:
            print("FAIL", file=sys.stderr)
            sys.exit(1)
        print("PASS")
        sys.exit(0)

    runtime = load_runtime_authority(args, task_spec)
    passed, msg = validate_authority(task_spec, runtime, dry_run=args.dry_run)
    print(msg)
    if not passed:
        print("\nVALIDATION FAILED", file=sys.stderr)
        print("RESULT: AUTHORITY_MISMATCH", file=sys.stderr)
        sys.exit(1)
    print("\nVALIDATION PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
