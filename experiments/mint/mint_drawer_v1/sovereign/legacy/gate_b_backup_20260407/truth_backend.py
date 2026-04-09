#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

CAMPAIGN_ROOT = Path(__file__).parent.parent.parent.resolve()
PROJECT_ROOT = CAMPAIGN_ROOT.parents[2]
SOVEREIGN = CAMPAIGN_ROOT / "sovereign"
ARTIFACTS = CAMPAIGN_ROOT / "artifacts"
EVIDENCE_DIR = SOVEREIGN / "evidence"
EXPERIMENT_SPECS_DIR = SOVEREIGN / "experiment_specs"
EXTERNAL_MINT = PROJECT_ROOT / "external" / "MINT"
LEGACY_DIR = SOVEREIGN / "legacy"

CURRENT_TRUTH_PATH = SOVEREIGN / "current_truth.json"
WORKSPACE_MANIFEST_PATH = SOVEREIGN / "workspace_manifest.json"
MODEL_LOAD_FIDELITY_PATH = SOVEREIGN / "model_load_fidelity.json"
RUN_LEDGER_PATH = SOVEREIGN / "run_ledger.yaml"
STATE_PATH = SOVEREIGN / "state.json"
NEXT_ACTIONS_PATH = SOVEREIGN / "next_actions.json"
CLAIMS_PATH = SOVEREIGN / "claims.yaml"
EVIDENCE_INDEX_PATH = EVIDENCE_DIR / "index.json"
DATASET_MANIFEST_PATH = ARTIFACTS / "current_dataset_manifest.json"
USAGE_GUIDE_PATH = CAMPAIGN_ROOT / "HARNESS_USAGE_GUIDE.md"
HANDOFF_PATH = SOVEREIGN / "handoff.md"
CAMPAIGN_TRUTH_PATH = SOVEREIGN / "CAMPAIGN_TRUTH.generated.md"

GENERATED_MARKER = "GENERATED FROM current_truth.json — NOT CANONICAL"
CANONICAL_TIER0 = [
    "sovereign/claims.yaml",
    "sovereign/evidence/index.json",
    "sovereign/evidence/*.yaml",
    "sovereign/next_actions.json",
    "artifacts/current_dataset_manifest.json",
    "sovereign/workspace_manifest.json",
    "sovereign/model_load_fidelity.json",
    "sovereign/current_truth.json",
]

CLAIM_ALLOWED_TOP_LEVEL_KEYS = {
    "claim_id",
    "current_revision",
    "lifecycle_status",
    "superseded_by",
    "superseded_by_reason",
    "supersedes",
    "split_from",
    "split_into",
    "revisions",
}

CURRENT_DRIVING_BY_ACTION_ID = {
    "gate_b_teacher_replayability": {
        "C_V59_OFFLINE_OVERFIT_SIGNAL",
        "C_V59_ENV_GATE_FAILED",
        "C_RCA1_TEACHER_NOT_ROOT_CAUSE",
        "C_V59_ACTION_NORMALIZATION_NOT_ROOT_CAUSE",
        "C_V59_STATE_REPRESENTATION_NOT_ROOT_CAUSE",
    },
}

CURRENT_DRIVING_BY_PHASE = {
    "v59_LEARNABILITY_AUDIT_GATE_B": {
        "C_V59_OFFLINE_OVERFIT_SIGNAL",
        "C_V59_ENV_GATE_FAILED",
        "C_RCA1_TEACHER_NOT_ROOT_CAUSE",
        "C_V59_ACTION_NORMALIZATION_NOT_ROOT_CAUSE",
        "C_V59_STATE_REPRESENTATION_NOT_ROOT_CAUSE",
    },
}

LIFECYCLE_REVIEW_PRIORITY_ORDER = {
    "P1": 1,
    "P2": 2,
    "P3": 3,
}


def now_ts() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return deepcopy(default)
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_yaml(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return deepcopy(default)
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    os.replace(tmp, path)


def save_text(path: Path, data: str) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(data)
    os.replace(tmp, path)


def read_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _git_candidates() -> list[str]:
    candidates = []
    which = shutil.which("git")
    if which:
        candidates.append(which)
    candidates.extend(["/usr/bin/git", "/bin/git", "git"])
    seen = []
    for item in candidates:
        if item not in seen:
            seen.append(item)
    return seen


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    last_error = None
    for git in _git_candidates():
        try:
            result = subprocess.run(
                [git] + args,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 or git != "git":
                return result
            last_error = result
        except FileNotFoundError as exc:
            last_error = exc
    cmd = " ".join(["git"] + args)
    fallback_cmds = [cmd]
    conda_sh = Path("/root/anaconda3/etc/profile.d/conda.sh")
    if conda_sh.exists():
        fallback_cmds.insert(0, f"source {conda_sh} && conda activate infinigen && {cmd}")
    result = None
    for fallback in fallback_cmds:
        result = subprocess.run(
            ["bash", "-lc", fallback],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result
    return result


def _iso_from_timestamp(value: Any) -> str:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value)).astimezone().isoformat(timespec="seconds")
    if isinstance(value, str):
        return value
    return now_ts()


def _parse_fraction(value: Any) -> tuple[int, int] | None:
    if isinstance(value, str):
        m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", value)
        if m:
            return int(m.group(1)), int(m.group(2))
    if isinstance(value, int):
        return value, value
    return None


def _recursive_has_field(data: Any, field_path: str) -> bool:
    cursor = data
    for part in field_path.split("."):
        if isinstance(cursor, dict) and part in cursor:
            cursor = cursor[part]
            continue
        return False
    return True


def _sanitize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _claim_live_revision(claim: dict[str, Any]) -> dict[str, Any]:
    return (claim.get("revisions") or [{}])[-1]


def _claim_summary(claim: dict[str, Any]) -> dict[str, Any]:
    live = _claim_live_revision(claim)
    return {
        "claim_id": claim.get("claim_id"),
        "revision": claim.get("current_revision"),
        "lifecycle_status": claim.get("lifecycle_status"),
        "status": live.get("status"),
        "scope": live.get("scope"),
        "statement": live.get("statement"),
        "evidence_ids": live.get("evidence_ids", []),
    }


def _claim_debt_flags(claim: dict[str, Any], current_phase: str) -> list[dict[str, Any]]:
    flags = []
    lifecycle = claim.get("lifecycle_status")
    revisions = claim.get("revisions") or []
    statements = [rev.get("statement") for rev in revisions if rev.get("statement")]
    unique_statement_count = len(set(statements))

    if lifecycle == "active" and len(revisions) >= 5 and unique_statement_count <= 1:
        flags.append({
            "flag": "revision_spam",
            "detail": f"{len(revisions)} revisions but only {unique_statement_count} distinct statement",
        })
    elif lifecycle == "active" and len(revisions) >= 8 and unique_statement_count <= 3:
        flags.append({
            "flag": "revision_bloat",
            "detail": f"{len(revisions)} revisions but only {unique_statement_count} distinct statements",
        })

    extra_keys = sorted(key for key in claim.keys() if key not in CLAIM_ALLOWED_TOP_LEVEL_KEYS)
    if extra_keys:
        flags.append({
            "flag": "redundant_top_level_fields",
            "detail": ", ".join(extra_keys),
        })

    superseded_by = claim.get("superseded_by")
    if claim.get("lifecycle_status") == "superseded" and superseded_by:
        flags.append({
            "flag": "review_superseded_link",
            "detail": f"top-level superseded_by={superseded_by}",
        })

    live = _claim_live_revision(claim)
    scope = _sanitize_text(live.get("scope"))
    if (
        claim.get("lifecycle_status") == "active"
        and scope.startswith("v58_")
        and current_phase.startswith("v59_")
    ):
        flags.append({
            "flag": "scope_not_current_phase",
            "detail": f"scope={scope} does not directly match current phase={current_phase}",
        })

    return flags


def _classify_claim_role(
    claim: dict[str, Any],
    current_phase: str,
    next_action: dict[str, Any] | None,
) -> tuple[str, str]:
    claim_id = claim.get("claim_id")
    lifecycle = claim.get("lifecycle_status")
    live = _claim_live_revision(claim)
    status = live.get("status")

    driving_ids = set()
    if next_action:
        driving_ids.update(CURRENT_DRIVING_BY_ACTION_ID.get(next_action.get("id"), set()))
    driving_ids.update(CURRENT_DRIVING_BY_PHASE.get(current_phase, set()))

    if lifecycle != "active":
        return "retired_display", f"lifecycle_status={lifecycle}"
    if claim_id in driving_ids:
        return "current_driving", "explicitly mapped to current phase/action"
    if status in {"candidate", "contradicted"}:
        return "retired_display", f"active but status={status} and not driving the current phase"
    return "historical_context", "active supporting context but not directly driving the current phase"


def _claim_lifecycle_review(
    summary: dict[str, Any],
    role: str,
    flags: list[dict[str, Any]],
) -> dict[str, Any] | None:
    flag_names = {item.get("flag") for item in flags}
    claim_id = summary.get("claim_id")

    if "review_superseded_link" in flag_names:
        return {
            "claim_id": claim_id,
            "priority": "P1",
            "recommended_action": "normalize_superseded_link",
            "reason": "superseded claim still carries a top-level successor pointer that should be reviewed before canonical archive cleanup",
            "display_role": role,
            "flags": sorted(flag_names),
        }

    if "revision_spam" in flag_names:
        return {
            "claim_id": claim_id,
            "priority": "P1",
            "recommended_action": "collapse_revision_chain",
            "reason": "historical active claim has many revisions but effectively one statement; queue a summary/retirement migration instead of letting the chain keep growing",
            "display_role": role,
            "flags": sorted(flag_names),
        }

    if "revision_bloat" in flag_names:
        return {
            "claim_id": claim_id,
            "priority": "P2",
            "recommended_action": "consolidate_revision_history",
            "reason": "historical active claim has a bloated revision chain with little statement change; prepare a canonical summary revision or archive migration",
            "display_role": role,
            "flags": sorted(flag_names),
        }

    if role == "retired_display" and summary.get("lifecycle_status") == "active":
        return {
            "claim_id": claim_id,
            "priority": "P3",
            "recommended_action": "consider_archiving_from_active_display",
            "reason": "claim is active in canonical data but no longer drives the current phase; review whether it should move to a stricter historical lifecycle",
            "display_role": role,
            "flags": sorted(flag_names),
        }

    return None


def gather_repo_status(repo_root: Path) -> dict[str, Any]:
    status = {
        "root": str(repo_root),
        "exists": repo_root.exists(),
        "branch": None,
        "head": None,
        "upstream": None,
        "ahead": None,
        "behind": None,
        "dirty": False,
        "dirty_files": [],
        "error": None,
    }
    if not repo_root.exists():
        status["error"] = "repo_missing"
        return status

    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    head = _run_git(["rev-parse", "HEAD"], repo_root)
    if branch.returncode == 0:
        status["branch"] = branch.stdout.strip()
    else:
        status["error"] = branch.stderr.strip() or branch.stdout.strip()
    if head.returncode == 0:
        status["head"] = head.stdout.strip()

    upstream = _run_git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], repo_root)
    if upstream.returncode == 0:
        status["upstream"] = upstream.stdout.strip()
        counts = _run_git(["rev-list", "--left-right", "--count", "HEAD...@{u}"], repo_root)
        if counts.returncode == 0:
            parts = counts.stdout.strip().split()
            if len(parts) == 2:
                status["behind"] = int(parts[0])
                status["ahead"] = int(parts[1])

    dirty = _run_git(["status", "--short"], repo_root)
    if dirty.returncode == 0:
        dirty_lines = [line.rstrip() for line in dirty.stdout.splitlines() if line.strip()]
        status["dirty_files"] = dirty_lines
        status["dirty"] = bool(dirty_lines)
    else:
        status["error"] = dirty.stderr.strip() or dirty.stdout.strip() or status["error"]
    return status


def build_workspace_manifest() -> dict[str, Any]:
    main_repo = gather_repo_status(PROJECT_ROOT)
    submodules = {}
    if EXTERNAL_MINT.exists():
        submodules["external/MINT"] = gather_repo_status(EXTERNAL_MINT)
    manifest = {
        "schema_version": 1,
        "generated_at": now_ts(),
        "campaign_root": str(CAMPAIGN_ROOT),
        "project_root": str(PROJECT_ROOT),
        "main_repo": main_repo,
        "submodules": submodules,
    }
    save_json(WORKSPACE_MANIFEST_PATH, manifest)
    return manifest


def _read_candidate_logs() -> list[Path]:
    logs = []
    night_log = CAMPAIGN_ROOT / "sovereign" / "night" / "night_runner.log"
    if night_log.exists():
        logs.append(night_log)
    p0b_logs = sorted((ARTIFACTS / "p0b_colored_rollouts" / "logs").glob("*.log"))
    logs.extend(p0b_logs[-5:])
    return logs


def _extract_model_metrics_from_logs() -> dict[str, Any]:
    metrics = {
        "missing_keys_count": None,
        "unexpected_keys_count": None,
        "architectural_key_mismatches_fixed": None,
        "sources": [],
    }
    missing_hits = []
    unexpected_hits = []
    fixed_hits = []
    for log_path in _read_candidate_logs():
        text = read_text(log_path)
        for m in re.finditer(r"Missing keys when loading state dict:\s*(\d+)", text):
            missing_hits.append((log_path, int(m.group(1))))
        for m in re.finditer(r"Unexpected keys when loading state dict:\s*(\d+)", text):
            unexpected_hits.append((log_path, int(m.group(1))))
        for m in re.finditer(r"Fixed\s+(\d+)\s+architectural key mismatches", text):
            fixed_hits.append((log_path, int(m.group(1))))
    if missing_hits:
        path, count = missing_hits[-1]
        metrics["missing_keys_count"] = count
        metrics["sources"].append(str(path))
    if unexpected_hits:
        path, count = unexpected_hits[-1]
        metrics["unexpected_keys_count"] = count
        metrics["sources"].append(str(path))
    if fixed_hits:
        path, count = fixed_hits[-1]
        metrics["architectural_key_mismatches_fixed"] = count
        metrics["sources"].append(str(path))
    metrics["sources"] = sorted(set(metrics["sources"]))
    return metrics


def _diff_for_file(repo_root: Path, file_path: str) -> str:
    result = _run_git(["diff", "--", file_path], repo_root)
    if result.returncode == 0:
        return result.stdout
    return ""


def _classify_patch(file_path: str, diff_text: str) -> tuple[str | None, str]:
    semantic_patterns = [
        r"to_bfloat16_for_selected_params",
        r"self\.to\(dtype=torch\.float32\)",
        r"load_state_dict",
        r"Missing keys",
        r"Unexpected keys",
        r"architectural key mismatches",
        r"language_model\.model",
    ]
    compatibility_patterns = [
        r"embed_tokens",
        r"get_image_features",
        r"\.model\.forward",
        r"\.language_model\.model",
    ]
    if "external/MINT" in file_path or file_path.endswith("modeling_mint.py"):
        if any(re.search(pattern, diff_text) for pattern in semantic_patterns):
            return "semantic", "model load/precision semantics changed"
        if any(re.search(pattern, diff_text) for pattern in compatibility_patterns):
            return "compatibility", "API/path compatibility shim"
        return "compatibility", "local MINT patch present"
    if file_path.startswith("scripts/mint/"):
        return "experimental", "local experiment/runtime shim"
    return None, ""


def build_model_load_fidelity(workspace_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    workspace_manifest = workspace_manifest or load_json(WORKSPACE_MANIFEST_PATH, {})
    main_dirty = [line[3:] if len(line) > 3 else line for line in workspace_manifest.get("main_repo", {}).get("dirty_files", [])]
    sub_dirty = [line[3:] if len(line) > 3 else line for line in workspace_manifest.get("submodules", {}).get("external/MINT", {}).get("dirty_files", [])]

    compatibility_patches = []
    semantic_patches = []
    experimental_patches = []

    main_repo_root = Path(workspace_manifest.get("main_repo", {}).get("root", PROJECT_ROOT))
    mint_repo_root = Path(workspace_manifest.get("submodules", {}).get("external/MINT", {}).get("root", EXTERNAL_MINT))

    for file_path in sorted(set(main_dirty + [f"external/MINT/{p}" for p in sub_dirty])):
        if file_path.startswith("external/MINT/"):
            repo_root = mint_repo_root
            repo_relative = file_path.replace("external/MINT/", "", 1)
        else:
            repo_root = main_repo_root
            repo_relative = file_path
        diff_text = _diff_for_file(repo_root, repo_relative)
        kind, reason = _classify_patch(file_path, diff_text)
        if kind == "compatibility":
            compatibility_patches.append({"file": file_path, "reason": reason})
        elif kind == "semantic":
            semantic_patches.append({"file": file_path, "reason": reason})
        elif kind == "experimental":
            experimental_patches.append({"file": file_path, "reason": reason})

    log_metrics = _extract_model_metrics_from_logs()

    fidelity_grade = "F0"
    summary = "Upstream-faithful"
    if semantic_patches or (log_metrics.get("missing_keys_count") or 0) > 0 or (log_metrics.get("unexpected_keys_count") or 0) > 0:
        fidelity_grade = "F2"
        summary = "Runnable but semantically drifted"
    elif compatibility_patches or experimental_patches:
        fidelity_grade = "F1"
        summary = "Compatibility-patched but high-fidelity"

    payload = {
        "schema_version": 1,
        "generated_at": now_ts(),
        "campaign": CAMPAIGN_ROOT.name,
        "main_repo_head": workspace_manifest.get("main_repo", {}).get("head"),
        "main_repo_branch": workspace_manifest.get("main_repo", {}).get("branch"),
        "external_mint_head": workspace_manifest.get("submodules", {}).get("external/MINT", {}).get("head"),
        "external_mint_branch": workspace_manifest.get("submodules", {}).get("external/MINT", {}).get("branch"),
        "compatibility_patches": compatibility_patches,
        "semantic_patches": semantic_patches,
        "experimental_patches": experimental_patches,
        "missing_keys_count": log_metrics.get("missing_keys_count"),
        "unexpected_keys_count": log_metrics.get("unexpected_keys_count"),
        "architectural_key_mismatches_fixed": log_metrics.get("architectural_key_mismatches_fixed"),
        "metric_sources": log_metrics.get("sources", []),
        "dtype_policy": "current runtime uses local patched precision path; see semantic patches",
        "pretrained_and_finetuned_same_shim": True,
        "fidelity_grade": fidelity_grade,
        "fidelity_summary": summary,
        "guardrail": "Scientific conclusions apply to the local patched MINT variant, not blindly to upstream MINT.",
    }
    save_json(MODEL_LOAD_FIDELITY_PATH, payload)
    return payload


def normalize_run_ledger() -> dict[str, Any]:
    data = load_yaml(RUN_LEDGER_PATH, default={}) or {}
    raw_entries = data.get("entries", [])
    if not isinstance(raw_entries, list):
        raw_entries = []
    normalized_entries = []
    for entry in raw_entries:
        ts = entry.get("timestamp") or entry.get("started_at") or now_ts()
        duration_hours = entry.get("duration_hours")
        ended_at = entry.get("ended_at")
        if duration_hours is not None and not ended_at:
            ended_at = ts
        normalized_entries.append({
            "session_id": entry.get("session_id", f"legacy_{len(normalized_entries)+1:03d}"),
            "started_at": entry.get("started_at", ts),
            "ended_at": ended_at or entry.get("finished_at", ts),
            "operator": entry.get("operator", "unknown"),
            "intent": entry.get("intent", entry.get("campaign", "legacy_session")),
            "truth_sources_used": entry.get("truth_sources_used", []),
            "scripts_run": entry.get("scripts_run", []),
            "mutations_performed": entry.get("mutations_performed", entry.get("sovereign_updated", [])),
            "failures_and_retries": entry.get("failures_and_retries", []),
            "route_changes": entry.get("route_changes", []),
            "raw_artifacts_created": entry.get("raw_artifacts_created", entry.get("new_evidence", [])),
            "scientific_conclusions_promoted": entry.get("scientific_conclusions_promoted", [
                item for item in entry.get("engineering_vs_scientific", []) if item.get("type") == "scientific_conclusion"
            ]),
            "engineering_conclusions_promoted": entry.get("engineering_conclusions_promoted", [
                item for item in entry.get("engineering_vs_scientific", []) if item.get("type") == "engineering_fix"
            ]),
            "new_risks": entry.get("new_risks", []),
            "followup_required": entry.get("followup_required", entry.get("next_session_primers", [])),
        })
    normalized = {
        "schema_version": 1,
        "campaign": CAMPAIGN_ROOT.name,
        "updated_at": now_ts(),
        "entries": normalized_entries,
    }
    save_yaml(RUN_LEDGER_PATH, normalized)
    return normalized


def append_run_ledger_entry(entry: dict[str, Any]) -> None:
    data = normalize_run_ledger()
    data["entries"].append(entry)
    data["updated_at"] = now_ts()
    save_yaml(RUN_LEDGER_PATH, data)


def _current_next_action(actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    in_progress = [item for item in actions if item.get("status") == "in_progress"]
    pending = [item for item in actions if item.get("status") == "pending"]
    if in_progress:
        return in_progress[0]
    if pending:
        return pending[0]
    return None


def _derive_runtime_from_actions(next_actions: dict[str, Any], fallback_state: dict[str, Any]) -> dict[str, Any]:
    actions = next_actions.get("actions", [])
    current_next = _current_next_action(actions) or {}

    phase = (
        next_actions.get("next_phase")
        or next_actions.get("phase")
        or fallback_state.get("phase")
        or "unknown"
    )
    phase_gate = next_actions.get("phase_gate") or fallback_state.get("phase_gate") or "unknown"

    if current_next and next_actions.get("decision"):
        verdict = next_actions.get("decision")
    else:
        verdict = next_actions.get("verdict") or fallback_state.get("verdict") or "unknown"

    return {
        "phase": phase,
        "phase_gate": phase_gate,
        "verdict": verdict,
        "decision": next_actions.get("decision") or verdict,
        "current_next": current_next,
    }


def normalize_next_actions(next_actions: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    data = deepcopy(next_actions)
    actions = data.get("actions", [])
    gate_b_idx = None
    sim_idx = None
    for idx, action in enumerate(actions):
        aid = action.get("id")
        atype = action.get("type")
        if aid == "gate_b_teacher_replayability":
            gate_b_idx = idx
        if atype == "simulator_physics_investigation":
            sim_idx = idx
    if gate_b_idx is not None and sim_idx is not None and gate_b_idx > sim_idx:
        gate_b = actions.pop(gate_b_idx)
        sim_idx = next(i for i, action in enumerate(actions) if action.get("type") == "simulator_physics_investigation")
        actions.insert(sim_idx, gate_b)

    runtime = _derive_runtime_from_actions(data, state)
    data["phase"] = runtime["phase"]
    data["phase_gate"] = runtime["phase_gate"]
    data["verdict"] = runtime["verdict"]
    data["decision"] = runtime["decision"]
    data["generated_at"] = now_ts()
    data["authority_note"] = (
        "Canonical action queue. current_truth.json and generated docs are derived from this file plus "
        "claims/evidence/dataset/workspace/model manifests."
    )
    return data


def migrate_state_file() -> dict[str, Any]:
    state = load_json(STATE_PATH, default={}) or {}
    next_actions = load_json(NEXT_ACTIONS_PATH, default={}) or {}
    actions = next_actions.get("actions", [])
    runtime = _derive_runtime_from_actions(next_actions, state)
    current_next = runtime["current_next"]

    if state.get("runtime_schema_version") == 2:
        state["phase"] = runtime["phase"]
        state["phase_gate"] = runtime["phase_gate"]
        state["verdict"] = runtime["verdict"]
        state["active_job"] = (current_next or {}).get("id") or (current_next or {}).get("type") or state.get("phase")
        state["runtime_pointers"] = {
            "current_next_action_id": (current_next or {}).get("id"),
            "current_next_action_type": (current_next or {}).get("type"),
            "last_error": state.get("runtime_pointers", {}).get("last_error"),
            "last_repair_action": state.get("runtime_pointers", {}).get("last_repair_action"),
            "generated_from": "next_actions.json",
        }
        state["updated_at"] = next_actions.get("generated_at", state.get("updated_at", now_ts()))
        save_json(STATE_PATH, state)
        return state

    ensure_dir(LEGACY_DIR)
    snapshot_name = f"state_legacy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    snapshot_path = LEGACY_DIR / snapshot_name
    save_json(snapshot_path, state)

    queue = _normalize_list(state.get("queue"))
    recent_history = _normalize_list(state.get("history"))[-5:]
    runtime = {
        "runtime_schema_version": 2,
        "phase": runtime["phase"],
        "phase_gate": runtime["phase_gate"],
        "verdict": runtime["verdict"],
        "updated_at": next_actions.get("generated_at", state.get("updated_at", now_ts())),
        "active_job": (current_next or {}).get("id") or (current_next or {}).get("type") or state.get("phase"),
        "runtime_pointers": {
            "current_next_action_id": (current_next or {}).get("id"),
            "current_next_action_type": (current_next or {}).get("type"),
            "last_error": state.get("last_error"),
            "last_repair_action": state.get("last_repair_action"),
            "generated_from": "next_actions.json",
        },
        "queue": queue,
        "queue_summary": {
            "total": len(queue),
            "pending": sum(1 for item in queue if item.get("status") == "pending"),
            "in_progress": sum(1 for item in queue if item.get("status") == "in_progress"),
            "completed": sum(1 for item in queue if item.get("status") == "completed"),
            "blocked": sum(1 for item in queue if item.get("status") == "blocked"),
            "cancelled": sum(1 for item in queue if item.get("status") == "cancelled"),
        },
        "experiments": state.get("experiments", []),
        "recent_history": recent_history,
        "legacy_snapshot": str(snapshot_path.relative_to(SOVEREIGN)),
        "legacy_notice": "Full historical narrative moved to sovereign/legacy snapshot. state.json is runtime-only.",
    }
    save_json(STATE_PATH, runtime)
    return runtime


def _canonical_mtime(paths: list[Path]) -> float:
    mtimes = [path.stat().st_mtime for path in paths if path.exists()]
    return max(mtimes) if mtimes else 0.0


def _latest_bootstrap_path() -> Path:
    return SOVEREIGN / f"SESSION_BOOTSTRAP.{datetime.now().strftime('%Y%m%d')}.md"


def _derived_doc_status(path: Path, verdict: str, phase: str, latest_canonical_mtime: float) -> dict[str, Any]:
    status = {
        "path": str(path.relative_to(CAMPAIGN_ROOT)),
        "exists": path.exists(),
        "stale": False,
        "reasons": [],
    }
    if not path.exists():
        status["stale"] = True
        status["reasons"].append("missing")
        return status
    text = read_text(path)
    if GENERATED_MARKER not in text:
        status["stale"] = True
        status["reasons"].append("missing_generated_marker")
    if "Source: `sovereign/current_truth.json`" not in text and "Source: sovereign/current_truth.json" not in text:
        status["stale"] = True
        status["reasons"].append("missing_source_pointer")
    if verdict and verdict not in text:
        status["stale"] = True
        status["reasons"].append("verdict_mismatch")
    if phase and phase not in text:
        status["stale"] = True
        status["reasons"].append("phase_mismatch")
    if path.stat().st_mtime < latest_canonical_mtime:
        status["stale"] = True
        status["reasons"].append("older_than_canonical")
    return status


def _sorted_evidence_entries(index: dict[str, Any]) -> list[dict[str, Any]]:
    entries = list(index.get("entries", []))
    entries.sort(key=lambda entry: _iso_from_timestamp(entry.get("timestamp")), reverse=True)
    return entries


def build_current_truth() -> dict[str, Any]:
    claims = load_yaml(CLAIMS_PATH, default={"claims": []}) or {"claims": []}
    evidence_index = load_json(EVIDENCE_INDEX_PATH, default={"entries": []}) or {"entries": []}
    next_actions = load_json(NEXT_ACTIONS_PATH, default={}) or {}
    dataset_manifest = load_json(DATASET_MANIFEST_PATH, default={}) or {}
    workspace_manifest = load_json(WORKSPACE_MANIFEST_PATH, default={}) or {}
    model_fidelity = load_json(MODEL_LOAD_FIDELITY_PATH, default={}) or {}
    state = load_json(STATE_PATH, default={}) or {}

    actions = next_actions.get("actions", [])
    pending = [action for action in actions if action.get("status") == "pending"]
    in_progress = [action for action in actions if action.get("status") == "in_progress"]
    blocked = [action for action in actions if action.get("status") == "blocked"]
    completed = [action for action in actions if action.get("status") == "completed"]
    next_action = in_progress[0] if in_progress else (pending[0] if pending else None)

    active_claims = []
    current_driving_claims = []
    historical_context_claims = []
    retired_display_claims = []
    claim_debt_flags = []
    lifecycle_review_queue = []
    for claim in claims.get("claims", []):
        summary = _claim_summary(claim)
        role, role_reason = _classify_claim_role(
            claim,
            state.get("phase", next_actions.get("phase", "unknown")),
            next_action,
        )
        summary["display_role"] = role
        summary["display_role_reason"] = role_reason
        flags = _claim_debt_flags(claim, state.get("phase", next_actions.get("phase", "unknown")))
        if flags:
            summary["debt_flags"] = [item["flag"] for item in flags]
            claim_debt_flags.append({
                "claim_id": summary["claim_id"],
                "flags": flags,
            })
            review_item = _claim_lifecycle_review(summary, role, flags)
            if review_item:
                lifecycle_review_queue.append(review_item)

        if claim.get("lifecycle_status") == "active":
            active_claims.append(summary)
        if role == "current_driving":
            current_driving_claims.append(summary)
        elif role == "historical_context":
            historical_context_claims.append(summary)
        else:
            retired_display_claims.append(summary)

    lifecycle_review_queue.sort(
        key=lambda item: (
            LIFECYCLE_REVIEW_PRIORITY_ORDER.get(item.get("priority", "P9"), 99),
            item.get("claim_id", ""),
        )
    )

    evidence_entries = _sorted_evidence_entries(evidence_index)
    latest_verified = [entry for entry in evidence_entries if entry.get("verified")][:5]
    latest_all = evidence_entries[:5]
    latest_canonical_mtime = _canonical_mtime(
        [
            CLAIMS_PATH,
            EVIDENCE_INDEX_PATH,
            NEXT_ACTIONS_PATH,
            DATASET_MANIFEST_PATH,
            WORKSPACE_MANIFEST_PATH,
            MODEL_LOAD_FIDELITY_PATH,
        ] + sorted(EVIDENCE_DIR.glob("E*.yaml"))
    )

    bootstrap_path = _latest_bootstrap_path()
    derived_status = [
        _derived_doc_status(bootstrap_path, state.get("verdict", ""), state.get("phase", ""), latest_canonical_mtime),
        _derived_doc_status(USAGE_GUIDE_PATH, state.get("verdict", ""), state.get("phase", ""), latest_canonical_mtime),
        _derived_doc_status(HANDOFF_PATH, state.get("verdict", ""), state.get("phase", ""), latest_canonical_mtime),
        _derived_doc_status(CAMPAIGN_TRUTH_PATH, state.get("verdict", ""), state.get("phase", ""), latest_canonical_mtime),
    ]

    truth = {
        "schema_version": 1,
        "generated_at": now_ts(),
        "campaign": {
            "id": CAMPAIGN_ROOT.name,
            "root": str(CAMPAIGN_ROOT),
        },
        "authority": {
            "tier0": CANONICAL_TIER0,
            "tier1": ["sovereign/state.json"],
            "tier2": [
                str(bootstrap_path.relative_to(CAMPAIGN_ROOT)),
                str(USAGE_GUIDE_PATH.relative_to(CAMPAIGN_ROOT)),
                str(HANDOFF_PATH.relative_to(CAMPAIGN_ROOT)),
                str(CAMPAIGN_TRUTH_PATH.relative_to(CAMPAIGN_ROOT)),
            ],
        },
        "current": {
            "phase": state.get("phase", next_actions.get("phase", "unknown")),
            "phase_gate": state.get("phase_gate", next_actions.get("phase_gate", "unknown")),
            "verdict": state.get("verdict", next_actions.get("verdict", "unknown")),
            "decision": next_actions.get("decision", state.get("verdict", "unknown")),
            "next_action": next_action,
            "open_gates": [action for action in actions if action.get("type") == "learnability_audit_gate" and action.get("status") != "completed"],
            "blockers": blocked,
            "stale_docs": [item for item in derived_status if item["stale"]],
            "authority_notes": [
                "current_truth.json is derived from Tier-0/Tier-1 only.",
                "Derived docs are convenience surfaces and must not override canonical files.",
                model_fidelity.get("guardrail", ""),
            ],
        },
        "dataset": dataset_manifest,
        "workspace": workspace_manifest,
        "model_load_fidelity": model_fidelity,
        "claims": {
            "active_count": len(active_claims),
            "summary": active_claims,
            "current_driving": current_driving_claims,
            "historical_context": historical_context_claims,
            "retired_display": retired_display_claims,
            "debt_flags": claim_debt_flags,
            "lifecycle_review_queue": lifecycle_review_queue,
            "counts": {
                "current_driving": len(current_driving_claims),
                "historical_context": len(historical_context_claims),
                "retired_display": len(retired_display_claims),
                "with_debt_flags": len(claim_debt_flags),
                "lifecycle_review_queue": len(lifecycle_review_queue),
            },
        },
        "evidence": {
            "total": len(evidence_index.get("entries", [])),
            "verified_count": sum(1 for entry in evidence_index.get("entries", []) if entry.get("verified")),
            "latest_verified": latest_verified,
            "latest_all": latest_all,
        },
        "actions": {
            "pending_count": len(pending),
            "in_progress_count": len(in_progress),
            "completed_count": len(completed),
            "blocked_count": len(blocked),
        },
        "derived_docs": {
            "latest_bootstrap": str(bootstrap_path.relative_to(CAMPAIGN_ROOT)),
            "usage_guide": str(USAGE_GUIDE_PATH.relative_to(CAMPAIGN_ROOT)),
            "handoff": str(HANDOFF_PATH.relative_to(CAMPAIGN_ROOT)),
            "campaign_truth": str(CAMPAIGN_TRUTH_PATH.relative_to(CAMPAIGN_ROOT)),
            "statuses": derived_status,
        },
    }
    save_json(CURRENT_TRUTH_PATH, truth)
    return truth


def _markdown_header(title: str, generated_at: str) -> str:
    return "\n".join([
        f"# {title}",
        f"**{GENERATED_MARKER}**",
        "",
        f"- Generated at: `{generated_at}`",
        "- Source: `sovereign/current_truth.json`",
        "- Stale policy: regenerate with `python scripts/harness/sovereign_cli.py go`",
        "",
    ])


def render_session_bootstrap(truth: dict[str, Any]) -> str:
    current = truth["current"]
    dataset = truth["dataset"]
    workspace = truth["workspace"]
    model_fidelity = truth["model_load_fidelity"]
    stale_docs = current.get("stale_docs", [])
    next_action = current.get("next_action") or {}
    bootstrap_date = datetime.now().strftime("%Y%m%d")
    lines = [_markdown_header(f"SESSION_BOOTSTRAP.{bootstrap_date}", truth["generated_at"])]
    lines += [
        "## Current Canonical Truth",
        f"- Verdict: `{current.get('verdict', 'unknown')}`",
        f"- Phase: `{current.get('phase', 'unknown')}`",
        f"- Phase gate: `{current.get('phase_gate', 'unknown')}`",
        f"- Decision: `{current.get('decision', 'unknown')}`",
        "",
        "## Canonical Sources",
        *[f"- `{item}`" for item in truth["authority"]["tier0"]],
        "",
        "## Dataset Anchor",
        f"- Version: `{dataset.get('dataset_version', 'unknown')}`",
        f"- Episodes: `{dataset.get('episode_count', 'unknown')}`",
        f"- Frames: `{dataset.get('frame_count', 'unknown')}`",
        f"- Loads: `{dataset.get('dataset_loads', 'unknown')}`",
        "",
        "## Workspace Snapshot",
        f"- Branch: `{workspace.get('main_repo', {}).get('branch')}`",
        f"- HEAD: `{workspace.get('main_repo', {}).get('head')}`",
        f"- Dirty files: `{len(workspace.get('main_repo', {}).get('dirty_files', []))}`",
        f"- external/MINT dirty files: `{len(workspace.get('submodules', {}).get('external/MINT', {}).get('dirty_files', []))}`",
        "",
        "## Model Load Fidelity",
        f"- Grade: `{model_fidelity.get('fidelity_grade', 'unknown')}`",
        f"- Summary: {model_fidelity.get('fidelity_summary', 'unknown')}",
        "",
        "## Next Action",
        f"- Type: `{next_action.get('type', 'none')}`",
        f"- Id: `{next_action.get('id', '-')}`",
        f"- Priority: `{next_action.get('priority', '-')}`",
        f"- Target: {next_action.get('target', '-')}",
        "",
        "## Stale / Unsafe Docs",
    ]
    if stale_docs:
        lines.extend([f"- `{item['path']}` — {', '.join(item['reasons'])}" for item in stale_docs])
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def render_usage_guide(truth: dict[str, Any]) -> str:
    current = truth["current"]
    next_action = current.get("next_action") or {}
    dataset = truth.get("dataset", {})
    workspace = truth.get("workspace", {})
    model_fidelity = truth.get("model_load_fidelity", {})
    stale_docs = current.get("stale_docs", [])
    latest_verified = truth.get("evidence", {}).get("latest_verified", [])
    open_gates = current.get("open_gates", [])
    blockers = current.get("blockers", [])
    current_driving_claims = truth.get("claims", {}).get("current_driving", [])
    historical_context_claims = truth.get("claims", {}).get("historical_context", [])
    claim_debt_flags = truth.get("claims", {}).get("debt_flags", [])
    lifecycle_review_queue = truth.get("claims", {}).get("lifecycle_review_queue", [])

    human_read = (
        f"We are in `{current.get('phase', 'unknown')}`. "
        f"The current top priority is `{next_action.get('id', '-')}`: {next_action.get('target', '-')}"
        if next_action
        else "No pending canonical next action is currently registered."
    )

    lines = [_markdown_header("Harness v2 — mint_drawer_v1", truth["generated_at"])]
    lines += [
        "## At A Glance",
        f"- Verdict: `{current.get('verdict', 'unknown')}`",
        f"- Phase: `{current.get('phase', 'unknown')}`",
        f"- Decision: `{current.get('decision', 'unknown')}`",
        f"- Human read: {human_read}",
        f"- Dataset anchor: `{dataset.get('dataset_version', 'unknown')}` / `{dataset.get('episode_count', 'unknown')}` episodes / `{dataset.get('frame_count', 'unknown')}` frames / loads=`{dataset.get('dataset_loads', 'unknown')}`",
        f"- Workspace: branch `{workspace.get('main_repo', {}).get('branch')}` @ `{workspace.get('main_repo', {}).get('head')}` / dirty files `{len(workspace.get('main_repo', {}).get('dirty_files', []))}`",
        f"- MINT fidelity: `{model_fidelity.get('fidelity_grade', 'unknown')}` — {model_fidelity.get('fidelity_summary', 'unknown')}",
        "",
        "## Claims Driving This Phase",
    ]
    if current_driving_claims:
        for claim in current_driving_claims:
            lines.append(
                f"- `{claim.get('claim_id')}` [{claim.get('status', 'unknown')}]: {claim.get('statement', '')}"
            )
    else:
        lines.append("- No current-driving claims were classified.")

    lines += [
        "",
        "## Historical Context Claims",
        f"- Historical context claims currently suppressed from the main dashboard: `{len(historical_context_claims)}`",
    ]
    for claim in historical_context_claims[:5]:
        lines.append(
            f"- `{claim.get('claim_id')}` [{claim.get('status', 'unknown')}] / scope `{claim.get('scope', 'unknown')}`"
        )
    if len(historical_context_claims) > 5:
        lines.append(f"- ... plus `{len(historical_context_claims) - 5}` more historical-context claims")

    lines += [
        "",
        "## What Is Settled Right Now",
    ]
    if latest_verified:
        for entry in latest_verified[:5]:
            lines.append(f"- `{entry.get('evidence_id')}`: {entry.get('summary', '(no summary)')}")
    else:
        lines.append("- No verified evidence registered yet.")

    lines += [
        "",
        "## What Is Still Open",
    ]
    if open_gates:
        for gate in open_gates:
            lines.append(
                f"- Open gate `{gate.get('id', gate.get('type', '?'))}` [{gate.get('priority', '-')}]"
                f": {gate.get('target', '-')}"
            )
    else:
        lines.append("- No open canonical gates.")
    if blockers:
        for blocker in blockers:
            lines.append(f"- Blocker `{blocker.get('id', blocker.get('type', '?'))}`: {blocker.get('target', '-')}")
    else:
        lines.append("- No canonical blockers are registered.")

    lines += [
        "",
        "## What To Read First In A New Session",
        "1. `python scripts/harness/sovereign_cli.py go`",
        "2. `sovereign/current_truth.json`",
        "3. The latest verified evidence listed in `current_truth.json`",
        "4. The spec for the current next action, if the next action is spec-backed",
        "",
        "## Tonight / Next Safe Move",
        f"- Canonical next action: `{next_action.get('type', 'none')}` / `{next_action.get('id', '-')}` / priority `{next_action.get('priority', '-')}`",
        f"- Target: {next_action.get('target', '-')}",
        f"- Scope: {next_action.get('description', 'No scoped description provided.')}",
        "",
        "## Canonical Rules",
        "- Tier 0 files are authoritative; generated docs are convenience views only.",
        "- If a generated doc conflicts with `current_truth.json`, regenerate and trust `current_truth.json`.",
        "- `state.json` is runtime-only. It is not the place to carry long scientific narrative.",
        "- Only spec-backed deterministic experiments are allowed to auto-publish scientific updates.",
        "",
        "## What Not To Trust",
        "- Do not treat bootstrap/handoff/usage guide as canonical truth.",
        "- Do not promote inconclusive raw artifacts to eliminated/confirmed narrative.",
        "- Do not read old Cursor summaries as current project truth.",
        "- Do not interpret the current campaign as evaluating upstream-faithful MINT; check `model_load_fidelity.json` first.",
        "",
        "## Quick Files",
        "- `sovereign/current_truth.json` — single generated truth surface for current status",
        "- `sovereign/next_actions.json` — canonical action queue",
        "- `sovereign/model_load_fidelity.json` — whether current MINT runtime is scientifically comparable",
        "- `sovereign/run_ledger.yaml` — append-only process log",
        "- `sovereign/experiment_specs/` — what night runner is allowed to auto-publish",
        "",
        "## Claim Debt To Clean Later",
        f"- Claims with debt flags: `{len(claim_debt_flags)}`",
    ]
    for item in claim_debt_flags[:5]:
        flags = ", ".join(flag["flag"] for flag in item.get("flags", []))
        lines.append(f"- `{item['claim_id']}` — {flags}")
    if len(claim_debt_flags) > 5:
        lines.append(f"- ... plus `{len(claim_debt_flags) - 5}` more claim-debt entries")

    lines += [
        "",
        "## Claim Lifecycle Review Queue",
        f"- Canonical cleanup candidates now: `{len(lifecycle_review_queue)}`",
    ]
    for item in lifecycle_review_queue[:5]:
        flags = ", ".join(item.get("flags", []))
        lines.append(
            f"- [{item.get('priority', '-')}] `{item.get('claim_id')}` → `{item.get('recommended_action')}`"
            f" ({flags})"
        )
    if lifecycle_review_queue:
        lines.append("- These are queued canonical cleanup candidates, not live scientific verdict changes.")
    else:
        lines.append("- none")

    lines += [
        "",
        "## Generated Doc Health",
        f"- Stale derived docs: `{len(stale_docs)}`",
    ]
    if stale_docs:
        for item in stale_docs[:8]:
            lines.append(f"- `{item['path']}` — {', '.join(item['reasons'])}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def render_handoff(truth: dict[str, Any]) -> str:
    current = truth["current"]
    next_action = current.get("next_action") or {}
    stale_docs = current.get("stale_docs", [])
    lifecycle_review_queue = truth.get("claims", {}).get("lifecycle_review_queue", [])
    lines = [_markdown_header("Handoff — mint_drawer_v1", truth["generated_at"])]
    lines += [
        "## Current Verdict",
        f"- Verdict: `{current.get('verdict', 'unknown')}`",
        f"- Phase: `{current.get('phase', 'unknown')}`",
        f"- Decision: `{current.get('decision', 'unknown')}`",
        "",
        "## What The Next Agent Must Do",
        "1. Run `python scripts/harness/sovereign_cli.py go` first.",
        "2. Read `sovereign/current_truth.json`, not historical summaries.",
        "3. Respect experiment specs before publishing any scientific conclusion.",
        f"4. Current next action: `{next_action.get('type', 'none')}` / `{next_action.get('id', '-')}`.",
        "",
        "## Current Risks",
        f"- Model fidelity: `{truth.get('model_load_fidelity', {}).get('fidelity_grade', 'unknown')}`",
        f"- Stale derived docs: `{len(stale_docs)}`",
        f"- Claim lifecycle review queue: `{len(lifecycle_review_queue)}`",
    ]
    if lifecycle_review_queue:
        top_item = lifecycle_review_queue[0]
        lines += [
            f"- Top canonical cleanup candidate: `[{top_item.get('priority', '-')}] {top_item.get('claim_id')}` -> `{top_item.get('recommended_action')}`",
        ]
    lines += [
        "",
        "## Do Not",
        "- Do not hand-edit generated docs.",
        "- Do not cite legacy V58 narrative from archived state snapshots as current truth.",
        "- Do not publish unsupported verdict upgrades from raw artifacts.",
    ]
    return "\n".join(lines) + "\n"


def render_campaign_truth(truth: dict[str, Any]) -> str:
    current = truth["current"]
    lines = [_markdown_header(f"CAMPAIGN_TRUTH — {truth['campaign']['id']}", truth["generated_at"])]
    lines += [
        "## Canonical Summary",
        f"- Verdict: `{current.get('verdict', 'unknown')}`",
        f"- Phase: `{current.get('phase', 'unknown')}`",
        f"- Phase gate: `{current.get('phase_gate', 'unknown')}`",
        f"- Decision: `{current.get('decision', 'unknown')}`",
        "",
        "## Active Claims",
    ]
    for item in truth["claims"]["summary"]:
        lines.append(
            f"- `{item['claim_id']}` rev{item['revision']} [{item['status']}] — {item['statement']}"
        )
    lines += [
        "",
        "## Latest Verified Evidence",
    ]
    for entry in truth["evidence"]["latest_verified"]:
        lines.append(
            f"- `{entry['evidence_id']}` `{entry.get('experiment_id', '')}` — {entry.get('summary', '(no summary)')}"
        )
    lines += [
        "",
        "## Actions",
        f"- Pending: `{truth['actions']['pending_count']}`",
        f"- In progress: `{truth['actions']['in_progress_count']}`",
        f"- Completed: `{truth['actions']['completed_count']}`",
        f"- Blocked: `{truth['actions']['blocked_count']}`",
    ]
    return "\n".join(lines) + "\n"


def derive_docs(truth: dict[str, Any] | None = None) -> dict[str, str]:
    truth = truth or load_json(CURRENT_TRUTH_PATH, default={}) or build_current_truth()
    bootstrap_path = _latest_bootstrap_path()
    outputs = {
        str(bootstrap_path): render_session_bootstrap(truth),
        str(USAGE_GUIDE_PATH): render_usage_guide(truth),
        str(HANDOFF_PATH): render_handoff(truth),
        str(CAMPAIGN_TRUTH_PATH): render_campaign_truth(truth),
    }
    for path_str, text in outputs.items():
        save_text(Path(path_str), text)
    return outputs


def load_experiment_specs() -> dict[str, Any]:
    specs = {}
    if not EXPERIMENT_SPECS_DIR.exists():
        return specs
    for path in sorted(EXPERIMENT_SPECS_DIR.glob("*.yaml")):
        if path.name.startswith("._"):
            continue
        data = load_yaml(path, default={}) or {}
        spec_id = data.get("spec_id") or path.stem
        data["path"] = str(path)
        specs[spec_id] = data
    return specs


def validate_experiment_artifact(spec: dict[str, Any], artifact: dict[str, Any]) -> list[str]:
    errors = []
    for field_path in spec.get("required_schema_fields", []):
        if not _recursive_has_field(artifact, field_path):
            errors.append(f"missing field: {field_path}")
    return errors


def _set_evidence_verified(evidence_id: str, verified: bool) -> None:
    index = load_json(EVIDENCE_INDEX_PATH, default={"entries": []}) or {"entries": []}
    for entry in index.get("entries", []):
        if entry.get("evidence_id") == evidence_id:
            entry["verified"] = verified
            break
    index["last_updated"] = now_ts()
    save_json(EVIDENCE_INDEX_PATH, index)


def _publish_p0b_drawer_only_ab(spec: dict[str, Any], artifact_path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    comparison = artifact.get("a_b_comparison", {})
    if not comparison.get("matched_A_B"):
        raise ValueError("artifact is not a matched A/B result")
    evidence_id = spec.get("canonical_evidence_id", "E026")
    _set_evidence_verified(evidence_id, True)

    next_actions = load_json(NEXT_ACTIONS_PATH, default={}) or {}
    for action in next_actions.get("actions", []):
        if action.get("type") == "P0b_colored_reroll":
            action["status"] = "completed"
            action["result"] = {
                "matched_A_B": True,
                "verdict": "DRAWER_COLOR_NOT_SOLE_BOTTLENECK",
                "evidence": evidence_id,
                "artifact": str(artifact_path.relative_to(CAMPAIGN_ROOT)),
            }
            action["note"] = "Published by spec-backed publisher from matched A/B raw artifact."
            break
    save_json(NEXT_ACTIONS_PATH, normalize_next_actions(next_actions, load_json(STATE_PATH, default={})))

    append_run_ledger_entry({
        "session_id": f"publish_{spec['spec_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "started_at": now_ts(),
        "ended_at": now_ts(),
        "operator": "harness_publisher",
        "intent": f"publish {spec['spec_id']}",
        "truth_sources_used": [
            str(artifact_path.relative_to(CAMPAIGN_ROOT)),
            str(EVIDENCE_INDEX_PATH.relative_to(CAMPAIGN_ROOT)),
            str(NEXT_ACTIONS_PATH.relative_to(CAMPAIGN_ROOT)),
        ],
        "scripts_run": ["sovereign_cli.py publish-experiment"],
        "mutations_performed": [
            {"file": str(EVIDENCE_INDEX_PATH.relative_to(CAMPAIGN_ROOT)), "change": f"{evidence_id}.verified=true"},
            {"file": str(NEXT_ACTIONS_PATH.relative_to(CAMPAIGN_ROOT)), "change": "P0b action normalized/published"},
        ],
        "failures_and_retries": [],
        "route_changes": [],
        "raw_artifacts_created": [str(artifact_path.relative_to(CAMPAIGN_ROOT))],
        "scientific_conclusions_promoted": [
            {
                "scope": "drawer-only matched A/B",
                "detail": "Drawer-only color not sole bottleneck",
            }
        ],
        "engineering_conclusions_promoted": [],
        "new_risks": [],
        "followup_required": ["Regenerate current truth and derived docs"],
    })
    return {
        "published": True,
        "evidence_id": evidence_id,
        "verdict": "DRAWER_COLOR_NOT_SOLE_BOTTLENECK",
    }


def publish_experiment(spec_id: str, artifact_path: Path) -> dict[str, Any]:
    specs = load_experiment_specs()
    if spec_id not in specs:
        raise ValueError(f"Unknown experiment spec: {spec_id}")
    spec = specs[spec_id]
    artifact = load_json(artifact_path, default=None)
    if artifact is None:
        raise ValueError(f"Artifact not found: {artifact_path}")
    errors = validate_experiment_artifact(spec, artifact)
    if errors:
        raise ValueError("; ".join(errors))

    if not spec.get("auto_publish", False):
        proposal = {
            "proposal_id": f"publish_{spec_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "spec_id": spec_id,
            "artifact": str(artifact_path.relative_to(CAMPAIGN_ROOT)),
            "created_at": now_ts(),
            "reason": "Spec is proposal-only; human/daylight review required.",
        }
        proposal_path = SOVEREIGN / "proposals" / f"{proposal['proposal_id']}.yaml"
        save_yaml(proposal_path, proposal)
        return {"published": False, "proposal": str(proposal_path.relative_to(CAMPAIGN_ROOT))}

    if spec_id == "p0b_drawer_only_ab":
        return _publish_p0b_drawer_only_ab(spec, artifact_path, artifact)

    proposal = {
        "proposal_id": f"publish_{spec_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "spec_id": spec_id,
        "artifact": str(artifact_path.relative_to(CAMPAIGN_ROOT)),
        "created_at": now_ts(),
        "reason": "Spec exists but no deterministic publisher implemented; proposal-only fallback.",
    }
    proposal_path = SOVEREIGN / "proposals" / f"{proposal['proposal_id']}.yaml"
    save_yaml(proposal_path, proposal)
    return {"published": False, "proposal": str(proposal_path.relative_to(CAMPAIGN_ROOT))}
