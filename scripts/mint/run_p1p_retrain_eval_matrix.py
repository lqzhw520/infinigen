#!/usr/bin/env python3
"""P2: retrain and reevaluate only after earlier gates pass."""

from __future__ import annotations

import json
from pathlib import Path

from mint_common import now_iso, write_json_atomic, write_text_atomic
from p1_execution_common import ARTIFACT_DIR, CAMPAIGN_DIR, OUTPUT_DIR, upsert_evidence
import run_mujoco_infinigen_mainline_night as mainline
from mujoco_mainline_common import default_heldout_seeds

ARTIFACT_PATH = ARTIFACT_DIR / "p1p_retrain_eval_matrix.json"
REPORT_PATH = OUTPUT_DIR / "p1p_retrain_eval_matrix.md"
EVIDENCE_ID = "E058"
EXPERIMENT_ID = "p1p_retrain_eval_matrix"
PREREQS = {
    "p1j": ARTIFACT_DIR / "p1j_control_rerun_v2.json",
    "p1k": ARTIFACT_DIR / "p1k_capture_control_traces.json",
    "p1l": ARTIFACT_DIR / "p1l_eval_parity_matrix.json",
    "p1m": ARTIFACT_DIR / "p1m_state_action_gap_audit.json",
    "p1n": ARTIFACT_DIR / "p1n_visual_semantics_canonical.json",
    "p1o": ARTIFACT_DIR / "p1o_dataset_expansion_manifest.json",
}
DATASET_ROOT = CAMPAIGN_DIR / "dataset_p1o_v2"
DATASET_REPO_ID = "infinigen_drawer_robot_v2"


def _check_prereqs() -> tuple[bool, dict[str, Any]]:
    status = {}
    all_ok = True
    for key, path in PREREQS.items():
        if not path.exists():
            status[key] = {"exists": False, "passed": False}
            all_ok = False
            continue
        payload = json.loads(path.read_text())
        passed = bool(payload.get("passed"))
        status[key] = {"exists": True, "passed": passed, "path": str(path)}
        all_ok = all_ok and passed
    return all_ok, status


def main() -> int:
    prereqs_ok, prereq_status = _check_prereqs()
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at": now_iso(),
        "passed": False,
        "prerequisites_ok": prereqs_ok,
        "prerequisite_status": prereq_status,
        "dataset_root": str(DATASET_ROOT),
        "dataset_repo_id": DATASET_REPO_ID,
        "heldout_seeds": default_heldout_seeds(),
    }
    if not prereqs_ok:
        payload["gate_reason"] = "Earlier gates are not all passing; retrain is intentionally blocked."
        write_json_atomic(ARTIFACT_PATH, payload)
        write_text_atomic(REPORT_PATH, "# p1p Retrain Eval Matrix\n\n- prerequisites_ok: false\n- retrain blocked until all prior gates pass\n")
        upsert_evidence(EVIDENCE_ID, EXPERIMENT_ID, "retrain_eval_matrix", ARTIFACT_PATH, "Retrain/eval matrix gate with prerequisite checking; currently blocked because not all prior gates pass.", verified=False)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1

    mainline.DATASET_DIR = DATASET_ROOT
    mainline.DATASET_REPO_ID = DATASET_REPO_ID
    mainline.M6_OUTPUT_DIR = OUTPUT_DIR / "p1p_m6_mujoco_mint_train_v2"
    mainline.M7_OUTPUT_DIR = OUTPUT_DIR / "p1p_m7_mujoco_eval_v2"
    train_result = mainline.run_m6_mint_train_gate()
    eval_result = mainline.run_m7_mint_eval_gate(train_result) if train_result.get("passed") else {"passed": False, "error": "train gate failed"}

    payload.update(
        {
            "passed": bool(train_result.get("passed") and eval_result.get("passed")),
            "train_result": train_result,
            "eval_result": eval_result,
            "gate_reason": "Retrain/eval matrix completed after all prior gates passed." if train_result.get("passed") and eval_result.get("passed") else "Retrain/eval matrix ran but one of the stages failed.",
        }
    )
    write_json_atomic(ARTIFACT_PATH, payload)
    lines = [
        "# p1p Retrain Eval Matrix",
        "",
        f"Generated: {payload['generated_at']}",
        f"- prerequisites_ok: {prereqs_ok}",
        f"- train_passed: {train_result.get('passed')}",
        f"- eval_passed: {eval_result.get('passed')}",
        f"- heldout_seeds: {payload['heldout_seeds']}",
    ]
    write_text_atomic(REPORT_PATH, "\n".join(lines).rstrip() + "\n")
    upsert_evidence(EVIDENCE_ID, EXPERIMENT_ID, "retrain_eval_matrix", ARTIFACT_PATH, "Retrain/eval matrix guarded by prerequisite gate checks and executed only on the expanded aligned dataset after prior gates pass.", verified=bool(payload['passed']))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
