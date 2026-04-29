#!/usr/bin/env python3
"""Phase 1H contact report smoke test — runs on A800 remote."""
import json
import traceback
import sys
import os

REMOTE_ROOT = "/mnt/afs2/zhuhaowu/infinigen"
OUTPUT_PATH = f"{REMOTE_ROOT}/runs/phase1h_execution/contact_report_smoke.json"

def run_smoke():
    os.chdir(f"{REMOTE_ROOT}/scripts/mint")
    sys.path.insert(0, f"{REMOTE_ROOT}/scripts/mint")

    from drawer_robot_env_mujoco import DrawerRobotEnvMuJoCoLibero
    from merged_model_builder import MergedModelBuilder
    DrawerRobotEnvMuJoCoLibero._MERGED_BUILDER_CLASS = MergedModelBuilder

    result = {
        "timestamp_utc": "2026-04-28T12:20:00Z",
        "phase": "1H",
        "test": "contact_report_smoke",
        "smoke_passed": False,
        "schema_keys_found": [],
        "schema_keys_missing": [],
        "report_after_reset": None,
        "report_after_step": None,
        "gripper_geom_ids_found": [],
        "pad_geom_ids_found": [],
        "error": None,
        "traceback": None,
    }

    try:
        env = DrawerRobotEnvMuJoCoLibero(seed=1, image_size=64, max_steps=10)
        result["env_instantiated"] = True

        obs = env.reset()
        result["env_reset"] = True

        report_reset = dict(getattr(env, "_last_full_robot_contact_report", {}))
        result["report_after_reset"] = {k: repr(v) for k, v in report_reset.items()}

        expected_keys = [
            "has_any_contact", "has_target_contact", "has_forbidden_contact",
            "target_contact_pairs", "forbidden_contact_pairs", "all_contact_pairs",
            "legal_gripper_geom_names", "forbidden_robot_geom_names",
            "handle_geom_names", "drawer_geom_names", "reason",
            "target_handle_contact", "target_handle_contact_count",
            "target_handle_contact_force_n", "target_handle_contact_min_dist_m",
            "target_handle_geom_ids", "gripper_contact_geom_ids",
        ]

        found_keys = [k for k in expected_keys if k in report_reset]
        missing_keys = [k for k in expected_keys if k not in report_reset]
        result["schema_keys_found"] = found_keys
        result["schema_keys_missing"] = missing_keys

        gripper_ids = list(getattr(env, "_gripper_contact_geom_ids", []))
        pad_ids = list(getattr(env, "_gripper_pad_geom_ids", []))
        result["gripper_geom_ids_found"] = gripper_ids
        result["pad_geom_ids_found"] = pad_ids

        gripper_names = []
        for gid in gripper_ids:
            try:
                name = str(env.model.geom(gid).name or "")
                gripper_names.append(name)
            except Exception:
                pass
        result["gripper_geom_names"] = gripper_names

        import numpy as np
        action = np.zeros(8, dtype=np.float32)
        obs, reward, done, info = env.step(action)
        result["step_count"] = int(env._step_count)
        result["env_step"] = True

        report_step = dict(getattr(env, "_last_full_robot_contact_report", {}))
        result["report_after_step"] = {k: repr(v) for k, v in report_step.items()}
        result["info_target_handle_contact"] = bool(info.get("target_handle_contact", False))
        result["info_has_any_contact"] = bool(info.get("has_any_contact", False))
        result["info_has_forbidden_contact"] = bool(info.get("has_forbidden_contact", False))
        result["info_legal_gripper_geom_names"] = list(info.get("legal_gripper_geom_names", []))
        result["info_forbidden_robot_geom_names"] = list(info.get("forbidden_robot_geom_names", []))

        smoke_passed = bool(
            len(missing_keys) == 0
            and len(found_keys) == len(expected_keys)
            and "has_any_contact" in report_step
            and len(gripper_ids) > 0
        )
        result["smoke_passed"] = smoke_passed

        if len(gripper_ids) == 0:
            result["smoke_failure_reason"] = "No gripper contact geoms discovered"
        elif len(missing_keys) > 0:
            result["smoke_failure_reason"] = f"Missing schema keys: {missing_keys}"
        else:
            result["smoke_failure_reason"] = None

        env.close()

    except Exception as e:
        result["smoke_passed"] = False
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))
    print(f"Output: {OUTPUT_PATH}")
    return 0 if result["smoke_passed"] else 1


if __name__ == "__main__":
    sys.exit(run_smoke())
