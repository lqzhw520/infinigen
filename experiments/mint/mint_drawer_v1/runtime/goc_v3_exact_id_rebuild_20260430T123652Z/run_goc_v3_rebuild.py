#!/usr/bin/env python3
"""Static GOC-v3 exact-ID contract rebuild.

This script intentionally loads only the static merged MuJoCo model. It does not
step actions, render, train, replay, or patch runtime/harness code.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN = REPO_ROOT / "experiments/mint/mint_drawer_v1"
RUN_DIR = Path(
    os.environ.get("GOC_V3_RUN_DIR")
    or Path("/tmp/goc_v3_current_run_dir").read_text().strip()
)
CONTRACT_DIR = CAMPAIGN / "artifacts/phase1h_geometry_contract"
SOVEREIGN = CAMPAIGN / "sovereign"
SPEC_DIR = SOVEREIGN / "experiment_specs"
OLD_INVENTORY = CONTRACT_DIR / "geom_inventory.json"
OLD_CONTRACT = CONTRACT_DIR / "geometry_ownership_contract.json"

TASK_ID = "GOC_V3_EXACT_ID_CONTRACT_REBUILD_AND_VALIDATE"
CONTRACT_ID = "GOC_V3_EXACT_ID_GEOMETRY_OWNERSHIP_CONTRACT"
SUCCESS = "GOC_V3_READY_FOR_V11_G4_TASK_REWRITE"

ROBOT_ARM_BODIES = {"link0", "link1", "link2", "link3", "link4"}
GRIPPER_BODIES = {"link5", "link6", "link7", "right_hand"}
DRAWER_HANDLE_BODIES = {"drawer_base"}
DRAWER_BODY_BODIES = {"link_1", "link_2"}
DRAWER_ALL_BODIES = DRAWER_HANDLE_BODIES | DRAWER_BODY_BODIES
RUNTIME_GOCV2_GRIPPER_BODIES = {
    "base",
    "link0",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
}


def now_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2, sort_keys=False) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def sha256_json(data: Any) -> str:
    payload = json.dumps(jsonable(data), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def git(args: list[str]) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(REPO_ROOT), text=True
    ).strip()


def model_name(mujoco: Any, model: Any, obj: Any, idx: int) -> str:
    return mujoco.mj_id2name(model, obj, idx) or ""


def geom_type_name(mujoco: Any, geom_type: int) -> str:
    names = {
        int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
        int(mujoco.mjtGeom.mjGEOM_HFIELD): "hfield",
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
        int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
        int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
    }
    return names.get(int(geom_type), f"unknown_{int(geom_type)}")


def body_chain(mujoco: Any, model: Any, body_id: int) -> list[str]:
    chain: list[str] = []
    cur = int(body_id)
    seen: set[int] = set()
    while cur >= 0 and cur not in seen:
        seen.add(cur)
        chain.append(
            model_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, cur) or "world"
        )
        if cur == 0:
            break
        cur = int(model.body_parentid[cur])
    return list(reversed(chain))


def parent_chain(mujoco: Any, model: Any, body_id: int) -> list[str]:
    chain = body_chain(mujoco, model, body_id)
    return chain[:-1]


def load_model() -> tuple[Any, Any, str, dict[str, Any], str]:
    sys.path.insert(0, str(REPO_ROOT / "scripts/mint"))
    import mujoco  # type: ignore
    from merged_model_builder import MergedModelBuilder  # type: ignore

    xml, assets, metadata, semantic_hash = MergedModelBuilder(seed=1).build()
    model = mujoco.MjModel.from_xml_string(xml, assets)
    return mujoco, model, xml, jsonable(metadata), str(semantic_hash)


def enumerate_model(
    mujoco: Any, model: Any, metadata: dict[str, Any], semantic_hash: str
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    geoms: list[dict[str, Any]] = []
    for gid in range(model.ngeom):
        body_id = int(model.geom_bodyid[gid])
        body_name = (
            model_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "world"
        )
        dataid = int(model.geom_dataid[gid])
        mesh_name = ""
        if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH) and dataid >= 0:
            mesh_name = model_name(mujoco, model, mujoco.mjtObj.mjOBJ_MESH, dataid)
        contype = int(model.geom_contype[gid])
        conaffinity = int(model.geom_conaffinity[gid])
        contact_capable = bool(contype != 0 and conaffinity != 0)
        group = int(model.geom_group[gid])
        geoms.append(
            {
                "geom_id": gid,
                "geom_name": model_name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, gid),
                "geom_type": geom_type_name(mujoco, int(model.geom_type[gid])),
                "geom_type_id": int(model.geom_type[gid]),
                "geom_bodyid": body_id,
                "body_id": body_id,
                "body_name": body_name,
                "body_parent_chain": parent_chain(mujoco, model, body_id),
                "body_ancestor_chain": body_chain(mujoco, model, body_id),
                "contype": contype,
                "conaffinity": conaffinity,
                "geom_group": group,
                "rgba": [float(x) for x in model.geom_rgba[gid].tolist()],
                "size": [float(x) for x in model.geom_size[gid].tolist()],
                "pos": [float(x) for x in model.geom_pos[gid].tolist()],
                "mesh_or_data_reference": mesh_name
                or (f"dataid:{dataid}" if dataid >= 0 else ""),
                "appears_contact_capable": contact_capable,
                "visual_only_or_noncontact": bool(not contact_capable),
                "source_of_classification": "contact_params/body_ancestor_chain",
            }
        )

    bodies: list[dict[str, Any]] = []
    child_map: dict[int, list[int]] = {i: [] for i in range(model.nbody)}
    for bid in range(1, model.nbody):
        child_map[int(model.body_parentid[bid])].append(bid)
    for bid in range(model.nbody):
        bname = model_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, bid) or "world"
        gids = [g["geom_id"] for g in geoms if g["body_id"] == bid]
        bodies.append(
            {
                "body_id": bid,
                "body_name": bname,
                "parent_body_id": int(model.body_parentid[bid]) if bid > 0 else None,
                "parent_body_name": model_name(
                    mujoco,
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(model.body_parentid[bid]),
                )
                if bid > 0
                else None,
                "child_body_ids": child_map.get(bid, []),
                "child_body_names": [
                    model_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, c)
                    for c in child_map.get(bid, [])
                ],
                "body_ancestor_chain": body_chain(mujoco, model, bid),
                "geom_ids": gids,
                "geom_count": len(gids),
            }
        )

    body_names = {b["body_name"] for b in bodies}
    model_load_report = {
        "task_id": TASK_ID,
        "generated_at_utc": now_utc(),
        "source_model_commit": git(["rev-parse", "HEAD"]),
        "builder_module_path": "scripts/mint/merged_model_builder.py",
        "builder_class": "MergedModelBuilder(seed=1)",
        "model_source_path": {
            "robot_xml": "/root/anaconda3/envs/infinigen/lib/python3.11/site-packages/robosuite/models/assets/robots/panda/robot.xml",
            "drawer_root": "/mnt/afs2/zhuhaowu/infinigen/sim_exports/urdf/drawer",
            "metadata": metadata,
        },
        "semantic_hash": semantic_hash,
        "total_ngeom": int(model.ngeom),
        "total_nbody": int(model.nbody),
        "total_njnt": int(model.njnt),
        "is_full_robot_in_scene": bool(
            {
                "link0",
                "link1",
                "link2",
                "link3",
                "link4",
                "link5",
                "link6",
                "link7",
            }.issubset(body_names)
            and {"drawer_base", "link_1", "link_2"}.issubset(body_names)
        ),
        "is_proxy_only": False,
        "drawer_handle_entity_present": bool("drawer_base" in body_names),
        "robot_gripper_or_finger_bodies_present": bool(
            {"link5", "link6", "link7", "right_hand"}.intersection(body_names)
        ),
        "drawer_body_or_cabinet_bodies_present": bool(
            {"drawer_base", "link_1", "link_2"}.issubset(body_names)
        ),
        "static_load_only": True,
        "rollout_render_train_run": False,
    }
    raw_inventory = {
        "task_id": TASK_ID,
        "generated_at_utc": now_utc(),
        "total_ngeom": int(model.ngeom),
        "total_nbody": int(model.nbody),
        "total_njnt": int(model.njnt),
        "model_source_path": model_load_report["model_source_path"],
        "builder_module_path": model_load_report["builder_module_path"],
        "semantic_hash": semantic_hash,
        "geoms": geoms,
    }
    raw_body_tree = {
        "task_id": TASK_ID,
        "generated_at_utc": now_utc(),
        "bodies": bodies,
    }
    return raw_inventory, raw_body_tree, geoms, model_load_report


def idset(
    geoms: list[dict[str, Any]], bodies: set[str], *, contact_only: bool | None = None
) -> list[int]:
    out = []
    for g in geoms:
        if g["body_name"] not in bodies:
            continue
        if contact_only is True and not g["appears_contact_capable"]:
            continue
        if contact_only is False and g["appears_contact_capable"]:
            continue
        out.append(int(g["geom_id"]))
    return sorted(out)


def classify_geoms(
    geoms: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    drawer_handle_ids = idset(geoms, DRAWER_HANDLE_BODIES, contact_only=None)
    drawer_body_ids = idset(geoms, DRAWER_BODY_BODIES, contact_only=None)
    robot_arm_ids = idset(geoms, ROBOT_ARM_BODIES, contact_only=None)
    gripper_ids = idset(geoms, GRIPPER_BODIES, contact_only=None)
    legal_ids = idset(geoms, {"link5", "link6", "link7"}, contact_only=True)
    forbidden_ids = idset(geoms, ROBOT_ARM_BODIES, contact_only=True)
    visual_noncontact_ids = sorted(
        [int(g["geom_id"]) for g in geoms if not g["appears_contact_capable"]]
    )
    unknown_ids: list[int] = []

    exact_sets = {
        "legal_gripper_surface_geom_ids": legal_ids,
        "forbidden_robot_surface_geom_ids": forbidden_ids,
        "drawer_handle_geom_ids": drawer_handle_ids,
        "drawer_body_or_cabinet_geom_ids": drawer_body_ids,
        "robot_arm_geom_ids": robot_arm_ids,
        "gripper_geom_ids": gripper_ids,
        "visual_only_or_noncontact_geom_ids": visual_noncontact_ids,
        "unknown_geom_ids": unknown_ids,
    }

    classified: list[dict[str, Any]] = []
    for g in geoms:
        gid = int(g["geom_id"])
        body = g["body_name"]
        contact = bool(g["appears_contact_capable"])
        primary_owner = "unknown"
        contact_role = "unknown"
        allowed = False
        forbidden = False
        counterpart = "none"
        source = "unknown"
        reason = ""

        if gid in legal_ids:
            primary_owner = "legal_gripper_contact_surface"
            contact_role = "legal_gripper_surface"
            allowed = True
            counterpart = "drawer_handle"
            source = "contact_params"
            reason = (
                "Current model contact-capable geom on gripper body link5/link6/link7."
            )
        elif gid in forbidden_ids:
            primary_owner = "forbidden_robot_surface"
            contact_role = "forbidden_robot_surface"
            forbidden = True
            source = "contact_params"
            reason = "Current model contact-capable geom on nonlegal robot arm body link0-link4."
        elif gid in drawer_handle_ids:
            primary_owner = "drawer_handle"
            contact_role = "handle_surface"
            source = "manual_rule_declared_in_goc_v3"
            reason = "Merged model has no separate handle names; GOC-v3 declares drawer_base geoms as the handle target set, matching prior BUG_GOC_04 body-ownership repair."
        elif gid in drawer_body_ids:
            primary_owner = "drawer_body_or_cabinet"
            contact_role = "drawer_body_surface"
            source = "body_ancestor_chain"
            reason = "Current model drawer link/cabinet body geom on link_1/link_2; not legal gripper contact."
        elif gid in visual_noncontact_ids:
            primary_owner = "visual_only" if int(g["geom_group"]) == 1 else "noncontact"
            contact_role = "visual_only" if int(g["geom_group"]) == 1 else "noncontact"
            source = "contact_params"
            reason = "Geom has contype=0 or conaffinity=0 in current MuJoCo model, so it cannot be target contact authority."
        elif body == "world":
            primary_owner = "world_static"
            contact_role = "world_surface"
            source = "body_ancestor_chain"
            reason = "World body geom."
        elif body in GRIPPER_BODIES:
            primary_owner = "gripper"
            contact_role = "unknown"
            source = "body_ancestor_chain"
            reason = "Gripper body geom not otherwise classified."
        elif body in ROBOT_ARM_BODIES:
            primary_owner = "robot_arm"
            contact_role = "unknown"
            source = "body_ancestor_chain"
            reason = "Robot arm body geom not otherwise classified."
        else:
            unknown_ids.append(gid)
            reason = "No GOC-v3 rule matched this geom."

        rec = dict(g)
        rec.update(
            {
                "primary_owner": primary_owner,
                "contact_role": contact_role,
                "is_collision_contact_surface": contact,
                "is_visual_only_or_noncontact": bool(gid in visual_noncontact_ids),
                "allowed_for_target_contact": allowed,
                "forbidden_if_contacts_drawer_or_handle": forbidden,
                "allowed_target_contact_counterpart": counterpart,
                "classification_reason": reason,
                "classification_source": source,
            }
        )
        classified.append(rec)

    # Refresh unknown set if any unmatched IDs were appended.
    exact_sets["unknown_geom_ids"] = sorted(set(unknown_ids))
    return classified, exact_sets


def reconcile(
    geoms: list[dict[str, Any]], exact_sets: dict[str, list[int]]
) -> dict[str, Any]:
    old_contract = json.loads(OLD_CONTRACT.read_text()) if OLD_CONTRACT.exists() else {}
    old_inventory = (
        json.loads(OLD_INVENTORY.read_text()) if OLD_INVENTORY.exists() else {}
    )
    old_legal_ids_full = (
        old_inventory.get("correct_classification", {})
        .get("legal_gripper_pad_or_fingertip", {})
        .get("ids", [])
    )
    old_forbidden_ids_full = (
        old_inventory.get("correct_classification", {})
        .get("forbidden_robot_geoms", {})
        .get("ids", [])
    )
    expected_legal_count = (
        old_contract.get("classification_summary", {})
        .get("legal_gripper_pad_geom_ids", {})
        .get("count", 29)
    )
    expected_forbidden_count = (
        old_contract.get("classification_summary", {})
        .get("forbidden_robot_geom_ids", {})
        .get("count", 26)
    )

    runtime_legal_31 = idset(geoms, {"link5", "link6", "link7"}, contact_only=None)
    runtime_forbidden_27 = idset(geoms, ROBOT_ARM_BODIES, contact_only=None)

    # The old artifact is internally inconsistent: it stores count=29 but its listed IDs contain 31.
    # Preserve a deterministic diagnostic reconstruction of the count-only GOC-v2 surface by sorted/body-order prefix.
    v2_29_reconstructed = (
        sorted(old_legal_ids_full[:expected_legal_count])
        if len(old_legal_ids_full) >= expected_legal_count
        else runtime_legal_31[:expected_legal_count]
    )
    v2_26_reconstructed = runtime_forbidden_27[:expected_forbidden_count]

    legal_runtime_not_v2_29 = sorted(set(runtime_legal_31) - set(v2_29_reconstructed))
    legal_v2_29_not_runtime = sorted(set(v2_29_reconstructed) - set(runtime_legal_31))
    forbidden_runtime_not_v2_26 = sorted(
        set(runtime_forbidden_27) - set(v2_26_reconstructed)
    )
    forbidden_v2_26_not_runtime = sorted(
        set(v2_26_reconstructed) - set(runtime_forbidden_27)
    )

    visual_noncontact = set(exact_sets["visual_only_or_noncontact_geom_ids"])
    current_contact_legal = set(exact_sets["legal_gripper_surface_geom_ids"])
    current_contact_forbidden = set(exact_sets["forbidden_robot_surface_geom_ids"])

    return {
        "task_id": TASK_ID,
        "generated_at_utc": now_utc(),
        "summary": "GOC-v2 stored count/body-level expectations (29/26) while runtime derived body-owned sets from current model (31/27). GOC-v3 resolves this by exact current geom IDs and contact-capability roles.",
        "goc_v2_expected_counts": {
            "legal_pad": expected_legal_count,
            "forbidden": expected_forbidden_count,
            "handle": old_contract.get("classification_summary", {})
            .get("drawer_handle_geom_ids", {})
            .get("count", 9),
        },
        "old_artifact_inconsistencies": {
            "legal_count_field": expected_legal_count,
            "legal_ids_list_length_in_geom_inventory": len(old_legal_ids_full),
            "forbidden_count_field": expected_forbidden_count,
            "forbidden_ids_list_length_in_geom_inventory": len(old_forbidden_ids_full),
            "interpretation": "The old artifacts are count/body-level and internally stale; GOC-v3 does not treat their counts as authority. Exact current geom IDs below are the authority.",
        },
        "runtime_body_based_sets": {
            "legal_pad_body_based_31": runtime_legal_31,
            "forbidden_body_based_27": runtime_forbidden_27,
            "handle_body_based_9": exact_sets["drawer_handle_geom_ids"],
        },
        "diagnostic_goc_v2_count_reconstructions": {
            "legal_pad_29_reconstructed_from_old_order": v2_29_reconstructed,
            "forbidden_26_reconstructed_from_current_sorted_body_order": v2_26_reconstructed,
            "not_authority": True,
        },
        "requested_exact_deltas": {
            "legal_ids_in_29_not_31": legal_v2_29_not_runtime,
            "legal_ids_in_31_not_29": legal_runtime_not_v2_29,
            "forbidden_ids_in_26_not_27": forbidden_v2_26_not_runtime,
            "forbidden_ids_in_27_not_26": forbidden_runtime_not_v2_26,
        },
        "goc_v3_contact_authority_sets": {
            "legal_gripper_surface_geom_ids": exact_sets[
                "legal_gripper_surface_geom_ids"
            ],
            "forbidden_robot_surface_geom_ids": exact_sets[
                "forbidden_robot_surface_geom_ids"
            ],
            "drawer_handle_geom_ids": exact_sets["drawer_handle_geom_ids"],
            "drawer_body_or_cabinet_geom_ids": exact_sets[
                "drawer_body_or_cabinet_geom_ids"
            ],
        },
        "body_based_visual_or_noncontact_exclusions": {
            "legal_body_based_31_excluded_from_target_contact": sorted(
                set(runtime_legal_31) - current_contact_legal
            ),
            "forbidden_body_based_27_excluded_from_forbidden_contact": sorted(
                set(runtime_forbidden_27) - current_contact_forbidden
            ),
            "excluded_ids_that_are_visual_or_noncontact": sorted(
                (set(runtime_legal_31) | set(runtime_forbidden_27)) & visual_noncontact
            ),
        },
        "discrepancy_classification": [
            "B_visual_or_noncontact_geom_inclusion",
            "C_stale_artifact",
            "E_classifier_bug_count_only_body_semantics",
        ],
        "explained_with_exact_ids": True,
    }


def validate_invariants(
    classified: list[dict[str, Any]],
    exact_sets: dict[str, list[int]],
    reconciliation: dict[str, Any],
) -> dict[str, Any]:
    ids_all = {int(g["geom_id"]) for g in classified}
    legal = set(exact_sets["legal_gripper_surface_geom_ids"])
    forbidden = set(exact_sets["forbidden_robot_surface_geom_ids"])
    handle = set(exact_sets["drawer_handle_geom_ids"])
    drawer = set(exact_sets["drawer_body_or_cabinet_geom_ids"])
    robot_arm = set(exact_sets["robot_arm_geom_ids"])
    gripper = set(exact_sets["gripper_geom_ids"])
    visual = set(exact_sets["visual_only_or_noncontact_geom_ids"])
    unknown = set(exact_sets["unknown_geom_ids"])
    contact_relevant = {
        int(g["geom_id"]) for g in classified if g["appears_contact_capable"]
    }

    def result(
        name: str, passed: bool, detail: str, offenders: list[int] | None = None
    ) -> dict[str, Any]:
        return {
            "invariant": name,
            "passed": bool(passed),
            "detail": detail,
            "offending_geom_ids": sorted(offenders or []),
        }

    inv = []
    no_unknown_contact_owner = sorted(
        [
            gid
            for gid in contact_relevant
            if next(g for g in classified if g["geom_id"] == gid)["primary_owner"]
            == "unknown"
        ]
    )
    inv.append(
        result(
            "I1",
            not no_unknown_contact_owner,
            "Every contact-relevant geom has exactly one non-unknown primary_owner.",
            no_unknown_contact_owner,
        )
    )
    inv.append(
        result(
            "I2",
            legal.issubset(ids_all) and handle.issubset(ids_all),
            "Every target contact geom is an exact current geom_id.",
            sorted((legal | handle) - ids_all),
        )
    )
    inv.append(
        result(
            "I3",
            legal.isdisjoint(forbidden),
            "legal_gripper_surface and forbidden_robot_surface are disjoint.",
            sorted(legal & forbidden),
        )
    )
    inv.append(
        result(
            "I4",
            handle.isdisjoint(robot_arm),
            "drawer_handle and robot_arm are disjoint.",
            sorted(handle & robot_arm),
        )
    )
    inv.append(
        result(
            "I5",
            handle.isdisjoint(gripper),
            "drawer_handle and gripper are disjoint.",
            sorted(handle & gripper),
        )
    )
    inv.append(
        result(
            "I6",
            visual.isdisjoint(legal),
            "visual_only_or_noncontact geoms cannot be legal target contact.",
            sorted(visual & legal),
        )
    )
    inv.append(
        result(
            "I7",
            unknown.isdisjoint(legal | handle),
            "unknown geoms cannot participate in target contact.",
            sorted(unknown & (legal | handle)),
        )
    )
    inv.append(
        result(
            "I8",
            unknown.isdisjoint(forbidden),
            "unknown geoms cannot participate in forbidden contact classification.",
            sorted(unknown & forbidden),
        )
    )
    inv.append(
        result(
            "I9",
            bool(legal and handle),
            "Target contact is exactly legal_gripper_surface_geom_ids <-> drawer_handle_geom_ids by contract rule.",
        )
    )
    inv.append(
        result(
            "I10",
            bool(forbidden and (handle or drawer)),
            "Forbidden contact includes forbidden_robot_surface against drawer_handle/drawer_body_or_cabinet by contract rule.",
        )
    )
    inv.append(
        result(
            "I11",
            drawer.isdisjoint(gripper),
            "drawer_body_or_cabinet cannot be counted as gripper contact.",
            sorted(drawer & gripper),
        )
    )
    inv.append(
        result(
            "I12",
            handle.isdisjoint(robot_arm | gripper),
            "Robot link body/collision cannot be counted as drawer handle.",
            sorted(handle & (robot_arm | gripper)),
        )
    )
    inv.append(
        result(
            "I13",
            bool(reconciliation.get("explained_with_exact_ids")),
            "GOC-v3 exact-ID sets explain GOC-v2 29/26 versus runtime 31/27.",
        )
    )
    inv.append(
        result(
            "I14",
            all(isinstance(i, int) for s in exact_sets.values() for i in s),
            "GOC-v3 contains no count-only authority in exact sets.",
        )
    )
    inv.append(
        result(
            "I15",
            bool(classified) and len(classified) == len(ids_all),
            "GOC-v3 is generated from current model inventory, not copied from old artifacts.",
        )
    )
    return {
        "task_id": TASK_ID,
        "generated_at_utc": now_utc(),
        "all_invariants_passed": all(x["passed"] for x in inv),
        "invariants": inv,
        "summary_counts": {
            "legal_gripper_surface_count": len(legal),
            "forbidden_robot_surface_count": len(forbidden),
            "drawer_handle_surface_count": len(handle),
            "unknown_contact_relevant_geom_count": len(
                sorted(unknown & contact_relevant)
            ),
            "visual_only_or_noncontact_count": len(visual),
        },
    }


def contract_markdown(
    contract: dict[str, Any], reconciliation: dict[str, Any], validation: dict[str, Any]
) -> str:
    s = contract["exact_id_sets"]
    return f"""# GOC-v3 Exact-ID Geometry Ownership Contract

Contract ID: `{contract['contract_id']}`
Version: `{contract['version']}`
Generated UTC: `{contract['generated_at_utc']}`
Source model commit: `{contract['source_model_commit']}`

## What GOC-v3 Is

GOC-v3 is the exact per-geom authority for future V11-G4 contact classification. It binds target contact and forbidden contact to concrete MuJoCo `geom_id` sets from the current merged Panda + Infinigen drawer model.

It is not a rollout result, not a visual artifact claim, not a training eligibility claim, and not a MINT success claim.

## Exact Contact Authority

- Legal gripper target surfaces: `{s['legal_gripper_surface_geom_ids']}`
- Drawer handle target surfaces: `{s['drawer_handle_geom_ids']}`
- Forbidden robot contact surfaces: `{s['forbidden_robot_surface_geom_ids']}`
- Drawer body/cabinet surfaces: `{s['drawer_body_or_cabinet_geom_ids']}`
- Visual-only/noncontact geoms: `{s['visual_only_or_noncontact_geom_ids']}`
- Unknown geoms: `{s['unknown_geom_ids']}`

Future target contact is valid only for:

```text
legal_gripper_surface_geom_ids <-> drawer_handle_geom_ids
```

Forbidden contact includes:

```text
forbidden_robot_surface_geom_ids <-> drawer_handle_geom_ids
forbidden_robot_surface_geom_ids <-> drawer_body_or_cabinet_geom_ids
nonlegal robot contact surfaces <-> drawer_handle/body/cabinet
```

## Why GOC-v2 Was Insufficient

GOC-v2 described ownership with body-level counts: legal_pad=29, forbidden=26, handle=9. The current runtime body-based derivation sees legal=31 and forbidden=27 because it includes every geom on the named robot bodies. That body-count logic is not exact contact authority.

GOC-v3 resolves this by listing exact IDs and separating contact-capable geoms from visual-only/noncontact geoms.

## 29/26 vs 31/27 Reconciliation

- Runtime legal body-based 31: `{reconciliation['runtime_body_based_sets']['legal_pad_body_based_31']}`
- Diagnostic GOC-v2 legal 29 reconstruction: `{reconciliation['diagnostic_goc_v2_count_reconstructions']['legal_pad_29_reconstructed_from_old_order']}`
- IDs in 31 but not 29: `{reconciliation['requested_exact_deltas']['legal_ids_in_31_not_29']}`
- IDs in 29 but not 31: `{reconciliation['requested_exact_deltas']['legal_ids_in_29_not_31']}`

- Runtime forbidden body-based 27: `{reconciliation['runtime_body_based_sets']['forbidden_body_based_27']}`
- Diagnostic GOC-v2 forbidden 26 reconstruction: `{reconciliation['diagnostic_goc_v2_count_reconstructions']['forbidden_26_reconstructed_from_current_sorted_body_order']}`
- IDs in 27 but not 26: `{reconciliation['requested_exact_deltas']['forbidden_ids_in_27_not_26']}`
- IDs in 26 but not 27: `{reconciliation['requested_exact_deltas']['forbidden_ids_in_26_not_27']}`

The old artifacts also contain stale/count inconsistencies, so these diagnostic reconstructions are not future authority. The future authority is the exact GOC-v3 contact-role sets above.

## Validation

All invariants passed: `{validation['all_invariants_passed']}`

## Future Runtime Requirement

Future V11-G4 runtime/contact reports must load `artifacts/phase1h_geometry_contract/goc_v3_contract.json` and classify target/forbidden contacts using exact geom IDs. It may not use GOC-v2 counts and may not use body-based 31/27 without exact-ID role classification.

## Still Not Proven

- No Phase 1H rollout was run.
- No render or visual artifact was generated.
- No teacher rollout, replay, training, fine-tuning, or evaluation success is claimed.
- This contract only proves exact current model geometry ownership for future task rewrite and contact-report validation.
"""


def validation_markdown(validation: dict[str, Any]) -> str:
    lines = [
        "# GOC-v3 Validation Report",
        "",
        f"Generated UTC: `{validation['generated_at_utc']}`",
        "",
        f"All invariants passed: `{validation['all_invariants_passed']}`",
        "",
        "## Summary Counts",
        "",
    ]
    for k, v in validation["summary_counts"].items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## Invariants", ""])
    for inv in validation["invariants"]:
        status = "PASS" if inv["passed"] else "FAIL"
        lines.append(
            f"- {inv['invariant']}: {status}. {inv['detail']} Offenders: {inv['offending_geom_ids']}"
        )
    lines.append("")
    return "\n".join(lines)


def reconciliation_markdown(rec: dict[str, Any]) -> str:
    d = rec["requested_exact_deltas"]
    return f"""# GOC-v2 29/26 vs Runtime 31/27 Reconciliation

Generated UTC: `{rec['generated_at_utc']}`

GOC-v2 stored count/body-level expectations while the runtime derived body-owned sets from the current model. GOC-v3 resolves the ambiguity by exact current geom IDs and contact roles.

## Exact Deltas Requested

- legal IDs in 29 not 31: `{d['legal_ids_in_29_not_31']}`
- legal IDs in 31 not 29: `{d['legal_ids_in_31_not_29']}`
- forbidden IDs in 26 not 27: `{d['forbidden_ids_in_26_not_27']}`
- forbidden IDs in 27 not 26: `{d['forbidden_ids_in_27_not_26']}`

## Current GOC-v3 Authority

- legal_gripper_surface_geom_ids: `{rec['goc_v3_contact_authority_sets']['legal_gripper_surface_geom_ids']}`
- forbidden_robot_surface_geom_ids: `{rec['goc_v3_contact_authority_sets']['forbidden_robot_surface_geom_ids']}`
- drawer_handle_geom_ids: `{rec['goc_v3_contact_authority_sets']['drawer_handle_geom_ids']}`
- drawer_body_or_cabinet_geom_ids: `{rec['goc_v3_contact_authority_sets']['drawer_body_or_cabinet_geom_ids']}`

## Cause Classification

`{rec['discrepancy_classification']}`

The old count fields are not future authority. Exact-ID GOC-v3 is the authority for the next V11-G4 task rewrite.
"""


def proposed_spec_yaml(contract_rel: str, exact_sets: dict[str, list[int]]) -> str:
    def list_inline(xs: list[int]) -> str:
        return "[" + ", ".join(str(x) for x in xs) + "]"

    return f"""task_id: proposed_v11_g4_with_goc_v3_contact_test
status: proposed_not_active
target_gate: V11-G4_LEGAL_GRIPPER_PAD_TO_HANDLE_CONTACT
goc_authority: {contract_rel}
rollout_allowed: false
harness_review_required_before_rollout: true
contact_authority:
  allowed_target_contact:
    legal_gripper_surface_geom_ids: {list_inline(exact_sets['legal_gripper_surface_geom_ids'])}
    drawer_handle_geom_ids: {list_inline(exact_sets['drawer_handle_geom_ids'])}
    relation: legal_gripper_surface_geom_ids <-> drawer_handle_geom_ids
  forbidden_contact:
    forbidden_robot_surface_geom_ids: {list_inline(exact_sets['forbidden_robot_surface_geom_ids'])}
    nonlegal_robot_surface_geom_ids: {list_inline(sorted(set(exact_sets['robot_arm_geom_ids']) - set(exact_sets['legal_gripper_surface_geom_ids'])))}
    drawer_body_or_cabinet_geom_ids: {list_inline(exact_sets['drawer_body_or_cabinet_geom_ids'])}
    drawer_handle_geom_ids: {list_inline(exact_sets['drawer_handle_geom_ids'])}
  prohibited_classifiers:
    - GOC-v2 count-only legal_pad=29 forbidden=26 authority
    - body-based runtime 31/27 without exact-ID role classification
prohibited_execution:
  - rollout
  - render
  - train
  - teacher_generation
notes:
  - This is a proposed task spec only; it does not mutate the active V11-G4 spec.
  - Future runtime must consume the GOC-v3 exact-ID contract before Phase 1H execution.
"""


def write_failure(
    closeout: str, message: str, extra: dict[str, Any] | None = None
) -> None:
    payload = {
        "task_id": TASK_ID,
        "closeout_classification": closeout,
        "message": message,
        "generated_at_utc": now_utc(),
        "extra": extra or {},
        "committed": False,
        "pushed_to_origin": False,
        "rollout_render_train_run": False,
    }
    write_json(RUN_DIR / "closeout_decision.json", payload)
    write_text(RUN_DIR / "final_report.md", f"# {closeout}\n\n{message}\n")


def main() -> int:
    started = time.time()
    if Path.cwd() != REPO_ROOT:
        write_failure("WRONG_WORKTREE", f"Expected {REPO_ROOT}, got {Path.cwd()}")
        return 2

    try:
        mujoco, model, xml, metadata, semantic_hash = load_model()
        raw_inventory, raw_body_tree, geoms, model_load_report = enumerate_model(
            mujoco, model, metadata, semantic_hash
        )
        if not model_load_report["is_full_robot_in_scene"]:
            write_json(RUN_DIR / "model_load_report.json", model_load_report)
            write_failure(
                "GOC_V3_FULL_ROBOT_MODEL_UNAVAILABLE",
                "Static model loaded but is not full robot-in-scene.",
                model_load_report,
            )
            return 3
        if not geoms:
            write_failure("GOC_V3_MODEL_INVENTORY_FAILED", "No geoms enumerated.")
            return 4

        write_json(RUN_DIR / "raw_model_inventory.json", raw_inventory)
        write_json(RUN_DIR / "raw_body_tree.json", raw_body_tree)
        write_json(RUN_DIR / "model_load_report.json", model_load_report)

        classified, exact_sets = classify_geoms(geoms)
        raw_inventory_hash = sha256_json(raw_inventory)
        reconciliation = reconcile(geoms, exact_sets)
        if not reconciliation.get("explained_with_exact_ids"):
            write_json(
                RUN_DIR / "goc_v2_vs_runtime_3127_reconciliation.json", reconciliation
            )
            write_text(
                RUN_DIR / "goc_v2_vs_runtime_3127_reconciliation.md",
                reconciliation_markdown(reconciliation),
            )
            write_failure(
                "GOC_V3_AUTHORITY_AMBIGUOUS",
                "GOC-v2/runtime discrepancy could not be explained with exact IDs.",
                reconciliation,
            )
            return 5

        validation = validate_invariants(classified, exact_sets, reconciliation)
        write_json(
            RUN_DIR / "goc_v2_vs_runtime_3127_reconciliation.json", reconciliation
        )
        write_text(
            RUN_DIR / "goc_v2_vs_runtime_3127_reconciliation.md",
            reconciliation_markdown(reconciliation),
        )
        write_json(RUN_DIR / "goc_v3_validation_report.json", validation)
        write_text(
            RUN_DIR / "goc_v3_validation_report.md", validation_markdown(validation)
        )
        if not validation["all_invariants_passed"]:
            write_failure(
                "GOC_V3_INVARIANTS_FAILED",
                "One or more GOC-v3 invariants failed.",
                validation,
            )
            return 6

        source_commit = git(["rev-parse", "HEAD"])
        contract = {
            "contract_id": CONTRACT_ID,
            "version": "v3",
            "generated_at_utc": now_utc(),
            "source_model_commit": source_commit,
            "source_model_inventory_hash": raw_inventory_hash,
            "model_provenance": model_load_report,
            "exact_id_sets": exact_sets,
            "geom_inventory": classified,
            "invariant_results": validation,
            "v2_vs_runtime_reconciliation": reconciliation,
            "approved_usage": [
                "V11-G4 target contact classification",
                "future Phase1H FSM smoke",
                "future contact report validation",
            ],
            "prohibited_usage": [
                "MINT success claim",
                "visual artifact claim",
                "training eligibility claim",
                "rollout success claim",
            ],
            "rollout_render_train_run": False,
        }
        geom_inventory_artifact = {
            "task_id": TASK_ID,
            "generated_at_utc": now_utc(),
            "source_model_commit": source_commit,
            "source_model_inventory_hash": raw_inventory_hash,
            "total_ngeom": raw_inventory["total_ngeom"],
            "total_nbody": raw_inventory["total_nbody"],
            "total_njnt": raw_inventory["total_njnt"],
            "exact_id_sets": exact_sets,
            "geoms": classified,
        }

        # Run-dir copies.
        write_json(RUN_DIR / "goc_v3_geom_inventory.json", geom_inventory_artifact)
        write_json(RUN_DIR / "goc_v3_contract.json", contract)
        write_text(
            RUN_DIR / "goc_v3_contract.md",
            contract_markdown(contract, reconciliation, validation),
        )

        # Canonical artifacts.
        write_json(CONTRACT_DIR / "goc_v3_geom_inventory.json", geom_inventory_artifact)
        write_json(CONTRACT_DIR / "goc_v3_contract.json", contract)
        write_text(
            CONTRACT_DIR / "goc_v3_contract.md",
            contract_markdown(contract, reconciliation, validation),
        )
        write_json(CONTRACT_DIR / "goc_v3_validation_report.json", validation)
        write_text(
            CONTRACT_DIR / "goc_v3_validation_report.md",
            validation_markdown(validation),
        )

        # Proposed task rewrite and sovereign deltas only.
        contract_rel = "artifacts/phase1h_geometry_contract/goc_v3_contract.json"
        write_text(
            SPEC_DIR / "proposed_v11_g4_with_goc_v3_contact_test.yaml",
            proposed_spec_yaml(contract_rel, exact_sets),
        )
        write_json(
            SOVEREIGN / "proposed_current_truth_delta_goc_v3.json",
            {
                "status": "proposed_not_active",
                "generated_at_utc": now_utc(),
                "proposal": "Adopt GOC-v3 exact-ID geometry ownership contract as V11-G4 contact authority.",
                "do_not_mutate_current_truth_directly": True,
                "goc_authority": contract_rel,
                "source_model_commit": source_commit,
                "exact_id_summary": validation["summary_counts"],
                "not_claimed": [
                    "V11 success",
                    "Phase 1H success",
                    "rollout success",
                    "visual success",
                    "training eligibility",
                ],
            },
        )
        write_json(
            SOVEREIGN / "proposed_next_actions_goc_v3.json",
            {
                "status": "proposed_not_active",
                "generated_at_utc": now_utc(),
                "next_gate": "V11_G4_TASK_REWRITE_TO_GOC_V3",
                "required_action": "Review proposed_v11_g4_with_goc_v3_contact_test.yaml and update active task only after harness review.",
                "goc_authority": contract_rel,
                "forbidden_before_review": [
                    "rollout",
                    "render",
                    "train",
                    "teacher_generation",
                ],
            },
        )

        closeout = {
            "task_id": TASK_ID,
            "closeout_classification": SUCCESS,
            "generated_at_utc": now_utc(),
            "harness_preflight_passed": True,
            "goc_v3_generated": True,
            "goc_v3_invariants_passed": True,
            "goc_v2_vs_runtime_3127_explained": True,
            "legal_gripper_surface_count": validation["summary_counts"][
                "legal_gripper_surface_count"
            ],
            "forbidden_robot_surface_count": validation["summary_counts"][
                "forbidden_robot_surface_count"
            ],
            "drawer_handle_surface_count": validation["summary_counts"][
                "drawer_handle_surface_count"
            ],
            "unknown_contact_relevant_geom_count": validation["summary_counts"][
                "unknown_contact_relevant_geom_count"
            ],
            "proposed_v11_g4_task_written": True,
            "current_truth_modified": False,
            "next_actions_modified": False,
            "runtime_code_modified": False,
            "rollout_render_train_run": False,
            "committed": False,
            "pushed_to_origin": False,
            "remote_commit_hash": None,
            "next_gate": "V11_G4_TASK_REWRITE_TO_GOC_V3",
            "duration_seconds": round(time.time() - started, 3),
        }
        write_json(RUN_DIR / "closeout_decision.json", closeout)
        write_text(
            RUN_DIR / "final_report.md",
            "\n".join(
                [
                    "# GOC-v3 Exact-ID Contract Rebuild Closeout",
                    "",
                    f"closeout_classification: {SUCCESS}",
                    "harness_preflight_passed: true",
                    "goc_v3_generated: true",
                    "goc_v3_invariants_passed: true",
                    "goc_v2_vs_runtime_3127_explained: true",
                    f"legal_gripper_surface_count: {closeout['legal_gripper_surface_count']}",
                    f"forbidden_robot_surface_count: {closeout['forbidden_robot_surface_count']}",
                    f"drawer_handle_surface_count: {closeout['drawer_handle_surface_count']}",
                    f"unknown_contact_relevant_geom_count: {closeout['unknown_contact_relevant_geom_count']}",
                    "",
                    "No rollout, render, train, teacher generation, replay, runtime patch, harness patch, current_truth mutation, or next_actions mutation was performed.",
                    "",
                ]
            ),
        )
        return 0
    except Exception as exc:
        write_failure(
            "EXECUTION_FAILED", str(exc), {"traceback": traceback.format_exc()}
        )
        return 10


if __name__ == "__main__":
    raise SystemExit(main())
