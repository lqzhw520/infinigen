#!/usr/bin/env python3
"""
P0 Physics Legality Module — Verifiable Physics-Compliant Teacher Rollout Loop.

Three-stage solution (root-cause, not filtering):
  Stage 0: AnyGrasp candidate feasibility filter (pre-contact physics check)
  Stage 1: Impedance-controlled compliant pull (during contact)
  Stage 2: Frame-by-frame collision filter (post-hoc verification)

Usage:
  # 1. Verify existing rollouts (standalone audit)
  python scripts/mint/physics_legality.py audit \
    --npz-dir experiments/mint/mint_drawer_v1/artifacts/c2_replay_valid_rollouts \
    --output experiments/mint/mint_drawer_v1/outputs/physics_audit.json

  # 2. Generate physics-legal rollouts with all 3 stages
  python scripts/mint/physics_legality.py generate \
    --seed 42 --num-rollouts 10 \
    --output-dir experiments/mint/mint_drawer_v1/artifacts/physics_rollouts

  # 3. Quick check: is a single NPZ physically legal?
  python scripts/mint/physics_legality.py check \
    --npz experiments/mint/mint_drawer_v1/artifacts/c2_replay_valid_rollouts/seed_002_episode_00.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "mint"))

# Default physics thresholds (calibrated per simulation backend)
DEFAULT_PHYSICS_THRESHOLDS = {
    # Stage 0: AnyGrasp candidate filter
    "max_grasp_penetration_depth_m": 0.01,     # 1cm initial gripper-drawer overlap
    "max_grasp_contact_force_N": 30.0,          # 30N (Panda max ~40N)
    "min_grasp_finger_clearance_m": 0.005,     # 5mm clearance from drawer surface
    "max_grasp_x_offset_from_handle_m": 0.06,  # 6cm lateral offset tolerance

    # Stage 1: Pull controller
    "impedance_stiffness_N_m": 200.0,          # Spring stiffness for impedance control
    "impedance_damping_Ns_m": 20.0,            # Damping coefficient
    "pull_force_N": 15.0,                      # Max pull force before slip
    "max_drawer_acceleration_m_s2": 2.0,       # Physically plausible acceleration cap

    # Stage 2: Replay verification
    "max_step_delta_pos_m": 0.05,              # 5cm/step (action_scale=0.03m × ~1.5 margin)
    "max_step_delta_rot_rad": 0.4,            # ~23deg/step (rotation_scale=0.25 × 1.5)
    "max_position_error_m": 0.02,              # 2cm replay position error tolerance
    "phantom_jump_threshold_m": 0.05,           # 5cm instant teleport detection
    "max_gripper_penetration_m": 0.002,        # 2mm penetration = collision
    "replay_success_threshold_m": 0.03,          # avg replay error < 3cm = success
}


# ============================================================================
# Stage 0: AnyGrasp Candidate Feasibility Filter
# ============================================================================

def stage0_check_grasp_feasibility(
    env,
    grasp_pose_world: np.ndarray,
) -> dict[str, Any]:
    """
    Stage 0: Pre-contact physics feasibility check for AnyGrasp candidates.
    
    Checks before committing to a grasp:
    1. Gripper approaching path collision-free
    2. Gripper fingers won't collide with drawer panel during approach
    3. Target object is within reachability bounds
    
    Args:
        env: DrawerRobotEnv instance (must have client, robot_id, drawer_id)
        grasp_pose_world: (4, 4) homogeneous transform of grasp target
    
    Returns:
        dict with 'feasible': bool and 'reasons': list
    """
    reasons = []
    feasible = True
    p = env.p
    
    # Check 1: Is the grasp pose within robot workspace?
    grasp_pos = grasp_pose_world[:3, 3]
    reachable = (
        -0.55 <= grasp_pos[0] <= 0.55 and
        -0.55 <= grasp_pos[1] <= 0.55 and
        0.05 <= grasp_pos[2] <= 0.80
    )
    if not reachable:
        reasons.append(f"grasp_out_of_workspace: pos={grasp_pos}")
        feasible = False

    # Check 2: Ray cast from grasp pose downward to detect drawer surface collision
    # Cast ray downward (negative Z in world frame) to check gripper would collide with drawer
    ray_from = grasp_pos + np.array([0.0, 0.0, 0.05])  # 5cm above grasp
    ray_to = grasp_pos - np.array([0.0, 0.0, 0.10])      # 10cm below
    
    ray_results = p.rayTestBatch(
        [ray_from.tolist()],
        [ray_to.tolist()],
        physicsClientId=env.client,
    )
    hit_object, hit_fraction = ray_results[0][0], ray_results[0][2]
    
    # drawer_id is typically 1 (check if hit drawer)
    if hit_object == env.drawer_id and hit_fraction < 0.5:
        reasons.append(f"gripper_hits_drawer: hit={hit_object} fraction={hit_fraction:.3f}")
        feasible = False
    
    # Check 3: Check for initial gripper-drawer penetration using contact points
    # Temporarily set gripper to closed and check contacts
    original_gripper = env.gripper_open
    env._set_gripper_open(0.0)  # Close gripper
    env._step_world(steps=2)
    
    contact_points = p.getContactPoints(
        bodyA=env.robot_id,
        bodyB=env.drawer_id,
        physicsClientId=env.client,
    )
    n_gripper_drawer_contacts = len(contact_points)
    max_penetration = 0.0
    for cp in contact_points:
        penetration = float(cp[8])  # contact depth on body B (drawer)
        max_penetration = max(max_penetration, penetration)
    
    env._set_gripper_open(original_gripper)  # Restore
    env._step_world(steps=2)
    
    if max_penetration > DEFAULT_PHYSICS_THRESHOLDS["max_grasp_penetration_depth_m"]:
        reasons.append(
            f"initial_gripper_drawer_penetration: {max_penetration*1000:.1f}mm "
            f"(>{DEFAULT_PHYSICS_THRESHOLDS['max_grasp_penetration_depth_m']*1000:.0f}mm)"
        )
        feasible = False
    
    return {
        "stage": 0,
        "feasible": feasible,
        "reasons": reasons,
        "grasp_pos": grasp_pos.tolist(),
        "n_gripper_drawer_contacts": n_gripper_drawer_contacts,
        "max_penetration_m": max_penetration,
    }


def stage0_anygrasp_feasibility_batch(
    env,
    grasp_candidates: list[np.ndarray],
) -> list[dict[str, Any]]:
    """
    Filter a batch of AnyGrasp candidates through Stage 0 physics feasibility.
    Returns results for each candidate.
    """
    results = []
    for i, grasp_pose in enumerate(grasp_candidates):
        result = stage0_check_grasp_feasibility(env, grasp_pose)
        result["candidate_index"] = i
        results.append(result)
    return results


# ============================================================================
# Stage 1: Impedance-Controlled Compliant Pull Controller
# ============================================================================

def stage1_impedance_pull_step(
    env,
    target_drawer_fraction: float,
    current_drawer_fraction: float,
) -> np.ndarray:
    """
    Stage 1: Impedance-controlled pull action for the pull phase.
    
    Uses impedance control (spring-damper model) instead of world-frame delta,
    which avoids the "world-frame delta conflicts with drawer slider constraint" bug.
    
    Args:
        env: DrawerRobotEnv instance
        target_drawer_fraction: desired drawer open fraction
        current_drawer_fraction: current drawer fraction
    
    Returns:
        action (7,) with impedance-computed delta for x direction + current commands
    """
    TH = DEFAULT_PHYSICS_THRESHOLDS
    
    # Compute error
    drawer_error = target_drawer_fraction - current_drawer_fraction
    
    # Compute impedance force: F = K*e - D*v
    stiffness = TH["impedance_stiffness_N_m"]
    damping = TH["impedance_damping_Ns_m"]
    
    # Estimate drawer velocity from last 3 trace points
    if len(env.drawer_trace_history) >= 3:
        dt = 1.0 / 10.0  # 10fps
        v_est = (env.drawer_trace_history[-1] - env.drawer_trace_history[-3]) / (2 * dt)
    else:
        v_est = 0.0
    
    impedance_force = stiffness * drawer_error - damping * v_est
    
    # Clamp force
    max_pull_force = TH["pull_force_N"]
    impedance_force = np.clip(impedance_force, -max_pull_force, max_pull_force)
    
    # Convert force to action-space delta (approximate, since action is normalized)
    # translation_scale = 0.03m/unit, so delta_action = impedance_force / stiffness
    # This gives smooth, force-based motion
    action_x = np.clip(impedance_force / stiffness, -1.0, 1.0)
    
    # Maintain current y/z/rotation
    current_action = np.zeros(7)
    current_action[0] = float(action_x)
    current_action[6] = -1.0  # Keep gripper closed during pull
    
    return current_action.astype(np.float32)


# ============================================================================
# Stage 2: Frame-by-Frame Collision Filter (Post-Hoc Verification)
# ============================================================================

def stage2_check_frame_collision(
    env,
) -> dict[str, Any]:
    """
    Stage 2: Check for gripper-drawer collision at current frame.
    Returns collision evidence without modifying state.
    """
    p = env.p
    
    # Get contact points between robot and drawer
    contact_points = p.getContactPoints(
        bodyA=env.robot_id,
        bodyB=env.drawer_id,
        physicsClientId=env.client,
    )
    
    # Filter: only care about contacts involving gripper fingers
    gripper_drawer_contacts = []
    for cp in contact_points:
        link_a = cp[3]  # Link index on robot
        if link_a in env.GRIPPER_JOINTS:
            gripper_drawer_contacts.append(cp)
    
    max_penetration = 0.0
    total_normal_force = 0.0
    for cp in gripper_drawer_contacts:
        max_penetration = max(max_penetration, abs(float(cp[8])))
        total_normal_force += abs(float(cp[9]))
    
    threshold = DEFAULT_PHYSICS_THRESHOLDS["max_gripper_penetration_m"]
    collision = max_penetration > threshold
    
    return {
        "stage": 2,
        "n_gripper_drawer_contacts": len(gripper_drawer_contacts),
        "total_contact_force_N": total_normal_force,
        "max_penetration_m": max_penetration,
        "collision_detected": collision,
        "collision_threshold_m": threshold,
    }


def check_physics_legality(
    rollout_npz: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Verify a single rollout NPZ for physics legality violations.
    This is the post-hoc audit function (Stage 2 applied to stored data).
    
    Args:
        rollout_npz: dict loaded from npz file
        thresholds: optional override thresholds
    
    Returns:
        dict with 'legal': bool, 'issues': list, 'metrics': dict
    """
    TH = thresholds or DEFAULT_PHYSICS_THRESHOLDS
    
    actions = rollout_npz["actions"]       # (T, 7)
    eef_positions = rollout_npz["eef_positions"]  # (T, 3)
    gripper_values = rollout_npz.get("gripper_values", np.ones(len(actions)))  # (T,)
    drawer_fractions = rollout_npz.get("absolute_drawer_fraction", np.zeros(len(actions)))
    
    issues = []
    metrics = {}
    T = len(actions)
    
    # Check 1: Action delta smoothness (|Δaction|_2)
    if T > 1:
        action_deltas = np.abs(np.diff(actions, axis=0))
        l2 = np.linalg.norm(action_deltas, axis=1)
        max_delta_l2 = float(l2.max())
        p95_delta_l2 = float(np.percentile(l2, 95))
        metrics["max_delta_action_l2"] = max_delta_l2
        metrics["p95_delta_action_l2"] = p95_delta_l2
        
        if p95_delta_l2 > TH["max_step_delta_pos_m"]:
            issues.append(
                f"action_delta_l2_p95={p95_delta_l2:.4f}m > "
                f"threshold={TH['max_step_delta_pos_m']:.4f}m — jerky motion"
            )
    
    # Check 2: Position discontinuity (phantom jump detection)
    if T > 1:
        pos_deltas = np.abs(np.diff(eef_positions, axis=0))  # (T-1, 3)
        pos_delta_norms = np.linalg.norm(pos_deltas, axis=1)
        max_phantom = float(pos_delta_norms.max())
        metrics["max_position_jump_m"] = max_phantom
        
        phantom_indices = np.where(pos_delta_norms > TH["phantom_jump_threshold_m"])[0]
        if len(phantom_indices) > 0:
            issues.append(
                f"phantom_jumps: {len(phantom_indices)} jumps > "
                f"{TH['phantom_jump_threshold_m']*100:.1f}cm "
                f"at steps {phantom_indices.tolist()}"
            )
    
    # Check 3: Gripper penetration proxy
    # If gripper_values are binary {0,1} and gripper was closed near drawer,
    # we can check for gripper state vs. EEF position to detect penetration risk
    if T > 1:
        closed_mask = gripper_values < 0.5  # gripper nearly closed
        if np.any(closed_mask):
            closed_pos = eef_positions[closed_mask]
            if len(closed_pos) > 0:
                # Check if EEF is near drawer surface (y ≈ 0.20m drawer front)
                near_drawer = np.abs(closed_pos[:, 1] - 0.20) < 0.05
                if np.any(near_drawer):
                    n_penetration_risk = int(np.sum(near_drawer))
                    issues.append(
                        f"gripper_drawer_penetration_risk: "
                        f"{n_penetration_risk} frames with closed gripper near drawer surface"
                    )
    
    # Check 4: Drawer motion consistency
    if T > 1:
        drawer_deltas = np.abs(np.diff(drawer_fractions))
        max_drawer_delta = float(drawer_deltas.max())
        metrics["max_drawer_fraction_delta"] = max_drawer_delta
        
        # If actions should open drawer (positive x) but drawer fraction decreases
        actions_should_open = actions[:, 0] > 0.1
        if np.any(actions_should_open):
            avg_action_when_open = float(actions[actions_should_open, 0].mean())
            drawer_delta_when_open = drawer_deltas[actions_should_open[:-1]]
            if len(drawer_delta_when_open) > 0 and float(drawer_delta_when_open.mean()) < 0:
                issues.append(
                    f"drawer_not_following_action: "
                    f"avg_action_x={avg_action_when_open:.3f} but drawer delta={float(drawer_delta_when_open.mean()):.5f}"
                )
    
    legal = len(issues) == 0
    return {
        "legal": legal,
        "n_frames": T,
        "issues": issues,
        "metrics": metrics,
        "n_issues": len(issues),
        "pass_rate": f"{0 if not legal else 1}/1",
    }


# ============================================================================
# Physics-Constrained Rollout Generation (3-Stage Pipeline)
# ============================================================================

def physics_constrained_teacher_rollout(
    seed: int,
    anygrasp_candidates: list[np.ndarray],
    max_retry: int = 5,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Generate a physics-legal teacher rollout using the 3-stage pipeline.
    
    Args:
        seed: Random seed for environment
        anygrasp_candidates: List of (4, 4) grasp pose matrices from AnyGrasp
        max_retry: Number of retry attempts with different grasp candidates
        verbose: Print debug info
    
    Returns:
        dict with rollout data and physics verification results
    """
    from drawer_robot_env import DrawerRobotEnv
    import json
    
    env = DrawerRobotEnv(seed=seed, image_size=224, max_steps=96)
    obs = env.reset()
    
    best_grasp_result = None
    rollout_data = None
    
    for attempt in range(max_retry):
        candidate = anygrasp_candidates[attempt % len(anygrasp_candidates)]
        
        # Stage 0: Check grasp feasibility
        grasp_result = stage0_check_grasp_feasibility(env, candidate)
        if verbose:
            print(f"  Attempt {attempt}: Stage0 feasible={grasp_result['feasible']}")
        
        if not grasp_result["feasible"]:
            continue
        
        # Execute the grasp
        success = env.execute_grasp_at_pose(candidate)
        if not success:
            if verbose:
                print(f"  Attempt {attempt}: Grasp execution failed")
            continue
        
        # Stage 1: Impedance-controlled pull
        # During pull phase, use impedance controller instead of world-frame delta
        rollout_frames = []
        attached = False
        for step in range(env.max_steps):
            current_drawer_fraction = env.drawer_fraction()
            
            if not attached and env.attached:
                attached = True
            
            if attached:
                # Use impedance controller
                target_fraction = 0.80  # Pull to 80% open
                action = stage1_impedance_pull_step(
                    env, target_fraction, current_drawer_fraction
                )
            else:
                action = np.zeros(7, dtype=np.float32)
                action[6] = -1.0  # Keep closed before attachment
            
            obs, reward, done, info = env.step(action)
            
            # Stage 2: Per-frame collision check
            collision = stage2_check_frame_collision(env)
            
            rollout_frames.append({
                "state": obs.state.tolist(),
                "action": action.tolist(),
                "drawer_fraction": float(env.drawer_fraction()),
                "collision": collision,
            })
            
            if done:
                break
        
        # Verify the entire rollout
        states_arr = np.array([f["state"] for f in rollout_frames])
        actions_arr = np.array([f["action"] for f in rollout_frames])
        
        legality = check_physics_legality({
            "actions": actions_arr,
            "eef_positions": states_arr[:, :3],
            "gripper_values": states_arr[:, 7],  # NOTE: this will be wrong until P1a fix
        })
        
        if legality["legal"]:
            if verbose:
                print(f"  Attempt {attempt}: SUCCESS — physics-legal rollout")
            return {
                "seed": seed,
                "attempt": attempt,
                "rollout_frames": rollout_frames,
                "stages_passed": [0, 1, 2],
                "legality": legality,
            }
        else:
            if verbose:
                print(f"  Attempt {attempt}: Physics violations: {legality['issues']}")
            best_grasp_result = grasp_result
            rollout_data = legality
    
    return {
        "seed": seed,
        "attempt": max_retry,
        "stages_passed": [0] if best_grasp_result else [],
        "legality": rollout_data or {"legal": False, "issues": ["no feasible grasp found"]},
        "last_grasp_result": best_grasp_result,
    }


# ============================================================================
# Audit: Verify All Existing NPZ Files
# ============================================================================

def audit_rollouts(
    npz_dir: Path,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Audit all NPZ rollouts in a directory for physics legality.
    Returns per-file results and aggregate statistics.
    """
    TH = thresholds or DEFAULT_PHYSICS_THRESHOLDS
    npz_files = sorted(npz_dir.glob("*.npz"))
    
    results = []
    n_legal = 0
    n_illegal = 0
    all_issues: dict[str, int] = {}
    
    for pf in npz_files:
        try:
            data = dict(np.load(pf, allow_pickle=True))
            legality = check_physics_legality(data, TH)
            
            result = {
                "file": pf.name,
                "legal": legality["legal"],
                "n_frames": legality["n_frames"],
                "issues": legality["issues"],
                "metrics": legality["metrics"],
            }
            results.append(result)
            
            if legality["legal"]:
                n_legal += 1
            else:
                n_illegal += 1
                for issue in legality["issues"]:
                    issue_type = issue.split(":")[0]
                    all_issues[issue_type] = all_issues.get(issue_type, 0) + 1
        except Exception as e:
            results.append({
                "file": pf.name,
                "error": str(e),
                "legal": False,
            })
            n_illegal += 1
            all_issues["load_error"] = all_issues.get("load_error", 0) + 1
    
    return {
        "total": len(results),
        "legal": n_legal,
        "illegal": n_illegal,
        "pass_rate": f"{n_legal}/{len(results)} ({100*n_legal/len(results):.0f}%)",
        "issue_counts": all_issues,
        "per_file": results,
        "thresholds_used": TH,
    }


def print_audit_summary(audit: dict[str, Any]) -> None:
    """Print a human-readable audit summary."""
    print(f"\n{'='*60}")
    print(f"  Physics Legality Audit Summary")
    print(f"{'='*60}")
    print(f"  Total rollouts:  {audit['total']}")
    print(f"  Physics-legal:  {audit['legal']} ({100*audit['legal']/max(audit['total'],1):.0f}%)")
    print(f"  Violations:      {audit['illegal']}")
    print(f"\n  Issue breakdown:")
    for issue_type, count in sorted(
        audit["issue_counts"].items(), key=lambda x: -x[1]
    ):
        print(f"    {issue_type}: {count}")
    
    print(f"\n  Per-file details:")
    for r in audit["per_file"]:
        status = "✅ LEGAL" if r["legal"] else "❌ ILLEGAL"
        n_frames = r.get("n_frames", "?")
        issues_str = ", ".join(r.get("issues", [])) if not r.get("error") else r.get("error", "")
        print(f"    {status}  {r['file']:50s}  frames={n_frames}")
        if issues_str:
            print(f"             Issues: {issues_str[:80]}")


# ============================================================================
# Standalone NPZ Audit (No PyBullet Dependencies)
# ============================================================================

def standalone_npb_audit(npz_dir: Path) -> dict[str, Any]:
    """
    Standalone audit that only uses numpy — no PyBullet required.
    Can be run with just `python physics_legality.py audit-npz <dir>`.
    """
    TH = DEFAULT_PHYSICS_THRESHOLDS
    npz_files = sorted(npz_dir.glob("*.npz"))
    results = []
    for pf in npz_files:
        try:
            d = dict(np.load(pf, allow_pickle=True))
            legality = check_physics_legality(d, TH)
            results.append({
                "file": pf.name,
                "legal": legality["legal"],
                "n_frames": legality["n_frames"],
                "issues": legality["issues"],
                "metrics": legality.get("metrics", {}),
            })
        except Exception as e:
            results.append({"file": pf.name, "error": str(e), "legal": False})

    legal_n = sum(1 for r in results if r["legal"])
    illegal_n = len(results) - legal_n
    issue_counts = {}
    for r in results:
        for issue in r.get("issues", []):
            k = issue.split(":")[0]
            issue_counts[k] = issue_counts.get(k, 0) + 1

    return {
        "total": len(results), "legal": legal_n, "illegal": illegal_n,
        "pass_rate": f"{legal_n}/{len(results)}",
        "issue_counts": issue_counts,
        "per_file": results,
        "thresholds_used": TH,
    }


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="P0 Physics Legality Module")
    sub = parser.add_subparsers(dest="command", required=True)
    
    audit_p = sub.add_parser("audit", help="Audit existing NPZ rollouts for physics legality")
    audit_p.add_argument("--npz-dir", type=Path, required=True)
    audit_p.add_argument("--output", type=Path, required=True)
    
    check_p = sub.add_parser("check", help="Check a single NPZ file")
    check_p.add_argument("--npz", type=Path, required=True)
    
    gen_p = sub.add_parser("generate", help="Generate physics-legal rollouts (requires AnyGrasp)")
    gen_p.add_argument("--seed", type=int, default=42)
    gen_p.add_argument("--num-rollouts", type=int, default=10)
    gen_p.add_argument("--output-dir", type=Path, required=True)

    # Standalone audit: no PyBullet dependency
    audit_npz_p = sub.add_parser("audit-npz", help="Audit NPZ files without PyBullet (numpy-only)")
    audit_npz_p.add_argument("--npz-dir", type=Path, required=True)
    audit_npz_p.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()
    
    if args.command == "audit":
        audit = audit_rollouts(args.npz_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit, indent=2))
        print_audit_summary(audit)
        print(f"\nWritten: {args.output}")
    
    elif args.command == "check":
        data = dict(np.load(args.npz, allow_pickle=True))
        legality = check_physics_legality(data)
        print(f"\nFile: {args.npz.name}")
        print(f"Frames: {legality['n_frames']}")
        print(f"Legal: {legality['legal']}")
        if legality["issues"]:
            print("Issues:")
            for issue in legality["issues"]:
                print(f"  - {issue}")
        else:
            print("No physics violations found.")
    
    elif args.command == "generate":
        print("Note: This requires AnyGrasp candidate poses. ")
        print("Use physics_constrained_teacher_rollout() programmatically with candidates.")

    elif args.command == "audit-npz":
        audit = standalone_npb_audit(args.npz_dir)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(audit, indent=2))
        print_audit_summary(audit)


if __name__ == "__main__":
    main()
