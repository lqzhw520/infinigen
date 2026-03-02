from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from .motion_planner_api import PlanRequest, PlanResult


class JointSpaceLinearPlanner:
    """
    Minimal, runnable planner baseline:
    - Linear interpolation in joint space
    - Optional self-collision checking in PyBullet

    This is NOT meant to be optimal; it is a clean integration target and a sanity-check
    baseline for Phase 3.2.
    """

    def plan(self, req: PlanRequest) -> PlanResult:
        if int(req.n_steps) < 2:
            return PlanResult(success=False, message="n_steps must be >= 2")

        joint_names = sorted(
            set(req.start_positions.keys()) | set(req.goal_positions.keys())
        )
        start = {jn: float(req.start_positions.get(jn, 0.0)) for jn in joint_names}
        goal = {jn: float(req.goal_positions.get(jn, start[jn])) for jn in joint_names}

        # Pre-check limits for start/goal (if provided)
        if req.joint_limits is not None:
            for jn in joint_names:
                lim = req.joint_limits.get(jn)
                if lim is None:
                    continue
                if not (lim.lower <= start[jn] <= lim.upper):
                    return PlanResult(
                        success=False,
                        message=f"start out of limits: {jn}={start[jn]} lim={lim}",
                    )
                if not (lim.lower <= goal[jn] <= lim.upper):
                    return PlanResult(
                        success=False,
                        message=f"goal out of limits: {jn}={goal[jn]} lim={lim}",
                    )

        traj: List[Dict[str, float]] = []
        for i in range(int(req.n_steps)):
            a = float(i) / float(int(req.n_steps) - 1)
            q = {jn: (1.0 - a) * start[jn] + a * goal[jn] for jn in joint_names}
            traj.append(q)

        if not req.collision_check:
            return PlanResult(
                success=True, message="planned (no collision checking)", trajectory=traj
            )

        # Collision checking in PyBullet
        try:
            import pybullet as p
        except Exception as e:
            return PlanResult(
                success=False, message=f"pybullet import failed: {e}", trajectory=traj
            )

        physics_id = p.connect(p.DIRECT)
        try:
            flags = 0
            if req.self_collision:
                flags |= p.URDF_USE_SELF_COLLISION
            if req.use_inertia_from_file:
                flags |= p.URDF_USE_INERTIA_FROM_FILE

            urdf_path = Path(str(req.urdf_path)).resolve()
            # Allow URDF meshes like "assets/geom_0.obj" to be resolved.
            p.setAdditionalSearchPath(str(urdf_path.parent))
            body = p.loadURDF(str(urdf_path), useFixedBase=True, flags=flags)

            name_to_idx: Dict[str, int] = {}
            for j in range(p.getNumJoints(body)):
                ji = p.getJointInfo(body, j)
                joint_name = ji[1].decode("utf-8")
                name_to_idx[joint_name] = j

            # Only check joints that exist in PyBullet model
            active = [jn for jn in joint_names if jn in name_to_idx]

            collided = False
            min_dist = float("inf")
            n_bad_steps = 0

            for step_idx, q in enumerate(traj):
                for jn in active:
                    p.resetJointState(body, name_to_idx[jn], float(q[jn]))

                p.performCollisionDetection()
                contacts = p.getContactPoints(bodyA=body, bodyB=body)
                if contacts:
                    # contactDistance is index 8 in pybullet getContactPoints tuple
                    dists = [float(c[8]) for c in contacts]
                    if dists:
                        min_dist = min(min_dist, min(dists))
                    has_penetration = any(d < -float(req.contact_eps) for d in dists)
                    if has_penetration:
                        collided = True
                        n_bad_steps += 1
                        break

            metrics = {
                "n_steps": int(req.n_steps),
                "n_active_joints": int(len(active)),
                "min_contact_distance": float(min_dist)
                if np.isfinite(min_dist)
                else None,
                "collided": bool(collided),
                "n_bad_steps": int(n_bad_steps),
            }
            if collided:
                return PlanResult(
                    success=False,
                    message="collision along trajectory",
                    trajectory=traj,
                    metrics=metrics,
                )
            return PlanResult(
                success=True,
                message="planned (collision-free)",
                trajectory=traj,
                metrics=metrics,
            )
        finally:
            p.disconnect(physics_id)
