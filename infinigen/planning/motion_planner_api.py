from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol


@dataclass(frozen=True)
class JointLimit:
    lower: float
    upper: float


@dataclass(frozen=True)
class PlanRequest:
    """
    Planner-agnostic joint-space planning request.

    This keeps the Phase-3 integration minimal: box manipulation goals are expressed
    as desired joint positions in URDF joint coordinates.
    """

    urdf_path: str
    start_positions: Dict[str, float]
    goal_positions: Dict[str, float]
    joint_limits: Optional[Dict[str, JointLimit]] = None

    # Execution / validation options
    n_steps: int = 60
    collision_check: bool = True
    self_collision: bool = True
    use_inertia_from_file: bool = True
    contact_eps: float = 1e-4


@dataclass
class PlanResult:
    success: bool
    message: str = ""
    trajectory: List[Dict[str, float]] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)


class MotionPlanner(Protocol):
    def plan(self, req: PlanRequest) -> PlanResult:  # pragma: no cover
        ...

