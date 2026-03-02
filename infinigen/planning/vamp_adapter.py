from __future__ import annotations

from .motion_planner_api import PlanRequest, PlanResult


class VampMotionPlanner:
    """
    Integration stub for the external VAMP Motion Planner.

    This repository does not vendor VAMP. The adapter defines the expected interface
    and where translation between data formats should occur.
    """

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)

        # Lazy import to avoid hard dependency.
        try:
            import vamp  # type: ignore  # noqa: F401
        except Exception as e:  # pragma: no cover
            self._import_error = e
        else:  # pragma: no cover
            self._import_error = None

    def plan(self, req: PlanRequest) -> PlanResult:
        if self._import_error is not None:  # pragma: no cover
            return PlanResult(
                success=False,
                message=(
                    "VAMP is not installed/available in this environment. "
                    f"Import error: {self._import_error}"
                ),
            )

        # TODO(Phase3.2): translate PlanRequest -> VAMP problem specification and call VAMP planner.
        # Expected responsibilities:
        # - Load URDF and joint limits
        # - Define collision constraints (self-collision + environment)
        # - Map start/goal joint states
        # - Return a time-parameterized trajectory
        return PlanResult(success=False, message="VAMP adapter not implemented yet")

