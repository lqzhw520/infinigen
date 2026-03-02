from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


@dataclass(frozen=True)
class LossWeights:
    box_type_ce: float = 1.0
    joint_state_l1: float = 1.0
    joint_type_ce: float = 0.5
    joint_axis_l1: float = 0.25
    joint_origin_l1: float = 0.25


def compute_losses(
    *,
    outputs,
    targets: Dict[str, "torch.Tensor"],
    joint_mask: Optional["torch.Tensor"] = None,
    weights: LossWeights = LossWeights(),
) -> Dict[str, "torch.Tensor"]:
    """
    Compute multi-task supervised losses for the Topo-Box-Net skeleton.

    Args:
      outputs: TopoBoxNetOutput
      targets:
        - box_type: (B,) int64
        - joint_state: (B,max_joints) float32
        - joint_type: (B,max_joints) int64 in {0,1,2} for {revolute, prismatic, fixed/pad}
        - joint_axis: (B,max_joints,3) float32 (unit)
        - joint_origin: (B,max_joints,3) float32
      joint_mask: (B,max_joints) float32 in {0,1}, masks padded joints
    """
    if torch is None:  # pragma: no cover
        raise ImportError("compute_losses requires PyTorch.")

    losses: Dict[str, torch.Tensor] = {}

    losses["box_type_ce"] = F.cross_entropy(outputs.box_type_logits, targets["box_type"])

    mask = joint_mask
    if mask is None:
        mask = torch.ones_like(outputs.joint_state, dtype=torch.float32)
    mask = mask.to(dtype=torch.float32)

    # Joint state regression (masked L1)
    l1 = torch.abs(outputs.joint_state - targets["joint_state"])
    losses["joint_state_l1"] = (l1 * mask).sum() / (mask.sum().clamp_min(1.0))

    # Joint type classification (masked CE)
    jt_logits = outputs.joint_type_logits  # (B,M,3)
    jt_tgt = targets["joint_type"]  # (B,M)
    ce = F.cross_entropy(jt_logits.reshape(-1, 3), jt_tgt.reshape(-1), reduction="none").view_as(jt_tgt)
    losses["joint_type_ce"] = (ce * mask).sum() / (mask.sum().clamp_min(1.0))

    # Axis/origin regression (masked L1)
    ax_l1 = torch.abs(outputs.joint_axis - targets["joint_axis"]).sum(dim=-1)  # (B,M)
    losses["joint_axis_l1"] = (ax_l1 * mask).sum() / (mask.sum().clamp_min(1.0))

    org_l1 = torch.abs(outputs.joint_origin - targets["joint_origin"]).sum(dim=-1)  # (B,M)
    losses["joint_origin_l1"] = (org_l1 * mask).sum() / (mask.sum().clamp_min(1.0))

    losses["total"] = (
        weights.box_type_ce * losses["box_type_ce"]
        + weights.joint_state_l1 * losses["joint_state_l1"]
        + weights.joint_type_ce * losses["joint_type_ce"]
        + weights.joint_axis_l1 * losses["joint_axis_l1"]
        + weights.joint_origin_l1 * losses["joint_origin_l1"]
    )

    return losses

