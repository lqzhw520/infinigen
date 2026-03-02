from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


@dataclass
class TopoBoxNetOutput:
    """
    Canonical outputs for Topo-Box-Net (research skeleton).

    Shapes:
      - box_type_logits: (B, num_box_types)
      - joint_state: (B, max_joints)
      - joint_type_logits: (B, max_joints, 3)  # {revolute/prismatic/fixed_or_pad}
      - joint_axis: (B, max_joints, 3)
      - joint_origin: (B, max_joints, 3)
    """

    box_type_logits: "torch.Tensor"
    joint_state: "torch.Tensor"
    joint_type_logits: "torch.Tensor"
    joint_axis: "torch.Tensor"
    joint_origin: "torch.Tensor"


if torch is not None:

    class SimpleConvEncoder(nn.Module):
        def __init__(self, in_channels: int, feat_dim: int = 256):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels, 32, 5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
            self.proj = nn.Linear(256, feat_dim)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B,C,H,W)
            f = self.backbone(x)
            # global average pooling
            f = f.mean(dim=(-2, -1))  # (B,256)
            return self.proj(f)


    class TopoBoxNet(nn.Module):
        """
        Topo-Box-Net (skeleton):
        - Image encoder over RGB-D (+ optional segmentation channel)
        - Heads for box type, joint state, and joint parameters.

        This is intentionally minimal: it provides a concrete, runnable shape for the
        Phase 2.1 design. Replace the encoder (e.g. ResNet/ViT) and heads as needed.
        """

        def __init__(
            self,
            *,
            num_box_types: int,
            max_joints: int,
            use_segmentation: bool = True,
            feat_dim: int = 256,
        ):
            super().__init__()
            self.num_box_types = int(num_box_types)
            self.max_joints = int(max_joints)
            self.use_segmentation = bool(use_segmentation)

            # Inputs: RGB(3) + Depth(1) + optional Seg(1)
            in_ch = 3 + 1 + (1 if self.use_segmentation else 0)
            self.encoder = SimpleConvEncoder(in_channels=in_ch, feat_dim=feat_dim)

            self.box_type_head = nn.Linear(feat_dim, self.num_box_types)

            self.joint_state_head = nn.Linear(feat_dim, self.max_joints)
            self.joint_type_head = nn.Linear(feat_dim, self.max_joints * 3)
            self.joint_axis_head = nn.Linear(feat_dim, self.max_joints * 3)
            self.joint_origin_head = nn.Linear(feat_dim, self.max_joints * 3)

        def forward(
            self,
            *,
            rgb: "torch.Tensor",
            depth: "torch.Tensor",
            segmentation: Optional["torch.Tensor"] = None,
        ) -> TopoBoxNetOutput:
            """
            Args:
              rgb: (B,3,H,W) float32 in [0,1]
              depth: (B,1,H,W) float32 in meters
              segmentation: (B,1,H,W) float32 (e.g. normalized link-id mask) if enabled
            """
            if self.use_segmentation:
                if segmentation is None:
                    raise ValueError("segmentation is required when use_segmentation=True")
                x = torch.cat([rgb, depth, segmentation], dim=1)
            else:
                x = torch.cat([rgb, depth], dim=1)

            feat = self.encoder(x)

            box_type_logits = self.box_type_head(feat)

            joint_state = self.joint_state_head(feat)
            joint_type_logits = self.joint_type_head(feat).view(-1, self.max_joints, 3)
            joint_axis = self.joint_axis_head(feat).view(-1, self.max_joints, 3)
            joint_origin = self.joint_origin_head(feat).view(-1, self.max_joints, 3)

            # Normalize axis to unit vectors (avoid NaN on zero vectors).
            joint_axis = F.normalize(joint_axis, dim=-1, eps=1e-8)

            return TopoBoxNetOutput(
                box_type_logits=box_type_logits,
                joint_state=joint_state,
                joint_type_logits=joint_type_logits,
                joint_axis=joint_axis,
                joint_origin=joint_origin,
            )

else:

    class TopoBoxNet:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TopoBoxNet requires PyTorch. Install torch (and optionally torchvision) to use this module."
            )

