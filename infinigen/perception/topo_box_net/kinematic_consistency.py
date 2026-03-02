from __future__ import annotations

from typing import Tuple

import numpy as np


def _normalize_np(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError(f"Zero/invalid vector: {v}")
    return v / n


def skew_np(a: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix [a]_x."""
    ax, ay, az = [float(x) for x in np.asarray(a, dtype=np.float64).tolist()]
    return np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]], dtype=np.float64)


def rodrigues_np(axis: np.ndarray, theta: float) -> np.ndarray:
    """
    Rodrigues' rotation formula.

    Args:
      axis: (3,) rotation axis (need not be unit)
      theta: rotation angle in radians
    Returns:
      R: (3,3) rotation matrix
    """
    a = _normalize_np(axis)
    K = skew_np(a)
    th = float(theta)
    I = np.eye(3, dtype=np.float64)
    return I + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def hinge_transform_np(axis: np.ndarray, theta: float, origin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Relative transform (R,t) for a *pure revolute* joint about an axis passing through `origin`.

    Convention:
      x_child = R x_parent + t

    For a rotation about a line (origin, axis):
      t = o - R o
    """
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    R = rodrigues_np(axis, theta)
    t = o - R @ o
    return R, t


def prismatic_transform_np(axis: np.ndarray, displacement: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Relative transform (R,t) for a *pure prismatic* joint translating along `axis` by `displacement`.

    Convention:
      x_child = R x_parent + t
    """
    a = _normalize_np(axis)
    d = float(displacement)
    R = np.eye(3, dtype=np.float64)
    t = a * d
    return R, t


def hinge_constraint_residual_np(
    *,
    R: np.ndarray,
    t: np.ndarray,
    axis: np.ndarray,
    origin: np.ndarray,
) -> float:
    """
    Kinematic constraint residual for a revolute joint, given an observed transform (R,t) and
    joint parameters (axis, origin) expressed in the parent frame.

    Residual terms (all should be ~0 for a valid hinge):
      1) Axis invariance: R a = a
      2) No translation along axis: a·t = 0
      3) Rotation about the axis line: (I - R) o = t
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    a = _normalize_np(axis)
    o = np.asarray(origin, dtype=np.float64).reshape(3)

    term1 = np.linalg.norm(R @ a - a)
    term2 = float(np.dot(a, t))
    term3 = np.linalg.norm((np.eye(3) - R) @ o - t)
    return float(term1 * term1 + term2 * term2 + term3 * term3)


def prismatic_constraint_residual_np(*, R: np.ndarray, t: np.ndarray, axis: np.ndarray) -> float:
    """
    Kinematic constraint residual for a prismatic joint, given an observed transform (R,t) and
    joint axis expressed in the parent frame.

    Residual terms (all should be ~0 for a valid prismatic):
      1) No relative rotation: R = I
      2) Translation parallel to axis: t_perp = (I - aa^T) t = 0
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    a = _normalize_np(axis)

    term1 = np.linalg.norm(R - np.eye(3))
    t_perp = t - a * float(np.dot(a, t))
    term2 = np.linalg.norm(t_perp)
    return float(term1 * term1 + term2 * term2)


# -----------------------------
# Optional PyTorch reference impl
# -----------------------------

try:  # pragma: no cover
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def rodrigues_torch(axis: "torch.Tensor", theta: "torch.Tensor", eps: float = 1e-12) -> "torch.Tensor":
    """
    Batched Rodrigues (PyTorch).
    axis: (...,3)
    theta: (...,) or (...,1)
    returns: (...,3,3)
    """
    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch is required for rodrigues_torch")

    if theta.ndim == axis.ndim:
        theta = theta[..., 0]
    axis = axis.to(dtype=torch.float64)
    theta = theta.to(dtype=torch.float64)

    n = torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(eps)
    a = axis / n
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]

    zeros = torch.zeros_like(ax)
    K = torch.stack(
        [
            torch.stack([zeros, -az, ay], dim=-1),
            torch.stack([az, zeros, -ax], dim=-1),
            torch.stack([-ay, ax, zeros], dim=-1),
        ],
        dim=-2,
    )  # (...,3,3)

    I = torch.eye(3, dtype=torch.float64, device=axis.device).expand(K.shape)
    th = theta[..., None, None]
    return I + torch.sin(th) * K + (1.0 - torch.cos(th)) * (K @ K)


def hinge_transform_torch(
    axis: "torch.Tensor", theta: "torch.Tensor", origin: "torch.Tensor"
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Return (R,t) for hinge joint; supports batching."""
    R = rodrigues_torch(axis, theta)
    o = origin.to(dtype=torch.float64)
    t = o - (R @ o[..., None]).squeeze(-1)
    return R, t


def prismatic_transform_torch(axis: "torch.Tensor", displacement: "torch.Tensor", eps: float = 1e-12) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Return (R,t) for prismatic joint; supports batching."""
    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch is required for prismatic_transform_torch")
    axis = axis.to(dtype=torch.float64)
    d = displacement.to(dtype=torch.float64)
    n = torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(eps)
    a = axis / n
    R = torch.eye(3, dtype=torch.float64, device=axis.device).expand(a.shape[:-1] + (3, 3))
    t = a * d[..., None]
    return R, t


def hinge_constraint_residual_torch(
    *,
    R: "torch.Tensor",
    t: "torch.Tensor",
    axis: "torch.Tensor",
    origin: "torch.Tensor",
    eps: float = 1e-12,
) -> "torch.Tensor":
    """Torch version of `hinge_constraint_residual_np` (returns (...,) residual)."""
    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch is required for hinge_constraint_residual_torch")
    R = R.to(dtype=torch.float64)
    t = t.to(dtype=torch.float64)
    axis = axis.to(dtype=torch.float64)
    origin = origin.to(dtype=torch.float64)

    n = torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(eps)
    a = axis / n
    term1 = torch.linalg.norm((R @ a[..., None]).squeeze(-1) - a, dim=-1)
    term2 = (a * t).sum(dim=-1)
    I = torch.eye(3, dtype=torch.float64, device=R.device).expand(R.shape)
    term3 = torch.linalg.norm(((I - R) @ origin[..., None]).squeeze(-1) - t, dim=-1)
    return term1 * term1 + term2 * term2 + term3 * term3


def prismatic_constraint_residual_torch(
    *,
    R: "torch.Tensor",
    t: "torch.Tensor",
    axis: "torch.Tensor",
    eps: float = 1e-12,
) -> "torch.Tensor":
    """Torch version of `prismatic_constraint_residual_np` (returns (...,) residual)."""
    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch is required for prismatic_constraint_residual_torch")
    R = R.to(dtype=torch.float64)
    t = t.to(dtype=torch.float64)
    axis = axis.to(dtype=torch.float64)
    n = torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(eps)
    a = axis / n
    I = torch.eye(3, dtype=torch.float64, device=R.device).expand(R.shape)
    term1 = torch.linalg.norm(R - I, dim=(-2, -1))
    t_par = (a * t).sum(dim=-1, keepdim=True)
    t_perp = t - a * t_par
    term2 = torch.linalg.norm(t_perp, dim=-1)
    return term1 * term1 + term2 * term2

