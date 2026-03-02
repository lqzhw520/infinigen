import numpy as np

from infinigen.perception.topo_box_net.kinematic_consistency import (
    hinge_constraint_residual_np,
    hinge_transform_np,
    prismatic_constraint_residual_np,
    prismatic_transform_np,
    rodrigues_np,
)


def _apply(R: np.ndarray, t: np.ndarray, x: np.ndarray) -> np.ndarray:
    return R @ x + t


def test_rodrigues_identity():
    R = rodrigues_np(np.array([1.0, 0.0, 0.0]), 0.0)
    assert np.allclose(R, np.eye(3), atol=1e-12)


def test_hinge_transform_keeps_origin_fixed():
    axis = np.array([0.0, 0.0, 1.0])
    origin = np.array([1.0, 0.0, 0.0])
    R, t = hinge_transform_np(axis, np.pi / 2.0, origin)
    origin2 = _apply(R, t, origin)
    assert np.allclose(origin2, origin, atol=1e-12)


def test_hinge_constraint_residual_zero_for_exact_transform():
    axis = np.array([0.3, -0.7, 1.1])
    origin = np.array([0.2, -0.1, 0.3])
    R, t = hinge_transform_np(axis, 0.3, origin)
    res = hinge_constraint_residual_np(R=R, t=t, axis=axis, origin=origin)
    assert res < 1e-10


def test_hinge_constraint_residual_positive_for_wrong_origin():
    axis = np.array([0.0, 0.0, 1.0])
    origin = np.array([0.2, -0.1, 0.3])
    R, t = hinge_transform_np(axis, 0.8, origin)
    res_good = hinge_constraint_residual_np(R=R, t=t, axis=axis, origin=origin)
    res_bad = hinge_constraint_residual_np(R=R, t=t, axis=axis, origin=origin + np.array([0.05, 0.0, 0.0]))
    assert res_good < 1e-10
    assert res_bad > 1e-6


def test_prismatic_constraint_residual_zero_for_exact_transform():
    axis = np.array([0.0, 1.0, 0.0])
    R, t = prismatic_transform_np(axis, 0.2)
    res = prismatic_constraint_residual_np(R=R, t=t, axis=axis)
    assert res < 1e-12


def test_prismatic_constraint_residual_positive_for_perp_translation():
    axis = np.array([0.0, 1.0, 0.0])
    R = np.eye(3)
    t = np.array([0.1, 0.0, 0.0])  # perpendicular to axis
    res = prismatic_constraint_residual_np(R=R, t=t, axis=axis)
    assert res > 1e-6

