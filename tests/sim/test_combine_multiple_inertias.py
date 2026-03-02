#!/usr/bin/env python3
"""
Unit tests: combine_multiple_inertias (Parallel Axis Theorem)

This validates the *mathematical correctness* of inertia merging used by the URDF exporter:
- total mass: M = Σ m_i
- combined center of mass: C = (1/M) Σ (m_i c_i)
- combined inertia about C:
    I = Σ [ I_i + m_i * ((r·r)E - r⊗r) ],  r = c_i - C

Run:
  cd /mnt/afs2/zhuhaowu/infinigen
  conda run -n infinigen pytest -q tests/sim/test_combine_multiple_inertias.py
"""

from __future__ import annotations

import numpy as np

from infinigen.core.sim.physics.thin_shell_inertia import combine_multiple_inertias


def _parallel_axis_term(m: float, r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    return float(m) * (float(r @ r) * np.eye(3) - np.outer(r, r))


def test_combine_multiple_inertias_two_bodies_symmetric_offsets() -> None:
    m = 1.0
    d = 0.4
    masses = [m, m]
    coms = [np.array([+d, 0.0, 0.0]), np.array([-d, 0.0, 0.0])]
    I0 = np.diag([0.1, 0.2, 0.3])
    inertias = [I0.copy(), I0.copy()]

    M, C, I = combine_multiple_inertias(
        masses=masses, centers_of_mass=coms, inertia_tensors=inertias
    )

    assert np.isclose(M, 2.0)
    assert np.allclose(C, np.zeros(3), atol=1e-12)

    # Expected: I = Σ(I_i + m * ((r·r)E - r⊗r)), with r = +/-[d,0,0]
    term = _parallel_axis_term(m, np.array([d, 0.0, 0.0]))
    I_expected = (I0 + term) + (I0 + term)

    assert np.allclose(I, I_expected, rtol=1e-10, atol=1e-12)
    assert np.allclose(I, I.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(I) > 0.0)


def test_combine_multiple_inertias_mass_weighted_com_and_inertia() -> None:
    masses = [2.0, 1.0]
    coms = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.5, 0.0])]
    inertias = [np.diag([0.2, 0.2, 0.2]), np.diag([0.1, 0.1, 0.1])]

    M, C, I = combine_multiple_inertias(
        masses=masses, centers_of_mass=coms, inertia_tensors=inertias
    )

    M_expected = 3.0
    C_expected = (masses[0] * coms[0] + masses[1] * coms[1]) / M_expected

    I_expected = np.zeros((3, 3), dtype=np.float64)
    for m, c, Ii in zip(masses, coms, inertias):
        r = c - C_expected
        I_expected += np.asarray(Ii, dtype=np.float64) + _parallel_axis_term(m, r)

    assert np.isclose(M, M_expected)
    assert np.allclose(C, C_expected, rtol=1e-12, atol=1e-12)
    assert np.allclose(I, I_expected, rtol=1e-10, atol=1e-12)
    assert np.allclose(I, I.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(I) > 0.0)
