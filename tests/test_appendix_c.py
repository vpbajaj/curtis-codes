"""Tests for the Appendix C planar three-body integration."""

import numpy as np
import pytest

from curtis.appendix_c.accel_3body import accel_3body
from curtis.appendix_c.threebody import threebody


def test_accel_3body_velocity_block():
    # The first eight derivative components must be the velocity components
    # carried in the state vector.
    G = 6.67259e-20
    m = np.array([1e29, 1e29, 1e29])
    f = np.arange(1.0, 17.0)  # arbitrary non-degenerate state
    dfdt = accel_3body(0.0, f, G, m)
    assert dfdt.shape == (16,)
    # d/dt of the positions == velocities (f[8:16]).
    assert dfdt[:8] == pytest.approx(f[8:16])
    # The centre of mass has zero acceleration (Eq C.5a).
    assert dfdt[14] == 0.0 and dfdt[15] == 0.0


def test_accel_3body_newtons_third_law():
    # Total force on the system is zero: sum of m_i * a_i == 0.
    G = 6.67259e-20
    m = np.array([1e29, 2e29, 3e29])
    f = np.array([0., 0., 3e5, 0., 6e5, 1e5,
                  0., 0., 0., 0., 250., 250., 0., 0., 0., 0.])
    dfdt = accel_3body(0.0, f, G, m)
    a1 = dfdt[8:10]
    a2 = dfdt[10:12]
    a3 = dfdt[12:14]
    total = m[0] * a1 + m[1] * a2 + m[2] * a3
    assert total == pytest.approx([0.0, 0.0], abs=1e-6)


def test_threebody_com_is_linear():
    # The centre of mass carried in the state vector must move in a straight
    # line: rG(t) = rG0 + vG0 * t (Equations C.5).
    t, f = threebody()
    m = np.array([1e29, 1e29, 1e29])
    r0 = np.array([[0., 0.], [3e5, 0.], [6e5, 0.]])
    v0 = np.array([[0., 0.], [250., 250.], [0., 0.]])
    rG0 = m @ r0 / m.sum()
    vG0 = m @ v0 / m.sum()

    xG, yG = f[:, 6], f[:, 7]
    assert xG == pytest.approx(rG0[0] + vG0[0] * t, abs=1e-3)
    assert yG == pytest.approx(rG0[1] + vG0[1] * t, abs=1e-3)


def test_threebody_com_matches_body_average():
    # Momentum conservation: the mass-weighted average of the three
    # integrated bodies must equal the independently integrated COM.
    t, f = threebody()
    m = np.array([1e29, 1e29, 1e29])
    x_avg = (m[0] * f[:, 0] + m[1] * f[:, 2] + m[2] * f[:, 4]) / m.sum()
    y_avg = (m[0] * f[:, 1] + m[1] * f[:, 3] + m[2] * f[:, 5]) / m.sum()
    assert x_avg == pytest.approx(f[:, 6], abs=1.0)
    assert y_avg == pytest.approx(f[:, 7], abs=1.0)


def test_threebody_final_com():
    # Final COM position predicted analytically.
    t, f = threebody()
    # rG0 = (300000, 0); vG0 = (250/3, 250/3); tf = 67000.
    assert f[-1, 6] == pytest.approx(300000 + (250 / 3) * 67000, rel=1e-6)
    assert f[-1, 7] == pytest.approx((250 / 3) * 67000, rel=1e-6)
