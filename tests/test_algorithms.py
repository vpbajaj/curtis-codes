"""Verify each Appendix D algorithm against the published example outputs in
Curtis, *Orbital Mechanics for Engineering Students*.

Expected values are the numbers printed in the book's "Output from
Example_x_yy" listings.
"""

import math

import numpy as np
import pytest

deg = math.pi / 180.0


# --- Algorithm 3.1: kepler_E (Example 3.2) --------------------------------
def test_kepler_E():
    from curtis.algorithms.alg_3_1_kepler_E import kepler_E
    E = kepler_E(0.37255, 3.6029)
    assert E == pytest.approx(3.47942, abs=1e-5)


# --- Algorithm 3.2: kepler_H (Example 3.5) --------------------------------
def test_kepler_H():
    from curtis.algorithms.alg_3_2_kepler_H import kepler_H
    F = kepler_H(2.7696, 40.69)
    assert F == pytest.approx(3.46309, abs=1e-5)


# --- D.4: Stumpff functions -----------------------------------------------
def test_stumpff_z_zero():
    from curtis.algorithms.stumpff import stumpC, stumpS
    assert stumpC(0.0) == pytest.approx(0.5)
    assert stumpS(0.0) == pytest.approx(1.0 / 6.0)


def test_stumpff_continuity():
    # C and S are entire; the three branches must agree near z = 0.
    from curtis.algorithms.stumpff import stumpC, stumpS
    for z in (1e-6, -1e-6):
        assert stumpC(z) == pytest.approx(0.5, abs=1e-6)
        assert stumpS(z) == pytest.approx(1.0 / 6.0, abs=1e-6)


# --- Algorithm 3.3: kepler_U (Example 3.6) --------------------------------
def test_kepler_U():
    from curtis.algorithms.alg_3_3_kepler_U import kepler_U
    x = kepler_U(3600.0, 10000.0, 3.0752, 1.0 / -19655.0, mu=398600.0)
    assert x == pytest.approx(128.511, abs=1e-3)


# --- Algorithm 3.4: rv_from_r0v0 (Example 3.7) ----------------------------
def test_rv_from_r0v0():
    from curtis.algorithms.alg_3_4_rv_from_r0v0 import rv_from_r0v0
    R, V = rv_from_r0v0([7000.0, -12124.0, 0.0],
                        [2.6679, 4.6210, 0.0], 3600.0)
    assert R == pytest.approx([-3297.77, 7413.40, 0.0], abs=1e-2)
    assert V == pytest.approx([-8.2976, -0.964045, 0.0], abs=1e-4)


# --- Algorithm 4.1: coe_from_sv (Example 4.3) -----------------------------
def test_coe_from_sv():
    from curtis.algorithms.alg_4_1_coe_from_sv import coe_from_sv
    coe = coe_from_sv([-6045.0, -3490.0, 2500.0],
                      [-3.457, 6.618, 2.533])
    h, e, RA, incl, w, TA, a = coe
    assert h == pytest.approx(58311.7, abs=1e-1)
    assert e == pytest.approx(0.171212, abs=1e-6)
    assert RA / deg == pytest.approx(255.279, abs=1e-3)
    assert incl / deg == pytest.approx(153.249, abs=1e-3)
    assert w / deg == pytest.approx(20.0683, abs=1e-3)
    assert TA / deg == pytest.approx(28.4456, abs=1e-3)
    assert a == pytest.approx(8788.1, abs=1e-1)


# --- Algorithm 4.2: sv_from_coe (Example 4.5) -----------------------------
def test_sv_from_coe():
    from curtis.algorithms.alg_4_2_sv_from_coe import sv_from_coe
    coe = [80000.0, 1.4, 40.0 * deg, 30.0 * deg, 60.0 * deg, 30.0 * deg]
    r, v = sv_from_coe(coe)
    assert r == pytest.approx([-4039.9, 4814.56, 3628.62], abs=1e-2)
    assert v == pytest.approx([-10.386, -4.77192, 1.74388], abs=1e-4)


# --- Algorithm 5.1: Gibbs (Example 5.1) -----------------------------------
def test_gibbs():
    from curtis.algorithms.alg_5_1_gibbs import gibbs
    v2, ierr = gibbs([-294.32, 4265.1, 5986.7],
                     [-1365.4, 3637.6, 6346.8],
                     [-2940.3, 2473.7, 6555.8])
    assert ierr == 0
    assert v2 == pytest.approx([-6.2176, -4.01237, 1.59915], abs=1e-4)


# --- Algorithm 5.2: Lambert (Example 5.2) ---------------------------------
def test_lambert():
    from curtis.algorithms.alg_5_2_lambert import lambert
    v1, v2 = lambert([5000.0, 10000.0, 2100.0],
                     [-14600.0, 2500.0, 7000.0], 3600.0, "pro")
    assert v1 == pytest.approx([-5.99249, 1.92536, 3.24564], abs=1e-4)
    assert v2 == pytest.approx([-3.31246, -4.19662, -0.385288], abs=1e-4)


# --- D.12: Julian day (Example 5.4) ---------------------------------------
def test_julian_day():
    from curtis.algorithms.julian_day import J0
    j0 = J0(2004, 5, 12)
    assert j0 == pytest.approx(2453137.5)
    jd = j0 + (14 + 45 / 60 + 30 / 3600) / 24
    assert jd == pytest.approx(2453138.115, abs=1e-3)


# --- Algorithm 5.3: LST (Example 5.6) -------------------------------------
def test_lst():
    from curtis.algorithms.alg_5_3_lst import LST
    EL = 139 + 47 / 60
    lst = LST(2004, 3, 3, 4.5, EL)
    assert lst == pytest.approx(8.57688, abs=1e-4)


# --- Algorithm 5.4: rv_from_observe (Example 5.10) ------------------------
def test_rv_from_observe():
    from curtis.algorithms.alg_5_4_rv_from_observe import rv_from_observe
    r, v = rv_from_observe(2551.0, 0.0, 90.0, 0.1130, 30.0, 0.05651,
                           300.0, 60.0, 0.0,
                           f=1.0 / 298.256421867, Re=6378.13655,
                           wE=7.292115e-5)
    assert r == pytest.approx([3830.68, -2216.47, 6605.09], abs=1e-2)
    assert v == pytest.approx([1.50357, -4.56099, -0.291536], abs=1e-4)


# --- Algorithms 5.5/5.6: Gauss (Example 5.11) -----------------------------
def test_gauss():
    from curtis.algorithms.alg_5_5_gauss import gauss

    mu = 398600.0
    Re = 6378.0
    f = 1.0 / 298.26
    H = 1.0
    phi = 40.0 * deg
    t = [0.0, 118.104, 237.577]
    ra = np.array([43.5365, 54.4196, 64.3178]) * deg
    dec = np.array([-8.78334, -12.0739, -15.1054]) * deg
    theta = np.array([44.5065, 45.000, 45.4992]) * deg

    fac1 = Re / math.sqrt(1.0 - (2.0 * f - f * f) * math.sin(phi) ** 2)
    fac2 = (Re * (1.0 - f) ** 2
            / math.sqrt(1.0 - (2.0 * f - f * f) * math.sin(phi) ** 2)
            + H) * math.sin(phi)
    R = np.zeros((3, 3))
    Rho = np.zeros((3, 3))
    for i in range(3):
        R[i] = [(fac1 + H) * math.cos(phi) * math.cos(theta[i]),
                (fac1 + H) * math.cos(phi) * math.sin(theta[i]),
                fac2]
        Rho[i] = [math.cos(dec[i]) * math.cos(ra[i]),
                  math.cos(dec[i]) * math.sin(ra[i]),
                  math.sin(dec[i])]

    r, v, r_old, v_old = gauss(Rho[0], Rho[1], Rho[2],
                               R[0], R[1], R[2], *t, mu=mu)

    # Without iterative improvement.
    assert r_old == pytest.approx([5659.03, 6533.74, 3270.15], abs=1e-2)
    assert v_old == pytest.approx([-3.90774, 5.05735, -2.22224], abs=1e-4)
    # With iterative improvement.
    assert r == pytest.approx([5662.04, 6537.95, 3269.05], abs=1e-2)
    assert v == pytest.approx([-3.88542, 5.12141, -2.2434], abs=1e-4)


# --- D.16: month_planet_names ---------------------------------------------
def test_month_planet_names():
    from curtis.algorithms.month_planet_names import month_planet_names
    month, planet = month_planet_names(8, 3)
    assert month == "August"
    assert planet == "Earth"


# --- Algorithm 8.1: planet_elements_and_sv (Example 8.7) ------------------
def test_planet_elements_and_sv():
    from curtis.algorithms.alg_8_1_planet_elements_and_sv import (
        planet_elements_and_sv,
    )
    mu = 1.327124e11
    coe, r, v, jd = planet_elements_and_sv(3, 2003, 8, 27, 12, 0, 0, mu=mu)
    h, e, RA, incl, w, TA, a, w_hat, L, M, E = coe
    assert jd == pytest.approx(2452879.0, abs=1e-3)
    assert h == pytest.approx(4.4551e9, rel=1e-4)
    assert e == pytest.approx(0.0167088, abs=1e-6)
    assert RA == pytest.approx(348.554, abs=1e-3)
    assert w == pytest.approx(114.405, abs=1e-3)
    assert TA == pytest.approx(230.812, abs=1e-3)
    assert a == pytest.approx(1.49598e8, rel=1e-4)
    assert M == pytest.approx(232.308, abs=1e-3)
    assert E == pytest.approx(231.558, abs=1e-3)
    assert r == pytest.approx([1.35589e8, -6.68029e7, 286.909], rel=1e-4)
    assert v == pytest.approx([12.6804, 26.61, -0.000212731], rel=1e-4)


# --- Algorithm 8.2: interplanetary (Example 8.8) --------------------------
def test_interplanetary():
    from curtis.algorithms.alg_4_1_coe_from_sv import coe_from_sv
    from curtis.algorithms.alg_8_2_interplanetary import interplanetary

    mu = 1.327124e11
    planet1, planet2, trajectory = interplanetary(
        [3, 1996, 11, 7, 0, 0, 0], [4, 1997, 9, 12, 0, 0, 0], mu=mu)
    R1, Vp1, jd1 = planet1
    R2, Vp2, jd2 = planet2
    V1, V2 = trajectory

    assert jd1 == pytest.approx(2450394.5, abs=1e-3)
    assert jd2 == pytest.approx(2450703.5, abs=1e-3)
    assert V1 == pytest.approx([-24.4282, 21.7819, 0.948049], rel=1e-4)
    assert V2 == pytest.approx([22.1581, -0.19666, -0.457847], rel=1e-3)

    vinf1 = V1 - Vp1
    vinf2 = V2 - Vp2
    assert np.linalg.norm(vinf1) == pytest.approx(3.16513, abs=1e-4)
    assert np.linalg.norm(vinf2) == pytest.approx(2.88518, abs=1e-4)

    coe = coe_from_sv(R1, V1, mu=mu)
    coe2 = coe_from_sv(R2, V2, mu=mu)
    assert coe[1] == pytest.approx(0.205785, abs=1e-5)
    assert coe[6] == pytest.approx(1.84742e8, rel=1e-4)
    assert coe[5] / deg == pytest.approx(340.039, abs=1e-2)
    assert coe2[5] / deg == pytest.approx(199.695, abs=1e-2)
