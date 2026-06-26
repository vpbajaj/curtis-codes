"""Algorithms 5.5 and 5.6 (Appendix D.15) - Gauss's method of preliminary
orbit determination with iterative improvement.

MATLAB source: ``gauss.m`` driven by ``Example_5_11.m``.
"""

import math

import numpy as np

from ..constants import MU_EARTH
from .alg_3_3_kepler_U import kepler_U
from .lagrange_coefficients import f_and_g


def posroot(Roots):
    """Extract the positive real root from a vector of polynomial roots.

    If more than one positive real root exists the candidates are printed
    and the first is returned (the MATLAB original prompts the user).
    """
    posroots = [r.real for r in Roots
                if abs(r.imag) < 1e-9 and r.real > 0]

    if len(posroots) == 0:
        raise ValueError("** There are no positive roots.")
    elif len(posroots) == 1:
        return posroots[0]
    else:
        print("\n\n ** There are two or more positive roots.")
        for i, r in enumerate(posroots, start=1):
            print(f"\n root #{i} = {r:g}")
        print(f"\n Using the first root: {posroots[0]:g}")
        return posroots[0]


def gauss(Rho1, Rho2, Rho3, R1, R2, R3, t1, t2, t3,
          mu=MU_EARTH, tol=1.0e-8, nmax=1000):
    """Gauss's method (with iterative improvement) for the state vector.

    Parameters
    ----------
    Rho1, Rho2, Rho3 : array_like
        Direction cosine vectors of the satellite at t1, t2, t3.
    R1, R2, R3 : array_like
        Observation site position vectors at t1, t2, t3 (km).
    t1, t2, t3 : float
        Times of the observations (s).
    mu : float
        Gravitational parameter (km^3/s^2).

    Returns
    -------
    tuple
        ``(r, v, r_old, v_old)`` - the iteratively improved state vector
        ``(r, v)`` and the un-improved state vector ``(r_old, v_old)``,
        all in km and km/s.
    """
    Rho1 = np.asarray(Rho1, dtype=float)
    Rho2 = np.asarray(Rho2, dtype=float)
    Rho3 = np.asarray(Rho3, dtype=float)
    R1 = np.asarray(R1, dtype=float)
    R2 = np.asarray(R2, dtype=float)
    R3 = np.asarray(R3, dtype=float)

    # Equations 5.98 and 5.101 - time intervals.
    tau1 = t1 - t2
    tau3 = t3 - t2
    tau = tau3 - tau1

    # Independent cross products among the direction cosine vectors.
    p1 = np.cross(Rho2, Rho3)
    p2 = np.cross(Rho1, Rho3)
    p3 = np.cross(Rho1, Rho2)

    # Equation 5.108
    Do = np.dot(Rho1, p1)

    # Equations 5.109b, 5.110b and 5.111b.
    D = np.array([
        [np.dot(R1, p1), np.dot(R1, p2), np.dot(R1, p3)],
        [np.dot(R2, p1), np.dot(R2, p2), np.dot(R2, p3)],
        [np.dot(R3, p1), np.dot(R3, p2), np.dot(R3, p3)],
    ])

    # Equation 5.115b
    E = np.dot(R2, Rho2)

    # Equations 5.112b and 5.112c.
    A = 1.0 / Do * (-D[0, 1] * tau3 / tau + D[1, 1] + D[2, 1] * tau1 / tau)
    B = 1.0 / 6.0 / Do * (D[0, 1] * (tau3**2 - tau**2) * tau3 / tau
                          + D[2, 1] * (tau**2 - tau1**2) * tau1 / tau)

    # Equations 5.117 - coefficients of the 8th order polynomial.
    a = -(A**2 + 2.0 * A * E + np.linalg.norm(R2) ** 2)
    b = -2.0 * mu * B * (A + E)
    c = -(mu * B) ** 2

    # Equation 5.116 - roots of the 8th order polynomial.
    Roots = np.roots([1, 0, a, 0, 0, b, 0, 0, c])
    x = posroot(Roots)

    # Equations 5.99a, 5.99b, 5.100a, 5.100b.
    f1 = 1.0 - 0.5 * mu * tau1**2 / x**3
    f3 = 1.0 - 0.5 * mu * tau3**2 / x**3
    g1 = tau1 - 1.0 / 6.0 * mu * (tau1 / x) ** 3
    g3 = tau3 - 1.0 / 6.0 * mu * (tau3 / x) ** 3

    # Equation 5.112a
    rho2 = A + mu * B / x**3

    # Equations 5.113 and 5.114.
    rho1 = 1.0 / Do * (
        (6.0 * (D[2, 0] * tau1 / tau3 + D[1, 0] * tau / tau3) * x**3
         + mu * D[2, 0] * (tau**2 - tau1**2) * tau1 / tau3)
        / (6.0 * x**3 + mu * (tau**2 - tau3**2)) - D[0, 0])
    # NOTE: The printed appendix code (Eq 5.114) has the denominator as
    # (tau^2 - tau1^2), but the derivation in Section 5.10 (Example 5.11,
    # Step 9) and the appendix's own printed output use (tau^2 - tau3^2).
    # The latter is used here so the result matches the book.
    rho3 = 1.0 / Do * (
        (6.0 * (D[0, 2] * tau3 / tau1 - D[1, 2] * tau / tau1) * x**3
         + mu * D[0, 2] * (tau**2 - tau3**2) * tau3 / tau1)
        / (6.0 * x**3 + mu * (tau**2 - tau3**2)) - D[2, 2])

    # Equation 5.86 - position vectors.
    r1 = R1 + rho1 * Rho1
    r2 = R2 + rho2 * Rho2
    r3 = R3 + rho3 * Rho3

    # Equation 5.118 - velocity at the middle observation.
    v2 = (-f3 * r1 + f1 * r3) / (f1 * g3 - f3 * g1)

    # Save the initial (un-improved) estimates of r2 and v2.
    r_old = r2.copy()
    v_old = v2.copy()

    # --- Algorithm 5.6: iterative improvement -----------------------------
    rho1_old, rho2_old, rho3_old = rho1, rho2, rho3
    diff1 = diff2 = diff3 = 1.0
    n = 0

    while ((diff1 > tol) and (diff2 > tol) and (diff3 > tol)) and (n < nmax):
        n += 1

        # Quantities required by the universal Kepler's equation.
        ro = np.linalg.norm(r2)
        vo = np.linalg.norm(v2)
        vro = np.dot(v2, r2) / ro
        a_inv = 2.0 / ro - vo**2 / mu

        # Universal anomalies at tau1 and tau3.
        x1 = kepler_U(tau1, ro, vro, a_inv, mu=mu)
        x3 = kepler_U(tau3, ro, vro, a_inv, mu=mu)

        # Lagrange f and g coefficients at tau1 and tau3.
        ff1, gg1 = f_and_g(x1, tau1, ro, a_inv, mu=mu)
        ff3, gg3 = f_and_g(x3, tau3, ro, a_inv, mu=mu)

        # Average old and new.
        f1 = (f1 + ff1) / 2.0
        f3 = (f3 + ff3) / 2.0
        g1 = (g1 + gg1) / 2.0
        g3 = (g3 + gg3) / 2.0

        # Equations 5.96 and 5.97.
        c1 = g3 / (f1 * g3 - f3 * g1)
        c3 = -g1 / (f1 * g3 - f3 * g1)

        # Equations 5.109a, 5.110a and 5.111a.
        rho1 = 1.0 / Do * (-D[0, 0] + 1.0 / c1 * D[1, 0] - c3 / c1 * D[2, 0])
        rho2 = 1.0 / Do * (-c1 * D[0, 1] + D[1, 1] - c3 * D[2, 1])
        rho3 = 1.0 / Do * (-c1 / c3 * D[0, 2] + 1.0 / c3 * D[1, 2] - D[2, 2])

        # Equation 5.86 - updated position vectors.
        r1 = R1 + rho1 * Rho1
        r2 = R2 + rho2 * Rho2
        r3 = R3 + rho3 * Rho3

        # Equation 5.118 - updated velocity.
        v2 = (-f3 * r1 + f1 * r3) / (f1 * g3 - f3 * g1)

        # Differences upon which to base convergence.
        diff1 = abs(rho1 - rho1_old)
        diff2 = abs(rho2 - rho2_old)
        diff3 = abs(rho3 - rho3_old)

        rho1_old, rho2_old, rho3_old = rho1, rho2, rho3

    print(f"\n( **Number of Gauss improvement iterations = {n})\n")
    if n >= nmax:
        print(f"\n\n **Number of iterations exceeds {nmax} \n\n ")

    # Return the state vector for the central observation.
    r = r2
    v = v2

    return r, v, r_old, v_old


def _example_5_11():
    """Reproduce the output of ``Example_5_11.m``."""
    from .alg_4_1_coe_from_sv import coe_from_sv

    deg = math.pi / 180.0
    mu = 398600.0
    Re = 6378.0
    f = 1.0 / 298.26

    # Input data.
    H = 1.0
    phi = 40.0 * deg
    t = np.array([0.0, 118.104, 237.577])
    ra = np.array([43.5365, 54.4196, 64.3178]) * deg
    dec = np.array([-8.78334, -12.0739, -15.1054]) * deg
    theta = np.array([44.5065, 45.000, 45.4992]) * deg

    # Equations 5.56 and 5.57.
    fac1 = Re / math.sqrt(1.0 - (2.0 * f - f * f) * math.sin(phi) ** 2)
    fac2 = (Re * (1.0 - f) ** 2
            / math.sqrt(1.0 - (2.0 * f - f * f) * math.sin(phi) ** 2)
            + H) * math.sin(phi)
    R = np.zeros((3, 3))
    Rho = np.zeros((3, 3))
    for i in range(3):
        R[i, 0] = (fac1 + H) * math.cos(phi) * math.cos(theta[i])
        R[i, 1] = (fac1 + H) * math.cos(phi) * math.sin(theta[i])
        R[i, 2] = fac2
        Rho[i, 0] = math.cos(dec[i]) * math.cos(ra[i])
        Rho[i, 1] = math.cos(dec[i]) * math.sin(ra[i])
        Rho[i, 2] = math.sin(dec[i])

    r, v, r_old, v_old = gauss(Rho[0], Rho[1], Rho[2],
                               R[0], R[1], R[2],
                               t[0], t[1], t[2], mu=mu)

    coe_old = coe_from_sv(r_old, v_old, mu=mu)
    coe = coe_from_sv(r, v, mu=mu)

    print("-" * 52)
    print(" Example 5.11: Orbit determination by the Gauss method\n")
    print(f" Radius of earth (km)               = {Re:g}")
    print(f" Flattening factor                  = {f:g}")
    print(f" Gravitational parameter (km^3/s^2) = {mu:g}\n")
    print(" Input data:\n")
    print(f" Latitude (deg)             = {phi / deg:g}")
    print(f" Altitude above sea level (km) = {H:g}\n")
    print(" Observations:")
    print("   Time (s)   Right ascension (deg)   Declination (deg)"
          "   Local sidereal time (deg)")
    for i in range(3):
        print(f"   {t[i]:9.4g} {ra[i] / deg:17.4f} {dec[i] / deg:19.4f}"
              f" {theta[i] / deg:23.4f}")

    print("\n Solution:\n")
    print(" Without iterative improvement...\n")
    _print_state_and_coe(r_old, v_old, coe_old, mu, deg)

    print("\n With iterative improvement...\n")
    _print_state_and_coe(r, v, coe, mu, deg)
    print("-" * 52)


def _print_state_and_coe(r, v, coe, mu, deg):
    print(f" r (km)   = [{r[0]:g}, {r[1]:g}, {r[2]:g}]")
    print(f" v (km/s) = [{v[0]:g}, {v[1]:g}, {v[2]:g}]")
    print(f"   Angular momentum (km^2/s)  = {coe[0]:g}")
    print(f"   Eccentricity               = {coe[1]:g}")
    print(f"   RA of ascending node (deg) = {coe[2] / deg:g}")
    print(f"   Inclination (deg)          = {coe[3] / deg:g}")
    print(f"   Argument of perigee (deg)  = {coe[4] / deg:g}")
    print(f"   True anomaly (deg)         = {coe[5] / deg:g}")
    print(f"   Semimajor axis (km)        = {coe[6]:g}")
    print(f"   Periapse radius (km)       = {coe[0]**2 / mu / (1 + coe[1]):g}")
    if coe[1] < 1:
        T = 2 * math.pi / math.sqrt(mu) * coe[6] ** 1.5
        print("   Period:")
        print(f"     Seconds                  = {T:g}")
        print(f"     Minutes                  = {T / 60:g}")
        print(f"     Hours                    = {T / 3600:g}")
        print(f"     Days                     = {T / 24 / 3600:g}")


if __name__ == "__main__":
    _example_5_11()
