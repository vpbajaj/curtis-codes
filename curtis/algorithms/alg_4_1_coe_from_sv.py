"""Algorithm 4.1 (Appendix D.8) - Calculation of the classical orbital
elements from the state vector.

MATLAB source: ``coe_from_sv.m`` driven by ``Example_4_03.m``.
"""

import math

import numpy as np

from ..constants import MU_EARTH


def coe_from_sv(R, V, mu=MU_EARTH):
    """Compute the classical orbital elements from the state vector.

    Parameters
    ----------
    R : array_like
        Position vector in the geocentric equatorial frame (km).
    V : array_like
        Velocity vector in the geocentric equatorial frame (km/s).
    mu : float
        Gravitational parameter (km^3/s^2).

    Returns
    -------
    numpy.ndarray
        Orbital elements ``[h, e, RA, incl, w, TA, a]`` with angles in
        radians, ``h`` in km^2/s and ``a`` in km.
    """
    eps = 1.0e-10

    R = np.asarray(R, dtype=float)
    V = np.asarray(V, dtype=float)

    r = np.linalg.norm(R)
    v = np.linalg.norm(V)

    vr = np.dot(R, V) / r

    H = np.cross(R, V)
    h = np.linalg.norm(H)

    # Equation 4.7
    incl = math.acos(H[2] / h)

    # Equation 4.8
    N = np.cross([0, 0, 1], H)
    n = np.linalg.norm(N)

    # Equation 4.9
    if n != 0:
        RA = math.acos(N[0] / n)
        if N[1] < 0:
            RA = 2 * math.pi - RA
    else:
        RA = 0.0

    # Equation 4.10
    E = 1.0 / mu * ((v**2 - mu / r) * R - r * vr * V)
    e = np.linalg.norm(E)

    # Equation 4.12 (incorporating the case e = 0)
    if n != 0:
        if e > eps:
            w = math.acos(np.dot(N, E) / n / e)
            if E[2] < 0:
                w = 2 * math.pi - w
        else:
            w = 0.0
    else:
        w = 0.0

    # Equation 4.13a (incorporating the case e = 0)
    if e > eps:
        TA = math.acos(np.dot(E, R) / e / r)
        if vr < 0:
            TA = 2 * math.pi - TA
    else:
        cp = np.cross(N, R)
        if cp[2] >= 0:
            TA = math.acos(np.dot(N, R) / n / r)
        else:
            TA = 2 * math.pi - math.acos(np.dot(N, R) / n / r)

    # Equation 2.61 (a < 0 for a hyperbola)
    a = h**2 / mu / (1.0 - e**2)

    return np.array([h, e, RA, incl, w, TA, a])


def _example_4_03():
    """Reproduce the output of ``Example_4_03.m``."""
    deg = math.pi / 180.0
    mu = 398600.0
    r = np.array([-6045.0, -3490.0, 2500.0])
    v = np.array([-3.457, 6.618, 2.533])

    coe = coe_from_sv(r, v, mu=mu)

    print("-" * 52)
    print(" Example 4.3\n")
    print(f" Gravitational parameter (km^3/s^2) = {mu:g}\n")
    print(" State vector:\n")
    print(f" r (km)                     = [{r[0]:g}  {r[1]:g}  {r[2]:g}]")
    print(f" v (km/s)                   = [{v[0]:g}  {v[1]:g}  {v[2]:g}]\n")
    print(f" Angular momentum (km^2/s)  = {coe[0]:g}")
    print(f" Eccentricity               = {coe[1]:g}")
    print(f" Right ascension (deg)      = {coe[2] / deg:g}")
    print(f" Inclination (deg)          = {coe[3] / deg:g}")
    print(f" Argument of perigee (deg)  = {coe[4] / deg:g}")
    print(f" True anomaly (deg)         = {coe[5] / deg:g}")
    print(f" Semimajor axis (km):       = {coe[6]:g}")

    if coe[1] < 1:
        T = 2 * math.pi / math.sqrt(mu) * coe[6] ** 1.5  # Equation 2.73
        print(" Period:")
        print(f"   Seconds                  = {T:g}")
        print(f"   Minutes                  = {T / 60:g}")
        print(f"   Hours                    = {T / 3600:g}")
        print(f"   Days                     = {T / 24 / 3600:g}")
    print("-" * 52)


if __name__ == "__main__":
    _example_4_03()
