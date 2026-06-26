"""Algorithm 5.2 (Appendix D.11) - Solution of Lambert's problem.

MATLAB source: ``lambert.m`` driven by ``Example_5_02.m``.
"""

import cmath
import math

import numpy as np

from ..constants import MU_EARTH


# Complex-aware Stumpff functions.  MATLAB performs the bracket search and
# the Newton iteration in complex arithmetic (taking the real part where a
# real comparison is needed); these entire-function forms are valid for any
# z and reduce to Equations 3.49/3.50 on the real axis.
def _stumpC(z):
    if z == 0:
        return 0.5
    return (1.0 - cmath.cos(cmath.sqrt(z))) / z


def _stumpS(z):
    if z == 0:
        return 1.0 / 6.0
    sz = cmath.sqrt(z)
    return (sz - cmath.sin(sz)) / sz**3


def lambert(R1, R2, t, string="pro", mu=MU_EARTH, tol=1.0e-8, nmax=5000):
    """Solve Lambert's problem for the terminal velocity vectors.

    Parameters
    ----------
    R1, R2 : array_like
        Initial and final position vectors (km).
    t : float
        Time of flight from R1 to R2 (s).
    string : str
        ``'pro'`` for a prograde orbit, ``'retro'`` for retrograde.
    mu : float
        Gravitational parameter (km^3/s^2).
    tol : float
        Convergence tolerance.
    nmax : int
        Maximum number of Newton iterations.

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray)
        Velocity vectors ``V1`` and ``V2`` (km/s).
    """
    R1 = np.asarray(R1, dtype=float)
    R2 = np.asarray(R2, dtype=float)

    # Magnitudes of R1 and R2.
    r1 = np.linalg.norm(R1)
    r2 = np.linalg.norm(R2)

    c12 = np.cross(R1, R2)
    theta = math.acos(np.dot(R1, R2) / r1 / r2)

    # Determine whether the orbit is prograde or retrograde.
    if string == "pro":
        if c12[2] <= 0:
            theta = 2 * math.pi - theta
    elif string == "retro":
        if c12[2] >= 0:
            theta = 2 * math.pi - theta
    else:
        string = "pro"
        print("\n ** Prograde trajectory assumed.\n")

    # Equation 5.35
    A = math.sin(theta) * math.sqrt(r1 * r2 / (1.0 - math.cos(theta)))

    # Equation 5.38
    def y(z):
        return r1 + r2 + A * (z * _stumpS(z) - 1.0) / cmath.sqrt(_stumpC(z))

    # Equation 5.40
    def F(z, t):
        return ((y(z) / _stumpC(z)) ** 1.5 * _stumpS(z)
                + A * cmath.sqrt(y(z)) - math.sqrt(mu) * t)

    # Equation 5.43
    def dFdz(z):
        if z == 0:
            return (math.sqrt(2) / 40 * y(0) ** 1.5
                    + A / 8 * (cmath.sqrt(y(0))
                               + A * cmath.sqrt(1.0 / 2.0 / y(0))))
        else:
            C = _stumpC(z)
            S = _stumpS(z)
            return ((y(z) / C) ** 1.5
                    * (1.0 / 2.0 / z * (C - 3.0 * S / 2.0 / C)
                       + 3.0 * S**2 / 4.0 / C)
                    + A / 8 * (3.0 * S / C * cmath.sqrt(y(z))
                              + A * cmath.sqrt(C / y(z))))

    # Determine approximately where F(z,t) changes sign, and use that value
    # of z as the starting value for Equation 5.45.  (The real part is used
    # for the comparison, as in the MATLAB original.)
    z = -100.0
    while complex(F(z, t)).real < 0:
        z = z + 0.1

    # Iterate on Equation 5.45 until z is within the error tolerance.
    ratio = 1.0
    n = 0
    while abs(ratio) > tol and n <= nmax:
        n += 1
        ratio = complex(F(z, t) / dFdz(z)).real
        z = z - ratio

    if n >= nmax:
        print("\n\n **Number of iterations exceeds")
        print(f" {nmax} \n\n ")

    # Equation 5.46a
    f = (1.0 - y(z) / r1).real
    # Equation 5.46b
    g = (A * cmath.sqrt(y(z) / mu)).real
    # Equation 5.46d
    gdot = (1.0 - y(z) / r2).real

    # Equation 5.28
    V1 = 1.0 / g * (R2 - f * R1)
    # Equation 5.29
    V2 = 1.0 / g * (gdot * R2 - R1)

    return V1, V2


def _example_5_02():
    """Reproduce the output of ``Example_5_02.m``."""
    from .alg_4_1_coe_from_sv import coe_from_sv

    deg = math.pi / 180.0
    mu = 398600.0

    r1 = np.array([5000.0, 10000.0, 2100.0])
    r2 = np.array([-14600.0, 2500.0, 7000.0])
    dt = 3600.0
    string = "pro"

    v1, v2 = lambert(r1, r2, dt, string, mu=mu)

    coe = coe_from_sv(r1, v1, mu=mu)
    TA1 = coe[5]
    coe = coe_from_sv(r2, v2, mu=mu)
    TA2 = coe[5]

    print("-" * 52)
    print(" Example 5.2: Lambert's Problem\n")
    print(" Input data:\n")
    print(f" Gravitational parameter (km^3/s^2) = {mu:g}")
    print(f" r1 (km) = [{r1[0]:g}  {r1[1]:g}  {r1[2]:g}]")
    print(f" r2 (km) = [{r2[0]:g}  {r2[1]:g}  {r2[2]:g}]")
    print(f" Elapsed time (s) = {dt:g}\n")
    print(" Solution:\n")
    print(f" v1 (km/s) = [{v1[0]:g}  {v1[1]:g}  {v1[2]:g}]")
    print(f" v2 (km/s) = [{v2[0]:g}  {v2[1]:g}  {v2[2]:g}]\n")
    print(" Orbital elements:")
    print(f"   Angular momentum (km^2/s)  = {coe[0]:g}")
    print(f"   Eccentricity               = {coe[1]:g}")
    print(f"   Inclination (deg)          = {coe[3] / deg:g}")
    print(f"   RA of ascending node (deg) = {coe[2] / deg:g}")
    print(f"   Argument of perigee (deg)  = {coe[4] / deg:g}")
    print(f"   True anomaly initial (deg) = {TA1 / deg:g}")
    print(f"   True anomaly final   (deg) = {TA2 / deg:g}")
    print(f"   Semimajor axis (km)        = {coe[6]:g}")
    print(f"   Periapse radius (km)       = {coe[0]**2 / mu / (1 + coe[1]):g}")
    if coe[1] < 1:
        T = 2 * math.pi / math.sqrt(mu) * coe[6] ** 1.5
        print("   Period:")
        print(f"     Seconds                  = {T:g}")
        print(f"     Minutes                  = {T / 60:g}")
        print(f"     Hours                    = {T / 3600:g}")
        print(f"     Days                     = {T / 24 / 3600:g}")
    print("-" * 52)


if __name__ == "__main__":
    _example_5_02()
