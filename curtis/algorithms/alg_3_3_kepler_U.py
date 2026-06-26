"""Algorithm 3.3 (Appendix D.5) - Solution of the universal Kepler's
equation using Newton's method.

MATLAB source: ``kepler_U.m`` driven by ``Example_3_06.m``.
"""

import math

from ..constants import MU_EARTH
from .stumpff import stumpC, stumpS


def kepler_U(dt, ro, vro, a, mu=MU_EARTH, tol=1.0e-8, nMax=1000):
    """Solve the universal Kepler equation for the universal anomaly ``x``.

    Parameters
    ----------
    dt : float
        Time since ``x = 0`` (s).
    ro : float
        Radial position when ``x = 0`` (km).
    vro : float
        Radial velocity when ``x = 0`` (km/s).
    a : float
        Reciprocal of the semimajor axis (1/km).
    mu : float
        Gravitational parameter (km^3/s^2).
    tol : float
        Convergence tolerance.
    nMax : int
        Maximum number of iterations.

    Returns
    -------
    float
        Universal anomaly ``x`` (km^0.5).
    """
    # Starting value for x.
    x = math.sqrt(mu) * abs(a) * dt

    # Iterate on Equation 3.62 until convergence.
    n = 0
    ratio = 1.0
    while abs(ratio) > tol and n <= nMax:
        n += 1
        C = stumpC(a * x**2)
        S = stumpS(a * x**2)
        F = (ro * vro / math.sqrt(mu) * x**2 * C
             + (1.0 - a * ro) * x**3 * S
             + ro * x - math.sqrt(mu) * dt)
        dFdx = (ro * vro / math.sqrt(mu) * x * (1.0 - a * x**2 * S)
                + (1.0 - a * ro) * x**2 * C + ro)
        ratio = F / dFdx
        x = x - ratio

    if n > nMax:
        print(f"\n **No. iterations of Kepler's equation = {n}")
        print(f"\n   F/dFdx                                = {F / dFdx:g}\n")

    return x


def _example_3_06():
    """Reproduce the output of ``Example_3_06.m``."""
    mu = 398600.0
    ro = 10000.0
    vro = 3.0752
    dt = 3600.0
    a = -19655.0

    # Universal Kepler's requires the reciprocal of the semimajor axis.
    x = kepler_U(dt, ro, vro, 1.0 / a, mu=mu)

    print("-" * 52)
    print(" Example 3.6\n")
    print(f" Initial radial coordinate (km) = {ro:g}")
    print(f" Initial radial velocity (km/s) = {vro:g}")
    print(f" Elapsed time (seconds)         = {dt:g}")
    print(f" Semimajor axis (km)            = {a:g}\n")
    print(f" Universal anomaly (km^0.5)     = {x:g}")
    print("-" * 52)


if __name__ == "__main__":
    _example_3_06()
