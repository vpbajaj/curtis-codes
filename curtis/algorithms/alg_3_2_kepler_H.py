"""Algorithm 3.2 (Appendix D.3) - Solution of Kepler's equation for the
hyperbola ``e*sinh(F) - F = M`` by Newton's method.

MATLAB source: ``kepler_H.m`` driven by ``Example_3_05.m``.
"""

import math


def kepler_H(e, M, tol=1.0e-8):
    """Solve the hyperbolic Kepler equation for the eccentric anomaly.

    Parameters
    ----------
    e : float
        Eccentricity (e > 1).
    M : float
        Hyperbolic mean anomaly (dimensionless).
    tol : float
        Error tolerance on the Newton iteration.

    Returns
    -------
    float
        Hyperbolic eccentric anomaly ``F`` (dimensionless).
    """
    # Starting value for F.
    F = M

    # Iterate on Equation 3.42 until F is within the error tolerance.
    ratio = 1.0
    while abs(ratio) > tol:
        ratio = (e * math.sinh(F) - F - M) / (e * math.cosh(F) - 1.0)
        F = F - ratio

    return F


def _example_3_05():
    """Reproduce the output of ``Example_3_05.m``."""
    e = 2.7696
    M = 40.69
    F = kepler_H(e, M)

    print("-" * 52)
    print(" Example 3.5\n")
    print(f" Eccentricity                  = {e:g}")
    print(f" Hyperbolic mean anomaly       = {M:g}\n")
    print(f" Hyperbolic eccentric anomaly  = {F:g}")
    print("-" * 52)


if __name__ == "__main__":
    _example_3_05()
