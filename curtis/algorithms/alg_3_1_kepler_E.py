"""Algorithm 3.1 (Appendix D.2) - Solution of Kepler's equation
``E - e*sin(E) = M`` by Newton's method.

MATLAB source: ``kepler_E.m`` driven by ``Example_3_02.m``.
"""

import math


def kepler_E(e, M, tol=1.0e-8):
    """Solve Kepler's equation for the eccentric anomaly.

    Parameters
    ----------
    e : float
        Eccentricity (0 <= e < 1).
    M : float
        Mean anomaly (radians).
    tol : float
        Error tolerance on the Newton iteration.

    Returns
    -------
    float
        Eccentric anomaly ``E`` (radians).
    """
    # Select a starting value for E.
    if M < math.pi:
        E = M + e / 2.0
    else:
        E = M - e / 2.0

    # Iterate on Equation 3.14 until E is within the error tolerance.
    ratio = 1.0
    while abs(ratio) > tol:
        ratio = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E = E - ratio

    return E


def _example_3_02():
    """Reproduce the output of ``Example_3_02.m``."""
    e = 0.37255
    M = 3.6029
    E = kepler_E(e, M)

    print("-" * 52)
    print(" Example 3.2\n")
    print(f" Eccentricity                = {e:g}")
    print(f" Mean anomaly (radians)      = {M:g}\n")
    print(f" Eccentric anomaly (radians) = {E:g}")
    print("-" * 52)


if __name__ == "__main__":
    _example_3_02()
