"""Appendix D.6 - Lagrange coefficients ``f`` and ``g`` and their time
derivatives (Equations 3.66), in universal-variable form.

MATLAB source: ``f_and_g.m`` and ``fDot_and_gDot.m``.
"""

import math

from ..constants import MU_EARTH
from .stumpff import stumpC, stumpS


def f_and_g(x, t, ro, a, mu=MU_EARTH):
    """Calculate the Lagrange ``f`` and ``g`` coefficients.

    Parameters
    ----------
    x : float
        Universal anomaly after time ``t`` (km^0.5).
    t : float
        Time elapsed since ``t0`` (s).
    ro : float
        Radial position at time ``t0`` (km).
    a : float
        Reciprocal of the semimajor axis (1/km).
    mu : float
        Gravitational parameter (km^3/s^2).

    Returns
    -------
    tuple(float, float)
        ``f`` (dimensionless) and ``g`` (s).
    """
    z = a * x**2

    # Equation 3.66a
    f = 1.0 - x**2 / ro * stumpC(z)
    # Equation 3.66b
    g = t - 1.0 / math.sqrt(mu) * x**3 * stumpS(z)

    return f, g


def fDot_and_gDot(x, r, ro, a, mu=MU_EARTH):
    """Calculate the time derivatives of the Lagrange coefficients.

    Parameters
    ----------
    x : float
        Universal anomaly after time ``t`` (km^0.5).
    r : float
        Radial position after time ``t`` (km).
    ro : float
        Radial position at time ``t0`` (km).
    a : float
        Reciprocal of the semimajor axis (1/km).
    mu : float
        Gravitational parameter (km^3/s^2).

    Returns
    -------
    tuple(float, float)
        ``fdot`` (1/s) and ``gdot`` (dimensionless).
    """
    z = a * x**2

    # Equation 3.66c
    fdot = math.sqrt(mu) / r / ro * (z * stumpS(z) - 1.0) * x
    # Equation 3.66d
    gdot = 1.0 - x**2 / r * stumpC(z)

    return fdot, gdot
