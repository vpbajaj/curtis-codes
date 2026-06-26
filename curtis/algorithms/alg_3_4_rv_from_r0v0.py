"""Algorithm 3.4 (Appendix D.7) - Calculation of the state vector
``(r, v)`` given the initial state vector ``(r0, v0)`` and the time lapse
``dt``.

MATLAB source: ``rv_from_r0v0.m`` driven by ``Example_3_07.m``.
"""

import numpy as np

from ..constants import MU_EARTH
from .alg_3_3_kepler_U import kepler_U
from .lagrange_coefficients import f_and_g, fDot_and_gDot


def rv_from_r0v0(R0, V0, t, mu=MU_EARTH):
    """Compute the state vector after a time lapse from an initial state.

    Parameters
    ----------
    R0 : array_like
        Initial position vector (km).
    V0 : array_like
        Initial velocity vector (km/s).
    t : float
        Elapsed time (s).
    mu : float
        Gravitational parameter (km^3/s^2).

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray)
        Final position vector ``R`` (km) and velocity vector ``V`` (km/s).
    """
    R0 = np.asarray(R0, dtype=float)
    V0 = np.asarray(V0, dtype=float)

    # Magnitudes of R0 and V0.
    r0 = np.linalg.norm(R0)
    v0 = np.linalg.norm(V0)

    # Initial radial velocity.
    vr0 = np.dot(R0, V0) / r0

    # Reciprocal of the semimajor axis (from the energy equation).
    alpha = 2.0 / r0 - v0**2 / mu

    # Universal anomaly.
    x = kepler_U(t, r0, vr0, alpha, mu=mu)

    # f and g functions.
    f, g = f_and_g(x, t, r0, alpha, mu=mu)

    # Final position vector and its magnitude.
    R = f * R0 + g * V0
    r = np.linalg.norm(R)

    # Derivatives of f and g.
    fdot, gdot = fDot_and_gDot(x, r, r0, alpha, mu=mu)

    # Final velocity vector.
    V = fdot * R0 + gdot * V0

    return R, V


def _example_3_07():
    """Reproduce the output of ``Example_3_07.m``."""
    R0 = np.array([7000.0, -12124.0, 0.0])
    V0 = np.array([2.6679, 4.6210, 0.0])
    t = 3600.0

    R, V = rv_from_r0v0(R0, V0, t)

    print("-" * 52)
    print(" Example 3.7\n")
    print(" Initial position vector (km):")
    print(f"   r0 = ({R0[0]:g}, {R0[1]:g}, {R0[2]:g})\n")
    print(" Initial velocity vector (km/s):")
    print(f"   v0 = ({V0[0]:g}, {V0[1]:g}, {V0[2]:g})\n")
    print(f" Elapsed time = {t:g} s\n")
    print(" Final position vector (km):")
    print(f"   r = ({R[0]:g}, {R[1]:g}, {R[2]:g})\n")
    print(" Final velocity vector (km/s):")
    print(f"   v = ({V[0]:g}, {V[1]:g}, {V[2]:g})")
    print("-" * 52)


if __name__ == "__main__":
    _example_3_07()
