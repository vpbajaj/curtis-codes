"""Appendix C.1 - ``accel_3body.m``.

Evaluates the acceleration of each member of a planar three-body system at
time ``t`` from their positions and velocities at that time (Equations C.8
and C.9).
"""

import numpy as np


def accel_3body(t, f, G, m):
    """Right-hand side of the planar three-body ODE system.

    Parameters
    ----------
    t : float
        Time (s). Unused (the system is autonomous) but kept for the
        solver's ``f(t, y)`` calling convention.
    f : array_like
        16-component state vector holding the position and velocity
        components of the three masses and the centre of mass at time
        ``t``: ``[r1x, r1y, r2x, r2y, r3x, r3y, rGx, rGy,
        v1x, v1y, v2x, v2y, v3x, v3y, vGx, vGy]``.
    G : float
        Gravitational constant (km^3/kg/s^2).
    m : array_like
        Masses ``[m1, m2, m3]`` (kg).

    Returns
    -------
    numpy.ndarray
        The 16-component time derivative ``df/dt`` (velocities and
        accelerations of the three masses and the centre of mass).
    """
    # For ease of reading, assign each component of f to a mnemonic variable.
    r1x, r1y = f[0], f[1]
    r2x, r2y = f[2], f[3]
    r3x, r3y = f[4], f[5]
    # f[6], f[7] are rGx, rGy (not needed for the accelerations).
    v1x, v1y = f[8], f[9]
    v2x, v2y = f[10], f[11]
    v3x, v3y = f[12], f[13]
    vGx, vGy = f[14], f[15]

    # Equations C.9 - cubes of the inter-body distances.
    r12 = np.linalg.norm([r2x - r1x, r2y - r1y]) ** 3
    r13 = np.linalg.norm([r3x - r1x, r3y - r1y]) ** 3
    r23 = np.linalg.norm([r3x - r2x, r3y - r2y]) ** 3

    # Equations C.8 - accelerations of the three masses.
    a1x = G * m[1] * (r2x - r1x) / r12 + G * m[2] * (r3x - r1x) / r13
    a1y = G * m[1] * (r2y - r1y) / r12 + G * m[2] * (r3y - r1y) / r13
    a2x = G * m[0] * (r1x - r2x) / r12 + G * m[2] * (r3x - r2x) / r23
    a2y = G * m[0] * (r1y - r2y) / r12 + G * m[2] * (r3y - r2y) / r23
    a3x = G * m[0] * (r1x - r3x) / r13 + G * m[1] * (r2x - r3x) / r23
    a3y = G * m[0] * (r1y - r3y) / r13 + G * m[1] * (r2y - r3y) / r23

    # Equation C.5a - the centre of mass has zero acceleration.
    aGx = 0.0
    aGy = 0.0

    # Place the velocity and acceleration components into dfdt.
    dfdt = np.array([
        v1x, v1y,
        v2x, v2y,
        v3x, v3y,
        vGx, vGy,
        a1x, a1y,
        a2x, a2y,
        a3x, a3y,
        aGx, aGy,
    ])

    return dfdt
