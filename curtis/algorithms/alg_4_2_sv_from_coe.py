"""Algorithm 4.2 (Appendix D.9) - Calculation of the state vector from the
classical orbital elements.

MATLAB source: ``sv_from_coe.m`` driven by ``Example_4_05.m``.
"""

import math

import numpy as np

from ..constants import MU_EARTH


def sv_from_coe(coe, mu=MU_EARTH):
    """Compute the state vector from the classical orbital elements.

    Parameters
    ----------
    coe : array_like
        Orbital elements ``[h, e, RA, incl, w, TA]`` with angles in
        radians and ``h`` in km^2/s.
    mu : float
        Gravitational parameter (km^3/s^2).

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray)
        Position vector ``r`` (km) and velocity vector ``v`` (km/s) in the
        geocentric equatorial frame.
    """
    h = coe[0]
    e = coe[1]
    RA = coe[2]
    incl = coe[3]
    w = coe[4]
    TA = coe[5]

    # Equations 4.37 and 4.38 (rp and vp are column vectors).
    rp = (h**2 / mu) * (1.0 / (1.0 + e * math.cos(TA))) * (
        math.cos(TA) * np.array([1.0, 0.0, 0.0])
        + math.sin(TA) * np.array([0.0, 1.0, 0.0]))
    vp = (mu / h) * (
        -math.sin(TA) * np.array([1.0, 0.0, 0.0])
        + (e + math.cos(TA)) * np.array([0.0, 1.0, 0.0]))

    # Equation 4.39 - rotation about the z-axis through RA.
    R3_W = np.array([
        [math.cos(RA), math.sin(RA), 0.0],
        [-math.sin(RA), math.cos(RA), 0.0],
        [0.0, 0.0, 1.0],
    ])

    # Equation 4.40 - rotation about the x-axis through incl.
    R1_i = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(incl), math.sin(incl)],
        [0.0, -math.sin(incl), math.cos(incl)],
    ])

    # Equation 4.41 - rotation about the z-axis through w.
    R3_w = np.array([
        [math.cos(w), math.sin(w), 0.0],
        [-math.sin(w), math.cos(w), 0.0],
        [0.0, 0.0, 1.0],
    ])

    # Equation 4.44 - transformation matrix from perifocal to geocentric
    # equatorial frame.
    Q_pX = (R3_w @ R1_i @ R3_W).T

    # Equation 4.46 (r and v are column vectors).
    r = Q_pX @ rp
    v = Q_pX @ vp

    return r, v


def _example_4_05():
    """Reproduce the output of ``Example_4_05.m``."""
    deg = math.pi / 180.0
    mu = 398600.0

    h = 80000.0
    e = 1.4
    RA = 40.0
    incl = 30.0
    w = 60.0
    TA = 30.0

    coe = [h, e, RA * deg, incl * deg, w * deg, TA * deg]
    r, v = sv_from_coe(coe, mu=mu)

    print("-" * 52)
    print(" Example 4.5\n")
    print(f" Gravitational parameter (km^3/s^2) = {mu:g}\n")
    print(f" Angular momentum (km^2/s)  = {h:g}")
    print(f" Eccentricity               = {e:g}")
    print(f" Right ascension (deg)      = {RA:g}")
    print(f" Argument of perigee (deg)  = {w:g}")
    print(f" True anomaly (deg)         = {TA:g}\n")
    print(" State vector:")
    print(f"   r (km)   = [{r[0]:g}  {r[1]:g}  {r[2]:g}]")
    print(f"   v (km/s) = [{v[0]:g}  {v[1]:g}  {v[2]:g}]")
    print("-" * 52)


if __name__ == "__main__":
    _example_4_05()
