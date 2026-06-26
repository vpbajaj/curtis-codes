"""Algorithm 5.4 (Appendix D.14) - Calculation of the state vector from
measurements of range, angular position and their rates.

MATLAB source: ``rv_from_observe.m`` driven by ``Example_5_10.m``.
"""

import math

import numpy as np

from ..constants import (
    F_EARTH,
    MU_EARTH_PRECISE,
    RE_EARTH_EQ,
    WE_EARTH,
)


def rv_from_observe(rho, rhodot, A, Adot, a, adot, theta, phi, H,
                    f=F_EARTH, Re=RE_EARTH_EQ, wE=WE_EARTH):
    """Geocentric equatorial state vector from radar observations.

    Parameters
    ----------
    rho : float
        Slant range of the object (km).
    rhodot : float
        Range rate (km/s).
    A : float
        Azimuth of the object relative to the site (degrees).
    Adot : float
        Azimuth rate (degrees/s).
    a : float
        Elevation angle (degrees).
    adot : float
        Elevation rate (degrees/s).
    theta : float
        Local sidereal time of the tracking site (degrees).
    phi : float
        Geodetic latitude of the site (degrees).
    H : float
        Elevation of the site above sea level (km).
    f : float
        Earth's flattening factor.
    Re : float
        Equatorial radius of the Earth (km).
    wE : float
        Angular velocity of the Earth (rad/s).

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray)
        Geocentric equatorial position ``r`` (km) and velocity ``v`` (km/s).
    """
    deg = math.pi / 180.0
    omega = np.array([0.0, 0.0, wE])

    # Convert angular quantities from degrees to radians.
    A = A * deg
    Adot = Adot * deg
    a = a * deg
    adot = adot * deg
    theta = theta * deg
    phi = phi * deg

    # Equation 5.56 - position vector of the tracking site.
    fac1 = Re / math.sqrt(1.0 - (2.0 * f - f * f) * math.sin(phi) ** 2)
    fac2 = Re * (1.0 - f) ** 2 / math.sqrt(
        1.0 - (2.0 * f - f * f) * math.sin(phi) ** 2)
    R = np.array([
        (fac1 + H) * math.cos(phi) * math.cos(theta),
        (fac1 + H) * math.cos(phi) * math.sin(theta),
        (fac2 + H) * math.sin(phi),
    ])

    # Equation 5.66 - inertial velocity of the site.
    Rdot = np.cross(omega, R)

    # Equation 5.83a - topocentric declination.
    dec = math.asin(math.cos(phi) * math.cos(A) * math.cos(a)
                    + math.sin(phi) * math.sin(a))

    # Equation 5.83b - hour angle.
    h = math.acos((math.cos(phi) * math.sin(a)
                   - math.sin(phi) * math.cos(A) * math.cos(a))
                  / math.cos(dec))
    if (A > 0) and (A < math.pi):
        h = 2.0 * math.pi - h

    # Equation 5.83c - right ascension.
    RA = theta - h

    # Equation 5.57 - direction cosine unit vector.
    Rho = np.array([
        math.cos(RA) * math.cos(dec),
        math.sin(RA) * math.cos(dec),
        math.sin(dec),
    ])

    # Equation 5.63 - geocentric position vector.
    r = R + rho * Rho

    # Equation 5.84 - declination rate.
    decdot = (-Adot * math.cos(phi) * math.sin(A) * math.cos(a)
              + adot * (math.sin(phi) * math.cos(a)
                        - math.cos(phi) * math.cos(A) * math.sin(a))
              ) / math.cos(dec)

    # Equation 5.85 - right ascension rate.
    RAdot = (wE
             + (Adot * math.cos(A) * math.cos(a)
                - adot * math.sin(A) * math.sin(a)
                + decdot * math.sin(A) * math.cos(a) * math.tan(dec))
             / (math.cos(phi) * math.sin(a)
                - math.sin(phi) * math.cos(A) * math.cos(a)))

    # Equations 5.69 and 5.72 - direction cosine rate vector.
    Rhodot = np.array([
        -RAdot * math.sin(RA) * math.cos(dec)
        - decdot * math.cos(RA) * math.sin(dec),
        RAdot * math.cos(RA) * math.cos(dec)
        - decdot * math.sin(RA) * math.sin(dec),
        decdot * math.cos(dec),
    ])

    # Equation 5.64 - geocentric velocity vector.
    v = Rdot + rhodot * Rho + rho * Rhodot

    return r, v


def _example_5_10():
    """Reproduce the output of ``Example_5_10.m``."""
    from .alg_4_1_coe_from_sv import coe_from_sv

    deg = math.pi / 180.0
    f = 1.0 / 298.256421867
    Re = 6378.13655
    wE = 7.292115e-5
    mu = MU_EARTH_PRECISE

    # Input data for Example 5.10.
    rho = 2551.0
    rhodot = 0.0
    A = 90.0
    Adot = 0.1130
    a = 30.0
    adot = 0.05651
    theta = 300.0
    phi = 60.0
    H = 0.0

    r, v = rv_from_observe(rho, rhodot, A, Adot, a, adot, theta, phi, H,
                           f=f, Re=Re, wE=wE)

    coe = coe_from_sv(r, v, mu=mu)
    h, e, RA, incl, w, TA, aa = coe
    rp = h**2 / mu / (1.0 + e)

    print("-" * 52)
    print(" Example 5.10\n")
    print(" Input data:\n")
    print(f"   Slant range (km)              = {rho:g}")
    print(f"   Slant range rate (km/s)       = {rhodot:g}")
    print(f"   Azimuth (deg)                 = {A:g}")
    print(f"   Azimuth rate (deg/s)          = {Adot:g}")
    print(f"   Elevation (deg)               = {a:g}")
    print(f"   Elevation rate (deg/s)        = {adot:g}")
    print(f"   Local sidereal time (deg)     = {theta:g}")
    print(f"   Latitude (deg)                = {phi:g}")
    print(f"   Altitude above sea level (km) = {H:g}\n")
    print(" Solution:\n")
    print(" State vector:")
    print(f"   r (km)   = [{r[0]:g}, {r[1]:g}, {r[2]:g}]")
    print(f"   v (km/s) = [{v[0]:g}, {v[1]:g}, {v[2]:g}]\n")
    print(" Orbital elements:")
    print(f"   Angular momentum (km^2/s)  = {h:g}")
    print(f"   Eccentricity               = {e:g}")
    print(f"   Inclination (deg)          = {incl / deg:g}")
    print(f"   RA of ascending node (deg) = {RA / deg:g}")
    print(f"   Argument of perigee (deg)  = {w / deg:g}")
    print(f"   True anomaly (deg)         = {TA / deg:g}")
    print(f"   Semimajor axis (km)        = {aa:g}")
    print(f"   Perigee radius (km)        = {rp:g}")
    if e < 1:
        T = 2 * math.pi / math.sqrt(mu) * aa ** 1.5
        print("   Period:")
        print(f"     Seconds                  = {T:g}")
        print(f"     Minutes                  = {T / 60:g}")
        print(f"     Hours                    = {T / 3600:g}")
        print(f"     Days                     = {T / 24 / 3600:g}")
    print("-" * 52)


if __name__ == "__main__":
    _example_5_10()
