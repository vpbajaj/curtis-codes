"""Algorithm 8.2 (Appendix D.18) - Calculation of the spacecraft trajectory
from planet 1 to planet 2.

MATLAB source: ``interplanetary.m`` driven by ``Example_8_08.m``.
"""

import math

import numpy as np

from ..constants import MU_SUN
from .alg_5_2_lambert import lambert
from .alg_8_1_planet_elements_and_sv import planet_elements_and_sv


def interplanetary(depart, arrive, mu=MU_SUN):
    """Spacecraft trajectory from the SOI of planet 1 to that of planet 2.

    Parameters
    ----------
    depart : sequence
        ``[planet_id, year, month, day, hour, minute, second]`` at departure.
    arrive : sequence
        ``[planet_id, year, month, day, hour, minute, second]`` at arrival.
    mu : float
        Gravitational parameter of the sun (km^3/s^2).

    Returns
    -------
    tuple
        ``(planet1, planet2, trajectory)`` where ``planet1`` is
        ``(Rp1, Vp1, jd1)``, ``planet2`` is ``(Rp2, Vp2, jd2)`` and
        ``trajectory`` is ``(V1, V2)`` - the heliocentric departure and
        arrival velocities of the spacecraft (km/s).
    """
    planet_id, year, month, day, hour, minute, second = depart

    # Algorithm 8.1 for planet 1's state vector (orbital elements not needed).
    _, Rp1, Vp1, jd1 = planet_elements_and_sv(
        planet_id, year, month, day, hour, minute, second, mu=mu)

    planet_id, year, month, day, hour, minute, second = arrive

    # Algorithm 8.1 for planet 2's state vector.
    _, Rp2, Vp2, jd2 = planet_elements_and_sv(
        planet_id, year, month, day, hour, minute, second, mu=mu)

    tof = (jd2 - jd1) * 24 * 3600

    # Patched conic assumption.
    R1 = Rp1
    R2 = Rp2

    # Algorithm 5.2 to find the spacecraft's velocity at departure and
    # arrival, assuming a prograde trajectory.
    V1, V2 = lambert(R1, R2, tof, "pro", mu=mu)

    planet1 = (Rp1, Vp1, jd1)
    planet2 = (Rp2, Vp2, jd2)
    trajectory = (V1, V2)

    return planet1, planet2, trajectory


def _example_8_08():
    """Reproduce the output of ``Example_8_08.m``."""
    from .alg_4_1_coe_from_sv import coe_from_sv
    from .month_planet_names import month_planet_names

    deg = math.pi / 180.0
    mu = 1.327124e11

    # Data for planet 1 (Earth).
    depart = [3, 1996, 11, 7, 0, 0, 0]
    # Data for planet 2 (Mars).
    arrive = [4, 1997, 9, 12, 0, 0, 0]

    planet1, planet2, trajectory = interplanetary(depart, arrive, mu=mu)
    R1, Vp1, jd1 = planet1
    R2, Vp2, jd2 = planet2
    V1, V2 = trajectory

    tof = jd2 - jd1

    # Algorithm 4.1 for the orbital elements of the trajectory.
    coe = coe_from_sv(R1, V1, mu=mu)
    coe2 = coe_from_sv(R2, V2, mu=mu)

    # Equations 8.102 and 8.103 - hyperbolic excess velocities.
    vinf1 = V1 - Vp1
    vinf2 = V2 - Vp2

    print("-" * 60)
    print(" Example 8.8\n")

    print(" Departure:")
    month_name, planet_name = month_planet_names(depart[2], depart[0])
    print(f"   Planet: {planet_name}")
    print(f"   Year  : {depart[1]:g}")
    print(f"   Month : {month_name}")
    print(f"   Day   : {depart[3]:g}")
    print(f"   Hour  : {depart[4]:g}")
    print(f"   Minute: {depart[5]:g}")
    print(f"   Second: {depart[6]:g}\n")
    print(f"   Julian day: {jd1:.3f}\n")
    print(f"   Planet position vector (km)   = "
          f"[{R1[0]:g}  {R1[1]:g}  {R1[2]:g}]")
    print(f"   Magnitude                     = {np.linalg.norm(R1):g}\n")
    print(f"   Planet velocity (km/s)        = "
          f"[{Vp1[0]:g}  {Vp1[1]:g}  {Vp1[2]:g}]")
    print(f"   Magnitude                     = {np.linalg.norm(Vp1):g}\n")
    print(f"   Spacecraft velocity (km/s)    = "
          f"[{V1[0]:g}  {V1[1]:g}  {V1[2]:g}]")
    print(f"   Magnitude                     = {np.linalg.norm(V1):g}\n")
    print(f"   v-infinity at departure (km/s) = "
          f"[{vinf1[0]:g}  {vinf1[1]:g}  {vinf1[2]:g}]")
    print(f"   Magnitude                     = {np.linalg.norm(vinf1):g}\n")

    print(f" Time of flight = {tof:g} days\n")

    print(" Arrival:")
    month_name, planet_name = month_planet_names(arrive[2], arrive[0])
    print(f"   Planet: {planet_name}")
    print(f"   Year  : {arrive[1]:g}")
    print(f"   Month : {month_name}")
    print(f"   Day   : {arrive[3]:g}")
    print(f"   Hour  : {arrive[4]:g}")
    print(f"   Minute: {arrive[5]:g}")
    print(f"   Second: {arrive[6]:g}\n")
    print(f"   Julian day: {jd2:.3f}\n")
    print(f"   Planet position vector (km)   = "
          f"[{R2[0]:g}  {R2[1]:g}  {R2[2]:g}]")
    print(f"   Magnitude                     = {np.linalg.norm(R2):g}\n")
    print(f"   Planet velocity (km/s)        = "
          f"[{Vp2[0]:g}  {Vp2[1]:g}  {Vp2[2]:g}]")
    print(f"   Magnitude                     = {np.linalg.norm(Vp2):g}\n")
    print(f"   Spacecraft Velocity (km/s)    = "
          f"[{V2[0]:g}  {V2[1]:g}  {V2[2]:g}]")
    print(f"   Magnitude                     = {np.linalg.norm(V2):g}\n")
    print(f"   v-infinity at arrival (km/s)  = "
          f"[{vinf2[0]:g}  {vinf2[1]:g}  {vinf2[2]:g}]")
    print(f"   Magnitude                     = {np.linalg.norm(vinf2):g}\n")

    print(" Orbital elements of flight trajectory:\n")
    print(f"  Angular momentum (km^2/s)                  = {coe[0]:g}")
    print(f"  Eccentricity                               = {coe[1]:g}")
    print(f"  Right ascension of the ascending node (deg) = {coe[2] / deg:g}")
    print(f"  Inclination to the ecliptic (deg)          = {coe[3] / deg:g}")
    print(f"  Argument of perihelion (deg)               = {coe[4] / deg:g}")
    print(f"  True anomaly at departure (deg)            = {coe[5] / deg:g}")
    print(f"  True anomaly at arrival (deg)              = {coe2[5] / deg:g}\n")
    print(f"  Semimajor axis (km)                        = {coe[6]:g}")
    if coe[1] < 1:
        T = 2 * math.pi / math.sqrt(mu) * coe[6] ** 1.5 / 24 / 3600
        print(f"  Period (days)                              = {T:g}")
    print("-" * 60)


if __name__ == "__main__":
    _example_8_08()
