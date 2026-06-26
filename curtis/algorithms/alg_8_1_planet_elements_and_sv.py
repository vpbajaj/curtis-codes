"""Algorithm 8.1 (Appendix D.17) - Calculation of the orbital elements and
state vector of a planet at a given epoch.

MATLAB source: ``planet_elements_and_sv.m`` (with the ``planetary_elements``
and ``zero_to_360`` subfunctions) driven by ``Example_8_07.m``.
"""

import math

import numpy as np

from ..constants import MU_SUN
from .alg_3_1_kepler_E import kepler_E
from .alg_4_2_sv_from_coe import sv_from_coe
from .julian_day import J0, _fix

# Table 8.1 - J2000 orbital elements of the nine planets.
# Columns: a (AU), e, i (deg), RA (deg), w_hat (deg), L (deg).
_J2000_ELEMENTS = np.array([
    [0.38709893, 0.20563069, 7.00487, 48.33167, 77.45645, 252.25084],
    [0.72333199, 0.00677323, 3.39471, 76.68069, 131.53298, 181.97973],
    [1.00000011, 0.01671022, 0.00005, -11.26064, 102.94719, 100.46435],
    [1.52366231, 0.09341233, 1.85061, 49.57854, 336.04084, 355.45332],
    [5.20336301, 0.04839266, 1.30530, 100.55615, 14.75385, 34.40438],
    [9.53707032, 0.05415060, 2.48446, 113.71504, 92.43194, 49.94432],
    [19.19126393, 0.04716771, 0.76986, 74.22988, 170.96424, 313.23218],
    [30.06896348, 0.00858587, 1.76917, 131.72169, 44.97135, 304.88003],
    [39.48168677, 0.24880766, 17.14175, 110.30347, 224.06676, 238.92881],
])

# Table 8.1 - rates of change per Julian century.
# Columns: a_dot (AU/Cy), e_dot (1/Cy), i_dot, RA_dot, w_hat_dot, Ldot
# (the last four in arcseconds/Cy).
_CENT_RATES = np.array([
    [0.00000066, 0.00002527, -23.51, -446.30, 573.57, 538101628.29],
    [0.00000092, -0.00004938, -2.86, -996.89, -108.80, 210664136.06],
    [-0.00000005, -0.00003804, -46.94, -18228.25, 1198.28, 129597740.63],
    [-0.00007221, 0.00011902, -25.47, -1020.19, 1560.78, 68905103.78],
    [0.00060737, -0.00012880, -4.15, 1217.17, 839.93, 10925078.35],
    [-0.00301530, -0.00036762, 6.11, -1591.05, -1948.89, 4401052.95],
    [0.00152025, -0.00019150, -2.09, -1681.4, 1312.56, 1542547.79],
    [-0.00125196, 0.00002514, -3.64, -151.25, -844.43, 786449.21],
    [-0.00076912, 0.00006465, 11.07, -37.33, -132.25, 522747.90],
])


def zero_to_360(x):
    """Reduce an angle (degrees) to the range 0 - 360 degrees."""
    if x >= 360:
        x = x - _fix(x / 360) * 360
    elif x < 0:
        x = x - (_fix(x / 360) - 1) * 360
    return x


def planetary_elements(planet_id):
    """Extract a planet's J2000 elements and centennial rates (Table 8.1).

    AU is converted to km and arcseconds to degrees, matching the units
    used by the rest of the algorithm.
    """
    J2000_coe = _J2000_ELEMENTS[planet_id - 1].copy()
    rates = _CENT_RATES[planet_id - 1].copy()

    # Convert from AU to km.
    au = 149597871.0
    J2000_coe[0] = J2000_coe[0] * au
    rates[0] = rates[0] * au

    # Convert from arcseconds to fractions of a degree.
    rates[2:6] = rates[2:6] / 3600.0

    return J2000_coe, rates


def planet_elements_and_sv(planet_id, year, month, day, hour, minute, second,
                           mu=MU_SUN):
    """Heliocentric orbital elements and state vector of a planet.

    Parameters
    ----------
    planet_id : int
        Planet identifier (1 = Mercury ... 9 = Pluto).
    year, month, day, hour, minute, second : int
        Date and Universal Time of the epoch.
    mu : float
        Gravitational parameter of the sun (km^3/s^2).

    Returns
    -------
    tuple
        ``(coe, r, v, jd)`` where ``coe`` is
        ``[h, e, RA, incl, w, TA, a, w_hat, L, M, E]`` (angles in degrees,
        ``h`` in km^2/s, ``a`` in km), ``r`` and ``v`` are the heliocentric
        position (km) and velocity (km/s) vectors and ``jd`` is the Julian
        day number of the epoch.
    """
    deg = math.pi / 180.0

    # Equation 5.48 and 5.47 - Julian day of the date and time.
    j0 = J0(year, month, day)
    ut = (hour + minute / 60 + second / 3600) / 24
    jd = j0 + ut

    # Obtain the data for the selected planet from Table 8.1.
    J2000_coe, rates = planetary_elements(planet_id)

    # Equation 8.104a - Julian centuries between J2000 and jd.
    t0 = (jd - 2451545) / 36525

    # Equation 8.104b - elements at jd.
    elements = J2000_coe + rates * t0

    a = elements[0]
    e = elements[1]

    # Equation 2.61 - angular momentum.
    h = math.sqrt(mu * a * (1 - e**2))

    # Reduce the angular elements to within the range 0 - 360 degrees.
    incl = elements[2]
    RA = zero_to_360(elements[3])
    w_hat = zero_to_360(elements[4])
    L = zero_to_360(elements[5])
    w = zero_to_360(w_hat - RA)
    M = zero_to_360((L - w_hat))

    # Algorithm 3.1 (M must be in radians).
    E = kepler_E(e, M * deg)

    # Equation 3.10 - true anomaly (converting the result to degrees).
    TA = zero_to_360(
        2 * math.atan(math.sqrt((1 + e) / (1 - e)) * math.tan(E / 2)) / deg)

    coe = [h, e, RA, incl, w, TA, a, w_hat, L, M, E / deg]

    # Algorithm 4.2 (all angles must be in radians).
    r, v = sv_from_coe([h, e, RA * deg, incl * deg, w * deg, TA * deg], mu=mu)

    return coe, r, v, jd


def _example_8_07():
    """Reproduce the output of ``Example_8_07.m``."""
    from .month_planet_names import month_planet_names

    mu = 1.327124e11

    planet_id = 3
    year = 2003
    month = 8
    day = 27
    hour = 12
    minute = 0
    second = 0

    coe, r, v, jd = planet_elements_and_sv(
        planet_id, year, month, day, hour, minute, second, mu=mu)
    month_name, planet_name = month_planet_names(month, planet_id)

    print("-" * 52)
    print(" Example 8.7\n")
    print(" Input data:\n")
    print(f"   Planet: {planet_name}")
    print(f"   Year  : {year:g}")
    print(f"   Month : {month_name}")
    print(f"   Day   : {day:g}")
    print(f"   Hour  : {hour:g}")
    print(f"   Minute: {minute:g}")
    print(f"   Second: {second:g}\n")
    print(f"   Julian day: {jd:.3f}\n")
    print(" Orbital elements:\n")
    print(f"  Angular momentum (km^2/s)                  = {coe[0]:g}")
    print(f"  Eccentricity                               = {coe[1]:g}")
    print(f"  Right ascension of the ascending node (deg) = {coe[2]:g}")
    print(f"  Inclination to the ecliptic (deg)          = {coe[3]:g}")
    print(f"  Argument of perihelion (deg)               = {coe[4]:g}")
    print(f"  True anomaly (deg)                         = {coe[5]:g}")
    print(f"  Semimajor axis (km)                        = {coe[6]:g}\n")
    print(f"  Longitude of perihelion (deg)              = {coe[7]:g}")
    print(f"  Mean longitude (deg)                       = {coe[8]:g}")
    print(f"  Mean anomaly (deg)                         = {coe[9]:g}")
    print(f"  Eccentric anomaly (deg)                    = {coe[10]:g}\n")
    print(" State vector:\n")
    print(f"  Position vector (km) = [{r[0]:g}  {r[1]:g}  {r[2]:g}]")
    print(f"  Magnitude            = {np.linalg.norm(r):g}")
    print(f"  Velocity (km/s)      = [{v[0]:g}  {v[1]:g}  {v[2]:g}]")
    print(f"  Magnitude            = {np.linalg.norm(v):g}")
    print("-" * 52)


if __name__ == "__main__":
    _example_8_07()
