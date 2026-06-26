"""Algorithm 5.3 (Appendix D.13) - Calculation of local sidereal time.

MATLAB source: ``LST.m`` driven by ``Example_5_06.m``.
"""

import math

from .julian_day import J0, _fix


def _zeroTo360(x):
    """Reduce an angle (degrees) to the range 0 - 360 degrees."""
    if x >= 360:
        x = x - _fix(x / 360) * 360
    elif x < 0:
        x = x - (_fix(x / 360) - 1) * 360
    return x


def LST(y, m, d, ut, EL):
    """Calculate the local sidereal time.

    Parameters
    ----------
    y, m, d : int
        Year, month, day.
    ut : float
        Universal Time (hours).
    EL : float
        East longitude (degrees).

    Returns
    -------
    float
        Local sidereal time (degrees).
    """
    # Equation 5.48
    j0 = J0(y, m, d)

    # Equation 5.49 - number of centuries since J2000.
    j = (j0 - 2451545) / 36525

    # Equation 5.50 - Greenwich sidereal time at 0 hr UT.
    g0 = (100.4606184 + 36000.77004 * j + 0.000387933 * j**2
          - 2.583e-8 * j**3)

    # Reduce g0 to the range 0 - 360 degrees.
    g0 = _zeroTo360(g0)

    # Equation 5.51 - Greenwich sidereal time at the specified UT.
    gst = g0 + 360.98564724 * ut / 24

    # Equation 5.52 - local sidereal time.
    lst = gst + EL

    # Reduce lst to the range 0 - 360 degrees.
    lst = lst - 360 * _fix(lst / 360)

    return lst


def _example_5_06():
    """Reproduce the output of ``Example_5_06.m``."""
    # East longitude.
    degrees = 139
    minutes = 47
    seconds = 0

    # Date.
    year, month, day = 2004, 3, 3

    # Universal time.
    hour, minute, second = 4, 30, 0

    # Convert negative (west) longitude to east longitude.
    if degrees < 0:
        degrees = degrees + 360

    EL = degrees + minutes / 60 + seconds / 3600
    WL = 360 - EL
    ut = hour + minute / 60 + second / 3600

    lst = LST(year, month, day, ut, EL)

    print("-" * 52)
    print(" Example 5.6: Local sidereal time calculation\n")
    print(" Input data:\n")
    print(f"   Year                  = {year:g}")
    print(f"   Month                 = {month:g}")
    print(f"   Day                   = {day:g}")
    print(f"   UT (hr)               = {ut:g}")
    print(f"   West Longitude (deg)  = {WL:g}")
    print(f"   East Longitude (deg)  = {EL:g}\n")
    print(" Solution:\n")
    print(f"   Local Sidereal Time (deg) = {lst:g}")
    print(f"   Local Sidereal Time (hr)  = {lst / 15:g}")
    print("-" * 52)


if __name__ == "__main__":
    _example_5_06()
