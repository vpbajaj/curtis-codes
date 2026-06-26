"""Appendix D.12 - Calculation of the Julian day number at 0 hr UT
(Equation 5.48).

MATLAB source: ``J0.m`` (exercised within ``Example_5_04.m``).
"""

import math


def _fix(x):
    """MATLAB ``fix``: round toward zero."""
    return math.trunc(x)


def J0(year, month, day):
    """Julian day number at 0 hr UT for any year between 1900 and 2100.

    Parameters
    ----------
    year : int
        Year, range 1901 - 2099.
    month : int
        Month, range 1 - 12.
    day : int
        Day, range 1 - 31.

    Returns
    -------
    float
        Julian day number at 0 hr UT (Equation 5.48).
    """
    return (367 * year
            - _fix(7 * (year + _fix((month + 9) / 12)) / 4)
            + _fix(275 * month / 9)
            + day + 1721013.5)


def _example_5_04():
    """Reproduce the J0/JD portion of ``Example_5_04.m``."""
    year = 2004
    month = 5
    day = 12
    hour = 14
    minute = 45
    second = 30

    ut = hour + minute / 60 + second / 3600
    j0 = J0(year, month, day)   # Equation 5.48
    jd = j0 + ut / 24           # Equation 5.47

    print("-" * 52)
    print(" Example 5.4: Julian day calculation\n")
    print(" Input data:\n")
    print(f"   Year             = {year:g}")
    print(f"   Month            = {month:g}")
    print(f"   Day              = {day:g}")
    print(f"   Hour             = {hour:g}")
    print(f"   Minute           = {minute:g}")
    print(f"   Second           = {second:g}\n")
    print(f" Julian day number at 0 hr UT = {j0:.11g}")
    print(f" Julian day number at given UT = {jd:.11g}")
    print("-" * 52)


if __name__ == "__main__":
    _example_5_04()
