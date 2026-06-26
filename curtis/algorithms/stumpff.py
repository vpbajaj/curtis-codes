"""Appendix D.4 - Stumpff functions ``S(z)`` and ``C(z)``.

These implement Equations 3.49 and 3.50 and are used by the
universal-variable routines.

MATLAB source: ``stumpS.m`` and ``stumpC.m``.
"""

import math


def stumpS(z):
    """Evaluate the Stumpff function ``S(z)`` (Equation 3.49)."""
    if z > 0:
        sz = math.sqrt(z)
        return (sz - math.sin(sz)) / sz**3
    elif z < 0:
        sz = math.sqrt(-z)
        return (math.sinh(sz) - sz) / sz**3
    else:
        return 1.0 / 6.0


def stumpC(z):
    """Evaluate the Stumpff function ``C(z)`` (Equation 3.50)."""
    if z > 0:
        return (1.0 - math.cos(math.sqrt(z))) / z
    elif z < 0:
        return (math.cosh(math.sqrt(-z)) - 1.0) / (-z)
    else:
        return 1.0 / 2.0
