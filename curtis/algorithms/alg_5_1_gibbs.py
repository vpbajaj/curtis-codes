"""Algorithm 5.1 (Appendix D.10) - Gibbs' method of preliminary orbit
determination.

MATLAB source: ``gibbs.m`` driven by ``Example_5_01.m``.
"""

import math

import numpy as np

from ..constants import MU_EARTH
from .alg_4_1_coe_from_sv import coe_from_sv


def gibbs(R1, R2, R3, mu=MU_EARTH, tol=1.0e-4):
    """Gibbs' method: velocity at R2 from three coplanar position vectors.

    Parameters
    ----------
    R1, R2, R3 : array_like
        Three coplanar geocentric position vectors (km).
    mu : float
        Gravitational parameter (km^3/s^2).
    tol : float
        Tolerance for the coplanarity check.

    Returns
    -------
    tuple(numpy.ndarray, int)
        Velocity ``V2`` corresponding to ``R2`` (km/s) and an error flag
        ``ierr`` (0 if R1, R2, R3 are coplanar, 1 otherwise).
    """
    R1 = np.asarray(R1, dtype=float)
    R2 = np.asarray(R2, dtype=float)
    R3 = np.asarray(R3, dtype=float)

    ierr = 0

    # Magnitudes of R1, R2 and R3.
    r1 = np.linalg.norm(R1)
    r2 = np.linalg.norm(R2)
    r3 = np.linalg.norm(R3)

    # Cross products among R1, R2 and R3.
    c12 = np.cross(R1, R2)
    c23 = np.cross(R2, R3)
    c31 = np.cross(R3, R1)

    # Check that R1, R2 and R3 are coplanar; if not, set the error flag.
    if abs(np.dot(R1, c23) / r1 / np.linalg.norm(c23)) > tol:
        ierr = 1

    # Equation 5.13
    N = r1 * c23 + r2 * c31 + r3 * c12
    # Equation 5.14
    D = c12 + c23 + c31
    # Equation 5.21
    S = R1 * (r2 - r3) + R2 * (r3 - r1) + R3 * (r1 - r2)
    # Equation 5.22
    V2 = math.sqrt(mu / np.linalg.norm(N) / np.linalg.norm(D)) * (
        np.cross(D, R2) / r2 + S)

    return V2, ierr


def _example_5_01():
    """Reproduce the output of ``Example_5_01.m``."""
    deg = math.pi / 180.0
    mu = 398600.0

    r1 = np.array([-294.32, 4265.1, 5986.7])
    r2 = np.array([-1365.4, 3637.6, 6346.8])
    r3 = np.array([-2940.3, 2473.7, 6555.8])

    print("-" * 52)
    print(" Example 5.1: Gibbs Method\n")
    print(" Input data:\n")
    print(f" Gravitational parameter (km^3/s^2)  = {mu:g}\n")
    print(f" r1 (km) = [{r1[0]:g}  {r1[1]:g}  {r1[2]:g}]")
    print(f" r2 (km) = [{r2[0]:g}  {r2[1]:g}  {r2[2]:g}]")
    print(f" r3 (km) = [{r3[0]:g}  {r3[1]:g}  {r3[2]:g}]")

    v2, ierr = gibbs(r1, r2, r3, mu=mu)
    if ierr == 1:
        print("\n  These vectors are not coplanar.\n")
        return

    coe = coe_from_sv(r2, v2, mu=mu)

    print("Solution:\n")
    print(f" v2 (km/s) = [{v2[0]:g}  {v2[1]:g}  {v2[2]:g}]\n")
    print(" Orbital elements:")
    print(f"   Angular momentum (km^2/s)  = {coe[0]:g}")
    print(f"   Eccentricity               = {coe[1]:g}")
    print(f"   Inclination (deg)          = {coe[3] / deg:g}")
    print(f"   RA of ascending node (deg) = {coe[2] / deg:g}")
    print(f"   Argument of perigee (deg)  = {coe[4] / deg:g}")
    print(f"   True anomaly (deg)         = {coe[5] / deg:g}")
    print(f"   Semimajor axis (km)        = {coe[6]:g}")
    if coe[1] < 1:
        T = 2 * math.pi / math.sqrt(mu) * coe[6] ** 1.5
        print(f"   Period (s)                 = {T:g}")
    print("-" * 52)


if __name__ == "__main__":
    _example_5_01()
