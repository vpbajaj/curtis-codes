"""Physical data from Appendix A of Curtis, *Orbital Mechanics for
Engineering Students*.

All values are given in the units used throughout the text: kilometres,
kilograms, seconds and degrees.
"""

# ---------------------------------------------------------------------------
# Table A.2 - Gravitational parameter mu (km^3/s^2)
# ---------------------------------------------------------------------------
MU = {
    "sun": 132_712_000_000.0,
    "mercury": 22_030.0,
    "venus": 324_900.0,
    "earth": 398_600.0,
    "moon": 4_903.0,
    "mars": 42_828.0,
    "jupiter": 126_686_000.0,
    "saturn": 37_931_000.0,
    "uranus": 5_794_000.0,
    "neptune": 6_835_100.0,
    "pluto": 830.0,
}

# Convenient shorthands for the bodies used most often in the text.
MU_SUN = MU["sun"]
MU_EARTH = MU["earth"]
MU_MOON = MU["moon"]

# ---------------------------------------------------------------------------
# Table A.2 - Sphere of influence (SOI) radius (km)
# ---------------------------------------------------------------------------
SOI = {
    "mercury": 112_000.0,
    "venus": 616_000.0,
    "earth": 925_000.0,
    "moon": 66_200.0,
    "mars": 577_000.0,
    "jupiter": 48_200_000.0,
    "saturn": 54_800_000.0,
    "uranus": 51_800_000.0,
    "neptune": 86_600_000.0,
    "pluto": 3_080_000.0,
}

# ---------------------------------------------------------------------------
# Table A.1 - Mean equatorial radius (km)
# ---------------------------------------------------------------------------
RADIUS = {
    "sun": 696_000.0,
    "mercury": 2_440.0,
    "venus": 6_052.0,
    "earth": 6_378.0,
    "moon": 1_737.0,
    "mars": 3_396.0,
    "jupiter": 71_490.0,
    "saturn": 60_270.0,
    "uranus": 25_560.0,
    "neptune": 24_760.0,
    "pluto": 1_195.0,
}

R_EARTH = RADIUS["earth"]

# Earth oblateness (Section 4.7)
J2_EARTH = 1.08263e-3

# Earth shape / rotation parameters (used by Algorithms 5.3 and 5.4).
RE_EARTH_EQ = 6378.13655   # equatorial radius (km)
F_EARTH = 1.0 / 298.256421867   # flattening factor
WE_EARTH = 7.292115e-5   # angular velocity (rad/s)
MU_EARTH_PRECISE = 398600.4418   # gravitational parameter (km^3/s^2)

# ---------------------------------------------------------------------------
# Table A.3 - Some conversion factors
# ---------------------------------------------------------------------------
FT_TO_M = 0.3048
MILE_TO_KM = 1.609
NAUTICAL_MILE_TO_KM = 1.852
MIPH_TO_KMPS = 0.0004469
LB_MASS_TO_KG = 0.4536
LB_FORCE_TO_N = 4.448
PSI_TO_KPA = 6895.0
