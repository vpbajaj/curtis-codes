"""Appendix D.16 - Converting the numerical designation of a month or a
planet into its name.

MATLAB source: ``month_planet_names.m``.
"""

_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

_PLANETS = [
    "Mercury", "Venus", "Earth", "Mars", "Jupiter",
    "Saturn", "Uranus", "Neptune", "Pluto",
]


def month_planet_names(month_id, planet_id):
    """Return the names of the month and planet for the given numbers.

    Parameters
    ----------
    month_id : int
        Month number (1 - 12).
    planet_id : int
        Planet number (1 - 9).

    Returns
    -------
    tuple(str, str)
        Name of the month and name of the planet.
    """
    return _MONTHS[month_id - 1], _PLANETS[planet_id - 1]
