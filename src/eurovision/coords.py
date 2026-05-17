"""Geographic coordinates for the Eurovision flow map.

Real (lat, lon) of capitals, except for `au` which is relocated to the south-east
corner of the European view so it appears on the same canvas.
"""
from __future__ import annotations

# (lat, lon) — capitals unless noted otherwise
COORDS: dict[str, tuple[float, float]] = {
    "al": (41.33, 19.82),   # Tirana
    "ad": (42.51, 1.52),    # Andorra la Vella
    "am": (40.18, 44.51),   # Yerevan
    "au": (32.00, 51.00),   # ⚑ Australia — relocated to SE corner of EU view
    "at": (48.21, 16.37),   # Vienna
    "az": (40.41, 49.87),   # Baku
    "by": (53.90, 27.57),   # Minsk
    "be": (50.85, 4.35),    # Brussels
    "ba": (43.86, 18.41),   # Sarajevo
    "bg": (42.70, 23.32),   # Sofia
    "hr": (45.81, 15.98),   # Zagreb
    "cy": (35.17, 33.37),   # Nicosia
    "cz": (50.08, 14.44),   # Prague
    "dk": (55.68, 12.57),   # Copenhagen
    "ee": (59.44, 24.75),   # Tallinn
    "fi": (60.17, 24.94),   # Helsinki
    "fr": (48.86, 2.35),    # Paris
    "ge": (41.72, 44.79),   # Tbilisi
    "de": (52.52, 13.40),   # Berlin
    "gr": (37.98, 23.73),   # Athens
    "hu": (47.50, 19.04),   # Budapest
    "is": (64.13, -21.94),  # Reykjavík
    "ie": (53.35, -6.26),   # Dublin
    "il": (31.78, 35.22),   # Jerusalem
    "it": (41.90, 12.50),   # Rome
    "lv": (56.95, 24.11),   # Riga
    "lt": (54.69, 25.28),   # Vilnius
    "lu": (49.61, 6.13),    # Luxembourg
    "mt": (35.90, 14.51),   # Valletta
    "md": (47.01, 28.86),   # Chișinău
    "mc": (43.74, 7.42),    # Monaco
    "me": (42.44, 19.26),   # Podgorica
    "ma": (33.97, -6.84),   # Rabat
    "nl": (52.37, 4.90),    # Amsterdam
    "mk": (41.99, 21.43),   # Skopje
    "no": (59.91, 10.75),   # Oslo
    "pl": (52.23, 21.01),   # Warsaw
    "pt": (38.72, -9.14),   # Lisbon
    "ro": (44.43, 26.10),   # Bucharest
    "ru": (55.75, 37.62),   # Moscow
    "sm": (43.94, 12.45),   # San Marino
    "rs": (44.79, 20.45),   # Belgrade
    "cs": (44.79, 20.45),   # Serbia & Montenegro — Belgrade
    "sk": (48.15, 17.11),   # Bratislava
    "si": (46.06, 14.51),   # Ljubljana
    "es": (40.42, -3.70),   # Madrid
    "se": (59.33, 18.07),   # Stockholm
    "ch": (46.95, 7.45),    # Bern
    "tr": (39.93, 32.87),   # Ankara
    "ua": (50.45, 30.52),   # Kyiv
    "gb": (51.51, -0.13),   # London
    "yu": (44.79, 20.45),   # Yugoslavia — Belgrade
}

# Countries we've intentionally relocated so the map view stays Europe-centric.
RELOCATED: set[str] = {"au"}


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing in degrees (0 = north, 90 = east) from p1 to p2."""
    import math
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def great_circle_point(lat1: float, lon1: float, lat2: float, lon2: float,
                       f: float = 0.5) -> tuple[float, float]:
    """Point at fraction f along the great-circle path from p1 to p2 (slerp).

    f=0 → p1, f=1 → p2, f=0.5 → midpoint along the great circle.
    """
    import math
    lat1r, lon1r = math.radians(lat1), math.radians(lon1)
    lat2r, lon2r = math.radians(lat2), math.radians(lon2)
    # Angular distance between the two points
    d = 2 * math.asin(math.sqrt(
        math.sin((lat2r - lat1r) / 2) ** 2
        + math.cos(lat1r) * math.cos(lat2r) * math.sin((lon2r - lon1r) / 2) ** 2
    ))
    if d < 1e-9:
        return lat1, lon1
    a = math.sin((1 - f) * d) / math.sin(d)
    b = math.sin(f * d) / math.sin(d)
    x = a * math.cos(lat1r) * math.cos(lon1r) + b * math.cos(lat2r) * math.cos(lon2r)
    y = a * math.cos(lat1r) * math.sin(lon1r) + b * math.cos(lat2r) * math.sin(lon2r)
    z = a * math.sin(lat1r) + b * math.sin(lat2r)
    lat = math.atan2(z, math.sqrt(x * x + y * y))
    lon = math.atan2(y, x)
    return math.degrees(lat), math.degrees(lon)
