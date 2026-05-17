"""Country name ↔ ISO-alpha2 code mapping used across Eurovision history.

Includes Yugoslavia, Serbia & Montenegro, and the F.Y.R. Macedonia → North Macedonia
rename. Codes follow the Spijkervet dataset conventions where they diverge from ISO.
"""
from __future__ import annotations

NAME_TO_CODE: dict[str, str] = {
    "Albania": "al", "Andorra": "ad", "Armenia": "am", "Australia": "au",
    "Austria": "at", "Azerbaijan": "az", "Belarus": "by", "Belgium": "be",
    "Bosnia & Herzegovina": "ba", "Bosnia and Herzegovina": "ba",
    "Bulgaria": "bg", "Croatia": "hr", "Cyprus": "cy", "Czechia": "cz",
    "Czech Republic": "cz", "Denmark": "dk", "Estonia": "ee", "Finland": "fi",
    "France": "fr", "Georgia": "ge", "Germany": "de", "Greece": "gr",
    "Hungary": "hu", "Iceland": "is", "Ireland": "ie", "Israel": "il",
    "Italy": "it", "Latvia": "lv", "Lithuania": "lt", "Luxembourg": "lu",
    "Malta": "mt", "Moldova": "md", "Monaco": "mc", "Montenegro": "me",
    "Morocco": "ma", "Netherlands": "nl", "The Netherlands": "nl",
    "North Macedonia": "mk", "Macedonia": "mk", "F.Y.R. Macedonia": "mk",
    "Norway": "no", "Poland": "pl", "Portugal": "pt", "Romania": "ro",
    "Russia": "ru", "San Marino": "sm", "Serbia": "rs",
    "Serbia & Montenegro": "cs", "Serbia and Montenegro": "cs",
    "Slovakia": "sk", "Slovenia": "si", "Spain": "es", "Sweden": "se",
    "Switzerland": "ch", "Turkey": "tr", "Türkiye": "tr",
    "Ukraine": "ua", "United Kingdom": "gb", "UK": "gb",
    "Yugoslavia": "yu",
}

CODE_TO_NAME: dict[str, str] = {
    "al": "Albania", "ad": "Andorra", "am": "Armenia", "au": "Australia",
    "at": "Austria", "az": "Azerbaijan", "by": "Belarus", "be": "Belgium",
    "ba": "Bosnia & Herzegovina", "bg": "Bulgaria", "hr": "Croatia",
    "cy": "Cyprus", "cz": "Czechia", "dk": "Denmark", "ee": "Estonia",
    "fi": "Finland", "fr": "France", "ge": "Georgia", "de": "Germany",
    "gr": "Greece", "hu": "Hungary", "is": "Iceland", "ie": "Ireland",
    "il": "Israel", "it": "Italy", "lv": "Latvia", "lt": "Lithuania",
    "lu": "Luxembourg", "mt": "Malta", "md": "Moldova", "mc": "Monaco",
    "me": "Montenegro", "ma": "Morocco", "nl": "Netherlands",
    "mk": "North Macedonia", "no": "Norway", "pl": "Poland", "pt": "Portugal",
    "ro": "Romania", "ru": "Russia", "sm": "San Marino", "rs": "Serbia",
    "cs": "Serbia & Montenegro", "sk": "Slovakia", "si": "Slovenia",
    "es": "Spain", "se": "Sweden", "ch": "Switzerland", "tr": "Turkey",
    "ua": "Ukraine", "gb": "United Kingdom", "yu": "Yugoslavia",
}


def to_code(name: str) -> str | None:
    if not name:
        return None
    return NAME_TO_CODE.get(name.strip())
