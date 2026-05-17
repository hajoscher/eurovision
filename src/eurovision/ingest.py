"""Download raw Eurovision data.

Sources:
- Spijkervet/eurovision-dataset GitHub releases (1956–2023): contestants, votes, betting odds
- Wikipedia (2024–2026 backfill): contestants + final results + jury/televote splits
"""
from __future__ import annotations

import logging
from pathlib import Path

import requests

log = logging.getLogger(__name__)

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

SPIJKERVET_BASE = "https://github.com/Spijkervet/eurovision-dataset/releases/download/2023"
SPIJKERVET_FILES = ["contestants.csv", "votes.csv", "betting_offices.csv"]

WIKI_YEARS = [2023, 2024, 2025, 2026]
WIKI_URL = "https://en.wikipedia.org/wiki/Eurovision_Song_Contest_{year}"

UA = {"User-Agent": "Mozilla/5.0 (eurovision-research; +https://github.com/)"}


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        log.info("skip (exists): %s", dest.name)
        return
    log.info("GET %s", url)
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    log.info("wrote %s (%d bytes)", dest.name, len(r.content))


def fetch_spijkervet() -> None:
    for name in SPIJKERVET_FILES:
        _download(f"{SPIJKERVET_BASE}/{name}", RAW / f"{name.replace('.csv', '')}_2023.csv")


def fetch_wikipedia() -> None:
    for y in WIKI_YEARS:
        _download(WIKI_URL.format(year=y), RAW / f"wiki_{y}.html")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fetch_spijkervet()
    fetch_wikipedia()
    log.info("done.")


if __name__ == "__main__":
    main()
