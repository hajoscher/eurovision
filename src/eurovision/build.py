"""Build cleaned parquet tables from raw inputs.

Outputs (under data/processed/):
- contestants.parquet — one row per (year, country) entry, all years 1956–latest
- final_results.parquet — flat final standings: year, country, place, total/jury/tele points
- votes.parquet — pairwise votes 1957–2023 (from_country, to_country, round, jury/tele/total)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .wiki_parse import parse_year, parse_pairwise_votes

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

WIKI_BACKFILL_YEARS = [2023, 2024, 2025, 2026]


def _normalize_country_id(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Spijkervet rows sometimes have the full country *name* in the id column.

    Coerce any value longer than 3 chars back to its 2-letter code.
    """
    from .countries import NAME_TO_CODE, CODE_TO_NAME
    for col in cols:
        if col not in df.columns:
            continue
        bad = df[col].astype(str).str.len() > 3
        df.loc[bad, col] = df.loc[bad, col].map(NAME_TO_CODE).fillna(df.loc[bad, col])
    return df


def _spijkervet_contestants() -> pd.DataFrame:
    df = pd.read_csv(RAW / "contestants_2023.csv", low_memory=False)
    df = _normalize_country_id(df, ["to_country_id"])
    return df


def _spijkervet_votes() -> pd.DataFrame:
    df = pd.read_csv(RAW / "votes_2023.csv", low_memory=False)
    df = _normalize_country_id(df, ["from_country_id", "to_country_id"])
    return df


def _wiki_backfill_contestants() -> pd.DataFrame:
    frames = []
    for y in WIKI_BACKFILL_YEARS:
        f = parse_year(y)
        # Match Spijkervet contestants columns
        f = f.rename(columns={})
        # Keep only common columns
        f["sf_num"] = pd.NA
        f["place_contest"] = f["place_final"]
        f["running_sf"] = pd.NA
        f["place_sf"] = pd.NA
        f["points_sf"] = pd.NA
        f["points_tele_sf"] = pd.NA
        f["points_jury_sf"] = pd.NA
        f["composers"] = pd.NA
        f["lyricists"] = pd.NA
        f["lyrics"] = pd.NA
        f["youtube_url"] = pd.NA
        f = f[[
            "year", "to_country_id", "to_country", "performer", "song", "language",
            "place_contest", "sf_num", "running_final", "running_sf",
            "place_final", "points_final", "place_sf", "points_sf",
            "points_tele_final", "points_jury_final", "points_tele_sf",
            "points_jury_sf", "composers", "lyricists", "lyrics", "youtube_url",
        ]]
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def build_contestants() -> pd.DataFrame:
    old = _spijkervet_contestants()
    new = _wiki_backfill_contestants()
    # Replace `to_country` with full name from countries.CODE_TO_NAME for new entries
    from .countries import CODE_TO_NAME
    new["to_country"] = new["to_country_id"].map(CODE_TO_NAME).fillna(new["to_country_id"])
    full = pd.concat([old, new], ignore_index=True)
    full = full.drop_duplicates(subset=["year", "to_country_id"], keep="last")
    full.to_parquet(OUT / "contestants.parquet", index=False)
    log.info("contestants: %d rows, years %d–%d", len(full), full.year.min(), full.year.max())
    return full


def build_final_results(contestants: pd.DataFrame) -> pd.DataFrame:
    """Flat finals table for quick dashboard slicing."""
    f = contestants[contestants["place_final"].notna()].copy()
    out = f[[
        "year", "to_country_id", "to_country", "performer", "song",
        "running_final", "place_final", "points_final",
        "points_jury_final", "points_tele_final",
    ]].copy()
    out["place_final"] = out["place_final"].astype(int)
    out.to_parquet(OUT / "final_results.parquet", index=False)
    log.info("final_results: %d rows", len(out))
    return out


def build_votes() -> pd.DataFrame:
    """1957–2023 from Spijkervet + 2024+ pairwise from Wikipedia matrices."""
    v = _spijkervet_votes()
    for c in ["total_points", "tele_points", "jury_points"]:
        v[c] = pd.to_numeric(v[c], errors="coerce")
    self_votes = v.from_country_id == v.to_country_id
    log.info("dropping %d self-vote rows", int(self_votes.sum()))
    v = v[~self_votes].reset_index(drop=True)
    # Append Wikipedia backfill years that aren't already in Spijkervet (2024+).
    new_frames = []
    for y in WIKI_BACKFILL_YEARS:
        if y in v.year.unique():
            continue
        try:
            new_frames.append(parse_pairwise_votes(y))
            log.info("wiki pairwise: %d", y)
        except Exception as e:
            log.warning("failed to parse pairwise votes for %d: %s", y, e)
    if new_frames:
        v = pd.concat([v, *new_frames], ignore_index=True)
    v.to_parquet(OUT / "votes.parquet", index=False)
    log.info("votes: %d rows, years %d–%d", len(v), v.year.min(), v.year.max())
    return v


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    c = build_contestants()
    build_final_results(c)
    build_votes()
    log.info("done. wrote to %s", OUT)


if __name__ == "__main__":
    main()
