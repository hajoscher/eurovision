"""Parse Eurovision data from Wikipedia HTML pages (2024+ backfill)."""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from .countries import to_code

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

_PLACE_NUM = re.compile(r"(\d+)")
_FOOTNOTE = re.compile(r"\[.*?\]")


def _strip_footnotes(s):
    if not isinstance(s, str):
        return s
    return _FOOTNOTE.sub("", s).strip()


def _clean_country(s: str) -> str:
    return _strip_footnotes(s)


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.map(_strip_footnotes), errors="coerce")


def _find_contestants_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    for t in tables:
        cols = [str(c) for c in t.columns]
        if {"Country", "Artist", "Song", "Broadcaster"}.issubset(set(cols)):
            return t
    raise ValueError("contestants table not found")


def _find_final_results_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """The simple 'R/O Country Artist Song Points Place' table for the Grand Final.

    There are 3 such tables per year (SF1, SF2, Final); the final is the largest.
    """
    candidates = []
    for t in tables:
        cols = [str(c) for c in t.columns]
        if {"R/O", "Country", "Artist", "Song", "Points", "Place"}.issubset(set(cols)):
            candidates.append(t)
    if not candidates:
        raise ValueError("no R/O results tables found")
    return max(candidates, key=len)


def _find_split_table(tables: list[pd.DataFrame], n_rows: int) -> pd.DataFrame:
    """The jury-vs-televote split table — multi-index columns Combined / Jury / Televoting.

    Multiple split tables exist (SF1, SF2, Final); pick the one matching n_rows.
    """
    for t in tables:
        if not isinstance(t.columns, pd.MultiIndex):
            continue
        lvl0 = {c[0] for c in t.columns}
        if {"Combined", "Jury", "Televoting"}.issubset(lvl0) and len(t) == n_rows:
            return t
    raise ValueError(f"no Combined/Jury/Televoting split table with {n_rows} rows")


def parse_contestants(year: int) -> pd.DataFrame:
    tables = pd.read_html(RAW / f"wiki_{year}.html")
    t = _find_contestants_table(tables).copy()
    t.columns = [str(c) for c in t.columns]
    lang_col = "Language(s)" if "Language(s)" in t.columns else "Language"
    out = pd.DataFrame({
        "year": year,
        "country": t["Country"].map(_clean_country),
        "broadcaster": t["Broadcaster"],
        "performer": t["Artist"],
        "song": t["Song"].str.strip('"'),
        "language": t[lang_col] if lang_col in t.columns else None,
    })
    out["to_country_id"] = out["country"].map(to_code)
    out["to_country"] = out["to_country_id"]  # match Spijkervet schema
    return out


def parse_final_results(year: int) -> pd.DataFrame:
    """Returns one row per finalist with running order, points, place, jury + televote split."""
    tables = pd.read_html(RAW / f"wiki_{year}.html")
    final = _find_final_results_table(tables).copy()
    final["Country"] = final["Country"].map(_clean_country)
    final["Place"] = final["Place"].astype(str).str.extract(_PLACE_NUM, expand=False)
    final = final.rename(columns={
        "R/O": "running_final", "Country": "country", "Artist": "performer",
        "Song": "song", "Points": "points_final", "Place": "place_final",
    })
    final["song"] = final["song"].str.strip('"')

    split = _find_split_table(tables, len(final)).copy()
    # Multi-index → flat
    split.columns = ["_".join([str(x) for x in c if x and str(x) != "nan"]).strip("_")
                     for c in split.columns]
    # Expected: Place_Place, Combined_Country, Combined_Points, Jury_Country, Jury_Points,
    # Televoting_Country, Televoting_Points
    jury = split[["Jury_Country", "Jury_Points"]].rename(
        columns={"Jury_Country": "country", "Jury_Points": "points_jury_final"})
    tele = split[["Televoting_Country", "Televoting_Points"]].rename(
        columns={"Televoting_Country": "country", "Televoting_Points": "points_tele_final"})
    jury["country"] = jury["country"].map(_clean_country)
    tele["country"] = tele["country"].map(_clean_country)

    out = final.merge(jury, on="country", how="left").merge(tele, on="country", how="left")
    out["year"] = year
    out["to_country_id"] = out["country"].map(to_code)
    out["to_country"] = out["to_country_id"]
    for c in ["points_final", "points_jury_final", "points_tele_final",
              "place_final", "running_final"]:
        out[c] = _to_num(out[c])
    return out


def parse_year(year: int) -> pd.DataFrame:
    """Merge contestants + final results into a single Spijkervet-shaped row per country."""
    c = parse_contestants(year)
    r = parse_final_results(year)
    merged = c.merge(
        r[["country", "running_final", "points_final", "place_final",
           "points_jury_final", "points_tele_final"]],
        on="country", how="left",
    )
    return merged


def _find_voting_matrix(tables: list[pd.DataFrame], kind: str) -> pd.DataFrame:
    """Find the per-country voting matrix for the final.

    `kind` is "jury" or "tele". Returns the largest matching table (the final;
    semi-final ones are smaller).
    """
    needles = {"jury": ["Jury vote"],
               "tele": ["Televote", "Televoting vote"]}[kind]
    candidates = []
    for t in tables:
        if t.shape[0] < 20 or t.shape[1] < 30:
            continue
        # Header in row 0 col 5 (or thereabouts) indicates jury vs tele
        try:
            header = str(t.iloc[0, 5])
        except Exception:
            continue
        if any(n in header for n in needles):
            candidates.append(t)
    if not candidates:
        raise ValueError(f"no {kind} voting matrix found")
    return max(candidates, key=lambda t: t.shape[0] * t.shape[1])


def parse_pairwise_votes(year: int) -> pd.DataFrame:
    """Long-format pairwise votes for the final: year, round, from/to country, jury/tele/total."""
    tables = pd.read_html(RAW / f"wiki_{year}.html")
    jury = _find_voting_matrix(tables, "jury")
    tele = _find_voting_matrix(tables, "tele")

    def _melt(matrix: pd.DataFrame, value_name: str) -> pd.DataFrame:
        voters = matrix.iloc[1, 5:].tolist()
        # Build a clean long-form: for each contestant row (rows 3+), for each voter col
        rows = []
        for ri in range(3, matrix.shape[0]):
            contestant = matrix.iloc[ri, 1]
            if not isinstance(contestant, str):
                continue  # skip the trailing NaN row
            contestant = _clean_country(contestant)
            for vi, voter in enumerate(voters):
                if not isinstance(voter, str):
                    continue
                pts = matrix.iloc[ri, 5 + vi]
                if pd.isna(pts):
                    continue
                rows.append({
                    "from_country": _clean_country(voter),
                    "to_country": contestant,
                    value_name: pd.to_numeric(str(pts).replace("[", "").split("]")[0],
                                              errors="coerce"),
                })
        return pd.DataFrame(rows)

    j = _melt(jury, "jury_points")
    t = _melt(tele, "tele_points")
    # Build the full (voter × contestant) grid so 0-point pairs are explicit too
    # (matches Spijkervet's convention; needed for correct mean-affinity calcs).
    voters_j = [_clean_country(x) for x in jury.iloc[1, 5:].tolist() if isinstance(x, str)]
    voters_t = [_clean_country(x) for x in tele.iloc[1, 5:].tolist() if isinstance(x, str)]
    voters = sorted(set(voters_j) | set(voters_t))
    contestants_list = [c for c in jury.iloc[3:, 1].tolist() if isinstance(c, str)]
    contestants_list = [_clean_country(c) for c in contestants_list]
    grid = pd.DataFrame(
        [(v, c) for v in voters for c in contestants_list if v != c],
        columns=["from_country", "to_country"],
    )
    merged = (grid.merge(j, on=["from_country", "to_country"], how="left")
                  .merge(t, on=["from_country", "to_country"], how="left"))
    merged["jury_points"] = merged["jury_points"].fillna(0)
    merged["tele_points"] = merged["tele_points"].fillna(0)
    merged["total_points"] = merged["jury_points"] + merged["tele_points"]
    merged["year"] = year
    merged["round"] = "final"
    merged["from_country_id"] = merged["from_country"].map(to_code)
    merged["to_country_id"]   = merged["to_country"].map(to_code)
    # "Rest of the World" televote has no country code — keep its rows but mark id=None;
    # downstream filters/Affinity views skip rows with missing ids.
    # Drop rows where to_country is missing a code (shouldn't happen).
    merged = merged.dropna(subset=["to_country_id"])
    cols = ["year", "round", "from_country_id", "to_country_id",
            "from_country", "to_country",
            "total_points", "tele_points", "jury_points"]
    return merged[cols]
