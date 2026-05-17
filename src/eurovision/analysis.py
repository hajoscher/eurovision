"""Higher-level analytics: affinity, blocs, snubs, predictors.

All functions accept `vote_type ∈ {'total', 'jury', 'tele'}` so every view can be
recomputed for total points, jury-only (2016+), or televote-only (2016+).
"""
from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd

from . import data as ed

POINTS_COL = {"total": "total_points", "jury": "jury_points", "tele": "tele_points"}

def max_points_per_voter(year: int) -> int:
    """Max points one voter (country) could give a single entry under that year's system."""
    if year < 1957:
        return 10  # 1956 was special / data missing; pick sane default
    if 1957 <= year <= 1961: return 10
    if year == 1962:         return 3   # 3-2-1
    if year == 1963:         return 5   # 5-4-3-2-1
    if 1964 <= year <= 1966: return 5   # 5+3+1 variations
    if 1967 <= year <= 1970: return 10  # 10 votes per jury
    if 1971 <= year <= 1973: return 10  # 2 jurors × 1-5 = max 10 per song
    if year == 1974:         return 10  # 10-vote distribution
    if 1975 <= year <= 2015: return 12  # classic 12-point system
    return 24                            # 2016+: jury 12 + tele 12 = 24


@cache
def n_voters_per_year() -> pd.Series:
    """Number of voting countries per year (final round). Falls back to number of
    competing countries from the contestants table for years where pairwise votes
    aren't yet ingested (2024+)."""
    v = ed.votes()
    from_votes = (v[v["round"] == "final"]
                  .groupby("year")["from_country_id"].nunique())
    c = ed.contestants()
    from_contestants = c.groupby("year")["to_country_id"].nunique()
    # Use votes count where available, else contestants count
    out = from_contestants.copy()
    out.update(from_votes)
    out.name = "n_voters"
    return out


def normalize_finals(round_: str = "final") -> pd.DataFrame:
    """Return final_results + columns: max_possible, pct_of_max, total_awarded,
    pct_of_total, runner_up_pts, margin_ratio (winner-only)."""
    f = ed.final_results().copy()
    nv = n_voters_per_year()
    f["n_voters"] = f["year"].map(nv)
    f["max_per_voter"] = f["year"].map(max_points_per_voter)
    f["max_possible"] = (f["n_voters"] - 1) * f["max_per_voter"]
    f["pct_of_max"] = (f["points_final"] / f["max_possible"] * 100).round(1)
    total_awarded = f.groupby("year")["points_final"].sum()
    f["total_awarded"] = f["year"].map(total_awarded)
    f["pct_of_total"] = (f["points_final"] / f["total_awarded"] * 100).round(2)
    # Margin over 2nd (only meaningful for the winner of each year)
    runner_up = (f[f.place_final == 2]
                 .groupby("year")["points_final"].max())
    f["runner_up_pts"] = f["year"].map(runner_up)
    f["margin_ratio"] = (f["points_final"] / f["runner_up_pts"]).round(3)
    return f


VOTING_ERAS = [
    # (start, end, short_name, max_pts_per_voter_per_song, voter_budget, description)
    (1956, 1956, "Secret jury",        None, None,
     "Single contest with undisclosed national jury scores; full results were never published."),
    (1957, 1961, "10 jurors, 1 vote each", 10, 10,
     "Each country's 10 jurors picked one favourite; a song could get 0–10 from a country."),
    (1962, 1962, "3-2-1",               3, 6,
     "Each country awarded 3 pts to its favourite, 2 to second, 1 to third."),
    (1963, 1963, "5-4-3-2-1",           5, 15,
     "Each country awarded 5 pts to favourite down to 1 to fifth."),
    (1964, 1966, "5+3+1 variants",      5, 9,
     "5-3-1 or 6-3 systems with juror-by-juror scoring."),
    (1967, 1970, "10 distributed votes", 10, 10,
     "Each country's jury distributed exactly 10 votes between songs."),
    (1971, 1973, "Two jurors × 1–5",   10, 87,
     "Two jurors per country gave each song 1–5 pts; budget per voter ≈ 80–100."),
    (1974, 1974, "10 jurors, free distribution", 10, 10,
     "10 jurors per country, each one vote — distributed in any way (ABBA's Waterloo year)."),
    (1975, 2015, "Classic 12-point system", 12, 58,
     "Each country awards 12-10-8-7-6-5-4-3-2-1 to its ten favourites. The Eurovision system most people recognise."),
    (2016, None, "Jury + televote split", 24, 116,
     "From 2016, jury and televote each give 12-10-…-1 separately; max single-voter contribution doubled to 24."),
]


def voting_eras() -> pd.DataFrame:
    rows = []
    for start, end, name, max_pts, budget, desc in VOTING_ERAS:
        end_actual = end if end is not None else 2026
        for y in range(start, end_actual + 1):
            rows.append({"year": y, "era": name, "max_pts_per_voter": max_pts,
                         "voter_budget": budget})
    return pd.DataFrame(rows)


def participation_per_year() -> pd.DataFrame:
    """Per year: countries that competed (incl. semi-finals)."""
    c = ed.contestants()
    out = (c.groupby("year")
             .agg(n_competing=("to_country_id", "nunique"),
                  n_finalists=("place_final", lambda s: int(s.notna().sum())))
             .reset_index())
    return out


def absences_per_year() -> pd.DataFrame:
    """For each year, count countries that competed nearby in time but skipped that year.

    A country counts as "absent" in year Y if it competed at least once in the prior
    5 years (Y-5 .. Y-1) AND at least once in the next 5 years (Y+1 .. Y+5). For the
    most recent few years where there is no "future" yet, just having recent past
    participation is enough.

    Proxy for boycotts / withdrawals — doesn't distinguish a true boycott from
    a financial/broadcaster pullout or a forced relegation (1996–2003).
    """
    from .countries import CODE_TO_NAME
    c = ed.contestants()[["year", "to_country_id"]].drop_duplicates()
    years_sorted = sorted(c.year.unique())
    max_year = max(years_sorted)
    by_year_codes = {y: set(c[c.year == y]["to_country_id"]) for y in years_sorted}
    rows = []
    for y in years_sorted:
        present = by_year_codes[y]
        # Recent past competitors (last 5 years)
        past = set().union(*(by_year_codes.get(yp, set()) for yp in range(y - 5, y)))
        # Next-5 future competitors
        future = set().union(*(by_year_codes.get(yf, set()) for yf in range(y + 1, y + 6)))
        if y >= max_year - 1:
            # Not enough future data — accept anyone with recent past participation
            candidates = past
        else:
            candidates = past & future
        absent = sorted(code for code in candidates if code not in present)
        absent_names = [CODE_TO_NAME.get(code, code.upper()) for code in absent]
        rows.append({"year": y, "n_absent": len(absent),
                     "absent_countries": ", ".join(absent_names)})
    return pd.DataFrame(rows)


NOTABLE_ABSENCES = [
    # (year, country, kind, note)
    (1970, "Austria, Finland, Norway, Portugal, Sweden", "boycott",
     "Protested the 4-way tie at 1969 (four-way winners with no tiebreaker)."),
    (1974, "Greece", "boycott",
     "Following the Turkish invasion of Cyprus."),
    (1975, "Greece, Malta, Turkey", "boycott / withdrawal",
     "Greece boycotted in response to Turkey's debut; Malta withdrew for financial reasons."),
    (1976, "Greece, Sweden, Yugoslavia", "boycott",
     "Sweden refused to host again citing cost; Greece boycotted over Turkey's entry."),
    (1978, "Tunisia", "withdrew",
     "Withdrew at the last moment in protest of Israel's participation."),
    (1979, "Turkey", "boycott",
     "Refused to compete in Jerusalem; pressured by other Arab broadcasters."),
    (1980, "Israel", "did not enter",
     "Israel could not host (date conflicted with Yom HaZikaron) and chose not to compete."),
    (1981, "Israel, Turkey, Yugoslavia", "various",
     "Israel: religious-calendar conflict. Turkey: continued Israel-related boycott."),
    (1982, "France, Greece", "withdrew", "Financial / programming reasons."),
    (1985, "Greece, Israel, Netherlands, Yugoslavia", "withdrew",
     "Netherlands: scheduling conflict (Remembrance Day). Israel: Holocaust Remembrance Day."),
    (1986, "Greece, Italy, Netherlands, Yugoslavia", "withdrew",
     "Italy: declining domestic interest. Netherlands: budget / scheduling."),
    (1991, "Yugoslavia", "broke up", "Last entry as Yugoslavia before dissolution."),
    (1994, "Germany, Italy", "relegated",
     "First year of the relegation system — bottom-ranked countries forced out."),
    (1996, "Germany, Romania, others", "qualifier failure",
     "Pre-televised audio qualifier; some countries didn't make it through."),
    (1996, "Several", "relegated", "Relegation continued."),
    (2020, "All", "contest cancelled",
     "COVID-19: only contest ever cancelled. Songs were celebrated in 'Europe Shine a Light'."),
    (2022, "Russia", "banned",
     "Excluded by the EBU after the invasion of Ukraine."),
    (2021, "Belarus", "banned",
     "Excluded after the BTRC submitted a politically partisan entry rejected by the EBU."),
    (2024, "Bulgaria", "withdrew", "Cited financial constraints."),
    (2025, "Romania, Moldova", "withdrew", "Financial / strategic reasons."),
    (2026, "Spain, Ireland, Iceland, Slovenia, Netherlands", "boycott",
     "Refused to participate over Israel's continued inclusion amid the Gaza war "
     "and the EBU's refusal to exclude Israel (cited as a double-standard versus "
     "the Russia ban). Largest single-event boycott since 1970."),
]


CANONICAL_BLOCS = {
    "Nordic":   ["se", "no", "dk", "fi", "is"],
    "Baltic":   ["ee", "lv", "lt"],
    "Ex-Yu":    ["hr", "si", "rs", "ba", "mk", "me", "yu", "cs"],
    "Caucasus": ["am", "az", "ge"],
    "Iberian":  ["es", "pt"],
    "British Isles": ["gb", "ie"],
    "Levant":   ["il", "cy", "gr"],
    "Eastern Slav": ["ru", "ua", "by", "md"],
    "Visegrád": ["pl", "cz", "sk", "hu"],
    "DACH":     ["de", "at", "ch"],
    "Benelux":  ["nl", "be", "lu"],
}


def _filter_votes(round_: str, year_from: int, year_to: int | None,
                  vote_type: str) -> pd.DataFrame:
    v = ed.votes()
    v = v[v["round"] == round_]
    v = v[v.year >= year_from]
    if year_to is not None:
        v = v[v.year <= year_to]
    col = POINTS_COL[vote_type]
    out = v[["year", "from_country_id", "to_country_id", col]].rename(columns={col: "pts"})
    return out.dropna(subset=["pts"])


def affinity_matrix(
    vote_type: str = "total",
    round_: str = "final",
    year_from: int = 1975,
    year_to: int | None = None,
    mode: str = "mean",
) -> pd.DataFrame:
    """Country×country average points sent.

    `mode`:
      - 'mean'   — mean points per joint appearance (raw affinity)
      - 'excess' — observed mean minus expected baseline where
                   expected[i,j] = row_mean[i] * col_mean[j] / grand_mean.
                   Positive ⇒ A votes for B more than B's overall popularity
                   and A's overall generosity would predict.
    """
    v = _filter_votes(round_, year_from, year_to, vote_type)
    if v.empty:
        return pd.DataFrame()
    m = (v.groupby(["from_country_id", "to_country_id"])["pts"].mean().unstack())
    if mode == "mean":
        return m
    if mode == "excess":
        # Use 0 for missing (countries never voted) so expected is finite.
        filled = m.fillna(0)
        row = filled.mean(axis=1)
        col = filled.mean(axis=0)
        grand = filled.values.mean()
        if grand == 0:
            return m
        expected = np.outer(row, col) / grand
        excess = filled - expected
        return excess
    raise ValueError(f"unknown mode {mode!r}")


def reorder_by_clustering(m: pd.DataFrame) -> pd.DataFrame:
    """Reorder rows/cols using hierarchical clustering of voting profiles.

    Bloc structure shows up as diagonal blocks. Symmetrizes the matrix first
    so voters and recipients share an ordering.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    common = m.index.intersection(m.columns)
    mm = m.loc[common, common].fillna(0).values
    sym = (mm + mm.T) / 2
    # Distance = -similarity (shifted to non-negative)
    dist = sym.max() - sym
    np.fill_diagonal(dist, 0)
    if (dist != dist.T).any():
        dist = (dist + dist.T) / 2
    cond = squareform(dist, checks=False)
    z = linkage(cond, method="average")
    order = leaves_list(z)
    ordered = common[order]
    return m.loc[ordered, ordered]


def bloc_strength_over_time(
    vote_type: str = "total",
    round_: str = "final",
    smooth_window: int = 5,
) -> pd.DataFrame:
    """For each canonical bloc, mean *within-bloc* points per pair-year.

    Returns rows = year, cols = bloc name. Optionally smoothed (rolling mean).
    """
    v = _filter_votes(round_, 1975, None, vote_type)
    frames = []
    for name, codes in CANONICAL_BLOCS.items():
        s = v[v.from_country_id.isin(codes) & v.to_country_id.isin(codes) &
              (v.from_country_id != v.to_country_id)]
        per_year = s.groupby("year")["pts"].mean()
        per_year.name = name
        frames.append(per_year)
    df = pd.concat(frames, axis=1)
    if smooth_window > 1:
        df = df.rolling(smooth_window, min_periods=1).mean()
    return df


def split_winners() -> pd.DataFrame:
    """Years where overall winner / jury winner / televote winner are not all the same."""
    f = ed.final_results()
    f = f.dropna(subset=["points_jury_final", "points_tele_final"])
    rows = []
    for y, grp in f.groupby("year"):
        overall = grp.loc[grp.points_final.idxmax()]
        jury    = grp.loc[grp.points_jury_final.idxmax()]
        tele    = grp.loc[grp.points_tele_final.idxmax()]
        winners = {overall.to_country, jury.to_country, tele.to_country}
        if len(winners) > 1:
            rows.append({
                "year": int(y),
                "overall": f"{overall.to_country} · {overall.performer}",
                "jury":    f"{jury.to_country} · {jury.performer}",
                "tele":    f"{tele.to_country} · {tele.performer}",
                "overall_pts": int(overall.points_final),
                "jury_pts":    int(jury.points_jury_final),
                "tele_pts":    int(tele.points_tele_final),
            })
    return pd.DataFrame(rows).sort_values("year", ascending=False)


def snub_boost(top_n: int = 20) -> pd.DataFrame:
    """Per-finalist jury-vs-televote gap. Positive `delta` = tele loved, jury didn't.

    Restricted to years with split (2016+).
    """
    f = ed.final_results().dropna(subset=["points_jury_final", "points_tele_final"]).copy()
    f["delta"] = f["points_tele_final"] - f["points_jury_final"]
    f["abs_delta"] = f["delta"].abs()
    f["year"] = f["year"].astype(int)
    f["place_final"] = f["place_final"].astype(int)
    return f.sort_values("abs_delta", ascending=False).head(top_n)


def running_order_effect(vote_type: str = "total") -> pd.DataFrame:
    """Per-finalist (year, running order, place, points). Vote_type picks which points col."""
    f = ed.final_results().copy()
    pts_col = {"total": "points_final",
               "jury":  "points_jury_final",
               "tele":  "points_tele_final"}[vote_type]
    out = f[["year", "to_country", "performer", "song", "running_final",
             "place_final", pts_col]].rename(columns={pts_col: "pts"})
    out = out.dropna(subset=["running_final", "pts"])
    out["running_final"] = out["running_final"].astype(int)
    out["place_final"]   = out["place_final"].astype(int)
    return out
