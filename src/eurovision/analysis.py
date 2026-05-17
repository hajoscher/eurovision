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
