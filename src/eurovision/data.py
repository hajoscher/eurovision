"""Read-only accessors for the processed parquet tables."""
from __future__ import annotations

from functools import cache
from pathlib import Path

import pandas as pd

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"


@cache
def contestants() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "contestants.parquet")


@cache
def final_results() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "final_results.parquet")


@cache
def votes() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "votes.parquet")


def winners() -> pd.DataFrame:
    f = final_results()
    return f[f.place_final == 1].sort_values("year")


def affinity_matrix(
    round_: str = "final",
    year_from: int = 1975,
    year_to: int | None = None,
    normalize: str = "per_opportunity",
) -> pd.DataFrame:
    """Average points sent from country A → country B, restricted to contests where both
    participated in the chosen round.

    `normalize`:
    - 'per_opportunity' — mean(total_points) across years both were eligible (default)
    - 'sum' — raw cumulative points sent
    """
    v = votes()
    v = v[v["round"] == round_].copy()
    v = v[v.year >= year_from]
    if year_to is not None:
        v = v[v.year <= year_to]
    if normalize == "sum":
        m = (v.groupby(["from_country_id", "to_country_id"])
               ["total_points"].sum().unstack(fill_value=0))
    else:
        m = (v.groupby(["from_country_id", "to_country_id"])
               ["total_points"].mean().unstack())
    return m
