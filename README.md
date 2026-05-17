# Eurovision

A quick [Claude Code](https://claude.ai/code) experiment: pull 70 years of
Eurovision Song Contest voting data and poke around to see what jumps out.
Built in a morning, take it as a sketch rather than a finished product.

## Live demo

(Deployed to Streamlit Community Cloud — link to be added once it's up.)

## What's in it

An interactive Streamlit dashboard with a handful of views over 1956–2026:

- **Winners** — every winning entry on a timeline, with optional normalization
  to "% of max possible" or margin over runner-up (the points system has
  changed several times, so absolute totals aren't comparable across eras).
- **Affinity heatmap** — who-votes-for-whom averages, with optional "excess
  over baseline" mode that strips out generic popularity to surface bilateral
  bias.
- **Jury vs televote** — diverging bar chart per year, all-time snub/boost
  ranking, table of years where jury and televote winners diverged.
- **Patterns & blocs** — excess-affinity heatmap reordered by hierarchical
  clustering (the famous Nordic / Balkan / Caucasus / ex-Yugoslav blocs pop
  out as diagonal blocks), bloc strength over time, mutual-surprise pairs.
- **Flow map** — points-flow arrows on a map of Europe (Australia relocated
  to a corner), with three modes: single year's 12/10-pointers, multi-year
  averaged top pairs, or all flows in/out of one focused country.
- **Predictors** — running-order effect on placement, language buckets,
  Big 5 / host underperformance since the semi-final era began.

All vote-type-aware (Total / Jury-only / Televote-only where the split is
available, i.e. 2016+).

## Data sources

- [Spijkervet/eurovision-dataset](https://github.com/Spijkervet/eurovision-dataset)
  for contestants, pairwise votes, and bookmaker odds, 1956–2023.
- Wikipedia (`Eurovision_Song_Contest_<year>` pages) for 2023–2026 backfill:
  finalists, jury / televote split, and per-country voting matrices.

## Caveats

- The 1956–1974 era used many different voting systems; raw point totals from
  those years are not comparable to today. The "% of max possible" toggle and
  "margin over runner-up" toggle on the Winners tab attempt to make eras
  comparable but neither is perfect (field size effects still aren't handled).
- The language-effect analysis only has the last few years populated.
- 1969 had a four-way tie so the "margin over 2nd" metric is undefined.
- Spijkervet's 2023 release was published mid-contest and missed the top
  finishers; those are backfilled from Wikipedia.
- Country geography on the flow map: Australia is relocated for visibility;
  the orange dotted halo indicates this.

## Local development

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -e .

# Re-fetch raw data (Spijkervet release + Wikipedia HTML pages)
python -m eurovision.ingest

# Re-build the processed parquet tables from raw
python -m eurovision.build

# Launch the dashboard
streamlit run dashboard/app.py
```

## Layout

```
src/eurovision/   data ingestion, cleaning, analysis helpers
dashboard/        streamlit app
data/raw/         scraped CSVs and Wikipedia HTML (gitignored)
data/processed/   cleaned parquet tables (committed)
```
