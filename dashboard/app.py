"""Eurovision dashboard — interactive visual exploration of 1956–2026 voting data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from eurovision import data as ed
from eurovision import analysis as an
from eurovision.countries import CODE_TO_NAME
from eurovision.coords import COORDS, RELOCATED, bearing, great_circle_point

# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Eurovision · 1956–2026",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

ESC_BLUE = "#1A1B7C"
ESC_PINK = "#FF1F8C"
ESC_GOLD = "#F6CF5C"
ESC_TEAL = "#3DCDC2"

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 1.5rem; padding-bottom: 1rem; }}
      h1, h2, h3 {{ font-family: 'Helvetica Neue', sans-serif; letter-spacing: -0.01em; }}
      .stMetric {{ background: linear-gradient(135deg, {ESC_BLUE}15, {ESC_PINK}10);
                   padding: 12px 16px; border-radius: 12px; }}
      .stTabs [data-baseweb="tab-list"] {{ gap: 24px; }}
      .stTabs [data-baseweb="tab"] {{ font-weight: 600; font-size: 0.95rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"<h1 style='margin-bottom:0;'>🎤 <span style='color:{ESC_PINK}'>Euro</span>"
    f"<span style='color:{ESC_BLUE}'>vision</span> "
    f"<span style='color:{ESC_GOLD}'>1956–2026</span></h1>"
    "<p style='color:#666; margin-top:4px;'>"
    "70 years of votes, snubs, twelve-pointers, and Nordic alliances."
    "</p>",
    unsafe_allow_html=True,
)

contestants = ed.contestants()
finals = an.normalize_finals()   # adds max_possible, pct_of_max, total_awarded, pct_of_total
votes = ed.votes()
winners = finals[finals.place_final == 1].sort_values("year")

# ─────────────────────────────────────────────────────────────────────────────
# Headline metrics
# ─────────────────────────────────────────────────────────────────────────────
latest = winners.iloc[-1]
m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Latest winner",
    f"{latest['to_country']} {int(latest['year'])}",
    f"{latest['performer']} · “{latest['song']}”",
)
m2.metric("Editions held", int(finals["year"].nunique()))
m3.metric("Distinct countries", int(contestants["to_country_id"].nunique()))
m4.metric("Pairwise votes", f"{len(votes):,}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Global filters")
    st.caption(
        "Year range affects every tab. Vote type affects everywhere it's meaningful "
        "(Affinity, Patterns, Flow map, Country focus, Winners — and toggles the "
        "'who would have won by jury/tele' view on the Winners tab). The Year "
        "drill-down tab has its own year picker."
    )
    yr_min, yr_max = int(votes.year.min()), int(finals.year.max())
    year_range = st.slider("Year range", yr_min, yr_max, (1975, yr_max), step=1)
    vote_type_global = st.radio(
        "Vote type",
        options=["total", "jury", "tele"],
        format_func=lambda v: {"total": "Total points",
                               "jury": "Jury only (2016+)",
                               "tele": "Televote only (2016+)"}[v],
        index=0,
        help="Jury/Tele restrict to 2016+ where the split is reported.",
    )

countries_list = sorted(contestants["to_country"].dropna().unique())
years_desc = sorted(finals.year.unique(), reverse=True)


def _ensure_split_years(yr: tuple[int, int]) -> tuple[int, int]:
    """For jury/tele views, clamp to 2016+ where the split exists."""
    if vote_type_global in ("jury", "tele"):
        return (max(2016, yr[0]), yr[1])
    return yr


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_winners, tab_affinity, tab_split, tab_patterns, tab_flow, tab_predictors, tab_country, tab_year, tab_history = st.tabs([
    "🏆 Winners",
    "🔥 Affinity",
    "⚖️ Jury vs Televote",
    "🧭 Patterns & blocs",
    "🗺️ Flow map",
    "🎲 Predictors",
    "🌍 Country focus",
    "📅 Year drill-down",
    "📊 History",
])

# ── Winners over time ───────────────────────────────────────────────────────
with tab_winners:
    # Vote-type-aware "winner": for jury/tele, show who would have won that year
    # by jury-only / televote-only. Restricted to 2016+ when split exists.
    if vote_type_global == "total":
        w_all = finals[finals.place_final == 1].copy()
        pts_col_w = "points_final"
        pts_label = "Total"
        max_per_voter = None
    else:
        split_col = "points_jury_final" if vote_type_global == "jury" else "points_tele_final"
        f_split = finals.dropna(subset=[split_col])
        idx = f_split.groupby("year")[split_col].idxmax()
        w_all = f_split.loc[idx].copy()
        pts_col_w = split_col
        pts_label = "Jury" if vote_type_global == "jury" else "Televote"
        max_per_voter = 12  # jury and televote each award up to 12 per voter
    # Apply global year-range filter
    w = w_all[w_all.year.between(*year_range)].copy()
    # Recompute pct_of_max + runner-up margin against the right point column
    if vote_type_global == "total":
        # Already in normalize_finals()
        pass
    else:
        # max possible for jury-only or tele-only = (n_voters - 1) × 12
        w["max_possible"] = (w["n_voters"] - 1) * max_per_voter
        w["pct_of_max"]   = (w[pts_col_w] / w["max_possible"] * 100).round(1)
        # Runner-up by this metric (the second-highest jury/tele scorer that year)
        runners = (finals.dropna(subset=[pts_col_w])
                         .sort_values(["year", pts_col_w], ascending=[True, False])
                         .groupby("year").nth(1)[pts_col_w])
        w["runner_up_pts"] = w["year"].map(runners)
        w["margin_ratio"]  = (w[pts_col_w] / w["runner_up_pts"]).round(3)

    st.subheader(f"Every Eurovision winner ({pts_label.lower()}-points basis)"
                 if vote_type_global != "total" else "Every Eurovision winner")
    if vote_type_global != "total":
        st.caption(
            f"Showing who would have won each year if only the **{pts_label.lower()}** "
            f"counted. Restricted to 2016+ where the jury/tele split is reported."
        )
    score_mode = st.radio(
        "Score", ["Total points", "% of max possible", "Margin over 2nd"],
        horizontal=True, key="winners_score",
        help=(
            "Three different lenses on dominance:\n"
            "- **Total points** — raw, but caps changed a lot (10 pts in 1974 vs 24+ "
            "today).\n"
            "- **% of max possible** — points / ((n_voters−1) × max_per_voter). "
            "Normalizes for voter count and per-voter point cap.\n"
            "- **Margin over 2nd** — winner_pts / runner-up_pts. Pure field-relative "
            "dominance. ABBA 1974 = 1.33×, Italy 1964 = 2.88× (most dominant ever)."
        ),
    )
    score_map = {
        "Total points":      (pts_col_w,      f"{pts_label} points"),
        "% of max possible": ("pct_of_max",   "% of maximum possible"),
        "Margin over 2nd":   ("margin_ratio", "Winner pts / runner-up pts"),
    }
    y_col, y_title = score_map[score_mode]
    w_plot = w.dropna(subset=[y_col]).copy()
    if w_plot.empty:
        st.warning("No winners match the current year-range / vote-type filter.")
    else:
        if score_mode == "Margin over 2nd":
            st.caption("1969 (4-way tie) is excluded — no runner-up.")
        fig = px.scatter(
            w_plot, x="year", y=y_col, text="to_country",
            hover_data={
                "performer": True, "song": True,
                pts_col_w: ":.0f", "max_possible": ":.0f", "pct_of_max": ":.1f",
                "runner_up_pts": ":.0f", "margin_ratio": ":.2f",
                y_col: False, "to_country": False,
            },
            color=y_col, color_continuous_scale="Plasma",
            size=y_col, size_max=22,
        )
        fig.update_traces(textposition="top center", textfont=dict(size=10))
        fig.update_layout(
            height=520, plot_bgcolor="white",
            yaxis_title=y_title, xaxis_title=None,
            coloraxis_showscale=False, margin=dict(l=20, r=20, t=20, b=20),
        )
        if score_mode == "Margin over 2nd":
            fig.add_hline(y=1.0, line_dash="dot", line_color="grey",
                          annotation_text="1.00 = tie", annotation_position="bottom right")
        st.plotly_chart(fig, width="stretch")

    leaderboard = (w.groupby("to_country").size()
                   .sort_values(ascending=False).head(15).reset_index(name="wins"))
    st.subheader(f"Most-winning countries ({year_range[0]}–{year_range[1]}, "
                 f"{pts_label.lower()} basis)")
    if leaderboard.empty:
        st.info("No winners in this range.")
    else:
        fig2 = px.bar(leaderboard, x="wins", y="to_country", orientation="h",
                      color="wins", color_continuous_scale="Viridis", text="wins")
        fig2.update_layout(yaxis=dict(autorange="reversed"), height=420,
                           plot_bgcolor="white", coloraxis_showscale=False,
                           margin=dict(l=20, r=20, t=10, b=20),
                           xaxis_title="Wins", yaxis_title=None)
        st.plotly_chart(fig2, width="stretch")

# ── Affinity heatmap ────────────────────────────────────────────────────────
with tab_affinity:
    yr = _ensure_split_years(year_range)
    st.subheader(f"Who votes for whom? · {vote_type_global.title()} · {yr[0]}–{yr[1]}")
    c1, c2 = st.columns([3, 1])
    with c2:
        mode = st.radio("Mode", ["mean", "excess"],
                        format_func=lambda x: {"mean": "Raw average pts",
                                               "excess": "Excess over baseline"}[x],
                        help="Excess = observed − expected (controls for how popular the "
                             "recipient is and how generous the voter is overall). "
                             "Reveals true bilateral bias.")
        min_appearances = st.slider("Min joint appearances", 3, 30, 8)
        cluster = st.checkbox("Reorder by clustering", value=(mode == "excess"))
    m = an.affinity_matrix(vote_type_global, year_from=yr[0], year_to=yr[1], mode=mode)
    if m.empty:
        st.warning("No data for this selection.")
    else:
        counts = (votes[(votes["round"] == "final") & votes.year.between(*yr)]
                  .groupby(["from_country_id", "to_country_id"]).size().unstack(fill_value=0))
        keep_rows = counts.sum(axis=1).ge(min_appearances * 5)
        keep_cols = counts.sum(axis=0).ge(min_appearances * 5)
        m = m.loc[m.index.intersection(keep_rows[keep_rows].index),
                  m.columns.intersection(keep_cols[keep_cols].index)]
        if m.size == 0:
            with c1:
                st.info("Year range / min-appearances filter leaves no data. "
                        "Widen the year range or lower the min-appearances threshold.")
        else:
            if cluster:
                m = an.reorder_by_clustering(m)
            m.index   = [CODE_TO_NAME.get(c, c) for c in m.index]
            m.columns = [CODE_TO_NAME.get(c, c) for c in m.columns]
            if not cluster:
                m = m.sort_index().sort_index(axis=1)
            if mode == "excess":
                zmax = float(np.nanmax(np.abs(m.values)))
                fig = go.Figure(go.Heatmap(
                    z=m.values, x=m.columns, y=m.index,
                    colorscale="RdBu", zmid=0, zmin=-zmax, zmax=zmax,
                    colorbar=dict(title="Excess pts"),
                    hovertemplate="Voter: %{y}<br>→ %{x}<br>Excess: %{z:.2f}<extra></extra>",
                ))
            else:
                fig = go.Figure(go.Heatmap(
                    z=m.values, x=m.columns, y=m.index,
                    colorscale="Magma", colorbar=dict(title="Avg pts"),
                    hovertemplate="Voter: %{y}<br>→ %{x}<br>Avg: %{z:.2f}<extra></extra>",
                ))
            with c1:
                fig.update_layout(height=720, plot_bgcolor="white",
                                  margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, width="stretch")

    st.subheader(f"Top voting affinities — {vote_type_global}")
    v = votes[(votes["round"] == "final") & votes.year.between(*yr)].copy()
    pts_col = an.POINTS_COL[vote_type_global]
    v = v[v[pts_col].notna()]
    pair = (v.groupby(["from_country_id", "to_country_id"])
              .agg(avg=(pts_col, "mean"), n=(pts_col, "size")).reset_index())
    pair = pair[pair.n >= min_appearances]
    pair["from"] = pair["from_country_id"].map(CODE_TO_NAME).fillna(pair["from_country_id"])
    pair["to"]   = pair["to_country_id"].map(CODE_TO_NAME).fillna(pair["to_country_id"])
    top = pair.sort_values("avg", ascending=False).head(20)[["from", "to", "avg", "n"]]
    top.columns = ["Voter", "Recipient", "Avg pts", "Joint years"]
    st.dataframe(top, hide_index=True, width="stretch")

# ── Jury vs televote ────────────────────────────────────────────────────────
with tab_split:
    split_years = sorted(
        finals[finals["points_jury_final"].notna()
               & finals["points_tele_final"].notna()]
        .year.unique(), reverse=True,
    )
    split_year = st.selectbox("Year", split_years, key="split_year",
                              help="Years with jury/televote split (2016+).")
    st.subheader(f"Jury and televote — {split_year}")
    st.caption(
        "Each country in that year's final is shown twice: jury points extend "
        "leftward (cool), televote rightward (warm). The wider the asymmetry, "
        "the more the two sides disagreed."
    )
    fy = finals[finals.year == split_year].sort_values("place_final").copy()
    fy = fy.dropna(subset=["points_jury_final", "points_tele_final"])
    if fy.empty:
        st.info(f"No jury/televote split available for {split_year}.")
    else:
        fy = fy.sort_values("points_final")  # ascending so winner is at top of horizontal chart
        fy["label"] = (fy["place_final"].astype(int).astype(str) + ".  "
                       + fy["to_country"] + " — " + fy["performer"])
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=fy["label"], x=-fy["points_jury_final"], orientation="h",
            name="Jury", marker_color=ESC_TEAL,
            customdata=np.stack([fy["points_jury_final"], fy["song"]], axis=1),
            hovertemplate="<b>%{y}</b><br>Jury: %{customdata[0]:.0f}<br>"
                          "Song: %{customdata[1]}<extra></extra>",
            text=fy["points_jury_final"].astype(int), textposition="outside",
        ))
        fig.add_trace(go.Bar(
            y=fy["label"], x=fy["points_tele_final"], orientation="h",
            name="Televote", marker_color=ESC_PINK,
            customdata=np.stack([fy["points_tele_final"], fy["song"]], axis=1),
            hovertemplate="<b>%{y}</b><br>Tele: %{customdata[0]:.0f}<br>"
                          "Song: %{customdata[1]}<extra></extra>",
            text=fy["points_tele_final"].astype(int), textposition="outside",
        ))
        max_x = max(fy["points_jury_final"].max(), fy["points_tele_final"].max()) * 1.15
        fig.update_layout(
            barmode="overlay", height=max(420, 32 * len(fy)),
            plot_bgcolor="white", margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(title="← Jury        Televote →", range=[-max_x, max_x],
                       tickformat="d", tickvals=[-max_x, -max_x/2, 0, max_x/2, max_x],
                       ticktext=[f"{int(max_x)}", f"{int(max_x/2)}", "0",
                                 f"{int(max_x/2)}", f"{int(max_x)}"]),
            yaxis=dict(title=None), legend=dict(orientation="h", y=1.05, x=0),
        )
        fig.add_vline(x=0, line_color="#333", line_width=1)
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader(f"Biggest jury-vs-televote disagreements ({year_range[0]}–{year_range[1]})")
    st.caption("Positive delta = televote loved them more; negative = jury did. "
               "Filtered by the global year range (sidebar).")
    sb = an.snub_boost(top_n=200).copy()
    sb = sb[sb.year.between(*year_range)].head(20)
    if sb.empty:
        st.info("No jury/televote split data in this year range. "
                "The split was only reported from 2016 onward — broaden the year "
                "slider to include 2016+ to see this.")
    else:
        sb["delta"] = sb["delta"].astype(int)
        sb["points_jury_final"] = sb["points_jury_final"].astype(int)
        sb["points_tele_final"] = sb["points_tele_final"].astype(int)
        fig = px.bar(
            sb.sort_values("delta"), x="delta", y=sb.sort_values("delta").apply(
                lambda r: f"{int(r['year'])} · {r['to_country']} · {r['performer']}", axis=1),
            orientation="h", color="delta", color_continuous_scale="RdBu", color_continuous_midpoint=0,
            hover_data={"points_jury_final": True, "points_tele_final": True, "place_final": True},
        )
        fig.update_layout(height=560, plot_bgcolor="white", yaxis_title=None,
                          xaxis_title="Televote − Jury", coloraxis_showscale=False,
                          margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Years the jury and the public didn't agree on the winner")
    st.caption("Editions where the overall, jury, and televote winners weren't all the same entry. "
               "Filtered by the global year range.")
    sw = an.split_winners()
    sw = sw[sw.year.between(*year_range)]
    if sw.empty:
        st.info("No data — jury/televote split is only reported from 2016 onward.")
    else:
        st.dataframe(
            sw.rename(columns={
                "year": "Year", "overall": "Overall winner", "jury": "Jury winner",
                "tele": "Televote winner", "overall_pts": "Pts (overall)",
                "jury_pts": "Pts (jury)", "tele_pts": "Pts (tele)",
            }),
            hide_index=True, width="stretch",
        )

# ── Patterns & blocs ────────────────────────────────────────────────────────
with tab_patterns:
    yr = _ensure_split_years(year_range)
    st.subheader(f"Voting blocs · {vote_type_global.title()} · {yr[0]}–{yr[1]}")
    st.caption(
        "Cells show **excess** affinity (observed − expected if voting were popularity-driven). "
        "Rows/cols reordered by hierarchical clustering — the red squares on the diagonal "
        "are voting blocs."
    )
    m = an.affinity_matrix(vote_type_global, year_from=yr[0], year_to=yr[1], mode="excess")
    if m.empty:
        st.warning("No data for this selection.")
    else:
        # Require some minimum mass
        counts = (votes[(votes["round"] == "final") & votes.year.between(*yr)]
                  .groupby(["from_country_id", "to_country_id"]).size().unstack(fill_value=0))
        thr = 5 * 8  # ~8 joint appearances
        kr = counts.sum(axis=1).ge(thr)
        kc = counts.sum(axis=0).ge(thr)
        m = m.loc[m.index.intersection(kr[kr].index),
                  m.columns.intersection(kc[kc].index)]
        if m.size == 0:
            st.info("Year range too narrow to show a bloc matrix — try a multi-year window.")
        else:
            m = an.reorder_by_clustering(m)
            m.index   = [CODE_TO_NAME.get(c, c) for c in m.index]
            m.columns = [CODE_TO_NAME.get(c, c) for c in m.columns]
            zmax = float(np.nanmax(np.abs(m.values)))
            fig = go.Figure(go.Heatmap(
                z=m.values, x=m.columns, y=m.index,
                colorscale="RdBu", zmid=0, zmin=-zmax, zmax=zmax,
                colorbar=dict(title="Excess"),
                hovertemplate="%{y} → %{x}: %{z:.2f}<extra></extra>",
            ))
            fig.update_layout(height=720, plot_bgcolor="white",
                              margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader(f"Strength of canonical blocs over time — {vote_type_global}")
    st.caption("Mean points exchanged inside each bloc per year, 5-year rolling mean.")
    bs = an.bloc_strength_over_time(vote_type_global)
    bs_long = bs.reset_index().melt("year", var_name="bloc", value_name="avg_pts").dropna()
    fig = px.line(bs_long, x="year", y="avg_pts", color="bloc",
                  color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(height=460, plot_bgcolor="white",
                      yaxis_title="Avg points exchanged within bloc",
                      xaxis_title=None,
                      margin=dict(l=20, r=20, t=10, b=20))
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader(f"Bilateral surprises · {vote_type_global}")
    st.caption("Country pairs whose mutual love most exceeds what popularity would predict.")
    v = votes[(votes["round"] == "final") & votes.year.between(*yr)].copy()
    pts_col = an.POINTS_COL[vote_type_global]
    v = v[v[pts_col].notna()]
    base = an.affinity_matrix(vote_type_global, year_from=yr[0], year_to=yr[1], mode="excess")
    if not base.empty:
        # Mutual excess = (a→b) + (b→a)
        common = base.index.intersection(base.columns)
        b = base.loc[common, common].fillna(0)
        mutual = (b + b.T) / 2
        # Pull upper-triangle ranking
        idx = np.triu_indices_from(mutual.values, k=1)
        pairs = pd.DataFrame({
            "A": [common[i] for i in idx[0]], "B": [common[j] for j in idx[1]],
            "mutual_excess": mutual.values[idx],
        })
        pairs["A"] = pairs["A"].map(CODE_TO_NAME).fillna(pairs["A"])
        pairs["B"] = pairs["B"].map(CODE_TO_NAME).fillna(pairs["B"])
        top = pairs.sort_values("mutual_excess", ascending=False).head(15)
        bot = pairs.sort_values("mutual_excess", ascending=True).head(10)
        c1, c2 = st.columns(2)
        c1.markdown("**Strongest mutual affinities**")
        c1.dataframe(top.assign(mutual_excess=top["mutual_excess"].round(2)),
                     hide_index=True, width="stretch")
        c2.markdown("**Strongest mutual coolness**")
        c2.dataframe(bot.assign(mutual_excess=bot["mutual_excess"].round(2)),
                     hide_index=True, width="stretch")

# ── Flow map ────────────────────────────────────────────────────────────────
with tab_flow:
    st.subheader(f"Points-flow map · {vote_type_global}")
    flow_mode = st.radio(
        "Mode", [
            "Single year — top votes (12 / 10 / …)",
            "Multi-year average — top pairs",
            "Country focus — flows in/out of one country",
        ],
        horizontal=False, key="flow_mode_radio",
    )
    st.caption(
        "🇦🇺 Australia is relocated to the south-east of the European frame so it "
        "stays on canvas. Other coords are real capital locations."
    )

    # Common figure builder helpers ─────────────────────────────────────────
    def _empty_geo_fig() -> go.Figure:
        f = go.Figure()
        f.update_geos(
            scope="europe", projection_type="natural earth",
            showcoastlines=True, coastlinecolor="#cfcfcf",
            showland=True, landcolor="#f8f4ea",
            showocean=True, oceancolor="#e8eff7",
            showcountries=True, countrycolor="#dcdcdc",
            lataxis_range=[30, 72], lonaxis_range=[-25, 55],
        )
        f.update_layout(height=720, margin=dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor="white")
        return f

    def _add_country_markers(f: go.Figure, codes: set[str]) -> None:
        codes = {c for c in codes if c in COORDS}
        if not codes:
            return
        f.add_trace(go.Scattergeo(
            lon=[COORDS[c][1] for c in codes],
            lat=[COORDS[c][0] for c in codes],
            text=[CODE_TO_NAME.get(c, c.upper()) for c in codes],
            mode="markers+text",
            marker=dict(size=8, color=ESC_BLUE, line=dict(color="white", width=1)),
            textposition="top center", textfont=dict(size=10, color="#222"),
            hoverinfo="text", showlegend=False,
        ))
        reloc = [c for c in RELOCATED if c in codes]
        if reloc:
            f.add_trace(go.Scattergeo(
                lon=[COORDS[c][1] for c in reloc],
                lat=[COORDS[c][0] for c in reloc],
                mode="markers",
                marker=dict(size=22, color="rgba(0,0,0,0)",
                            line=dict(color=ESC_GOLD, width=2, dash="dot")),
                hoverinfo="skip", showlegend=False,
            ))

    def _add_flows(f: go.Figure, df: pd.DataFrame, width_col: str,
                   color: str, dash: str = "solid",
                   max_width: float | None = None,
                   pts_label: str = "pts",
                   thickness_floor: float = 1.0,
                   thickness_scale: float = 7.0,
                   show_dest_dot: bool = True,
                   legend_name: str | None = None,
                   legendgroup: str | None = None,
                   line_opacity: float = 0.85,
                   arrowhead_size: int = 22) -> None:
        if df.empty:
            return
        if max_width is None:
            max_width = float(df[width_col].max())
        first = True
        for _, r in df.iterrows():
            if r["from_country_id"] not in COORDS or r["to_country_id"] not in COORDS:
                continue
            lat0, lon0 = COORDS[r["from_country_id"]]
            lat1, lon1 = COORDS[r["to_country_id"]]
            w = thickness_floor + thickness_scale * (r[width_col] / max_width) ** 1.2
            from_name = CODE_TO_NAME.get(r["from_country_id"], r["from_country_id"])
            to_name   = CODE_TO_NAME.get(r["to_country_id"], r["to_country_id"])
            hover = (f"<b>{from_name} → {to_name}</b><br>"
                     f"{r[width_col]:.1f} {pts_label}<extra></extra>")
            # Line as one trace, midpoint chevron as a separate single-point
            # trace with scalar angle. Two traces per arrow but reliable rotation.
            f.add_trace(go.Scattergeo(
                lon=[lon0, lon1], lat=[lat0, lat1], mode="lines",
                line=dict(width=w, color=color, dash=dash),
                opacity=line_opacity,
                showlegend=(legend_name is not None and first),
                name=legend_name or "",
                legendgroup=legendgroup,
                hovertemplate=hover,
            ))
            first = False
            if show_dest_dot:
                # Plotly draws scattergeo lines as great-circle arcs.
                # Put the arrowhead on the same arc (slerp midpoint), and use
                # the *local* bearing at that midpoint toward the destination so
                # the arrow tangents the curve.
                mid_lat, mid_lon = great_circle_point(lat0, lon0, lat1, lon1, 0.5)
                angle_deg = bearing(mid_lat, mid_lon, lat1, lon1)
                if angle_deg > 180:
                    angle_deg -= 360
                f.add_trace(go.Scattergeo(
                    lon=[mid_lon], lat=[mid_lat], mode="markers",
                    marker=dict(
                        symbol="arrow-wide", size=arrowhead_size,
                        angle=angle_deg, angleref="up",
                        color=color, line=dict(color="white", width=1),
                    ),
                    opacity=0.95, showlegend=False, hovertemplate=hover,
                    legendgroup=legendgroup,
                ))

    # ── Mode 1: Single year — top votes ─────────────────────────────────────
    if flow_mode.startswith("Single year"):
        v_split = votes[(votes["round"] == "final")
                        & votes["jury_points"].notna()
                        & votes["tele_points"].notna()]
        years_with_split = sorted(v_split.year.unique(), reverse=True)
        all_years        = sorted(votes[votes["round"] == "final"].year.unique(), reverse=True)
        c_y, c_pts = st.columns([1, 2])
        avail_years = all_years if vote_type_global == "total" else years_with_split
        sel_year = c_y.selectbox("Year", avail_years, key="flow_single_year")
        which = c_pts.multiselect(
            "Which top votes to show",
            options=[12, 10, 8, 7, 6], default=[12, 10],
            help="Each voter awards 12, 10, 8, 7, 6, 5, 4, 3, 2, 1 to ten recipients (per jury "
                 "and per televote in 2016+).",
        )
        pts_col = an.POINTS_COL[vote_type_global]
        v_yr = votes[(votes["round"] == "final") & (votes.year == sel_year)].copy()
        v_yr = v_yr[v_yr[pts_col].notna()]

        # In 2016+ with vote_type=total, jury and tele are summed → the 1-12 scale
        # only applies per category. Overlay both as separate layers.
        show_overlay = (vote_type_global == "total" and sel_year >= 2016)

        fig = _empty_geo_fig()
        color_for_pts = {12: ESC_PINK, 10: ESC_GOLD, 8: ESC_TEAL,
                         7: "#7B7BFF", 6: "#9C7B5E"}
        all_codes: set[str] = set()
        if show_overlay:
            for pts in which:
                jury_arrows = v_yr[v_yr["jury_points"] == pts]
                tele_arrows = v_yr[v_yr["tele_points"] == pts]
                if not jury_arrows.empty:
                    _add_flows(fig, jury_arrows, "jury_points", color_for_pts.get(pts, ESC_PINK),
                               dash="solid", pts_label=f"jury {pts}-pt",
                               thickness_floor=1.5, thickness_scale=2.5,
                               legend_name=f"Jury {pts}", legendgroup=f"jury_{pts}")
                if not tele_arrows.empty:
                    _add_flows(fig, tele_arrows, "tele_points", color_for_pts.get(pts, ESC_PINK),
                               dash="dot", pts_label=f"tele {pts}-pt",
                               thickness_floor=1.5, thickness_scale=2.5,
                               legend_name=f"Tele {pts}", legendgroup=f"tele_{pts}")
                all_codes |= set(jury_arrows["from_country_id"]) | set(jury_arrows["to_country_id"])
                all_codes |= set(tele_arrows["from_country_id"]) | set(tele_arrows["to_country_id"])
        else:
            for pts in which:
                arrows = v_yr[v_yr[pts_col] == pts]
                if arrows.empty:
                    continue
                _add_flows(fig, arrows, pts_col, color_for_pts.get(pts, ESC_PINK),
                           pts_label=f"{pts}-pt", thickness_floor=1.5, thickness_scale=2.5,
                           legend_name=f"{pts} pts", legendgroup=f"pts_{pts}")
                all_codes |= set(arrows["from_country_id"]) | set(arrows["to_country_id"])

        # Always show all competing countries for context, not just senders
        all_codes |= set(v_yr["from_country_id"]) | set(v_yr["to_country_id"])
        _add_country_markers(fig, all_codes)
        fig.update_layout(legend=dict(orientation="h", y=1.05, x=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("Solid lines = jury, dotted = televote (when both shown). "
                   "12 = pink, 10 = gold, 8 = teal, 7 = blue, 6 = brown.")

    # ── Mode 2: Multi-year average — top pairs ─────────────────────────────
    elif flow_mode.startswith("Multi-year"):
        c_yr, c_n, c_min = st.columns(3)
        floor_yr = 2016 if vote_type_global != "total" else yr_min
        dflt = (max(floor_yr, yr_max - 4), yr_max)
        flow_years = c_yr.slider("Years to average", floor_yr, yr_max, dflt,
                                 step=1, key="flow_yr_range")
        top_n = c_n.slider("Show top N pairs", 5, 150, 40, step=5, key="flow_top_n")
        min_app = c_min.slider("Min joint years", 1, 15, 3, key="flow_min_app")

        pts_col = an.POINTS_COL[vote_type_global]
        v = votes[(votes["round"] == "final") & votes.year.between(*flow_years)].copy()
        v = v[v[pts_col].notna()]
        flow = (v.groupby(["from_country_id", "to_country_id"])[pts_col]
                  .agg(["mean", "count"]).reset_index()
                  .rename(columns={"mean": "avg_pts", "count": "joint_years"}))
        flow = (flow[flow["joint_years"] >= min_app]
                .sort_values("avg_pts", ascending=False).head(top_n))
        fig = _empty_geo_fig()
        if flow.empty:
            st.warning("No flows match those filters.")
        else:
            _add_flows(fig, flow, "avg_pts", ESC_PINK, pts_label="avg pts",
                       thickness_floor=1.0, thickness_scale=7.0)
            codes = set(flow["from_country_id"]) | set(flow["to_country_id"])
            _add_country_markers(fig, codes)
        st.plotly_chart(fig, width="stretch")

        if not flow.empty:
            ranked = flow.copy()
            ranked["From"] = ranked["from_country_id"].map(CODE_TO_NAME).fillna(ranked["from_country_id"])
            ranked["To"]   = ranked["to_country_id"].map(CODE_TO_NAME).fillna(ranked["to_country_id"])
            ranked = ranked[["From", "To", "avg_pts", "joint_years"]].rename(
                columns={"avg_pts": "Avg pts", "joint_years": "Years"})
            ranked["Avg pts"] = ranked["Avg pts"].round(2)
            st.dataframe(ranked, hide_index=True, width="stretch")

    # ── Mode 3: Country focus — flows in/out of one country ────────────────
    else:
        c_cty, c_yr = st.columns([2, 3])
        codes_with_coords = sorted(
            (c for c in votes["to_country_id"].unique() if c in COORDS),
            key=lambda c: CODE_TO_NAME.get(c, c),
        )
        fc_code = c_cty.selectbox(
            "Country", codes_with_coords,
            format_func=lambda c: CODE_TO_NAME.get(c, c.upper()),
            index=codes_with_coords.index("se") if "se" in codes_with_coords else 0,
            key="flow_focus_country",
        )
        floor_yr = 2016 if vote_type_global != "total" else yr_min
        fc_years = c_yr.slider("Years to average", floor_yr, yr_max,
                               (max(floor_yr, yr_max - 4), yr_max),
                               step=1, key="flow_focus_years")
        pts_col = an.POINTS_COL[vote_type_global]
        v = votes[(votes["round"] == "final") & votes.year.between(*fc_years)].copy()
        v = v[v[pts_col].notna()]

        out_flow = (v[v.from_country_id == fc_code]
                    .groupby("to_country_id")[pts_col]
                    .agg(["mean", "count"]).reset_index()
                    .rename(columns={"mean": "avg_pts", "count": "joint_years",
                                     "to_country_id": "other"}))
        in_flow = (v[v.to_country_id == fc_code]
                   .groupby("from_country_id")[pts_col]
                   .agg(["mean", "count"]).reset_index()
                   .rename(columns={"mean": "avg_pts", "count": "joint_years",
                                    "from_country_id": "other"}))
        # Drop zero-pt averages — clutter that says "they never voted for each other"
        out_flow = out_flow[out_flow["avg_pts"] > 0].copy()
        in_flow  = in_flow[in_flow["avg_pts"] > 0].copy()
        out_flow["from_country_id"] = fc_code
        out_flow["to_country_id"]   = out_flow["other"]
        in_flow["from_country_id"]  = in_flow["other"]
        in_flow["to_country_id"]    = fc_code

        fig = _empty_geo_fig()
        max_w = max(out_flow["avg_pts"].max() if not out_flow.empty else 0,
                    in_flow["avg_pts"].max() if not in_flow.empty else 0,
                    1.0)
        out_name = f"{CODE_TO_NAME.get(fc_code, fc_code)} → others"
        in_name  = f"others → {CODE_TO_NAME.get(fc_code, fc_code)}"
        _add_flows(fig, out_flow, "avg_pts", ESC_PINK,
                   pts_label="avg pts (out)", max_width=max_w,
                   thickness_floor=2.0, thickness_scale=9.0, line_opacity=0.55,
                   legend_name=out_name, legendgroup="out")
        _add_flows(fig, in_flow, "avg_pts", ESC_TEAL, dash="solid",
                   pts_label="avg pts (in)", max_width=max_w,
                   thickness_floor=2.0, thickness_scale=9.0, line_opacity=0.55,
                   show_dest_dot=True,
                   legend_name=in_name, legendgroup="in")
        codes = (set(out_flow["other"].tolist()) | set(in_flow["other"].tolist())
                 | {fc_code})
        _add_country_markers(fig, codes)
        fig.update_layout(legend=dict(orientation="h", y=1.05, x=0))
        st.plotly_chart(fig, width="stretch")

        c_out, c_in = st.columns(2)
        c_out.markdown(f"**{CODE_TO_NAME.get(fc_code, fc_code)} → others (avg pts sent)**")
        if not out_flow.empty:
            out_tbl = out_flow.assign(
                Recipient=out_flow["other"].map(CODE_TO_NAME).fillna(out_flow["other"]),
                **{"Avg pts": out_flow["avg_pts"].round(2)},
                Years=out_flow["joint_years"].astype(int),
            )[["Recipient", "Avg pts", "Years"]].sort_values("Avg pts", ascending=False)
            c_out.dataframe(out_tbl, hide_index=True, width="stretch")
        c_in.markdown(f"**others → {CODE_TO_NAME.get(fc_code, fc_code)} (avg pts received)**")
        if not in_flow.empty:
            in_tbl = in_flow.assign(
                Voter=in_flow["other"].map(CODE_TO_NAME).fillna(in_flow["other"]),
                **{"Avg pts": in_flow["avg_pts"].round(2)},
                Years=in_flow["joint_years"].astype(int),
            )[["Voter", "Avg pts", "Years"]].sort_values("Avg pts", ascending=False)
            c_in.dataframe(in_tbl, hide_index=True, width="stretch")


# ── Predictors ──────────────────────────────────────────────────────────────
with tab_predictors:
    st.subheader("Running order effect on final placement")
    st.caption(
        "Performers who go on later in the show tend to place better. "
        "Toggle the vote type to see if the effect is stronger for jury or televote."
    )
    pv = st.radio("Points", ["total", "jury", "tele"], horizontal=True, key="ro_pv",
                  format_func=lambda v: {"total": "Total", "jury": "Jury", "tele": "Tele"}[v])
    ro = an.running_order_effect(pv)
    if pv != "total":
        ro = ro[ro.year >= 2016]
    ro = ro[ro.year.between(*year_range)]
    if ro.empty:
        st.info(
            f"No data for this combination. Jury/televote running-order data only "
            f"exists from 2016 onward; year-range slider is {year_range[0]}–"
            f"{year_range[1]}."
        )
        fig = None
    else:
        fig = px.scatter(
            ro, x="running_final", y="pts", color="place_final",
            color_continuous_scale="Plasma_r", opacity=0.6,
            hover_data=["year", "to_country", "performer", "song", "place_final"],
            trendline="lowess",
        )
    if fig is not None:
        fig.update_layout(height=460, plot_bgcolor="white",
                          xaxis_title="Running order (1 = opens show)",
                          yaxis_title={"total": "Total points",
                                       "jury":  "Jury points",
                                       "tele":  "Televote points"}[pv],
                          coloraxis_colorbar=dict(title="Place"),
                          margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig, width="stretch")

        avg_by_slot = ro.groupby("running_final").agg(
            mean_pts=("pts", "mean"), n=("pts", "size")).reset_index()
        avg_by_slot = avg_by_slot[avg_by_slot.n >= 5]
        if not avg_by_slot.empty:
            fig2 = px.bar(avg_by_slot, x="running_final", y="mean_pts",
                          color="mean_pts", color_continuous_scale="Plasma",
                          labels={"running_final": "Running-order slot",
                                  "mean_pts": "Mean points"})
            fig2.update_layout(height=320, plot_bgcolor="white", coloraxis_showscale=False,
                               margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig2, width="stretch")

    st.divider()
    st.subheader("Language effect on placement")
    st.caption(
        "English-only, native-only, or mixed lyrics. Currently limited to 2024–2026 "
        "(Wikipedia backfill) — adding language for 1956–2023 from lyrics-column "
        "inference is on the TODO list."
    )
    if "language" in contestants.columns:
        cont = contestants[contestants["place_final"].notna()
                           & contestants["language"].notna()].copy()
    else:
        cont = pd.DataFrame()
    if cont.empty:
        st.info("Language data not yet populated in the parquet cache. "
                "Re-run `python -m eurovision.build` after the next ingest pass.")
    else:
        def _lang_bucket(s: str) -> str:
            s = str(s).lower()
            if "english" in s and "," in s:
                return "Mixed (incl. English)"
            if "english" in s:
                return "English"
            return "Native / other"
        cont["lang_bucket"] = cont["language"].map(_lang_bucket)
        cont["place_final"] = cont["place_final"].astype(int)
        fig = px.box(cont, x="lang_bucket", y="place_final",
                     color="lang_bucket", points="all",
                     color_discrete_sequence=[ESC_PINK, ESC_TEAL, ESC_GOLD])
        fig.update_yaxes(autorange="reversed", title="Final placement (1 = win)")
        fig.update_layout(height=420, plot_bgcolor="white", showlegend=False,
                          xaxis_title=None, margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig, width="stretch")
        summary = (cont.groupby("lang_bucket")
                   .agg(median_place=("place_final", "median"),
                        mean_place=("place_final", "mean"),
                        wins=("place_final", lambda s: int((s == 1).sum())),
                        entries=("place_final", "size"))
                   .round(1).reset_index())
        st.dataframe(summary.rename(columns={"lang_bucket": "Bucket"}),
                     hide_index=True, width="stretch")

    st.divider()
    st.subheader("Big-5 + host: do automatic qualifiers underperform?")
    st.caption("The Big 5 (UK, France, Germany, Spain, Italy) and the host country skip the "
               "semi-finals. Skipping the semis = no live-test of the song with the audience.")
    big5 = ["gb", "fr", "de", "es", "it"]
    # Host = country that hosted year Y (we infer as previous year's winner — close enough)
    w_by_yr = winners.set_index("year")["to_country_id"].to_dict()
    host = {(y+1): w_by_yr[y] for y in w_by_yr}
    f = finals.copy()
    f["host_code"] = f["year"].map(host)
    f["auto"] = (f["to_country_id"].isin(big5) | (f["to_country_id"] == f["host_code"]))
    f = f[f.year >= 2004]  # semi-final era
    f["place_final"] = f["place_final"].astype(int)
    auto_summary = (f.groupby("auto").agg(
        mean_place=("place_final", "mean"),
        median_place=("place_final", "median"),
        wins=("place_final", lambda s: int((s == 1).sum())),
        entries=("place_final", "size")).round(2).reset_index())
    auto_summary["auto"] = auto_summary["auto"].map({True: "Auto-qualified", False: "Came via semi"})
    st.dataframe(auto_summary, hide_index=True, width="stretch")

# ── Country focus ───────────────────────────────────────────────────────────
with tab_country:
    yr = _ensure_split_years(year_range)
    sel_col, score_col = st.columns([2, 3])
    focus_country = sel_col.selectbox(
        "Country", countries_list,
        index=countries_list.index("Sweden") if "Sweden" in countries_list else 0,
        key="country_focus_pick",
    )
    score_mode = score_col.radio(
        "Score", ["Total points", "% of max possible"], horizontal=True,
        key="country_score",
    )
    st.subheader(f"Eurovision history of {focus_country} ({year_range[0]}–{year_range[1]})")
    code = next((c for c, n in CODE_TO_NAME.items() if n == focus_country), None)
    cf = (finals[finals["to_country"] == focus_country]
          .loc[lambda d: d.year.between(*year_range)]
          .sort_values("year"))
    size_col = "points_final" if score_mode == "Total points" else "pct_of_max"
    cf = cf.dropna(subset=[size_col, "place_final"])
    fig = px.scatter(
        cf, x="year", y="place_final", size=size_col, color=size_col,
        color_continuous_scale="Plasma",
        hover_data={
            "performer": True, "song": True,
            "points_final": ":.0f", "max_possible": ":.0f", "pct_of_max": ":.1f",
            "place_final": True, "year": False, size_col: False,
        },
    )
    fig.update_yaxes(autorange="reversed", title="Final placement")
    fig.update_layout(height=420, plot_bgcolor="white", coloraxis_showscale=False,
                      margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, width="stretch")

    if code:
        pts_col = an.POINTS_COL[vote_type_global]
        v = votes[(votes["round"] == "final") & votes.year.between(*yr)]
        v = v[v[pts_col].notna()]
        # Scale min joint years with the selected range — single year ⇒ 1, wide ⇒ 3
        min_joint = min(3, yr[1] - yr[0] + 1)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Top fans of {focus_country} ({vote_type_global})")
            fans = (v[v.to_country_id == code]
                    .groupby("from_country_id")[pts_col].agg(["mean", "count"]).reset_index()
                    .rename(columns={"from_country_id": "voter"}))
            fans = fans[fans["count"] >= min_joint].sort_values("mean", ascending=False).head(12)
            fans["voter"] = fans["voter"].map(CODE_TO_NAME).fillna(fans["voter"])
            fig = px.bar(fans, x="mean", y="voter", orientation="h",
                         color="mean", color_continuous_scale="Magma", text="mean")
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig.update_layout(yaxis=dict(autorange="reversed"), height=420,
                              plot_bgcolor="white", coloraxis_showscale=False,
                              xaxis_title="Avg pts given", yaxis_title=None,
                              margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader(f"{focus_country}'s top recipients ({vote_type_global})")
            sent = (v[v.from_country_id == code]
                    .groupby("to_country_id")[pts_col].agg(["mean", "count"]).reset_index()
                    .rename(columns={"to_country_id": "recipient"}))
            sent = sent[sent["count"] >= min_joint].sort_values("mean", ascending=False).head(12)
            sent["recipient"] = sent["recipient"].map(CODE_TO_NAME).fillna(sent["recipient"])
            fig = px.bar(sent, x="mean", y="recipient", orientation="h",
                         color="mean", color_continuous_scale="Viridis", text="mean")
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig.update_layout(yaxis=dict(autorange="reversed"), height=420,
                              plot_bgcolor="white", coloraxis_showscale=False,
                              xaxis_title="Avg pts received", yaxis_title=None,
                              margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig, width="stretch")

# ── Year drill-down ─────────────────────────────────────────────────────────
with tab_year:
    focus_year = st.selectbox("Year", years_desc, key="year_drilldown_pick")
    st.subheader(f"Grand Final — {focus_year}")
    fy = finals[finals.year == focus_year].sort_values("place_final")
    if fy.empty:
        st.warning("No final data for this year.")
    else:
        has_split = fy["points_jury_final"].notna().any() and fy["points_tele_final"].notna().any()
        if has_split:
            fy_plot = fy.copy()
            fy_plot["Jury"] = fy_plot["points_jury_final"].fillna(0)
            fy_plot["Televote"] = fy_plot["points_tele_final"].fillna(0)
            long = fy_plot.melt(
                id_vars=["to_country", "performer", "song", "place_final", "points_final"],
                value_vars=["Jury", "Televote"],
                var_name="Source", value_name="Points",
            )
            long["label"] = (long["place_final"].astype(int).astype(str) + ". " +
                             long["to_country"])
            order = (fy_plot.sort_values("points_final", ascending=True)
                     .assign(label=lambda d: d["place_final"].astype(int).astype(str) + ". " +
                             d["to_country"])["label"].tolist())
            fig = px.bar(
                long, x="Points", y="label", color="Source", orientation="h",
                color_discrete_map={"Jury": ESC_TEAL, "Televote": ESC_PINK},
                hover_data=["performer", "song"],
            )
            fig.update_layout(
                height=max(420, 26 * len(fy_plot)),
                yaxis=dict(categoryorder="array", categoryarray=order),
                barmode="stack", plot_bgcolor="white",
                margin=dict(l=20, r=20, t=10, b=20), xaxis_title="Points",
                yaxis_title=None, legend_title=None,
            )
            st.plotly_chart(fig, width="stretch")
        else:
            fig = px.bar(
                fy.sort_values("points_final"), x="points_final", y="to_country",
                orientation="h", color="points_final", color_continuous_scale="Plasma",
                hover_data=["performer", "song", "place_final"],
            )
            fig.update_layout(height=max(420, 26 * len(fy)), plot_bgcolor="white",
                              coloraxis_showscale=False, yaxis_title=None,
                              margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig, width="stretch")

        st.dataframe(
            fy[["place_final", "to_country", "performer", "song",
                "points_final", "points_jury_final", "points_tele_final",
                "running_final"]].rename(columns={
                "place_final": "Place", "to_country": "Country",
                "performer": "Performer", "song": "Song",
                "points_final": "Total", "points_jury_final": "Jury",
                "points_tele_final": "Tele", "running_final": "R/O",
            }),
            hide_index=True, width="stretch",
        )

# ── History tab ─────────────────────────────────────────────────────────────
with tab_history:
    st.subheader("Voting-system eras")
    st.caption(
        "Eurovision's scoring system has been re-invented many times. Each era band "
        "below shows the max points a single voter could give one entry and that voter's "
        "total point budget."
    )
    era_df = an.voting_eras()
    # Plot eras as colored vertical bands with one line for max_per_voter and one for budget
    fig = go.Figure()
    # Era bands
    era_colors = px.colors.qualitative.Pastel * 3
    for i, (start, end, name, max_pts, budget, desc) in enumerate(an.VOTING_ERAS):
        end_a = end if end is not None else 2026
        fig.add_vrect(
            x0=start - 0.5, x1=end_a + 0.5,
            fillcolor=era_colors[i], opacity=0.35, line_width=0,
            annotation_text=name, annotation_position="top left",
            annotation_font_size=10,
        )
    # Lines (drop NaN for the secret-jury year)
    era_clean = era_df.dropna(subset=["max_pts_per_voter"])
    fig.add_trace(go.Scatter(
        x=era_clean["year"], y=era_clean["max_pts_per_voter"], mode="lines",
        name="Max pts/voter to one entry",
        line=dict(color=ESC_PINK, width=3, shape="hv"),
    ))
    fig.add_trace(go.Scatter(
        x=era_clean["year"], y=era_clean["voter_budget"], mode="lines",
        name="Voter point budget",
        line=dict(color=ESC_BLUE, width=3, shape="hv"),
        yaxis="y2",
    ))
    fig.update_layout(
        height=460, plot_bgcolor="white", margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(title="Max pts per voter → one entry", side="left",
                   range=[0, 28]),
        yaxis2=dict(title="Voter point budget (total)", side="right",
                    overlaying="y", range=[0, 130]),
        legend=dict(orientation="h", y=1.05, x=0),
    )
    st.plotly_chart(fig, width="stretch")
    with st.expander("Notes on each era"):
        for start, end, name, _, _, desc in an.VOTING_ERAS:
            end_str = f"{end}" if end is not None else "present"
            st.markdown(f"**{start}–{end_str} · {name}** — {desc}")

    st.divider()
    st.subheader("Participating countries over time")
    st.caption(
        "Countries that competed at all (incl. semi-finals from 2004 onward). The "
        "growth around 1993 reflects post-Cold-War expansion; the dip in the 1990s "
        "reflects the relegation system; the further plateau from 2003 reflects "
        "semi-finals taking pressure off the limit on finalist slots."
    )
    pp = an.participation_per_year()
    annotations = {
        1956: "First contest, 7 countries",
        1993: "Eastern European expansion",
        2004: "Semi-final introduced",
        2008: "Two semi-finals",
        2016: "Jury + televote split",
        2020: "Contest cancelled (COVID)",
        2022: "Russia banned",
    }
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pp["year"], y=pp["n_competing"], mode="lines+markers",
        name="Competing", line=dict(color=ESC_BLUE, width=3),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=pp["year"], y=pp["n_finalists"], mode="lines",
        name="Finalists", line=dict(color=ESC_PINK, width=2, dash="dot"),
    ))
    for y, label in annotations.items():
        if y in pp.year.values:
            n = int(pp.loc[pp.year == y, "n_competing"].iloc[0])
            fig.add_annotation(x=y, y=n, text=label,
                               showarrow=True, arrowhead=2, ax=0, ay=-30,
                               font=dict(size=10, color="#444"),
                               bgcolor="rgba(255,255,255,0.85)", bordercolor="#888",
                               borderwidth=1, borderpad=2)
    fig.update_layout(
        height=460, plot_bgcolor="white", margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="Number of countries", xaxis_title=None,
        legend=dict(orientation="h", y=1.05, x=0),
    )
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Non-participation by year")
    st.caption(
        "For each year, the count of countries that had competed both *before* and "
        "*after* that year but didn't compete in it. **This conflates several different "
        "kinds of absence:**\n"
        "- *Voluntary boycotts* (Sweden 1976, Greece-Turkey 1970s/80s)\n"
        "- *Financial / scheduling withdrawals* (Bulgaria 2024, various NL/IT 1980s)\n"
        "- *Forced relegation* (1996–2003 system that excluded bottom-finishers — "
        "drives the big spike around 2000–2002, not boycotts)\n"
        "- *EBU bans* (Russia 2022+, Belarus 2021+)\n"
        "- *Contest cancellation* (2020 COVID)\n\n"
        "See the curated table below for cases where the reason is documented."
    )
    ab = an.absences_per_year()
    # Shade the relegation era so users can mentally subtract it
    fig = go.Figure()
    fig.add_vrect(x0=1995.5, x1=2003.5, fillcolor="#cccccc", opacity=0.35,
                  line_width=0, annotation_text="Relegation era",
                  annotation_position="top left", annotation_font_size=10)
    fig.add_trace(go.Bar(
        x=ab["year"], y=ab["n_absent"],
        marker=dict(color=ab["n_absent"], colorscale="OrRd"),
        customdata=ab["absent_countries"],
        hovertemplate="<b>%{x}</b><br>%{y} country(ies) absent<br>"
                      "%{customdata}<extra></extra>",
    ))
    fig.update_layout(
        height=380, plot_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="# of countries non-participating", xaxis_title=None,
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Notable boycotts, withdrawals, and bans (curated)**")
    st.caption("Hand-collected from Wikipedia. Not exhaustive.")
    nb = pd.DataFrame(an.NOTABLE_ABSENCES,
                      columns=["Year", "Country/-ies", "Kind", "Reason"])
    st.dataframe(nb.sort_values("Year"), hide_index=True, width="stretch")


st.divider()
st.caption(
    "Data: Spijkervet/eurovision-dataset (1956–2023) + Wikipedia (2024–2026 backfill). "
    "Built with Streamlit + Plotly."
)
