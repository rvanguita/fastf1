"""F1 Champion Win Predictor — Streamlit dashboard.

Layout: an always-visible header (championship strip · top-5 probability cards ·
auto insights) over four tabs — Prediction, Season, Deep Dives, Data. All data
access and caching is in ``data.py``; pure aggregates in ``analytics.py``;
Plotly builders in ``charts.py``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import analytics
import charts
import data

RECENT_RACES_DEFAULT = 5


# ── small render helpers ───────────────────────────────────────────────────


def _driver_options(preds: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """(driver_team_id options ordered by latest win prob, {id: label})."""
    latest = preds["dt_ref"].max()
    ordered = (
        preds[preds["dt_ref"] == latest]
        .sort_values("prob_win", ascending=False)
        .drop_duplicates("driver_team_id")
    )
    labels = {
        row["driver_team_id"]: analytics.driver_label(row["FullName"], row["TeamName"])
        for _, row in ordered.iterrows()
    }
    return ordered["driver_team_id"].tolist(), labels


def _championship_strip(dstats: pd.DataFrame) -> None:
    if dstats.empty:
        return
    leader = dstats.iloc[0]
    gap = leader["Points"] - dstats.iloc[1]["Points"] if len(dstats) > 1 else 0.0
    most_wins = dstats.sort_values("Wins", ascending=False).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏁 Rounds", int(dstats["Races"].max()))
    c2.metric(
        "👑 Points leader", leader["FullName"], help=f"{leader['Points']:.0f} pts"
    )
    c3.metric("📏 Gap to P2", f"{gap:.0f} pts")
    c4.metric("🥇 Most wins", f"{most_wins['FullName']} ({int(most_wins['Wins'])})")


def _probability_cards(momentum: pd.DataFrame, n: int = 5) -> None:
    if momentum.empty:
        return
    top = momentum.dropna(subset=["latest"]).head(n)
    if top.empty:
        return
    for col, (_, row) in zip(st.columns(len(top)), top.iterrows()):
        with col:
            url = row.get("HeadshotUrl")
            if isinstance(url, str) and url.startswith("http"):
                st.image(url, width=72)
            delta = row.get("delta_prev")
            col.metric(
                label=row.get("FullName", row["DriverId"]),
                value=f"{row['latest']:.0%}",
                delta=None if pd.isna(delta) else f"{delta * 100:+.1f} pp",
                help=row.get("TeamName", ""),
            )


def _snapshot(dstats: pd.DataFrame, races_season: pd.DataFrame) -> None:
    if dstats.empty:
        st.info("No race data for this season yet.")
        return
    leader = dstats.iloc[0]
    gap = leader["Points"] - dstats.iloc[1]["Points"] if len(dstats) > 1 else 0.0
    most_wins = dstats.sort_values("Wins", ascending=False).iloc[0]

    latest_round = races_season["RoundNumber"].max()
    last = races_season[races_season["RoundNumber"] == latest_round].copy()
    last["Gain"] = last["GridPosition"] - last["Position"]
    movers = last.dropna(subset=["Gain"])
    mover = (
        f"{movers.loc[movers['Gain'].idxmax(), 'Abbreviation']} "
        f"({movers['Gain'].max():+.0f})"
        if not movers.empty
        else "—"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🏁 Rounds completed", int(dstats["Races"].max()))
    c2.metric(
        "👑 Points leader", leader["FullName"], help=f"{leader['Points']:.0f} pts"
    )
    c3.metric("📏 Championship gap", f"{gap:.0f} pts")
    c4.metric("🥇 Most wins", f"{most_wins['FullName']} ({int(most_wins['Wins'])})")
    c5.metric("🚀 Biggest mover", mover, help="Grid → finish, last race")


def _recent_race_cards(races: pd.DataFrame, n: int) -> None:
    """Top-5 finishers for each of the last ``n`` race rounds (any season)."""
    df = races.dropna(subset=["Position"]).copy()
    df["Position"] = df["Position"].astype(int)

    recent = (
        df.drop_duplicates(subset=["Year", "RoundNumber"])
        .sort_values("Date", ascending=False)
        .head(n)[["Year", "RoundNumber", "EventName", "Date", "Country"]]
    )
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}

    for col, (_, rnd) in zip(st.columns(len(recent)), recent.iterrows()):
        top5 = (
            df[
                (df["Year"] == rnd["Year"])
                & (df["RoundNumber"] == rnd["RoundNumber"])
                & (df["Position"] <= 5)
            ]
            .sort_values("Position")
            .reset_index(drop=True)
        )
        with col.container(border=True):
            st.markdown(f"**{rnd['EventName']}**")
            st.caption(
                f"{rnd.get('Country', '')} · {pd.to_datetime(rnd['Date']):%b %d, %Y}"
            )
            for _, d in top5.iterrows():
                pos = int(d["Position"])
                grid = d["GridPosition"]
                move = ""
                if pd.notna(grid):
                    diff = int(grid) - pos
                    move = (
                        f" ▲{diff}"
                        if diff > 0
                        else f" ▼{abs(diff)}"
                        if diff < 0
                        else ""
                    )
                st.markdown(
                    f"<span style='color:{d['TeamColor'] or charts.FLAT}'>"
                    f"{medals.get(pos, f'P{pos}')} {d['Abbreviation']}</span>"
                    f"<span style='float:right;opacity:0.7'>{d['Points']:.0f}{move}</span>",
                    unsafe_allow_html=True,
                )


# ── tabs ──────────────────────────────────────────────────────────────────


def _tab_prediction(
    preds: pd.DataFrame, selected: list[str], momentum: pd.DataFrame, last_n: int
) -> None:
    sel = preds[preds["driver_team_id"].isin(selected)].copy()
    if sel.empty:
        st.warning("Select at least one driver in the sidebar.")
        return

    sel["Driver"] = sel["FullName"].map(analytics.short_name)
    long_df = sel.rename(columns={"prob_win": "WinProb"})[
        ["dt_ref", "Driver", "WinProb"]
    ]
    color_map = sel.drop_duplicates("Driver").set_index("Driver")["TeamColor"].to_dict()
    order = (
        sel[sel["dt_ref"] == sel["dt_ref"].max()]
        .sort_values("prob_win", ascending=False)["Driver"]
        .tolist()
    )
    st.subheader("Win probability over time")
    st.plotly_chart(charts.win_prob_lines(long_df, color_map, order), width="stretch")

    st.subheader("Momentum")
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(charts.momentum_bars(momentum), width="stretch")
    with right:
        moves = momentum[momentum["rank_change"].fillna(0) != 0][
            ["FullName", "rank", "rank_change"]
        ].copy()
        moves["rank"] = moves["rank"].astype(int)
        moves["Move"] = moves["rank_change"].map(
            lambda c: f"▲ {int(c)}" if c > 0 else f"▼ {int(-c)}"
        )
        st.caption("Rank changes since previous round")
        st.dataframe(
            moves[["FullName", "rank", "Move"]].rename(
                columns={"FullName": "Driver", "rank": "Rank"}
            ),
            hide_index=True,
            width="stretch",
        )

    with st.expander("🔬 What drives the prediction (model explainability)"):
        info = data.model_info()
        importances = info.get("importances") if info else None
        if not importances:
            st.caption("Model info unavailable — start the API to populate this.")
            return
        st.plotly_chart(charts.importance_bar(importances, 15), width="stretch")

        latest = preds[preds["dt_ref"] == preds["dt_ref"].max()]
        feats = [f for f in importances if f in latest.columns]
        field_median = latest[feats].median(numeric_only=True)
        who = st.selectbox("Top factors for", options=order, key="explain_driver")
        drv_row = sel[(sel["Driver"] == who) & (sel["dt_ref"] == sel["dt_ref"].max())]
        if not drv_row.empty:
            tf = analytics.top_factors(
                importances, drv_row.iloc[0][feats], field_median, k=8
            )
            st.dataframe(tf, hide_index=True, width="stretch")


def _tab_season(season: int, last_n: int) -> None:
    dstats = data.driver_stats(season)
    races_all = data.races_frame()
    races_season = races_all[races_all["Year"] == season]

    _snapshot(dstats, races_season)
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Points ranking")
        if not dstats.empty:
            pts = dstats[["FullName", "TeamName", "TeamColor", "Points"]].sort_values(
                "Points"
            )
            st.plotly_chart(
                charts.points_bar(pts, analytics._color_map(pts, "TeamName")),
                width="stretch",
            )
    with c2:
        st.subheader("Season progression")
        cumulative = _cumulative(races_season)
        if not cumulative.empty:
            order = (
                cumulative.groupby("FullName")["Cumulative Points"]
                .last()
                .sort_values(ascending=False)
                .index.tolist()
            )
            st.plotly_chart(
                charts.cumulative_points(
                    cumulative,
                    analytics._color_map(cumulative, "FullName", keep="first"),
                    order,
                ),
                width="stretch",
            )

    st.divider()
    st.subheader(f"Last {last_n} race results")
    _recent_race_cards(races_all, last_n)

    st.divider()
    st.subheader("Finishing position by round")
    _position_heatmap(races_season, dstats)


def _tab_deep_dives(season: int, preds: pd.DataFrame) -> None:
    t_drv, t_con, t_mate, t_rel = st.tabs(
        ["🧑‍🚀 Drivers", "🏗️ Constructors", "⚔️ Teammates", "🎯 Quali & reliability"]
    )

    with t_drv:
        dstats = data.driver_stats(season)
        if dstats.empty:
            st.info("No race data for this season.")
        else:
            heads = preds.drop_duplicates("FullName")[["FullName", "HeadshotUrl"]]
            table = dstats.merge(heads, on="FullName", how="left")
            st.dataframe(
                table[
                    [
                        "Rank",
                        "HeadshotUrl",
                        "FullName",
                        "TeamName",
                        "Races",
                        "Wins",
                        "Podiums",
                        "Poles",
                        "DNFs",
                        "Points",
                        "BestFinish",
                        "AvgFinish",
                        "AvgGrid",
                        "AvgGain",
                        "PodiumRate",
                    ]
                ],
                column_config={
                    "Rank": st.column_config.NumberColumn("#"),
                    "HeadshotUrl": st.column_config.ImageColumn(""),
                    "FullName": st.column_config.TextColumn("Driver"),
                    "TeamName": st.column_config.TextColumn("Team"),
                    "Wins": st.column_config.NumberColumn("🏆"),
                    "Podiums": st.column_config.NumberColumn("🥇"),
                    "Poles": st.column_config.NumberColumn("🎯"),
                    "DNFs": st.column_config.NumberColumn("❌"),
                    "Points": st.column_config.NumberColumn("Points", format="%.0f"),
                    "BestFinish": st.column_config.NumberColumn("Best", format="%.0f"),
                    "AvgFinish": st.column_config.NumberColumn(
                        "Avg fin", format="%.1f"
                    ),
                    "AvgGrid": st.column_config.NumberColumn("Avg grid", format="%.1f"),
                    "AvgGain": st.column_config.NumberColumn("Avg +/−", format="%.1f"),
                    "PodiumRate": st.column_config.ProgressColumn(
                        "Podium rate", format="%.0f%%", min_value=0, max_value=1
                    ),
                },
                hide_index=True,
                width="stretch",
                height=min(38 * (len(table) + 1) + 3, 700),
            )

    with t_con:
        tstats = data.team_stats(season)
        if tstats.empty:
            st.info("No race data for this season.")
        else:
            st.plotly_chart(
                charts.constructor_bar(
                    tstats, analytics._color_map(tstats, "TeamName")
                ),
                width="stretch",
            )
            st.dataframe(
                tstats[["Rank", "TeamName", "Wins", "Podiums", "Points"]],
                column_config={
                    "Rank": st.column_config.NumberColumn("#"),
                    "TeamName": st.column_config.TextColumn("Team"),
                    "Points": st.column_config.NumberColumn("Points", format="%.0f"),
                },
                hide_index=True,
                width="stretch",
            )

    with t_mate:
        h2h = data.teammate_h2h(season)
        if h2h.empty:
            st.info("No two-car constructor data for this season.")
        else:
            st.plotly_chart(charts.teammate_h2h_bars(h2h), width="stretch")
            rec = pd.DataFrame(
                {
                    "Team": h2h["TeamName"],
                    "Race (A–B)": h2h["RaceWinsA"].astype(int).astype(str)
                    + "–"
                    + h2h["RaceWinsB"].astype(int).astype(str),
                    "Quali (A–B)": h2h["QualiWinsA"].astype(int).astype(str)
                    + "–"
                    + h2h["QualiWinsB"].astype(int).astype(str),
                    "A": h2h["abbr_a"],
                    "B": h2h["abbr_b"],
                    "Points (A–B)": h2h["PointsA"].round().astype(int).astype(str)
                    + "–"
                    + h2h["PointsB"].round().astype(int).astype(str),
                }
            )
            st.dataframe(rec, hide_index=True, width="stretch")

    with t_rel:
        rel = data.reliability(season)
        if rel.empty:
            st.info("No race data for this season.")
        else:
            best = rel[rel["Starts"] >= 3].sort_values("AvgGrid")
            if not best.empty:
                st.caption(
                    f"Best qualifier: **{best.iloc[0]['FullName']}** "
                    f"(avg grid P{best.iloc[0]['AvgGrid']:.1f})"
                )
            st.dataframe(
                rel[
                    [
                        "FullName",
                        "TeamName",
                        "Starts",
                        "DNFs",
                        "DNFRate",
                        "PointsFinishRate",
                        "AvgGrid",
                        "BestGrid",
                        "AvgGain",
                    ]
                ],
                column_config={
                    "FullName": st.column_config.TextColumn("Driver"),
                    "TeamName": st.column_config.TextColumn("Team"),
                    "DNFRate": st.column_config.ProgressColumn(
                        "DNF rate", format="%.0f%%", min_value=0, max_value=1
                    ),
                    "PointsFinishRate": st.column_config.ProgressColumn(
                        "Points rate", format="%.0f%%", min_value=0, max_value=1
                    ),
                    "AvgGrid": st.column_config.NumberColumn("Avg grid", format="%.1f"),
                    "BestGrid": st.column_config.NumberColumn(
                        "Best grid", format="%.0f"
                    ),
                    "AvgGain": st.column_config.NumberColumn("Avg +/−", format="%.1f"),
                },
                hide_index=True,
                width="stretch",
            )
            races_season = data.races_frame()
            races_season = races_season[races_season["Year"] == season].copy()
            races_season["Gain"] = (
                races_season["GridPosition"] - races_season["Position"]
            )
            order = (
                races_season.groupby("Abbreviation")["Gain"]
                .median()
                .sort_values(ascending=False)
                .index.tolist()
            )
            st.plotly_chart(charts.gain_box(races_season, order), width="stretch")


def _tab_data(preds: pd.DataFrame, selected: list[str]) -> None:
    sel = preds[preds["driver_team_id"].isin(selected)]
    pivot = sel.pivot_table(
        index="dt_ref", columns="driver_team_id", values="prob_win"
    ).reset_index()
    cfg = {
        c: st.column_config.NumberColumn(c, format="percent") for c in pivot.columns[1:]
    }
    cfg["dt_ref"] = st.column_config.TextColumn("Prediction date")
    st.markdown("#### Win probability by prediction date")
    st.dataframe(pivot, column_config=cfg, width="stretch", hide_index=True)
    with st.expander("Full feature table (nulls preserved)"):
        st.dataframe(sel, width="stretch", hide_index=True)


# ── shared derivations ────────────────────────────────────────────────────


def _cumulative(races_season: pd.DataFrame) -> pd.DataFrame:
    df = races_season.dropna(subset=["Points", "RoundNumber"]).sort_values(
        "RoundNumber"
    )
    if df.empty:
        return df
    cum = (
        df.groupby(["FullName", "TeamName", "TeamColor", "RoundNumber", "EventName"])[
            "Points"
        ]
        .sum()
        .reset_index()
        .sort_values(["FullName", "RoundNumber"])
    )
    cum["Cumulative Points"] = cum.groupby("FullName")["Points"].cumsum()
    return cum


def _position_heatmap(races_season: pd.DataFrame, dstats: pd.DataFrame) -> None:
    df = races_season.dropna(subset=["Position", "RoundNumber"]).copy()
    if df.empty:
        st.info("No race data for this season.")
        return
    df["Position"] = df["Position"].astype(int)
    pivot = df.pivot_table(
        index="Abbreviation", columns="EventName", values="Position", aggfunc="min"
    )
    if not dstats.empty:
        order = [a for a in dstats["Abbreviation"] if a in pivot.index]
        order += [a for a in pivot.index if a not in order]
        pivot = pivot.reindex(order[::-1])
    st.plotly_chart(charts.position_heatmap(pivot), width="stretch")


# ── main ──────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="F1 Champion Predictor",
        page_icon="🏎️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    seasons = data.available_seasons()

    with st.sidebar:
        st.header("Filters")
        season = st.selectbox("📅 Season", options=seasons, index=0, key="season")
        last_n = st.slider(
            "Momentum / recent-race window", 3, 10, RECENT_RACES_DEFAULT, key="window"
        )

    with st.spinner("Loading predictions…"):
        preds = data.load_predictions(season)

    st.markdown("# 🏁 F1 — Champion Win Predictor")
    if preds.empty or preds["prob_win"].isna().all():
        st.warning(
            f"No predictions available for {season}. "
            "Is the API running and the model registered?"
        )
        return

    st.caption(
        "Model-estimated probability that each driver finishes the season as "
        f"championship points leader · latest prediction {preds['dt_ref'].max()}"
    )

    options, labels = _driver_options(preds)
    with st.sidebar:
        selected = st.multiselect(
            "🏎️ Drivers",
            options=options,
            default=options[:5],
            format_func=lambda x: labels.get(x, x),
            key="drivers",
        )
    if not selected:
        selected = options[:5]

    momentum = data.momentum(season, last_n)
    dstats = data.driver_stats(season)
    h2h = data.teammate_h2h(season)
    rel = data.reliability(season)

    st.divider()
    _championship_strip(dstats)
    st.divider()
    _probability_cards(momentum)

    insights = analytics.build_insights(momentum, h2h, rel, last_n)
    if insights:
        st.info("  \n".join(f"- {line}" for line in insights))

    st.divider()
    tab_pred, tab_season, tab_deep, tab_data = st.tabs(
        ["🔮 Prediction", "📅 Season", "🔬 Deep Dives", "📋 Data"]
    )
    with tab_pred:
        _tab_prediction(preds, selected, momentum, last_n)
    with tab_season:
        _tab_season(season, last_n)
    with tab_deep:
        _tab_deep_dives(season, preds)
    with tab_data:
        _tab_data(preds, selected)


if __name__ == "__main__":
    main()
