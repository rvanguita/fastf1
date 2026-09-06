"""Theme-aware Plotly figure builders.

Every figure inherits the viewer's Streamlit theme (light/dark) and paints a
transparent background so it sits flush on the page. Builders take
already-shaped DataFrames / dicts and return a ``go.Figure``; the caller does
``st.plotly_chart(fig, use_container_width=True)``.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

UP = "#2ecc71"
DOWN = "#e74c3c"
FLAT = "#8a8a8a"
GRID = "rgba(128,128,128,0.18)"


def _template() -> str:
    base = st.get_option("theme.base") or "light"
    return "plotly_dark" if base == "dark" else "plotly_white"


def _apply(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template=_template(),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t": 30, "b": 10, "l": 10, "r": 10},
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ── prediction ─────────────────────────────────────────────────────────────


def win_prob_lines(
    long_df: pd.DataFrame, color_map: dict, order: list[str]
) -> go.Figure:
    """``long_df``: columns ``dt_ref``, ``Driver`` (label), ``WinProb`` (0-1)."""
    df = long_df.copy()
    df["Win Probability (%)"] = df["WinProb"] * 100
    fig = px.line(
        df,
        x="dt_ref",
        y="Win Probability (%)",
        color="Driver",
        category_orders={"Driver": order},
        color_discrete_map=color_map,
        markers=True,
        labels={"dt_ref": "Post-race date"},
    )
    fig.update_traces(line={"width": 2.5}, marker={"size": 6})
    fig.update_layout(
        hovermode="x unified",
        legend={"title": "Driver", "orientation": "h", "y": -0.25},
        yaxis={"ticksuffix": "%", "range": [0, 100]},
    )
    return _apply(fig, height=460)


def momentum_bars(df: pd.DataFrame, label_col: str = "Abbreviation") -> go.Figure:
    """``df`` from ``compute_momentum`` — horizontal Δ-since-last-round bars."""
    d = df.dropna(subset=["delta_prev"]).copy()
    d["pp"] = d["delta_prev"] * 100
    d = d.sort_values("pp")
    colors = [UP if v > 0 else DOWN if v < 0 else FLAT for v in d["pp"]]
    fig = go.Figure(
        go.Bar(
            x=d["pp"],
            y=d[label_col],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.1f}" for v in d["pp"]],
            textposition="outside",
            hovertemplate="%{y}: %{x:+.1f} pp<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Δ win probability vs previous round (pp)",
        yaxis_title="",
    )
    return _apply(fig, height=max(280, len(d) * 26))


def importance_bar(importances: dict[str, float], k: int = 15) -> go.Figure:
    ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:k]
    feats = [f for f, _ in ranked][::-1]
    vals = [v for _, v in ranked][::-1]
    fig = go.Figure(go.Bar(x=vals, y=feats, orientation="h", marker_color="#4781D7"))
    fig.update_layout(xaxis_title="Random-forest importance", yaxis_title="")
    return _apply(fig, height=max(300, len(feats) * 26))


# ── season ────────────────────────────────────────────────────────────────


def points_bar(points_total: pd.DataFrame, color_map: dict) -> go.Figure:
    """``points_total``: columns ``FullName``, ``TeamName``, ``Points`` (sorted asc)."""
    fig = px.bar(
        points_total,
        x="Points",
        y="FullName",
        orientation="h",
        color="TeamName",
        color_discrete_map=color_map,
        text="Points",
        labels={"FullName": "", "Points": "Championship points"},
    )
    fig.update_traces(
        texttemplate="%{text:.0f}", textposition="outside", cliponaxis=False
    )
    fig.update_layout(
        showlegend=False,
        bargap=0.35,
        xaxis={"range": [0, points_total["Points"].max() * 1.08]},
    )
    return _apply(fig, height=max(350, len(points_total) * 28))


def constructor_bar(stats: pd.DataFrame, color_map: dict) -> go.Figure:
    fig = px.bar(
        stats.sort_values("Points"),
        x="Points",
        y="TeamName",
        orientation="h",
        color="TeamName",
        color_discrete_map=color_map,
        text="Points",
        labels={"TeamName": "", "Points": "Constructor points"},
    )
    fig.update_traces(
        texttemplate="%{text:.0f}", textposition="outside", cliponaxis=False
    )
    fig.update_layout(
        showlegend=False,
        bargap=0.35,
        xaxis={"range": [0, stats["Points"].max() * 1.08]},
    )
    return _apply(fig, height=max(300, len(stats) * 40))


def cumulative_points(
    cumulative: pd.DataFrame, color_map: dict, order: list[str]
) -> go.Figure:
    """``cumulative``: ``FullName``, ``RoundNumber``, ``EventName``, ``Cumulative Points``."""
    fig = px.line(
        cumulative,
        x="RoundNumber",
        y="Cumulative Points",
        color="FullName",
        category_orders={"FullName": order},
        color_discrete_map=color_map,
        markers=True,
        hover_data={"EventName": True},
        labels={"RoundNumber": "Round", "FullName": "Driver"},
    )
    fig.update_traces(line={"width": 2}, marker={"size": 5})
    fig.update_layout(
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.3, "title": "Driver"},
    )
    return _apply(fig, height=430)


def position_heatmap(pivot: pd.DataFrame) -> go.Figure:
    """``pivot``: index driver abbrev, columns event name, values finishing position."""
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdYlGn_r",
            zmin=1,
            zmax=20,
            colorbar={"title": "Finish"},
            hovertemplate="%{y} · %{x}<br>Position: %{z}<extra></extra>",
        )
    )
    fig.update_layout(xaxis={"tickangle": -35})
    return _apply(fig, height=max(320, len(pivot) * 26))


# ── deep dives ────────────────────────────────────────────────────────────


def teammate_h2h_bars(h2h: pd.DataFrame) -> go.Figure:
    """Diverging race-day win count per constructor (driver A right, B left)."""
    d = h2h.sort_values("PointsA")
    fig = go.Figure()
    fig.add_bar(
        x=d["RaceWinsA"],
        y=d["TeamName"],
        orientation="h",
        marker_color="#4781D7",
        text=[f"{a} {int(n)}" for a, n in zip(d["abbr_a"], d["RaceWinsA"])],
        textposition="inside",
        name="Driver A ahead",
        hovertemplate="%{y}: %{text}<extra></extra>",
    )
    fig.add_bar(
        x=-d["RaceWinsB"],
        y=d["TeamName"],
        orientation="h",
        marker_color="#ED9121",
        text=[f"{b} {int(n)}" for b, n in zip(d["abbr_b"], d["RaceWinsB"])],
        textposition="inside",
        name="Driver B ahead",
        hovertemplate="%{y}: %{text}<extra></extra>",
    )
    fig.update_layout(
        barmode="relative",
        legend={"orientation": "h", "y": -0.2},
        xaxis_title="◄ teammate ahead on race day ►",
        yaxis_title="",
    )
    return _apply(fig, height=max(300, len(d) * 44))


def gain_box(races: pd.DataFrame, order: list[str]) -> go.Figure:
    """``races``: season race rows with ``Abbreviation`` and ``Gain`` (grid − finish)."""
    fig = px.box(
        races.dropna(subset=["Gain"]),
        x="Abbreviation",
        y="Gain",
        category_orders={"Abbreviation": order},
        points="outliers",
        labels={"Abbreviation": "", "Gain": "Places gained (grid → finish)"},
    )
    fig.add_hline(y=0, line_dash="dot", line_color=FLAT)
    return _apply(fig, height=420)
