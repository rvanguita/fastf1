"""Visualizações editoriais Plotly do Lake FastF1."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ACCENT = "#ff4b44"
POSITIVE = "#19a974"
NEGATIVE = "#e05252"
NEUTRAL = "#737b8c"
GRID = "rgba(128,128,128,.16)"


def _template() -> str:
    return (
        "plotly_dark"
        if (st.get_option("theme.base") or "light") == "dark"
        else "plotly_white"
    )


def _apply(fig: go.Figure, height: int = 420, *, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template=_template(),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t": 36, "b": 28, "l": 12, "r": 12},
        font={"family": "Inter, ui-sans-serif, system-ui", "size": 13},
        hoverlabel={"namelength": -1, "font_size": 13},
        showlegend=legend,
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, automargin=True)
    return fig


def probability_ranking(momentum: pd.DataFrame, n: int = 10) -> go.Figure:
    d = momentum.dropna(subset=["latest"]).head(n).sort_values("latest")
    colors = [d.iloc[i].get("TeamColor", NEUTRAL) for i in range(len(d))]
    d = d.assign(
        delta_label=d["delta_prev"].map(
            lambda value: (
                "sem rodada anterior" if pd.isna(value) else f"{value * 100:+.1f} pp"
            )
        )
    )
    custom = d[["delta_label", "TeamName"]].to_numpy()
    fig = go.Figure(
        go.Bar(
            x=d["latest"] * 100,
            y=d["FullName"],
            orientation="h",
            marker={"color": colors, "line": {"color": GRID, "width": 1}},
            text=[f"{v:.1%}" for v in d["latest"]],
            textposition="outside",
            cliponaxis=False,
            customdata=custom,
            hovertemplate="<b>%{y}</b><br>Chance: %{x:.1f}%<br>Variação: %{customdata[0]}<br>%{customdata[1]}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis={
            "title": "Probabilidade normalizada",
            "ticksuffix": "%",
            "range": [0, max(5, d["latest"].max() * 115)],
        },
        yaxis_title="",
    )
    return _apply(fig, max(320, len(d) * 36), legend=False)


def probability_history(preds: pd.DataFrame, selected: list[str]) -> go.Figure:
    d = preds[preds["DriverId"].isin(selected)].copy()
    d["Probability"] = d["prob_win"] * 100
    colors = (
        d.drop_duplicates("DriverId", keep="last")
        .set_index("FullName")["TeamColor"]
        .to_dict()
    )
    fig = px.line(
        d,
        x="dt_ref",
        y="Probability",
        color="FullName",
        color_discrete_map=colors,
        markers=True,
        labels={
            "dt_ref": "Data da previsão",
            "Probability": "Chance do título",
            "FullName": "Piloto",
        },
    )
    fig.update_traces(line={"width": 2.8}, marker={"size": 6})
    fig.update_layout(
        hovermode="x unified",
        yaxis={"ticksuffix": "%", "range": [0, 100]},
        legend={"orientation": "h", "y": -0.24},
    )
    return _apply(fig, 450)


def standings_bar(stats: pd.DataFrame, n: int = 20) -> go.Figure:
    d = stats.head(n).sort_values("Points")
    leader = float(stats["Points"].max())
    d = d.assign(Gap=d["Points"] - leader)
    fig = go.Figure(
        go.Bar(
            x=d["Points"],
            y=d["FullName"],
            orientation="h",
            marker={"color": d["TeamColor"], "line": {"color": GRID, "width": 1}},
            text=[
                f"{p:.0f}  ({g:+.0f})" if g else f"{p:.0f}"
                for p, g in zip(d["Points"], d["Gap"])
            ],
            textposition="outside",
            cliponaxis=False,
            customdata=d[["Wins", "Podiums", "DNFs", "TeamName"]],
            hovertemplate="<b>%{y}</b><br>%{x:.0f} pts<br>Vitórias: %{customdata[0]}<br>Pódios: %{customdata[1]}<br>DNFs: %{customdata[2]}<br>%{customdata[3]}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis={
            "title": "Pontos · diferença para o líder entre parênteses",
            "range": [0, leader * 1.17],
        },
        yaxis_title="",
    )
    return _apply(fig, max(390, len(d) * 31), legend=False)


def points_history(history: pd.DataFrame, selected: list[str]) -> go.Figure:
    d = (
        history[history["DriverId"].isin(selected)]
        if "DriverId" in history
        else history[history["FullName"].isin(selected)]
    )
    colors = (
        d.drop_duplicates("FullName", keep="last")
        .set_index("FullName")["TeamColor"]
        .to_dict()
    )
    fig = px.line(
        d,
        x="RoundNumber",
        y="CumulativePoints",
        color="FullName",
        color_discrete_map=colors,
        markers=True,
        custom_data=["EventName"],
        labels={
            "RoundNumber": "Rodada",
            "CumulativePoints": "Pontos acumulados",
            "FullName": "Piloto",
        },
    )
    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>R%{x} · %{customdata[0]}<br>%{y:.0f} pts<extra></extra>"
    )
    fig.update_layout(hovermode="x unified", legend={"orientation": "h", "y": -0.24})
    return _apply(fig, 450)


def rank_bump(history: pd.DataFrame, selected: list[str]) -> go.Figure:
    d = (
        history[history["DriverId"].isin(selected)]
        if "DriverId" in history
        else history[history["FullName"].isin(selected)]
    )
    colors = (
        d.drop_duplicates("FullName", keep="last")
        .set_index("FullName")["TeamColor"]
        .to_dict()
    )
    fig = px.line(
        d,
        x="RoundNumber",
        y="ChampionshipRank",
        color="FullName",
        color_discrete_map=colors,
        markers=True,
        custom_data=["EventName"],
        labels={
            "RoundNumber": "Rodada",
            "ChampionshipRank": "Posição",
            "FullName": "Piloto",
        },
    )
    fig.update_yaxes(autorange="reversed", dtick=1)
    fig.update_traces(
        line={"width": 2.5},
        hovertemplate="<b>%{fullData.name}</b><br>R%{x} · %{customdata[0]}<br>P%{y:.0f}<extra></extra>",
    )
    fig.update_layout(legend={"orientation": "h", "y": -0.24})
    return _apply(fig, 450)


def result_heatmap(results: pd.DataFrame, driver_order: list[str]) -> go.Figure:
    d = results.copy()
    events = d.sort_values("RoundNumber").drop_duplicates("RoundNumber")
    event_order = events["EventName"].tolist()
    value = d.pivot_table(
        index="Abbreviation",
        columns="EventName",
        values="OfficialFinish",
        aggfunc="first",
    ).reindex(index=driver_order, columns=event_order)
    labels = (
        d.pivot_table(
            index="Abbreviation",
            columns="EventName",
            values="ResultLabel",
            aggfunc="first",
        )
        .reindex(index=driver_order, columns=event_order)
        .fillna("—")
    )
    z = value.astype("Float64").fillna(21).astype(float)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=z.columns,
            y=z.index,
            text=labels,
            texttemplate="%{text}",
            colorscale=[
                [0, "#16865f"],
                [0.12, "#66b98d"],
                [0.45, "#e7c66b"],
                [1, "#d86666"],
            ],
            zmin=1,
            zmax=21,
            showscale=False,
            hovertemplate="<b>%{y}</b> · %{x}<br>Resultado: %{text}<extra></extra>",
        )
    )
    fig.update_layout(xaxis={"tickangle": -35}, yaxis={"autorange": "reversed"})
    return _apply(fig, max(360, len(value) * 30), legend=False)


def grid_to_finish(results: pd.DataFrame, round_number: int) -> go.Figure:
    d = results[
        (results["RoundNumber"] == round_number) & results["OfficialGrid"].notna()
    ].copy()
    finish = d["OfficialFinish"].fillna(d["Position"])
    fig = go.Figure()
    for (_, row), end in zip(d.iterrows(), finish):
        color = (
            POSITIVE
            if row["OfficialGrid"] > end
            else NEGATIVE
            if row["OfficialGrid"] < end
            else NEUTRAL
        )
        fig.add_trace(
            go.Scatter(
                x=["Grid", "Chegada"],
                y=[row["OfficialGrid"], end],
                mode="lines+markers+text",
                line={"color": color, "width": 2},
                marker={"size": 9},
                text=[row["Abbreviation"], row["ResultLabel"]],
                textposition=["middle left", "middle right"],
                name=row["FullName"],
                hovertemplate=f"<b>{row['FullName']}</b><br>Grid P{row['OfficialGrid']:.0f}<br>Resultado {row['ResultLabel']}<extra></extra>",
            )
        )
    fig.update_yaxes(autorange="reversed", dtick=1, title="Posição")
    fig.update_layout(showlegend=False, xaxis={"side": "top", "title": ""})
    return _apply(fig, 560, legend=False)


def driver_dumbbell(stats: pd.DataFrame, drivers: list[str]) -> go.Figure:
    d = stats[stats["DriverId"].isin(drivers)].set_index("DriverId")
    metrics = [
        ("Points", "Pontos"),
        ("Wins", "Vitórias"),
        ("Podiums", "Pódios"),
        ("AvgGrid", "Média grid"),
        ("AvgFinish", "Média chegada"),
        ("DNFRate", "Taxa DNF"),
    ]
    names = d["FullName"].to_dict()

    def display_value(metric: str, value: float) -> str:
        if pd.isna(value):
            return "—"
        if metric == "DNFRate":
            return f"{value:.1%}"
        if metric in {"AvgGrid", "AvgFinish"}:
            return f"P{value:.1f}"
        return f"{value:.0f}"

    fig = go.Figure()
    for metric, label in metrics:
        field = stats[metric].dropna()
        lo, hi = field.min(), field.max()
        vals = []
        for driver in drivers:
            value = d.loc[driver, metric]
            score = 50 if hi == lo else (value - lo) / (hi - lo) * 100
            if metric in {"AvgGrid", "AvgFinish", "DNFRate"}:
                score = 100 - score
            vals.append((score, value))
        fig.add_trace(
            go.Scatter(
                x=[vals[0][0], vals[1][0]],
                y=[label, label],
                mode="lines",
                line={"color": GRID, "width": 4},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        for i, driver in enumerate(drivers):
            fig.add_trace(
                go.Scatter(
                    x=[vals[i][0]],
                    y=[label],
                    mode="markers",
                    marker={"size": 13, "color": d.loc[driver, "TeamColor"]},
                    name=names[driver],
                    legendgroup=driver,
                    showlegend=metric == "Points",
                    customdata=[[display_value(metric, vals[i][1])]],
                    hovertemplate=f"<b>{names[driver]}</b><br>{label}: %{{customdata[0]}}<extra></extra>",
                )
            )
    fig.update_layout(
        xaxis={
            "title": "Desempenho relativo · melhor à direita",
            "range": [-5, 105],
            "tickvals": [0, 50, 100],
            "ticktext": ["Pior", "Médio", "Melhor"],
        },
        yaxis_title="",
        legend={"orientation": "h", "y": -0.23},
    )
    return _apply(fig, 420)


def constructor_points(teams: pd.DataFrame) -> go.Figure:
    d = teams.sort_values("Points")
    fig = go.Figure(
        go.Bar(
            x=d["Points"],
            y=d["TeamName"],
            orientation="h",
            marker_color=d["TeamColor"],
            text=d["Points"],
            texttemplate="%{text:.0f}",
            textposition="outside",
            customdata=d[["Wins", "Podiums", "Reliability"]],
            hovertemplate="<b>%{y}</b><br>%{x:.0f} pts<br>Vitórias: %{customdata[0]}<br>Pódios: %{customdata[1]}<br>Confiabilidade: %{customdata[2]:.0%}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Pontos dos construtores", yaxis_title="")
    return _apply(fig, max(340, len(d) * 42), legend=False)


def teammate_duels(h2h: pd.DataFrame) -> go.Figure:
    d = h2h.sort_values("PointsA")
    limit = max(1, int(d[["RaceWinsA", "RaceWinsB"]].to_numpy().max()))
    ticks = list(range(-limit, limit + 1))
    fig = go.Figure()
    fig.add_bar(
        x=d["RaceWinsA"],
        y=d["TeamName"],
        orientation="h",
        marker_color="#2f78d1",
        text=[f"{a} {int(n)}" for a, n in zip(d["abbr_a"], d["RaceWinsA"])],
        textposition="inside",
        name="Piloto A",
    )
    fig.add_bar(
        x=-d["RaceWinsB"],
        y=d["TeamName"],
        orientation="h",
        marker_color="#e99b32",
        text=[f"{b} {int(n)}" for b, n in zip(d["abbr_b"], d["RaceWinsB"])],
        textposition="inside",
        name="Piloto B",
    )
    fig.update_layout(
        barmode="relative",
        xaxis_title="← B à frente · corridas em comum · A à frente →",
        xaxis={
            "range": [-limit - 0.5, limit + 0.5],
            "tickvals": ticks,
            "ticktext": [str(abs(value)) for value in ticks],
        },
        yaxis_title="",
        legend={"orientation": "h", "y": -0.22},
    )
    fig.add_vline(x=0, line_color=GRID, line_width=1)
    return _apply(fig, max(340, len(d) * 46))


def importance_bar(importances: dict[str, float], k: int = 15) -> go.Figure:
    ranked = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:k][
        ::-1
    ]
    fig = go.Figure(
        go.Bar(
            x=[v for _, v in ranked],
            y=[f for f, _ in ranked],
            orientation="h",
            marker_color=ACCENT,
            hovertemplate="%{y}<br>Importância: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Importância global do modelo", yaxis_title="")
    return _apply(fig, max(340, len(ranked) * 28), legend=False)


def contribution_bar(explanation: dict) -> go.Figure:
    rows = pd.DataFrame(explanation.get("contributions", []))
    if rows.empty:
        return _apply(go.Figure(), 320, legend=False)
    rows = rows.sort_values("contribution")
    fig = go.Figure(
        go.Bar(
            x=rows["contribution"],
            y=rows["feature"],
            orientation="h",
            marker_color=[
                POSITIVE if v >= 0 else NEGATIVE for v in rows["contribution"]
            ],
            customdata=rows[["value"]],
            hovertemplate="%{y}<br>Contribuição: %{x:+.4f}<br>Valor: %{customdata[0]}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color=NEUTRAL)
    fig.update_layout(
        xaxis_title="Contribuição SHAP para a classe campeão", yaxis_title=""
    )
    return _apply(fig, max(340, len(rows) * 30), legend=False)


def model_performance(evaluations: list[dict]) -> go.Figure:
    d = pd.DataFrame(evaluations)
    fig = go.Figure()
    if not d.empty and {"season", "top1_accuracy"}.issubset(d):
        fig.add_trace(
            go.Scatter(
                x=d["season"],
                y=d["top1_accuracy"],
                mode="lines+markers",
                name="Modelo",
                line={"color": ACCENT, "width": 3},
            )
        )
        if "baseline_accuracy" in d:
            fig.add_trace(
                go.Scatter(
                    x=d["season"],
                    y=d["baseline_accuracy"],
                    mode="lines+markers",
                    name="Líder atual",
                    line={"color": NEUTRAL, "dash": "dot"},
                )
            )
    fig.update_layout(
        yaxis={"title": "Acerto do campeão", "tickformat": ".0%", "range": [0, 1]},
        xaxis_title="Temporada de teste",
        legend={"orientation": "h", "y": -0.22},
    )
    return _apply(fig, 380)


def calibration_curve(points: list[dict]) -> go.Figure:
    d = pd.DataFrame(points)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Calibração ideal",
            line={"color": NEUTRAL, "dash": "dot"},
        )
    )
    if not d.empty and {"predicted", "observed"}.issubset(d):
        fig.add_trace(
            go.Scatter(
                x=d["predicted"],
                y=d["observed"],
                mode="lines+markers",
                name="Modelo",
                line={"color": ACCENT, "width": 3},
            )
        )
    fig.update_layout(
        xaxis={"title": "Probabilidade prevista", "tickformat": ".0%", "range": [0, 1]},
        yaxis={"title": "Frequência observada", "tickformat": ".0%", "range": [0, 1]},
        legend={"orientation": "h", "y": -0.22},
    )
    return _apply(fig, 380)
