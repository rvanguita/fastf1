"""Lake FastF1 — produto analítico editorial em Streamlit."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

import pandas as pd
import streamlit as st

import analytics
import charts
import data

_PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


@dataclass(frozen=True)
class Selection:
    season: int
    window: int
    drivers: list[str]


def _css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1500px}
        h1, h2, h3 {letter-spacing: -.025em}
        [data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,.18);
            border-top: 3px solid #ff4b44;
            border-radius: .65rem;
            padding: .75rem .9rem;
            background: rgba(128,128,128,.035);
        }
        [data-testid="stMetricValue"] {font-weight: 720; letter-spacing: -.035em}
        .eyebrow {font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; opacity:.62; font-weight:700}
        .lede {font-size:1.08rem; opacity:.75; max-width:850px; margin-bottom:1.4rem}
        [data-testid="stExpander"] {border-color: rgba(128,128,128,.22)}
        @media (max-width: 800px) {
            .block-container {padding-top: 1.4rem}
            .lede {font-size: 1rem}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(f'<div class="eyebrow">{escape(kicker)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(
        f'<div class="lede">{escape(description)}</div>', unsafe_allow_html=True
    )


def _plot(figure) -> None:
    st.plotly_chart(
        figure,
        width="stretch",
        config=_PLOTLY_CONFIG,
    )


def _context_caption(
    races: pd.DataFrame, predictions: pd.DataFrame | None = None
) -> None:
    items = []
    if not races.empty:
        latest = races.sort_values("RoundNumber").iloc[-1]
        items.append(
            f"resultados até R{int(latest['RoundNumber'])} · {latest['EventName']}"
        )
    if (
        predictions is not None
        and not predictions.empty
        and predictions["dt_ref"].notna().any()
    ):
        snapshot = pd.Timestamp(predictions["dt_ref"].max()).strftime("%d/%m/%Y")
        items.append(f"previsão de {snapshot}")
    if items:
        st.caption("Atualização: " + " · ".join(items))


def _current_selection() -> Selection:
    return Selection(
        int(st.session_state["season"]),
        int(st.session_state["window"]),
        list(st.session_state.get("drivers", [])),
    )


def _global_filters() -> bool:
    seasons = data.available_seasons()
    with st.sidebar:
        st.divider()
        st.caption("CONTEXTO DA ANÁLISE")
        if not seasons:
            st.error("Nenhuma temporada foi encontrada nas tabelas Delta.")
            return False
        season = st.selectbox("Temporada", seasons, key="season")
        stats = data.driver_stats(season)
        ids = stats["DriverId"].tolist() if "DriverId" in stats else []
        labels = (
            stats.set_index("DriverId")
            .apply(
                lambda row: analytics.driver_label(row["FullName"], row["TeamName"]),
                axis=1,
            )
            .to_dict()
            if ids
            else {}
        )
        current = [
            driver
            for driver in st.session_state.get("drivers", ids[:5])
            if driver in ids
        ]
        st.multiselect(
            "Pilotos em destaque",
            ids,
            default=current or ids[:5],
            format_func=lambda value: labels.get(value, value),
            key="drivers",
        )
        st.slider(
            "Janela de tendência",
            3,
            10,
            5,
            key="window",
            help="Número de rodadas usado para medir a tendência recente.",
        )
        st.caption("Resultados FastF1 · atualização semanal")
    return True


def _probability_cards(momentum: pd.DataFrame) -> None:
    if momentum.empty or "latest" not in momentum:
        st.info(
            "A API preditiva está indisponível. A análise descritiva continua ativa."
        )
        return
    top = momentum.dropna(subset=["latest"]).head(4)
    if top.empty:
        st.info(
            "A API preditiva está indisponível. A análise descritiva continua ativa."
        )
        return
    cols = st.columns(4)
    for col, (_, row) in zip(cols, top.iterrows()):
        delta = row.get("delta_prev")
        col.metric(
            row["FullName"],
            f"{row['latest']:.1%}",
            None if pd.isna(delta) else f"{delta * 100:+.1f} pp",
            help=f"{row['TeamName']} · probabilidade normalizada dentro do grid",
        )


def page_overview() -> None:
    selection = _current_selection()
    stats = data.driver_stats(selection.season)
    races = data.results(selection.season)
    predictions = data.load_predictions(selection.season)
    momentum = (
        analytics.compute_momentum(predictions, last_n=selection.window)
        if not predictions.empty and "prob_win" in predictions
        else pd.DataFrame()
    )
    _page_header(
        "TEMPORADA EM UMA PÁGINA",
        f"O campeonato de {selection.season}, agora",
        "Classificação, disputa pelo título e sinais recentes reunidos sem repetir a mesma informação em formatos diferentes.",
    )
    _context_caption(races, predictions)
    if stats.empty:
        st.warning("Não há resultados para esta temporada.")
        return
    leader = stats.iloc[0]
    runner = stats.iloc[1] if len(stats) > 1 else leader
    latest_round = int(races["RoundNumber"].max()) if not races.empty else 0
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Líder", leader["FullName"], f"{leader['Points']:.0f} pts")
    k2.metric(
        "Vantagem",
        f"{leader['Points'] - runner['Points']:.0f} pts",
        f"sobre {analytics.short_name(runner['FullName'])}",
        delta_color="off",
    )
    k3.metric("Rodadas concluídas", latest_round)
    k4.metric(
        "Vitórias do líder",
        int(leader["Wins"]),
        f"{leader['Podiums']:.0f} pódios",
        delta_color="off",
    )
    st.divider()
    st.subheader("Quem controla a disputa pelo título?")
    st.caption(
        "Probabilidades normalizadas por data; os competidores do mesmo snapshot somam 100%."
    )
    _probability_cards(momentum)
    if not momentum.empty:
        _plot(charts.probability_ranking(momentum))
    st.divider()
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Classificação do campeonato")
        _plot(charts.standings_bar(stats))
    with right:
        st.subheader("Leituras rápidas")
        insights = analytics.build_insights(
            momentum,
            data.teammate_h2h(selection.season),
            data.reliability(selection.season),
            selection.window,
        )
        for insight in insights:
            with st.container(border=True):
                st.markdown(insight)
        if not insights:
            st.caption(
                "Os insights serão exibidos após três rodadas e uma previsão válida."
            )


def page_championship() -> None:
    selection = _current_selection()
    stats = data.driver_stats(selection.season)
    races = data.results(selection.season)
    history = data.standings_history(selection.season)
    _page_header(
        "EVOLUÇÃO DA TEMPORADA",
        "Como o campeonato chegou até aqui?",
        "Pontos, posições e resultados por rodada. Use os mesmos pilotos em destaque em todos os gráficos.",
    )
    _context_caption(races)
    if history.empty:
        st.warning("Não há histórico para esta temporada.")
        return
    selected = selection.drivers or stats["DriverId"].head(5).tolist()
    tab_points, tab_rank = st.tabs(["Pontos acumulados", "Posição no campeonato"])
    with tab_points:
        _plot(charts.points_history(history, selected))
    with tab_rank:
        _plot(charts.rank_bump(history, selected))
    st.divider()
    st.subheader("Cada corrida, em contexto")
    st.caption(
        "Resultados oficiais; DNF, DNS, DSQ e NC são preservados como categorias, não convertidos em posições."
    )
    order = [
        abbr for abbr in stats["Abbreviation"] if abbr in set(races["Abbreviation"])
    ]
    _plot(charts.result_heatmap(races, order))
    rounds = races.sort_values("RoundNumber").drop_duplicates("RoundNumber")[
        ["RoundNumber", "EventName"]
    ]
    if rounds.empty:
        st.info("Ainda não há corridas para detalhar nesta temporada.")
        return
    round_map = dict(zip(rounds["RoundNumber"].astype(int), rounds["EventName"]))
    chosen = st.selectbox(
        "Analisar grid → chegada",
        list(round_map),
        format_func=lambda value: f"R{value} · {round_map[value]}",
        index=len(round_map) - 1,
    )
    summary = analytics.summarize_race(races, chosen)
    st.subheader(f"R{chosen} · {summary.get('event_name', round_map[chosen])}")
    s1, s2, s3 = st.columns(3)
    s1.metric("Vencedor", summary.get("winner", "—"))
    s2.metric("Maior avanço", summary.get("biggest_gainer", "—"))
    s3.metric("Não classificados", summary.get("incidents", 0))
    st.caption(f"Pódio: {summary.get('podium', '—')}")
    _plot(charts.grid_to_finish(races, chosen))


def page_compare() -> None:
    selection = _current_selection()
    stats = data.driver_stats(selection.season)
    teams = data.team_stats(selection.season)
    races = data.results(selection.season)
    _page_header(
        "COMPARADOR",
        "Onde a vantagem realmente aparece?",
        "Compare pilotos na mesma escala, investigue duelos internos e veja a contribuição dos construtores.",
    )
    _context_caption(races)
    if len(stats) < 2:
        st.warning("São necessários dois pilotos para comparar.")
        return
    ids = stats["DriverId"].tolist()
    labels = stats.set_index("DriverId")["FullName"].to_dict()
    defaults = (selection.drivers + ids)[:2]
    left, right = st.columns(2)
    a = left.selectbox(
        "Piloto A",
        ids,
        index=ids.index(defaults[0]),
        format_func=lambda value: labels[value],
    )
    b_options = [value for value in ids if value != a]
    b_default = defaults[1] if defaults[1] in b_options else b_options[0]
    b = right.selectbox(
        "Piloto B",
        b_options,
        index=b_options.index(b_default),
        format_func=lambda value: labels[value],
    )
    _plot(charts.driver_dumbbell(stats, [a, b]))
    st.caption(
        "A escala relativa é calculada dentro do grid da temporada; para média de grid, chegada e DNF, menor é melhor."
    )
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Força dos construtores")
        _plot(charts.constructor_points(teams))
    with c2:
        st.subheader("Duelos entre companheiros")
        h2h = data.teammate_h2h(selection.season)
        if h2h.empty:
            st.info("Não há pares completos nesta temporada.")
        else:
            _plot(charts.teammate_duels(h2h))


def page_model_data() -> None:
    selection = _current_selection()
    stats = data.driver_stats(selection.season)
    races = data.results(selection.season)
    predictions = data.load_predictions(selection.season)
    _page_header(
        "TRANSPARÊNCIA",
        "O que sustenta a previsão?",
        "Desempenho fora do tempo, explicações individuais, limitações e saúde dos dados no mesmo lugar.",
    )
    _context_caption(races, predictions)
    card = data.model_card()
    info = data.model_info()
    evaluations = card.get("evaluations", [])
    status = card.get("status", "indisponível" if not info else "experimental")
    st.info(
        f"Status do modelo: **{status}** · previsões são estimativas, não garantias."
    )
    if card:
        trained = card.get("trained_through", "—")
        c1, c2, c3 = st.columns(3)
        c1.metric("Treinado até", trained)
        c2.metric("Temporadas em backtest", len(evaluations))
        c3.metric("Estratégia", "Rolling origin")
    if not predictions.empty and predictions["prob_win"].notna().any():
        selected = selection.drivers or stats["DriverId"].head(5).tolist()
        st.subheader("Evolução das probabilidades")
        _plot(charts.probability_history(predictions, selected))
    else:
        st.warning("A API não retornou previsões válidas para esta temporada.")
    st.divider()
    m1, m2 = st.columns(2)
    with m1:
        st.subheader("Validação temporal")
        if evaluations:
            _plot(charts.model_performance(evaluations))
        else:
            st.caption("O modelo registrado ainda não contém backtests temporais.")
    with m2:
        st.subheader("Calibração")
        calibration = card.get("calibration", [])
        if calibration:
            _plot(charts.calibration_curve(calibration))
        else:
            st.caption("O modelo registrado ainda não contém uma curva de calibração.")
    importances = info.get("importances", {})
    if importances:
        st.subheader("O que o modelo usa globalmente")
        st.caption(
            "Importância global não indica a direção do efeito para um piloto específico."
        )
        _plot(charts.importance_bar(importances))
    if not predictions.empty and info.get("features"):
        latest = predictions[predictions["dt_ref"] == predictions["dt_ref"].max()]
        candidates = latest["DriverId"].tolist()
        if candidates:
            label_map = latest.set_index("DriverId")["FullName"].to_dict()
            chosen = st.selectbox(
                "Explicação individual",
                candidates,
                format_func=lambda value: label_map[value],
            )
            row = latest[latest["DriverId"] == chosen].iloc[0]
            payload_cols = [
                "id",
                *[column for column in info["features"] if column in latest],
            ]
            explanations = data.explain(pd.DataFrame([row[payload_cols].to_dict()]))
            if row["id"] in explanations:
                _plot(charts.contribution_bar(explanations[row["id"]]))
            else:
                st.caption("Explicação SHAP indisponível para esta versão do modelo.")
    st.divider()
    st.subheader("Saúde e acesso aos dados")
    health = data.data_health(selection.season)
    h1, h2, h3, h4 = st.columns(4)
    h1.metric(
        "Último resultado",
        health["latest_result"].strftime("%d/%m/%Y")
        if pd.notna(health["latest_result"])
        else "—",
    )
    h2.metric("Rodadas", health["rounds"])
    h3.metric("Snapshots preditivos", health["prediction_snapshots"])
    h4.metric("Chaves duplicadas", health["duplicate_results"])
    if health["duplicate_results"]:
        st.error(
            f"Foram encontradas {health['duplicate_results']} chaves duplicadas; revise a camada Bronze."
        )
    else:
        st.success("Integridade das chaves verificada: nenhuma duplicidade encontrada.")
    with st.expander("Dados analíticos e metodologia"):
        columns = {
            "Rank": "Pos.",
            "FullName": "Piloto",
            "TeamName": "Equipe",
            "Points": "Pontos",
            "Wins": "Vitórias",
            "Podiums": "Pódios",
            "AvgGrid": "Grid médio",
            "AvgFinish": "Chegada média",
            "DNFRate": "Taxa DNF",
        }
        standings = stats[[column for column in columns if column in stats]].rename(
            columns=columns
        )
        st.dataframe(
            standings,
            hide_index=True,
            width="stretch",
            column_config={
                "Pontos": st.column_config.NumberColumn(format="%.0f"),
                "Grid médio": st.column_config.NumberColumn(format="P%.1f"),
                "Chegada média": st.column_config.NumberColumn(format="P%.1f"),
                "Taxa DNF": st.column_config.NumberColumn(format="percent"),
            },
        )
        st.download_button(
            "Baixar classificação em CSV",
            stats.to_csv(index=False).encode(),
            file_name=f"classificacao_f1_{selection.season}.csv",
            mime="text/csv",
        )
        st.markdown(
            "Posições médias excluem DNF/DNS/DSQ; grid zero (pit lane) não é tratado como posição. Probabilidades são normalizadas por snapshot."
        )
        limitations = card.get("limitations", [])
        if limitations:
            st.markdown("**Limitações do modelo**")
            for limitation in limitations:
                st.markdown(f"- {limitation}")


def main() -> None:
    st.set_page_config(
        page_title="Lake FastF1",
        page_icon="🏁",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _css()
    page = st.navigation(
        {
            "Lake FastF1": [
                st.Page(page_overview, title="Visão geral", icon="🏁", default=True),
                st.Page(page_championship, title="Campeonato", icon="📈"),
                st.Page(page_compare, title="Comparador", icon="⚖️"),
                st.Page(page_model_data, title="Modelo & dados", icon="🔬"),
            ]
        }
    )
    if not _global_filters():
        st.error(
            "O dashboard precisa de ao menos uma temporada na camada Bronze. "
            "Execute a ingestão e a consolidação antes de abrir a aplicação."
        )
        return
    page.run()


if __name__ == "__main__":
    main()
