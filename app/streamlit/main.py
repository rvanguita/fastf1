"""Lake FastF1 — produto analítico editorial em Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

import analytics
import charts
import data


@dataclass(frozen=True)
class Context:
    season: int
    window: int
    drivers: list[str]
    stats: pd.DataFrame
    teams: pd.DataFrame
    races: pd.DataFrame
    history: pd.DataFrame
    predictions: pd.DataFrame
    momentum: pd.DataFrame


def _css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1500px}
        h1, h2, h3 {letter-spacing: -.025em}
        [data-testid="stMetric"] {border-top: 2px solid rgba(128,128,128,.25); padding-top: .7rem}
        [data-testid="stMetricValue"] {font-weight: 720; letter-spacing: -.035em}
        .eyebrow {font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; opacity:.62; font-weight:700}
        .lede {font-size:1.08rem; opacity:.75; max-width:850px; margin-bottom:1.4rem}
        .insight {border-left:3px solid #ff4b44; padding:.45rem 0 .45rem 1rem; margin:.4rem 0}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="lede">{description}</div>', unsafe_allow_html=True)


@st.cache_data(ttl="15m")
def _context(season: int, window: int, drivers: tuple[str, ...]) -> Context:
    stats = data.driver_stats(season)
    teams = data.team_stats(season)
    races = data.results(season)
    history = data.standings_history(season)
    predictions = data.load_predictions(season)
    momentum = (
        analytics.compute_momentum(predictions, last_n=window)
        if not predictions.empty and "prob_win" in predictions
        else pd.DataFrame()
    )
    return Context(
        season,
        window,
        list(drivers),
        stats,
        teams,
        races,
        history,
        predictions,
        momentum,
    )


def _current_context() -> Context:
    return _context(
        int(st.session_state["season"]),
        int(st.session_state["window"]),
        tuple(st.session_state.get("drivers", [])),
    )


def _global_filters() -> None:
    seasons = data.available_seasons()
    with st.sidebar:
        st.divider()
        st.caption("CONTEXTO DA ANÁLISE")
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


def _probability_cards(momentum: pd.DataFrame) -> None:
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
    ctx = _current_context()
    _page_header(
        "TEMPORADA EM UMA PÁGINA",
        f"O campeonato de {ctx.season}, agora",
        "Classificação, disputa pelo título e sinais recentes reunidos sem repetir a mesma informação em formatos diferentes.",
    )
    if ctx.stats.empty:
        st.warning("Não há resultados para esta temporada.")
        return
    leader = ctx.stats.iloc[0]
    runner = ctx.stats.iloc[1] if len(ctx.stats) > 1 else leader
    latest_round = int(ctx.races["RoundNumber"].max()) if not ctx.races.empty else 0
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
    _probability_cards(ctx.momentum)
    if not ctx.momentum.empty:
        st.plotly_chart(charts.probability_ranking(ctx.momentum), width="stretch")
    st.divider()
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Classificação do campeonato")
        st.plotly_chart(charts.standings_bar(ctx.stats), width="stretch")
    with right:
        st.subheader("Leituras rápidas")
        insights = analytics.build_insights(
            ctx.momentum,
            data.teammate_h2h(ctx.season),
            data.reliability(ctx.season),
            ctx.window,
        )
        for insight in insights:
            st.markdown(f'<div class="insight">{insight}</div>', unsafe_allow_html=True)
        if not insights:
            st.caption(
                "Os insights serão exibidos após três rodadas e uma previsão válida."
            )


def page_championship() -> None:
    ctx = _current_context()
    _page_header(
        "EVOLUÇÃO DA TEMPORADA",
        "Como o campeonato chegou até aqui?",
        "Pontos, posições e resultados por rodada. Use os mesmos pilotos em destaque em todos os gráficos.",
    )
    if ctx.history.empty:
        st.warning("Não há histórico para esta temporada.")
        return
    selected = ctx.drivers or ctx.stats["DriverId"].head(5).tolist()
    tab_points, tab_rank = st.tabs(["Pontos acumulados", "Posição no campeonato"])
    with tab_points:
        st.plotly_chart(charts.points_history(ctx.history, selected), width="stretch")
    with tab_rank:
        st.plotly_chart(charts.rank_bump(ctx.history, selected), width="stretch")
    st.divider()
    st.subheader("Cada corrida, em contexto")
    st.caption(
        "Resultados oficiais; DNF, DNS, DSQ e NC são preservados como categorias, não convertidos em posições."
    )
    order = [
        abbr
        for abbr in ctx.stats["Abbreviation"]
        if abbr in set(ctx.races["Abbreviation"])
    ]
    st.plotly_chart(charts.result_heatmap(ctx.races, order), width="stretch")
    rounds = ctx.races.sort_values("RoundNumber").drop_duplicates("RoundNumber")[
        ["RoundNumber", "EventName"]
    ]
    round_map = dict(zip(rounds["RoundNumber"].astype(int), rounds["EventName"]))
    chosen = st.selectbox(
        "Analisar grid → chegada",
        list(round_map),
        format_func=lambda value: f"R{value} · {round_map[value]}",
        index=len(round_map) - 1,
    )
    st.plotly_chart(charts.grid_to_finish(ctx.races, chosen), width="stretch")


def page_compare() -> None:
    ctx = _current_context()
    _page_header(
        "COMPARADOR",
        "Onde a vantagem realmente aparece?",
        "Compare pilotos na mesma escala, investigue duelos internos e veja a contribuição dos construtores.",
    )
    if len(ctx.stats) < 2:
        st.warning("São necessários dois pilotos para comparar.")
        return
    ids = ctx.stats["DriverId"].tolist()
    labels = ctx.stats.set_index("DriverId")["FullName"].to_dict()
    defaults = (ctx.drivers + ids)[:2]
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
    st.plotly_chart(charts.driver_dumbbell(ctx.stats, [a, b]), width="stretch")
    st.caption(
        "A escala relativa é calculada dentro do grid da temporada; para média de grid, chegada e DNF, menor é melhor."
    )
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Força dos construtores")
        st.plotly_chart(charts.constructor_points(ctx.teams), width="stretch")
    with c2:
        st.subheader("Duelos entre companheiros")
        h2h = data.teammate_h2h(ctx.season)
        if h2h.empty:
            st.info("Não há pares completos nesta temporada.")
        else:
            st.plotly_chart(charts.teammate_duels(h2h), width="stretch")


def page_model_data() -> None:
    ctx = _current_context()
    _page_header(
        "TRANSPARÊNCIA",
        "O que sustenta a previsão?",
        "Desempenho fora do tempo, explicações individuais, limitações e saúde dos dados no mesmo lugar.",
    )
    card = data.model_card()
    info = data.model_info()
    status = card.get("status", "indisponível" if not info else "experimental")
    st.info(
        f"Status do modelo: **{status}** · previsões são estimativas, não garantias."
    )
    if not ctx.predictions.empty and ctx.predictions["prob_win"].notna().any():
        selected = ctx.drivers or ctx.stats["DriverId"].head(5).tolist()
        st.subheader("Evolução das probabilidades")
        st.plotly_chart(
            charts.probability_history(ctx.predictions, selected), width="stretch"
        )
    else:
        st.warning("A API não retornou previsões válidas para esta temporada.")
    st.divider()
    m1, m2 = st.columns(2)
    with m1:
        st.subheader("Validação temporal")
        evaluations = card.get("evaluations", [])
        if evaluations:
            st.plotly_chart(charts.model_performance(evaluations), width="stretch")
        else:
            st.caption("O modelo registrado ainda não contém backtests temporais.")
    with m2:
        st.subheader("Calibração")
        calibration = card.get("calibration", [])
        if calibration:
            st.plotly_chart(charts.calibration_curve(calibration), width="stretch")
        else:
            st.caption("O modelo registrado ainda não contém uma curva de calibração.")
    importances = info.get("importances", {})
    if importances:
        st.subheader("O que o modelo usa globalmente")
        st.caption(
            "Importância global não indica a direção do efeito para um piloto específico."
        )
        st.plotly_chart(charts.importance_bar(importances), width="stretch")
    if not ctx.predictions.empty and info.get("features"):
        latest = ctx.predictions[
            ctx.predictions["dt_ref"] == ctx.predictions["dt_ref"].max()
        ]
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
                st.plotly_chart(
                    charts.contribution_bar(explanations[row["id"]]), width="stretch"
                )
            else:
                st.caption("Explicação SHAP indisponível para esta versão do modelo.")
    st.divider()
    st.subheader("Saúde e acesso aos dados")
    health = data.data_health(ctx.season)
    h1, h2, h3, h4 = st.columns(4)
    h1.metric(
        "Último resultado",
        health["latest_result"].strftime("%d/%m/%Y")
        if health["latest_result"] is not None
        else "—",
    )
    h2.metric("Rodadas", health["rounds"])
    h3.metric("Snapshots preditivos", health["prediction_snapshots"])
    h4.metric("Chaves duplicadas", health["duplicate_results"])
    with st.expander("Dados analíticos e metodologia"):
        st.dataframe(ctx.stats, hide_index=True, width="stretch")
        st.download_button(
            "Baixar classificação em CSV",
            ctx.stats.to_csv(index=False).encode(),
            file_name=f"classificacao_f1_{ctx.season}.csv",
            mime="text/csv",
        )
        st.markdown(
            "Posições médias excluem DNF/DNS/DSQ; grid zero (pit lane) não é tratado como posição. Probabilidades são normalizadas por snapshot."
        )


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
    _global_filters()
    page.run()


if __name__ == "__main__":
    main()
