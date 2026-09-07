"""Acesso a Delta e à API preditiva, com degradação segura e cache."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from deltalake import DeltaTable
from deltalake.exceptions import DeltaError

import analytics

URI_API = (
    os.getenv("API_URL")
    or f"http://api-driver-champion:{os.getenv('API_PORT', '5002')}"
).rstrip("/")
TABLE_PATH_SILVER = os.getenv("TABLE_PATH_SILVER", "/data/silver/tb_abt")
TABLE_PATH_BRONZE = os.getenv("TABLE_PATH_BRONZE", "/data/bronze/results")
_SILVER_ROOT = Path(TABLE_PATH_SILVER).parent
TABLE_PATH_MART_DRIVER_ROUND = os.environ.get(
    "TABLE_PATH_MART_DRIVER_ROUND", str(_SILVER_ROOT / "mart_driver_round")
)
TABLE_PATH_MART_STANDINGS = os.environ.get(
    "TABLE_PATH_MART_STANDINGS", str(_SILVER_ROOT / "mart_standings")
)
_NON_FEATURE = {"dt_ref", "DriverId", "Year", "id", "flChampion"}
_MODEL_FILL = -10000
_BRONZE_ANALYTICS_COLUMNS = (
    "Year",
    "Date",
    "Mode",
    "RoundNumber",
    "EventName",
    "DriverId",
    "FullName",
    "Abbreviation",
    "CountryCode",
    "HeadshotUrl",
    "TeamId",
    "TeamName",
    "TeamColor",
    "Position",
    "ClassifiedPosition",
    "GridPosition",
    "Points",
    "Status",
)
DeltaFilter = tuple[str, str, Any]


def predict_legacy(values: pd.DataFrame) -> dict:
    response = requests.post(
        f"{URI_API}/predict",
        json={"values": values.to_dict(orient="records")},
        timeout=60,
    )
    response.raise_for_status()
    return response.json().get("predictions", {})


def predict(values: pd.DataFrame) -> dict:
    """Compatibilidade pública com o helper anterior."""
    return predict_legacy(values)


def predict_v1(
    values: pd.DataFrame, *, include_intervals: bool = False
) -> tuple[dict, dict]:
    response = requests.post(
        f"{URI_API}/v1/predict",
        json={
            "values": values.to_dict(orient="records"),
            "include_intervals": include_intervals,
        },
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    return body.get("predictions", {}), body.get("metadata", {})


@st.cache_data(ttl="15m")
def model_info() -> dict:
    try:
        response = requests.get(f"{URI_API}/model_info", timeout=10)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return {}


@st.cache_data(ttl="15m")
def model_card() -> dict:
    try:
        response = requests.get(f"{URI_API}/v1/model-card", timeout=10)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return {}


def explain(values: pd.DataFrame, top_n: int = 10) -> dict:
    try:
        response = requests.post(
            f"{URI_API}/v1/explain",
            json={"values": values.to_dict(orient="records"), "top_n": top_n},
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("explanations", {})
    except (requests.RequestException, ValueError):
        return {}


def _delta_version(path: str, *, optional: bool = False) -> int | None:
    try:
        return DeltaTable(path).version()
    except (DeltaError, OSError):
        if optional:
            return None
        raise


@st.cache_data(ttl="1d")
def _read_delta(
    path: str,
    version: int,
    columns: tuple[str, ...] | None = None,
    filters: tuple[DeltaFilter, ...] = (),
) -> pd.DataFrame:
    """Lê uma versão Delta com projeção/predicados incluídos na chave de cache."""
    table = DeltaTable(path, version=version).to_pyarrow_table(
        columns=list(columns) if columns else None,
        filters=list(filters) if filters else None,
    )
    return table.to_pandas()


def load_bronze(
    year: int | None = None, columns: tuple[str, ...] | None = None
) -> pd.DataFrame:
    filters: tuple[DeltaFilter, ...] = (("Year", "=", year),) if year else ()
    version = _delta_version(TABLE_PATH_BRONZE)
    assert version is not None
    df = _read_delta(TABLE_PATH_BRONZE, version, columns, filters)
    if "TeamColor" in df:
        df["TeamColor"] = df["TeamColor"].apply(analytics.format_color)
    if "Date" in df:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def races_frame(year: int | None = None) -> pd.DataFrame:
    """Todas as sessões pontuáveis; a camada semântica separa Race e Sprint."""
    return load_bronze(year, columns=_BRONZE_ANALYTICS_COLUMNS).copy()


def _abt_raw(year: int | None = None) -> pd.DataFrame:
    version = _delta_version(TABLE_PATH_SILVER, optional=True)
    if version is None:
        return pd.DataFrame(columns=["dt_ref", "Year"])
    filters: tuple[DeltaFilter, ...] = ()
    if year is not None:
        filters = (
            ("dt_ref", ">=", date(year, 1, 1)),
            ("dt_ref", "<", date(year + 1, 1, 1)),
        )
    raw = _read_delta(TABLE_PATH_SILVER, version, filters=filters)
    raw["dt_ref"] = pd.to_datetime(raw["dt_ref"], errors="coerce")
    raw["Year"] = raw["dt_ref"].dt.year
    return raw


def _optional_mart(path: str, year: int) -> pd.DataFrame:
    """Lê um mart novo; instalações ainda não migradas usam o fallback Bronze."""
    version = _delta_version(path, optional=True)
    if version is None:
        return pd.DataFrame()
    return _read_delta(path, version, filters=(("season", "=", year),))


def _driver_metadata(year: int) -> pd.DataFrame:
    bronze = races_frame(year)
    season = bronze[bronze["Mode"] == "Race"]
    cols = [
        "DriverId",
        "TeamName",
        "TeamColor",
        "FullName",
        "TeamId",
        "HeadshotUrl",
        "CountryCode",
        "Abbreviation",
    ]
    latest = season.sort_values(["RoundNumber", "Date"]).drop_duplicates(
        "DriverId", keep="last"
    )[cols]
    first_race = (
        season.groupby("DriverId", as_index=False)["Date"]
        .min()
        .rename(columns={"Date": "FirstRaceDate"})
    )
    return latest.merge(first_race, on="DriverId", how="left")


def load_predictions(year: int) -> pd.DataFrame:
    abt_version = _delta_version(TABLE_PATH_SILVER, optional=True)
    bronze_version = _delta_version(TABLE_PATH_BRONZE)
    assert bronze_version is not None
    return _load_predictions(year, abt_version, bronze_version)


@st.cache_data(ttl="15m")
def _load_predictions(
    year: int, abt_version: int | None, bronze_version: int
) -> pd.DataFrame:
    del abt_version, bronze_version  # invalidam a cache após atualização Delta
    season = _abt_raw(year).copy()
    if season.empty:
        return season
    feature_cols = [column for column in season if column not in _NON_FEATURE]
    # A ABT traz pilotos vistos em temporadas anteriores. O universo de cada
    # snapshot contém apenas quem já estreou na temporada selecionada.
    season = season.merge(_driver_metadata(year), on="DriverId", how="inner")
    season = season[season["dt_ref"] >= season["FirstRaceDate"]].copy()
    season["id"] = season["dt_ref"].dt.strftime("%Y-%m-%d") + "_" + season["DriverId"]
    season["prediction_group"] = season["dt_ref"].dt.strftime("%Y-%m-%d")
    payload = season[["id", "prediction_group", *feature_cols]].fillna(_MODEL_FILL)
    metadata: dict = {}
    try:
        predictions, metadata = predict_v1(payload, include_intervals=False)
        mapped = pd.DataFrame.from_dict(predictions, orient="index")
        mapped.index.name = "id"
        mapped = mapped.reset_index().rename(columns={"probability": "prob_win"})
        season = season.merge(mapped, on="id", how="left")
    except (requests.RequestException, ValueError, KeyError):
        try:
            predictions = predict_legacy(payload.drop(columns="prediction_group"))
            mapped = (
                pd.DataFrame.from_dict(predictions, orient="index")
                .reset_index()
                .rename(columns={"index": "id", "1": "raw_score"})
            )
            season = season.merge(mapped[["id", "raw_score"]], on="id", how="left")
            season = analytics.normalize_probabilities(season)
        except (requests.RequestException, ValueError, KeyError):
            season["prob_win"] = pd.NA
            season["raw_score"] = pd.NA
    for column in ("prob_win", "raw_score", "lower", "upper"):
        if column in season:
            season[column] = pd.to_numeric(season[column], errors="coerce")
    season = season.drop(columns="FirstRaceDate")
    season.attrs["prediction_metadata"] = metadata
    return season


def available_seasons() -> list[int]:
    bronze = load_bronze(columns=("Year",))
    years = pd.to_numeric(bronze["Year"], errors="coerce").dropna().astype(int)
    return sorted(years.unique().tolist(), reverse=True)


def driver_stats(year: int) -> pd.DataFrame:
    return analytics.compute_driver_stats(races_frame(year), year)


def team_stats(year: int) -> pd.DataFrame:
    return analytics.compute_team_stats(races_frame(year), year)


def reliability(year: int) -> pd.DataFrame:
    return analytics.compute_reliability(races_frame(year), year)


def teammate_h2h(year: int) -> pd.DataFrame:
    return analytics.compute_teammate_h2h(races_frame(year), year)


def standings_history(year: int) -> pd.DataFrame:
    mart = _optional_mart(TABLE_PATH_MART_STANDINGS, year)
    if not mart.empty:
        season = mart.rename(
            columns={
                "driver_id": "DriverId",
                "driver_name": "FullName",
                "abbreviation": "Abbreviation",
                "team_name": "TeamName",
                "team_color": "TeamColor",
                "round_number": "RoundNumber",
                "event_name": "EventName",
                "event_date": "Date",
                "round_points": "Points",
                "cumulative_points": "CumulativePoints",
                "championship_rank": "ChampionshipRank",
            }
        )
        season["TeamColor"] = season["TeamColor"].map(analytics.format_color)
        return season
    return analytics.compute_standings_history(races_frame(year), year)


def results(year: int) -> pd.DataFrame:
    mart = _optional_mart(TABLE_PATH_MART_DRIVER_ROUND, year)
    if not mart.empty:
        season = mart.rename(
            columns={
                "driver_id": "DriverId",
                "driver_name": "FullName",
                "abbreviation": "Abbreviation",
                "team_name": "TeamName",
                "team_color": "TeamColor",
                "round_number": "RoundNumber",
                "event_name": "EventName",
                "event_date": "Date",
                "points": "Points",
                "result_order": "Position",
                "official_grid": "OfficialGrid",
                "official_finish": "OfficialFinish",
                "result_status": "ResultStatus",
            }
        )
        season["TeamColor"] = season["TeamColor"].map(analytics.format_color)
        season["Gain"] = season["OfficialGrid"] - season["OfficialFinish"]
        season["ResultLabel"] = season["OfficialFinish"].map(
            lambda value: f"P{int(value)}" if pd.notna(value) else ""
        )
        season.loc[season["ResultStatus"] != "FINISHED", "ResultLabel"] = season[
            "ResultStatus"
        ]
        return season.sort_values(["RoundNumber", "Position"])
    return analytics.result_matrix(races_frame(year), year)


def momentum(year: int, last_n: int = 5) -> pd.DataFrame:
    return analytics.compute_momentum(load_predictions(year), last_n=last_n)


def data_health(year: int) -> dict:
    races = analytics._season_races(races_frame(year), year)
    season_abt = _abt_raw(year)
    return {
        "latest_result": races["Date"].max() if not races.empty else None,
        "rounds": int(races["RoundNumber"].nunique()) if not races.empty else 0,
        "race_rows": len(races),
        "prediction_snapshots": int(season_abt["dt_ref"].nunique()),
        "duplicate_results": int(
            races.duplicated(["Year", "RoundNumber", "DriverId"]).sum()
        )
        if not races.empty
        else 0,
    }
