"""Data access for the dashboard: Delta reads, the prediction API, and thin
``st.cache_data`` wrappers around the pure helpers in ``analytics``.

Every loader is cached for a day. The per-season wrappers take only an ``int``
year as their cache key and pull their DataFrames from the cached loaders, so a
widget change never re-reads Delta or re-hits the API.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st
from deltalake import DeltaTable

import analytics

URI_API = f"http://api-driver-champion:{os.environ['API_PORT']}"
TABLE_PATH_SILVER = os.environ["TABLE_PATH_SILVER"]
TABLE_PATH_BRONZE = os.environ["TABLE_PATH_BRONZE"]

# columns of tb_abt that are not model features
_NON_FEATURE = {"dt_ref", "DriverId", "Year", "id", "flChampion"}
_MODEL_FILL = -10000  # SimpleImputer(fill_value=-10000) used at training time

_DRIVER_INFO_COLS = [
    "DriverId",
    "TeamName",
    "TeamColor",
    "FullName",
    "TeamId",
    "HeadshotUrl",
    "CountryCode",
    "Abbreviation",
]


# ── API ─────────────────────────────────────────────────────────────────────


def predict(values: pd.DataFrame):
    """POST feature rows to ``/predict``; returns ``{id: {"0": p, "1": p}}``."""
    data = {"values": values.to_dict(orient="records")}
    resp = requests.post(f"{URI_API}/predict", json=data)
    return resp.json().get("predictions")


@st.cache_data(ttl="1d")
def model_info() -> dict:
    """``GET /model_info`` — feature importances etc.; ``{}`` if unavailable."""
    try:
        resp = requests.get(f"{URI_API}/model_info", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError):
        return {}


# ── Delta loaders ──────────────────────────────────────────────────────────


@st.cache_data(ttl="1d")
def load_bronze() -> pd.DataFrame:
    """Full race/sprint results from the Bronze layer."""
    df = DeltaTable(TABLE_PATH_BRONZE).to_pyarrow_table().to_pandas()
    df["TeamColor"] = df["TeamColor"].apply(analytics.format_color)
    return df


@st.cache_data(ttl="1d")
def races_frame() -> pd.DataFrame:
    """Race-mode Bronze rows with numeric position/grid/points + a DNF flag —
    the shared input for every season aggregate (coerced once)."""
    df = load_bronze()
    df = df[df["Mode"] == "Race"].copy()
    for col in ("Position", "GridPosition", "Points"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["DNF"] = df["ClassifiedPosition"] == "R"
    return df


@st.cache_data(ttl="1d")
def _abt_raw() -> pd.DataFrame:
    """The full Silver ABT (read once; ``load_predictions`` slices it by year)."""
    raw = DeltaTable(TABLE_PATH_SILVER).to_pyarrow_table().to_pandas()
    raw["dt_ref"] = raw["dt_ref"].astype("string")
    raw["Year"] = pd.to_datetime(raw["dt_ref"]).dt.year
    return raw


@st.cache_data(ttl="1d")
def load_predictions(year: int) -> pd.DataFrame:
    """One season of the Silver ABT with win probabilities from ``/predict``,
    enriched with driver metadata. Nulls are preserved for display; the API
    payload is a separate ``-10000``-filled copy."""
    season = _abt_raw()
    season = season[season["Year"] == year].copy()
    if season.empty:
        return season
    season["id"] = season["dt_ref"] + "_" + season["DriverId"]

    feature_cols = [c for c in season.columns if c not in _NON_FEATURE]
    payload = season[["id", *feature_cols]].fillna(_MODEL_FILL)
    try:
        preds = predict(payload) or {}
        prob = (
            pd.DataFrame(preds)
            .T.reset_index()
            .rename(columns={"index": "id", "1": "prob_win"})
        )
        season = season.merge(prob[["id", "prob_win"]], on="id", how="left")
    except (requests.RequestException, ValueError, KeyError):
        # API unreachable / model unregistered — surface a friendly notice
        # in main() rather than a traceback.
        season["prob_win"] = pd.NA
    season["prob_win"] = pd.to_numeric(season["prob_win"], errors="coerce")

    driver_info = load_bronze().drop_duplicates("DriverId")[_DRIVER_INFO_COLS]
    season = season.merge(driver_info, on="DriverId", how="left")
    season["driver_team_id"] = season["DriverId"] + "_" + season["TeamId"]
    return season


def available_seasons() -> list[int]:
    """Descending list of seasons present in the ABT."""
    years = _abt_raw()["Year"].dropna().astype(int).unique().tolist()
    return sorted(years, reverse=True)


# ── cached per-season aggregates ──────────────────────────────────────────


@st.cache_data(ttl="1d")
def driver_stats(year: int) -> pd.DataFrame:
    return analytics.compute_driver_stats(races_frame(), year)


@st.cache_data(ttl="1d")
def team_stats(year: int) -> pd.DataFrame:
    return analytics.compute_team_stats(races_frame(), year)


@st.cache_data(ttl="1d")
def reliability(year: int) -> pd.DataFrame:
    return analytics.compute_reliability(races_frame(), year)


@st.cache_data(ttl="1d")
def teammate_h2h(year: int) -> pd.DataFrame:
    return analytics.compute_teammate_h2h(races_frame(), year)


@st.cache_data(ttl="1d")
def momentum(year: int, last_n: int = 5) -> pd.DataFrame:
    return analytics.compute_momentum(load_predictions(year), last_n=last_n)
