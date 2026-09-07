"""Unit tests for the pure helpers in ``analytics`` (and ``data.predict``).

The ``render_*`` / tab functions need a live Streamlit runtime and are out of
scope; only the pandas-only logic is exercised here.
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import analytics
import data

# ── format_color / short_name / driver_label ───────────────────────────────


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "#737b8c"),
        ("#AABBCC", "#aabbcc"),
        ("1A2B3C", "#1a2b3c"),
        ("#already", "#already"),
        ("nan", "#737b8c"),
        (float("nan"), "#737b8c"),
        ("", "#737b8c"),
    ],
)
def test_format_color(value, expected):
    assert analytics.format_color(value) == expected


def test_short_name():
    assert analytics.short_name("Max Verstappen") == "M. Verstappen"
    assert analytics.short_name("Nurse") == "Nurse"


def test_driver_label():
    assert (
        analytics.driver_label("Max Verstappen", "Red Bull")
        == "M. Verstappen — Red Bull"
    )
    assert analytics.driver_label("Max Verstappen", "") == "M. Verstappen"


# ── _rank_by / _color_map ──────────────────────────────────────────────────


def test_rank_by_sorts_desc_and_prepends_rank():
    df = pd.DataFrame({"FullName": ["a", "b", "c"], "Points": [10, 30, 20]})
    out = analytics._rank_by(df)

    assert next(iter(out.columns)) == "Rank"
    assert out["Points"].tolist() == [30, 20, 10]
    assert out["Rank"].tolist() == [1, 2, 3]


def test_rank_by_custom_column():
    df = pd.DataFrame({"x": [1, 2], "Wins": [5, 2]})
    out = analytics._rank_by(df, col="Wins")
    assert out["Wins"].tolist() == [5, 2]
    assert out["Rank"].tolist() == [1, 2]


@pytest.fixture
def team_color_df():
    return pd.DataFrame(
        {"TeamName": ["RB", "RB", "McLaren"], "TeamColor": ["#aaa", "#bbb", "#ccc"]}
    )


def test_color_map_keep_last(team_color_df):
    assert analytics._color_map(team_color_df, "TeamName") == {
        "RB": "#bbb",
        "McLaren": "#ccc",
    }


def test_color_map_keep_first(team_color_df):
    assert analytics._color_map(team_color_df, "TeamName", keep="first")["RB"] == "#aaa"


# ── data.predict ───────────────────────────────────────────────────────────


def test_predict_calls_api_and_unwraps(monkeypatch):
    resp = Mock()
    resp.json.return_value = {"predictions": {"x": {"1": 0.9}}}
    post = Mock(return_value=resp)
    monkeypatch.setattr(data.requests, "post", post)

    out = data.predict(pd.DataFrame([{"id": "x", "f": 1}]))

    assert out == {"x": {"1": 0.9}}
    post.assert_called_once_with(
        f"{data.URI_API}/predict",
        json={"values": [{"id": "x", "f": 1}]},
        timeout=60,
    )


# ── season aggregates ──────────────────────────────────────────────────────


@pytest.fixture
def bronze():
    """Two drivers over 2024 R1-R3; a Sprint row and a 2023 row must be ignored.
    Max: P1/P1/P2 from grids 1/2/1. Lando: P2/P3/DNF from grids 3/3/4."""
    return pd.DataFrame(
        {
            "Mode": ["Race"] * 6 + ["Sprint", "Race"],
            "Year": [2024, 2024, 2024, 2024, 2024, 2024, 2024, 2023],
            "RoundNumber": [1, 2, 3, 1, 2, 3, 1, 1],
            "Position": [1.0, 1.0, 2.0, 2.0, 3.0, np.nan, 1.0, 1.0],
            "GridPosition": [1.0, 2.0, 1.0, 3.0, 3.0, 4.0, 1.0, 1.0],
            "Points": [25.0, 25.0, 18.0, 18.0, 15.0, 0.0, 8.0, 25.0],
            "FullName": ["Max V"] * 3 + ["Lando N"] * 3 + ["Max V", "Max V"],
            "TeamName": ["RB"] * 3 + ["McLaren"] * 3 + ["RB", "RB"],
            "TeamColor": ["#3671C6"] * 3 + ["#FF8000"] * 3 + ["#3671C6", "#3671C6"],
            "Abbreviation": ["VER"] * 3 + ["NOR"] * 3 + ["VER", "VER"],
            "ClassifiedPosition": ["1", "1", "2", "2", "3", "R", "1", "1"],
            "DriverId": ["max"] * 3 + ["lando"] * 3 + ["max", "max"],
            "TeamId": ["rb"] * 3 + ["mcl"] * 3 + ["rb", "rb"],
        }
    )


def test_compute_driver_stats_empty_when_year_absent(bronze):
    assert analytics.compute_driver_stats(bronze, 1999).empty


def test_compute_driver_stats_values(bronze):
    stats = analytics.compute_driver_stats(bronze, 2024).set_index("FullName")

    mx = stats.loc["Max V"]
    assert mx["Rank"] == 1
    assert mx["Races"] == 3
    assert mx["Points"] == 76.0  # race + sprint points
    assert mx["Wins"] == 2
    assert mx["Podiums"] == 3
    assert mx["Poles"] == 2
    assert mx["DNFs"] == 0
    assert mx["BestFinish"] == 1.0
    assert mx["AvgGain"] == pytest.approx(0.0)
    assert mx["PodiumRate"] == pytest.approx(1.0)

    ln = stats.loc["Lando N"]
    assert ln["Rank"] == 2
    assert ln["Points"] == 33.0
    assert ln["Wins"] == 0
    assert ln["Podiums"] == 2  # NaN finish is not a podium
    assert ln["DNFs"] == 1
    assert ln["AvgFinish"] == pytest.approx(2.5)  # NaN skipped
    assert ln["PodiumRate"] == pytest.approx(2 / 3)


def test_driver_stats_keeps_team_switch_as_one_driver(bronze):
    switched = bronze.iloc[[0]].copy()
    switched["RoundNumber"] = 4
    switched["TeamId"] = "ferrari"
    switched["TeamName"] = "Ferrari"
    switched["TeamColor"] = "#ff0000"
    switched["Points"] = 10
    frame = pd.concat([bronze, switched], ignore_index=True)
    stats = analytics.compute_driver_stats(frame, 2024)
    max_rows = stats[stats["DriverId"] == "max"]
    assert len(max_rows) == 1
    assert max_rows.iloc[0]["TeamName"] == "Ferrari"
    assert max_rows.iloc[0]["Points"] == 86


def test_grid_zero_and_dnf_do_not_bias_position_averages(bronze):
    frame = bronze.copy()
    frame.loc[
        (frame["DriverId"] == "lando") & (frame["RoundNumber"] == 3),
        "GridPosition",
    ] = 0
    stats = analytics.compute_driver_stats(frame, 2024).set_index("DriverId")
    assert stats.loc["lando", "AvgGrid"] == pytest.approx(3.0)
    assert stats.loc["lando", "AvgFinish"] == pytest.approx(2.5)


def test_compute_team_stats_values(bronze):
    teams = analytics.compute_team_stats(bronze, 2024).set_index("TeamName")

    assert teams.loc["RB", "Rank"] == 1
    assert teams.loc["RB", "Points"] == 76.0
    assert teams.loc["RB", "Wins"] == 2
    assert teams.loc["RB", "Podiums"] == 3
    assert teams.loc["McLaren", "Points"] == 33.0
    assert teams.loc["McLaren", "Wins"] == 0


def test_compute_reliability_values(bronze):
    rel = analytics.compute_reliability(bronze, 2024).set_index("FullName")

    ln = rel.loc["Lando N"]
    assert ln["Starts"] == 3
    assert ln["DNFs"] == 1
    assert ln["DNFRate"] == pytest.approx(1 / 3)
    assert ln["PointsFinishRate"] == pytest.approx(2 / 3)  # R3 scored 0

    mx = rel.loc["Max V"]
    assert mx["DNFs"] == 0
    assert mx["AvgGain"] == pytest.approx(0.0)  # (0 + 1 + -1) / 3


# ── teammate head-to-head ──────────────────────────────────────────────────


@pytest.fixture
def bronze_pairs():
    """RB with two cars over R1-R2 (Max ahead both races, quali split) plus a
    one-car team that must be dropped."""
    return pd.DataFrame(
        {
            "Mode": ["Race"] * 5,
            "Year": [2024] * 5,
            "RoundNumber": [1, 2, 1, 2, 1],
            "Position": [1.0, 1.0, 4.0, 3.0, 2.0],
            "GridPosition": [1.0, 2.0, 3.0, 1.0, 2.0],
            "Points": [25.0, 25.0, 12.0, 15.0, 18.0],
            "FullName": ["Max V", "Max V", "Sergio P", "Sergio P", "Solo D"],
            "TeamName": ["RB", "RB", "RB", "RB", "Solo"],
            "TeamColor": ["#3671C6", "#3671C6", "#3671C6", "#3671C6", "#111111"],
            "Abbreviation": ["VER", "VER", "PER", "PER", "SOL"],
            "ClassifiedPosition": ["1", "1", "4", "3", "2"],
            "DriverId": ["max", "max", "checo", "checo", "solo"],
            "TeamId": ["rb", "rb", "rb", "rb", "solo"],
        }
    )


def test_compute_teammate_h2h(bronze_pairs):
    h2h = analytics.compute_teammate_h2h(bronze_pairs, 2024)
    assert len(h2h) == 1  # Solo team dropped

    row = h2h.iloc[0]
    assert (row["driver_a"], row["driver_b"]) == ("max", "checo")  # a = higher scorer
    assert row["Rounds"] == 2
    assert (row["RaceWinsA"], row["RaceWinsB"]) == (2, 0)
    assert (row["QualiWinsA"], row["QualiWinsB"]) == (1, 1)
    assert row["PointsA"] == 50.0
    assert row["PointsB"] == 27.0


# ── momentum & insights ────────────────────────────────────────────────────


@pytest.fixture
def preds():
    dates = ["2024-03-01", "2024-03-15", "2024-03-29"]
    prob = {
        "aaa": [0.20, 0.30, 0.60],
        "bbb": [0.50, 0.40, 0.30],
        "ccc": [0.10, 0.10, 0.10],
    }
    rows = [
        {
            "dt_ref": dt,
            "DriverId": did,
            "FullName": did.upper(),
            "Abbreviation": did[:3].upper(),
            "TeamName": "T",
            "TeamColor": "#123456",
            "HeadshotUrl": "",
            "prob_win": p,
        }
        for did, series in prob.items()
        for dt, p in zip(dates, series)
    ]
    return pd.DataFrame(rows)


def test_compute_momentum(preds):
    mom = analytics.compute_momentum(preds, last_n=3).set_index("DriverId")

    assert mom.index[0] == "aaa"  # sorted by latest desc
    assert mom.loc["aaa", "latest"] == pytest.approx(0.60)
    assert mom.loc["aaa", "delta_prev"] == pytest.approx(0.30)
    assert mom.loc["aaa", "trend"] == "up"
    assert mom.loc["bbb", "trend"] == "down"
    assert mom.loc["ccc", "trend"] == "flat"
    assert mom.loc["aaa", "rank_change"] == 1  # P2 previous round -> P1 now


def test_compute_momentum_single_date():
    one = pd.DataFrame(
        [{"dt_ref": "2024-03-01", "DriverId": "x", "FullName": "X", "prob_win": 0.4}]
    )
    mom = analytics.compute_momentum(one)
    assert pd.isna(mom.iloc[0]["delta_prev"])
    assert mom.iloc[0]["trend"] == "flat"


def test_build_insights(preds, bronze_pairs):
    mom = analytics.compute_momentum(preds, last_n=3)
    h2h = analytics.compute_teammate_h2h(bronze_pairs, 2024)
    rel = analytics.compute_reliability(bronze_pairs, 2024)

    lines = analytics.build_insights(mom, h2h, rel, last_n=3)

    assert 1 <= len(lines) <= 5
    assert all(isinstance(x, str) and x for x in lines)
    assert any("AAA" in line and "maior mudança" in line for line in lines)


def test_build_insights_empty_inputs():
    empty = pd.DataFrame()
    assert analytics.build_insights(empty, empty, empty) == []


# ── top_factors ────────────────────────────────────────────────────────────


def test_top_factors_ranks_by_importance_and_flags_direction():
    importances = {"f_hi": 0.6, "f_lo": 0.1, "f_missing": 0.3}
    row = pd.Series({"f_hi": 10.0, "f_lo": 1.0})
    median = pd.Series({"f_hi": 4.0, "f_lo": 5.0})

    tf = analytics.top_factors(importances, row, median, k=5)

    assert tf["feature"].tolist() == ["f_hi", "f_lo"]  # f_missing skipped (not in row)
    assert tf.iloc[0]["vs_field"] == "acima"
    assert tf.iloc[1]["vs_field"] == "abaixo"


def test_classify_result_preserves_non_finish_categories():
    assert analytics.classify_result("1", "Finished") == "FINISHED"
    assert analytics.classify_result("R", "Engine") == "DNF"
    assert analytics.classify_result("D", "Disqualified") == "DSQ"
    assert analytics.classify_result("W", "Withdrew") == "DNS"


def test_normalize_probabilities_sums_one_per_snapshot():
    frame = pd.DataFrame(
        {
            "dt_ref": ["2024-01-01", "2024-01-01", "2024-02-01"],
            "raw_score": [0.8, 0.2, 0.4],
        }
    )
    out = analytics.normalize_probabilities(frame)
    assert out.groupby("dt_ref")["prob_win"].sum().tolist() == pytest.approx([1, 1])


def test_normalize_probabilities_uses_uniform_fallback_for_zero_scores():
    frame = pd.DataFrame({"dt_ref": ["r1", "r1"], "raw_score": [0, 0]})
    out = analytics.normalize_probabilities(frame)
    assert out["prob_win"].tolist() == pytest.approx([0.5, 0.5])


def test_standings_history_carries_points_through_missing_round(bronze):
    frame = bronze[~((bronze["FullName"] == "Lando N") & (bronze["RoundNumber"] == 2))]
    history = analytics.compute_standings_history(frame, 2024)
    lando = history[history["DriverId"] == "lando"].set_index("RoundNumber")
    assert lando.loc[2, "CumulativePoints"] == lando.loc[1, "CumulativePoints"]
