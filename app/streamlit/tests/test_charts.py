"""Contratos estruturais dos gráficos editoriais."""

import pandas as pd

import charts


def test_probability_ranking_is_ordered_horizontal_bar():
    frame = pd.DataFrame(
        {
            "FullName": ["A", "B"],
            "TeamName": ["T1", "T2"],
            "TeamColor": ["#111111", "#222222"],
            "latest": [0.7, 0.3],
            "delta_prev": [0.1, -0.1],
        }
    )
    figure = charts.probability_ranking(frame)
    assert figure.data[0].orientation == "h"
    assert list(figure.data[0].y) == ["B", "A"]


def test_result_heatmap_respects_round_and_driver_order():
    frame = pd.DataFrame(
        {
            "RoundNumber": [2, 1, 2, 1],
            "EventName": ["Second", "First", "Second", "First"],
            "Abbreviation": ["AAA", "AAA", "BBB", "BBB"],
            "OfficialFinish": [2, 1, 1, pd.NA],
            "ResultLabel": ["P2", "P1", "P1", "DNF"],
        }
    )
    figure = charts.result_heatmap(frame, ["BBB", "AAA"])
    assert list(figure.data[0].x) == ["First", "Second"]
    assert list(figure.data[0].y) == ["BBB", "AAA"]


def test_calibration_chart_always_contains_reference_line():
    figure = charts.calibration_curve([])
    assert len(figure.data) == 1
    assert list(figure.data[0].x) == [0, 1]
