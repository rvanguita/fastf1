"""Tests for Silver helpers without starting a SparkSession."""

from unittest.mock import Mock, call

import pytest

from src.silver_data import CURRENT_YEAR, SilverData


def read_sql_file(name):
    """``SilverData.read_sql_file`` ignores ``self`` — call it with a dummy."""
    return SilverData.read_sql_file(None, name)


def test_read_sql_file_returns_query_text():
    query = read_sql_file("champions")
    assert "results" in query
    assert "rank_driver" in query


@pytest.mark.parametrize(
    "name",
    [
        "champions",
        "driver_statistic",
        "tb_abt",
        "mart_driver_round",
        "mart_standings",
    ],
)
def test_query_file_is_non_empty(name):
    assert read_sql_file(name).strip()


def test_driver_statistic_is_str_formattable():
    """`driver_statistic.sql` is `.format()`-ed by ``driver_n_race``; a stray
    unescaped brace anywhere in the file would raise here."""
    query = read_sql_file("driver_statistic")
    formatted = query.format(year_start=1980, year_stop=CURRENT_YEAR, last_rounds=5)
    assert "{" not in formatted and "}" not in formatted


def test_missing_query_file_raises():
    with pytest.raises(FileNotFoundError):
        read_sql_file("does_not_exist")


def test_driver_statistics_reuses_cached_results_for_all_windows():
    silver = SilverData.__new__(SilverData)
    silver.cache_results = Mock()
    silver.driver_n_race = Mock()

    silver.driver_statistics(rounds=(5, 10))

    silver.cache_results.assert_called_once_with()
    assert silver.driver_n_race.call_args_list == [
        call(query_name="driver_statistic", round=5),
        call(query_name="driver_statistic", round=10),
    ]
