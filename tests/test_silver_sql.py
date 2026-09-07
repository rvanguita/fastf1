"""Tests for SQL-file handling in ``src.silver_data`` — no SparkSession is
started (``SilverData`` is never instantiated); only ``read_sql_file`` and the
``.format(...)`` contract of the query files are exercised."""

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
