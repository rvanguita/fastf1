"""Tests for ``src.extract_data.ExtractData`` — FastF1 is never contacted;
``get_data`` / ``prepare_data`` inputs are supplied directly."""

from pathlib import Path

import pandas as pd
import pytest

from src.extract_data import CURRENT_YEAR, ExtractData


@pytest.fixture
def extractor(tmp_path, monkeypatch):
    """An ``ExtractData`` whose Raw directory is an isolated tmp path."""
    monkeypatch.setenv("PATH_RAW", str(tmp_path))
    # PATH_RAW is read at import time into a module global, so patch that too.
    monkeypatch.setattr("src.extract_data.PATH_RAW", str(tmp_path))
    return ExtractData()


class TestInit:
    def test_default_years_is_current_year(self):
        assert ExtractData().years == [CURRENT_YEAR]

    def test_reload_data_expands_year_range(self):
        ext = ExtractData(reload_data=True)
        assert ext.years == list(range(1980, CURRENT_YEAR))
        assert ext.reload_data is True

    def test_path_save_data_uses_base_data(self, extractor):
        assert extractor.path_save_data.endswith("/results")

    def test_init_creates_save_directory(self, extractor):
        assert Path(extractor.path_save_data).is_dir()


class TestProcessData:
    def test_skips_when_file_exists(self, extractor, monkeypatch):
        target = Path(extractor.path_save_data) / "2024_01_R.parquet"
        target.write_bytes(b"")

        def _boom(*a, **k):  # pragma: no cover - must not be reached
            raise AssertionError("get_data should not be called for existing file")

        monkeypatch.setattr(extractor, "get_data", _boom)
        assert extractor.process_data(2024, 1, "R") is False

    def test_returns_false_on_empty_dataframe(self, extractor, monkeypatch):
        monkeypatch.setattr(extractor, "get_data", lambda *a, **k: pd.DataFrame())
        assert extractor.process_data(2024, 2, "R") is False
        assert not (Path(extractor.path_save_data) / "2024_02_R.parquet").exists()

    def test_writes_parquet_and_returns_true(self, extractor, monkeypatch):
        df = pd.DataFrame({"DriverId": ["max"], "Points": [25.0]})
        monkeypatch.setattr(extractor, "get_data", lambda *a, **k: df)

        assert extractor.process_data(2024, 3, "R") is True

        written = Path(extractor.path_save_data) / "2024_03_R.parquet"
        assert written.is_file()
        pd.testing.assert_frame_equal(pd.read_parquet(written), df)

    def test_filename_zero_pads_round(self, extractor, monkeypatch):
        df = pd.DataFrame({"DriverId": ["max"]})
        monkeypatch.setattr(extractor, "get_data", lambda *a, **k: df)
        extractor.process_data(2024, 7, "S")
        assert (Path(extractor.path_save_data) / "2024_07_S.parquet").is_file()


class TestPrepareData:
    def test_adds_event_columns_and_casts(self, extractor, fake_fastf1_session):
        session = fake_fastf1_session(year=2024, mode="Race", round_number=4)

        out = extractor.prepare_data(session)

        assert (out["Year"] == 2024).all()
        assert (out["Mode"] == "Race").all()
        assert (out["RoundNumber"] == 4).all()
        assert out["EventName"].unique().tolist() == ["Test Grand Prix"]
        assert str(out["Date"].dtype) == "datetime64[us]"
        # Q1/Q2/Q3/Time converted from timedelta to float seconds
        for col in ("Q1", "Q2", "Q3", "Time"):
            assert out[col].dtype == "float64"
        assert out["Q1"].iloc[0] == pytest.approx(80.5)
        assert out["Time"].iloc[0] == pytest.approx(5400.0)


class TestLoops:
    def test_process_years_returns_true_when_any_year_has_new_data(
        self, extractor, monkeypatch
    ):
        monkeypatch.setattr("src.extract_data.time.sleep", lambda *_: None)
        extractor.years = [2023, 2024]
        results = iter([True, False])
        monkeypatch.setattr(extractor, "process_identifiers", lambda *_: next(results))
        assert extractor.process_years() is True

    def test_process_identifiers_returns_true_when_any_session_has_new_data(
        self, extractor, monkeypatch
    ):
        monkeypatch.setattr("src.extract_data.time.sleep", lambda *_: None)
        extractor.identifiers = ["R"]
        results = iter([True, *([False] * 28)])
        monkeypatch.setattr(extractor, "process_data", lambda *_: next(results))
        assert extractor.process_identifiers(2024) is True

    def test_process_years_empty_years_does_not_raise(self, extractor, monkeypatch):
        monkeypatch.setattr("src.extract_data.time.sleep", lambda *_: None)
        extractor.years = []
        assert extractor.process_years() is False

    def test_process_identifiers_empty_identifiers_does_not_raise(
        self, extractor, monkeypatch
    ):
        monkeypatch.setattr("src.extract_data.time.sleep", lambda *_: None)
        extractor.identifiers = []
        assert extractor.process_identifiers(2024) is False
