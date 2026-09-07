from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.train_driver_champion import _normalize_by_snapshot, prepare_training_frame


def test_prepare_training_frame_excludes_current_season():
    current_year = datetime.now(UTC).year
    frame = pd.DataFrame(
        {
            "dt_ref": [f"{current_year - 1}-03-01", f"{current_year}-03-01"],
            "DriverId": ["a", "b"],
            "feature": [1.0, 2.0],
            "flChampion": [1, 0],
        }
    )
    # Include both classes in the completed season so the validation guard passes.
    frame = pd.concat(
        [
            frame,
            pd.DataFrame(
                [
                    {
                        "dt_ref": f"{current_year - 1}-03-01",
                        "DriverId": "c",
                        "feature": 0.0,
                        "flChampion": 0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    prepared, features = prepare_training_frame(frame, current_year=current_year)
    assert prepared["Year"].unique().tolist() == [current_year - 1]
    assert features == ["feature"]


def test_normalize_by_snapshot_is_mutually_exclusive():
    scores = np.array([0.8, 0.2, 0.3, 0.3])
    dates = pd.Series(["r1", "r1", "r2", "r2"])
    normalized = _normalize_by_snapshot(scores, dates)
    assert normalized[:2].sum() == pytest.approx(1)
    assert normalized[2:].sum() == pytest.approx(1)
