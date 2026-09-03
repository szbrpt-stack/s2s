from datetime import datetime, timezone

import training_dataset


def test_training_row_cutoff_never_after_kickoff():
    row = training_dataset.TrainingRow(
        fixture_id="1",
        kickoff_utc="2026-01-02T00:00:00+00:00",
        feature_cutoff_utc="2026-01-01T23:00:00+00:00",
        model_version="test",
        feature_version="test",
        actual_home=1,
        actual_away=0,
        prediction={},
        metrics={},
        snapshot={},
        advanced=None,
        advanced_available=False,
    )
    assert datetime.fromisoformat(row.feature_cutoff_utc) <= datetime.fromisoformat(row.kickoff_utc)


def test_training_rows_sort_chronologically():
    rows = [
        training_dataset.TrainingRow("2", "2026-01-02T00:00:00+00:00", "2026-01-01T00:00:00+00:00", "m", "f", 0, 0, {}, {}, {}, None, False),
        training_dataset.TrainingRow("1", "2026-01-01T00:00:00+00:00", "2025-12-31T00:00:00+00:00", "m", "f", 1, 0, {}, {}, {}, None, False),
    ]
    rows.sort(key=lambda row: (row.kickoff_utc, row.fixture_id))
    assert [row.fixture_id for row in rows] == ["1", "2"]
