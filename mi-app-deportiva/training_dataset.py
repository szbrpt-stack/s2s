"""Leakage-safe, egress-aware dataset builder for S2S calibration.

PostgreSQL mode performs the filtering and joins in-database and returns only
rows belonging to the requested model cohort. It never downloads the complete
advanced_fixture_stats warehouse.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import main

UTC = timezone.utc


def _dt(value: Any) -> datetime | None:
    return main.parse_dt(value)


def _json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


@dataclass(frozen=True)
class TrainingRow:
    fixture_id: str
    kickoff_utc: str
    feature_cutoff_utc: str
    model_version: str
    feature_version: str
    actual_home: int
    actual_away: int
    prediction: dict[str, Any]
    metrics: dict[str, Any]
    snapshot: dict[str, Any]
    advanced: dict[str, Any] | None
    advanced_available: bool


def _fetch_joined_postgres(model_version: str | None) -> list[tuple[Any, ...]]:
    """One bounded query; advanced payload is returned only for matching outcomes."""
    where = "AND o.model_version=%s AND s.model_version=%s" if model_version else ""
    params = (model_version, model_version) if model_version else ()
    query = f"""
        SELECT
            o.fixture_id,
            o.kickoff_utc,
            o.model_version,
            o.prediction,
            o.actual_home,
            o.actual_away,
            o.metrics,
            s.cutoff_utc,
            s.payload,
            a.payload
        FROM prediction_outcomes o
        JOIN snapshots s ON s.fixture_id=o.fixture_id
        LEFT JOIN advanced_fixture_stats a
          ON a.fixture_id = CASE
              WHEN o.fixture_id ~ '^[0-9]+$' THEN o.fixture_id::bigint
              ELSE NULL
          END
        WHERE o.kickoff_utc IS NOT NULL
          AND s.cutoff_utc IS NOT NULL
          AND s.cutoff_utc::timestamptz <= o.kickoff_utc::timestamptz
          {where}
        ORDER BY o.kickoff_utc::timestamptz, o.fixture_id
    """
    with main.psycopg.connect(main.DATABASE_URL, connect_timeout=15, prepare_threshold=None) as db:
        return db.execute(query, params).fetchall()


def _fetch_joined_sqlite(model_version: str | None) -> list[tuple[Any, ...]]:
    import sqlite3
    where = "AND o.model_version=? AND s.model_version=?" if model_version else ""
    params = (model_version, model_version) if model_version else ()
    query = f"""
        SELECT o.fixture_id,o.kickoff_utc,o.model_version,o.prediction,
               o.actual_home,o.actual_away,o.metrics,s.cutoff_utc,s.payload,a.payload
        FROM prediction_outcomes o
        JOIN snapshots s ON s.fixture_id=o.fixture_id
        LEFT JOIN advanced_fixture_stats a ON CAST(a.fixture_id AS TEXT)=o.fixture_id
        WHERE o.kickoff_utc IS NOT NULL AND s.cutoff_utc IS NOT NULL {where}
        ORDER BY o.kickoff_utc,o.fixture_id
    """
    with sqlite3.connect(main.STATE_DB_PATH) as db:
        return db.execute(query, params).fetchall()


def build_dataset(model_version: str | None = None) -> tuple[list[TrainingRow], dict[str, Any]]:
    joined = _fetch_joined_postgres(model_version) if main.DATABASE_URL else _fetch_joined_sqlite(model_version)
    rows: list[TrainingRow] = []
    rejected = {"missing_kickoff": 0, "future_cutoff": 0, "invalid_outcome": 0}
    advanced_supported = 0

    for item in joined:
        fixture_id = str(item[0])
        kickoff, cutoff = _dt(item[1]), _dt(item[7])
        if kickoff is None:
            rejected["missing_kickoff"] += 1
            continue
        if cutoff is None or cutoff > kickoff:
            rejected["future_cutoff"] += 1
            continue
        try:
            actual_home, actual_away = int(item[4]), int(item[5])
        except (TypeError, ValueError):
            rejected["invalid_outcome"] += 1
            continue

        snapshot_payload = _json(item[8])
        advanced_payload = _json(item[9]) if item[9] else {}
        availability = str(advanced_payload.get("availability") or "").upper()
        teams = advanced_payload.get("teams")
        supported = availability != "UNAVAILABLE" and isinstance(teams, dict) and bool(teams)
        advanced_supported += int(supported)

        rows.append(TrainingRow(
            fixture_id=fixture_id,
            kickoff_utc=kickoff.astimezone(UTC).isoformat(),
            feature_cutoff_utc=cutoff.astimezone(UTC).isoformat(),
            model_version=str(item[2]),
            feature_version=str(snapshot_payload.get("feature_version") or main.ENGINE_VERSION),
            actual_home=actual_home,
            actual_away=actual_away,
            prediction=_json(item[3]),
            metrics=_json(item[6]),
            snapshot=snapshot_payload,
            advanced=advanced_payload if item[9] else None,
            advanced_available=supported,
        ))

    n = len(rows)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_version": model_version or "ALL",
        "query_strategy": "server_side_join_filtered_by_model_version",
        "usable_rows": n,
        "rejected": rejected,
        "leakage_violations_in_output": 0,
        "advanced_supported_rows": advanced_supported,
        "advanced_supported_rate": round(advanced_supported / n, 4) if n else 0.0,
        "split": {
            "method": "chronological_70_15_15",
            "train": train_end,
            "validation": validation_end - train_end,
            "holdout": n - validation_end,
            "train_end_kickoff": rows[train_end - 1].kickoff_utc if train_end else None,
            "validation_end_kickoff": rows[validation_end - 1].kickoff_utc if validation_end > train_end else None,
        },
    }
    return rows, summary


def export_jsonl(path: str, rows: list[TrainingRow]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe chronological S2S calibration dataset")
    parser.add_argument("--model-version", default=main.MODEL_VERSION)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    main.init_db()
    rows, summary = build_dataset(args.model_version or None)
    if args.output:
        export_jsonl(args.output, rows)
        summary["output"] = args.output
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main_cli()
