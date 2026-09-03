"""Leakage-safe dataset builder for S2S model calibration.

Builds a reproducible, chronological dataset from persisted predictions,
resolved outcomes, pre-kickoff snapshots and optional advanced fixture data.
It is strictly a sports forecasting/calibration artifact: no odds, staking,
or wagering logic is produced.
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


def _fetch_rows(model_version: str | None = None) -> tuple[list[tuple[Any, ...]], dict[str, tuple[Any, ...]], dict[str, tuple[Any, ...]]]:
    where = " WHERE model_version=%s" if model_version and main.DATABASE_URL else ""
    sqlite_where = " WHERE model_version=?" if model_version and not main.DATABASE_URL else ""
    params = (model_version,) if model_version else ()
    if main.DATABASE_URL:
        with main.psycopg.connect(main.DATABASE_URL, connect_timeout=15, prepare_threshold=None) as db:
            outcomes = db.execute(
                "SELECT fixture_id,kickoff_utc,model_version,prediction,actual_home,actual_away,metrics,resolved_at FROM prediction_outcomes" + where,
                params,
            ).fetchall()
            snapshots = db.execute(
                "SELECT fixture_id,fixture_date,cutoff_utc,model_version,payload,created_at FROM snapshots" + where,
                params,
            ).fetchall()
            advanced = db.execute(
                "SELECT fixture_id,payload,created_at FROM advanced_fixture_stats"
            ).fetchall()
    else:
        import sqlite3
        with sqlite3.connect(main.STATE_DB_PATH) as db:
            outcomes = db.execute(
                "SELECT fixture_id,kickoff_utc,model_version,prediction,actual_home,actual_away,metrics,resolved_at FROM prediction_outcomes" + sqlite_where,
                params,
            ).fetchall()
            snapshots = db.execute(
                "SELECT fixture_id,fixture_date,cutoff_utc,model_version,payload,created_at FROM snapshots" + sqlite_where,
                params,
            ).fetchall()
            advanced = db.execute("SELECT fixture_id,payload,created_at FROM advanced_fixture_stats").fetchall()
    return outcomes, {str(row[0]): row for row in snapshots}, {str(row[0]): row for row in advanced}


def build_dataset(model_version: str | None = None) -> tuple[list[TrainingRow], dict[str, Any]]:
    outcomes, snapshots, advanced = _fetch_rows(model_version)
    rows: list[TrainingRow] = []
    rejected = {"missing_kickoff": 0, "missing_snapshot": 0, "future_cutoff": 0, "invalid_outcome": 0}
    advanced_supported = 0

    for outcome in outcomes:
        fixture_id = str(outcome[0])
        kickoff = _dt(outcome[1])
        snapshot_row = snapshots.get(fixture_id)
        if kickoff is None:
            rejected["missing_kickoff"] += 1
            continue
        if snapshot_row is None:
            rejected["missing_snapshot"] += 1
            continue
        cutoff = _dt(snapshot_row[2])
        if cutoff is None or cutoff > kickoff:
            rejected["future_cutoff"] += 1
            continue
        try:
            actual_home, actual_away = int(outcome[4]), int(outcome[5])
        except (TypeError, ValueError):
            rejected["invalid_outcome"] += 1
            continue

        snapshot_payload = _json(snapshot_row[4])
        advanced_row = advanced.get(fixture_id)
        advanced_payload = _json(advanced_row[1]) if advanced_row else {}
        availability = str(advanced_payload.get("availability") or "").upper()
        teams = advanced_payload.get("teams")
        supported = availability != "UNAVAILABLE" and isinstance(teams, dict) and bool(teams)
        if supported:
            advanced_supported += 1

        rows.append(TrainingRow(
            fixture_id=fixture_id,
            kickoff_utc=kickoff.astimezone(UTC).isoformat(),
            feature_cutoff_utc=cutoff.astimezone(UTC).isoformat(),
            model_version=str(outcome[2]),
            feature_version=str(snapshot_payload.get("feature_version") or main.ENGINE_VERSION),
            actual_home=actual_home,
            actual_away=actual_away,
            prediction=_json(outcome[3]),
            metrics=_json(outcome[6]),
            snapshot=snapshot_payload,
            advanced=advanced_payload if advanced_row else None,
            advanced_available=supported,
        ))

    rows.sort(key=lambda row: (row.kickoff_utc, row.fixture_id))
    n = len(rows)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_version": model_version or "ALL",
        "raw_outcomes": len(outcomes),
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
