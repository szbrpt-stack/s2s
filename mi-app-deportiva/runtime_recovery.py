from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import main

app = main.app
BOOT_ID = str(uuid.uuid4())
BOOTED_AT = datetime.now(timezone.utc)
STALE_JOB_MINUTES = max(10, int(os.getenv("CALIBRATION_STALE_MINUTES", "45")))
WATCHDOG_SECONDS = max(60, int(os.getenv("RUNTIME_WATCHDOG_SECONDS", "300")))
SNAPSHOT_WARN_HOURS = max(6, int(os.getenv("SNAPSHOT_WARN_HOURS", "36")))
_watchdog_task: asyncio.Task[None] | None = None


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _age_minutes(value: Any) -> float | None:
    parsed = _parse_dt(value)
    if not parsed:
        return None
    return max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() / 60.0)


def _scorecard_sample_size(scorecard: dict[str, Any] | None) -> int:
    if not isinstance(scorecard, dict):
        return 0
    for value in (
        scorecard.get("resolved_predictions"),
        (scorecard.get("overall") or {}).get("n") if isinstance(scorecard.get("overall"), dict) else None,
        scorecard.get("n"),
        scorecard.get("total"),
    ):
        try:
            if value is not None:
                return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return 0


def _db_runtime_snapshot() -> dict[str, Any]:
    if not main.DATABASE_URL or main.psycopg is None:
        return {
            "database": main.DATABASE_BACKEND,
            "persistent": bool(main.DATABASE_URL),
            "running_jobs": [],
            "latest_snapshot_at": None,
            "latest_advanced_stats_at": None,
            "latest_outcome_at": None,
        }
    with main.psycopg.connect(main.DATABASE_URL, connect_timeout=15, prepare_threshold=None) as db:
        running = db.execute(
            "SELECT job_id,status,phase,done,total,payload,updated_at "
            "FROM calibration_jobs WHERE status='RUNNING' ORDER BY updated_at"
        ).fetchall()
        latest_snapshot = db.execute("SELECT MAX(created_at) FROM snapshots").fetchone()[0]
        latest_advanced = db.execute("SELECT MAX(created_at) FROM advanced_fixture_stats").fetchone()[0]
        latest_outcome = db.execute("SELECT MAX(resolved_at) FROM prediction_outcomes").fetchone()[0]
    jobs = []
    for job_id, status, phase, done, total, payload, updated_at in running:
        try:
            payload_obj = json.loads(payload) if payload else {}
        except (TypeError, json.JSONDecodeError):
            payload_obj = {"raw": payload}
        jobs.append({
            "job_id": job_id,
            "status": status,
            "phase": phase,
            "done": done,
            "total": total,
            "payload": payload_obj,
            "updated_at": updated_at,
            "age_minutes": _age_minutes(updated_at),
        })
    return {
        "database": main.DATABASE_BACKEND,
        "persistent": bool(main.DATABASE_URL),
        "running_jobs": jobs,
        "latest_snapshot_at": latest_snapshot,
        "latest_advanced_stats_at": latest_advanced,
        "latest_outcome_at": latest_outcome,
    }


def _recover_stale_jobs() -> list[str]:
    if not main.DATABASE_URL or main.psycopg is None:
        return []
    recovered: list[str] = []
    now = datetime.now(timezone.utc)
    with main.psycopg.connect(main.DATABASE_URL, connect_timeout=15, prepare_threshold=None) as db:
        rows = db.execute(
            "SELECT job_id,payload,updated_at FROM calibration_jobs WHERE status='RUNNING'"
        ).fetchall()
        for job_id, payload, updated_at in rows:
            age = _age_minutes(updated_at)
            if age is None or age < STALE_JOB_MINUTES:
                continue
            try:
                data = json.loads(payload) if payload else {}
            except (TypeError, json.JSONDecodeError):
                data = {"raw_payload": payload}
            data.update({
                "recovered_at": now.isoformat(),
                "recovered_by_boot_id": BOOT_ID,
                "recovery_reason": f"No heartbeat for at least {STALE_JOB_MINUTES} minutes",
            })
            db.execute(
                "UPDATE calibration_jobs SET status=%s,phase=%s,payload=%s,updated_at=%s WHERE job_id=%s",
                ("ABORTED", "STALE_RECOVERED", json.dumps(data, ensure_ascii=False), now.isoformat(), job_id),
            )
            recovered.append(str(job_id))
    if recovered:
        main._calibration_job_id = None
        main._calibration_state.update({
            "running": False,
            "finished_at": now.isoformat(),
            "error": "Recovered stale durable calibration job(s) after process loss",
            "run_id": None,
        })
        main.log.warning("Recovered stale calibration jobs: %s", recovered)
    return recovered


async def _watchdog() -> None:
    while True:
        try:
            await asyncio.to_thread(_recover_stale_jobs)
        except Exception:
            main.log.exception("Runtime watchdog failed")
        await asyncio.sleep(WATCHDOG_SECONDS)


async def _startup_recovery() -> None:
    global _watchdog_task
    recovered = await asyncio.to_thread(_recover_stale_jobs)
    main.log.info("Runtime recovery boot_id=%s recovered=%s", BOOT_ID, recovered)
    if _watchdog_task is None or _watchdog_task.done():
        _watchdog_task = asyncio.create_task(_watchdog())


app.add_event_handler("startup", _startup_recovery)


@app.get("/api/v1/runtime/integrity")
async def runtime_integrity() -> dict[str, Any]:
    snapshot = await asyncio.to_thread(_db_runtime_snapshot)
    memory_running = bool(main._calibration_task and not main._calibration_task.done())
    stale_jobs = [
        job for job in snapshot["running_jobs"]
        if job.get("age_minutes") is not None and float(job["age_minutes"]) >= STALE_JOB_MINUTES
    ]
    snapshot_age = _age_minutes(snapshot.get("latest_snapshot_at"))
    advanced_age = _age_minutes(snapshot.get("latest_advanced_stats_at"))
    issues: list[str] = []
    if stale_jobs:
        issues.append("STALE_CALIBRATION_JOB")
    if snapshot_age is not None and snapshot_age > SNAPSHOT_WARN_HOURS * 60:
        issues.append("SNAPSHOT_FRESHNESS_DEGRADED")
    if snapshot["running_jobs"] and not memory_running:
        issues.append("DURABLE_MEMORY_JOB_MISMATCH")
    return {
        "status": "ok" if not issues else "degraded",
        "issues": issues,
        "boot_id": BOOT_ID,
        "booted_at": BOOTED_AT.isoformat(),
        "engine_version": main.ENGINE_VERSION,
        "model_version": main.MODEL_VERSION,
        "database": snapshot["database"],
        "database_persistent": snapshot["persistent"],
        "analysis_running": bool(main._analysis_task and not main._analysis_task.done()),
        "calibration_memory_running": memory_running,
        "calibration_jobs_running": snapshot["running_jobs"],
        "refresh_policy": {
            "snapshot_mode": "ON_DEMAND",
            "automatic_scheduler_configured": False,
            "triggers": ["/api/v1/sync", "/api/v1/coverage", "/api/v1/props"],
        },
        "freshness": {
            "latest_snapshot_at": snapshot.get("latest_snapshot_at"),
            "snapshot_age_minutes": round(snapshot_age, 2) if snapshot_age is not None else None,
            "latest_advanced_stats_at": snapshot.get("latest_advanced_stats_at"),
            "advanced_stats_age_minutes": round(advanced_age, 2) if advanced_age is not None else None,
            "latest_outcome_at": snapshot.get("latest_outcome_at"),
            "outcome_age_minutes": round(_age_minutes(snapshot.get("latest_outcome_at")), 2)
            if _age_minutes(snapshot.get("latest_outcome_at")) is not None else None,
        },
        "stale_job_minutes": STALE_JOB_MINUTES,
    }


@app.get("/api/v1/model/integrity")
async def model_integrity() -> dict[str, Any]:
    latest = await asyncio.to_thread(main.db_latest_calibration)
    scorecard = await asyncio.to_thread(main.db_model_scorecard)
    latest = latest or {}
    holdout = latest.get("holdout") or {}
    walk = latest.get("walk_forward") or {}
    holdout_n = int(holdout.get("n") or 0)
    score_n = _scorecard_sample_size(scorecard)
    passing_folds = int(walk.get("passing_folds") or 0)
    folds = int(walk.get("folds") or 0)
    gates = {
        "holdout_sample_sufficient": holdout_n >= 500,
        "walk_forward_evaluated": folds >= 4,
        "walk_forward_stable": bool(walk.get("stable_candidate")) and passing_folds >= 3,
        "prequential_sample_sufficient": score_n >= 200,
    }
    return {
        "model_version": main.MODEL_VERSION,
        "validation_status": main.MODEL_VALIDATION_STATUS,
        "latest_calibration_status": latest.get("status"),
        "latest_calibration_date": latest.get("date"),
        "parameters": latest.get("parameters"),
        "holdout": holdout,
        "walk_forward": {
            "status": walk.get("status"),
            "folds": folds,
            "passing_folds": passing_folds,
            "required_passing_folds": walk.get("required_passing_folds"),
            "stable_candidate": walk.get("stable_candidate"),
        },
        "prequential_sample_size": score_n,
        "prequential_scorecard": scorecard,
        "reliability_gates": gates,
        "reliable_for_reporting": all(gates.values()),
        "traceability": {
            "chronological_holdout": True,
            "walk_forward_required": True,
            "pre_kickoff_outcomes_only": True,
            "automatic_parameter_promotion": False,
        },
    }
