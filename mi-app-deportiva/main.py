"""S2S Sigma Engine v5.

Render start command:
    uvicorn main:app --host 0.0.0.0 --port $PORT

Required environment variable:
    API_FOOTBALL_KEY

This service publishes sports evidence and probabilistic estimates. It does not
publish betting recommendations, odds, edge, expected value, or certified
viability. Until an external backtest calibrates the model, viability is null.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sqlite3
import time
import uuid
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, Header, HTTPException, Query

try:
    import psycopg
except ImportError:  # El desarrollo local puede continuar con SQLite.
    psycopg = None

ENGINE_VERSION = "5.6.0"
CONTRACT_VERSION = "5.0"
MODEL_VERSION = "dc-shrunk-v3-league-coherent-evaluation"
MODEL_VALIDATION_STATUS = "WALK_FORWARD_EVALUATED_NOT_EXTERNALLY_CERTIFIED"
BASE_URL = "https://v3.football.api-sports.io"
BOGOTA = ZoneInfo("America/Bogota")
UTC = timezone.utc

API_KEY = os.getenv("API_FOOTBALL_KEY", "").strip()
REQUESTS_PER_MINUTE = max(60, min(int(os.getenv("REQUESTS_PER_MINUTE", "360")), 420))
HTTP_CONCURRENCY = max(2, min(int(os.getenv("HTTP_CONCURRENCY", "12")), 20))
ANALYSIS_WORKERS = max(1, min(int(os.getenv("ANALYSIS_WORKERS", "8")), 12))
CATALOG_TTL = max(30, int(os.getenv("CATALOG_TTL_SECONDS", "60")))
TEAM_TTL = max(900, int(os.getenv("TEAM_TTL_SECONDS", "21600")))
LEAGUE_TTL = max(900, int(os.getenv("LEAGUE_TTL_SECONDS", "21600")))
FIXTURE_STATS_TTL = max(3600, int(os.getenv("FIXTURE_STATS_TTL_SECONDS", "86400")))
H2H_TTL = max(3600, int(os.getenv("H2H_TTL_SECONDS", "86400")))
HISTORY_SIZE = max(5, min(int(os.getenv("HISTORY_SIZE", "10")), 20))
MIN_HISTORY = max(3, min(int(os.getenv("MIN_HISTORY", "5")), HISTORY_SIZE))
MIN_SEASON_PLAYED = max(3, int(os.getenv("MIN_SEASON_PLAYED", "5")))
MIN_EVIDENCE_QUALITY = max(0, min(int(os.getenv("MIN_EVIDENCE_QUALITY", "55")), 100))
MAX_GOALS_HISTORY_GAP = max(0.10, min(float(os.getenv("MAX_GOALS_HISTORY_GAP", "0.30")), 0.50))
STALE_NS_MINUTES = max(5, int(os.getenv("STALE_NS_MINUTES", "15")))
SHRINKAGE_MATCHES = max(3.0, float(os.getenv("SHRINKAGE_MATCHES", "12")))
DIXON_COLES_RHO = max(-0.20, min(float(os.getenv("DIXON_COLES_RHO", "-0.05")), 0.05))
RECENCY_STRENGTH = max(0.0, min(float(os.getenv("RECENCY_STRENGTH", "0.0")), 1.0))
DEFAULT_DB_PATH = "/var/data/s2s_sigma_v5.db" if Path("/var/data").exists() else "/tmp/s2s_sigma_v5.db"
STATE_DB_PATH = os.getenv("STATE_DB_PATH", DEFAULT_DB_PATH)
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DATABASE_BACKEND = "postgresql" if DATABASE_URL else "sqlite"
ADMIN_TOKEN = os.getenv("S2S_ADMIN_TOKEN", "").strip()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("S2S")
app = FastAPI(title="S2S Sigma Engine", version=ENGINE_VERSION)

_http_slots = asyncio.Semaphore(HTTP_CONCURRENCY)
_catalog_lock = asyncio.Lock()
_key_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
_memory_cache: dict[str, tuple[float, Any]] = {}
_catalog: list[dict[str, Any]] = []
_catalog_date: str | None = None
_catalog_loaded_monotonic = 0.0
_analysis_task: asyncio.Task[None] | None = None
_analysis_errors: dict[str, str] = {}
_progress = {"phase": "IDLE", "done": 0, "total": 0, "overall_done": 0.0, "overall_total": 1.0, "started_at": None}
_quota = {"daily_remaining": None, "minute_remaining": None, "last_call_at": None}
_calibration_task: asyncio.Task[None] | None = None
_calibration_job_id: str | None = None
_calibration_state: dict[str, Any] = {"running": False, "started_at": None, "finished_at": None, "error": None, "run_id": None}


class PacedRateLimiter:
    """Evenly spaces calls; never emits a large initial burst."""

    def __init__(self, requests_per_minute: int) -> None:
        self.interval = 60.0 / requests_per_minute
        self.next_at = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self.lock:
            now = time.monotonic()
            wait = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval
        if wait:
            await asyncio.sleep(wait)


_limiter = PacedRateLimiter(REQUESTS_PER_MINUTE)


def now_local() -> datetime:
    return datetime.now(BOGOTA)


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return None


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        if isinstance(value, str):
            value = value.replace("%", "").strip()
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def cache_get(key: str) -> Any | None:
    item = _memory_cache.get(key)
    if item and item[0] > time.monotonic():
        return item[1]
    _memory_cache.pop(key, None)
    return None


def cache_put(key: str, value: Any, ttl: int) -> Any:
    _memory_cache[key] = (time.monotonic() + ttl, value)
    return value


def chunks(values: Iterable[int], size: int = 20) -> list[list[int]]:
    rows = list(values)
    return [rows[index:index + size] for index in range(0, len(rows), size)]


def init_db() -> None:
    if DATABASE_URL:
        if psycopg is None:
            raise RuntimeError("DATABASE_URL está configurada pero falta psycopg")
        with psycopg.connect(DATABASE_URL, connect_timeout=15, prepare_threshold=None) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS snapshots (
                fixture_id TEXT PRIMARY KEY, fixture_date TEXT NOT NULL,
                cutoff_utc TEXT NOT NULL, model_version TEXT NOT NULL,
                payload TEXT NOT NULL, created_at TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS calibration_runs (
                id BIGSERIAL PRIMARY KEY, created_at TEXT NOT NULL,
                status TEXT NOT NULL, promoted INTEGER NOT NULL DEFAULT 0,
                payload TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS advanced_fixture_stats (
                fixture_id BIGINT PRIMARY KEY, payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS prediction_outcomes (
                fixture_id TEXT PRIMARY KEY, kickoff_utc TEXT,
                model_version TEXT NOT NULL, prediction TEXT NOT NULL,
                actual_home INTEGER NOT NULL, actual_away INTEGER NOT NULL,
                metrics TEXT NOT NULL, resolved_at TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS calibration_jobs (
                job_id TEXT PRIMARY KEY, status TEXT NOT NULL, phase TEXT NOT NULL,
                done INTEGER NOT NULL, total INTEGER NOT NULL, payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )""")
        log.info("Persistencia PostgreSQL inicializada")
        return
    path = Path(STATE_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(
            """CREATE TABLE IF NOT EXISTS snapshots (
                fixture_id TEXT PRIMARY KEY,
                fixture_date TEXT NOT NULL,
                cutoff_utc TEXT NOT NULL,
                model_version TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        db.execute(
            """CREATE TABLE IF NOT EXISTS calibration_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                promoted INTEGER NOT NULL DEFAULT 0,
                payload TEXT NOT NULL
            )"""
        )
        db.execute(
            """CREATE TABLE IF NOT EXISTS advanced_fixture_stats (
                fixture_id INTEGER PRIMARY KEY,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        db.execute(
            """CREATE TABLE IF NOT EXISTS prediction_outcomes (
                fixture_id TEXT PRIMARY KEY, kickoff_utc TEXT,
                model_version TEXT NOT NULL, prediction TEXT NOT NULL,
                actual_home INTEGER NOT NULL, actual_away INTEGER NOT NULL,
                metrics TEXT NOT NULL, resolved_at TEXT NOT NULL
            )"""
        )
        db.execute(
            """CREATE TABLE IF NOT EXISTS calibration_jobs (
                job_id TEXT PRIMARY KEY, status TEXT NOT NULL, phase TEXT NOT NULL,
                done INTEGER NOT NULL, total INTEGER NOT NULL, payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )
        db.commit()


def db_load_snapshots(fixture_date: str) -> dict[str, dict[str, Any]]:
    if DATABASE_URL:
        with psycopg.connect(DATABASE_URL, connect_timeout=15, prepare_threshold=None) as db:
            rows = db.execute(
                "SELECT fixture_id, payload FROM snapshots WHERE fixture_date=%s", (fixture_date,)
            ).fetchall()
    else:
        with closing(sqlite3.connect(STATE_DB_PATH)) as db:
            rows = db.execute(
                "SELECT fixture_id, payload FROM snapshots WHERE fixture_date=?", (fixture_date,)
            ).fetchall()
    result: dict[str, dict[str, Any]] = {}
    for fixture_id, payload in rows:
        try:
            result[str(fixture_id)] = json.loads(payload)
        except json.JSONDecodeError:
            log.warning("Snapshot corrupto fixture=%s", fixture_id)
    return result


def db_save_snapshot(fixture_date: str, fixture_id: str, cutoff: str, payload: dict[str, Any]) -> None:
    values = (fixture_id, fixture_date, cutoff, MODEL_VERSION,
              json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
              datetime.now(UTC).isoformat())
    statement = """INSERT INTO snapshots(fixture_id, fixture_date, cutoff_utc, model_version, payload, created_at)
       VALUES({placeholders}) ON CONFLICT(fixture_id) DO UPDATE SET
       fixture_date=excluded.fixture_date, cutoff_utc=excluded.cutoff_utc,
       model_version=excluded.model_version, payload=excluded.payload,
       created_at=excluded.created_at"""
    if DATABASE_URL:
        with psycopg.connect(DATABASE_URL, connect_timeout=15, prepare_threshold=None) as db:
            db.execute(statement.format(placeholders="%s,%s,%s,%s,%s,%s"), values)
    else:
        with closing(sqlite3.connect(STATE_DB_PATH)) as db:
            db.execute(statement.format(placeholders="?,?,?,?,?,?"), values)
            db.commit()


def db_latest_calibration() -> dict[str, Any] | None:
    query = "SELECT id, created_at, status, promoted, payload FROM calibration_runs ORDER BY id DESC LIMIT 1"
    if DATABASE_URL:
        with psycopg.connect(DATABASE_URL, connect_timeout=15, prepare_threshold=None) as db:
            row = db.execute(query).fetchone()
    else:
        with closing(sqlite3.connect(STATE_DB_PATH)) as db:
            row = db.execute(query).fetchone()
    if not row:
        return None
    try:
        payload = json.loads(row[4])
    except json.JSONDecodeError:
        return {"run_id": row[0], "created_at": row[1], "status": "CORRUPT", "promoted": False}
    return {**payload, "run_id": row[0], "created_at": row[1], "status": row[2], "promoted": bool(row[3])}


def db_save_calibration(report: dict[str, Any]) -> int:
    created_at = datetime.now(UTC).isoformat()
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    if DATABASE_URL:
        with psycopg.connect(DATABASE_URL, connect_timeout=15, prepare_threshold=None) as db:
            row = db.execute(
                "INSERT INTO calibration_runs(created_at,status,promoted,payload) VALUES(%s,%s,0,%s) RETURNING id",
                (created_at, report["status"], payload),
            ).fetchone()
            return int(row[0])
    with closing(sqlite3.connect(STATE_DB_PATH)) as db:
        cursor = db.execute(
            "INSERT INTO calibration_runs(created_at,status,promoted,payload) VALUES(?,?,0,?)",
            (created_at, report["status"], payload),
        )
        db.commit()
        return int(cursor.lastrowid)


def db_checkpoint_calibration_job(job_id: str, status: str, phase: str, done: int, total: int, payload: dict[str, Any]) -> None:
    values = (job_id, status, phase, int(done), max(int(total), 1),
              json.dumps(payload, ensure_ascii=False, separators=(",", ":")), datetime.now(UTC).isoformat())
    statement = """INSERT INTO calibration_jobs(job_id,status,phase,done,total,payload,updated_at)
       VALUES({placeholders}) ON CONFLICT(job_id) DO UPDATE SET status=excluded.status,
       phase=excluded.phase,done=excluded.done,total=excluded.total,payload=excluded.payload,updated_at=excluded.updated_at"""
    if DATABASE_URL:
        with psycopg.connect(DATABASE_URL, connect_timeout=15, prepare_threshold=None) as db:
            db.execute(statement.format(placeholders="%s,%s,%s,%s,%s,%s,%s"), values)
    else:
        with closing(sqlite3.connect(STATE_DB_PATH)) as db:
            db.execute(statement.format(placeholders="?,?,?,?,?,?,?"), values)
            db.commit()


def db_latest_calibration_job() -> dict[str, Any] | None:
    query = "SELECT job_id,status,phase,done,total,payload,updated_at FROM calibration_jobs ORDER BY updated_at DESC LIMIT 1"
    if DATABASE_URL:
        with psycopg.connect(DATABASE_URL, connect_timeout=15, prepare_threshold=None) as db:
            row = db.execute(query).fetchone()
    else:
        with closing(sqlite3.connect(STATE_DB_PATH)) as db:
            row = db.execute(query).fetchone()
    if not row:
        return None
    try:
        payload = json.loads(row[5])
    except (json.JSONDecodeError, TypeError):
        payload = {}
    return {"job_id": row[0], "status": row[1], "phase": row[2], "done": row[3], "total": row[4],
            "payload": payload, "updated_at": row[6]}


def db_load_advanced_stats(fixture_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not fixture_ids:
        return {}
    result: dict[int, dict[str, Any]] = {}
    for batch in chunks(fixture_ids, 500):
        if DATABASE_URL:
            with psycopg.connect(DATABASE_URL, connect_timeout=15, prepare_threshold=None) as db:
                rows = db.execute(
                    "SELECT fixture_id,payload FROM advanced_fixture_stats WHERE fixture_id = ANY(%s)",
                    (batch,),
                ).fetchall()
        else:
            placeholders = ",".join("?" for _ in batch)
            with closing(sqlite3.connect(STATE_DB_PATH)) as db:
                rows = db.execute(
                    f"SELECT fixture_id,payload FROM advanced_fixture_stats WHERE fixture_id IN ({placeholders})",
                    batch,
                ).fetchall()
        for fixture_id, payload in rows:
            try:
                result[int(fixture_id)] = json.loads(payload)
            except json.JSONDecodeError:
                log.warning("Estadística avanzada corrupta fixture=%s", fixture_id)
    return result


def db_save_advanced_stats(rows: dict[int, dict[str, Any]]) -> None:
    if not rows:
        return
    created_at = datetime.now(UTC).isoformat()
    values = [(fixture_id, json.dumps(payload, ensure_ascii=False, separators=(",", ":")), created_at)
              for fixture_id, payload in rows.items()]
    if DATABASE_URL:
        with psycopg.connect(DATABASE_URL, connect_timeout=15, prepare_threshold=None) as db:
            with db.cursor() as cursor:
                cursor.executemany(
                    """INSERT INTO advanced_fixture_stats(fixture_id,payload,created_at) VALUES(%s,%s,%s)
                       ON CONFLICT(fixture_id) DO UPDATE SET payload=excluded.payload,created_at=excluded.created_at""",
                    values,
                )
    else:
        with closing(sqlite3.connect(STATE_DB_PATH)) as db:
            db.executemany(
                """INSERT INTO advanced_fixture_stats(fixture_id,payload,created_at) VALUES(?,?,?)
                   ON CONFLICT(fixture_id) DO UPDATE SET payload=excluded.payload,created_at=excluded.created_at""",
                values,
            )
            db.commit()


def db_resolve_outcomes(catalog: list[dict[str, Any]], snapshots: dict[str, dict[str, Any]]) -> int:
    values = []
    for fixture in catalog:
        shell = base_shell(fixture)
        snapshot = snapshots.get(shell["fixture_id"])
        goals = fixture.get("goals") or {}
        if not snapshot or snapshot.get("analysis_status") != "READY" or not shell["is_finished"]:
            continue
        if goals.get("home") is None or goals.get("away") is None:
            continue
        kickoff = parse_dt(shell.get("kickoff_utc"))
        cutoff = parse_dt(snapshot.get("snapshot_cutoff_utc"))
        if not kickoff or not cutoff or cutoff >= kickoff:
            log.warning("Outcome excluido por integridad temporal fixture=%s cutoff=%s kickoff=%s", shell["fixture_id"], cutoff, kickoff)
            continue
        actual_home, actual_away = int(goals["home"]), int(goals["away"])
        actual = 0 if actual_home > actual_away else 1 if actual_home == actual_away else 2
        probabilities = [float(snapshot.get(key) or 0) / 100 for key in ("p_home", "p_draw", "p_away")]
        total_probability = sum(probabilities)
        if total_probability <= 0:
            continue
        probabilities = [value / total_probability for value in probabilities]
        log_loss = -math.log(max(probabilities[actual], 1e-12))
        brier = sum((probability - (1.0 if index == actual else 0.0)) ** 2
                    for index, probability in enumerate(probabilities))
        goals_metric = ((snapshot.get("metrics") or {}).get("goals_2_5") or {})
        direction_probability = as_float(goals_metric.get("probability"))
        probability_over = None
        brier_over = None
        if direction_probability is not None:
            probability_over = direction_probability if goals_metric.get("direction") == "OVER" else 1 - direction_probability
            brier_over = (probability_over - (1.0 if actual_home + actual_away >= 3 else 0.0)) ** 2
        metrics = {
            "log_loss_1x2": round(log_loss, 8), "brier_1x2": round(brier, 8),
            "correct_1x2": int(max(range(3), key=lambda index: probabilities[index]) == actual),
            "modal_correct": int(snapshot.get("marcador_estimado") == f"{actual_home} - {actual_away}"),
            "brier_over25": round(brier_over, 8) if brier_over is not None else None,
            "probability_over25": round(probability_over, 8) if probability_over is not None else None,
            "baseline_log_loss_1x2": round(-math.log((0.45, 0.28, 0.27)[actual]), 8),
            "baseline_brier_1x2": round(sum((p - (1.0 if i == actual else 0.0)) ** 2 for i, p in enumerate((0.45, 0.28, 0.27))), 8),
            "baseline_brier_over25": round((0.5 - (1.0 if actual_home + actual_away >= 3 else 0.0)) ** 2, 8),
            "actual_1x2": actual,
        }
        values.append((
            shell["fixture_id"], shell.get("kickoff_utc"), snapshot.get("model_version") or MODEL_VERSION,
            json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")), actual_home, actual_away,
            json.dumps(metrics, separators=(",", ":")), datetime.now(UTC).isoformat(),
        ))
    if not values:
        return 0
    if DATABASE_URL:
        with psycopg.connect(DATABASE_URL, connect_timeout=15, prepare_threshold=None) as db:
            with db.cursor() as cursor:
                cursor.executemany(
                    """INSERT INTO prediction_outcomes
                       (fixture_id,kickoff_utc,model_version,prediction,actual_home,actual_away,metrics,resolved_at)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(fixture_id) DO NOTHING""", values)
    else:
        with closing(sqlite3.connect(STATE_DB_PATH)) as db:
            db.executemany(
                """INSERT INTO prediction_outcomes
                   (fixture_id,kickoff_utc,model_version,prediction,actual_home,actual_away,metrics,resolved_at)
                   VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(fixture_id) DO NOTHING""", values)
            db.commit()
    return len(values)


def db_model_scorecard() -> dict[str, Any]:
    query = "SELECT kickoff_utc,prediction,metrics,resolved_at,actual_home,actual_away FROM prediction_outcomes ORDER BY resolved_at DESC LIMIT 5000"
    if DATABASE_URL:
        with psycopg.connect(DATABASE_URL, connect_timeout=15, prepare_threshold=None) as db:
            rows = db.execute(query).fetchall()
    else:
        with closing(sqlite3.connect(STATE_DB_PATH)) as db:
            rows = db.execute(query).fetchall()
    records: list[dict[str, Any]] = []
    for row in rows:
        try:
            records.append({"kickoff": row[0], "prediction": json.loads(row[1]), "metrics": json.loads(row[2]),
                            "resolved_at": row[3], "actual_home": int(row[4]), "actual_away": int(row[5])})
        except (json.JSONDecodeError, TypeError):
            continue
    if not records:
        return {"status": "INSUFFICIENT", "resolved_predictions": 0, "minimum_for_reporting": 30}

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        metric_rows = [record["metrics"] for record in selected]
        def average(field: str) -> float | None:
            values = [float(row[field]) for row in metric_rows if row.get(field) is not None]
            return round(sum(values) / len(values), 6) if values else None
        n = len(metric_rows)
        brier = average("brier_1x2")
        baseline_brier = average("baseline_brier_1x2")
        over_brier = average("brier_over25")
        baseline_over = average("baseline_brier_over25")
        return {
            "n": n,
            "log_loss_1x2": average("log_loss_1x2"), "brier_1x2": brier,
            "accuracy_1x2": average("correct_1x2"), "brier_over25": over_brier,
            "baseline_brier_1x2": baseline_brier, "baseline_brier_over25": baseline_over,
            "brier_improvement_pct": round((baseline_brier - brier) / baseline_brier * 100, 4) if brier is not None and baseline_brier else None,
            "over25_brier_improvement_pct": round((baseline_over - over_brier) / baseline_over * 100, 4) if over_brier is not None and baseline_over else None,
        }

    def calibration_bins(selected: list[dict[str, Any]], market: str) -> list[dict[str, Any]]:
        buckets: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for record in selected:
            prediction, metric = record["prediction"], record["metrics"]
            if market == "over25":
                probability = as_float(metric.get("probability_over25"))
                actual = float(record["actual_home"] + record["actual_away"] >= 3) if probability is not None else None
            else:
                probs = [as_float(prediction.get(key), 0.0) / 100 for key in ("p_home", "p_draw", "p_away")]
                probability = max(probs)
                actual = float(metric.get("correct_1x2", 0))
            if probability is None or actual is None:
                continue
            bucket = min(int(probability * 10), 9)
            buckets[bucket].append((probability, actual))
        return [{"from": bucket / 10, "to": (bucket + 1) / 10, "n": len(values),
                 "mean_probability": round(sum(v[0] for v in values) / len(values), 4),
                 "observed_frequency": round(sum(v[1] for v in values) / len(values), 4)}
                for bucket, values in sorted(buckets.items())]

    now = datetime.now(UTC)
    horizons: dict[str, Any] = {}
    for label, days in (("7d", 7), ("30d", 30), ("90d", 90)):
        selected = [record for record in records if (parsed := parse_dt(record["kickoff"])) and parsed >= now - timedelta(days=days)]
        horizons[label] = summarize(selected)
    leagues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        prediction = record["prediction"]
        leagues[f'{prediction.get("league_id", 0)}:{prediction.get("league_name") or prediction.get("liga") or "N/D"}'].append(record)
    by_league = {key: summarize(value) for key, value in leagues.items()}
    overall = summarize(records)
    n = len(records)
    promotion = {
        "1x2": {"eligible": n >= 300 and (overall.get("brier_improvement_pct") or -999) > 0,
                "requirements": {"minimum_n": 300, "positive_brier_improvement": True}},
        "goals_2_5": {"eligible": n >= 300 and (overall.get("over25_brier_improvement_pct") or -999) > 0,
                      "requirements": {"minimum_n": 300, "positive_brier_improvement": True}},
        "corners": {"eligible": False, "reason": "Requiere scorecard prequential independiente"},
        "cards": {"eligible": False, "reason": "Requiere scorecard prequential independiente"},
        "shots": {"eligible": False, "reason": "Requiere scorecard prequential independiente"},
    }
    return {
        "status": "PRELIMINARY" if n < 100 else "MONITORED",
        "resolved_predictions": n, "minimum_for_reporting": 30,
        "overall": overall, "horizons": horizons, "by_league": by_league,
        "calibration_bins_1x2": calibration_bins(records, "1x2"),
        "calibration_bins_over25": calibration_bins(records, "over25"),
        "promotion_gates": promotion,
        "log_loss_1x2": overall.get("log_loss_1x2"), "brier_1x2": overall.get("brier_1x2"),
        "accuracy_1x2": overall.get("accuracy_1x2"), "brier_over25": overall.get("brier_over25"),
        "modal_accuracy": round(sum(float(record["metrics"].get("modal_correct", 0)) for record in records) / n, 6),
        "warning": "Métrica prequential descriptiva; promoción nunca automática.",
    }


init_db()
_snapshots: dict[str, dict[str, Any]] = {}


async def provider_get(
    client: httpx.AsyncClient,
    path: str,
    params: dict[str, Any],
    attempts: int = 4,
) -> list[Any] | dict[str, Any]:
    if not API_KEY:
        raise RuntimeError("Falta API_FOOTBALL_KEY en Render")
    last_error = "Error desconocido"
    for attempt in range(attempts):
        try:
            await _limiter.acquire()
            async with _http_slots:
                response = await client.get(path, params=params)
            _quota["daily_remaining"] = response.headers.get("x-ratelimit-requests-remaining")
            _quota["minute_remaining"] = response.headers.get("x-ratelimit-remaining")
            _quota["last_call_at"] = datetime.now(UTC).isoformat()
            if response.status_code == 429:
                retry_after = int(as_float(response.headers.get("Retry-After"), 60.0) or 60)
                last_error = f"HTTP 429 en {path}"
                log.warning("%s; reintento en %ss", last_error, retry_after)
                await asyncio.sleep(max(5, retry_after))
                continue
            response.raise_for_status()
            payload = response.json()
            errors = payload.get("errors") or {}
            if errors:
                last_error = f"API-Football rechazó {path}: {errors}"
                if "rateLimit" in str(errors):
                    await asyncio.sleep(60)
                    continue
                raise RuntimeError(last_error)
            return payload.get("response", [])
        except (httpx.HTTPError, ValueError, RuntimeError) as exc:
            last_error = str(exc)
            if attempt + 1 < attempts:
                await asyncio.sleep(min(8.0, 0.75 * (2**attempt)))
    raise RuntimeError(last_error)


LIVE_CODES = {"1H", "HT", "2H", "ET", "BT", "P", "LIVE", "INT"}
FINISHED_CODES = {"FT", "AET", "PEN"}
STOPPED_CODES = {"PST", "CANC", "ABD", "AWD", "WO", "SUSP"}
SCHEDULED_CODES = {"NS", "TBD"}


def fixture_state(info: dict[str, Any]) -> dict[str, Any]:
    status = info.get("status") or {}
    official = str(status.get("short") or "TBD").upper()
    kickoff = parse_dt(info.get("date"))
    local = kickoff.astimezone(BOGOTA) if kickoff else None
    now = now_local()
    elapsed = int(as_float(status.get("elapsed"), 0.0) or 0)
    stale = bool(local and official in SCHEDULED_CODES and now > local + timedelta(minutes=STALE_NS_MINUTES))

    if official in LIVE_CODES:
        interpreted, label, group = "LIVE", f"EN VIVO · {elapsed}'", "LIVE"
    elif official in FINISHED_CODES:
        interpreted, label, group = "FINISHED", "FINALIZADO", "FINISHED"
    elif official in STOPPED_CODES:
        labels = {
            "PST": "APLAZADO", "CANC": "CANCELADO", "ABD": "ABANDONADO",
            "AWD": "DECISIÓN ADMINISTRATIVA", "WO": "WALKOVER", "SUSP": "SUSPENDIDO",
        }
        interpreted, label, group = official, labels.get(official, official), "STOPPED"
    elif stale:
        interpreted, label, group = "UNCONFIRMED", "ESTADO SIN CONFIRMAR", "UNCONFIRMED"
    else:
        interpreted, group = "SCHEDULED", "UPCOMING"
        prefix = "HOY" if local and local.date() == now.date() else local.strftime("%d/%m") if local else "FECHA PENDIENTE"
        label = f"{prefix} · {local.strftime('%I:%M %p')}" if local else prefix

    return {
        "official": official,
        "interpreted": interpreted,
        "group": group,
        "display": label,
        "is_live": group == "LIVE",
        "is_finished": group == "FINISHED",
        "is_upcoming": group == "UPCOMING",
        "kickoff_utc": kickoff.astimezone(UTC).isoformat() if kickoff else None,
        "kickoff_local": local.isoformat() if local else None,
        "timestamp": kickoff.timestamp() if kickoff else 0.0,
    }


def country_flag(country: str) -> str:
    code = {
        "argentina": "🇦🇷", "brazil": "🇧🇷", "brasil": "🇧🇷", "colombia": "🇨🇴",
        "spain": "🇪🇸", "england": "🏴", "mexico": "🇲🇽", "uruguay": "🇺🇾",
        "usa": "🇺🇸", "united states": "🇺🇸", "france": "🇫🇷", "germany": "🇩🇪",
        "italy": "🇮🇹", "portugal": "🇵🇹", "netherlands": "🇳🇱", "belgium": "🇧🇪",
        "chile": "🇨🇱", "peru": "🇵🇪", "ecuador": "🇪🇨", "paraguay": "🇵🇾",
        "bolivia": "🇧🇴", "venezuela": "🇻🇪", "costa rica": "🇨🇷", "panama": "🇵🇦",
        "honduras": "🇭🇳", "guatemala": "🇬🇹", "el salvador": "🇸🇻", "canada": "🇨🇦",
    }.get(country.strip().lower())
    return code or "🌐"


def stat_map(block: dict[str, Any]) -> dict[str, float | None]:
    values = {str(row.get("type")): row.get("value") for row in block.get("statistics", [])}
    yellow = as_float(values.get("Yellow Cards"))
    red = as_float(values.get("Red Cards"))
    cards = None if yellow is None and red is None else (yellow or 0.0) + (red or 0.0)
    return {
        "corners": as_float(values.get("Corner Kicks")),
        "cards": cards,
        "shots": as_float(values.get("Total Shots")),
        "shots_on": as_float(values.get("Shots on Goal")),
    }


async def preload_fixture_stats(client: httpx.AsyncClient, fixture_ids: set[int], overall_weight: float = 0.0) -> None:
    missing = sorted(fid for fid in fixture_ids if cache_get(f"fixture-stats:{fid}") is None)
    batches = chunks(missing, 20)
    if not batches:
        return
    _progress.update(phase="HISTORICAL_STATS", done=0, total=len(batches))
    batch_weight = overall_weight / len(batches)

    async def load(batch: list[int]) -> None:
        try:
            rows = await provider_get(client, "/fixtures", {"ids": "-".join(map(str, batch))})
            found: set[int] = set()
            for row in rows if isinstance(rows, list) else []:
                fixture_id = int((row.get("fixture") or {}).get("id") or 0)
                if not fixture_id:
                    continue
                found.add(fixture_id)
                parsed = {
                    int(block.get("team", {}).get("id")): stat_map(block)
                    for block in row.get("statistics") or [] if block.get("team", {}).get("id")
                }
                cache_put(f"fixture-stats:{fixture_id}", parsed, FIXTURE_STATS_TTL)
            for fixture_id in set(batch) - found:
                cache_put(f"fixture-stats:{fixture_id}", {}, FIXTURE_STATS_TTL)
        finally:
            _progress["done"] += 1
            _progress["overall_done"] += batch_weight

    results = await asyncio.gather(*(load(batch) for batch in batches), return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            log.warning("Lote de estadísticas no disponible: %s", result)


def parse_match(row: dict[str, Any], team_id: int) -> dict[str, Any] | None:
    info, teams, goals = row.get("fixture") or {}, row.get("teams") or {}, row.get("goals") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    is_home = int(home.get("id") or 0) == team_id
    opponent = away if is_home else home
    gf = goals.get("home" if is_home else "away")
    gc = goals.get("away" if is_home else "home")
    played = parse_dt(info.get("date"))
    fixture_id = int(info.get("id") or 0)
    if not fixture_id or not opponent.get("name") or gf is None or gc is None or not played:
        return None
    return {
        "fixture_id": fixture_id,
        "timestamp": played.timestamp(),
        "date": played.astimezone(BOGOTA).strftime("%d/%m"),
        "rival": str(opponent["name"]),
        "competition": str((row.get("league") or {}).get("name") or ""),
        "venue": "HOME" if is_home else "AWAY",
        "gf": int(gf), "gc": int(gc), "goals": int(gf) + int(gc),
        "score": f"{gf} - {gc}",
        "result": "V" if gf > gc else "E" if gf == gc else "D",
    }


def season_average(stats: dict[str, Any], side: str, venue: str, fallback: float | None = None) -> float | None:
    goals = stats.get("goals") or {}
    block = goals.get(side) or {}
    average = block.get("average") or {}
    return as_float(average.get(venue), fallback)


async def team_profile(
    client: httpx.AsyncClient,
    team_id: int,
    league_id: int,
    season: int,
    cutoff: datetime,
) -> dict[str, Any]:
    cutoff_date = (cutoff.astimezone(BOGOTA).date() - timedelta(days=1)).isoformat()
    key = f"team:{team_id}:{league_id}:{season}:{cutoff_date}"
    if (hit := cache_get(key)) is not None:
        return hit
    async with _key_locks[key]:
        if (hit := cache_get(key)) is not None:
            return hit
        # API-Football rechaza actualmente la combinación `last` + `status`.
        # Solicitamos una ventana mayor y conservamos localmente solo finalizados
        # anteriores al kickoff, hasta completar HISTORY_SIZE observaciones.
        # Forma reciente coherente con la competición objetivo. Mezclar copas o
        # amistosos con medias de liga sesga la comparación interna.
        fixtures_task = provider_get(
            client, "/fixtures",
            {"team": team_id, "league": league_id, "season": season, "last": min(HISTORY_SIZE * 2, 40)},
        )
        stats_task = provider_get(
            client,
            "/teams/statistics",
            {"team": team_id, "league": league_id, "season": season, "date": cutoff_date},
        )
        fixtures, season_stats = await asyncio.gather(fixtures_task, stats_task)
        season_stats = season_stats if isinstance(season_stats, dict) else {}
        matches = [
            parsed
            for row in fixtures
            if fixture_state((row.get("fixture") or {}))["group"] == "FINISHED"
            and (parsed := parse_match(row, team_id))
        ]
        matches = sorted((m for m in matches if m["timestamp"] < cutoff.timestamp()), key=lambda m: m["timestamp"], reverse=True)
        played_block = (season_stats.get("fixtures") or {}).get("played") or {}
        result = {
            "team_id": team_id,
            "matches": matches[:HISTORY_SIZE],
            "played_total": int(as_float(played_block.get("total"), 0.0) or 0),
            "played_home": int(as_float(played_block.get("home"), 0.0) or 0),
            "played_away": int(as_float(played_block.get("away"), 0.0) or 0),
            "gf_total": season_average(season_stats, "for", "total"),
            "gc_total": season_average(season_stats, "against", "total"),
            "gf_home": season_average(season_stats, "for", "home"),
            "gc_home": season_average(season_stats, "against", "home"),
            "gf_away": season_average(season_stats, "for", "away"),
            "gc_away": season_average(season_stats, "against", "away"),
        }
        return cache_put(key, result, TEAM_TTL)


async def league_profile(client: httpx.AsyncClient, league_id: int, season: int, cutoff: datetime) -> dict[str, Any]:
    cutoff_date = (cutoff.astimezone(BOGOTA).date() - timedelta(days=1)).isoformat()
    key = f"league:{league_id}:{season}:{cutoff_date}"
    if (hit := cache_get(key)) is not None:
        return hit
    async with _key_locks[key]:
        if (hit := cache_get(key)) is not None:
            return hit
        rows = await provider_get(
            client,
            "/fixtures",
            {"league": league_id, "season": season, "status": "FT"},
        )
        # No combinamos `last` con `status`; API-Football los considera
        # incompatibles. El corte local impide fuga de información del propio día.
        eligible = []
        for row in rows if isinstance(rows, list) else []:
            played = parse_dt((row.get("fixture") or {}).get("date")) if isinstance(row, dict) else None
            if played and played < cutoff:
                eligible.append(row)
        eligible.sort(key=lambda row: parse_dt((row.get("fixture") or {}).get("date")).timestamp(), reverse=True)
        scores = [row.get("goals") or {} for row in eligible[:100]]
        valid = [(as_float(s.get("home")), as_float(s.get("away"))) for s in scores]
        valid = [(h, a) for h, a in valid if h is not None and a is not None]
        result = {
            "sample": len(valid),
            "home_rate": sum(h for h, _ in valid) / len(valid) if valid else None,
            "away_rate": sum(a for _, a in valid) / len(valid) if valid else None,
        }
        return cache_put(key, result, LEAGUE_TTL)


async def head_to_head(client: httpx.AsyncClient, home_id: int, away_id: int, cutoff: datetime) -> list[dict[str, Any]]:
    key = f"h2h:{min(home_id, away_id)}:{max(home_id, away_id)}:{cutoff.date()}"
    if (hit := cache_get(key)) is not None:
        return hit
    try:
        rows = await provider_get(client, "/fixtures/headtohead", {"h2h": f"{home_id}-{away_id}", "last": 5})
        result = []
        for row in rows if isinstance(rows, list) else []:
            info, teams, goals = row.get("fixture") or {}, row.get("teams") or {}, row.get("goals") or {}
            played = parse_dt(info.get("date"))
            if not played or played >= cutoff or goals.get("home") is None or goals.get("away") is None:
                continue
            result.append({
                "fecha": played.astimezone(BOGOTA).strftime("%d/%m/%Y"),
                "home": str((teams.get("home") or {}).get("name") or ""),
                "away": str((teams.get("away") or {}).get("name") or ""),
                "score": f"{goals['home']} - {goals['away']}",
            })
        return cache_put(key, result, H2H_TTL)
    except Exception as exc:
        log.warning("H2H no disponible %s-%s: %s", home_id, away_id, exc)
        return []


def poisson(k: int, rate: float) -> float:
    return math.exp(-rate) * rate**k / math.factorial(k)


def score_matrix(home_rate: float, away_rate: float, size: int = 11) -> list[list[float]]:
    matrix = [[poisson(h, home_rate) * poisson(a, away_rate) for a in range(size)] for h in range(size)]
    corrections = {
        (0, 0): 1 - home_rate * away_rate * DIXON_COLES_RHO,
        (0, 1): 1 + home_rate * DIXON_COLES_RHO,
        (1, 0): 1 + away_rate * DIXON_COLES_RHO,
        (1, 1): 1 - DIXON_COLES_RHO,
    }
    for (h, a), factor in corrections.items():
        matrix[h][a] *= max(factor, 1e-9)
    total = sum(map(sum, matrix))
    return [[value / total for value in row] for row in matrix]


def shrink(observed: float | None, baseline: float, matches: int) -> float:
    if observed is None or matches <= 0:
        return baseline
    weight = matches / (matches + SHRINKAGE_MATCHES)
    return weight * observed + (1 - weight) * baseline


def recent_rate(matches: list[dict[str, Any]], field: str) -> float | None:
    if not matches:
        return None
    weights = [0.82**index for index in range(len(matches))]
    return sum(float(match[field]) * weight for match, weight in zip(matches, weights)) / sum(weights)


def attach_advanced(matches: list[dict[str, Any]], team_id: int) -> None:
    for match in matches:
        blocks = cache_get(f"fixture-stats:{match['fixture_id']}") or {}
        own = blocks.get(team_id) or {}
        totals: dict[str, float | None] = {}
        for metric in ("corners", "cards", "shots", "shots_on"):
            values = [block.get(metric) for block in blocks.values() if block.get(metric) is not None]
            totals[metric] = sum(values) if values else None
        match.update({f"{metric}_team": own.get(metric) for metric in totals})
        match.update(totals)


def history(matches: list[dict[str, Any]], metric: str, line: float, direction: str) -> list[dict[str, Any]]:
    rows = []
    for match in matches:
        value = match.get(metric)
        if value is None:
            continue
        complies = value > line if direction == "OVER" else value < line
        rows.append({
            "rival": match["rival"], "score": match["score"], "resultado": match["result"],
            "valor_numerico": round(float(value), 2), "cumple": complies,
            "fecha": match["date"], "competicion": match["competition"], "dato_disponible": True,
        })
    return rows


def btts_history(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "rival": m["rival"], "score": m["score"], "resultado": m["result"],
        "valor_numerico": 1.0 if m["gf"] > 0 and m["gc"] > 0 else 0.0,
        "cumple": m["gf"] > 0 and m["gc"] > 0, "fecha": m["date"],
        "competicion": m["competition"], "dato_disponible": True,
    } for m in matches]


def metric_evidence(home: list[dict[str, Any]], away: list[dict[str, Any]], metric: str, line: float) -> dict[str, Any]:
    home_values = [float(m[metric]) for m in home if m.get(metric) is not None]
    away_values = [float(m[metric]) for m in away if m.get(metric) is not None]
    team_field = f"{metric}_team"
    home_own = [float(m[team_field]) for m in home if m.get(team_field) is not None and m.get(metric) is not None]
    away_own = [float(m[team_field]) for m in away if m.get(team_field) is not None and m.get(metric) is not None]
    home_allowed = [float(m[metric]) - float(m[team_field]) for m in home if m.get(team_field) is not None and m.get(metric) is not None]
    away_allowed = [float(m[metric]) - float(m[team_field]) for m in away if m.get(team_field) is not None and m.get(metric) is not None]
    if min(len(home_values), len(away_values), len(home_own), len(away_own)) < 3:
        return {"available": False, "reason": "Cobertura oficial insuficiente", "sample": len(home_values) + len(away_values)}
    combined = home_values + away_values
    league_team_prior = sum(combined) / (2 * len(combined))

    def venue_or_all(rows: list[dict[str, Any]], values: list[float], venue: str, allowed: bool = False) -> list[float]:
        selected = []
        for row in rows:
            if row.get("venue") != venue or row.get(team_field) is None or row.get(metric) is None:
                continue
            selected.append(float(row[metric]) - float(row[team_field]) if allowed else float(row[team_field]))
        return selected if len(selected) >= 3 else values

    def regularized(values: list[float]) -> float:
        observed = sum(values) / len(values)
        weight = len(values) / (len(values) + SHRINKAGE_MATCHES)
        return weight * observed + (1 - weight) * league_team_prior

    # Producción propia y concesión rival se estiman por separado. La muestra
    # casa/fuera se usa cuando alcanza el mínimo; en otro caso se retrocede a
    # todos los partidos, siempre anteriores al kickoff.
    expected_home = (regularized(venue_or_all(home, home_own, "HOME")) + regularized(venue_or_all(away, away_allowed, "AWAY", True))) / 2
    expected_away = (regularized(venue_or_all(away, away_own, "AWAY")) + regularized(venue_or_all(home, home_allowed, "HOME", True))) / 2
    projection = max(0.05, expected_home + expected_away)
    sample_mean = sum(combined) / len(combined)
    variance = sum((value - sample_mean) ** 2 for value in combined) / max(len(combined) - 1, 1)
    dispersion = sample_mean * sample_mean / (variance - sample_mean) if variance > sample_mean + 0.01 else None

    def probability_at_most(limit: int) -> float:
        if dispersion is None:
            probability = math.exp(-projection)
            total = probability
            for count in range(1, limit + 1):
                probability *= projection / count
                total += probability
            return clamp(total, 0.0, 1.0)
        size = clamp(dispersion, 0.25, 1000.0)
        success = size / (size + projection)
        probability = success**size
        total = probability
        for count in range(1, limit + 1):
            probability *= ((count - 1 + size) / count) * (1 - success)
            total += probability
        return clamp(total, 0.0, 1.0)

    over_probability = 1.0 - probability_at_most(math.floor(line))
    direction = "OVER" if over_probability >= 0.5 else "UNDER"
    frequency = over_probability if direction == "OVER" else 1 - over_probability
    return {
        "available": True, "line": line, "direction": direction,
        "projection": round(projection, 2), "frequency": round(frequency, 4),
        "probability_over": round(over_probability, 6),
        "expected_home": round(expected_home, 3), "expected_away": round(expected_away, 3),
        "distribution": "NEGATIVE_BINOMIAL" if dispersion is not None else "POISSON",
        "dispersion": round(dispersion, 4) if dispersion is not None else None,
        "validation_status": "EXPERIMENTAL_NOT_WALK_FORWARD_VALIDATED",
        "home_frequency": round(sum((v > line) if direction == "OVER" else (v < line) for v in home_values) / len(home_values), 4),
        "away_frequency": round(sum((v > line) if direction == "OVER" else (v < line) for v in away_values) / len(away_values), 4),
        "sample": len(combined),
    }


def evidence_quality(home: dict[str, Any], away: dict[str, Any], league: dict[str, Any], advanced_ratio: float) -> dict[str, Any]:
    history_n = min(len(home["matches"]), len(away["matches"]))
    season_n = min(home["played_total"], away["played_total"])
    history_score = min(history_n / HISTORY_SIZE, 1.0)
    season_score = min(season_n / 20.0, 1.0)
    league_score = min(int(league.get("sample") or 0) / 60.0, 1.0)
    score = round(100 * (0.35 * history_score + 0.35 * season_score + 0.20 * league_score + 0.10 * advanced_ratio))
    label = "HIGH" if score >= 80 else "MODERATE" if score >= 60 else "LIMITED" if score >= 40 else "INSUFFICIENT"
    return {
        "score": score, "label": label, "history_per_team": history_n,
        "season_matches_min": season_n, "league_matches": int(league.get("sample") or 0),
        "advanced_coverage": round(advanced_ratio, 3),
    }


def base_shell(fixture: dict[str, Any]) -> dict[str, Any]:
    info, league, teams = fixture.get("fixture") or {}, fixture.get("league") or {}, fixture.get("teams") or {}
    home, away = teams.get("home") or {}, teams.get("away") or {}
    state = fixture_state(info)
    country = str(league.get("country") or "Internacional")
    flag = country_flag(country)
    goals = fixture.get("goals") or {}
    score_real = None if goals.get("home") is None or goals.get("away") is None else f"{goals['home']} - {goals['away']}"
    unavailable = state["group"] in {"LIVE", "FINISHED", "STOPPED", "UNCONFIRMED"}
    message = {
        "LIVE": "Partido iniciado sin snapshot prematch persistido",
        "FINISHED": "Partido finalizado",
        "STOPPED": state["display"].title(),
        "UNCONFIRMED": "La hora pasó pero el proveedor aún no confirma el estado",
    }.get(state["group"], "Esperando análisis estadístico")
    return {
        "contract_version": CONTRACT_VERSION, "id": str(info.get("id") or ""),
        "fixture_id": str(info.get("id") or ""), "league_id": int(league.get("id") or 0),
        "season": int(league.get("season") or now_local().year), "league_name": str(league.get("name") or "Liga"),
        "country": country, "flag": flag, "liga": f"{country} • {str(league.get('name') or 'Liga')}",
        "pais": country, "bandera": flag, "home_name": str(home.get("name") or "Equipo local"),
        "away_name": str(away.get("name") or "Equipo visitante"), "home_logo": str(home.get("logo") or ""),
        "away_logo": str(away.get("logo") or ""), "home_id": int(home.get("id") or 0), "away_id": int(away.get("id") or 0),
        "fecha": state["display"], "kickoff_utc": state["kickoff_utc"], "kickoff_local": state["kickoff_local"],
        "timezone": "America/Bogota", "official_status": state["official"], "interpreted_status": state["interpreted"],
        "status_group": state["group"], "status_code": state["interpreted"], "status_display": state["display"],
        "is_live": state["is_live"], "is_finished": state["is_finished"], "is_upcoming": state["is_upcoming"],
        "score_real": score_real, "last_status_update": datetime.now(UTC).isoformat(),
        "analysis_status": "UNAVAILABLE" if unavailable else "PENDING", "analysis_message": message,
        "status_verdict": "SIN_SNAPSHOT_PREVIO" if unavailable else "ANALISIS_PENDIENTE",
        "model_version": None, "model_calibrated": False, "snapshot_cutoff_utc": None,
        "probabilities": None, "p_home": 0, "p_draw": 0, "p_away": 0, "prob_1x2": "N/D",
        "expected_goals": None, "marcador_estimado": "—", "scoreline_top": [],
        "evidence_quality": {"score": 0, "label": "INSUFFICIENT"}, "data_quality": 0, "sample_size": 0,
        "confidence": None, "viability": None, "viability_status": "NOT_CERTIFIED",
        "abstention_reasons": [], "metrics": {}, "h2h": [],
        "mercado": "ANÁLISIS ESTADÍSTICO", "cr_mercado": "N/D", "cr_score_num": "0",
        "cr_home_casa": "N/D", "cr_away_fora": "N/D", "cr_combinado_split": "N/D", "proyeccion_val": "N/D",
        "home_goles": [], "away_goles": [], "home_corners": [], "away_corners": [],
        "home_tarjetas": [], "away_tarjetas": [], "home_remates": [], "away_remates": [],
        "home_btts": [], "away_btts": [], "split_vs_list": [], "h2h_matches": [],
        "home_matches_20": [], "away_matches_20": [],
        "corners_label": "DATOS NO DISPONIBLES", "corners_conf": 0, "corners_proyeccion": "N/D",
        "tarjetas_label": "DATOS NO DISPONIBLES", "tarjetas_conf": 0, "tarjetas_proyeccion": "N/D",
        "disparos_label": "DATOS NO DISPONIBLES", "disparos_conf": 0, "disparos_proyeccion": "N/D",
        "btts_label": "PROBABILIDAD AMBOS ANOTAN", "btts_conf": 0, "btts_proyeccion": "N/D",
        "cr_home_l10": "N/D", "cr_away_l10": "N/D", "cr_combinado_l10": "N/D",
        "metrics_home": {}, "metrics_away": {}, "_sort": {"LIVE": 0, "UPCOMING": 1, "UNCONFIRMED": 2, "STOPPED": 3, "FINISHED": 4}.get(state["group"], 5),
        "_timestamp": state["timestamp"],
    }


async def build_analysis(client: httpx.AsyncClient, fixture: dict[str, Any]) -> dict[str, Any]:
    shell = base_shell(fixture)
    cutoff = parse_dt(shell["kickoff_utc"])
    if not cutoff or datetime.now(UTC) >= cutoff:
        raise RuntimeError("La hora de inicio ya pasó; no se fabricará un snapshot retrospectivo")
    if not all((shell["fixture_id"], shell["league_id"], shell["home_id"], shell["away_id"])):
        raise RuntimeError("Fixture sin identificadores obligatorios")

    home, away, league = await asyncio.gather(
        team_profile(client, shell["home_id"], shell["league_id"], shell["season"], cutoff),
        team_profile(client, shell["away_id"], shell["league_id"], shell["season"], cutoff),
        league_profile(client, shell["league_id"], shell["season"], cutoff),
    )
    reasons = []
    if len(home["matches"]) < MIN_HISTORY:
        reasons.append(f"Historial local insuficiente: {len(home['matches'])}/{MIN_HISTORY}")
    if len(away["matches"]) < MIN_HISTORY:
        reasons.append(f"Historial visitante insuficiente: {len(away['matches'])}/{MIN_HISTORY}")
    if home["played_total"] < MIN_SEASON_PLAYED or away["played_total"] < MIN_SEASON_PLAYED:
        reasons.append("Temporada con muestra insuficiente")
    if int(league.get("sample") or 0) < 10 or league.get("home_rate") is None or league.get("away_rate") is None:
        reasons.append("Media de liga insuficiente")
    required = (home["gf_home"], home["gc_home"], away["gf_away"], away["gc_away"])
    if any(value is None for value in required):
        reasons.append("Promedios casa/fuera incompletos")
    if reasons:
        return {**shell, "analysis_status": "ABSTAINED", "analysis_message": "; ".join(reasons), "abstention_reasons": reasons, "status_verdict": "EVIDENCIA_INSUFICIENTE"}

    attach_advanced(home["matches"], shell["home_id"])
    attach_advanced(away["matches"], shell["away_id"])
    base_h = clamp(float(league["home_rate"]), 0.4, 3.5)
    base_a = clamp(float(league["away_rate"]), 0.3, 3.2)
    home_attack = shrink(home["gf_home"], base_h, home["played_home"]) / base_h
    away_defence = shrink(away["gc_away"], base_h, away["played_away"]) / base_h
    away_attack = shrink(away["gf_away"], base_a, away["played_away"]) / base_a
    home_defence = shrink(home["gc_home"], base_a, home["played_home"]) / base_a
    recent_h = recent_rate(home["matches"], "gf")
    recent_a = recent_rate(away["matches"], "gf")
    recency_h = clamp((recent_h / max(home["gf_total"] or base_h, 0.2)) if recent_h is not None else 1.0, 0.85, 1.15)
    recency_a = clamp((recent_a / max(away["gf_total"] or base_a, 0.2)) if recent_a is not None else 1.0, 0.85, 1.15)
    lambda_home = clamp(base_h * home_attack * away_defence * (recency_h**RECENCY_STRENGTH), 0.15, 4.5)
    lambda_away = clamp(base_a * away_attack * home_defence * (recency_a**RECENCY_STRENGTH), 0.15, 4.5)
    matrix = score_matrix(lambda_home, lambda_away)
    p_home = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[h])) if h > a)
    p_draw = sum(matrix[n][n] for n in range(len(matrix)))
    p_away = max(0.0, 1.0 - p_home - p_draw)
    top = sorted(((matrix[h][a], h, a) for h in range(len(matrix)) for a in range(len(matrix[h]))), reverse=True)[:5]
    modal = top[0]
    pcts = [round(p_home * 100), round(p_draw * 100)]
    pcts.append(100 - sum(pcts))

    over25 = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[h])) if h + a >= 3)
    goals_direction = "OVER" if over25 >= 0.5 else "UNDER"
    goals_probability = over25 if goals_direction == "OVER" else 1 - over25
    goals_line = 2.5
    home_goal_hist = history(home["matches"], "goals", goals_line, goals_direction)
    away_goal_hist = history(away["matches"], "goals", goals_line, goals_direction)
    goal_hist = home_goal_hist + away_goal_hist
    empirical_direction_rate = sum(bool(row.get("cumple")) for row in goal_hist) / len(goal_hist)
    sampling_margin = 2.0 * math.sqrt(max(empirical_direction_rate * (1 - empirical_direction_rate), 0.01) / len(goal_hist))
    coherence_limit = max(MAX_GOALS_HISTORY_GAP, sampling_margin)
    coherence_gap = abs(goals_probability - empirical_direction_rate)
    if coherence_gap > coherence_limit:
        reason = (
            f"Modelo e historial no convergen: probabilidad={goals_probability:.1%}, "
            f"frecuencia={empirical_direction_rate:.1%}, límite={coherence_limit:.1%}"
        )
        return {**shell, "analysis_status": "ABSTAINED", "analysis_message": reason,
                "abstention_reasons": [reason], "status_verdict": "INCOHERENCIA_MODELO_HISTORIAL",
                "data_quality": 0, "sample_size": len(goal_hist),
                "coherence": {"probability": round(goals_probability, 6),
                              "historical_frequency": round(empirical_direction_rate, 6),
                              "gap": round(coherence_gap, 6), "limit": round(coherence_limit, 6)}}
    corners = metric_evidence(home["matches"], away["matches"], "corners", 8.5)
    cards = metric_evidence(home["matches"], away["matches"], "cards", 4.5)
    shots = metric_evidence(home["matches"], away["matches"], "shots", 20.5)
    advanced_ratio = sum(metric["available"] for metric in (corners, cards, shots)) / 3
    quality = evidence_quality(home, away, league, advanced_ratio)
    if quality["score"] < MIN_EVIDENCE_QUALITY:
        reason = f"Calidad de evidencia insuficiente: {quality['score']}/{MIN_EVIDENCE_QUALITY}"
        return {**shell, "analysis_status": "ABSTAINED", "analysis_message": reason,
                "abstention_reasons": [reason], "status_verdict": "EVIDENCIA_INSUFICIENTE",
                "evidence_quality": quality, "data_quality": quality["score"],
                "sample_size": len(home["matches"]) + len(away["matches"])}
    h2h = await head_to_head(client, shell["home_id"], shell["away_id"], cutoff)
    btts = sum(matrix[h][a] for h in range(1, len(matrix)) for a in range(1, len(matrix[h])))

    def metric_label(name: str, evidence: dict[str, Any]) -> str:
        if not evidence["available"]:
            return "DATOS NO DISPONIBLES"
        direction = "MÁS DE" if evidence["direction"] == "OVER" else "MENOS DE"
        return f"TENDENCIA: {direction} {evidence['line']} {name}"

    def metric_hist(rows: list[dict[str, Any]], evidence: dict[str, Any], field: str) -> list[dict[str, Any]]:
        return history(rows, field, evidence["line"], evidence["direction"]) if evidence["available"] else []

    def legacy_team_average(rows: list[dict[str, Any]], field: str) -> float:
        """Numeric compatibility for the current Android non-null Float DTO."""
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        return round(sum(values) / len(values), 2) if values else 0.0

    snapshot = {
        **shell,
        "analysis_status": "READY", "analysis_message": "Estimación preliminar disponible",
        "status_verdict": "ANALISIS_LISTO", "model_version": MODEL_VERSION,
        "model_calibrated": False, "snapshot_cutoff_utc": datetime.now(UTC).isoformat(),
        "probabilities": {"home": round(p_home, 6), "draw": round(p_draw, 6), "away": round(p_away, 6), "sum": round(p_home + p_draw + p_away, 6)},
        "p_home": pcts[0], "p_draw": pcts[1], "p_away": pcts[2], "prob_1x2": f"{pcts[0]}% • {pcts[1]}% • {pcts[2]}%",
        "expected_goals": {"home": round(lambda_home, 4), "away": round(lambda_away, 4), "total": round(lambda_home + lambda_away, 4)},
        "marcador_estimado": f"{modal[1]} - {modal[2]}",
        "scoreline_top": [{"score": f"{h} - {a}", "probability": round(prob, 6)} for prob, h, a in top],
        "evidence_quality": quality, "data_quality": quality["score"],
        "sample_size": len(home["matches"]) + len(away["matches"]),
        "confidence": {
            "status": MODEL_VALIDATION_STATUS,
            "reason": "Evaluado internamente en ventanas cronológicas; no certificado ni promovido",
        },
        "viability": None, "viability_status": "NOT_CERTIFIED",
        "metrics": {
            "goals_2_5": {"available": True, "direction": goals_direction, "line": goals_line, "probability": round(goals_probability, 6), "sample": len(home_goal_hist) + len(away_goal_hist)},
            "corners_8_5": corners, "cards_4_5": cards, "shots_20_5": shots,
            "btts": {"available": True, "probability": round(btts, 6)},
        },
        "h2h": h2h,
        "mercado": f"TENDENCIA: {'MÁS DE' if goals_direction == 'OVER' else 'MENOS DE'} 2.5 GOLES",
        "cr_mercado": "NO CALIBRADO", "cr_score_num": "0",
        "cr_home_casa": "N/D", "cr_away_fora": "N/D", "cr_combinado_split": "N/D",
        "proyeccion_val": f"{lambda_home + lambda_away:.2f}",
        "home_goles": home_goal_hist, "away_goles": away_goal_hist,
        "home_corners": metric_hist(home["matches"], corners, "corners"), "away_corners": metric_hist(away["matches"], corners, "corners"),
        "home_tarjetas": metric_hist(home["matches"], cards, "cards"), "away_tarjetas": metric_hist(away["matches"], cards, "cards"),
        "home_remates": metric_hist(home["matches"], shots, "shots"), "away_remates": metric_hist(away["matches"], shots, "shots"),
        "home_btts": btts_history(home["matches"]), "away_btts": btts_history(away["matches"]),
        "home_matches_20": home_goal_hist, "away_matches_20": away_goal_hist,
        # El cliente v4 interpreta `cumple=false` como evidencia negativa. No
        # degradamos el H2H rico de v5 a ese contrato semánticamente incorrecto.
        "h2h_matches": [],
        "corners_label": metric_label("CÓRNERS", corners), "corners_conf": round(corners.get("frequency", 0) * 100), "corners_proyeccion": str(corners.get("projection", "N/D")),
        "tarjetas_label": metric_label("TARJETAS", cards), "tarjetas_conf": round(cards.get("frequency", 0) * 100), "tarjetas_proyeccion": str(cards.get("projection", "N/D")),
        "disparos_label": metric_label("REMATES", shots), "disparos_conf": round(shots.get("frequency", 0) * 100), "disparos_proyeccion": str(shots.get("projection", "N/D")),
        "btts_conf": round(btts * 100), "btts_proyeccion": f"{lambda_home:.2f} - {lambda_away:.2f}",
        "cr_home_l10": "N/D", "cr_away_l10": "N/D", "cr_combinado_l10": "N/D",
        "metrics_home": {
            "gf_prom": round(float(home["gf_total"] or 0.0), 2),
            "gc_prom": round(float(home["gc_total"] or 0.0), 2),
            "corn_prom": legacy_team_average(home["matches"], "corners_team"),
            "tarj_prom": legacy_team_average(home["matches"], "cards_team"),
            "rem_prom": legacy_team_average(home["matches"], "shots_team"),
        },
        "metrics_away": {
            "gf_prom": round(float(away["gf_total"] or 0.0), 2),
            "gc_prom": round(float(away["gc_total"] or 0.0), 2),
            "corn_prom": legacy_team_average(away["matches"], "corners_team"),
            "tarj_prom": legacy_team_average(away["matches"], "cards_team"),
            "rem_prom": legacy_team_average(away["matches"], "shots_team"),
        },
    }
    return snapshot


def merged_props() -> list[dict[str, Any]]:
    rows = []
    for fixture in _catalog:
        shell = base_shell(fixture)
        snapshot = _snapshots.get(shell["fixture_id"])
        row = snapshot or shell
        if snapshot:
            live_fields = (
                "fecha", "official_status", "interpreted_status", "status_group", "status_code", "status_display",
                "is_live", "is_finished", "is_upcoming", "score_real", "last_status_update", "_sort", "_timestamp",
            )
            row = {**snapshot, **{key: shell[key] for key in live_fields}}
        if shell["fixture_id"] in _analysis_errors and not snapshot:
            message = _analysis_errors[shell["fixture_id"]]
            started_during_run = message == "La hora de inicio ya pasó; no se fabricará un snapshot retrospectivo"
            row = {
                **shell,
                "analysis_status": "UNAVAILABLE" if started_during_run else "ERROR",
                "analysis_message": message,
                "status_verdict": "INICIO_SUPERADO" if started_during_run else "ERROR_TECNICO",
                "abstention_reasons": [] if started_during_run else [message],
            }
        rows.append(row)
    rows.sort(key=lambda row: (row.get("_sort", 9), row.get("_timestamp", 0), row.get("league_name", "")))
    return [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]


async def analyze_catalog(fixtures: list[dict[str, Any]], fixture_date: str) -> None:
    global _snapshots
    candidates = []
    for fixture in fixtures:
        shell = base_shell(fixture)
        snapshot = _snapshots.get(shell["fixture_id"])
        outdated = bool(snapshot and snapshot.get("model_version") != MODEL_VERSION)
        if shell["is_upcoming"] and (snapshot is None or outdated):
            candidates.append(fixture)
    candidates.sort(key=lambda fixture: fixture_state(fixture.get("fixture") or {})["timestamp"])
    _progress.update(
        phase="TEAM_AND_LEAGUE_DATA", done=0, total=max(len(candidates), 1),
        overall_done=0.0, overall_total=float(max(len(candidates) * 3, 1)),
        started_at=datetime.now(UTC).isoformat(),
    )
    if not candidates:
        _progress.update(phase="COMPLETE", done=1, total=1)
        _progress.update(overall_done=1.0, overall_total=1.0)
        return
    timeout = httpx.Timeout(connect=10, read=35, write=10, pool=35)
    async with httpx.AsyncClient(base_url=BASE_URL, headers={"x-apisports-key": API_KEY}, timeout=timeout) as client:
        historical_ids: set[int] = set()

        async def warm(fixture: dict[str, Any]) -> None:
            try:
                shell = base_shell(fixture)
                cutoff = parse_dt(shell["kickoff_utc"])
                if not cutoff or datetime.now(UTC) >= cutoff:
                    return
                home, away = await asyncio.gather(
                    team_profile(client, shell["home_id"], shell["league_id"], shell["season"], cutoff),
                    team_profile(client, shell["away_id"], shell["league_id"], shell["season"], cutoff),
                )
                historical_ids.update(match["fixture_id"] for match in home["matches"] + away["matches"])
            except Exception as exc:
                log.warning("Precarga incompleta fixture=%s: %s", base_shell(fixture)["fixture_id"], exc)
            finally:
                _progress["done"] += 1
                _progress["overall_done"] += 1.0

        await asyncio.gather(*(warm(fixture) for fixture in candidates))
        await preload_fixture_stats(client, historical_ids, overall_weight=float(len(candidates)))
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        for fixture in candidates:
            queue.put_nowait(fixture)
        _progress.update(phase="PROBABILITY_MODEL", done=0, total=queue.qsize())

        async def worker() -> None:
            while True:
                try:
                    fixture = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                shell = base_shell(fixture)
                fixture_id = shell["fixture_id"]
                try:
                    result = await build_analysis(client, fixture)
                    if result["analysis_status"] in {"READY", "ABSTAINED"}:
                        _snapshots[fixture_id] = result
                        await asyncio.to_thread(db_save_snapshot, fixture_date, fixture_id, result.get("snapshot_cutoff_utc") or datetime.now(UTC).isoformat(), result)
                    _analysis_errors.pop(fixture_id, None)
                except Exception as exc:
                    _analysis_errors[fixture_id] = str(exc)
                    log.error("Análisis fallido fixture=%s: %s", fixture_id, exc)
                finally:
                    _progress["done"] += 1
                    _progress["overall_done"] += 1.0
                    queue.task_done()

        await asyncio.gather(*(worker() for _ in range(ANALYSIS_WORKERS)))
    _progress.update(phase="COMPLETE", done=_progress["total"], total=_progress["total"])
    _progress["overall_done"] = _progress["overall_total"]
    ready = sum(row.get("analysis_status") == "READY" for row in _snapshots.values())
    abstained = sum(row.get("analysis_status") == "ABSTAINED" for row in _snapshots.values())
    log.info("Día procesado: ready=%s abstained=%s errors=%s", ready, abstained, len(_analysis_errors))


async def load_catalog(fixture_date: str, force: bool = False) -> None:
    global _catalog, _catalog_date, _catalog_loaded_monotonic, _analysis_task, _snapshots
    async with _catalog_lock:
        fresh = _catalog and _catalog_date == fixture_date and time.monotonic() - _catalog_loaded_monotonic < CATALOG_TTL
        if fresh and not force:
            return
        timeout = httpx.Timeout(connect=10, read=35, write=10, pool=35)
        async with httpx.AsyncClient(base_url=BASE_URL, headers={"x-apisports-key": API_KEY}, timeout=timeout) as client:
            rows = await provider_get(client, "/fixtures", {"date": fixture_date, "timezone": "America/Bogota"})
        if not isinstance(rows, list):
            raise RuntimeError("API-Football no devolvió un catálogo válido")
        date_changed = _catalog_date != fixture_date
        _catalog = rows
        _catalog_date = fixture_date
        _catalog_loaded_monotonic = time.monotonic()
        if date_changed:
            _analysis_errors.clear()
            _snapshots = await asyncio.to_thread(db_load_snapshots, fixture_date)
        resolved = await asyncio.to_thread(db_resolve_outcomes, _catalog, _snapshots)
        if resolved:
            log.info("Journal verificado: %s predicciones resueltas", resolved)
        if _analysis_task is None or _analysis_task.done():
            _analysis_task = asyncio.create_task(analyze_catalog(_catalog.copy(), fixture_date))


def sync_payload() -> dict[str, Any]:
    props = merged_props()
    counts: defaultdict[str, int] = defaultdict(int)
    for row in props:
        counts[row["analysis_status"]] += 1
    running = bool(_analysis_task and not _analysis_task.done())
    phase_progress = _progress["done"] / max(_progress["total"], 1)
    overall_progress = _progress["overall_done"] / max(_progress["overall_total"], 1.0)
    return {
        "contract_version": CONTRACT_VERSION, "date": _catalog_date, "total": len(props),
        "eligible": sum(row["is_upcoming"] for row in props), "ready": counts["READY"],
        "abstained": counts["ABSTAINED"], "pending": counts["PENDING"],
        "unavailable": counts["UNAVAILABLE"], "failed": counts["ERROR"],
        "phase": _progress["phase"], "phase_done": _progress["done"], "phase_total": _progress["total"],
        "phase_progress": round(clamp(phase_progress, 0, 1), 4),
        "progress": round(clamp(overall_progress, 0, 1), 4), "running": running,
        "started_at": _progress["started_at"], "catalog_updated_at": datetime.now(UTC).isoformat(),
    }


async def execute_calibration(date: str, max_leagues: int, advanced: bool) -> None:
    global _calibration_state, _calibration_job_id
    _calibration_job_id = str(uuid.uuid4())
    _calibration_state = {
        "running": True, "started_at": datetime.now(UTC).isoformat(),
        "finished_at": None, "error": None, "run_id": None,
    }
    await asyncio.to_thread(db_checkpoint_calibration_job, _calibration_job_id, "RUNNING", "STARTING", 0, 1,
                            {"date": date, "max_leagues": max_leagues, "advanced": advanced})
    try:
        # Importación diferida: calibration reutiliza las primitivas verificadas
        # de este módulo sin crear un ciclo durante el arranque de Uvicorn.
        import calibration

        report = await calibration.run_calibration(date, max_leagues, advanced)
        _calibration_state["run_id"] = report.get("run_id")
        _calibration_state.update({"phase": "COMPLETE", "done": 1, "total": 1, "progress": 1.0})
        await asyncio.to_thread(db_checkpoint_calibration_job, _calibration_job_id, "COMPLETE", "COMPLETE", 1, 1,
                                {"date": date, "max_leagues": max_leagues, "advanced": advanced, "run_id": report.get("run_id")})
    except Exception as exc:
        log.exception("Calibración histórica fallida")
        _calibration_state["error"] = str(exc)
        await asyncio.to_thread(db_checkpoint_calibration_job, _calibration_job_id, "FAILED", _calibration_state.get("phase", "UNKNOWN"),
                                _calibration_state.get("done", 0), _calibration_state.get("total", 1),
                                {"date": date, "max_leagues": max_leagues, "advanced": advanced, "error": str(exc)})
    finally:
        _calibration_state["running"] = False
        _calibration_state["finished_at"] = datetime.now(UTC).isoformat()


def require_admin(token: str | None) -> None:
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="S2S_ADMIN_TOKEN no está configurado")
    # compare_digest reduce filtraciones temporales al validar el secreto.
    import hmac
    if not token or not hmac.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Token administrativo inválido")


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "status": "ok", "service": "S2S Sigma Engine", "version": ENGINE_VERSION,
        "contract_version": CONTRACT_VERSION, "model_version": MODEL_VERSION,
        "model_calibrated": False, "model_validation_status": MODEL_VALIDATION_STATUS,
        "api_key_configured": bool(API_KEY),
        "purpose": "sports_evidence_not_betting_advice",
        "foundation": {"founder": "Sebastián Betancourt"},
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok", "catalog_date": _catalog_date, "catalog_matches": len(_catalog),
        "analysis_running": bool(_analysis_task and not _analysis_task.done()), "quota": _quota,
        "database": DATABASE_BACKEND, "database_persistent": bool(DATABASE_URL), "snapshots_loaded": len(_snapshots),
    }


@app.get("/api/v1/meta")
async def meta() -> dict[str, Any]:
    latest = db_latest_calibration()
    return {
        "contract_version": CONTRACT_VERSION, "engine_version": ENGINE_VERSION,
        "model_version": MODEL_VERSION,
        "model_parameters": {"rho": DIXON_COLES_RHO, "shrinkage": SHRINKAGE_MATCHES, "recency_strength": RECENCY_STRENGTH},
        "calibration_status": latest.get("status") if latest else MODEL_VALIDATION_STATUS,
        "calibration_run_id": latest.get("run_id") if latest else None,
        "viability_status": "NOT_CERTIFIED", "timezone": "America/Bogota",
        "foundation": {"founder": "Sebastián Betancourt"},
        "rate_policy": {"configured_per_minute": REQUESTS_PER_MINUTE, "hard_maximum": 420, "pacing": "uniform"},
        "disclaimer": "Análisis estadístico deportivo; no constituye recomendación de apuesta.",
    }


@app.get("/api/v1/calibration/status")
async def calibration_status() -> dict[str, Any]:
    latest = db_latest_calibration()
    durable_job = await asyncio.to_thread(db_latest_calibration_job)
    summary = None
    if latest:
        summary = {
            "run_id": latest.get("run_id"), "created_at": latest.get("created_at"),
            "status": latest.get("status"), "promoted": latest.get("promoted", False),
            "points": latest.get("points"), "train_n": latest.get("train_n"),
            "holdout_n": latest.get("holdout_n"), "parameters": latest.get("parameters"),
            "holdout": latest.get("holdout"), "advanced": latest.get("advanced", {}),
            "legacy_v1_holdout": latest.get("legacy_v1_holdout"),
            "naive_holdout": latest.get("naive_holdout"),
            "improvement": latest.get("improvement"),
            "walk_forward": latest.get("walk_forward"),
            "expected_calibration_error": latest.get("expected_calibration_error"),
            "promotion_candidate": latest.get("promotion_candidate", False),
            "promotion_policy": latest.get("promotion_policy"),
        }
    return {"task": _calibration_state, "durable_job": durable_job, "latest": summary}


@app.get("/api/v1/model/scorecard")
async def model_scorecard() -> dict[str, Any]:
    """Seguimiento prequential: sólo predicciones selladas antes del kickoff."""
    return await asyncio.to_thread(db_model_scorecard)


@app.post("/api/v1/calibration/run", status_code=202)
async def start_calibration(
    date: str | None = Query(None),
    max_leagues: int = Query(30, ge=1, le=100),
    advanced: bool = Query(True),
    resume: bool = Query(False),
    x_s2s_admin: str | None = Header(None, alias="X-S2S-Admin"),
) -> dict[str, Any]:
    global _calibration_task
    require_admin(x_s2s_admin)
    if _calibration_task and not _calibration_task.done():
        raise HTTPException(status_code=409, detail="Ya existe una calibración en ejecución")
    if resume:
        previous = await asyncio.to_thread(db_latest_calibration_job)
        if not previous or previous.get("status") not in {"RUNNING", "FAILED", "INTERRUPTED"}:
            raise HTTPException(status_code=409, detail="No existe una calibración interrumpida que reanudar")
        config = previous.get("payload") or {}
        date = str(config.get("date") or date or now_local().strftime("%Y-%m-%d"))
        max_leagues = int(config.get("max_leagues") or max_leagues)
        advanced = bool(config.get("advanced", advanced))
    requested = date or now_local().strftime("%Y-%m-%d")
    try:
        datetime.strptime(requested, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="date debe usar YYYY-MM-DD") from exc
    _calibration_task = asyncio.create_task(execute_calibration(requested, max_leagues, advanced))
    return {
        "accepted": True, "date": requested, "max_leagues": max_leagues,
        "advanced": advanced, "resumed": resume,
        "message": "Backtesting cronológico iniciado con checkpoints durables; consulta /api/v1/calibration/status",
    }


@app.get("/api/v1/sync")
async def sync_status(date: str | None = Query(None)) -> dict[str, Any]:
    requested = date or now_local().strftime("%Y-%m-%d")
    try:
        await load_catalog(requested, False)
        return sync_payload()
    except Exception as exc:
        log.exception("No se pudo sincronizar el catálogo")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/v1/props")
async def get_props(
    all: bool = Query(True),
    refresh: bool = Query(False),
    date: str | None = Query(None),
) -> list[dict[str, Any]]:
    del all
    requested = date or now_local().strftime("%Y-%m-%d")
    try:
        await load_catalog(requested, refresh)
        return merged_props()
    except Exception as exc:
        log.exception("No se pudo cargar la cartelera")
        if _catalog and _catalog_date == requested:
            return merged_props()
        raise HTTPException(status_code=502, detail=str(exc)) from exc
