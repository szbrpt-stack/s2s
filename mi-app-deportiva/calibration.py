"""S2S Sigma historical calibration and backtesting engine.

This job uses complete league-season fixture sets, walks them chronologically,
and never lets a fixture use information from its own future. It does not
produce betting advice. A report is persisted for audit; parameters are not
promoted into production automatically.

Render shell example:
    python calibration.py --date 2026-08-16 --max-leagues 30 --advanced
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

import main

EPS = 1e-12
RHO_GRID = (-0.15, -0.12, -0.10, -0.08, -0.05, 0.0)
SHRINKAGE_GRID = (4.0, 6.0, 8.0, 10.0, 12.0)
RECENCY_GRID = (0.0, 0.10, 0.20)
ADVANCED_LINES = {"corners": 8.5, "cards": 4.5, "shots": 20.5}
CALL_BUDGET = max(100, min(int(os.getenv("CALIBRATION_CALL_BUDGET", "5000")), 20_000))


def progress(phase: str, done: int, total: int) -> None:
    main._calibration_state.update({
        "phase": phase, "done": done, "total": max(total, 1),
        "progress": round(done / max(total, 1), 4),
    })


@dataclass(frozen=True)
class PlayedFixture:
    fixture_id: int
    league_id: int
    season: int
    kickoff: datetime
    home_id: int
    away_id: int
    home_goals: int
    away_goals: int


@dataclass
class TeamHistory:
    home_for: list[float]
    home_against: list[float]
    away_for: list[float]
    away_against: list[float]
    recent_for: list[float]

    @classmethod
    def empty(cls) -> "TeamHistory":
        return cls([], [], [], [], [])


@dataclass(frozen=True)
class BacktestPoint:
    fixture_id: int
    kickoff: str
    league_id: int
    season: int
    home_id: int
    away_id: int
    home_goals: int
    away_goals: int
    league_home_rate: float
    league_away_rate: float
    home_home_for: tuple[float, ...]
    home_home_against: tuple[float, ...]
    away_away_for: tuple[float, ...]
    away_away_against: tuple[float, ...]
    home_recent_for: tuple[float, ...]
    away_recent_for: tuple[float, ...]


def mean(values: list[float] | tuple[float, ...]) -> float | None:
    return sum(values) / len(values) if values else None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def poisson(k: int, rate: float) -> float:
    return math.exp(-rate) * rate**k / math.factorial(k)


def matrix(home_rate: float, away_rate: float, rho: float, size: int = 11) -> list[list[float]]:
    values = [[poisson(h, home_rate) * poisson(a, away_rate) for a in range(size)] for h in range(size)]
    corrections = {
        (0, 0): 1 - home_rate * away_rate * rho,
        (0, 1): 1 + home_rate * rho,
        (1, 0): 1 + away_rate * rho,
        (1, 1): 1 - rho,
    }
    for (h, a), factor in corrections.items():
        values[h][a] *= max(factor, EPS)
    total = sum(map(sum, values))
    return [[cell / total for cell in row] for row in values]


def weighted_recent(values: tuple[float, ...], decay: float = 0.82) -> float | None:
    if not values:
        return None
    recent = tuple(reversed(values[-10:]))
    weights = [decay**index for index in range(len(recent))]
    return sum(value * weight for value, weight in zip(recent, weights)) / sum(weights)


def shrunk(observed: float | None, baseline: float, sample: int, k: float) -> float:
    if observed is None or sample <= 0:
        return baseline
    weight = sample / (sample + k)
    return observed * weight + baseline * (1 - weight)


def predict(point: BacktestPoint, rho: float, k: float, recency_strength: float) -> dict[str, Any]:
    base_h = clamp(point.league_home_rate, 0.4, 3.5)
    base_a = clamp(point.league_away_rate, 0.3, 3.2)
    hf = shrunk(mean(point.home_home_for), base_h, len(point.home_home_for), k)
    ha = shrunk(mean(point.home_home_against), base_a, len(point.home_home_against), k)
    af = shrunk(mean(point.away_away_for), base_a, len(point.away_away_for), k)
    aa = shrunk(mean(point.away_away_against), base_h, len(point.away_away_against), k)
    home_recent = weighted_recent(point.home_recent_for)
    away_recent = weighted_recent(point.away_recent_for)
    home_ratio = clamp((home_recent / max(mean(point.home_recent_for) or base_h, 0.2)), 0.8, 1.2) if home_recent else 1.0
    away_ratio = clamp((away_recent / max(mean(point.away_recent_for) or base_a, 0.2)), 0.8, 1.2) if away_recent else 1.0
    lambda_h = clamp(base_h * (hf / base_h) * (aa / base_h) * (home_ratio**recency_strength), 0.15, 4.5)
    lambda_a = clamp(base_a * (af / base_a) * (ha / base_a) * (away_ratio**recency_strength), 0.15, 4.5)
    scores = matrix(lambda_h, lambda_a, rho)
    p_home = sum(scores[h][a] for h in range(11) for a in range(11) if h > a)
    p_draw = sum(scores[n][n] for n in range(11))
    p_away = max(0.0, 1.0 - p_home - p_draw)
    p_over25 = sum(scores[h][a] for h in range(11) for a in range(11) if h + a >= 3)
    modal = max((scores[h][a], h, a) for h in range(11) for a in range(11))
    return {
        "p": (p_home, p_draw, p_away),
        "over25": p_over25,
        "lambda_home": lambda_h,
        "lambda_away": lambda_a,
        "modal": (modal[1], modal[2]),
    }


def parse_fixture(row: dict[str, Any]) -> PlayedFixture | None:
    info, league, teams, goals = row.get("fixture") or {}, row.get("league") or {}, row.get("teams") or {}, row.get("goals") or {}
    kickoff = main.parse_dt(info.get("date"))
    home_id = int((teams.get("home") or {}).get("id") or 0)
    away_id = int((teams.get("away") or {}).get("id") or 0)
    if (
        main.fixture_state(info)["group"] != "FINISHED"
        or not kickoff or not home_id or not away_id
        or goals.get("home") is None or goals.get("away") is None
    ):
        return None
    return PlayedFixture(
        fixture_id=int(info.get("id") or 0), league_id=int(league.get("id") or 0),
        season=int(league.get("season") or 0), kickoff=kickoff, home_id=home_id, away_id=away_id,
        home_goals=int(goals["home"]), away_goals=int(goals["away"]),
    )


def build_points(fixtures: list[PlayedFixture], min_team: int = 5, min_league: int = 30) -> list[BacktestPoint]:
    teams: defaultdict[int, TeamHistory] = defaultdict(TeamHistory.empty)
    league_home: list[float] = []
    league_away: list[float] = []
    points: list[BacktestPoint] = []
    for fixture in sorted(fixtures, key=lambda item: item.kickoff):
        home = teams[fixture.home_id]
        away = teams[fixture.away_id]
        if (
            len(league_home) >= min_league
            and len(home.recent_for) >= min_team and len(away.recent_for) >= min_team
            and len(home.home_for) >= 3 and len(away.away_for) >= 3
        ):
            points.append(BacktestPoint(
                fixture_id=fixture.fixture_id, kickoff=fixture.kickoff.isoformat(),
                league_id=fixture.league_id, season=fixture.season,
                home_id=fixture.home_id, away_id=fixture.away_id,
                home_goals=fixture.home_goals, away_goals=fixture.away_goals,
                league_home_rate=sum(league_home[-100:]) / len(league_home[-100:]),
                league_away_rate=sum(league_away[-100:]) / len(league_away[-100:]),
                home_home_for=tuple(home.home_for[-20:]), home_home_against=tuple(home.home_against[-20:]),
                away_away_for=tuple(away.away_for[-20:]), away_away_against=tuple(away.away_against[-20:]),
                home_recent_for=tuple(home.recent_for[-10:]), away_recent_for=tuple(away.recent_for[-10:]),
            ))
        hg, ag = float(fixture.home_goals), float(fixture.away_goals)
        home.home_for.append(hg); home.home_against.append(ag); home.recent_for.append(hg)
        away.away_for.append(ag); away.away_against.append(hg); away.recent_for.append(ag)
        league_home.append(hg); league_away.append(ag)
    return points


def outcome_index(point: BacktestPoint) -> int:
    return 0 if point.home_goals > point.away_goals else 1 if point.home_goals == point.away_goals else 2


def score_points(points: list[BacktestPoint], rho: float, k: float, recency: float) -> dict[str, float | int]:
    if not points:
        return {"n": 0, "log_loss": 0.0, "brier_1x2": 0.0, "accuracy_1x2": 0.0, "brier_over25": 0.0, "modal_accuracy": 0.0}
    log_loss = brier = over_brier = correct = modal_correct = 0.0
    for point in points:
        prediction = predict(point, rho, k, recency)
        actual = outcome_index(point)
        probabilities = prediction["p"]
        log_loss -= math.log(max(probabilities[actual], EPS))
        brier += sum((probability - (1.0 if index == actual else 0.0)) ** 2 for index, probability in enumerate(probabilities))
        correct += int(max(range(3), key=lambda index: probabilities[index]) == actual)
        actual_over = 1.0 if point.home_goals + point.away_goals >= 3 else 0.0
        over_brier += (prediction["over25"] - actual_over) ** 2
        modal_correct += int(prediction["modal"] == (point.home_goals, point.away_goals))
    n = len(points)
    return {
        "n": n, "log_loss": round(log_loss / n, 6), "brier_1x2": round(brier / n, 6),
        "accuracy_1x2": round(correct / n, 6), "brier_over25": round(over_brier / n, 6),
        "modal_accuracy": round(modal_correct / n, 6),
    }


def calibration_bins(points: list[BacktestPoint], rho: float, k: float, recency: float) -> list[dict[str, Any]]:
    bins: defaultdict[int, list[tuple[float, int]]] = defaultdict(list)
    for point in points:
        prediction = predict(point, rho, k, recency)
        actual = outcome_index(point)
        for index, probability in enumerate(prediction["p"]):
            bucket = min(int(probability * 10), 9)
            bins[bucket].append((probability, int(index == actual)))
    return [{
        "range": f"{bucket * 10}-{(bucket + 1) * 10}%", "n": len(rows),
        "mean_prediction": round(sum(row[0] for row in rows) / len(rows), 4),
        "observed_frequency": round(sum(row[1] for row in rows) / len(rows), 4),
    } for bucket, rows in sorted(bins.items()) if rows]


def choose_parameters(train: list[BacktestPoint]) -> tuple[dict[str, float], dict[str, Any]]:
    best_params = {"rho": -0.08, "shrinkage": 6.0, "recency": 0.10}
    best_score: dict[str, Any] | None = None
    for rho in RHO_GRID:
        for k in SHRINKAGE_GRID:
            for recency in RECENCY_GRID:
                result = score_points(train, rho, k, recency)
                if best_score is None or (result["log_loss"], result["brier_1x2"]) < (best_score["log_loss"], best_score["brier_1x2"]):
                    best_params = {"rho": rho, "shrinkage": k, "recency": recency}
                    best_score = result
    return best_params, best_score or {}


def advanced_totals(row: dict[str, Any]) -> dict[str, float | None]:
    totals: dict[str, list[float]] = defaultdict(list)
    for block in row.get("statistics") or []:
        values = main.stat_map(block)
        for metric in ADVANCED_LINES:
            if values.get(metric) is not None:
                totals[metric].append(float(values[metric]))
    return {metric: sum(totals[metric]) if totals[metric] else None for metric in ADVANCED_LINES}


def advanced_report(points: list[BacktestPoint], stats: dict[int, dict[str, float | None]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric, line in ADVANCED_LINES.items():
        chron = [point for point in points if stats.get(point.fixture_id, {}).get(metric) is not None]
        split = int(len(chron) * 0.7)
        train_values = [float(stats[p.fixture_id][metric]) for p in chron[:split]]
        test = chron[split:]
        if len(train_values) < 30 or not test:
            result[metric] = {"status": "INSUFFICIENT", "n": len(chron), "line": line}
            continue
        prior = (sum(value > line for value in train_values) + 1) / (len(train_values) + 2)
        brier = sum((prior - (1.0 if float(stats[p.fixture_id][metric]) > line else 0.0)) ** 2 for p in test) / len(test)
        result[metric] = {
            "status": "EVALUATED", "line": line, "train_n": len(train_values), "test_n": len(test),
            "train_over_frequency": round(prior, 6), "holdout_brier": round(brier, 6),
            "note": "Baseline empírico; no se promueve como modelo predictivo.",
        }
    return result


def save_report(report: dict[str, Any]) -> int:
    with sqlite3.connect(main.STATE_DB_PATH) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS calibration_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL,
            promoted INTEGER NOT NULL DEFAULT 0,
            payload TEXT NOT NULL
        )""")
        cursor = db.execute(
            "INSERT INTO calibration_runs(created_at,status,promoted,payload) VALUES(?,?,0,?)",
            (datetime.now(main.UTC).isoformat(), report["status"], json.dumps(report, ensure_ascii=False, separators=(",", ":"))),
        )
        db.commit()
        return int(cursor.lastrowid)


async def fetch_rows(client: httpx.AsyncClient, league_id: int, season: int) -> list[dict[str, Any]]:
    response = await main.provider_get(client, "/fixtures", {"league": league_id, "season": season, "status": "FT"})
    return [row for row in response if isinstance(row, dict)] if isinstance(response, list) else []


async def fetch_advanced(client: httpx.AsyncClient, fixture_ids: list[int]) -> dict[int, dict[str, float | None]]:
    result: dict[int, dict[str, float | None]] = {}
    batches = main.chunks(fixture_ids, 20)
    for index, batch in enumerate(batches, 1):
        rows = await main.provider_get(client, "/fixtures", {"ids": "-".join(map(str, batch))})
        for row in rows if isinstance(rows, list) else []:
            fixture_id = int((row.get("fixture") or {}).get("id") or 0)
            if fixture_id:
                result[fixture_id] = advanced_totals(row)
        progress("ADVANCED_STATS", index, len(batches))
    return result


async def run_calibration(date: str, max_leagues: int, include_advanced: bool) -> dict[str, Any]:
    timeout = httpx.Timeout(connect=10, read=60, write=10, pool=60)
    async with httpx.AsyncClient(base_url=main.BASE_URL, headers={"x-apisports-key": main.API_KEY}, timeout=timeout) as client:
        catalog_raw = await main.provider_get(client, "/fixtures", {"date": date, "timezone": "America/Bogota"})
        catalog = [row for row in catalog_raw if isinstance(row, dict)] if isinstance(catalog_raw, list) else []
        league_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        for row in catalog:
            league = row.get("league") or {}
            key = (int(league.get("id") or 0), int(league.get("season") or 0))
            if all(key):
                league_counts[key] += 1
        selected = [key for key, _ in sorted(league_counts.items(), key=lambda item: (-item[1], item[0]))[:max_leagues]]
        semaphore = asyncio.Semaphore(6)
        loaded_count = 0

        async def load(key: tuple[int, int]) -> tuple[tuple[int, int], list[dict[str, Any]]]:
            nonlocal loaded_count
            async with semaphore:
                try:
                    return key, await fetch_rows(client, *key)
                finally:
                    loaded_count += 1
                    progress("LEAGUE_SEASONS", loaded_count, len(selected))

        loaded = await asyncio.gather(*(load(key) for key in selected), return_exceptions=True)
        all_points: list[BacktestPoint] = []
        league_points: dict[str, list[BacktestPoint]] = {}
        source_fixtures: dict[int, PlayedFixture] = {}
        errors: list[str] = []
        for item in loaded:
            if isinstance(item, Exception):
                errors.append(str(item)); continue
            key, rows = item
            fixtures = [fixture for row in rows if (fixture := parse_fixture(row))]
            for fixture in fixtures:
                source_fixtures[fixture.fixture_id] = fixture
            points = build_points(fixtures)
            league_points[f"{key[0]}:{key[1]}"] = points
            all_points.extend(points)

        all_points.sort(key=lambda point: point.kickoff)
        split = int(len(all_points) * 0.7)
        train, holdout = all_points[:split], all_points[split:]
        if len(train) < 100 or len(holdout) < 30:
            report = {
                "status": "INSUFFICIENT", "date": date, "leagues_requested": len(selected),
                "points": len(all_points), "errors": errors,
                "reason": "Se requieren al menos 100 observaciones de entrenamiento y 30 de holdout.",
            }
            report["run_id"] = save_report(report)
            return report

        progress("PARAMETER_SEARCH", 0, 1)
        params, train_score = choose_parameters(train)
        progress("PARAMETER_SEARCH", 1, 1)
        holdout_score = score_points(holdout, params["rho"], params["shrinkage"], params["recency"])
        per_league: dict[str, Any] = {}
        for key, points in league_points.items():
            if len(points) < 60:
                per_league[key] = {"status": "INSUFFICIENT", "n": len(points)}
                continue
            league_split = int(len(points) * 0.7)
            league_train, league_test = points[:league_split], points[league_split:]
            league_params, _ = choose_parameters(league_train)
            per_league[key] = {
                "status": "EVALUATED", "parameters": league_params,
                "holdout": score_points(league_test, league_params["rho"], league_params["shrinkage"], league_params["recency"]),
            }

        advanced = {}
        if include_advanced:
            remaining_calls = max(CALL_BUDGET - len(selected) - 1, 0)
            max_fixture_stats = remaining_calls * 20
            advanced_ids = [point.fixture_id for point in all_points[:max_fixture_stats]]
            advanced_stats = await fetch_advanced(client, advanced_ids)
            advanced = advanced_report(all_points, advanced_stats)

        report = {
            "status": "EVALUATED_NOT_PROMOTED", "date": date,
            "created_at": datetime.now(main.UTC).isoformat(), "leagues_requested": len(selected),
            "leagues_loaded": len(league_points), "points": len(all_points), "train_n": len(train), "holdout_n": len(holdout),
            "parameters": params, "train": train_score, "holdout": holdout_score,
            "calibration_bins": calibration_bins(holdout, params["rho"], params["shrinkage"], params["recency"]),
            "per_league": per_league, "advanced": advanced, "provider_errors": errors,
            "api_budget": {"configured_calls": CALL_BUDGET, "advanced_fixtures_requested": len(advanced_ids) if include_advanced else 0},
            "leakage_policy": "Cada predicción usa únicamente fixtures anteriores al kickoff.",
            "promotion_policy": "Nunca automática; requiere auditoría humana y umbrales predefinidos.",
        }
        report["run_id"] = save_report(report)
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S2S Sigma chronological calibration")
    parser.add_argument("--date", default=main.now_local().strftime("%Y-%m-%d"))
    parser.add_argument("--max-leagues", type=int, default=30)
    parser.add_argument("--advanced", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = asyncio.run(run_calibration(args.date, max(1, min(args.max_leagues, 100)), args.advanced))
    print(json.dumps(output, ensure_ascii=False, indent=2))
