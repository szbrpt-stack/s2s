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
ADVANCED_SHRINKAGE_GRID = (6.0, 12.0, 20.0)
BTTS_HISTORY_K_GRID = (8.0, 12.0, 16.0, 24.0)
BTTS_MAX_WEIGHT_GRID = (0.35, 0.45, 0.55)
CALL_BUDGET = max(100, min(int(os.getenv("CALIBRATION_CALL_BUDGET", "5000")), 20_000))
V2_CANDIDATE = {"rho": -0.05, "shrinkage": 12.0, "recency": 0.0}


def progress(phase: str, done: int, total: int) -> None:
    main._calibration_state.update({
        "phase": phase, "done": done, "total": max(total, 1),
        "progress": round(done / max(total, 1), 4),
    })
    job_id = getattr(main, "_calibration_job_id", None)
    if job_id:
        main.db_checkpoint_calibration_job(
            job_id, "RUNNING", phase, done, total,
            {key: main._calibration_state.get(key) for key in ("started_at", "run_id", "error")},
        )


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
    home_ratio = clamp((home_recent / max(mean(point.home_recent_for) or base_h, 0.2)), 0.85, 1.15) if home_recent else 1.0
    away_ratio = clamp((away_recent / max(mean(point.away_recent_for) or base_a, 0.2)), 0.85, 1.15) if away_recent else 1.0
    lambda_h = clamp(base_h * (hf / base_h) * (aa / base_h) * (home_ratio**recency_strength), 0.15, 4.5)
    lambda_a = clamp(base_a * (af / base_a) * (ha / base_a) * (away_ratio**recency_strength), 0.15, 4.5)
    scores = matrix(lambda_h, lambda_a, rho)
    p_home = sum(scores[h][a] for h in range(11) for a in range(11) if h > a)
    p_draw = sum(scores[n][n] for n in range(11))
    p_away = max(0.0, 1.0 - p_home - p_draw)
    p_over25 = sum(scores[h][a] for h in range(11) for a in range(11) if h + a >= 3)
    p_btts = sum(scores[h][a] for h in range(1, 11) for a in range(1, 11))
    modal = max((scores[h][a], h, a) for h in range(11) for a in range(11))
    return {
        "p": (p_home, p_draw, p_away),
        "over25": p_over25,
        "btts": p_btts,
        "lambda_home": lambda_h,
        "lambda_away": lambda_a,
        "modal": (modal[1], modal[2]),
    }


def predict_btts_fused(
    point: BacktestPoint, rho: float, shrinkage: float, recency: float,
    history_k: float, max_history_weight: float,
) -> dict[str, float | int]:
    structural = float(predict(point, rho, shrinkage, recency)["btts"])
    paired = [
        *(zip(point.home_home_for, point.home_home_against)),
        *(zip(point.away_away_for, point.away_away_against)),
    ]
    sample = len(paired)
    hits = sum(for_goals > 0 and against_goals > 0 for for_goals, against_goals in paired)
    posterior = (hits + 2.0) / (sample + 4.0) if sample else structural
    weight = min(sample / (sample + history_k), max_history_weight)
    probability = clamp(structural * (1 - weight) + posterior * weight, 0.05, 0.95)
    return {"probability": probability, "structural": structural, "posterior": posterior,
            "sample": sample, "hits": hits, "history_weight": weight}


def score_btts_points(
    points: list[BacktestPoint], base_params: dict[str, float], history_k: float, max_history_weight: float,
) -> dict[str, Any]:
    if not points:
        return {"n": 0, "brier": None, "log_loss": None}
    brier = log_loss = 0.0
    bins: defaultdict[int, list[tuple[float, float]]] = defaultdict(list)
    for point in points:
        result = predict_btts_fused(
            point, base_params["rho"], base_params["shrinkage"], base_params["recency"],
            history_k, max_history_weight,
        )
        probability = float(result["probability"])
        actual = float(point.home_goals > 0 and point.away_goals > 0)
        brier += (probability - actual) ** 2
        log_loss -= actual * math.log(max(probability, EPS)) + (1 - actual) * math.log(max(1 - probability, EPS))
        bins[min(int(probability * 10), 9)].append((probability, actual))
    n = len(points)
    calibration = [{
        "from": bucket / 10, "to": (bucket + 1) / 10, "n": len(rows),
        "mean_probability": round(sum(row[0] for row in rows) / len(rows), 4),
        "observed_frequency": round(sum(row[1] for row in rows) / len(rows), 4),
    } for bucket, rows in sorted(bins.items())]
    ece = sum(row["n"] * abs(row["mean_probability"] - row["observed_frequency"]) for row in calibration) / n
    return {"n": n, "brier": round(brier / n, 6), "log_loss": round(log_loss / n, 6),
            "ece": round(ece, 6), "calibration_bins": calibration}


def choose_btts_parameters(points: list[BacktestPoint], base_params: dict[str, float]) -> tuple[dict[str, float], dict[str, Any]]:
    candidates = []
    for history_k in BTTS_HISTORY_K_GRID:
        for max_weight in BTTS_MAX_WEIGHT_GRID:
            score = score_btts_points(points, base_params, history_k, max_weight)
            candidates.append((float(score["brier"]), float(score["log_loss"]), history_k, max_weight, score))
    _, _, history_k, max_weight, score = min(candidates)
    return {"history_k": history_k, "max_history_weight": max_weight}, score


def btts_walk_forward(points: list[BacktestPoint], folds: int = 4) -> dict[str, Any]:
    if len(points) < 400:
        return {"status": "INSUFFICIENT", "folds": 0, "points": len(points)}
    initial_train = len(points) // 2
    block = max((len(points) - initial_train) // folds, 1)
    reports = []
    for index in range(folds):
        start = initial_train + index * block
        end = len(points) if index == folds - 1 else min(start + block, len(points))
        train, test = points[:start], points[start:end]
        base_params, _ = choose_parameters(train)
        params, _ = choose_btts_parameters(train, base_params)
        candidate = score_btts_points(test, base_params, params["history_k"], params["max_history_weight"])
        structural = score_btts_points(test, base_params, 1.0, 0.0)
        naive = naive_baseline(train, test)
        naive_brier = float(naive["brier_btts"])
        improvement = {
            "vs_structural_brier_pct": improvement_percent(float(structural["brier"]), float(candidate["brier"])),
            "vs_naive_brier_pct": improvement_percent(naive_brier, float(candidate["brier"])),
        }
        reports.append({"fold": index + 1, "train_n": len(train), "test_n": len(test),
                        "parameters": params, "candidate": candidate, "dixon_coles_only": structural,
                        "naive_brier": naive_brier, "improvement": improvement,
                        "passes": all(value > 0 for value in improvement.values())})
    passing = sum(bool(report["passes"]) for report in reports)
    return {"status": "EVALUATED", "folds": len(reports), "passing_folds": passing,
            "stable_candidate": len(reports) == folds and passing >= 3,
            "required_passing_folds": 3, "reports": reports}


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
        return {"n": 0, "log_loss": 0.0, "brier_1x2": 0.0, "accuracy_1x2": 0.0, "brier_over25": 0.0, "brier_btts": 0.0, "modal_accuracy": 0.0}
    log_loss = brier = over_brier = btts_brier = correct = modal_correct = 0.0
    for point in points:
        prediction = predict(point, rho, k, recency)
        actual = outcome_index(point)
        probabilities = prediction["p"]
        log_loss -= math.log(max(probabilities[actual], EPS))
        brier += sum((probability - (1.0 if index == actual else 0.0)) ** 2 for index, probability in enumerate(probabilities))
        correct += int(max(range(3), key=lambda index: probabilities[index]) == actual)
        actual_over = 1.0 if point.home_goals + point.away_goals >= 3 else 0.0
        over_brier += (prediction["over25"] - actual_over) ** 2
        actual_btts = 1.0 if point.home_goals > 0 and point.away_goals > 0 else 0.0
        btts_brier += (prediction["btts"] - actual_btts) ** 2
        modal_correct += int(prediction["modal"] == (point.home_goals, point.away_goals))
    n = len(points)
    return {
        "n": n, "log_loss": round(log_loss / n, 6), "brier_1x2": round(brier / n, 6),
        "accuracy_1x2": round(correct / n, 6), "brier_over25": round(over_brier / n, 6),
        "brier_btts": round(btts_brier / n, 6),
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


def expected_calibration_error(bins: list[dict[str, Any]]) -> float:
    total = sum(int(row["n"]) for row in bins)
    if total <= 0:
        return 0.0
    return round(sum(int(row["n"]) * abs(float(row["mean_prediction"]) - float(row["observed_frequency"])) for row in bins) / total, 6)


def naive_baseline(train: list[BacktestPoint], test: list[BacktestPoint]) -> dict[str, Any]:
    counts = [1.0, 1.0, 1.0]
    over_count = 1.0
    btts_count = 1.0
    score_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    for point in train:
        counts[outcome_index(point)] += 1
        over_count += int(point.home_goals + point.away_goals >= 3)
        btts_count += int(point.home_goals > 0 and point.away_goals > 0)
        score_counts[(point.home_goals, point.away_goals)] += 1
    total = sum(counts)
    probabilities = tuple(value / total for value in counts)
    over_probability = over_count / (len(train) + 2)
    btts_probability = btts_count / (len(train) + 2)
    modal = max(score_counts, key=score_counts.get) if score_counts else (1, 1)
    if not test:
        return {"n": 0}
    log_loss = brier = over_brier = btts_brier = correct = modal_correct = 0.0
    predicted = max(range(3), key=lambda index: probabilities[index])
    for point in test:
        actual = outcome_index(point)
        log_loss -= math.log(max(probabilities[actual], EPS))
        brier += sum((probability - (1.0 if index == actual else 0.0)) ** 2 for index, probability in enumerate(probabilities))
        correct += int(predicted == actual)
        actual_over = 1.0 if point.home_goals + point.away_goals >= 3 else 0.0
        over_brier += (over_probability - actual_over) ** 2
        actual_btts = 1.0 if point.home_goals > 0 and point.away_goals > 0 else 0.0
        btts_brier += (btts_probability - actual_btts) ** 2
        modal_correct += int(modal == (point.home_goals, point.away_goals))
    n = len(test)
    return {
        "n": n, "log_loss": round(log_loss / n, 6), "brier_1x2": round(brier / n, 6),
        "accuracy_1x2": round(correct / n, 6), "brier_over25": round(over_brier / n, 6),
        "brier_btts": round(btts_brier / n, 6),
        "modal_accuracy": round(modal_correct / n, 6),
    }


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


def improvement_percent(reference: float, candidate: float) -> float:
    return round(100 * (reference - candidate) / reference, 4) if reference else 0.0


def walk_forward_report(points: list[BacktestPoint], folds: int = 4) -> dict[str, Any]:
    """Repeated expanding-window validation with no future leakage."""
    if len(points) < 400 or folds < 2:
        return {"status": "INSUFFICIENT", "points": len(points), "folds": 0}
    initial_train = len(points) // 2
    block = max((len(points) - initial_train) // folds, 1)
    reports: list[dict[str, Any]] = []
    for index in range(folds):
        test_start = initial_train + index * block
        test_end = len(points) if index == folds - 1 else min(test_start + block, len(points))
        train, test = points[:test_start], points[test_start:test_end]
        if len(test) < 30:
            continue
        selected_params, _ = choose_parameters(train)
        candidate = score_points(test, V2_CANDIDATE["rho"], V2_CANDIDATE["shrinkage"], V2_CANDIDATE["recency"])
        legacy = score_points(test, -0.08, 6.0, 1.0)
        naive = naive_baseline(train, test)
        improvement = {
            "vs_legacy_log_loss_pct": improvement_percent(legacy["log_loss"], candidate["log_loss"]),
            "vs_legacy_brier_pct": improvement_percent(legacy["brier_1x2"], candidate["brier_1x2"]),
            "vs_naive_log_loss_pct": improvement_percent(naive["log_loss"], candidate["log_loss"]),
            "vs_naive_brier_pct": improvement_percent(naive["brier_1x2"], candidate["brier_1x2"]),
            "btts_vs_legacy_brier_pct": improvement_percent(legacy["brier_btts"], candidate["brier_btts"]),
            "btts_vs_naive_brier_pct": improvement_percent(naive["brier_btts"], candidate["brier_btts"]),
        }
        reports.append({
            "fold": index + 1, "train_n": len(train), "test_n": len(test),
            "test_from": test[0].kickoff, "test_to": test[-1].kickoff,
            "selected_parameters_on_fold_train": selected_params,
            "v2_candidate_parameters": V2_CANDIDATE,
            "candidate": candidate, "legacy_v1": legacy,
            "naive": naive, "improvement": improvement,
            "passes": all(value > 0.0 for value in improvement.values()),
        })
    pass_count = sum(bool(row["passes"]) for row in reports)
    stable = len(reports) >= 4 and pass_count >= 3
    return {
        "status": "EVALUATED", "folds": len(reports), "passing_folds": pass_count,
        "stable_candidate": stable, "required_passing_folds": 3,
        "reports": reports,
    }


def advanced_totals(row: dict[str, Any]) -> dict[str, Any]:
    totals: dict[str, list[float]] = defaultdict(list)
    teams: dict[int, dict[str, float | None]] = {}
    for block in row.get("statistics") or []:
        values = main.stat_map(block)
        team_id = int((block.get("team") or {}).get("id") or 0)
        if team_id:
            teams[team_id] = values
        for metric in ADVANCED_LINES:
            if values.get(metric) is not None:
                totals[metric].append(float(values[metric]))
    result: dict[str, Any] = {metric: sum(totals[metric]) if totals[metric] else None for metric in ADVANCED_LINES}
    result["teams"] = teams
    return result


def count_cdf(limit: int, expected: float, history: list[float]) -> tuple[float, str, float | None]:
    """CDF Poisson/NB sin SciPy; NB sólo cuando la muestra demuestra sobredispersión."""
    expected = max(expected, 0.01)
    sample_mean = mean(history) or expected
    variance = sum((value - sample_mean) ** 2 for value in history) / max(len(history) - 1, 1)
    dispersion = sample_mean * sample_mean / (variance - sample_mean) if len(history) >= 20 and variance > sample_mean + 0.01 else None
    if dispersion is None:
        probability = math.exp(-expected)
        total = probability
        for count in range(1, limit + 1):
            probability *= expected / count
            total += probability
        return clamp(total, 0.0, 1.0), "POISSON", None
    size = clamp(dispersion, 0.25, 1000.0)
    success = size / (size + expected)
    probability = success**size
    total = probability
    for count in range(1, limit + 1):
        probability *= ((count - 1 + size) / count) * (1 - success)
        total += probability
    return clamp(total, 0.0, 1.0), "NEGATIVE_BINOMIAL", size


def advanced_records(points: list[BacktestPoint], stats: dict[int, dict[str, Any]], metric: str) -> list[tuple[BacktestPoint, float, float]]:
    records = []
    for point in points:
        row = stats.get(point.fixture_id) or {}
        teams = row.get("teams") or {}
        home = (teams.get(point.home_id) or teams.get(str(point.home_id)) or {}).get(metric)
        away = (teams.get(point.away_id) or teams.get(str(point.away_id)) or {}).get(metric)
        # Compatibilidad con reportes antiguos que sólo conservaban el total:
        # sirven como baseline descriptivo, pero no para el modelo individual.
        if home is not None and away is not None:
            records.append((point, float(home), float(away)))
    return records


def score_advanced_window(
    records: list[tuple[BacktestPoint, float, float]], train_end: int, test_end: int,
    line: float, shrinkage: float,
) -> dict[str, Any]:
    team_home_for: defaultdict[int, list[float]] = defaultdict(list)
    team_home_against: defaultdict[int, list[float]] = defaultdict(list)
    team_away_for: defaultdict[int, list[float]] = defaultdict(list)
    team_away_against: defaultdict[int, list[float]] = defaultdict(list)
    league_home: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
    league_away: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
    totals: list[float] = []
    overs = 1.0

    def update(record: tuple[BacktestPoint, float, float]) -> None:
        nonlocal overs
        point, home_value, away_value = record
        key = (point.league_id, point.season)
        team_home_for[point.home_id].append(home_value)
        team_home_against[point.home_id].append(away_value)
        team_away_for[point.away_id].append(away_value)
        team_away_against[point.away_id].append(home_value)
        league_home[key].append(home_value); league_away[key].append(away_value)
        total = home_value + away_value
        totals.append(total); overs += int(total > line)

    for record in records[:train_end]:
        update(record)
    if train_end <= 0 or test_end <= train_end:
        return {"n": 0}
    baseline_probability = overs / (train_end + 2)
    candidate_brier = baseline_brier = log_loss = 0.0
    distributions: defaultdict[str, int] = defaultdict(int)
    for record in records[train_end:test_end]:
        point, home_value, away_value = record
        key = (point.league_id, point.season)
        base_home = mean(league_home[key]) or mean([v for values in league_home.values() for v in values]) or max((mean(totals) or 2.0) / 2, 0.1)
        base_away = mean(league_away[key]) or mean([v for values in league_away.values() for v in values]) or base_home
        hf = shrunk(mean(team_home_for[point.home_id][-20:]), base_home, len(team_home_for[point.home_id][-20:]), shrinkage)
        aa = shrunk(mean(team_away_against[point.away_id][-20:]), base_home, len(team_away_against[point.away_id][-20:]), shrinkage)
        af = shrunk(mean(team_away_for[point.away_id][-20:]), base_away, len(team_away_for[point.away_id][-20:]), shrinkage)
        ha = shrunk(mean(team_home_against[point.home_id][-20:]), base_away, len(team_home_against[point.home_id][-20:]), shrinkage)
        expected = max(0.05, (hf + aa) / 2 + (af + ha) / 2)
        cdf, distribution, _ = count_cdf(math.floor(line), expected, totals[-500:])
        probability = clamp(1 - cdf, EPS, 1 - EPS)
        actual = 1.0 if home_value + away_value > line else 0.0
        candidate_brier += (probability - actual) ** 2
        baseline_brier += (baseline_probability - actual) ** 2
        log_loss -= actual * math.log(probability) + (1 - actual) * math.log(1 - probability)
        distributions[distribution] += 1
        update(record)
    n = test_end - train_end
    candidate = candidate_brier / n
    baseline = baseline_brier / n
    return {
        "n": n, "brier": round(candidate, 6), "log_loss": round(log_loss / n, 6),
        "baseline_brier": round(baseline, 6),
        "brier_improvement_pct": improvement_percent(baseline, candidate),
        "distributions": dict(distributions),
    }


def advanced_report(points: list[BacktestPoint], stats: dict[int, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric, line in ADVANCED_LINES.items():
        records = advanced_records(points, stats, metric)
        split = int(len(records) * 0.7)
        if split < 100 or len(records) - split < 40:
            result[metric] = {"status": "INSUFFICIENT", "n": len(records), "line": line,
                              "reason": "Se requieren estadísticas por equipo: mínimo 100 train / 40 holdout."}
            continue
        validation_start = max(60, int(split * 0.75))
        choices = [(score_advanced_window(records[:split], validation_start, split, line, k)["brier"], k) for k in ADVANCED_SHRINKAGE_GRID]
        shrinkage = min(choices)[1]
        holdout = score_advanced_window(records, split, len(records), line, shrinkage)
        initial = max(80, int(len(records) * 0.4))
        block = max(1, (len(records) - initial) // 4)
        folds = []
        for index in range(4):
            train_end = initial + index * block
            test_end = len(records) if index == 3 else min(len(records), train_end + block)
            fold = score_advanced_window(records, train_end, test_end, line, shrinkage)
            fold["fold"] = index + 1
            fold["passes"] = fold.get("brier_improvement_pct", -1) > 0
            folds.append(fold)
        passing = sum(bool(fold["passes"]) for fold in folds)
        candidate = holdout.get("brier_improvement_pct", -1) > 0 and passing >= 3
        result[metric] = {
            "status": "EVALUATED_NOT_PROMOTED", "line": line, "n": len(records),
            "train_n": split, "test_n": len(records) - split, "shrinkage": shrinkage,
            "holdout": holdout, "walk_forward": {"folds": folds, "passing_folds": passing},
            "promotion_candidate": candidate,
            "note": "Modelo individual de conteo; requiere auditoría humana antes de cualquier promoción.",
        }
    return result


def save_report(report: dict[str, Any]) -> int:
    return main.db_save_calibration(report)


async def fetch_rows(client: httpx.AsyncClient, league_id: int, season: int) -> list[dict[str, Any]]:
    response = await main.provider_get(client, "/fixtures", {"league": league_id, "season": season, "status": "FT"})
    return [row for row in response if isinstance(row, dict)] if isinstance(response, list) else []


async def fetch_advanced(client: httpx.AsyncClient, fixture_ids: list[int]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = await asyncio.to_thread(main.db_load_advanced_stats, fixture_ids)
    missing = [fixture_id for fixture_id in fixture_ids if fixture_id not in result]
    batches = main.chunks(missing, 20)
    fetched: dict[int, dict[str, Any]] = {}
    for index, batch in enumerate(batches, 1):
        rows = await main.provider_get(client, "/fixtures", {"ids": "-".join(map(str, batch))})
        for row in rows if isinstance(rows, list) else []:
            fixture_id = int((row.get("fixture") or {}).get("id") or 0)
            if fixture_id:
                parsed = advanced_totals(row)
                result[fixture_id] = parsed
                fetched[fixture_id] = parsed
        progress("ADVANCED_STATS", index, len(batches))
    await asyncio.to_thread(main.db_save_advanced_stats, fetched)
    main.log.info("Dataset avanzado: cache=%s nuevos=%s solicitados=%s", len(result) - len(fetched), len(fetched), len(fixture_ids))
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
        current = [key for key, _ in sorted(league_counts.items(), key=lambda item: (-item[1], item[0]))[:max_leagues]]
        # Dos temporadas aumentan potencia estadística sin mezclar el futuro:
        # cada serie se reconstruye cronológicamente y conserva liga/temporada.
        selected = list(dict.fromkeys(current + [(league_id, season - 1) for league_id, season in current if season > 1900]))
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
        params, train_score = await asyncio.to_thread(choose_parameters, train)
        progress("PARAMETER_SEARCH", 1, 1)
        holdout_score = await asyncio.to_thread(score_points, holdout, params["rho"], params["shrinkage"], params["recency"])
        legacy_v1_holdout = await asyncio.to_thread(score_points, holdout, -0.08, 6.0, 1.0)
        naive_holdout = await asyncio.to_thread(naive_baseline, train, holdout)
        bins = await asyncio.to_thread(calibration_bins, holdout, params["rho"], params["shrinkage"], params["recency"])
        progress("BTTS_PARAMETER_SEARCH", 0, 1)
        btts_params, btts_train = await asyncio.to_thread(choose_btts_parameters, train, params)
        btts_holdout = await asyncio.to_thread(
            score_btts_points, holdout, params, btts_params["history_k"], btts_params["max_history_weight"]
        )
        btts_structural = await asyncio.to_thread(score_btts_points, holdout, params, 1.0, 0.0)
        btts_naive = float(naive_holdout["brier_btts"])
        btts_walk = await asyncio.to_thread(btts_walk_forward, all_points, 4)
        btts_report = {
            "status": "EVALUATED_NOT_PROMOTED",
            "parameters": btts_params,
            "train": btts_train,
            "holdout": btts_holdout,
            "dixon_coles_only_holdout": btts_structural,
            "naive_holdout_brier": btts_naive,
            "improvement": {
                "vs_dixon_coles_only_brier_pct": improvement_percent(float(btts_structural["brier"]), float(btts_holdout["brier"])),
                "vs_naive_brier_pct": improvement_percent(btts_naive, float(btts_holdout["brier"])),
            },
            "walk_forward": btts_walk,
            "promotion_candidate": (
                len(holdout) >= 500
                and bool(btts_walk.get("stable_candidate"))
                and improvement_percent(float(btts_structural["brier"]), float(btts_holdout["brier"])) > 0
                and improvement_percent(btts_naive, float(btts_holdout["brier"])) > 0
            ),
            "promotion_policy": "Nunca automática; requiere auditoría humana.",
        }
        progress("BTTS_PARAMETER_SEARCH", 1, 1)
        per_league: dict[str, Any] = {}
        for key, points in league_points.items():
            if len(points) < 60:
                per_league[key] = {"status": "INSUFFICIENT", "n": len(points)}
                continue
            league_split = int(len(points) * 0.7)
            league_train, league_test = points[:league_split], points[league_split:]
            league_params, _ = await asyncio.to_thread(choose_parameters, league_train)
            per_league[key] = {
                "status": "EVALUATED", "parameters": league_params,
                "holdout": await asyncio.to_thread(score_points, league_test, league_params["rho"], league_params["shrinkage"], league_params["recency"]),
            }

        advanced = {}
        if include_advanced:
            remaining_calls = max(CALL_BUDGET - len(selected) - 1, 0)
            max_fixture_stats = remaining_calls * 20
            advanced_ids = [point.fixture_id for point in all_points[:max_fixture_stats]]
            advanced_stats = await fetch_advanced(client, advanced_ids)
            advanced = await asyncio.to_thread(advanced_report, all_points, advanced_stats)

        improvement = {
            "vs_legacy_log_loss_pct": improvement_percent(legacy_v1_holdout["log_loss"], holdout_score["log_loss"]),
            "vs_legacy_brier_pct": improvement_percent(legacy_v1_holdout["brier_1x2"], holdout_score["brier_1x2"]),
            "vs_naive_log_loss_pct": improvement_percent(naive_holdout["log_loss"], holdout_score["log_loss"]),
            "vs_naive_brier_pct": improvement_percent(naive_holdout["brier_1x2"], holdout_score["brier_1x2"]),
            "btts_vs_legacy_brier_pct": improvement_percent(legacy_v1_holdout["brier_btts"], holdout_score["brier_btts"]),
            "btts_vs_naive_brier_pct": improvement_percent(naive_holdout["brier_btts"], holdout_score["brier_btts"]),
        }
        progress("WALK_FORWARD", 0, 1)
        walk_forward = await asyncio.to_thread(walk_forward_report, all_points, 4)
        progress("WALK_FORWARD", 1, 1)
        promotion_candidate = (
            len(holdout) >= 500
            and params == V2_CANDIDATE
            and improvement["vs_legacy_log_loss_pct"] > 1.0
            and improvement["vs_legacy_brier_pct"] > 1.0
            and improvement["vs_naive_log_loss_pct"] > 0.0
            and improvement["vs_naive_brier_pct"] > 0.0
            and bool(walk_forward.get("stable_candidate"))
        )

        report = {
            "status": "EVALUATED_NOT_PROMOTED", "date": date,
            "created_at": datetime.now(main.UTC).isoformat(), "leagues_requested": len(selected),
            "leagues_loaded": len(league_points), "points": len(all_points), "train_n": len(train), "holdout_n": len(holdout),
            "parameters": params, "train": train_score, "holdout": holdout_score,
            "legacy_v1_holdout": legacy_v1_holdout, "naive_holdout": naive_holdout,
            "improvement": improvement, "promotion_candidate": promotion_candidate,
            "walk_forward": walk_forward,
            "calibration_bins": bins, "expected_calibration_error": expected_calibration_error(bins),
            "per_league": per_league, "advanced": advanced, "provider_errors": errors,
            "btts": btts_report,
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
