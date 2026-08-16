"""S2S Sigma backend.

Render command: uvicorn render_main:app --host 0.0.0.0 --port $PORT
Required secret: API_FOOTBALL_KEY
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(title="S2S Sigma Engine", version="3.0.0")
log = logging.getLogger("s2s")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

API_KEY = os.getenv("API_FOOTBALL_KEY", "").strip()
BASE_URL = "https://v3.football.api-sports.io"
BOGOTA = ZoneInfo("America/Bogota")
HTTP_CONCURRENCY = max(2, min(int(os.getenv("HTTP_CONCURRENCY", "8")), 16))
ANALYSIS_WORKERS = max(1, min(int(os.getenv("ANALYSIS_WORKERS", "4")), 8))
REQUESTS_PER_MINUTE = max(30, min(int(os.getenv("REQUESTS_PER_MINUTE", "120")), 120))
RESPONSE_TTL = int(os.getenv("RESPONSE_TTL_SECONDS", "900"))
TEAM_TTL = int(os.getenv("TEAM_TTL_SECONDS", "21600"))
FIXTURE_STATS_TTL = int(os.getenv("FIXTURE_STATS_TTL_SECONDS", "86400"))
RHO = float(os.getenv("DIXON_COLES_RHO", "-0.13"))
EPSILON = 1e-9

_request_slots = asyncio.Semaphore(HTTP_CONCURRENCY)
_refresh_lock = asyncio.Lock()
_cache: dict[str, tuple[float, Any]] = {}
_key_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
_catalog: list[dict[str, Any]] = []
_analyses: dict[str, dict[str, Any]] = {}
_analysis_errors: dict[str, str] = {}
_analysis_task: asyncio.Task[None] | None = None
_sync_started_at: str | None = None


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, period_seconds: float = 60.0) -> None:
        self.limit = limit
        self.period = period_seconds
        self.timestamps: deque[float] = deque()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self.lock:
                now = time.monotonic()
                while self.timestamps and now - self.timestamps[0] >= self.period:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.limit:
                    self.timestamps.append(now)
                    return
                wait = self.period - (now - self.timestamps[0]) + 0.05
            await asyncio.sleep(max(wait, 0.05))


_provider_limiter = SlidingWindowRateLimiter(REQUESTS_PER_MINUTE)

BANDERAS = {
    "ARGENTINA": ("🇦🇷", "Argentina"), "BRAZIL": ("🇧🇷", "Brasil"),
    "BRASIL": ("🇧🇷", "Brasil"), "COLOMBIA": ("🇨🇴", "Colombia"),
    "SPAIN": ("🇪🇸", "España"), "ESPAÑA": ("🇪🇸", "España"),
    "USA": ("🇺🇸", "Estados Unidos"), "MEXICO": ("🇲🇽", "México"),
    "MÉXICO": ("🇲🇽", "México"), "URUGUAY": ("🇺🇾", "Uruguay"),
    "ENGLAND": ("🏴", "Inglaterra"), "INGLATERRA": ("🏴", "Inglaterra"),
}


def cached(key: str) -> Any | None:
    item = _cache.get(key)
    if item and item[0] > time.monotonic():
        return item[1]
    return None


def cache_put(key: str, value: Any, ttl: int) -> Any:
    _cache[key] = (time.monotonic() + ttl, value)
    return value


async def api_get(client: httpx.AsyncClient, path: str, params: dict[str, Any]) -> list[Any] | dict[str, Any]:
    """Call API-Sports and never disguise provider errors as an empty result."""
    if not API_KEY:
        raise RuntimeError("Falta la variable secreta API_FOOTBALL_KEY en Render")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            await _provider_limiter.acquire()
            async with _request_slots:
                response = await client.get(path, params=params)
            if response.status_code == 429:
                last_error = RuntimeError(f"API-Sports rate limit HTTP 429 en {path}")
                await asyncio.sleep(number(response.headers.get("Retry-After"), 60.0))
                continue
            response.raise_for_status()
            payload = response.json()
            errors = payload.get("errors")
            if errors:
                last_error = RuntimeError(f"API-Sports rechazó la consulta {path}: {errors}")
                if isinstance(errors, dict) and "rateLimit" in errors:
                    await asyncio.sleep(60)
                    continue
                raise last_error
            return payload.get("response", [])
        except (httpx.HTTPError, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt < 2:
                await asyncio.sleep(0.5 * (2**attempt))
    raise RuntimeError(str(last_error) or f"Fallo consultando {path}")


def country_flag(country: str, league: str) -> tuple[str, str]:
    key = (country or "").upper().strip()
    if key in BANDERAS:
        return BANDERAS[key]
    league_key = (league or "").upper()
    for candidate, value in BANDERAS.items():
        if candidate in league_key:
            return value
    return "🌐", country.title() if country else "Internacional"


def fixture_state(data: dict[str, Any]) -> dict[str, Any]:
    status = data.get("status") or {}
    short = status.get("short") or "NS"
    if short in {"1H", "2H", "HT", "ET", "P", "LIVE", "BT"}:
        elapsed = status.get("elapsed")
        return {"code": "LIVE", "display": f"EN VIVO · {elapsed or 0}'", "is_live": True, "is_finished": False, "sort": 0}
    if short in {"FT", "AET", "PEN"}:
        return {"code": "FT", "display": "FINALIZADO", "is_live": False, "is_finished": True, "sort": 2}
    try:
        local_dt = datetime.fromisoformat(data["date"].replace("Z", "+00:00")).astimezone(BOGOTA)
        prefix = "HOY" if local_dt.date() == datetime.now(BOGOTA).date() else local_dt.strftime("%d/%m")
        return {"code": short, "display": f"{prefix} · {local_dt.strftime('%I:%M %p')}", "is_live": False, "is_finished": False, "sort": 1, "timestamp": local_dt.timestamp()}
    except (KeyError, TypeError, ValueError):
        return {"code": short, "display": "PROGRAMADO", "is_live": False, "is_finished": False, "sort": 1, "timestamp": 0}


def number(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value) if value is not None else fallback
    except (TypeError, ValueError):
        return fallback


def stat_map(block: dict[str, Any]) -> dict[str, float]:
    values = {str(row.get("type")): row.get("value") for row in block.get("statistics", [])}
    yellow = number(values.get("Yellow Cards"))
    red = number(values.get("Red Cards"))
    return {
        "corners": number(values.get("Corner Kicks")),
        "cards": yellow + red,
        "shots": number(values.get("Total Shots")),
    }


async def fixture_statistics(client: httpx.AsyncClient, fixture_id: int) -> dict[int, dict[str, float]]:
    key = f"fixture-stats:{fixture_id}"
    if (hit := cached(key)) is not None:
        return hit
    async with _key_locks[key]:
        if (hit := cached(key)) is not None:
            return hit
        response = await api_get(client, "/fixtures/statistics", {"fixture": fixture_id})
        result = {int(row.get("team", {}).get("id", 0)): stat_map(row) for row in response if row.get("team", {}).get("id")}
        return cache_put(key, result, FIXTURE_STATS_TTL)


async def team_data(client: httpx.AsyncClient, team_id: int, league_id: int, season: int) -> dict[str, Any]:
    key = f"team:{team_id}:{league_id}:{season}"
    if (hit := cached(key)) is not None:
        return hit

    async with _key_locks[key]:
        if (hit := cached(key)) is not None:
            return hit
        fixtures_task = api_get(client, "/fixtures", {"team": team_id, "last": 5, "status": "FT"})
        season_task = api_get(client, "/teams/statistics", {"team": team_id, "league": league_id, "season": season})
        fixtures, season_stats = await asyncio.gather(fixtures_task, season_task)
        season_stats = season_stats if isinstance(season_stats, dict) else {}
        stats_results = await asyncio.gather(
            *(fixture_statistics(client, int(row["fixture"]["id"])) for row in fixtures),
            return_exceptions=True,
        )
        matches: list[dict[str, Any]] = []
        for fixture, stats_result in zip(fixtures, stats_results):
            teams, goals, info = fixture.get("teams", {}), fixture.get("goals", {}), fixture.get("fixture", {})
            is_home = teams.get("home", {}).get("id") == team_id
            opponent = teams.get("away" if is_home else "home", {}).get("name")
            gf = goals.get("home" if is_home else "away")
            gc = goals.get("away" if is_home else "home")
            if not opponent or gf is None or gc is None:
                continue
            stats_available = isinstance(stats_result, dict) and bool(stats_result)
            if not stats_available:
                log.warning("Sin estadísticas oficiales fixture=%s team=%s: %s", info.get("id"), team_id, stats_result)
            all_stats = stats_result if stats_available else {}
            own = all_stats.get(team_id, {})
            totals = {
                metric: sum(team_stats.get(metric, 0.0) for team_stats in all_stats.values()) if stats_available else None
                for metric in ("corners", "cards", "shots")
            }
            try:
                played_at = datetime.fromisoformat(info["date"].replace("Z", "+00:00"))
                date = played_at.strftime("%d/%m")
                timestamp = played_at.timestamp()
            except (KeyError, TypeError, ValueError):
                date, timestamp = "", 0.0
            matches.append({
                "fixture_id": str(info.get("id") or ""), "timestamp": timestamp,
                "rival": opponent, "score": f"{gf} - {gc}", "gf": int(gf), "gc": int(gc),
                "resultado": "V" if gf > gc else "E" if gf == gc else "D", "fecha": date,
                "corners": totals["corners"], "cards": totals["cards"], "shots": totals["shots"],
                "corners_team": own.get("corners"), "cards_team": own.get("cards"), "shots_team": own.get("shots"),
            })
        goals = season_stats.get("goals") or {}
        gf_avg = number((((goals.get("for") or {}).get("average") or {}).get("total")), 1.20)
        gc_avg = number((((goals.get("against") or {}).get("average") or {}).get("total")), 1.10)
        result = {"matches": matches, "gf_avg": max(gf_avg, 0.05), "gc_avg": max(gc_avg, 0.05)}
        return cache_put(key, result, TEAM_TTL)


def poisson_probability(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def probability_matrix(lambda_home: float, lambda_away: float, size: int = 9) -> list[list[float]]:
    matrix = [[poisson_probability(h, lambda_home) * poisson_probability(a, lambda_away) for a in range(size)] for h in range(size)]
    # Dixon-Coles correction for low scores.
    corrections = {
        (0, 0): 1 - lambda_home * lambda_away * RHO,
        (0, 1): 1 + lambda_home * RHO,
        (1, 0): 1 + lambda_away * RHO,
        (1, 1): 1 - RHO,
    }
    for (h, a), factor in corrections.items():
        matrix[h][a] *= max(factor, EPSILON)
    total = max(sum(map(sum, matrix)), EPSILON)
    return [[value / total for value in row] for row in matrix]


def market_history(matches: list[dict[str, Any]], metric: str, line: float, over: bool) -> list[dict[str, Any]]:
    result = []
    for match in matches:
        value = match[metric]
        if value is None:
            continue
        result.append({
            "rival": match["rival"], "score": match["score"], "resultado": match["resultado"],
            "valor_numerico": value, "cumple": value > line if over else value < line, "fecha": match["fecha"],
        })
    return result


def btts_history(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rival": match["rival"], "score": match["score"], "resultado": match["resultado"],
            "valor_numerico": 1.0 if match["gf"] > 0 and match["gc"] > 0 else 0.0,
            "cumple": match["gf"] > 0 and match["gc"] > 0, "fecha": match["fecha"],
        }
        for match in matches
    ]


def rate(matches: list[dict[str, Any]], metric: str, line: float, over: bool) -> float:
    known = [m[metric] for m in matches if m.get(metric) is not None]
    return sum((value > line) if over else (value < line) for value in known) / max(len(known), 1)


def average(matches: list[dict[str, Any]], metric: str) -> float:
    known = [m[metric] for m in matches if m.get(metric) is not None]
    return sum(known) / max(len(known), 1)


def choose_total_market(matrix: list[list[float]], home: list[dict[str, Any]], away: list[dict[str, Any]]) -> tuple[str, float, bool, float, float, float]:
    over25 = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[h])) if h + a >= 3)
    candidates = [("MÁS DE 2.5 GOLES", 2.5, True, over25), ("MENOS DE 2.5 GOLES", 2.5, False, 1 - over25)]
    label, line, over, theoretical = max(candidates, key=lambda row: row[3])
    home_rate = rate(home, "goals", line, over)
    away_rate = rate(away, "goals", line, over)
    confidence = 0.60 * theoretical + 0.40 * ((home_rate + away_rate) / 2)
    return label, line, over, confidence, home_rate, away_rate


def enrich_goals(matches: list[dict[str, Any]]) -> None:
    for match in matches:
        match["goals"] = match["gf"] + match["gc"]


async def build_prop(client: httpx.AsyncClient, fixture: dict[str, Any]) -> dict[str, Any]:
    info, league, teams = fixture.get("fixture", {}), fixture.get("league", {}), fixture.get("teams", {})
    home, away = teams.get("home", {}), teams.get("away", {})
    if not all((info.get("id"), league.get("id"), home.get("id"), away.get("id"))):
        raise ValueError("Fixture sin IDs obligatorios")
    home_data, away_data = await asyncio.gather(
        team_data(client, int(home["id"]), int(league["id"]), int(league.get("season") or datetime.now().year)),
        team_data(client, int(away["id"]), int(league["id"]), int(league.get("season") or datetime.now().year)),
    )
    target_timestamp = datetime.fromisoformat(str(info["date"]).replace("Z", "+00:00")).timestamp()
    target_id = str(info["id"])
    def before_target(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        valid = [row for row in rows if row.get("fixture_id") != target_id and 0 < row.get("timestamp", 0) < target_timestamp]
        return sorted(valid, key=lambda row: row["timestamp"], reverse=True)[:5]
    home_matches = before_target(home_data["matches"])
    away_matches = before_target(away_data["matches"])
    if not home_matches or not away_matches:
        raise RuntimeError(f"Historial oficial insuficiente: {home.get('name')} vs {away.get('name')}")
    enrich_goals(home_matches)
    enrich_goals(away_matches)

    lambda_home = max(0.20, (home_data["gf_avg"] + away_data["gc_avg"]) / 2)
    lambda_away = max(0.20, (away_data["gf_avg"] + home_data["gc_avg"]) / 2)
    matrix = probability_matrix(lambda_home, lambda_away)
    p_home = round(sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[h])) if h > a) * 100)
    p_draw = round(sum(matrix[n][n] for n in range(len(matrix))) * 100)
    p_away = 100 - p_home - p_draw
    mode_index = max(((matrix[h][a], h, a) for h in range(len(matrix)) for a in range(len(matrix[h]))))[1:]

    goal_label, goal_line, goal_over, confidence, home_rate, away_rate = choose_total_market(matrix, home_matches, away_matches)
    cr = round(max(0, min(confidence, 1)) * 100)
    corner_line, card_line, shot_line = 8.5, 4.5, 20.5
    state = fixture_state(info)
    flag, country = country_flag(str(league.get("country") or ""), str(league.get("name") or ""))

    def hist(matches: list[dict[str, Any]], metric: str, line: float, over: bool = True) -> list[dict[str, Any]]:
        return market_history(matches, metric, line, over)

    home_goals, away_goals = hist(home_matches, "goals", goal_line, goal_over), hist(away_matches, "goals", goal_line, goal_over)
    score = fixture.get("goals") or {}
    score_real = None if score.get("home") is None else f"{score.get('home')} - {score.get('away')}"
    return {
        "id": str(info["id"]), "liga": f"{flag}  {country} • {str(league.get('name') or 'Liga')}",
        "pais": country, "bandera": flag, "home_name": home.get("name"), "away_name": away.get("name"),
        "home_logo": home.get("logo") or "", "away_logo": away.get("logo") or "", "fecha": state["display"],
        "status_code": state["code"], "status_display": state["display"], "is_live": state["is_live"],
        "is_finished": state["is_finished"], "score_real": score_real, "status_verdict": "PENDIENTE",
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away, "prob_1x2": f"{p_home}% • {p_draw}% • {p_away}%",
        "marcador_estimado": f"{mode_index[0]} - {mode_index[1]}", "mercado": goal_label,
        "cr_mercado": f"{cr}%", "cr_score_num": str(cr), "cr_home_casa": f"{round(home_rate * 100)}%",
        "cr_away_fora": f"{round(away_rate * 100)}%", "cr_combinado_split": f"{round((home_rate + away_rate) * 50)}%",
        "proyeccion_val": f"{lambda_home + lambda_away:.2f}",
        "home_goles": home_goals, "away_goles": away_goals,
        "home_corners": hist(home_matches, "corners", corner_line), "away_corners": hist(away_matches, "corners", corner_line),
        "home_tarjetas": hist(home_matches, "cards", card_line, False), "away_tarjetas": hist(away_matches, "cards", card_line, False),
        "home_remates": hist(home_matches, "shots", shot_line), "away_remates": hist(away_matches, "shots", shot_line),
        "home_btts": btts_history(home_matches), "away_btts": btts_history(away_matches),
        "split_vs_list": [], "h2h_matches": [], "home_matches_20": home_goals, "away_matches_20": away_goals,
        "corners_label": "MÁS DE 8.5 CÓRNERS", "corners_conf": round((rate(home_matches, "corners", corner_line, True) + rate(away_matches, "corners", corner_line, True)) * 50),
        "corners_proyeccion": f"{(average(home_matches, 'corners') + average(away_matches, 'corners')) / 2:.1f}",
        "tarjetas_label": "MENOS DE 4.5 TARJETAS", "tarjetas_conf": round((rate(home_matches, "cards", card_line, False) + rate(away_matches, "cards", card_line, False)) * 50),
        "tarjetas_proyeccion": f"{(average(home_matches, 'cards') + average(away_matches, 'cards')) / 2:.1f}",
        "disparos_label": "MÁS DE 20.5 REMATES", "disparos_conf": round((rate(home_matches, "shots", shot_line, True) + rate(away_matches, "shots", shot_line, True)) * 50),
        "disparos_proyeccion": f"{(average(home_matches, 'shots') + average(away_matches, 'shots')) / 2:.1f}",
        "btts_label": "PROBABILIDAD AMBOS ANOTAN", "btts_conf": round(sum(matrix[h][a] for h in range(1, len(matrix)) for a in range(1, len(matrix[h]))) * 100),
        "btts_proyeccion": f"{lambda_home:.2f} - {lambda_away:.2f}",
        "cr_home_l10": f"{round(home_rate * 100)}%", "cr_away_l10": f"{round(away_rate * 100)}%", "cr_combinado_l10": f"{round((home_rate + away_rate) * 50)}%",
        "metrics_home": {"gf_prom": home_data["gf_avg"], "gc_prom": home_data["gc_avg"], "corn_prom": average(home_matches, "corners_team"), "tarj_prom": average(home_matches, "cards_team"), "rem_prom": average(home_matches, "shots_team")},
        "metrics_away": {"gf_prom": away_data["gf_avg"], "gc_prom": away_data["gc_avg"], "corn_prom": average(away_matches, "corners_team"), "tarj_prom": average(away_matches, "cards_team"), "rem_prom": average(away_matches, "shots_team")},
        "_sort": state["sort"], "_timestamp": state.get("timestamp", 0),
    }


_catalog_date: str | None = None


def catalog_prop(fixture: dict[str, Any]) -> dict[str, Any]:
    info, league, teams = fixture.get("fixture", {}), fixture.get("league", {}), fixture.get("teams", {})
    home, away = teams.get("home", {}), teams.get("away", {})
    state = fixture_state(info)
    flag, country = country_flag(str(league.get("country") or ""), str(league.get("name") or ""))
    goals = fixture.get("goals") or {}
    score_real = None if goals.get("home") is None else f"{goals.get('home')} - {goals.get('away')}"
    unavailable = state["is_finished"] or state["is_live"]
    return {
        "id": str(info.get("id") or ""), "liga": f"{flag}  {country} • {str(league.get('name') or 'Liga')}",
        "pais": country, "bandera": flag, "home_name": home.get("name") or "Equipo local",
        "away_name": away.get("name") or "Equipo visitante", "home_logo": home.get("logo") or "",
        "away_logo": away.get("logo") or "", "fecha": state["display"], "status_code": state["code"],
        "status_display": state["display"], "is_live": state["is_live"], "is_finished": state["is_finished"],
        "score_real": score_real, "status_verdict": "SIN_SNAPSHOT_PREVIO" if unavailable else "ANALISIS_PENDIENTE",
        "p_home": 0, "p_draw": 0, "p_away": 0, "prob_1x2": "PENDIENTE",
        "marcador_estimado": "—", "mercado": "ANÁLISIS ESTADÍSTICO", "cr_mercado": "N/D", "cr_score_num": "0",
        "home_goles": [], "away_goles": [], "home_corners": [], "away_corners": [],
        "home_tarjetas": [], "away_tarjetas": [], "home_remates": [], "away_remates": [],
        "analysis_status": "UNAVAILABLE" if unavailable else "PENDING",
        "analysis_message": "No existe snapshot previo al inicio" if unavailable else "Esperando procesamiento",
        "data_quality": 0, "sample_size": 0, "_sort": state["sort"], "_timestamp": state.get("timestamp", 0),
    }


def public_prop(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def merged_props() -> list[dict[str, Any]]:
    rows = []
    for fixture in _catalog:
        shell = catalog_prop(fixture)
        row = _analyses.get(shell["id"], shell)
        if shell["id"] in _analysis_errors:
            row = {**shell, "analysis_status": "ERROR", "analysis_message": _analysis_errors[shell["id"]]}
        rows.append(row)
    rows.sort(key=lambda row: (row.get("_sort", 1), row.get("_timestamp", 0), row.get("pais", "")))
    return [public_prop(row) for row in rows]


async def analyze_catalog(fixtures: list[dict[str, Any]]) -> None:
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for fixture in fixtures:
        shell = catalog_prop(fixture)
        if shell["analysis_status"] == "PENDING" and shell["id"] not in _analyses:
            queue.put_nowait(fixture)

    async def worker() -> None:
        timeout = httpx.Timeout(connect=10, read=20, write=10, pool=20)
        async with httpx.AsyncClient(base_url=BASE_URL, headers={"x-apisports-key": API_KEY}, timeout=timeout) as client:
            while True:
                try:
                    fixture = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                fixture_id = str(fixture.get("fixture", {}).get("id") or "")
                try:
                    result = await build_prop(client, fixture)
                    result["status_verdict"] = "ANALISIS_LISTO"
                    result["analysis_status"] = "READY"
                    result["analysis_message"] = "Análisis estadístico disponible"
                    result["sample_size"] = min(len(result.get("home_goles", [])), len(result.get("away_goles", []))) * 2
                    advanced = sum(bool(result.get(key)) for key in ("home_corners", "away_corners", "home_tarjetas", "away_tarjetas", "home_remates", "away_remates"))
                    result["data_quality"] = round(55 + advanced / 6 * 45)
                    _analyses[fixture_id] = result
                    _analysis_errors.pop(fixture_id, None)
                except Exception as exc:
                    _analysis_errors[fixture_id] = str(exc)
                    log.error("Análisis fallido fixture=%s: %s", fixture_id, exc)
                finally:
                    queue.task_done()

    await asyncio.gather(*(worker() for _ in range(ANALYSIS_WORKERS)))
    log.info("Procesamiento diario terminado: %s listos, %s fallidos", len(_analyses), len(_analysis_errors))


async def load_catalog(date: str, force: bool) -> None:
    global _catalog, _catalog_date, _analysis_task, _sync_started_at
    async with _refresh_lock:
        if _catalog and _catalog_date == date and not force:
            return
        timeout = httpx.Timeout(connect=10, read=20, write=10, pool=20)
        async with httpx.AsyncClient(base_url=BASE_URL, headers={"x-apisports-key": API_KEY}, timeout=timeout) as client:
            fixtures = await api_get(client, "/fixtures", {"date": date, "timezone": "America/Bogota"})
        if _catalog_date != date:
            _analyses.clear()
            _analysis_errors.clear()
        _catalog, _catalog_date = list(fixtures), date
        _sync_started_at = datetime.now(BOGOTA).isoformat()
        if _analysis_task is None or _analysis_task.done():
            _analysis_task = asyncio.create_task(analyze_catalog(_catalog.copy()))


@app.get("/")
async def root() -> dict[str, Any]:
    return {"status": "ok", "service": "S2S Sigma Engine", "version": "4.0.0", "api_key_configured": bool(API_KEY)}


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "api_key_configured": bool(API_KEY), "catalog_matches": len(_catalog), "ready_matches": len(_analyses), "analysis_running": bool(_analysis_task and not _analysis_task.done())}


@app.get("/api/v1/sync")
async def sync_status() -> dict[str, Any]:
    total = len(_catalog)
    unavailable = sum(catalog_prop(fixture)["analysis_status"] == "UNAVAILABLE" for fixture in _catalog)
    eligible = max(total - unavailable, 0)
    ready, failed = len(_analyses), len(_analysis_errors)
    completed = min(ready + failed, eligible)
    return {
        "date": _catalog_date, "total": total, "eligible": eligible, "ready": ready,
        "pending": max(eligible - completed, 0), "unavailable": unavailable, "failed": failed,
        "progress": completed / max(eligible, 1), "running": bool(_analysis_task and not _analysis_task.done()),
        "started_at": _sync_started_at,
    }


@app.get("/api/v1/props")
async def get_props(all: bool = Query(True), refresh: bool = Query(False), date: str | None = Query(None)) -> list[dict[str, Any]]:
    del all  # Kept for Android contract compatibility.
    requested_date = date or datetime.now(BOGOTA).strftime("%Y-%m-%d")
    try:
        await load_catalog(requested_date, refresh)
        return merged_props()
    except Exception as exc:
        log.exception("No se pudo cargar el catálogo diario")
        if _catalog and _catalog_date == requested_date:
            return merged_props()
        raise HTTPException(status_code=502, detail=str(exc)) from exc
