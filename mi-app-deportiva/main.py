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
MAX_FIXTURES = max(1, min(int(os.getenv("MAX_FIXTURES", "6")), 20))
HTTP_CONCURRENCY = max(2, min(int(os.getenv("HTTP_CONCURRENCY", "8")), 16))
RESPONSE_TTL = int(os.getenv("RESPONSE_TTL_SECONDS", "900"))
TEAM_TTL = int(os.getenv("TEAM_TTL_SECONDS", "21600"))
FIXTURE_STATS_TTL = int(os.getenv("FIXTURE_STATS_TTL_SECONDS", "86400"))
RHO = float(os.getenv("DIXON_COLES_RHO", "-0.13"))
EPSILON = 1e-9

_request_slots = asyncio.Semaphore(HTTP_CONCURRENCY)
_refresh_lock = asyncio.Lock()
_cache: dict[str, tuple[float, Any]] = {}

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
            async with _request_slots:
                response = await client.get(path, params=params)
            response.raise_for_status()
            payload = response.json()
            errors = payload.get("errors")
            if errors:
                raise RuntimeError(f"API-Sports rechazó la consulta {path}: {errors}")
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
    response = await api_get(client, "/fixtures/statistics", {"fixture": fixture_id})
    result = {int(row.get("team", {}).get("id", 0)): stat_map(row) for row in response if row.get("team", {}).get("id")}
    return cache_put(key, result, FIXTURE_STATS_TTL)


async def team_data(client: httpx.AsyncClient, team_id: int, league_id: int, season: int) -> dict[str, Any]:
    key = f"team:{team_id}:{league_id}:{season}"
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
            log.warning("Historial incompleto team=%s fixture=%s", team_id, info.get("id"))
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
            date = datetime.fromisoformat(info["date"].replace("Z", "+00:00")).strftime("%d/%m")
        except (KeyError, TypeError, ValueError):
            date = ""
        matches.append({
            "rival": opponent, "score": f"{gf} - {gc}", "gf": int(gf), "gc": int(gc),
            "resultado": "V" if gf > gc else "E" if gf == gc else "D", "fecha": date,
            "corners": totals["corners"], "cards": totals["cards"], "shots": totals["shots"],
            "corners_team": own.get("corners"), "cards_team": own.get("cards"), "shots_team": own.get("shots"),
        })

    goals = season_stats.get("goals") or {}
    # Correct schema: average.total is itself a numeric string, not a dictionary.
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
    home_matches, away_matches = home_data["matches"], away_data["matches"]
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
        "home_btts": hist(home_matches, "goals", 1.5), "away_btts": hist(away_matches, "goals", 1.5),
        "split_vs_list": [], "h2h_matches": [], "home_matches_20": home_goals, "away_matches_20": away_goals,
        "corners_label": "MÁS DE 8.5 CÓRNERS", "corners_conf": round((rate(home_matches, "corners", corner_line, True) + rate(away_matches, "corners", corner_line, True)) * 50),
        "corners_proyeccion": f"{(average(home_matches, 'corners') + average(away_matches, 'corners')) / 2:.1f}",
        "tarjetas_label": "MENOS DE 4.5 TARJETAS", "tarjetas_conf": round((rate(home_matches, "cards", card_line, False) + rate(away_matches, "cards", card_line, False)) * 50),
        "tarjetas_proyeccion": f"{(average(home_matches, 'cards') + average(away_matches, 'cards')) / 2:.1f}",
        "disparos_label": "MÁS DE 20.5 REMATES", "disparos_conf": round((rate(home_matches, "shots", shot_line, True) + rate(away_matches, "shots", shot_line, True)) * 50),
        "disparos_proyeccion": f"{(average(home_matches, 'shots') + average(away_matches, 'shots')) / 2:.1f}",
        "btts_label": "AMBOS ANOTAN", "btts_conf": round(sum(matrix[h][a] for h in range(1, len(matrix)) for a in range(1, len(matrix[h]))) * 100),
        "btts_proyeccion": f"{lambda_home:.2f} - {lambda_away:.2f}",
        "cr_home_l10": f"{round(home_rate * 100)}%", "cr_away_l10": f"{round(away_rate * 100)}%", "cr_combinado_l10": f"{round((home_rate + away_rate) * 50)}%",
        "metrics_home": {"gf_prom": home_data["gf_avg"], "gc_prom": home_data["gc_avg"], "corn_prom": average(home_matches, "corners_team"), "tarj_prom": average(home_matches, "cards_team"), "rem_prom": average(home_matches, "shots_team")},
        "metrics_away": {"gf_prom": away_data["gf_avg"], "gc_prom": away_data["gc_avg"], "corn_prom": average(away_matches, "corners_team"), "tarj_prom": average(away_matches, "cards_team"), "rem_prom": average(away_matches, "shots_team")},
        "_sort": state["sort"], "_timestamp": state.get("timestamp", 0),
    }


async def compute_props(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    today = datetime.now(BOGOTA).strftime("%Y-%m-%d")
    fixtures = await api_get(client, "/fixtures", {"date": today, "timezone": "America/Bogota"})
    if not fixtures:
        fixtures = await api_get(client, "/fixtures", {"next": MAX_FIXTURES, "timezone": "America/Bogota"})
    fixtures = list(fixtures)[:MAX_FIXTURES]
    results = await asyncio.gather(*(build_prop(client, fixture) for fixture in fixtures), return_exceptions=True)
    props = []
    for fixture, result in zip(fixtures, results):
        if isinstance(result, Exception):
            log.error(
                "Fixture descartado id=%s: %s",
                fixture.get("fixture", {}).get("id"), result,
                exc_info=(type(result), result, result.__traceback__),
            )
        else:
            props.append(result)
    if fixtures and not props:
        raise RuntimeError("Ningún fixture pudo construirse; revisa los errores anteriores")
    props.sort(key=lambda row: (row["_sort"], row["_timestamp"], row["pais"]))
    for row in props:
        row.pop("_sort", None)
        row.pop("_timestamp", None)
    log.info("Respuesta construida con %s partidos independientes", len(props))
    return props


@app.get("/")
async def root() -> dict[str, Any]:
    return {"status": "ok", "service": "S2S Sigma Engine", "version": "3.0.0", "api_key_configured": bool(API_KEY)}


@app.get("/health")
async def health() -> dict[str, Any]:
    response = cached("props")
    return {"status": "ok", "api_key_configured": bool(API_KEY), "cached_matches": len(response or []), "max_fixtures": MAX_FIXTURES}


@app.get("/api/v1/props")
async def get_props(all: bool = Query(True), refresh: bool = Query(False)) -> list[dict[str, Any]]:
    del all  # Kept for Android contract compatibility.
    if not refresh and (hit := cached("props")) is not None:
        return hit
    async with _refresh_lock:
        if not refresh and (hit := cached("props")) is not None:
            return hit
        try:
            timeout = httpx.Timeout(connect=10, read=15, write=10, pool=10)
            async with httpx.AsyncClient(base_url=BASE_URL, headers={"x-apisports-key": API_KEY}, timeout=timeout) as client:
                return cache_put("props", await compute_props(client), RESPONSE_TTL)
        except Exception as exc:
            log.exception("No se pudo construir /api/v1/props")
            stale = _cache.get("props")
            if stale:
                log.warning("Entregando última respuesta conocida por fallo temporal")
                return stale[1]
            raise HTTPException(status_code=502, detail=str(exc)) from exc
