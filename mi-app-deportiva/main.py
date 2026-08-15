from fastapi import FastAPI
import httpx
import numpy as np
from datetime import datetime
import zoneinfo
from typing import Dict, List, Any
import asyncio

app = FastAPI(title="S2S Sigma Engine - Mass Direct Concurrent Core")

API_KEY = "9cf313ae66d39a8f1aa2674401de70ce"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

CACHE_EQUIPOS: Dict[int, List[Dict[str, Any]]] = {}

BANDERA_MAP = {
    "ARGENTINA": ("\U0001F1E6\U0001F1F7", "Argentina"),
    "BRASIL": ("\U0001F1E7\U0001F1F7", "Brasil"),
    "BRAZIL": ("\U0001F1E7\U0001F1F7", "Brasil"),
    "COLOMBIA": ("\U0001F1E8\U0001F1F4", "Colombia"),
    "ESPAÑA": ("\U0001F1EA\U0001F1F8", "España"),
    "SPAIN": ("\U0001F1EA\U0001F1F8", "España"),
    "ESTADOS UNIDOS": ("\U0001F1FA\U0001F1F8", "Estados Unidos"),
    "USA": ("\U0001F1FA\U0001F1F8", "Estados Unidos"),
    "MEXICO": ("\U0001F1F2\U0001F1FD", "México"),
    "MÉXICO": ("\U0001F1F2\U0001F1FD", "México"),
    "URUGUAY": ("\U0001F1FA\U0001F1FE", "Uruguay"),
    "INGLATERRA": ("\U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006E\U000E0067\U000E007F", "Inglaterra"),
}

def obtener_pais_y_bandera(pais_raw: str, liga_raw: str) -> tuple:
    pais_key = pais_raw.upper().strip() if pais_raw else ""
    if pais_key in BANDERA_MAP:
        return BANDERA_MAP[pais_key]
    for k, v in BANDERA_MAP.items():
        if k in liga_raw.upper():
            return v
    return ("\U0001F310", pais_raw.title() if pais_raw else "Internacional")

def parsear_estado_cronologico(fixture_data: dict) -> dict:
    status = fixture_data.get("status", {})
    status_short = status.get("short", "")
    elapsed = status.get("elapsed", 0)
    
    if status_short in ["1H", "2H", "HT", "ET", "P", "LIVE"]:
        disp = "ENTRETIEMPO" if status_short == "HT" else f"EN VIVO · {elapsed}'"
        return {"code": "LIVE", "display": disp, "is_live": True, "is_finished": False}
        
    if status_short in ["FT", "AET", "PEN"]:
        return {"code": "FT", "display": "FINALIZADO", "is_live": False, "is_finished": True}

    date_str = fixture_data.get("date", "")
    try:
        dt_utc = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        tz_col = zoneinfo.ZoneInfo("America/Bogota")
        dt_col = dt_utc.astimezone(tz_col)
        hoy = datetime.now(tz_col).date()
        prefijo = "HOY" if dt_col.date() == hoy else dt_col.strftime("%d/%m")
        return {"code": "NS", "display": f"{prefijo} · {dt_col.strftime('%I:%M %p')}", "is_live": False, "is_finished": False}
    except Exception:
        return {"code": "NS", "display": "HOY", "is_live": False, "is_finished": False}

async def fetch_historial_rapido(client: httpx.AsyncClient, team_id: int) -> List[Dict[str, Any]]:
    if team_id in CACHE_EQUIPOS:
        return CACHE_EQUIPOS[team_id]
        
    url = f"{BASE_URL}/fixtures?team={team_id}&last=5&status=FT"
    partidos = []
    try:
        r = await client.get(url, headers=HEADERS, timeout=5.0)
        if r.status_code == 200:
            for fix in r.json().get("response", []):
                teams = fix.get("teams", {})
                goals = fix.get("goals", {})
                is_home = teams.get("home", {}).get("id") == team_id
                rival = teams.get("away", {}).get("name") if is_home else teams.get("home", {}).get("name")
                gf = goals.get("home") if is_home else goals.get("away")
                gc = goals.get("away") if is_home else goals.get("home")
                
                partidos.append({
                    "rival": rival or "Rival",
                    "score": f"{gf if gf is not None else 0} - {gc if gc is not None else 0}",
                    "gf": gf if gf is not None else 1,
                    "gc": gc if gc is not None else 1,
                    "corn_fav": 5, "corn_con": 4,
                    "tarj_prop": 2, "tarj_prov": 2,
                    "rem_fav": 12, "rem_con": 10,
                    "resultado": "V" if (gf or 0) > (gc or 0) else ("E" if gf == gc else "D"),
                    "fecha": "Reciente"
                })
    except Exception:
        pass
    
    if not partidos:
        partidos = [{"rival": "Rival Genérico", "score": "1 - 1", "gf": 1, "gc": 1, "corn_fav": 5, "corn_con": 4, "tarj_prop": 2, "tarj_prov": 2, "rem_fav": 12, "rem_con": 10, "resultado": "E", "fecha": "Reciente"}]

    CACHE_EQUIPOS[team_id] = partidos
    return partidos

@app.get("/")
def root():
    return {"status": "ok", "service": "S2S Mass Concurrent Core Operational"}

@app.get("/api/v1/props")
async def get_props():
    tz_col = zoneinfo.ZoneInfo("America/Bogota")
    hoy_str = datetime.now(tz_col).strftime("%Y-%m-%d")
    
    url_dia = f"{BASE_URL}/fixtures?date={hoy_str}&timezone=America/Bogota"
    
    async with httpx.AsyncClient(timeout=50.0) as client:
        try:
            resp = await client.get(url_dia, headers=HEADERS)
            fixtures = resp.json().get("response", []) if resp.status_code == 200 else []

            # Si la fecha exacta viene acotada, traemos de inmediato un bloque masivo global ampliado
            if len(fixtures) < 15:
                url_next = f"{BASE_URL}/fixtures?next=150&timezone=America/Bogota"
                resp_next = await client.get(url_next, headers=HEADERS)
                fixtures = resp_next.json().get("response", []) if resp_next.status_code == 200 else []

            async def procesar_fixture(idx, fix):
                try:
                    fixture_data = fix.get("fixture", {})
                    league_data = fix.get("league", {})
                    teams_data = fix.get("teams", {})
                    
                    estado = parsear_estado_cronologico(fixture_data)
                    fix_id = str(fixture_data.get("id", idx))
                    
                    pais_raw = league_data.get("country", "")
                    liga_nombre_raw = league_data.get("name", "Liga").upper()
                    bandera_emoji, pais_formateado = obtener_pais_y_bandera(pais_raw, liga_nombre_raw)
                    liga_agrupada = f"{bandera_emoji}  {pais_formateado} • {liga_nombre_raw.title()}"
                    
                    home_id = teams_data.get("home", {}).get("id", 0)
                    away_id = teams_data.get("away", {}).get("id", 0)
                    home_name = teams_data.get("home", {}).get("name", "Local")
                    away_name = teams_data.get("away", {}).get("name", "Visita")
                    
                    f_home_real = await fetch_historial_rapido(client, home_id)
                    f_away_real = await fetch_historial_rapido(client, away_id)
                    
                    return {
                        "id": fix_id,
                        "deporte": "FÚTBOL",
                        "pais": pais_formateado,
                        "bandera": bandera_emoji,
                        "liga": liga_agrupada,
                        "evento": f"{home_name} vs {away_name}",
                        "status_code": estado["code"],
                        "status_display": estado["display"],
                        "is_live": estado["is_live"],
                        "is_finished": estado["is_finished"],
                        "score_real": None,
                        "status_verdict": "PENDIENTE",
                        "home_name": home_name, "away_name": away_name,
                        "home_logo": teams_data.get("home", {}).get("logo", ""),
                        "away_logo": teams_data.get("away", {}).get("logo", ""),
                        "p_home": 45, "p_draw": 30, "p_away": 25,
                        "prob_1x2": "45% • 30% • 25%",
                        "marcador_estimado": "1 - 1",
                        "cr_mercado": "75%",
                        "cr_score_num": "75",
                        "cr_home_casa": "75%",
                        "cr_away_fora": "70%",
                        "cr_combinado_split": "72%",
                        "mercado": "MÁS DE 1.5 GOLES",
                        "linea": 1.5,
                        "proyeccion_val": "2.45",
                        "promedio_l10": 2.45,
                        "home_goles": f_home_real,
                        "away_goles": f_away_real,
                        "home_corners": f_home_real,
                        "away_corners": f_home_real,
                        "home_tarjetas": f_home_real,
                        "away_tarjetas": f_home_real,
                        "home_remates": f_home_real,
                        "away_remates": f_home_real,
                        "home_btts": f_home_real,
                        "away_btts": f_home_real,
                        "split_vs_list": [],
                        "h2h_matches": f_home_real,
                        "home_matches_20": f_home_real,
                        "away_matches_20": f_away_real,
                        "corners_label": "MÁS DE 8.5 CÓRNERS",
                        "corners_conf": 68.0,
                        "corners_proyeccion": "9.5",
                        "tarjetas_label": "MENOS DE 4.5 TARJETAS",
                        "tarjetas_conf": 70.0,
                        "tarjetas_proyeccion": "3.8",
                        "disparos_label": "MÁS DE 10.5 REMATES",
                        "disparos_conf": 72.0,
                        "disparos_proyeccion": "13.2",
                        "btts_label": "AMBOS ANOTAN: SÍ",
                        "btts_conf": 65.0,
                        "btts_proyeccion": "1.3 - 1.1"
                    }
                except Exception:
                    return None

            # Procesamos todos los partidos en paralelo de forma masiva y rápida
            resultados = await asyncio.gather(*(procesar_fixture(idx, fix) for idx, fix in enumerate(fixtures)))
            partidos_consolidados = [p for p in resultados if p is not None]

        except Exception as e:
            print(f"[ERROR MASSIVE API]: {e}")
            return []

    estado_orden = {"LIVE": 0, "NS": 1, "FT": 2}
    return sorted(partidos_consolidados, key=lambda x: (x.get("pais", "Z"), estado_orden.get(x.get("status_code", "NS"), 1)))
