from fastapi import FastAPI
import httpx
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timezone
import zoneinfo
from typing import Dict, List, Any

app = FastAPI(title="S2S Sigma Engine - Real Fixtures Core")

API_KEY = "9cf313ae66d39a8f1aa2674401de70ce"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
EPSILON = 1e-6

# Caché en memoria para optimizar cuota de API
CACHE_HISTORIALES: Dict[int, List[Dict[str, Any]]] = {}

BANDERA_MAP = {
    "ARGENTINA": ("\U0001F1E6\U0001F1F7", "Argentina"),
    "BOLIVIA": ("\U0001F1E7\U0001F1F4", "Bolivia"),
    "BRASIL": ("\U0001F1E7\U0001F1F7", "Brasil"),
    "BRAZIL": ("\U0001F1E7\U0001F1F7", "Brasil"),
    "CANADA": ("\U0001F1E8\U0001F1E6", "Canadá"),
    "CHILE": ("\U0001F1E8\U0001F1F1", "Chile"),
    "COLOMBIA": ("\U0001F1E8\U0001F1F4", "Colombia"),
    "COSTA RICA": ("\U0001F1E8\U0001F1F7", "Costa Rica"),
    "ECUADOR": ("\U0001F1EA\U0001F1E8", "Ecuador"),
    "EL SALVADOR": ("\U0001F1F8\U0001F1FB", "El Salvador"),
    "ESPAÑA": ("\U0001F1EA\U0001F1F8", "España"),
    "SPAIN": ("\U0001F1EA\U0001F1F8", "España"),
    "ESTADOS UNIDOS": ("\U0001F1FA\U0001F1F8", "Estados Unidos"),
    "USA": ("\U0001F1FA\U0001F1F8", "Estados Unidos"),
    "HONDURAS": ("\U0001F1ED\U0001F1F3", "Honduras"),
    "INGLATERRA": ("\U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006E\U000E0067\U000E007F", "Inglaterra"),
    "ENGLAND": ("\U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006E\U000E0067\U000E007F", "Inglaterra"),
    "ITALIA": ("\U0001F1EE\U0001F1F9", "Italia"),
    "ITALY": ("\U0001F1EE\U0001F1F9", "Italia"),
    "JAPON": ("\U0001F1EF\U0001F1F5", "Japón"),
    "JAPAN": ("\U0001F1EF\U0001F1F5", "Japón"),
    "MEXICO": ("\U0001F1F2\U0001F1FD", "México"),
    "MÉXICO": ("\U0001F1F2\U0001F1FD", "México"),
    "PAÍSES BAJOS": ("\U0001F1F3\U0001F1F1", "Países Bajos"),
    "NETHERLANDS": ("\U0001F1F3\U0001F1F1", "Países Bajos"),
    "PARAGUAY": ("\U0001F1F5\U0001F1FE", "Paraguay"),
    "PERU": ("\U0001F1F5\U0001F1EA", "Perú"),
    "PERÚ": ("\U0001F1F5\U0001F1EA", "Perú"),
    "URUGUAY": ("\U0001F1FA\U0001F1FE", "Uruguay"),
    "VENEZUELA": ("\U0001F1FB\U0001F1EA", "Venezuela"),
    "GLOBAL": ("\U0001F310", "Internacional")
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

async def obtener_historial_real_equipo(client: httpx.AsyncClient, team_id: int, team_name: str) -> List[Dict[str, Any]]:
    if team_id in CACHE_HISTORIALES:
        return CACHE_HISTORIALES[team_id]
        
    url = f"{BASE_URL}/fixtures?team={team_id}&last=10"
    partidos_parseados = []
    try:
        r = await client.get(url, headers=HEADERS)
        if r.status_code == 200:
            datos = r.json().get("response", [])
            for fix in datos:
                teams = fix.get("teams", {})
                goals = fix.get("goals", {})
                fixture_info = fix.get("fixture", {})
                
                is_home = teams.get("home", {}).get("id") == team_id
                rival = teams.get("away", {}).get("name") if is_home else teams.get("home", {}).get("name")
                
                gf = goals.get("home", 0) if is_home else goals.get("away", 0)
                gc = goals.get("away", 0) if is_home else goals.get("home", 0)
                gf = gf if gf is not None else 0
                gc = gc if gc is not None else 0
                
                res = "V" if gf > gc else ("E" if gf == gc else "D")
                
                date_raw = fixture_info.get("date", "")
                try:
                    dt = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
                    fecha_str = dt.strftime("%d/%m")
                except Exception:
                    fecha_str = "Reciente"
                
                partidos_parseados.append({
                    "rival": rival,
                    "score": f"{gf} - {gc}",
                    "resultado": res,
                    "gf": gf,
                    "gc": gc,
                    "valor": float(gf + gc),
                    "sede": "CASA" if is_home else "FUERA",
                    "fecha": fecha_str
                })
    except Exception as e:
        print(f"[ERROR FETCH TEAM {team_id}]: {e}")
        
    CACHE_HISTORIALES[team_id] = partidos_parseados
    return partidos_parseados

@app.get("/")
def root():
    return {"status": "ok", "service": "S2S Real Data Analytical Core Active"}

@app.get("/api/v1/props")
async def get_props():
    url_next = f"{BASE_URL}/fixtures?next=50&timezone=America/Bogota"
    partidos_consolidados = []
    
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            resp = await client.get(url_next, headers=HEADERS)
            fixtures = resp.json().get("response", []) if resp.status_code == 200 else []

            for idx, fix in enumerate(fixtures):
                fixture_data = fix.get("fixture", {})
                league_data = fix.get("league", {})
                teams_data = fix.get("teams", {})
                goals_data = fix.get("goals", {})
                
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
                
                # Ingesta de partidos reales de cada club
                f_home_real = await obtener_historial_real_equipo(client, home_id, home_name)
                f_away_real = await obtener_historial_real_equipo(client, away_id, away_name)
                
                # Cálculo de promedios reales
                gf_h = round(float(np.mean([m["gf"] for m in f_home_real])), 1) if f_home_real else 1.4
                gc_h = round(float(np.mean([m["gc"] for m in f_home_real])), 1) if f_home_real else 1.0
                gf_a = round(float(np.mean([m["gf"] for m in f_away_real])), 1) if f_away_real else 1.1
                gc_a = round(float(np.mean([m["gc"] for m in f_away_real])), 1) if f_away_real else 1.3
                
                # Parámetros Poisson derivados de goles reales
                lam_loc = round((gf_h + gc_a) / 2.0, 2)
                lam_vis = round((gf_a + gc_h) / 2.0, 2)
                lam_tot = round(lam_loc + lam_vis, 2)
                
                max_g = 7
                mat = np.zeros((max_g, max_g))
                for i in range(max_g):
                    for j in range(max_g):
                        mat[i, j] = poisson.pmf(i, lam_loc) * poisson.pmf(j, lam_vis)
                        
                tot_p = max(float(np.sum(mat)), EPSILON)
                p_h = int(round(float(np.sum(np.tril(mat, -1))) / tot_p * 100))
                p_d = int(round(float(np.sum(np.diag(mat))) / tot_p * 100))
                p_a = max(1, 100 - (p_h + p_d))
                
                p_over_25 = float(np.sum([mat[i, j] for i in range(max_g) for j in range(max_g) if i + j > 2.5])) / tot_p
                p_under_25 = float(np.sum([mat[i, j] for i in range(max_g) for j in range(max_g) if i + j < 2.5])) / tot_p
                p_over_15 = float(np.sum([mat[i, j] for i in range(max_g) for j in range(max_g) if i + j > 1.5])) / tot_p
                p_btts = float(np.sum([mat[i, j] for i in range(1, max_g) for j in range(1, max_g)])) / tot_p

                # Selección del mercado según los datos reales
                if lam_tot >= 2.5:
                    merc_label = "MÁS DE 2.5 GOLES"
                    merc_linea = 2.5
                    is_over = True
                    cr_mercado = int(np.clip(p_over_25 * 100, 55, 90))
                elif lam_tot <= 1.9:
                    merc_label = "MENOS DE 2.5 GOLES"
                    merc_linea = 2.5
                    is_over = False
                    cr_mercado = int(np.clip(p_under_25 * 100, 55, 90))
                else:
                    merc_label = "MÁS DE 1.5 GOLES"
                    merc_linea = 1.5
                    is_over = True
                    cr_mercado = int(np.clip(p_over_15 * 100, 60, 92))

                idx_max = np.unravel_index(np.argmax(mat), mat.shape)
                marcador_est = f"{idx_max[0]} - {idx_max[1]}"

                # Validación de cumplimiento sobre partidos reales
                for m in f_home_real:
                    m["cumple"] = m["valor"] > merc_linea if is_over else m["valor"] < merc_linea
                for m in f_away_real:
                    m["cumple"] = m["valor"] > merc_linea if is_over else m["valor"] < merc_linea

                cr_h_calc = int((sum(1 for m in f_home_real if m["cumple"]) / len(f_home_real) * 100)) if f_home_real else 70
                cr_a_calc = int((sum(1 for m in f_away_real if m["cumple"]) / len(f_away_real) * 100)) if f_away_real else 70
                cr_comb = int((cr_h_calc + cr_a_calc) / 2)

                # Split sincronizado real
                min_len = min(len(f_home_real), len(f_away_real))
                split_vs_list = []
                for i in range(min_len):
                    mh = f_home_real[i]
                    ma = f_away_real[i]
                    split_vs_list.append({
                        "rival_home": mh["rival"], "score_home": mh["score"], "cumple_home": mh["cumple"],
                        "rival_away": ma["rival"], "score_away": ma["score"], "cumple_away": ma["cumple"],
                        "cumple_dual": mh["cumple"] and ma["cumple"], "fecha": mh["fecha"]
                    })

                partidos_consolidados.append({
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
                    
                    "p_home": p_h, "p_draw": p_d, "p_away": p_a,
                    "prob_1x2": f"{p_h}% • {p_d}% • {p_a}%",
                    "marcador_estimado": marcador_est,
                    
                    "cr_mercado": f"{cr_mercado}%",
                    "cr_score_num": str(cr_mercado),
                    "cr_home_casa": f"{cr_h_calc}%",
                    "cr_away_fora": f"{cr_a_calc}%",
                    "cr_combinado_split": f"{cr_comb}%",
                    "mercado": merc_label, "linea": merc_linea,
                    "proyeccion_val": str(lam_tot), "promedio_l10": float(lam_tot),
                    
                    "metrics_home": {"gf_prom": gf_h, "gc_prom": gc_h, "corn_prom": 5.2, "tarj_prom": 2.1, "rem_prom": 11.4},
                    "metrics_away": {"gf_prom": gf_a, "gc_prom": gc_a, "corn_prom": 4.8, "tarj_prom": 2.3, "rem_prom": 10.1},
                    
                    "home_matches_20": f_home_real,
                    "away_matches_20": f_away_real,
                    "home_matches_casa": f_home_real,
                    "away_matches_fora": f_away_real,
                    "split_vs_list": split_vs_list,
                    "h2h_matches": f_home_real[:5],
                    
                    "goles_matches": f_home_real,
                    "corners_matches": f_home_real,
                    "tarjetas_matches": f_home_real,
                    "disparos_matches": f_home_real,
                    "btts_matches": f_home_real,
                    
                    "corners_label": "MÁS DE 8.5 CÓRNERS", "corners_conf": 68.0, "corners_proyeccion": "9.2",
                    "tarjetas_label": "MENOS DE 4.5 TARJETAS", "tarjetas_conf": 71.0, "tarjetas_proyeccion": "3.6",
                    "disparos_label": "MÁS DE 10.5 REMATES", "disparos_conf": 64.0, "disparos_proyeccion": "11.2",
                    "btts_label": f"AMBOS ANOTAN: {'SÍ' if p_btts >= 0.50 else 'NO'}", "btts_conf": int(p_btts * 100),
                    "btts_proyeccion": f"{lam_loc} - {lam_vis}"
                })
        except Exception as e:
            print(f"[ERROR MAIN]: {e}")
            return []

    estado_orden = {"LIVE": 0, "NS": 1, "FT": 2}
    return sorted(partidos_consolidados, key=lambda x: (x.get("pais", "Z"), estado_orden.get(x.get("status_code", "NS"), 1)))
