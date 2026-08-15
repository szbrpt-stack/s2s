from fastapi import FastAPI
import httpx
import numpy as np
from scipy.stats import poisson
from datetime import datetime
import zoneinfo
from typing import Dict, List, Any
import asyncio

app = FastAPI(title="S2S Sigma Engine - Formal Statistical & Deep API Core")

API_KEY = "9cf313ae66d39a8f1aa2674401de70ce"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
EPSILON = 1e-6
RHO_DIXON_COLES = -0.13

# Medias de referencia internacional
MU_GOLES_LOCAL = 1.45
MU_GOLES_VISITA = 1.15
MU_CORNERS_LIGA = 9.80
MU_TARJETAS_LIGA = 4.30

# Caché en memoria para optimizar peticiones
CACHE_EQUIPOS: Dict[int, List[Dict[str, Any]]] = {}

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

def tau_dixon_coles(x: int, y: int, lambda_h: float, lambda_a: float, rho: float = RHO_DIXON_COLES) -> float:
    if x == 0 and y == 0:
        return 1.0 - (lambda_h * lambda_a * rho)
    elif x == 1 and y == 0:
        return 1.0 + (lambda_a * rho)
    elif x == 0 and y == 1:
        return 1.0 + (lambda_h * rho)
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0

async def fetch_historial_profundo(client: httpx.AsyncClient, team_id: int) -> List[Dict[str, Any]]:
    if team_id in CACHE_EQUIPOS:
        return CACHE_EQUIPOS[team_id]
        
    url = f"{BASE_URL}/fixtures?team={team_id}&last=10&status=FT"
    partidos_procesados = []
    try:
        r = await client.get(url, headers=HEADERS, timeout=12.0)
        if r.status_code == 200:
            fixtures = r.json().get("response", [])
            for fix in fixtures:
                fix_id = fix.get("fixture", {}).get("id")
                teams = fix.get("teams", {})
                goals = fix.get("goals", {})
                fix_info = fix.get("fixture", {})
                
                is_home = teams.get("home", {}).get("id") == team_id
                rival = teams.get("away", {}).get("name") if is_home else teams.get("home", {}).get("name")
                
                gf = goals.get("home") if is_home else goals.get("away")
                gc = goals.get("away") if is_home else goals.get("home")
                gf = gf if gf is not None else 0
                gc = gc if gc is not None else 0
                
                url_stats = f"{BASE_URL}/fixtures/statistics?fixture={fix_id}"
                corn_fav, corn_con = 5, 4
                tarj_prop, tarj_prov = 2, 2
                rem_fav, rem_con = 11, 10
                
                try:
                    r_stats = await client.get(url_stats, headers=HEADERS, timeout=6.0)
                    if r_stats.status_code == 200:
                        stats_resp = r_stats.json().get("response", [])
                        for team_stat in stats_resp:
                            t_id = team_stat.get("team", {}).get("id")
                            s_list = {item["type"]: item["value"] for item in team_stat.get("statistics", [])}
                            
                            c_val = s_list.get("Corner Kicks") or 0
                            y_card = s_list.get("Yellow Cards") or 0
                            r_card = s_list.get("Red Cards") or 0
                            t_shots = s_list.get("Total Shots") or 0
                            
                            if t_id == team_id:
                                corn_fav = int(c_val)
                                tarj_prop = int(y_card) + (int(r_card) * 2)
                                rem_fav = int(t_shots)
                            else:
                                corn_con = int(c_val)
                                tarj_prov = int(y_card) + (int(r_card) * 2)
                                rem_con = int(t_shots)
                except Exception:
                    pass

                dt_str = fix_info.get("date", "")
                try:
                    fecha_str = datetime.fromisoformat(dt_str.replace("Z", "+00:00")).strftime("%d/%m")
                except Exception:
                    fecha_str = "Reciente"
                
                res = "V" if gf > gc else ("E" if gf == gc else "D")
                
                partidos_procesados.append({
                    "rival": rival or "Rival Oficial",
                    "score": f"{gf} - {gc}",
                    "resultado": res,
                    "gf": gf, "gc": gc,
                    "corn_fav": corn_fav, "corn_con": corn_con,
                    "tarj_prop": tarj_prop, "tarj_prov": tarj_prov,
                    "rem_fav": rem_fav, "rem_con": rem_con,
                    "sede": "CASA" if is_home else "FUERA",
                    "fecha": fecha_str
                })
    except Exception as e:
        print(f"[API ERROR FETCH TEAM {team_id}]: {e}")
        
    CACHE_EQUIPOS[team_id] = partidos_procesados
    return partidos_procesados

@app.get("/")
def root():
    return {"status": "ok", "service": "S2S Formal Metrology Core Operational"}

@app.get("/api/v1/props")
async def get_props():
    url_next = f"{BASE_URL}/fixtures?next=30&timezone=America/Bogota"
    partidos_consolidados = []
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(url_next, headers=HEADERS)
            fixtures = resp.json().get("response", []) if resp.status_code == 200 else []

            for idx, fix in enumerate(fixtures):
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
                
                f_home_real = await fetch_historial_profundo(client, home_id)
                f_away_real = await fetch_historial_profundo(client, away_id)
                
                if not f_home_real or not f_away_real:
                    continue

                # 1. Normalización de Fuerzas Relativas (Goles)
                gf_h_mean = np.mean([m["gf"] for m in f_home_real])
                gc_h_mean = np.mean([m["gc"] for m in f_home_real])
                gf_a_mean = np.mean([m["gf"] for m in f_away_real])
                gc_a_mean = np.mean([m["gc"] for m in f_away_real])
                
                alpha_h = max(0.2, gf_h_mean / MU_GOLES_LOCAL)
                beta_h = max(0.2, gc_h_mean / MU_GOLES_VISITA)
                alpha_a = max(0.2, gf_a_mean / MU_GOLES_VISITA)
                beta_a = max(0.2, gc_a_mean / MU_GOLES_LOCAL)
                
                lambda_h = round(alpha_h * beta_a * MU_GOLES_LOCAL, 2)
                lambda_a = round(alpha_a * beta_h * MU_GOLES_VISITA, 2)
                lambda_tot = round(lambda_h + lambda_a, 2)
                
                max_g = 7
                mat = np.zeros((max_g, max_g))
                for x in range(max_g):
                    for y in range(max_g):
                        p_base = (poisson.pmf(x, lambda_h) * poisson.pmf(y, lambda_a))
                        mat[x, y] = p_base * tau_dixon_coles(x, y, lambda_h, lambda_a)
                        
                tot_p = max(float(np.sum(mat)), EPSILON)
                mat /= tot_p
                
                p_h = int(round(float(np.sum(np.tril(mat, -1))) * 100))
                p_d = int(round(float(np.sum(np.diag(mat))) * 100))
                p_a = max(1, 100 - (p_h + p_d))
                
                p_o25 = float(np.sum([mat[x, y] for x in range(max_g) for y in range(max_g) if x + y > 2.5]))
                p_u25 = 1.0 - p_o25
                p_o15 = float(np.sum([mat[x, y] for x in range(max_g) for y in range(max_g) if x + y > 1.5]))
                p_u15 = 1.0 - p_o15
                p_btts = float(np.sum([mat[x, y] for x in range(1, max_g) for y in range(1, max_g)]))

                # Regla de Viabilidad Dominante (CR >= 50%)
                if p_o25 >= 0.50:
                    merc_label = "MÁS DE 2.5 GOLES"
                    merc_linea = 2.5
                    is_over = True
                    prob_teo = p_o25
                    marcador_est = "2 - 1" if p_h >= p_a else ("1 - 2" if p_a > p_h else "2 - 2")
                elif p_u25 >= 0.50:
                    merc_label = "MENOS DE 2.5 GOLES"
                    merc_linea = 2.5
                    is_over = False
                    prob_teo = p_u25
                    marcador_est = "1 - 0" if p_h > p_a else ("0 - 1" if p_a > p_h else "1 - 1")
                elif p_o15 >= 0.60:
                    merc_label = "MÁS DE 1.5 GOLES"
                    merc_linea = 1.5
                    is_over = True
                    prob_teo = p_o15
                    marcador_est = "2 - 0" if p_h > p_a else ("0 - 2" if p_a > p_h else "1 - 1")
                else:
                    merc_label = "MENOS DE 1.5 GOLES"
                    merc_linea = 1.5
                    is_over = False
                    prob_teo = p_u15
                    marcador_est = "1 - 0" if p_h >= p_a else "0 - 1"

                cump_h = sum(1 for m in f_home_real if ((m["gf"] + m["gc"] > merc_linea) if is_over else (m["gf"] + m["gc"] < merc_linea))) / len(f_home_real)
                cump_a = sum(1 for m in f_away_real if ((m["gf"] + m["gc"] > merc_linea) if is_over else (m["gf"] + m["gc"] < merc_linea))) / len(f_away_real)
                cump_empirico = (cump_h + cump_a) / 2.0
                
                cr_mercado = int(np.clip((0.70 * prob_teo + 0.30 * cump_empirico) * 100, 50, 96))

                # 2. Modelo de Córners
                cf_h_mean = np.mean([m["corn_fav"] for m in f_home_real])
                cc_h_mean = np.mean([m["corn_con"] for m in f_home_real])
                cf_a_mean = np.mean([m["corn_fav"] for m in f_away_real])
                cc_a_mean = np.mean([m["corn_con"] for m in f_away_real])
                
                exp_corn_tot = round(((cf_h_mean * cc_a_mean) / MU_CORNERS_LIGA) + ((cf_a_mean * cc_h_mean) / MU_CORNERS_LIGA), 1)
                exp_corn_tot = max(6.0, exp_corn_tot)
                
                if exp_corn_tot >= 9.5:
                    corn_label = "MÁS DE 8.5 CÓRNERS"
                    corn_conf = int(np.clip((exp_corn_tot / 13.5) * 100, 52, 90))
                else:
                    corn_label = "MENOS DE 10.5 CÓRNERS"
                    corn_conf = int(np.clip((1.0 - (exp_corn_tot / 16.0)) * 100, 52, 90))

                # 3. Modelo de Tarjetas
                tp_h_mean = np.mean([m["tarj_prop"] for m in f_home_real])
                tprov_h_mean = np.mean([m["tarj_prov"] for m in f_home_real])
                tp_a_mean = np.mean([m["tarj_prop"] for m in f_away_real])
                tprov_a_mean = np.mean([m["tarj_prov"] for m in f_away_real])
                
                exp_tarj_tot = round((tp_h_mean + tprov_a_mean + tp_a_mean + tprov_h_mean) / 2.0, 1)
                exp_tarj_tot = max(2.0, exp_tarj_tot)
                
                if exp_tarj_tot <= 4.2:
                    tarj_label = "MENOS DE 4.5 TARJETAS"
                    tarj_conf = int(np.clip((1.0 - (exp_tarj_tot / 7.5)) * 100, 52, 90))
                else:
                    tarj_label = "MÁS DE 3.5 TARJETAS"
                    tarj_conf = int(np.clip((exp_tarj_tot / 6.5) * 100, 52, 90))

                # 4. Remates y Ambos Anotan
                rf_h_mean = np.mean([m["rem_fav"] for m in f_home_real])
                rc_a_mean = np.mean([m["rem_con"] for m in f_away_real])
                rf_a_mean = np.mean([m["rem_fav"] for m in f_away_real])
                rc_h_mean = np.mean([m["rem_con"] for m in f_home_real])
                exp_rem_tot = round((rf_h_mean + rc_a_mean + rf_a_mean + rc_h_mean) / 2.0, 1)
                exp_rem_tot = max(14.0, exp_rem_tot)
                
                rem_label = "MÁS DE 10.5 REMATES" if exp_rem_tot >= 16.0 else "MENOS DE 18.5 REMATES"
                rem_conf = int(np.clip(55 + (exp_rem_tot / 35.0) * 35, 52, 90))

                btts_recom = "SÍ" if p_btts >= 0.50 else "NO"
                btts_cr = int(np.clip((p_btts if btts_recom == "SÍ" else (1.0 - p_btts)) * 100, 50, 92))
                btts_label = f"AMBOS ANOTAN: {btts_recom}"

                # Listas por mercado
                home_goles = [{"rival": m["rival"], "score": m["score"], "resultado": m["resultado"], "cumple": ((m["gf"] + m["gc"] > merc_linea) if is_over else (m["gf"] + m["gc"] < merc_linea)), "fecha": m["fecha"]} for m in f_home_real]
                away_goles = [{"rival": m["rival"], "score": m["score"], "resultado": m["resultado"], "cumple": ((m["gf"] + m["gc"] > merc_linea) if is_over else (m["gf"] + m["gc"] < merc_linea)), "fecha": m["fecha"]} for m in f_away_real]

                home_corners = [{"rival": m["rival"], "score": f"{m['corn_fav']} córners", "resultado": m["resultado"], "cumple": m["corn_fav"] > 4, "fecha": m["fecha"]} for m in f_home_real]
                away_corners = [{"rival": m["rival"], "score": f"{m['corn_fav']} córners", "resultado": m["resultado"], "cumple": m["corn_fav"] > 4, "fecha": m["fecha"]} for m in f_away_real]

                home_tarjetas = [{"rival": m["rival"], "score": f"{m['tarj_prop']} tarjetas", "resultado": m["resultado"], "cumple": m["tarj_prop"] < 3, "fecha": m["fecha"]} for m in f_home_real]
                away_tarjetas = [{"rival": m["rival"], "score": f"{m['tarj_prop']} tarjetas", "resultado": m["resultado"], "cumple": m["tarj_prop"] < 3, "fecha": m["fecha"]} for m in f_away_real]

                home_remates = [{"rival": m["rival"], "score": f"{m['rem_fav']} remates", "resultado": m["resultado"], "cumple": m["rem_fav"] > 9, "fecha": m["fecha"]} for m in f_home_real]
                away_remates = [{"rival": m["rival"], "score": f"{m['rem_fav']} remates", "resultado": m["resultado"], "cumple": m["rem_fav"] > 9, "fecha": m["fecha"]} for m in f_away_real]

                home_btts = [{"rival": m["rival"], "score": m["score"], "resultado": m["resultado"], "cumple": (m["gf"] > 0 and m["gc"] > 0) if btts_recom == "SÍ" else (m["gf"] == 0 or m["gc"] == 0), "fecha": m["fecha"]} for m in f_home_real]
                away_btts = [{"rival": m["rival"], "score": m["score"], "resultado": m["resultado"], "cumple": (m["gf"] > 0 and m["gc"] > 0) if btts_recom == "SÍ" else (m["gf"] == 0 or m["gc"] == 0), "fecha": m["fecha"]} for m in f_away_real]

                # Fila Dual Paralela
                min_len = min(len(f_home_real), len(f_away_real))
                split_vs_list = []
                for i in range(min_len):
                    mh = f_home_real[i]
                    ma = f_away_real[i]
                    c_gh = (mh["gf"] + mh["gc"] > merc_linea) if is_over else (mh["gf"] + mh["gc"] < merc_linea)
                    c_ga = (ma["gf"] + ma["gc"] > merc_linea) if is_over else (ma["gf"] + ma["gc"] < merc_linea)
                    
                    split_vs_list.append({
                        "rival_home": mh["rival"], "score_home": mh["score"], "cumple_home": bool(c_gh),
                        "rival_away": ma["rival"], "score_away": ma["score"], "cumple_away": bool(c_ga),
                        "cumple_dual": bool(c_gh and c_ga),
                        "corners_home": f"{mh['corn_fav']} córners", "cumple_corners_h": bool(mh['corn_fav'] > 4),
                        "corners_away": f"{ma['corn_fav']} córners", "cumple_corners_a": bool(ma['corn_fav'] > 4),
                        "tarj_home": f"{mh['tarj_prop']} tarjetas", "cumple_tarj_h": bool(mh['tarj_prop'] < 3),
                        "tarj_away": f"{ma['tarj_prop']} tarjetas", "cumple_tarj_a": bool(ma['tarj_prop'] < 3),
                        "rem_home": f"{mh['rem_fav']} remates", "cumple_rem_h": bool(mh['rem_fav'] > 9),
                        "rem_away": f"{ma['rem_fav']} remates", "cumple_rem_a": bool(ma['rem_fav'] > 9),
                        "fecha": mh["fecha"]
                    })

                cr_h_disp = int(cump_h * 100)
                cr_a_disp = int(cump_a * 100)
                cr_comb_disp = int((cr_h_disp + cr_a_disp) / 2)

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
                    "cr_home_casa": f"{cr_h_disp}%",
                    "cr_away_fora": f"{cr_a_disp}%",
                    "cr_combinado_split": f"{cr_comb_disp}%",
                    "mercado": merc_label, "linea": merc_linea,
                    "proyeccion_val": str(lambda_tot), "promedio_l10": float(lambda_tot),
                    
                    "home_goles": home_goles, "away_goles": away_goles,
                    "home_corners": home_corners, "away_corners": away_corners,
                    "home_tarjetas": home_tarjetas, "away_tarjetas": away_tarjetas,
                    "home_remates": home_remates, "away_remates": away_remates,
                    "home_btts": home_btts, "away_btts": away_btts,
                    
                    "split_vs_list": split_vs_list,
                    "h2h_matches": home_goles[:5],
                    "home_matches_20": home_goles,
                    "away_matches_20": away_goles,
                    
                    "corners_label": corn_label, "corners_conf": float(corn_conf), "corners_proyeccion": str(exp_corn_tot),
                    "tarjetas_label": tarj_label, "tarjetas_conf": float(tarj_conf), "tarjetas_proyeccion": str(exp_tarj_tot),
                    "disparos_label": rem_label, "disparos_conf": float(rem_conf), "disparos_proyeccion": str(exp_rem_tot),
                    "btts_label": btts_label, "btts_conf": float(btts_cr), "btts_proyeccion": f"{lambda_h} - {lambda_a}"
                })
        except Exception as e:
            print(f"[ERROR PIPELINE]: {e}")
            return []

    estado_orden = {"LIVE": 0, "NS": 1, "FT": 2}
    return sorted(partidos_consolidados, key=lambda x: (x.get("pais", "Z"), estado_orden.get(x.get("status_code", "NS"), 1)))
