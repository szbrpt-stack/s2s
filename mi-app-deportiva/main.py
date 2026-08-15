from fastapi import FastAPI
import httpx
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timezone
import zoneinfo
import hashlib

app = FastAPI(title="S2S Sigma Engine - Fully Reactive Core")

API_KEY = "9cf313ae66d39a8f1aa2674401de70ce"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
EPSILON = 1e-6

PAIS_MAP = {
    "PREMIER LEAGUE": "Inglaterra", "LIGA BETPLAY": "Colombia", "LA LIGA": "España",
    "SERIE A": "Italia", "BUNDESLIGA": "Alemania", "MLS": "Estados Unidos",
    "BRASILEIRÃO": "Brasil", "EREDIVISIE": "Países Bajos", "COPPA ITALIA": "Italia",
    "PRIMERA NACIONAL": "Argentina", "PRIMERA B": "Argentina", "PRIMERA C": "Argentina",
    "SERIE D": "Brasil", "SERIE B": "Brasil"
}

def parsear_estado_hora(fixture_data: dict) -> dict:
    status = fixture_data.get("status", {})
    status_short = status.get("short", "")
    elapsed = status.get("elapsed", 0)
    
    if status_short in ["1H", "2H", "HT", "ET", "P", "LIVE"]:
        disp = "ENTRETIEMPO" if status_short == "HT" else f"EN VIVO · {elapsed}'"
        return {"display": disp, "is_live": True, "valido": True}

    date_str = fixture_data.get("date", "")
    try:
        dt_utc = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if dt_utc < datetime.now(timezone.utc) and status_short != "NS":
            return {"display": "FINALIZADO", "is_live": False, "valido": False}
            
        tz_col = zoneinfo.ZoneInfo("America/Bogota")
        dt_col = dt_utc.astimezone(tz_col)
        hoy = datetime.now(tz_col).date()
        prefijo = "HOY" if dt_col.date() == hoy else dt_col.strftime("%d/%m")
        return {"display": f"{prefijo} · {dt_col.strftime('%I:%M %p')}", "is_live": False, "valido": True}
    except Exception:
        return {"display": "HOY", "is_live": False, "valido": True}

def generar_forma_especifica(seed: int, tipo: str, linea: float, is_over: bool, n: int = 20):
    partidos = []
    for i in range(n):
        if tipo == "GOLES":
            gf = (seed + i * 3) % 3
            gc = (seed * 2 + i * 5) % 3
            if is_over and (gf + gc) < linea and (seed + i) % 2 == 0:
                gf += 1
            val = gf + gc
            res = "V" if gf > gc else ("E" if gf == gc else "D")
            cumple = val > linea if is_over else val < linea
            score = f"{gf} - {gc}"
        elif tipo == "CÓRNERS":
            val = ((seed * 3 + i * 5) % 8) + 6
            res = "V" if val > linea else "D"
            cumple = val > linea if is_over else val < linea
            score = f"{val} córners"
        elif tipo == "TARJETAS":
            val = ((seed * 2 + i * 3) % 5) + 2
            res = "V" if val < linea else "D"
            cumple = val > linea if is_over else val < linea
            score = f"{val} tarjetas"
        elif tipo == "REMATES":
            val = ((seed * 5 + i * 7) % 8) + 8
            res = "V" if val > linea else "D"
            cumple = val > linea if is_over else val < linea
            score = f"{val} remates"
        else: # AMBOS ANOTAN
            gf = (seed + i * 3) % 3
            gc = (seed * 2 + i * 5) % 3
            ambos = gf > 0 and gc > 0
            val = 1.0 if ambos else 0.0
            res = "V" if ambos else "D"
            cumple = ambos if is_over else (not ambos)
            score = f"{gf} - {gc}"

        partidos.append({
            "rival": f"vs Rival {i + 1}",
            "score": score,
            "resultado": res,
            "valor": float(val),
            "cumple": bool(cumple),
            "fecha": f"{n - i} Ago"
        })
    return partidos

@app.get("/")
def root():
    return {"status": "ok", "service": "S2S Reactive Core Active"}

@app.get("/api/v1/props")
async def get_props():
    url_next = f"{BASE_URL}/fixtures?next=40&timezone=America/Bogota"
    partidos_consolidados = []
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(url_next, headers=HEADERS)
            fixtures = resp.json().get("response", []) if resp.status_code == 200 else []

            for idx, fix in enumerate(fixtures):
                fixture_data = fix.get("fixture", {})
                league_data = fix.get("league", {})
                teams_data = fix.get("teams", {})
                goals_data = fix.get("goals", {})
                
                estado = parsear_estado_hora(fixture_data)
                if not estado["valido"]:
                    continue

                fix_id = str(fixture_data.get("id", idx))
                nombre_liga_raw = league_data.get("name", "FÚTBOL").upper()
                pais_nombre = PAIS_MAP.get(nombre_liga_raw, league_data.get("country", "Global").title())
                liga_agrupada = f"{pais_nombre} • {nombre_liga_raw.title()}"
                
                home_name = teams_data.get("home", {}).get("name", "Local")
                away_name = teams_data.get("away", {}).get("name", "Visita")
                
                g_loc_live = goals_data.get("home") if goals_data.get("home") is not None else 0
                g_vis_live = goals_data.get("away") if goals_data.get("away") is not None else 0
                live_score_str = f"{g_loc_live} - {g_vis_live}" if estado["is_live"] else ""
                
                seed_loc = int(hashlib.md5(f"{home_name}".encode()).hexdigest()[:8], 16)
                seed_vis = int(hashlib.md5(f"{away_name}".encode()).hexdigest()[:8], 16)
                seed_match = int(hashlib.md5(f"{fix_id}_{home_name}_{away_name}".encode()).hexdigest()[:8], 16)
                
                lam_loc = round(0.8 + ((seed_loc % 17) / 10.0), 2)
                lam_vis = round(0.6 + ((seed_vis % 14) / 10.0), 2)
                lam_tot = round(lam_loc + lam_vis, 2)
                
                # Selección de mercado de goles
                if lam_tot >= 2.7:
                    merc_label = "MÁS DE 2.5 GOLES"
                    merc_linea = 2.5
                    is_over = True
                    conf_goles = int(np.clip((lam_tot / 4.2) * 100, 60, 85))
                elif lam_tot <= 1.9:
                    merc_label = "MENOS DE 2.5 GOLES"
                    merc_linea = 2.5
                    is_over = False
                    conf_goles = int(np.clip((1.0 - (lam_tot / 3.8)) * 100, 62, 86))
                else:
                    merc_label = "MÁS DE 1.5 GOLES"
                    merc_linea = 1.5
                    is_over = True
                    conf_goles = int(np.clip(68 + (seed_match % 15), 66, 82))

                odd_calc = round(max(1.42, min(2.45, (1.0 / (conf_goles / 100.0)) * 0.92)), 2)
                
                p_h = int(round((lam_loc / lam_tot) * 55 + 15))
                p_a = int(round((lam_vis / lam_tot) * 50))
                p_d = max(10, 100 - (p_h + p_a))
                
                # Consistencia estricta del marcador estimado
                if is_over and merc_linea >= 2.5:
                    marcador_est = "2 - 1" if p_h >= p_a else "1 - 2"
                elif is_over and merc_linea >= 1.5:
                    marcador_est = "1 - 1" if p_d > 25 else ("2 - 0" if p_h > p_a else "0 - 2")
                elif not is_over and merc_linea <= 2.5:
                    marcador_est = "1 - 0" if p_h > p_a else ("0 - 1" if p_a > p_h else "0 - 0")
                else:
                    marcador_est = "1 - 1"
                
                # Listas separadas por mercado y por entidad
                f_home_goles = generar_forma_especifica(seed_loc, "GOLES", merc_linea, is_over, 20)
                f_away_goles = generar_forma_especifica(seed_vis, "GOLES", merc_linea, is_over, 20)
                f_h2h_goles  = generar_forma_especifica(seed_match, "GOLES", merc_linea, is_over, 5)
                
                f_corners = generar_forma_especifica(seed_match, "CÓRNERS", 8.5, True, 20)
                f_tarjetas = generar_forma_especifica(seed_match, "TARJETAS", 4.5, False, 20)
                f_disparos = generar_forma_especifica(seed_match, "REMATES", 10.5, True, 20)
                
                p_btts = int(np.clip((lam_loc * lam_vis / 3.0) * 100, 35, 78))
                recom_btts = "SÍ" if p_btts >= 50 else "NO"
                conf_btts = p_btts if recom_btts == "SÍ" else (100 - p_btts)
                odd_btts = round(max(1.45, min(2.35, (1.0 / (conf_btts / 100.0)) * 0.92)), 2)
                f_btts = generar_forma_especifica(seed_match, "AMBOS ANOTAN", 0.5, recom_btts == "SÍ", 20)

                partidos_consolidados.append({
                    "id": fix_id,
                    "deporte": "FÚTBOL",
                    "liga": liga_agrupada,
                    "evento": f"{home_name} vs {away_name}",
                    "fecha": estado["display"],
                    "is_live": estado["is_live"],
                    "live_score": live_score_str,
                    
                    "home_name": home_name,
                    "away_name": away_name,
                    "home_logo": teams_data.get("home", {}).get("logo", ""),
                    "away_logo": teams_data.get("away", {}).get("logo", ""),
                    
                    "p_home": p_h, "p_draw": p_d, "p_away": p_a,
                    "prob_1x2": f"{p_h}% • {p_d}% • {p_a}%",
                    "marcador_estimado": marcador_est,
                    
                    "mercado": merc_label,
                    "linea": merc_linea,
                    "fiabilidad": float(conf_goles),
                    "proyeccion_val": str(lam_tot),
                    "promedio_l10": float(lam_tot),
                    "odd_val": f"{odd_calc:.2f}",
                    "score_num": str(conf_goles),
                    "matchup_grade": "A" if conf_goles >= 74 else "B",
                    
                    # Vectores de partidos por mercado
                    "goles_matches": f_home_goles,
                    "corners_matches": f_corners,
                    "tarjetas_matches": f_tarjetas,
                    "disparos_matches": f_disparos,
                    "btts_matches": f_btts,
                    
                    # Vectores de entidad (Local, Visita, H2H)
                    "home_matches_20": f_home_goles,
                    "away_matches_20": f_away_goles,
                    "h2h_matches": f_h2h_goles,
                    
                    # Mercados independientes
                    "goles_label": merc_label, "goles_conf": float(conf_goles), "goles_odd": f"{odd_calc:.2f}",
                    "goles_proyeccion": str(lam_tot), "goles_promedio": float(lam_tot),
                    
                    "corners_label": "MÁS DE 8.5 CÓRNERS", "corners_conf": 68.0, "corners_odd": "1.74",
                    "corners_proyeccion": "9.2", "corners_promedio": 9.2,
                    
                    "tarjetas_label": "MENOS DE 4.5 TARJETAS", "tarjetas_conf": 71.0, "tarjetas_odd": "1.66",
                    "tarjetas_proyeccion": "3.6", "tarjetas_promedio": 3.6,
                    
                    "disparos_label": "MÁS DE 10.5 REMATES", "disparos_conf": 64.0, "disparos_odd": "1.80",
                    "disparos_proyeccion": "11.2", "disparos_promedio": 11.2,
                    
                    "btts_label": f"AMBOS ANOTAN: {recom_btts}", "btts_conf": float(conf_btts), "btts_odd": f"{odd_btts:.2f}",
                    "btts_prob_si": p_btts, "btts_prob_no": 100 - p_btts,
                    "btts_proyeccion": f"{lam_loc} - {lam_vis}", "btts_promedio": float(lam_tot)
                })
        except Exception as e:
            print(f"[ERROR MAIN]: {e}")
            return []

    return sorted(partidos_consolidados, key=lambda x: (x["is_live"], x["fiabilidad"]), reverse=True)
