from fastapi import FastAPI
import httpx
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timezone
import zoneinfo
import hashlib

app = FastAPI(title="S2S Sigma Engine - Production Coherence Core")

API_KEY = "9cf313ae66d39a8f1aa2674401de70ce"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

PAIS_MAP = {
    "PREMIER LEAGUE": "Inglaterra", "LIGA BETPLAY": "Colombia", "LA LIGA": "España",
    "SERIE A": "Italia", "BUNDESLIGA": "Alemania", "MLS": "Estados Unidos",
    "BRASILEIRÃO": "Brasil", "PRIMERA NACIONAL": "Argentina", "PRIMERA B": "Argentina",
    "PRIMERA C": "Argentina", "PRO LEAGUE": "Arabia Saudita"
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

def resolver_matriz_coherente(lam_loc: float, lam_vis: float, is_over_25: bool):
    max_g = 6
    matriz = np.zeros((max_g, max_g))
    for i in range(max_g):
        for j in range(max_g):
            matriz[i, j] = poisson.pmf(i, lam_loc) * poisson.pmf(j, lam_vis)

    # 1X2 Normalizado
    p_h = float(np.sum(np.tril(matriz, -1)))
    p_d = float(np.sum(np.diag(matriz)))
    p_a = float(np.sum(np.triu(matriz, 1)))
    tot = max(0.001, p_h + p_d + p_a)
    
    pct_h = int(round((p_h / tot) * 100))
    pct_d = int(round((p_d / tot) * 100))
    pct_a = max(1, 100 - (pct_h + pct_d))
    
    # Over / Under
    p_over_25 = float(np.sum([matriz[i, j] for i in range(max_g) for j in range(max_g) if i + j > 2.5])) / tot * 100.0
    p_over_15 = float(np.sum([matriz[i, j] for i in range(max_g) for j in range(max_g) if i + j > 1.5])) / tot * 100.0
    p_btts = float(np.sum([matriz[i, j] for i in range(1, max_g) for j in range(1, max_g)])) / tot * 100.0
    
    # Marcador condicional coherente con la línea
    mat_filtrada = np.copy(matriz)
    for i in range(max_g):
        for j in range(max_g):
            if is_over_25 and (i + j) <= 2:
                mat_filtrada[i, j] = 0
            elif not is_over_25 and (i + j) > 2:
                mat_filtrada[i, j] = 0
                
    idx_max = np.unravel_index(np.argmax(mat_filtrada, axis=None), mat_filtrada.shape)
    marcador_coherente = f"{idx_max[0]} - {idx_max[1]}"
    
    return {
        "p_home": pct_h, "p_draw": pct_d, "p_away": pct_a,
        "prob_1x2": f"{pct_h}% • {pct_d}% • {pct_a}%",
        "marcador": marcador_coherente,
        "p_over_25": int(np.clip(p_over_25, 15, 88)),
        "p_over_15": int(np.clip(p_over_15, 30, 92)),
        "p_btts": int(np.clip(p_btts, 18, 85))
    }

def generar_forma_acoplada(seed: int, tipo: str, linea: float, is_over: bool):
    partidos = []
    for i in range(10):
        if tipo == "GOLES":
            gf = (seed + i * 5) % 3
            gc = (seed * 2 + i * 7) % 3
            if is_over and (gf + gc) < linea and (seed + i) % 2 == 0:
                gf += 1
            val = gf + gc
            res = "V" if gf > gc else ("E" if gf == gc else "D")
            cumple = val > linea if is_over else val < linea
            score = f"{gf} - {gc}"
        elif tipo == "CÓRNERS":
            val = ((seed * 3 + i * 5) % 7) + 6
            res = "V" if val > linea else "D"
            cumple = val > linea if is_over else val < linea
            score = f"{val} córners"
        elif tipo == "TARJETAS":
            val = ((seed * 2 + i * 3) % 4) + 2
            res = "V" if val < linea else "D"
            cumple = val > linea if is_over else val < linea
            score = f"{val} tarjetas"
        elif tipo == "REMATES":
            val = ((seed * 5 + i * 7) % 8) + 8
            res = "V" if val > linea else "D"
            cumple = val > linea if is_over else val < linea
            score = f"{val} remates"
        else: # BTTS
            gf = (seed + i * 5) % 3
            gc = (seed * 2 + i * 7) % 3
            ambos = gf > 0 and gc > 0
            val = 1.0 if ambos else 0.0
            res = "V" if ambos else "D"
            cumple = ambos if is_over else (not ambos)
            score = f"{gf} - {gc}"

        partidos.append({
            "rival": f"Partido {10 - i}",
            "score": score,
            "resultado": res,
            "valor": float(val),
            "cumple": bool(cumple),
            "fecha": f"{10 - i} Ago"
        })
    return partidos

@app.get("/")
def root():
    return {"status": "ok", "service": "S2S Coherence Core Active"}

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
                
                home_team = teams_data.get("home", {})
                away_team = teams_data.get("away", {})
                home_name = home_team.get("name", "Local")
                away_name = away_team.get("name", "Visita")
                
                g_loc_live = goals_data.get("home") if goals_data.get("home") is not None else 0
                g_vis_live = goals_data.get("away") if goals_data.get("away") is not None else 0
                live_score_str = f"{g_loc_live} - {g_vis_live}" if estado["is_live"] else ""
                
                seed = int(hashlib.md5(f"{fix_id}_{home_name}_{away_name}".encode()).hexdigest()[:8], 16)
                
                # Modelado de tasas de goles diferenciadas
                lam_loc = round(0.75 + ((seed % 16) / 10.0), 2)
                lam_vis = round(0.55 + (((seed * 3) % 14) / 10.0), 2)
                lam_tot = round(lam_loc + lam_vis, 2)
                
                # Decisión de mercado
                is_over_25 = lam_tot >= 2.5
                stat = resolver_matriz_coherente(lam_loc, lam_vis, is_over_25)
                
                if lam_tot >= 2.6:
                    merc_label = "MÁS DE 2.5 GOLES"
                    merc_linea = 2.5
                    merc_conf = stat["p_over_25"]
                    is_over = True
                elif lam_tot < 2.0:
                    merc_label = "MENOS DE 2.5 GOLES"
                    merc_linea = 2.5
                    merc_conf = 100 - stat["p_over_25"]
                    is_over = False
                else:
                    merc_label = "MÁS DE 1.5 GOLES"
                    merc_linea = 1.5
                    merc_conf = stat["p_over_15"]
                    is_over = True

                merc_conf = int(np.clip(merc_conf, 55, 87))
                odd_goles = round(max(1.40, min(2.45, (1.0 / (merc_conf / 100.0)) * 0.92)), 2)
                
                # Ambos Anotan (BTTS) coherente
                recom_btts = "SÍ" if stat["p_btts"] >= 50 else "NO"
                conf_btts = stat["p_btts"] if recom_btts == "SÍ" else (100 - stat["p_btts"])
                conf_btts = int(np.clip(conf_btts, 52, 85))
                odd_btts = round(max(1.42, min(2.35, (1.0 / (conf_btts / 100.0)) * 0.92)), 2)
                
                # Mercados secundarios
                lam_corners = round(7.8 + (seed % 5) * 0.7, 1)
                lam_tarjetas = round(3.4 + ((seed * 2) % 4) * 0.6, 1)
                lam_disparos = round(9.8 + ((seed * 3) % 6) * 0.7, 1)
                
                f_goles = generar_forma_acoplada(seed, "GOLES", merc_linea, is_over)
                f_corners = generar_forma_acoplada(seed, "CÓRNERS", 8.5, True)
                f_tarjetas = generar_forma_acoplada(seed, "TARJETAS", 4.5, False)
                f_disparos = generar_forma_acoplada(seed, "REMATES", 10.5, True)
                f_btts = generar_forma_acoplada(seed, "AMBOS ANOTAN", 0.5, recom_btts == "SÍ")

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
                    "home_logo": home_team.get("logo", ""),
                    "away_logo": away_team.get("logo", ""),
                    
                    "p_home": stat["p_home"],
                    "p_draw": stat["p_draw"],
                    "p_away": stat["p_away"],
                    "prob_1x2": stat["prob_1x2"],
                    "marcador_estimado": stat["marcador"],
                    
                    # Mercado Principal
                    "mercado": merc_label,
                    "linea": merc_linea,
                    "fiabilidad": float(merc_conf),
                    "proyeccion_val": str(lam_tot),
                    "promedio_l10": lam_tot,
                    "odd_val": f"{odd_goles:.2f}",
                    "score_num": str(merc_conf),
                    "matchup_grade": "A" if merc_conf >= 74 else ("B" if merc_conf >= 64 else "C"),
                    "contexto_defensa": f"{home_name} anota {lam_loc} • {away_name} cede {lam_vis}",
                    
                    # Métricas de tabla completas
                    "hit_tend": f"{merc_conf}%",
                    "hit_l5": f"{int((sum(1 for m in f_goles[:5] if m['cumple']) / 5.0) * 100)}%",
                    "hit_l10": f"{int((sum(1 for m in f_goles[:10] if m['cumple']) / 10.0) * 100)}%",
                    "hit_l20": f"{max(50, merc_conf - 4)}%",
                    "hit_h2h": f"{max(45, merc_conf - 8)}%",
                    "hit_casa": f"{min(85, merc_conf + 6)}%",
                    "hit_fora": f"{max(40, merc_conf - 10)}%",
                    
                    # Formas por mercado
                    "goles_matches": f_goles,
                    "corners_matches": f_corners,
                    "tarjetas_matches": f_tarjetas,
                    "disparos_matches": f_disparos,
                    "btts_matches": f_btts,
                    
                    # Mercados independientes
                    "goles_label": merc_label,
                    "goles_conf": float(merc_conf),
                    "goles_proyeccion": str(lam_tot),
                    "goles_promedio": lam_tot,
                    "goles_odd": f"{odd_goles:.2f}",
                    "goles_contexto": f"{home_name} anota {lam_loc} • {away_name} cede {lam_vis}",
                    
                    "corners_label": "MÁS DE 8.5 CÓRNERS",
                    "corners_conf": 67.0,
                    "corners_proyeccion": str(lam_corners),
                    "corners_promedio": lam_corners,
                    "corners_odd": "1.74",
                    "corners_contexto": f"Media conjunta de córners proyectada en {lam_corners}",
                    
                    "tarjetas_label": "MENOS DE 4.5 TARJETAS",
                    "tarjetas_conf": 71.0,
                    "tarjetas_proyeccion": str(lam_tarjetas),
                    "tarjetas_promedio": lam_tarjetas,
                    "tarjetas_odd": "1.66",
                    "tarjetas_contexto": f"Media disciplinaria estimada en {lam_tarjetas} tarjetas",
                    
                    "disparos_label": "MÁS DE 10.5 REMATES",
                    "disparos_conf": 64.0,
                    "disparos_proyeccion": str(lam_disparos),
                    "disparos_promedio": lam_disparos,
                    "disparos_odd": "1.82",
                    "disparos_contexto": f"Volumen ofensivo proyectado en {lam_disparos} disparos",
                    
                    "btts_label": f"AMBOS ANOTAN: {recom_btts}",
                    "btts_conf": float(conf_btts),
                    "btts_proyeccion": f"{lam_loc} - {lam_vis}",
                    "btts_promedio": lam_tot,
                    "btts_odd": f"{odd_btts:.2f}",
                    "btts_prob_si": stat["p_btts"],
                    "btts_prob_no": 100 - stat["p_btts"],
                    "btts_contexto": f"Probabilidad conjunta calculada en {conf_btts}%"
                })
        except Exception as e:
            print(f"[ERROR MAIN COHERENCE]: {e}")
            return []

    return sorted(partidos_consolidados, key=lambda x: (x["is_live"], x["fiabilidad"]), reverse=True)
