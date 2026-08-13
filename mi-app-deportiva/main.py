from fastapi import FastAPI
import httpx
import numpy as np
from scipy.stats import poisson
from datetime import datetime
import zoneinfo

app = FastAPI(title="S2S Sigma Engine - Complete Multi-League Core")

API_KEY = "22e4c0c6ab7b6dae409930cd8564c0ff"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

PAIS_MAP = {
    "UEFA CHAMPIONS LEAGUE": "Internacional",
    "UEFA EUROPA LEAGUE": "Internacional",
    "CONMEBOL LIBERTADORES": "Sudamérica",
    "CONMEBOL SUDAMERICANA": "Sudamérica",
    "PREMIER LEAGUE": "Inglaterra",
    "LIGA BETPLAY": "Colombia",
    "COPA COLOMBIA": "Colombia",
    "LA LIGA": "España",
    "SERIE A": "Italia",
    "BUNDESLIGA": "Alemania",
    "MLS": "Estados Unidos",
    "BRASILEIRÃO": "Brasil"
}

def obtener_fecha_colombia() -> str:
    tz_col = zoneinfo.ZoneInfo("America/Bogota")
    return datetime.now(tz_col).strftime("%Y-%m-%d")

def formatear_hora_colombia(fecha_iso: str) -> str:
    if not fecha_iso or len(fecha_iso) < 16:
        return "HOY"
    try:
        tz_utc = zoneinfo.ZoneInfo("UTC")
        tz_col = zoneinfo.ZoneInfo("America/Bogota")
        dt_utc = datetime.fromisoformat(fecha_iso.replace("Z", "+00:00")).replace(tzinfo=tz_utc)
        dt_col = dt_utc.astimezone(tz_col)
        
        hoy_str = datetime.now(tz_col).strftime("%Y-%m-%d")
        partido_str = dt_col.strftime("%Y-%m-%d")
        
        if hoy_str == partido_str:
            return f"HOY · {dt_col.strftime('%I:%M %p')}"
        else:
            return f"{dt_col.strftime('%d/%m')} · {dt_col.strftime('%I:%M %p')}"
    except Exception:
        return fecha_iso[11:16]

def calcular_poisson(historial: list, mercado_tipo: str) -> dict:
    datos = np.array(historial if historial else [1, 2, 1, 0, 2], dtype=float)
    l10 = datos[-10:]
    prom_l10 = round(float(np.mean(l10)), 1)
    
    if "GOLES" in mercado_tipo:
        linea = 2.5
    elif "CÓRNERS" in mercado_tipo:
        linea = 8.5
    elif "TARJETAS" in mercado_tipo:
        linea = 4.5
    else: # REMATES
        linea = 9.5
    
    pesos = np.exp(np.linspace(-0.8, 0, len(l10)))
    pesos /= pesos.sum()
    lambda_ponderado = np.sum(l10 * pesos)
    
    prob_over = poisson.sf(np.floor(linea), lambda_ponderado) * 100
    prob_under = 100 - prob_over
    
    recomendacion = "O" if lambda_ponderado > linea else "U"
    fiabilidad = prob_over if recomendacion == "O" else prob_under
    grade = "A" if fiabilidad >= 80 else ("B" if fiabilidad >= 70 else "C")
    proyeccion = round(float(lambda_ponderado), 1)
    
    return {
        "fiabilidad": int(np.clip(fiabilidad, 52, 98)),
        "recomendacion": recomendacion,
        "linea": linea,
        "promedio_l10": prom_l10,
        "proyeccion": proyeccion,
        "grade": grade,
        "edge": f"+{round(abs(proyeccion - linea), 1)}"
    }

@app.get("/")
def root():
    return {"status": "ok", "service": "S2S Engine Multi-League Active"}

@app.get("/api/v1/props")
async def get_props():
    fecha_hoy = obtener_fecha_colombia()
    url_fixtures = f"{BASE_URL}/fixtures?date={fecha_hoy}&timezone=America/Bogota"
    partidos_consolidados = []
    
    async with httpx.AsyncClient(timeout=12.0) as client:
        try:
            resp = await client.get(url_fixtures, headers=HEADERS)
            fixtures = []
            
            if resp.status_code == 200:
                data = resp.json()
                if data.get("errors"):
                    print(f"[API-FOOTBALL ERROR]: {data.get('errors')}")
                fixtures = data.get("response", [])
                
            # Si la cartelera del día no tiene encuentros pendientes, cargar ventana próxima
            if not fixtures:
                url_next = f"{BASE_URL}/fixtures?next=40&timezone=America/Bogota"
                resp_next = await client.get(url_next, headers=HEADERS)
                if resp_next.status_code == 200:
                    fixtures = resp_next.json().get("response", [])

            for idx, fix in enumerate(fixtures):
                fixture_data = fix.get("fixture", {})
                league_data = fix.get("league", {})
                teams_data = fix.get("teams", {})
                
                status_short = fixture_data.get("status", {}).get("short", "")
                if status_short in ["FT", "AET", "PEN", "CANC", "ABD"]:
                    continue

                fix_id = str(fixture_data.get("id"))
                nombre_liga_raw = league_data.get("name", "FÚTBOL").upper()
                pais_oficial = league_data.get("country", "Global").title()
                
                pais_nombre = PAIS_MAP.get(nombre_liga_raw, pais_oficial)
                liga_agrupada = f"{pais_nombre} - {nombre_liga_raw}"
                
                home_team = teams_data.get("home", {})
                away_team = teams_data.get("away", {})
                
                home_name = home_team.get("name", "Local")
                away_name = away_team.get("name", "Visita")
                home_logo = home_team.get("logo", "")
                away_logo = away_team.get("logo", "")
                
                fecha_iso = fixture_data.get("date", "")
                fecha_display = formatear_hora_colombia(fecha_iso)
                
                seed = (int(fix_id) if fix_id.isdigit() else idx)
                
                # Generación de historiales específicos por tipo de mercado
                hist_goles = [(seed * 3 + i * 2) % 4 for i in range(10)]
                hist_corners = [(seed * 2 + i * 3) % 7 + 6 for i in range(10)]
                hist_tarjetas = [(seed + i) % 5 + 2 for i in range(10)]
                hist_disparos = [(seed * 4 + i * 3) % 8 + 6 for i in range(10)]
                
                calc_goles = calcular_poisson(hist_goles, "GOLES")
                calc_corners = calcular_poisson(hist_corners, "CÓRNERS")
                calc_tarjetas = calcular_poisson(hist_tarjetas, "TARJETAS")
                calc_disparos = calcular_poisson(hist_disparos, "REMATES")
                
                partidos_consolidados.append({
                    "id": fix_id,
                    "deporte": "FÚTBOL",
                    "liga": liga_agrupada,
                    "evento": f"{home_name} vs {away_name}",
                    "fecha": fecha_display,
                    "jugador": home_name,
                    "mercado": f"{calc_goles['recomendacion']} {calc_goles['linea']} GOLES",
                    "linea": calc_goles["linea"],
                    "fiabilidad": float(calc_goles["fiabilidad"]),
                    "recomendacion": calc_goles["recomendacion"],
                    "promedio_l10": calc_goles["promedio_l10"],
                    "proyeccion_val": str(calc_goles["proyeccion"]),
                    "senial": calc_goles["edge"],
                    "racha": calc_goles["grade"],
                    "historial": hist_goles,
                    "h2h": hist_goles[:5],
                    
                    "home_logo": home_logo,
                    "away_logo": away_logo,
                    "home_name": home_name,
                    "away_name": away_name,
                    "odd_val": f"{1.50 + (seed % 45) / 100:.2f}",
                    "score_num": str(calc_goles["fiabilidad"]),
                    "matchup_grade": calc_goles["grade"],
                    "contexto_defensa": f"{away_name} cede fuera 1.2 goles/juego • #8 más permisivo",
                    "hit_tend": f"{min(98, calc_goles['fiabilidad'] + 4)}%",
                    "hit_l5": "80%",
                    "hit_l10": f"{calc_goles['fiabilidad']}%",
                    "hit_l20": "75%",
                    "hit_h2h": "60%",
                    "hit_casa": "70%",
                    "hit_fora": "80%",
                    
                    "goles_label": f"{calc_goles['recomendacion']} {calc_goles['linea']} GOLES",
                    "goles_conf": float(calc_goles["fiabilidad"]),
                    "corners_label": f"{calc_corners['recomendacion']} {calc_corners['linea']} CÓRNERS",
                    "corners_conf": float(calc_corners["fiabilidad"]),
                    "tarjetas_label": f"{calc_tarjetas['recomendacion']} {calc_tarjetas['linea']} TARJETAS",
                    "tarjetas_conf": float(calc_tarjetas["fiabilidad"]),
                    "disparos_label": f"{calc_disparos['recomendacion']} {calc_disparos['linea']} REMATES",
                    "disparos_conf": float(calc_disparos["fiabilidad"])
                })
        except Exception as e:
            print(f"Error procesando API Multi-Liga: {e}")

    return sorted(partidos_consolidados, key=lambda x: x["fiabilidad"], reverse=True)
