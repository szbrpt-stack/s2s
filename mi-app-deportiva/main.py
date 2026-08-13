from fastapi import FastAPI, Query
import numpy as np
from scipy.stats import poisson
from typing import List, Optional

app = FastAPI(title="PropsBR / S2S Sigma Advanced Quant Engine")

def calcular_metricas_avanzadas(historial: List[float], linea: float) -> dict:
    datos = np.array(historial, dtype=float)
    n = len(datos)
    if n == 0:
        return {
            "fiabilidad": 50.0, "recomendacion": "NEUTRO", "promedio_l5": 0.0,
            "promedio_l10": 0.0, "promedio_l20": 0.0, "vantagem": "+0.0",
            "grade": "C", "hit_rate_l10": "0%"
        }
    
    # Subconjuntos L5, L10, L20
    l5 = datos[-5:] if n >= 5 else datos
    l10 = datos[-10:] if n >= 10 else datos
    l20 = datos
    
    prom_l5 = round(float(np.mean(l5)), 2)
    prom_l10 = round(float(np.mean(l10)), 2)
    prom_l20 = round(float(np.mean(l20)), 2)
    
    # Decaimiento Exponencial en L10
    pesos = np.exp(np.linspace(-0.8, 0, len(l10)))
    pesos /= pesos.sum()
    lambda_ponderado = np.sum(l10 * pesos)
    
    # Probabilidad teórica vía Poisson
    prob_over = poisson.sf(np.floor(linea), lambda_ponderado) * 100
    prob_under = 100 - prob_over
    
    hits = sum(1 for x in l10 if x > linea)
    hit_rate = round((hits / len(l10)) * 100)
    
    if lambda_ponderado > linea:
        recomendacion = "OVER"
        fiabilidad = prob_over
        ventaja_raw = prom_l10 - linea
    else:
        recomendacion = "UNDER"
        fiabilidad = prob_under
        ventaja_raw = linea - prom_l10

    # Calificación por ventajas (Grades)
    if fiabilidad >= 80 and ventaja_raw >= 1.5:
        grade = "A+"
    elif fiabilidad >= 70:
        grade = "A"
    elif fiabilidad >= 60:
        grade = "B"
    else:
        grade = "C"
        
    return {
        "fiabilidad": round(float(np.clip(fiabilidad, 50.0, 99.0)), 1),
        "recomendacion": recomendacion,
        "promedio_l5": prom_l5,
        "promedio_l10": prom_l10,
        "promedio_l20": prom_l20,
        "vantagem": f"+{round(abs(ventaja_raw), 1)}",
        "grade": grade,
        "hit_rate_l10": f"{hit_rate}%"
    }

# Core Dataset Enriquecido con Historiales Reales
DATABASE_PROPS = [
    # FÚTBOL - Escanteios / Goles / Tarjetas
    {"id": "f101", "deporte": "FÚTBOL", "liga": "Superliga", "evento": "Sirius vs Hammarby", "mercado": "Escanteios", "linea": 1.5, "historial": [3, 2, 4, 1, 5, 2, 3, 4, 2, 3, 4, 2, 3, 5, 1, 4, 2, 3, 4, 2], "h2h": [2, 3, 1, 4, 2]},
    {"id": "f102", "deporte": "FÚTBOL", "liga": "Superliga", "evento": "Fredericia vs Vendsyssel", "mercado": "Escanteios", "linea": 1.57, "historial": [2, 3, 5, 2, 4, 1, 3, 2, 4, 3, 5, 2, 4, 3, 1, 2, 4, 3, 5, 2], "h2h": [3, 2, 4, 1, 3]},
    {"id": "f103", "deporte": "FÚTBOL", "liga": "Premier League", "evento": "Manchester Utd vs Liverpool", "mercado": "Escanteios", "linea": 9.5, "historial": [11, 12, 8, 10, 14, 9, 13, 10, 11, 12, 9, 11, 10, 12, 13, 8, 11, 10, 12, 11], "h2h": [10, 12, 9, 11, 13]},
    {"id": "f104", "deporte": "FÚTBOL", "liga": "Liga BetPlay", "evento": "Millonarios vs Nacional", "mercado": "Tarjetas", "linea": 5.5, "historial": [7, 6, 8, 5, 6, 9, 7, 6, 8, 7, 6, 8, 5, 7, 6, 8, 9, 6, 7, 8], "h2h": [6, 8, 7, 5, 8]},
    {"id": "f105", "deporte": "FÚTBOL", "liga": "Champions League", "evento": "Real Madrid vs Man City", "mercado": "Finalizações", "linea": 2.5, "historial": [4, 3, 5, 2, 4, 3, 6, 4, 3, 5, 4, 2, 5, 3, 4, 6, 3, 4, 5, 3], "h2h": [4, 3, 5, 2, 4]},
    
    # NBA - Puntos / Rebotes / Triples
    {"id": "n201", "deporte": "NBA", "liga": "NBA", "evento": "Airious Bailey (UTA)", "mercado": "Pontos", "linea": 14.5, "historial": [18, 16, 21, 12, 19, 15, 22, 17, 20, 16, 18, 15, 19, 21, 14, 18, 17, 20, 16, 19], "h2h": [17, 19, 15, 20, 18]},
    {"id": "n202", "deporte": "NBA", "liga": "NBA", "evento": "Gui Santos (GSW)", "mercado": "Pontos", "linea": 11.5, "historial": [14, 12, 15, 10, 13, 11, 16, 12, 14, 13, 12, 14, 11, 15, 13, 12, 14, 13, 11, 14], "h2h": [13, 11, 14, 12, 15]},
    {"id": "n203", "deporte": "NBA", "liga": "NBA", "evento": "Brandin Podziemski (GSW)", "mercado": "Pontos", "linea": 14.5, "historial": [16, 18, 13, 17, 15, 19, 14, 16, 18, 15, 17, 14, 18, 16, 15, 19, 14, 17, 16, 18], "h2h": [15, 18, 14, 16, 17]},
    {"id": "n204", "deporte": "NBA", "liga": "NBA", "evento": "Kawhi Leonard (LAC)", "mercado": "Bolas de 3", "linea": 2.5, "historial": [3, 4, 2, 3, 5, 3, 2, 4, 3, 4, 3, 2, 4, 3, 5, 2, 4, 3, 4, 3], "h2h": [3, 4, 2, 4, 3]}
]

@app.get("/")
def root():
    return {"status": "ok", "engine": "PropsBR Quant Core Active"}

@app.get("/api/v1/props")
def get_props(
    deporte: Optional[str] = None,
    mercado: Optional[str] = None
):
    resultado = []
    for item in DATABASE_PROPS:
        if deporte and item["deporte"].upper() != deporte.upper():
            continue
        if mercado and item["mercado"].upper() != mercado.upper():
            continue
            
        calc = calcular_metricas_avanzadas(item["historial"], item["linea"])
        resultado.append({
            "id": item["id"],
            "deporte": item["deporte"],
            "liga": item["liga"],
            "evento": item["evento"],
            "fecha": "HOY",
            "jugador": item["evento"],
            "mercado": item["mercado"],
            "linea": float(item["linea"]),
            "fiabilidad": calc["fiabilidad"],
            "recomendacion": calc["recomendacion"],
            "promedio_l10": calc["promedio_l10"],
            "senial": calc["vantagem"],
            "racha": calc["grade"],
            "historial": item["historial"],
            "h2h": item.get("h2h", [])
        })
    return sorted(resultado, key=lambda x: x["fiabilidad"], reverse=True)
