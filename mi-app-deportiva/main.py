from fastapi import FastAPI
import numpy as np
from scipy.stats import poisson
from datetime import datetime

app = FastAPI(title="S2S Sigma Core Engine")

def calcular_sigma(historial: list, linea: float) -> dict:
    datos = np.array(historial)
    n = len(datos)
    if n == 0:
        return {"fiabilidad": 50.0, "recomendacion": "NEUTRO", "promedio_l10": 0.0, "racha": "0/0"}
    
    # Decaimiento Exponencial (Peso mayor a lo reciente)
    pesos = np.exp(np.linspace(-0.8, 0, n))
    pesos /= pesos.sum()
    lambda_ponderado = np.sum(datos * pesos)
    
    # Cálculo de racha (cuántas veces superó/quedó por debajo de la línea en L10)
    aciertos = sum(1 for x in datos if x > linea) if lambda_ponderado > linea else sum(1 for x in datos if x < linea)
    racha_str = f"{aciertos}/{n} L10"
    
    # Probabilidad teórica Poisson
    prob_over = poisson.sf(np.floor(linea), lambda_ponderado) * 100
    prob_under = 100 - prob_over
    
    # Factor de consistencia
    desviacion = np.std(datos)
    cv = desviacion / lambda_ponderado if lambda_ponderado > 0 else 1.0
    factor_consistencia = np.clip(1.0 - (cv * 0.25), 0.65, 1.0)
    
    if lambda_ponderado > linea:
        recomendacion = "OVER"
        fiabilidad = prob_over * factor_consistencia
    else:
        recomendacion = "UNDER"
        fiabilidad = prob_under * factor_consistencia
        
    return {
        "fiabilidad": round(float(np.clip(fiabilidad, 55.0, 98.5)), 1),
        "recomendacion": recomendacion,
        "promedio_l10": round(float(lambda_ponderado), 1),
        "racha": racha_str
    }

# Dataset Dinámico de la Cartelera de HOY
PROPS_ACTUALES = [
    {
        "id": "p101",
        "deporte": "FÚTBOL",
        "liga": "LIGA BETPLAY",
        "evento": "Millonarios vs Atlético Nacional",
        "fecha": "HOY · 20:00",
        "jugador": "Córners Totales",
        "mercado": "Córners",
        "linea": 9.5,
        "senial": "ALTA FRECUENCIA",
        "historial": [11, 12, 10, 8, 13, 11, 9, 10, 12, 11]
    },
    {
        "id": "p102",
        "deporte": "FÚTBOL",
        "liga": "CHAMPIONS LEAGUE",
        "evento": "Real Madrid vs Manchester City",
        "fecha": "HOY · 14:00",
        "jugador": "Erling Haaland",
        "mercado": "Remates a Puerta",
        "linea": 2.5,
        "senial": "RACHA POSITIVA",
        "historial": [3, 4, 2, 5, 3, 4, 2, 3, 4, 3]
    },
    {
        "id": "p103",
        "deporte": "BALONCESTO",
        "liga": "NBA",
        "evento": "Lakers vs Warriors",
        "fecha": "HOY · 21:30",
        "jugador": "Luka Dončić",
        "mercado": "Puntos",
        "linea": 31.5,
        "senial": "VALOR EN LÍNEA",
        "historial": [35, 38, 29, 36, 33, 37, 34, 32, 39, 35]
    },
    {
        "id": "p104",
        "deporte": "FÚTBOL",
        "liga": "PREMIER LEAGUE",
        "evento": "Arsenal vs Chelsea",
        "fecha": "MAÑANA · 11:30",
        "jugador": "Faltas Totales",
        "mercado": "Faltas",
        "linea": 24.5,
        "senial": "TENDENCIA ESTABLE",
        "historial": [26, 28, 23, 27, 25, 29, 24, 26, 28, 25]
    },
    {
        "id": "p105",
        "deporte": "FÚTBOL",
        "liga": "COPA LIBERTADORES",
        "evento": "Flamengo vs River Plate",
        "fecha": "MAÑANA · 19:30",
        "jugador": "Tarjetas Amarillas",
        "mercado": "Tarjetas",
        "linea": 5.5,
        "senial": "ALTA VOLATILIDAD",
        "historial": [7, 6, 8, 5, 6, 9, 7, 6, 8, 7]
    }
]

@app.get("/")
def root():
    return {"status": "ok", "service": "S2S Sigma Engine"}

@app.get("/api/v1/props")
def get_props():
    resultado = []
    for item in PROPS_ACTUALES:
        calc = calcular_sigma(item["historial"], item["linea"])
        resultado.append({
            "id": item["id"],
            "deporte": item["deporte"],
            "liga": item["liga"],
            "evento": item["evento"],
            "fecha": item["fecha"],
            "jugador": item["jugador"],
            "mercado": item["mercado"],
            "linea": item["linea"],
            "fiabilidad": calc["fiabilidad"],
            "recomendacion": calc["recomendacion"],
            "promedio_l10": calc["promedio_l10"],
            "senial": item["senial"],
            "racha": calc["racha"],
            "historial": item["historial"]
        })
    return sorted(resultado, key=lambda x: x["fiabilidad"], reverse=True)
