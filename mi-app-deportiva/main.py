from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
import os
from datetime import datetime
import pytz
import logging

# Configuración de zona horaria
bogota_tz = pytz.timezone('America/Bogota')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("S2S-SIGMA-API")

app = FastAPI(title="S2S SIGMA API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///db/propsbr_warehouse.db")
engine = create_engine(DATABASE_URL)

@app.get("/")
async def root():
    return {"status": "S2S SIGMA Online", "timezone": "America/Bogota"}

@app.get("/api/v1/props")
async def get_props():
    """Endpoint principal para la App Android"""
    now = datetime.now(bogota_tz).strftime("%H:%M")
    
    # Aquí simulamos la respuesta con la estructura exacta que la app audita
    # En producción, esto debería venir de un SELECT a tu tabla de predicciones
    return [
        {
            "id": "1",
            "deporte": "FÚTBOL",
            "liga": "PREMIER LEAGUE",
            "evento": "Man City vs Arsenal",
            "fecha": f"HOY {now}",
            "mercado": "MÁS 2.5 GOLES",
            "linea": 2.5,
            "fiabilidad": 88.0,
            "recomendacion": "OVER",
            "promedio_l10": 3.2,
            "proyeccion_val": "3.5",
            "senial": "+1.2",
            "racha": "WWWDW",
            "historial": [2, 3, 4, 1, 5],
            "h2h": [2, 2, 1],
            "home_name": "Man City",
            "away_name": "Arsenal",
            "home_logo": "https://media.api-sports.io/football/teams/50.png",
            "away_logo": "https://media.api-sports.io/football/teams/42.png",
            "odd_val": "1.75",
            "score_num": "94",
            "matchup_grade": "A",
            "contexto_defensa": "Arsenal sin centrales titulares",
            "hit_tend": "80%",
            "hit_l5": "4/5",
            "hit_l10": "8/10",
            "hit_l20": "15/20",
            "hit_h2h": "3/5",
            "hit_casa": "90%",
            "hit_fora": "70%",
            "goles_label": "MÁS 2.5",
            "goles_conf": 88.0,
            "corners_label": "MÁS 9.5",
            "corners_conf": 75.0,
            "tarjetas_label": "MENOS 4.5",
            "tarjetas_conf": 65.0,
            "disparos_label": "MÁS 12.5",
            "disparos_conf": 82.0
        }
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
