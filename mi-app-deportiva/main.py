from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Motor Analítico Deportivo API", version="1.0.0")

# Esquema de respuesta para la App Móvil
class PropResponse(BaseModel):
    id: str
    deporte: str
    jugador: str
    mercado: str
    linea: float
    fiabilidad: float
    recomendacion: str
    promedio_l10: float

@app.get("/")
def home():
    return {"status": "ok", "mensaje": "Servidor de Analítica Activo"}

@app.get("/api/v1/props", response_model=List[PropResponse])
def obtener_props():
    return [
        PropResponse(
            id="nba-001",
            deporte="NBA",
            jugador="LeBron James",
            mercado="Puntos",
            linea=24.5,
            fiabilidad=88.5,
            recomendacion="OVER",
            promedio_l10=27.4
        ),
        PropResponse(
            id="nba-002",
            deporte="NBA",
            jugador="Stephen Curry",
            mercado="Triples",
            linea=4.5,
            fiabilidad=76.2,
            recomendacion="OVER",
            promedio_l10=5.1
        )
    ]
