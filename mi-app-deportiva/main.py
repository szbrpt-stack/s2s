from fastapi import FastAPI
import numpy as np

app = FastAPI(title="PropsBR / S2S Engine API")

# Generador de Dataset Denso Multideporte
DEPORTES = ["NBA", "FÚTBOL", "NHL", "MLB", "TENIS"]

DATASET_DENSE = [
    # FÚTBOL - Escanteios / Goles / Tarjetas
    {"id": "f1", "deporte": "FÚTBOL", "liga": "Superliga", "evento": "Sirius vs Hammarby", "mercado": "Escanteios", "linea": 1.5, "vantagem": "+1.5", "confianza": 85, "grade": "A+", "historial": [3, 2, 4, 1, 5, 2, 3, 4, 2, 3]},
    {"id": "f2", "deporte": "FÚTBOL", "liga": "Superliga", "evento": "Fredericia vs Vendsyssel", "mercado": "Escanteios", "linea": 1.57, "vantagem": "+4.1", "confianza": 80, "grade": "A", "historial": [2, 3, 5, 2, 4, 1, 3, 2, 4, 3]},
    {"id": "f3", "deporte": "FÚTBOL", "liga": "Premier League", "evento": "Manchester Utd vs Liverpool", "mercado": "Escanteios", "linea": 1.83, "vantagem": "+0.4", "confianza": 59, "grade": "C+", "historial": [1, 2, 0, 3, 1, 2, 1, 0, 2, 1]},
    {"id": "f4", "deporte": "FÚTBOL", "liga": "Ligue 2", "evento": "Rodez vs Troyes", "mercado": "Escanteios", "linea": 1.50, "vantagem": "+1.2", "confianza": 59, "grade": "C", "historial": [2, 1, 3, 0, 2, 1, 2, 3, 1, 2]},
    {"id": "f5", "deporte": "FÚTBOL", "liga": "Liga BetPlay", "evento": "Millonarios vs Nacional", "mercado": "Tarjetas", "linea": 5.5, "vantagem": "+2.1", "confianza": 78, "grade": "A", "historial": [7, 6, 8, 5, 6, 9, 7, 6, 8, 7]},
    {"id": "f6", "deporte": "FÚTBOL", "liga": "Champions", "evento": "Real Madrid vs Man City", "mercado": "Finalizações", "linea": 2.5, "vantagem": "+1.8", "confianza": 82, "grade": "A+", "historial": [4, 3, 5, 2, 4, 3, 6, 4, 3, 5]},
    
    # NBA - Puntos / Rebotes / Asistencias
    {"id": "n1", "deporte": "NBA", "liga": "NBA", "evento": "Airious Bailey (UTA)", "mercado": "Pontos", "linea": 14.5, "vantagem": "+20.3", "confianza": 81, "grade": "B", "historial": [18, 16, 21, 12, 19, 15, 22, 17, 20, 16]},
    {"id": "n2", "deporte": "NBA", "liga": "NBA", "evento": "Gui Santos (GSW)", "mercado": "Pontos", "linea": 11.5, "vantagem": "+14.7", "confianza": 78, "grade": "C", "historial": [14, 12, 15, 10, 13, 11, 16, 12, 14, 13]},
    {"id": "n3", "deporte": "NBA", "liga": "NBA", "evento": "Brandin Podziemski (GSW)", "mercado": "Pontos", "linea": 14.5, "vantagem": "+21.1", "confianza": 77, "grade": "C+", "historial": [16, 18, 13, 17, 15, 19, 14, 16, 18, 15]},
    {"id": "n4", "deporte": "NBA", "liga": "NBA", "evento": "Kawhi Leonard (LAC)", "mercado": "Bolas de 3", "linea": 2.5, "vantagem": "+2.8", "confianza": 75, "grade": "C+", "historial": [3, 4, 2, 3, 5, 3, 2, 4, 3, 4]}
]

@app.get("/")
def root():
    return {"status": "ok", "service": "PropsBR Engine"}

@app.get("/api/v1/props")
def get_props():
    resultado = []
    for item in DATASET_DENSE:
        resultado.append({
            "id": item["id"],
            "deporte": item["deporte"],
            "liga": item["liga"],
            "evento": item["evento"],
            "fecha": "HOY",
            "jugador": item["evento"],
            "mercado": item["mercado"],
            "linea": float(item["linea"]),
            "fiabilidad": float(item["confianza"]),
            "recomendacion": "OVER",
            "promedio_l10": float(round(np.mean(item["historial"]), 1)),
            "senial": item["vantagem"],
            "racha": f"Grade {item['grade']}",
            "historial": item["historial"]
        })
    return resultado
