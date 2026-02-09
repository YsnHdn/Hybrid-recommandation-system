"""
Point d'entrée de l'API FastAPI
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Créer l'application FastAPI
app = FastAPI(
    title="Recommendation System API",
    description="API REST pour un système de recommandation hybride de films",
    version="1.0.0"
)

# Configurer CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "API is running"
    }