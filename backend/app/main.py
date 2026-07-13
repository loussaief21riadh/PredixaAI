from fastapi import FastAPI

from app.core.config import settings
from app.database import Base, engine

# Import des modèles
from app.models.draw import Draw

# Import des routes
from app.routers import health, version, predict, draws

# Création des tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend API for LottoVisionAI",
)

# Routes
app.include_router(health.router)
app.include_router(version.router)
app.include_router(predict.router)
app.include_router(draws.router)


@app.get("/")
def root():
    return {
        "application": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "database": "connected",
    }