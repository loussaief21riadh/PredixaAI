from fastapi import FastAPI

from app.database import Base, engine

# Création des tables SQLAlchemy
from app.auth.models import User

# Routeurs
from app.auth.router import router as auth_router
from app.dashboard.router import router as dashboard_router

from app.routers import (
    ai,
    draws,
    health,
    import_csv,
    predict,
    statistics,
    version,
)

# Création automatique des tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Predixa AI Enterprise",
    version="2.3.0",
    description="Enterprise AI Lottery Prediction Platform",
)


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Welcome to Predixa AI Enterprise",
        "version": app.version,
        "status": "running",
    }


# ==========================
# Authentication
# ==========================
app.include_router(auth_router)

# ==========================
# Dashboard
# ==========================
app.include_router(dashboard_router)

# ==========================
# System
# ==========================
app.include_router(health.router)
app.include_router(version.router)

# ==========================
# Draws
# ==========================
app.include_router(draws.router)

# ==========================
# Prediction
# ==========================
app.include_router(predict.router)

# ==========================
# CSV Import
# ==========================
app.include_router(import_csv.router)

# ==========================
# Statistics
# ==========================
app.include_router(statistics.router)

# ==========================
# Artificial Intelligence
# ==========================
app.include_router(ai.router)