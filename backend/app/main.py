from fastapi import FastAPI

from app.database import Base, engine

from app.routers import (
    ai,
    draws,
    health,
    import_csv,
    predict,
    statistics,
    version,
)

# Création des tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Predixa AI",
    version="1.0.0",
    description="AI Lottery Prediction Platform",
)


@app.get("/")
def root():
    return {
        "message": "Welcome to Predixa AI",
    }


app.include_router(health.router)
app.include_router(version.router)
app.include_router(draws.router)
app.include_router(predict.router)
app.include_router(import_csv.router)
app.include_router(statistics.router)
app.include_router(ai.router)