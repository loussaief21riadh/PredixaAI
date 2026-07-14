from fastapi import FastAPI

from app.database import Base, engine
from app.models.draw import Draw

from app.routers.health import router as health_router
from app.routers.version import router as version_router
from app.routers.predict import router as predict_router
from app.routers.draws import router as draws_router
from app.routers.import_csv import router as import_router
from app.routers.statistics import router as statistics_router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Predixa AI",
    version="1.0.0",
    description="AI Lottery Prediction Platform",
)

app.include_router(health_router)
app.include_router(version_router)
app.include_router(draws_router)
app.include_router(predict_router)
app.include_router(import_router)
app.include_router(statistics_router)


@app.get("/")
def root():
    return {
        "application": "Predixa AI",
        "version": "1.0.0",
        "status": "running",
        "database": "connected",
    }