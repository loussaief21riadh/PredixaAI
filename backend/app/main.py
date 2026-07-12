from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.routers import health, version

app = FastAPI()

app.include_router(health.router)
app.include_router(version.router)

@app.get("/")
def read_root():
    return {"message": "LottoVisionAI Backend Running"}
