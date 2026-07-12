from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.routers import health, version, predict
from backend.app.core.config import settings

app = FastAPI()

app.include_router(health.router)
app.include_router(version.router)
app.include_router(predict.router)

@app.get("/")
def read_root():
    return {"message": f"{settings.app_name} Backend Running"}
