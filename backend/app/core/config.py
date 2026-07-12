import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = os.getenv("APP_NAME", "LottoVisionAI")
    version: str = os.getenv("VERSION", "0.1.0")

settings = Settings()
