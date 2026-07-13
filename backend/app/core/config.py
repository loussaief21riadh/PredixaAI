from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "LottoVisionAI"
    app_version: str = "1.0.0"
    database_url: str = "sqlite:///./sql_app.db"

    model_config = {
        "env_file": ".env"
    }


settings = Settings()