from pydantic import BaseModel


class DashboardResponse(BaseModel):
    users: int
    draws: int
    models: int
    predictions: int
    accuracy: float
    last_training: str