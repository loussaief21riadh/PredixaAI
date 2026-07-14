from datetime import date, datetime

from pydantic import BaseModel, Field


class DrawBase(BaseModel):
    game: str = "Loto FDJ"

    draw_date: date

    n1: int = Field(..., ge=1, le=49)
    n2: int = Field(..., ge=1, le=49)
    n3: int = Field(..., ge=1, le=49)
    n4: int = Field(..., ge=1, le=49)
    n5: int = Field(..., ge=1, le=49)

    chance: int = Field(..., ge=1, le=10)


class DrawCreate(DrawBase):
    pass


class DrawUpdate(DrawBase):
    pass


class DrawResponse(DrawBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True