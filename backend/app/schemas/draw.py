from datetime import date

from pydantic import BaseModel, Field


class DrawBase(BaseModel):
    draw_date: date

    n1: int = Field(..., ge=1, le=99)
    n2: int = Field(..., ge=1, le=99)
    n3: int = Field(..., ge=1, le=99)
    n4: int = Field(..., ge=1, le=99)
    n5: int = Field(..., ge=1, le=99)
    n6: int = Field(..., ge=1, le=99)

    bonus: int | None = None


class DrawCreate(DrawBase):
    pass


class DrawResponse(DrawBase):
    id: int

    class Config:
        from_attributes = True