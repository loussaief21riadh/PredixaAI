from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.draw import Draw
from app.schemas.draw import DrawCreate, DrawResponse, DrawUpdate

router = APIRouter(
    prefix="/draws",
    tags=["Draws"],
)


@router.post("/", response_model=DrawResponse)
def create_draw(draw: DrawCreate, db: Session = Depends(get_db)):
    db_draw = Draw(**draw.model_dump())

    db.add(db_draw)
    db.commit()
    db.refresh(db_draw)

    return db_draw


@router.get("/", response_model=list[DrawResponse])
def get_draws(db: Session = Depends(get_db)):
    return db.query(Draw).order_by(Draw.draw_date.desc()).all()


@router.get("/{draw_id}", response_model=DrawResponse)
def get_draw(draw_id: int, db: Session = Depends(get_db)):
    draw = db.query(Draw).filter(Draw.id == draw_id).first()

    if draw is None:
        raise HTTPException(status_code=404, detail="Draw not found")

    return draw


@router.put("/{draw_id}", response_model=DrawResponse)
def update_draw(
    draw_id: int,
    updated_draw: DrawUpdate,
    db: Session = Depends(get_db),
):
    draw = db.query(Draw).filter(Draw.id == draw_id).first()

    if draw is None:
        raise HTTPException(status_code=404, detail="Draw not found")

    data = updated_draw.model_dump()

    for key, value in data.items():
        setattr(draw, key, value)

    db.commit()
    db.refresh(draw)

    return draw


@router.delete("/{draw_id}")
def delete_draw(draw_id: int, db: Session = Depends(get_db)):
    draw = db.query(Draw).filter(Draw.id == draw_id).first()

    if draw is None:
        raise HTTPException(status_code=404, detail="Draw not found")

    db.delete(draw)
    db.commit()

    return {
        "success": True,
        "message": "Draw deleted successfully",
    }