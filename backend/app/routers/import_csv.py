from fastapi import APIRouter, HTTPException
from pathlib import Path

from app.database import SessionLocal
from app.models.draw import Draw
from app.services.csv_importer import CSVImporter
from app.services.loto_parser import LotoParser

router = APIRouter(
    prefix="/import",
    tags=["Import"],
)


@router.post("/csv")
def import_csv():

    csv_path = Path("/Users/loussaiefriadh/Downloads/loto.csv")

    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail="CSV file not found.",
        )

    importer = CSVImporter(str(csv_path))

    rows = importer.load()

    db = SessionLocal()

    imported = 0

    try:

        for row in rows:

            draw = LotoParser.parse(row)

            exists = (
                db.query(Draw)
                .filter(Draw.draw_date == draw["draw_date"])
                .first()
            )

            if exists:
                continue

            db_draw = Draw(**draw)

            db.add(db_draw)

            imported += 1

        db.commit()

    finally:

        db.close()

    return {
        "success": True,
        "imported": imported,
        "total_csv": len(rows),
    }