import shutil
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_admin
from app.auth.models import User
from app.database import get_db
from app.models.draw import Draw
from app.services.csv_importer import CSVImporter
from app.services.loto_parser import LotoParser

router = APIRouter(
    prefix="/import",
    tags=["Import"],
)

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)


@router.post("/upload")
def upload_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    """
    Upload and import a lottery CSV file.

    Administrator only.
    """

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are allowed.",
        )

    destination = UPLOAD_FOLDER / file.filename

    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    importer = CSVImporter(destination)

    rows = importer.load()

    imported = 0
    skipped = 0

    try:

        for row in rows:

            draw = LotoParser.parse(row)

            exists = (
                db.query(Draw)
                .filter(Draw.draw_date == draw["draw_date"])
                .first()
            )

            if exists:
                skipped += 1
                continue

            db.add(Draw(**draw))
            imported += 1

        db.commit()

    except Exception as exc:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    finally:

        if destination.exists():
            destination.unlink()

    return {
        "success": True,
        "filename": file.filename,
        "rows": len(rows),
        "imported": imported,
        "skipped": skipped,
    }