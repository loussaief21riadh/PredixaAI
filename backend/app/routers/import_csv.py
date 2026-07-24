import shutil
from pathlib import Path
from uuid import uuid4

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

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


@router.post("/upload")
def upload_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    """
    Upload and import a lottery CSV file.

    Access:
        Administrator only.

    Features:
        - CSV validation
        - Automatic delimiter detection
        - Duplicate detection
        - Legacy and modern Loto support
        - Per-row error handling
        - Temporary file cleanup
    """

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing filename.",
        )

    original_filename = Path(file.filename).name

    if Path(original_filename).suffix.lower() != ".csv":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are allowed.",
        )

    destination = (
        UPLOAD_FOLDER
        / f"{uuid4().hex}_{original_filename}"
    )

    imported = 0
    skipped = 0
    errors = []

    try:

        # Save temporary uploaded file
        with destination.open("wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        importer = CSVImporter(destination)

        rows = importer.load()

        for row_number, row in enumerate(
            rows,
            start=2,
        ):

            try:

                draw = LotoParser.parse(row)

                # -------------------------------------------------
                # Duplicate detection
                #
                # Same date is NOT sufficient because historical
                # FDJ datasets may contain more than one draw
                # on the same day.
                # -------------------------------------------------

                duplicate_query = (
                    db.query(Draw)
                    .filter(
                        Draw.game == draw["game"],
                        Draw.draw_date == draw["draw_date"],
                        Draw.n1 == draw["n1"],
                        Draw.n2 == draw["n2"],
                        Draw.n3 == draw["n3"],
                        Draw.n4 == draw["n4"],
                        Draw.n5 == draw["n5"],
                    )
                )

                # Legacy fields
                if draw.get("n6") is not None:
                    duplicate_query = duplicate_query.filter(
                        Draw.n6 == draw["n6"]
                    )

                if draw.get("bonus") is not None:
                    duplicate_query = duplicate_query.filter(
                        Draw.bonus == draw["bonus"]
                    )

                # Modern Loto field
                if draw.get("chance") is not None:
                    duplicate_query = duplicate_query.filter(
                        Draw.chance == draw["chance"]
                    )

                exists = duplicate_query.first()

                if exists:
                    skipped += 1
                    continue

                db_draw = Draw(
                    **draw
                )

                db.add(db_draw)

                imported += 1

            except Exception as exc:

                errors.append(
                    {
                        "row": row_number,
                        "error": str(exc),
                    }
                )

        db.commit()

        return {
            "success": True,
            "filename": original_filename,
            "total_rows": len(rows),
            "imported": imported,
            "skipped": skipped,
            "errors": len(errors),
            "error_details": errors[:20],
        }

    except (
        ValueError,
        FileNotFoundError,
    ) as exc:

        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    except Exception as exc:

        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "CSV import failed: "
                f"{str(exc)}"
            ),
        )

    finally:

        try:
            file.file.close()
        except Exception:
            pass

        if destination.exists():
            destination.unlink()