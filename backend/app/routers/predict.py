import pandas as pd

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)
from sqlalchemy.orm import Session

from app.ai.dataset_builder import DatasetBuilder
from app.ai.feature_engineering import FeatureEngineering
from app.auth.dependencies import get_current_user
from app.auth.models import User
from app.database import get_db
from app.models.draw import Draw
from app.registry.model_registry import ModelRegistry


router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


WINDOW_SIZE = DatasetBuilder.WINDOW_SIZE

MODERN_LOTO_START_DATE = (
    DatasetBuilder.MODERN_LOTO_START_DATE
)


@router.get("/")
def predict(
    top_n: int = Query(
        default=10,
        ge=5,
        le=49,
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Predixa AI V2 ranked prediction endpoint.

    Access:
        Authenticated users only.

    Uses:
        - Modern Loto regime only
        - Last 100 draws
        - V2 feature engineering
        - 49 Random Forest models

    Important:
        Scores are model ranking outputs.
        They are not guaranteed probabilities
        of future lottery outcomes.
    """

    # --------------------------------------------------
    # Load modern Loto history only
    # --------------------------------------------------

    draws = (
        db.query(Draw)
        .filter(
            Draw.draw_date
            >= MODERN_LOTO_START_DATE
        )
        .order_by(
            Draw.draw_date.asc(),
            Draw.id.asc(),
        )
        .all()
    )

    if len(draws) < WINDOW_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Not enough modern historical draws. "
                f"At least {WINDOW_SIZE} draws are required."
            ),
        )

    history = draws[-WINDOW_SIZE:]

    # --------------------------------------------------
    # Build V2 features
    # --------------------------------------------------

    try:

        features = (
            FeatureEngineering
            .build_from_history(
                history,
                window_size=WINDOW_SIZE,
            )
        )

    except Exception as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Feature generation failed: "
                f"{str(exc)}"
            ),
        )

    X = pd.DataFrame(
        [features]
    )

    ranked_numbers = []

    # --------------------------------------------------
    # Load and score all 49 models
    # --------------------------------------------------

    for number in range(
        1,
        50,
    ):

        model_name = (
            f"random_forest_target_{number}"
        )

        try:

            model = (
                ModelRegistry.load_model(
                    model_name
                )
            )

        except Exception as exc:

            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail=(
                    f"Unable to load model "
                    f"{model_name}: "
                    f"{str(exc)}"
                ),
            )

        try:

            # Ensure feature order matches
            # the model training schema.
            if hasattr(
                model,
                "feature_names_in_",
            ):

                expected_features = list(
                    model.feature_names_in_
                )

                missing_features = [
                    feature
                    for feature
                    in expected_features
                    if feature
                    not in X.columns
                ]

                if missing_features:

                    raise ValueError(
                        "Missing features: "
                        + ", ".join(
                            missing_features[:10]
                        )
                    )

                X_model = X[
                    expected_features
                ]

            else:

                X_model = X

            probabilities = (
                model.predict_proba(
                    X_model
                )
            )

            classes = list(
                model.classes_
            )

            if 1 in classes:

                positive_index = (
                    classes.index(1)
                )

                score = float(
                    probabilities[
                        0
                    ][
                        positive_index
                    ]
                )

            else:

                score = 0.0

        except Exception as exc:

            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail=(
                    f"Prediction failed for "
                    f"number {number}: "
                    f"{str(exc)}"
                ),
            )

        ranked_numbers.append(
            {
                "number": number,

                "score": round(
                    score,
                    6,
                ),
            }
        )

    # --------------------------------------------------
    # Ranking
    # --------------------------------------------------

    ranked_numbers.sort(
        key=lambda item: (
            item["score"]
        ),
        reverse=True,
    )

    top_numbers = (
        ranked_numbers[
            :top_n
        ]
    )

    top_5 = [
        item["number"]
        for item
        in ranked_numbers[:5]
    ]

    # --------------------------------------------------
    # Response
    # --------------------------------------------------

    return {
        "success": True,

        "version": "V2",

        "user": (
            current_user.username
        ),

        "model_regime": (
            "modern_loto"
        ),

        "modern_loto_start_date": (
            MODERN_LOTO_START_DATE
        ),

        "window_size": (
            WINDOW_SIZE
        ),

        "draws_used": (
            len(history)
        ),

        "latest_draw_date": str(
            history[-1].draw_date
        ),

        "feature_count": (
            len(X.columns)
        ),

        "top_5_numbers": (
            top_5
        ),

        "top_numbers": (
            top_numbers
        ),

        "all_ranked_numbers": (
            ranked_numbers
        ),

        "disclaimer": (
            "Model scores are ranking outputs "
            "derived from historical lottery data. "
            "They do not represent guaranteed "
            "probabilities of future outcomes."
        ),
    }