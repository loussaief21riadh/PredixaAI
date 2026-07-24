from collections import Counter

import pandas as pd
from sqlalchemy.orm import Session

from app.ai.dataset_builder import DatasetBuilder
from app.ai.feature_engineering import FeatureEngineering
from app.models.draw import Draw
from app.registry.model_registry import ModelRegistry


class Backtester:
    """
    Predixa AI V2 ranking backtester.

    Measures how often the Top-5 ranked numbers
    match the actual next draw.

    This version uses the already-trained models
    and replays historical windows chronologically.

    Important:
        This is not yet a full retrain-at-each-step
        walk-forward backtest.
    """

    TOP_K = 5

    @staticmethod
    def _load_models():
        """
        Load all 49 trained Random Forest models.
        """

        models = {}

        for number in range(1, 50):

            model_name = (
                f"random_forest_target_{number}"
            )

            models[number] = (
                ModelRegistry.load_model(
                    model_name
                )
            )

        return models

    @staticmethod
    def _score_numbers(
        models: dict,
        X: pd.DataFrame,
    ) -> list[dict]:
        """
        Generate ranking scores for all numbers.
        """

        ranked = []

        for number in range(1, 50):

            model = models[number]

            if hasattr(
                model,
                "feature_names_in_",
            ):

                expected_features = list(
                    model.feature_names_in_
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

            ranked.append(
                {
                    "number": number,
                    "score": score,
                }
            )

        ranked.sort(
            key=lambda item: (
                item["score"]
            ),
            reverse=True,
        )

        return ranked

    @staticmethod
    def _frequency_baseline(
        history: list[Draw],
    ) -> list[int]:
        """
        Frequency-based Top-5 baseline.
        """

        counter = Counter()

        for draw in history:

            numbers = [
                draw.n1,
                draw.n2,
                draw.n3,
                draw.n4,
                draw.n5,
            ]

            counter.update(
                numbers
            )

        return [
            number
            for number, _
            in counter.most_common(
                Backtester.TOP_K
            )
        ]

    @staticmethod
    def run(
        db: Session,
        test_draws: int = 200,
    ):
        """
        Run chronological ranking backtest.

        Parameters
        ----------
        test_draws:
            Number of most recent historical draws
            to evaluate.
        """

        window_size = (
            DatasetBuilder.WINDOW_SIZE
        )

        modern_start = (
            DatasetBuilder
            .MODERN_LOTO_START_DATE
        )

        draws = (
            db.query(Draw)
            .filter(
                Draw.draw_date
                >= modern_start
            )
            .order_by(
                Draw.draw_date.asc(),
                Draw.id.asc(),
            )
            .all()
        )

        if len(draws) <= window_size:
            raise ValueError(
                "Not enough modern draws "
                "for backtesting."
            )

        available_tests = (
            len(draws)
            - window_size
        )

        test_draws = min(
            test_draws,
            available_tests,
        )

        start_index = (
            len(draws)
            - test_draws
        )

        models = (
            Backtester._load_models()
        )

        hit_distribution = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
        }

        frequency_hit_distribution = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
        }

        total_hits = 0
        total_frequency_hits = 0

        evaluated = 0

        details = []

        for target_index in range(
            start_index,
            len(draws),
        ):

            history_start = (
                target_index
                - window_size
            )

            if history_start < 0:
                continue

            history = draws[
                history_start:
                target_index
            ]

            target_draw = draws[
                target_index
            ]

            features = (
                FeatureEngineering
                .build_from_history(
                    history,
                    window_size=window_size,
                )
            )

            X = pd.DataFrame(
                [features]
            )

            ranked = (
                Backtester
                ._score_numbers(
                    models,
                    X,
                )
            )

            predicted_top_5 = [
                item["number"]
                for item
                in ranked[
                    :Backtester.TOP_K
                ]
            ]

            actual_numbers = {
                target_draw.n1,
                target_draw.n2,
                target_draw.n3,
                target_draw.n4,
                target_draw.n5,
            }

            hits = len(
                set(
                    predicted_top_5
                )
                & actual_numbers
            )

            hit_distribution[
                hits
            ] += 1

            total_hits += hits

            frequency_top_5 = (
                Backtester
                ._frequency_baseline(
                    history
                )
            )

            frequency_hits = len(
                set(
                    frequency_top_5
                )
                & actual_numbers
            )

            frequency_hit_distribution[
                frequency_hits
            ] += 1

            total_frequency_hits += (
                frequency_hits
            )

            evaluated += 1

            if len(details) < 20:

                details.append(
                    {
                        "draw_date": str(
                            target_draw
                            .draw_date
                        ),

                        "predicted_top_5": (
                            predicted_top_5
                        ),

                        "actual_numbers": sorted(
                            actual_numbers
                        ),

                        "hits": hits,

                        "frequency_top_5": (
                            frequency_top_5
                        ),

                        "frequency_hits": (
                            frequency_hits
                        ),
                    }
                )

        if evaluated == 0:

            raise ValueError(
                "No draws were evaluated."
            )

        average_hits = (
            total_hits
            / evaluated
        )

        average_frequency_hits = (
            total_frequency_hits
            / evaluated
        )

        precision_at_5 = (
            total_hits
            / (
                evaluated
                * Backtester.TOP_K
            )
        )

        recall_at_5 = (
            precision_at_5
        )

        frequency_precision_at_5 = (
            total_frequency_hits
            / (
                evaluated
                * Backtester.TOP_K
            )
        )

        return {
            "status": "success",

            "version": "V2",

            "evaluated_draws": (
                evaluated
            ),

            "window_size": (
                window_size
            ),

            "top_k": (
                Backtester.TOP_K
            ),

            "model": {
                "average_hits_at_5": round(
                    average_hits,
                    4,
                ),

                "precision_at_5": round(
                    precision_at_5,
                    4,
                ),

                "recall_at_5": round(
                    recall_at_5,
                    4,
                ),

                "hit_distribution": (
                    hit_distribution
                ),

                "at_least_1_hit_rate": round(
                    (
                        evaluated
                        - hit_distribution[0]
                    )
                    / evaluated,
                    4,
                ),

                "at_least_2_hit_rate": round(
                    (
                        hit_distribution[2]
                        + hit_distribution[3]
                        + hit_distribution[4]
                        + hit_distribution[5]
                    )
                    / evaluated,
                    4,
                ),
            },

            "frequency_baseline": {
                "average_hits_at_5": round(
                    average_frequency_hits,
                    4,
                ),

                "precision_at_5": round(
                    frequency_precision_at_5,
                    4,
                ),

                "hit_distribution": (
                    frequency_hit_distribution
                ),
            },

            "details_sample": (
                details
            ),
        }