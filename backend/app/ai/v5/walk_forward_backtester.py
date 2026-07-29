from collections import Counter

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from app.ai.v5.feature_engineering import V5FeatureEngineering
from app.core.logger import logger
from app.core.settings import (
    RANDOM_STATE,
    N_ESTIMATORS,
    MAX_DEPTH,
)
from app.models.draw import Draw


class V5WalkForwardBacktester:
    """
    Predixa AI V5-A Purged T-2 Walk-Forward.

    Rules
    -----
    Prediction of target T:

        Features:
            100 historical draws ending at T-2.

        T-1:
            excluded from prediction features.

        Training:
            T-1 is also purged from the training targets.

    This reproduces the temporal safeguards validated
    with Predixa V4-F while using the new V5-A features.

    V5-C diagnostic extension:
        Complete probability scores for numbers 1..49
        are included in every walk-forward detail record.

    Important:
        This does not change model training,
        ranking, or Top-5 selection.
    """

    MODERN_LOTO_START_DATE = "2008-10-06"

    WINDOW_SIZE = 100
    TOP_K = 5
    MAX_TRAINING_SAMPLES = 1500
    MONTE_CARLO_SIMULATIONS = 10000

    LAG_DRAWS = 1
    PURGE_DRAWS = 1

    # --------------------------------------------------
    # Draw helper
    # --------------------------------------------------

    @staticmethod
    def _main_numbers(
        draw: Draw,
    ) -> list[int]:
        return [
            draw.n1,
            draw.n2,
            draw.n3,
            draw.n4,
            draw.n5,
        ]

    # --------------------------------------------------
    # V5 training dataset
    # --------------------------------------------------

    @staticmethod
    def _build_training_dataset(
        draws: list[Draw],
        window_size: int,
        max_training_samples: int,
    ):
        """
        For each training target K:

            features end at K-2
            K-1 is excluded
            target = K

        The caller is responsible for purging T-1
        from the complete training period before this
        function is called.
        """

        minimum_required = (
            window_size
            + V5WalkForwardBacktester.LAG_DRAWS
            + 1
        )

        if len(draws) < minimum_required:
            raise ValueError(
                "Not enough historical draws to build "
                "the V5-A training dataset."
            )

        first_target_index = (
            window_size
            + V5WalkForwardBacktester.LAG_DRAWS
        )

        if max_training_samples > 0:
            first_target_index = max(
                first_target_index,
                len(draws) - max_training_samples,
            )

        feature_rows = []
        target_rows = []

        for target_index in range(
            first_target_index,
            len(draws),
        ):
            feature_end_index = (
                target_index
                - V5WalkForwardBacktester.LAG_DRAWS
            )

            feature_start_index = (
                feature_end_index
                - window_size
            )

            history = draws[
                feature_start_index:
                feature_end_index
            ]

            if len(history) != window_size:
                continue

            target_draw = draws[
                target_index
            ]

            features = (
                V5FeatureEngineering
                .build_from_history(
                    history,
                    window_size=window_size,
                )
            )

            actual_numbers = set(
                V5WalkForwardBacktester
                ._main_numbers(
                    target_draw
                )
            )

            targets = {
                number: (
                    1
                    if number in actual_numbers
                    else 0
                )
                for number in range(
                    1,
                    50,
                )
            }

            feature_rows.append(
                features
            )

            target_rows.append(
                targets
            )

        X = pd.DataFrame(
            feature_rows
        )

        y = pd.DataFrame(
            target_rows
        )

        if X.empty or y.empty:
            raise ValueError(
                "V5-A walk-forward training dataset is empty."
            )

        if len(X) != len(y):
            raise ValueError(
                "V5-A X and y sizes do not match."
            )

        if X.isnull().any().any():
            raise ValueError(
                "V5-A training features contain missing values."
            )

        if y.isnull().any().any():
            raise ValueError(
                "V5-A training targets contain missing values."
            )

        if X.shape[1] != 396:
            raise ValueError(
                "Unexpected V5-A feature count. "
                f"Expected 396, received {X.shape[1]}."
            )

        return X, y

    # --------------------------------------------------
    # Probability helper
    # --------------------------------------------------

    @staticmethod
    def _positive_probability(
        model,
        X_prediction,
    ) -> float:
        probabilities = (
            model.predict_proba(
                X_prediction
            )
        )

        classes = list(
            model.classes_
        )

        if 1 not in classes:
            return 0.0

        positive_index = (
            classes.index(1)
        )

        return float(
            probabilities[0][positive_index]
        )

    # --------------------------------------------------
    # Frequency baseline
    # --------------------------------------------------

    @staticmethod
    def _frequency_top_5(
        history: list[Draw],
        window_size: int,
    ) -> list[int]:
        recent_history = history[
            -window_size:
        ]

        counter = Counter()

        for draw in recent_history:
            counter.update(
                V5WalkForwardBacktester
                ._main_numbers(draw)
            )

        ranked = sorted(
            range(1, 50),
            key=lambda number: (
                -counter.get(
                    number,
                    0,
                ),
                number,
            ),
        )

        return ranked[
            :V5WalkForwardBacktester.TOP_K
        ]

    # --------------------------------------------------
    # Previous-draw baseline
    # --------------------------------------------------

    @staticmethod
    def _previous_draw_top_5(
        history: list[Draw],
    ) -> list[int]:
        if not history:
            raise ValueError(
                "History cannot be empty."
            )

        return sorted(
            V5WalkForwardBacktester
            ._main_numbers(
                history[-1]
            )
        )

    # --------------------------------------------------
    # Monte-Carlo
    # --------------------------------------------------

    @staticmethod
    def _monte_carlo_random_baseline(
        evaluated_draws: int,
        observed_average_hits: float,
        simulations: int,
    ) -> dict:
        if evaluated_draws <= 0:
            raise ValueError(
                "evaluated_draws must be positive."
            )

        if simulations < 100:
            raise ValueError(
                "At least 100 simulations are required."
            )

        rng = np.random.default_rng(
            RANDOM_STATE
        )

        universe = np.arange(
            1,
            50,
        )

        experiment_averages = np.empty(
            simulations,
            dtype=float,
        )

        for simulation_index in range(
            simulations
        ):
            total_hits = 0

            for _ in range(
                evaluated_draws
            ):
                actual = rng.choice(
                    universe,
                    size=5,
                    replace=False,
                )

                predicted = rng.choice(
                    universe,
                    size=5,
                    replace=False,
                )

                total_hits += len(
                    set(actual.tolist())
                    & set(predicted.tolist())
                )

            experiment_averages[
                simulation_index
            ] = (
                total_hits
                / evaluated_draws
            )

        mean_hits = float(
            np.mean(
                experiment_averages
            )
        )

        std_hits = float(
            np.std(
                experiment_averages,
                ddof=1,
            )
        )

        lower_95 = float(
            np.percentile(
                experiment_averages,
                2.5,
            )
        )

        upper_95 = float(
            np.percentile(
                experiment_averages,
                97.5,
            )
        )

        empirical_p_value = float(
            (
                np.sum(
                    experiment_averages
                    >= observed_average_hits
                )
                + 1
            )
            / (
                simulations
                + 1
            )
        )

        percentile = float(
            np.mean(
                experiment_averages
                <= observed_average_hits
            )
        )

        return {
            "simulations": simulations,

            "mean_average_hits_at_5": round(
                mean_hits,
                4,
            ),

            "std_average_hits_at_5": round(
                std_hits,
                4,
            ),

            "95_percent_interval": {
                "lower": round(
                    lower_95,
                    4,
                ),
                "upper": round(
                    upper_95,
                    4,
                ),
            },

            "predixa_percentile": round(
                percentile,
                4,
            ),

            "empirical_p_value": round(
                empirical_p_value,
                4,
            ),
        }

    # --------------------------------------------------
    # Main V5 walk-forward
    # --------------------------------------------------

    @staticmethod
    def run(
        db,
        test_draws: int = 5,
        window_size: int = 100,
        max_training_samples: int = 1500,
        monte_carlo_simulations: int = 10000,
    ):
        if test_draws < 5:
            raise ValueError(
                "test_draws must be at least 5."
            )

        if window_size < 100:
            raise ValueError(
                "V5-A requires window_size >= 100."
            )

        if max_training_samples < 0:
            raise ValueError(
                "max_training_samples cannot be negative."
            )

        if monte_carlo_simulations < 100:
            raise ValueError(
                "monte_carlo_simulations must be at least 100."
            )

        logger.info(
            "=" * 60
        )

        logger.info(
            "PREDIXA AI V5-A - PURGED T-2 WALK-FORWARD"
        )

        logger.info(
            "=" * 60
        )

        # --------------------------------------------------
        # Load data
        # --------------------------------------------------

        all_draws = (
            db.query(Draw)
            .filter(
                Draw.draw_date
                >= V5WalkForwardBacktester
                .MODERN_LOTO_START_DATE
            )
            .order_by(
                Draw.draw_date.asc(),
                Draw.id.asc(),
            )
            .all()
        )

        minimum_required = (
            window_size
            + V5WalkForwardBacktester.LAG_DRAWS
            + V5WalkForwardBacktester.PURGE_DRAWS
            + test_draws
            + 1
        )

        if len(all_draws) < minimum_required:
            raise ValueError(
                "Not enough modern draws."
            )

        first_test_index = (
            len(all_draws)
            - test_draws
        )

        details = []

        total_hits = 0
        total_frequency_hits = 0
        total_previous_draw_hits = 0

        hit_distribution = {
            i: 0
            for i in range(6)
        }

        frequency_hit_distribution = {
            i: 0
            for i in range(6)
        }

        previous_draw_hit_distribution = {
            i: 0
            for i in range(6)
        }

        at_least_1 = 0
        at_least_2 = 0

        frequency_at_least_1 = 0
        frequency_at_least_2 = 0

        previous_at_least_1 = 0
        previous_at_least_2 = 0

        total_overlap_with_previous_draw = 0
        exact_previous_draw_copies = 0

        evaluated = 0

        # --------------------------------------------------
        # Walk forward
        # --------------------------------------------------

        for test_position, target_index in enumerate(
            range(
                first_test_index,
                len(all_draws),
            ),
            start=1,
        ):
            target_draw = all_draws[
                target_index
            ]

            historical_draws = all_draws[
                :target_index
            ]

            logger.info(
                "V5 walk-forward %s/%s | target=%s",
                test_position,
                test_draws,
                target_draw.draw_date,
            )

            # ----------------------------------------------
            # Purge T-1 from training.
            # ----------------------------------------------

            purged_training_draws = (
                historical_draws[
                    :-V5WalkForwardBacktester.PURGE_DRAWS
                ]
            )

            X_train, y_train = (
                V5WalkForwardBacktester
                ._build_training_dataset(
                    purged_training_draws,
                    window_size=window_size,
                    max_training_samples=(
                        max_training_samples
                    ),
                )
            )

            # ----------------------------------------------
            # Prediction history ends at T-2.
            # ----------------------------------------------

            feature_history = (
                historical_draws[
                    -(
                        window_size
                        + V5WalkForwardBacktester.LAG_DRAWS
                    ):
                    -V5WalkForwardBacktester.LAG_DRAWS
                ]
            )

            if len(feature_history) != window_size:
                raise ValueError(
                    "Invalid V5-A prediction history size."
                )

            prediction_features = (
                V5FeatureEngineering
                .build_from_history(
                    feature_history,
                    window_size=window_size,
                )
            )

            X_prediction = pd.DataFrame(
                [prediction_features]
            )

            X_prediction = (
                X_prediction[
                    X_train.columns
                ]
            )

            # ----------------------------------------------
            # Train 49 binary models
            # ----------------------------------------------

            ranked_numbers = []

            for number in range(
                1,
                50,
            ):
                target = y_train[
                    number
                ]

                if target.nunique() < 2:
                    score = float(
                        target.iloc[0]
                    )

                else:
                    model = (
                        RandomForestClassifier(
                            n_estimators=N_ESTIMATORS,
                            max_depth=MAX_DEPTH,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                            class_weight="balanced",
                        )
                    )

                    model.fit(
                        X_train,
                        target,
                    )

                    score = (
                        V5WalkForwardBacktester
                        ._positive_probability(
                            model,
                            X_prediction,
                        )
                    )

                ranked_numbers.append(
                    {
                        "number": number,
                        "score": score,
                    }
                )

            ranked_numbers.sort(
                key=lambda item: (
                    -item["score"],
                    item["number"],
                )
            )

            # ----------------------------------------------
            # Complete probability vector
            # ----------------------------------------------
            #
            # These are exactly the same scores already used
            # to rank the 49 numbers.
            #
            # No additional model is trained here.
            # No ranking logic is changed.
            # ----------------------------------------------

            probabilities = {
                item["number"]: float(
                    item["score"]
                )
                for item in ranked_numbers
            }

            if len(probabilities) != 49:
                raise ValueError(
                    "Expected 49 prediction probabilities."
                )

            predicted_top_5 = [
                item["number"]
                for item in ranked_numbers[
                    :V5WalkForwardBacktester.TOP_K
                ]
            ]

            # ----------------------------------------------
            # Actual numbers
            # ----------------------------------------------

            actual_numbers = sorted(
                V5WalkForwardBacktester
                ._main_numbers(
                    target_draw
                )
            )

            actual_set = set(
                actual_numbers
            )

            predicted_set = set(
                predicted_top_5
            )

            hits = len(
                predicted_set
                & actual_set
            )

            total_hits += hits

            hit_distribution[
                hits
            ] += 1

            if hits >= 1:
                at_least_1 += 1

            if hits >= 2:
                at_least_2 += 1

            # ----------------------------------------------
            # Frequency baseline
            # ----------------------------------------------

            frequency_top_5 = (
                V5WalkForwardBacktester
                ._frequency_top_5(
                    historical_draws,
                    window_size,
                )
            )

            frequency_hits = len(
                set(
                    frequency_top_5
                )
                & actual_set
            )

            total_frequency_hits += (
                frequency_hits
            )

            frequency_hit_distribution[
                frequency_hits
            ] += 1

            if frequency_hits >= 1:
                frequency_at_least_1 += 1

            if frequency_hits >= 2:
                frequency_at_least_2 += 1

            # ----------------------------------------------
            # Previous-draw baseline
            # ----------------------------------------------

            previous_draw_top_5 = (
                V5WalkForwardBacktester
                ._previous_draw_top_5(
                    historical_draws
                )
            )

            previous_set = set(
                previous_draw_top_5
            )

            previous_hits = len(
                previous_set
                & actual_set
            )

            total_previous_draw_hits += (
                previous_hits
            )

            previous_draw_hit_distribution[
                previous_hits
            ] += 1

            if previous_hits >= 1:
                previous_at_least_1 += 1

            if previous_hits >= 2:
                previous_at_least_2 += 1

            # ----------------------------------------------
            # Copy diagnostic
            # ----------------------------------------------

            overlap_previous = len(
                predicted_set
                & previous_set
            )

            total_overlap_with_previous_draw += (
                overlap_previous
            )

            exact_copy = (
                predicted_set
                == previous_set
            )

            if exact_copy:
                exact_previous_draw_copies += 1

            evaluated += 1

            details.append(
                {
                    "draw_date": str(
                        target_draw.draw_date
                    ),

                    "predicted_top_5": (
                        predicted_top_5
                    ),

                    "probabilities": (
                        probabilities
                    ),

                    "actual_numbers": (
                        actual_numbers
                    ),

                    "hits": hits,

                    "frequency_top_5": (
                        frequency_top_5
                    ),

                    "frequency_hits": (
                        frequency_hits
                    ),

                    "previous_draw_top_5": (
                        previous_draw_top_5
                    ),

                    "previous_draw_hits": (
                        previous_hits
                    ),

                    "predixa_overlap_with_previous_draw": (
                        overlap_previous
                    ),

                    "predixa_exact_previous_draw_copy": (
                        exact_copy
                    ),

                    "training_samples": len(
                        X_train
                    ),

                    "feature_count": (
                        X_train.shape[1]
                    ),

                    "last_allowed_feature_draw": str(
                        feature_history[-1]
                        .draw_date
                    ),

                    "excluded_previous_draw": str(
                        historical_draws[-1]
                        .draw_date
                    ),

                    "last_training_target_available": str(
                        purged_training_draws[-1]
                        .draw_date
                    ),
                }
            )

        if evaluated == 0:
            raise ValueError(
                "No draws were evaluated."
            )

        # --------------------------------------------------
        # Aggregate metrics
        # --------------------------------------------------

        average_hits = (
            total_hits
            / evaluated
        )

        frequency_average_hits = (
            total_frequency_hits
            / evaluated
        )

        previous_average_hits = (
            total_previous_draw_hits
            / evaluated
        )

        theoretical_random_hits = (
            V5WalkForwardBacktester.TOP_K
            * 5
            / 49
        )

        monte_carlo = (
            V5WalkForwardBacktester
            ._monte_carlo_random_baseline(
                evaluated_draws=evaluated,
                observed_average_hits=(
                    average_hits
                ),
                simulations=(
                    monte_carlo_simulations
                ),
            )
        )

        average_overlap = (
            total_overlap_with_previous_draw
            / evaluated
        )

        exact_copy_rate = (
            exact_previous_draw_copies
            / evaluated
        )

        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        result = {
            "status": "success",

            "version": (
                "V5-A-PURGED-T2"
            ),

            "evaluation_type": (
                "strict_walk_forward_purged"
            ),

            "evaluated_draws": (
                evaluated
            ),

            "window_size": (
                window_size
            ),

            "max_training_samples": (
                max_training_samples
            ),

            "feature_count": 396,

            "top_k": (
                V5WalkForwardBacktester.TOP_K
            ),

            "probability_vectors_included": True,

            "probability_vector_size": 49,

            "model": {
                "average_hits_at_5": round(
                    average_hits,
                    4,
                ),

                "precision_at_5": round(
                    total_hits
                    / (
                        evaluated
                        * V5WalkForwardBacktester.TOP_K
                    ),
                    4,
                ),

                "recall_at_5": round(
                    total_hits
                    / (
                        evaluated
                        * 5
                    ),
                    4,
                ),

                "total_hits": (
                    total_hits
                ),

                "hit_distribution": (
                    hit_distribution
                ),

                "at_least_1_hit_rate": round(
                    at_least_1
                    / evaluated,
                    4,
                ),

                "at_least_2_hit_rate": round(
                    at_least_2
                    / evaluated,
                    4,
                ),
            },

            "frequency_baseline": {
                "average_hits_at_5": round(
                    frequency_average_hits,
                    4,
                ),

                "total_hits": (
                    total_frequency_hits
                ),

                "hit_distribution": (
                    frequency_hit_distribution
                ),

                "at_least_1_hit_rate": round(
                    frequency_at_least_1
                    / evaluated,
                    4,
                ),

                "at_least_2_hit_rate": round(
                    frequency_at_least_2
                    / evaluated,
                    4,
                ),
            },

            "previous_draw_baseline": {
                "average_hits_at_5": round(
                    previous_average_hits,
                    4,
                ),

                "total_hits": (
                    total_previous_draw_hits
                ),

                "hit_distribution": (
                    previous_draw_hit_distribution
                ),

                "at_least_1_hit_rate": round(
                    previous_at_least_1
                    / evaluated,
                    4,
                ),

                "at_least_2_hit_rate": round(
                    previous_at_least_2
                    / evaluated,
                    4,
                ),
            },

            "random_baseline": {
                "theoretical": {
                    "expected_average_hits_at_5": round(
                        theoretical_random_hits,
                        4,
                    ),
                },

                "monte_carlo": (
                    monte_carlo
                ),
            },

            "previous_draw_diagnostics": {
                "average_predixa_overlap_with_previous_draw": round(
                    average_overlap,
                    4,
                ),

                "exact_previous_draw_copy_count": (
                    exact_previous_draw_copies
                ),

                "exact_previous_draw_copy_rate": round(
                    exact_copy_rate,
                    4,
                ),

                "maximum_possible_overlap": 5,
            },

            "comparison": {
                "absolute_lift_vs_frequency": round(
                    average_hits
                    - frequency_average_hits,
                    4,
                ),

                "absolute_lift_vs_previous_draw": round(
                    average_hits
                    - previous_average_hits,
                    4,
                ),

                "absolute_lift_vs_random": round(
                    average_hits
                    - theoretical_random_hits,
                    4,
                ),

                "beats_frequency": (
                    average_hits
                    > frequency_average_hits
                ),

                "beats_previous_draw": (
                    average_hits
                    > previous_average_hits
                ),

                "beats_random_expectation": (
                    average_hits
                    > theoretical_random_hits
                ),
            },

            "details": details,
        }

        logger.info(
            "V5-A completed | "
            "Predixa=%.4f | "
            "Frequency=%.4f | "
            "Previous=%.4f | "
            "Random=%.4f | "
            "T-1 overlap=%.4f/5 | "
            "T-1 copy rate=%.4f | "
            "p=%.4f",
            average_hits,
            frequency_average_hits,
            previous_average_hits,
            theoretical_random_hits,
            average_overlap,
            exact_copy_rate,
            monte_carlo[
                "empirical_p_value"
            ],
        )

        return result