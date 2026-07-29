from collections import Counter

import numpy as np

from app.ai.v5.feature_engineering_v5b import (
    V5BFeatureEngineering,
)
from app.ai.v6.ranking_dataset import (
    V6RankingDataset,
)
from app.ai.v6.ranking_model import (
    V6RankingModel,
)
from app.core.logger import logger
from app.core.settings import RANDOM_STATE
from app.models.draw import Draw


class V6WalkForwardBacktester:
    """
    Predixa AI V6 - Global Ranking Walk-Forward.

    Temporal protocol:
        target = T
        prediction features end at T-2
        T-1 excluded from prediction features
        T-1 purged from training targets
        strict chronological walk-forward

    Architecture:
        - one global Random Forest per evaluation step
        - candidate-level dataset
        - 49 candidates scored by the same model
        - Top-5 selected from the ranking
    """

    VERSION = "V6-GLOBAL-RANKING-PURGED-T2"

    MODERN_LOTO_START_DATE = "2008-10-06"

    TOP_K = 5

    LAG_DRAWS = 1
    PURGE_DRAWS = 1

    RANDOM_EXPECTATION = 25 / 49

    # ==========================================================
    # DRAW HELPERS
    # ==========================================================

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

    # ==========================================================
    # FREQUENCY BASELINE
    # ==========================================================

    @classmethod
    def _frequency_top_5(
        cls,
        history: list[Draw],
        window_size: int,
    ) -> list[int]:
        recent = history[
            -window_size:
        ]

        counter = Counter()

        for draw in recent:
            counter.update(
                cls._main_numbers(
                    draw
                )
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
            :cls.TOP_K
        ]

    # ==========================================================
    # PREVIOUS-DRAW BASELINE
    # ==========================================================

    @classmethod
    def _previous_draw_top_5(
        cls,
        history: list[Draw],
    ) -> list[int]:
        if not history:
            raise ValueError(
                "History cannot be empty."
            )

        return sorted(
            cls._main_numbers(
                history[-1]
            )
        )

    # ==========================================================
    # MONTE CARLO BASELINE
    # ==========================================================

    @classmethod
    def _monte_carlo_random_baseline(
        cls,
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

        averages = np.empty(
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
                    &
                    set(predicted.tolist())
                )

            averages[
                simulation_index
            ] = (
                total_hits
                / evaluated_draws
            )

        empirical_p = (
            (
                np.sum(
                    averages
                    >= observed_average_hits
                )
                + 1
            )
            /
            (
                simulations
                + 1
            )
        )

        return {
            "simulations": simulations,

            "mean_average_hits_at_5": round(
                float(
                    np.mean(
                        averages
                    )
                ),
                4,
            ),

            "std_average_hits_at_5": round(
                float(
                    np.std(
                        averages,
                        ddof=1,
                    )
                ),
                4,
            ),

            "95_percent_interval": {
                "lower": round(
                    float(
                        np.percentile(
                            averages,
                            2.5,
                        )
                    ),
                    4,
                ),

                "upper": round(
                    float(
                        np.percentile(
                            averages,
                            97.5,
                        )
                    ),
                    4,
                ),
            },

            "predixa_percentile": round(
                float(
                    np.mean(
                        averages
                        <= observed_average_hits
                    )
                ),
                4,
            ),

            "empirical_p_value": round(
                float(
                    empirical_p
                ),
                4,
            ),
        }

    # ==========================================================
    # MAIN WALK-FORWARD
    # ==========================================================

    @classmethod
    def run(
        cls,
        db,
        test_draws: int = 5,
        window_size: int = 100,
        max_training_targets: int = 1500,
        monte_carlo_simulations: int = 10000,
    ):
        if test_draws < 5:
            raise ValueError(
                "test_draws must be at least 5."
            )

        if window_size < 100:
            raise ValueError(
                "V6 requires window_size >= 100."
            )

        if max_training_targets < 0:
            raise ValueError(
                "max_training_targets cannot be negative."
            )

        if monte_carlo_simulations < 100:
            raise ValueError(
                "monte_carlo_simulations must be at least 100."
            )

        logger.info(
            "=" * 60
        )

        logger.info(
            "PREDIXA AI V6 - GLOBAL RANKING WALK-FORWARD"
        )

        logger.info(
            "=" * 60
        )

        # --------------------------------------------------
        # LOAD MODERN DRAWS
        # --------------------------------------------------

        all_draws = (
            db.query(Draw)
            .filter(
                Draw.draw_date
                >= cls.MODERN_LOTO_START_DATE
            )
            .order_by(
                Draw.draw_date.asc(),
                Draw.id.asc(),
            )
            .all()
        )

        minimum_required = (
            window_size
            + cls.LAG_DRAWS
            + cls.PURGE_DRAWS
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
        total_previous_hits = 0

        hit_distribution = {
            i: 0
            for i in range(6)
        }

        frequency_hit_distribution = {
            i: 0
            for i in range(6)
        }

        previous_hit_distribution = {
            i: 0
            for i in range(6)
        }

        at_least_1 = 0
        at_least_2 = 0

        frequency_at_least_1 = 0
        frequency_at_least_2 = 0

        previous_at_least_1 = 0
        previous_at_least_2 = 0

        total_overlap_previous = 0
        exact_previous_copies = 0

        evaluated = 0

        # --------------------------------------------------
        # WALK FORWARD
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
                "V6 walk-forward %s/%s | target=%s",
                test_position,
                test_draws,
                target_draw.draw_date,
            )

            # --------------------------------------------------
            # PURGE T-1 FROM TRAINING TARGETS
            # --------------------------------------------------

            purged_training_draws = (
                historical_draws[
                    :-cls.PURGE_DRAWS
                ]
            )

            training_dataset, training_metadata = (
                V6RankingDataset
                .build_from_draws(
                    draws=purged_training_draws,
                    window_size=window_size,
                    max_training_targets=(
                        max_training_targets
                    ),
                )
            )

            # --------------------------------------------------
            # TRAIN ONE GLOBAL MODEL
            # --------------------------------------------------

            model = V6RankingModel()

            model.fit(
                training_dataset
            )

            # --------------------------------------------------
            # PREDICTION HISTORY ENDS AT T-2
            # --------------------------------------------------

            feature_history = (
                historical_draws[
                    -(
                        window_size
                        + cls.LAG_DRAWS
                    ):
                    -cls.LAG_DRAWS
                ]
            )

            if len(feature_history) != window_size:
                raise ValueError(
                    "Invalid V6 prediction history size."
                )

            features = (
                V5BFeatureEngineering
                .build_from_history(
                    feature_history,
                    window_size=window_size,
                    variant="full",
                )
            )

            prediction = (
                model.predict_top_k(
                    features=features,
                    top_k=cls.TOP_K,
                )
            )

            predicted_top_5 = (
                prediction[
                    "predicted_numbers"
                ]
            )

            probabilities = (
                prediction[
                    "probabilities"
                ]
            )

            ranking = (
                prediction[
                    "ranking"
                ]
            )

            # --------------------------------------------------
            # ACTUAL NUMBERS
            # --------------------------------------------------

            actual_numbers = sorted(
                cls._main_numbers(
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

            # --------------------------------------------------
            # FREQUENCY BASELINE
            # --------------------------------------------------

            frequency_top_5 = (
                cls._frequency_top_5(
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

            # --------------------------------------------------
            # PREVIOUS DRAW BASELINE
            # --------------------------------------------------

            previous_top_5 = (
                cls._previous_draw_top_5(
                    historical_draws
                )
            )

            previous_set = set(
                previous_top_5
            )

            previous_hits = len(
                previous_set
                & actual_set
            )

            total_previous_hits += (
                previous_hits
            )

            previous_hit_distribution[
                previous_hits
            ] += 1

            if previous_hits >= 1:
                previous_at_least_1 += 1

            if previous_hits >= 2:
                previous_at_least_2 += 1

            # --------------------------------------------------
            # PREVIOUS-DRAW COPY DIAGNOSTIC
            # --------------------------------------------------

            overlap_previous = len(
                predicted_set
                & previous_set
            )

            total_overlap_previous += (
                overlap_previous
            )

            exact_previous_copy = (
                predicted_set
                == previous_set
            )

            if exact_previous_copy:
                exact_previous_copies += 1

            evaluated += 1

            # --------------------------------------------------
            # DETAIL
            # --------------------------------------------------

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

                    "ranking": (
                        ranking
                    ),

                    "actual_numbers": (
                        actual_numbers
                    ),

                    "hits": (
                        hits
                    ),

                    "frequency_top_5": (
                        frequency_top_5
                    ),

                    "frequency_hits": (
                        frequency_hits
                    ),

                    "previous_draw_top_5": (
                        previous_top_5
                    ),

                    "previous_draw_hits": (
                        previous_hits
                    ),

                    "predixa_overlap_with_previous_draw": (
                        overlap_previous
                    ),

                    "predixa_exact_previous_draw_copy": (
                        exact_previous_copy
                    ),

                    "training_target_count": (
                        len(
                            training_metadata
                        )
                    ),

                    "training_candidate_rows": (
                        len(
                            training_dataset
                        )
                    ),

                    "feature_count": (
                        prediction[
                            "feature_count"
                        ]
                    ),

                    "candidate_count": (
                        prediction[
                            "candidate_count"
                        ]
                    ),

                    "last_allowed_feature_draw": str(
                        feature_history[
                            -1
                        ].draw_date
                    ),

                    "excluded_previous_draw": str(
                        historical_draws[
                            -1
                        ].draw_date
                    ),

                    "last_training_target_available": (
                        training_metadata[
                            -1
                        ][
                            "target_date"
                        ]
                    ),
                }
            )

        if evaluated == 0:
            raise ValueError(
                "No V6 draws were evaluated."
            )

        # --------------------------------------------------
        # AGGREGATE METRICS
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
            total_previous_hits
            / evaluated
        )

        average_overlap_previous = (
            total_overlap_previous
            / evaluated
        )

        exact_copy_rate = (
            exact_previous_copies
            / evaluated
        )

        monte_carlo = (
            cls._monte_carlo_random_baseline(
                evaluated_draws=evaluated,
                observed_average_hits=(
                    average_hits
                ),
                simulations=(
                    monte_carlo_simulations
                ),
            )
        )

        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        result = {
            "status": "success",

            "version": cls.VERSION,

            "evaluation_type": (
                "strict_walk_forward_purged"
            ),

            "architecture": (
                "single_global_candidate_ranking_model"
            ),

            "evaluated_draws": (
                evaluated
            ),

            "window_size": (
                window_size
            ),

            "max_training_targets": (
                max_training_targets
            ),

            "feature_count": 12,

            "candidate_count": 49,

            "top_k": cls.TOP_K,

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
                        * cls.TOP_K
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
                    total_previous_hits
                ),

                "hit_distribution": (
                    previous_hit_distribution
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
                        cls.RANDOM_EXPECTATION,
                        4,
                    ),
                },

                "monte_carlo": (
                    monte_carlo
                ),
            },

            "previous_draw_diagnostics": {
                "average_predixa_overlap_with_previous_draw": round(
                    average_overlap_previous,
                    4,
                ),

                "exact_previous_draw_copy_count": (
                    exact_previous_copies
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
                    - cls.RANDOM_EXPECTATION,
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
                    > cls.RANDOM_EXPECTATION
                ),
            },

            "details": (
                details
            ),
        }

        logger.info(
            "V6 completed | "
            "Predixa=%.4f | "
            "Frequency=%.4f | "
            "Previous=%.4f | "
            "Random=%.4f | "
            "p=%.4f",
            average_hits,
            frequency_average_hits,
            previous_average_hits,
            cls.RANDOM_EXPECTATION,
            monte_carlo[
                "empirical_p_value"
            ],
        )

        return result