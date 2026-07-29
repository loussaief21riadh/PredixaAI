from typing import Any

import numpy as np


class V6LocalSignalDiagnostics:
    """
    Predixa AI V6 - Local Ranking Signal Diagnostics.

    Works only from saved walk-forward results.

    No model training is performed.

    Purpose:
        Determine whether useful information is concentrated
        in particular regions of the 49-number ranking.

    Diagnostics:
        - Hits by rank bands
        - Winner rank distribution
        - Recall@K
        - Top-K enrichment versus random
        - Mean rank of actual winning numbers
        - Score distribution for winners vs non-winners
        - Score gap between winners and non-winners
        - Top-5 score separation
    """

    VERSION = "V6-LOCAL-SIGNAL-DIAGNOSTICS"

    NUMBER_COUNT = 49
    WINNING_NUMBERS = 5

    RANK_BANDS = (
        (1, 5),
        (6, 10),
        (11, 20),
        (21, 30),
        (31, 40),
        (41, 49),
    )

    RECALL_K_VALUES = (
        1,
        3,
        5,
        10,
        20,
        30,
        40,
        49,
    )

    # ==========================================================
    # VALIDATION
    # ==========================================================

    @classmethod
    def _validate(
        cls,
        result: dict[str, Any],
    ) -> list[dict[str, Any]]:

        if not isinstance(result, dict):
            raise ValueError(
                "Result must be a dictionary."
            )

        details = result.get("details")

        if not isinstance(details, list):
            raise ValueError(
                "Result does not contain a valid details list."
            )

        if not details:
            raise ValueError(
                "Result details are empty."
            )

        for index, detail in enumerate(
            details,
            start=1,
        ):
            probabilities = detail.get(
                "probabilities"
            )

            actual = detail.get(
                "actual_numbers"
            )

            if not isinstance(
                probabilities,
                dict,
            ):
                raise ValueError(
                    f"Detail {index} has no probabilities."
                )

            if len(probabilities) != cls.NUMBER_COUNT:
                raise ValueError(
                    f"Detail {index}: expected "
                    f"{cls.NUMBER_COUNT} probabilities."
                )

            if not isinstance(
                actual,
                list,
            ):
                raise ValueError(
                    f"Detail {index} has no actual_numbers."
                )

            if len(actual) != cls.WINNING_NUMBERS:
                raise ValueError(
                    f"Detail {index}: expected "
                    f"{cls.WINNING_NUMBERS} actual numbers."
                )

            if len(set(actual)) != cls.WINNING_NUMBERS:
                raise ValueError(
                    f"Detail {index} contains duplicate "
                    "actual numbers."
                )

        return details

    # ==========================================================
    # SCORE
    # ==========================================================

    @staticmethod
    def _score(
        probabilities: dict,
        number: int,
    ) -> float:

        value = probabilities.get(
            number
        )

        if value is None:
            value = probabilities.get(
                str(number)
            )

        if value is None:
            raise ValueError(
                f"Missing score for number {number}."
            )

        value = float(
            value
        )

        if not np.isfinite(
            value
        ):
            raise ValueError(
                f"Non-finite score for number {number}."
            )

        return value

    # ==========================================================
    # BUILD RANKING
    # ==========================================================

    @classmethod
    def _ranking(
        cls,
        detail: dict[str, Any],
    ) -> list[dict[str, float | int]]:

        probabilities = detail[
            "probabilities"
        ]

        ranking = [
            {
                "number": number,
                "score": cls._score(
                    probabilities,
                    number,
                ),
            }
            for number in range(
                1,
                cls.NUMBER_COUNT + 1,
            )
        ]

        ranking.sort(
            key=lambda row: (
                -row["score"],
                row["number"],
            )
        )

        return ranking

    # ==========================================================
    # MAIN ANALYSIS
    # ==========================================================

    @classmethod
    def analyze(
        cls,
        result: dict[str, Any],
    ) -> dict[str, Any]:

        details = cls._validate(
            result
        )

        evaluated_draws = len(
            details
        )

        total_actual_numbers = (
            evaluated_draws
            * cls.WINNING_NUMBERS
        )

        band_hits = {
            f"{start}-{end}": 0
            for start, end
            in cls.RANK_BANDS
        }

        recall_hits = {
            k: 0
            for k in cls.RECALL_K_VALUES
        }

        all_winner_ranks = []

        winner_scores = []
        loser_scores = []

        top5_scores = []
        rest_scores = []

        per_draw_mean_winner_rank = []
        per_draw_best_winner_rank = []
        per_draw_top5_hits = []

        for detail in details:

            actual = set(
                int(number)
                for number in detail[
                    "actual_numbers"
                ]
            )

            ranking = cls._ranking(
                detail
            )

            rank_lookup = {
                int(row["number"]): (
                    index + 1
                )
                for index, row
                in enumerate(
                    ranking
                )
            }

            draw_winner_ranks = sorted(
                rank_lookup[number]
                for number in actual
            )

            all_winner_ranks.extend(
                draw_winner_ranks
            )

            per_draw_mean_winner_rank.append(
                float(
                    np.mean(
                        draw_winner_ranks
                    )
                )
            )

            per_draw_best_winner_rank.append(
                min(
                    draw_winner_ranks
                )
            )

            top5_numbers = {
                int(row["number"])
                for row in ranking[:5]
            }

            top5_hits = len(
                top5_numbers
                & actual
            )

            per_draw_top5_hits.append(
                top5_hits
            )

            # ----------------------------------------------
            # Rank bands
            # ----------------------------------------------

            for start, end in cls.RANK_BANDS:

                band_name = (
                    f"{start}-{end}"
                )

                band_numbers = {
                    int(row["number"])
                    for row in ranking[
                        start - 1:
                        end
                    ]
                }

                band_hits[
                    band_name
                ] += len(
                    band_numbers
                    & actual
                )

            # ----------------------------------------------
            # Recall @ K
            # ----------------------------------------------

            for k in cls.RECALL_K_VALUES:

                top_k_numbers = {
                    int(row["number"])
                    for row in ranking[:k]
                }

                recall_hits[
                    k
                ] += len(
                    top_k_numbers
                    & actual
                )

            # ----------------------------------------------
            # Score groups
            # ----------------------------------------------

            for rank_index, row in enumerate(
                ranking,
                start=1,
            ):
                number = int(
                    row["number"]
                )

                score = float(
                    row["score"]
                )

                if number in actual:
                    winner_scores.append(
                        score
                    )
                else:
                    loser_scores.append(
                        score
                    )

                if rank_index <= 5:
                    top5_scores.append(
                        score
                    )
                else:
                    rest_scores.append(
                        score
                    )

        # ======================================================
        # RANK BAND SUMMARY
        # ======================================================

        band_summary = {}

        for start, end in cls.RANK_BANDS:

            name = (
                f"{start}-{end}"
            )

            band_size = (
                end
                - start
                + 1
            )

            observed_hits = (
                band_hits[
                    name
                ]
            )

            observed_share = (
                observed_hits
                / total_actual_numbers
            )

            random_expected_share = (
                band_size
                / cls.NUMBER_COUNT
            )

            expected_random_hits = (
                total_actual_numbers
                * random_expected_share
            )

            enrichment = (
                observed_share
                / random_expected_share
                if random_expected_share > 0
                else 0.0
            )

            band_summary[
                name
            ] = {
                "rank_start": start,

                "rank_end": end,

                "band_size": (
                    band_size
                ),

                "observed_hits": (
                    observed_hits
                ),

                "observed_share_of_winners": round(
                    observed_share,
                    8,
                ),

                "random_expected_share": round(
                    random_expected_share,
                    8,
                ),

                "random_expected_hits": round(
                    expected_random_hits,
                    8,
                ),

                "enrichment_ratio": round(
                    enrichment,
                    8,
                ),
            }

        # ======================================================
        # RECALL SUMMARY
        # ======================================================

        recall_summary = {}

        for k in cls.RECALL_K_VALUES:

            hits = (
                recall_hits[
                    k
                ]
            )

            recall = (
                hits
                / total_actual_numbers
            )

            random_recall = (
                k
                / cls.NUMBER_COUNT
            )

            recall_summary[
                f"top_{k}"
            ] = {
                "winner_hits": (
                    hits
                ),

                "recall": round(
                    recall,
                    8,
                ),

                "random_expected_recall": round(
                    random_recall,
                    8,
                ),

                "lift": round(
                    recall
                    - random_recall,
                    8,
                ),

                "enrichment_ratio": round(
                    (
                        recall
                        / random_recall
                    )
                    if random_recall > 0
                    else 0.0,
                    8,
                ),
            }

        # ======================================================
        # SCORE SUMMARY
        # ======================================================

        winner_scores_array = np.asarray(
            winner_scores,
            dtype=float,
        )

        loser_scores_array = np.asarray(
            loser_scores,
            dtype=float,
        )

        top5_scores_array = np.asarray(
            top5_scores,
            dtype=float,
        )

        rest_scores_array = np.asarray(
            rest_scores,
            dtype=float,
        )

        mean_winner_score = float(
            np.mean(
                winner_scores_array
            )
        )

        mean_loser_score = float(
            np.mean(
                loser_scores_array
            )
        )

        score_summary = {
            "winner_count": len(
                winner_scores
            ),

            "loser_count": len(
                loser_scores
            ),

            "mean_winner_score": round(
                mean_winner_score,
                8,
            ),

            "mean_loser_score": round(
                mean_loser_score,
                8,
            ),

            "winner_minus_loser": round(
                mean_winner_score
                - mean_loser_score,
                8,
            ),

            "median_winner_score": round(
                float(
                    np.median(
                        winner_scores_array
                    )
                ),
                8,
            ),

            "median_loser_score": round(
                float(
                    np.median(
                        loser_scores_array
                    )
                ),
                8,
            ),

            "mean_top5_score": round(
                float(
                    np.mean(
                        top5_scores_array
                    )
                ),
                8,
            ),

            "mean_rest_score": round(
                float(
                    np.mean(
                        rest_scores_array
                    )
                ),
                8,
            ),

            "top5_minus_rest_score": round(
                float(
                    np.mean(
                        top5_scores_array
                    )
                    - np.mean(
                        rest_scores_array
                    )
                ),
                8,
            ),
        }

        # ======================================================
        # WINNER RANK SUMMARY
        # ======================================================

        winner_rank_array = np.asarray(
            all_winner_ranks,
            dtype=float,
        )

        random_expected_mean_rank = (
            cls.NUMBER_COUNT
            + 1
        ) / 2

        winner_rank_summary = {
            "winner_count": len(
                all_winner_ranks
            ),

            "mean_winner_rank": round(
                float(
                    np.mean(
                        winner_rank_array
                    )
                ),
                8,
            ),

            "median_winner_rank": round(
                float(
                    np.median(
                        winner_rank_array
                    )
                ),
                8,
            ),

            "random_expected_mean_rank": round(
                random_expected_mean_rank,
                8,
            ),

            "mean_rank_improvement": round(
                random_expected_mean_rank
                - float(
                    np.mean(
                        winner_rank_array
                    )
                ),
                8,
            ),

            "mean_best_winner_rank_per_draw": round(
                float(
                    np.mean(
                        per_draw_best_winner_rank
                    )
                ),
                8,
            ),

            "mean_winner_rank_per_draw": round(
                float(
                    np.mean(
                        per_draw_mean_winner_rank
                    )
                ),
                8,
            ),
        }

        # ======================================================
        # TOP-5 SUMMARY
        # ======================================================

        top5_hits_array = np.asarray(
            per_draw_top5_hits,
            dtype=float,
        )

        top5_summary = {
            "total_hits": int(
                np.sum(
                    top5_hits_array
                )
            ),

            "average_hits_at_5": round(
                float(
                    np.mean(
                        top5_hits_array
                    )
                ),
                8,
            ),

            "draws_with_0_hits": int(
                np.sum(
                    top5_hits_array == 0
                )
            ),

            "draws_with_1_hit": int(
                np.sum(
                    top5_hits_array == 1
                )
            ),

            "draws_with_2_plus_hits": int(
                np.sum(
                    top5_hits_array >= 2
                )
            ),

            "at_least_1_hit_rate": round(
                float(
                    np.mean(
                        top5_hits_array >= 1
                    )
                ),
                8,
            ),

            "at_least_2_hit_rate": round(
                float(
                    np.mean(
                        top5_hits_array >= 2
                    )
                ),
                8,
            ),
        }

        return {
            "status": "success",

            "version": (
                cls.VERSION
            ),

            "evaluated_draws": (
                evaluated_draws
            ),

            "total_actual_winning_numbers": (
                total_actual_numbers
            ),

            "rank_bands": (
                band_summary
            ),

            "recall_at_k": (
                recall_summary
            ),

            "winner_rank_summary": (
                winner_rank_summary
            ),

            "score_summary": (
                score_summary
            ),

            "top5_summary": (
                top5_summary
            ),
        }
