"""
Predixa AI V6B-CLEAN
Relational Feature Builder

This module converts one historical draw sequence and one candidate
number into a dictionary of machine-learning features.

Only historical information may be used.
The caller is responsible for enforcing the validated T-2 protocol.
"""

from __future__ import annotations

from typing import Iterable

from app.ai.v6b_clean.feature_registry import FeatureRegistry
from app.models.draw import Draw


class RelationalFeatureBuilder:
    """
    Builds candidate-level relational features.

    One instance converts:

        history + candidate_number

    into

        {
            feature_name: value,
            ...
        }

    The returned dictionary MUST always match the FeatureRegistry.
    """

    NUMBER_MIN = 1
    NUMBER_MAX = 49

    def build(
        self,
        history: Iterable[Draw],
        candidate_number: int,
    ) -> dict[str, float]:
        """
        Build all registered features.

        Parameters
        ----------
        history
            Chronological draw history ending at T-2.

        candidate_number
            Candidate lottery number.

        Returns
        -------
        dict[str, float]
        """

        if not (
            self.NUMBER_MIN
            <= candidate_number
            <= self.NUMBER_MAX
        ):
            raise ValueError(
                f"Candidate must be between "
                f"{self.NUMBER_MIN} and {self.NUMBER_MAX}."
            )

        history = list(history)

        if len(history) == 0:
            raise ValueError(
                "History cannot be empty."
            )

        features = {}

        # ======================================================
        # Frequency
        # ======================================================

        features.update(
            self._frequency_features(
                history,
                candidate_number,
            )
        )

        # ======================================================
        # Recency
        # ======================================================

        features.update(
            self._recency_features(
                history,
                candidate_number,
            )
        )

        # ======================================================
        # Trend
        # ======================================================

        features.update(
            self._trend_features(
                history,
                candidate_number,
            )
        )

        # ======================================================
        # Draw statistics
        # ======================================================

        features.update(
            self._draw_statistics(
                history,
            )
        )

        FeatureRegistry.validate(
            list(features.keys())
        )

        return features

    # ==========================================================
    # FEATURE FAMILIES
    # ==========================================================

    def _frequency_features(
        self,
        history,
        candidate,
    ) -> dict[str, float]:

        raise NotImplementedError

    def _recency_features(
        self,
        history,
        candidate,
    ) -> dict[str, float]:

        raise NotImplementedError

    def _trend_features(
        self,
        history,
        candidate,
    ) -> dict[str, float]:

        raise NotImplementedError

    def _draw_statistics(
        self,
        history,
    ) -> dict[str, float]:

        raise NotImplementedError