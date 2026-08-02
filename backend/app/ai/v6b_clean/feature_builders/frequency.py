from __future__ import annotations

from collections.abc import Sequence

from app.ai.v6b_clean.feature_builders.base import (
    BaseFeatureBuilder,
)
from app.ai.v6b_clean.utils.history import (
    frequency_rate,
)
from app.models.draw import Draw


class FrequencyBuilder(BaseFeatureBuilder):
    """
    PredixaAI V6B frequency feature builder.

    Produces the same four frequency features as V5:

    - rate_10
    - rate_20
    - rate_50
    - rate_100

    Values are rounded to six decimal places to preserve
    compatibility with V5FeatureEngineering.
    """

    WINDOWS = (
        10,
        20,
        50,
        100,
    )

    def build(
        self,
        history: Sequence[Draw],
        candidate_number: int,
    ) -> dict[str, float]:
        """
        Build frequency features for one candidate number.
        """

        self.validate_history(history)
        self.validate_candidate(candidate_number)

        features: dict[str, float] = {}

        for window in self.WINDOWS:
            value = frequency_rate(
                history=history,
                number=candidate_number,
                window=window,
            )

            features[f"rate_{window}"] = round(
                value,
                6,
            )

        return features