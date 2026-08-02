from __future__ import annotations

from app.ai.v6b_clean.feature_builders.base import (
    BaseFeatureBuilder,
)
from app.ai.v6b_clean.utils.history import (
    main_numbers,
)


class RecencyBuilder(BaseFeatureBuilder):
    """
    V6B implementation of the V5 recency feature.

    Returns:
        {
            "recency": int
        }
    """

    def build(
        self,
        history,
        candidate_number,
    ) -> dict[str, int]:

        self.validate_history(history)
        self.validate_candidate(candidate_number)

        for distance, draw in enumerate(reversed(history)):
            if candidate_number in main_numbers(draw):
                return {
                    "recency": distance,
                }

        return {
            "recency": len(history),
        }