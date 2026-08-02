from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.models.draw import Draw


class BaseFeatureBuilder(ABC):
    """
    Base class for all PredixaAI feature builders.

    Each builder receives:
        - a historical sequence of Draw objects
        - one candidate number (1-49)

    and returns a dictionary of features.
    """

    NUMBER_MIN = 1
    NUMBER_MAX = 49

    @abstractmethod
    def build(
        self,
        history: Sequence[Draw],
        candidate_number: int,
    ) -> dict[str, float]:
        """
        Build the feature family for one candidate.
        """
        raise NotImplementedError

    @classmethod
    def validate_history(
        cls,
        history: Sequence[Draw],
    ) -> None:
        if len(history) == 0:
            raise ValueError(
                "History cannot be empty."
            )

    @classmethod
    def validate_candidate(
        cls,
        candidate_number: int,
    ) -> None:
        if not (
            cls.NUMBER_MIN
            <= candidate_number
            <= cls.NUMBER_MAX
        ):
            raise ValueError(
                f"Candidate number must be between "
                f"{cls.NUMBER_MIN} and "
                f"{cls.NUMBER_MAX}."
            )