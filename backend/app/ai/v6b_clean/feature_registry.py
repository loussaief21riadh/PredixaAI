"""
Predixa AI V6B-CLEAN
Feature Registry

This module defines every feature exposed to the machine-learning
pipeline.

All downstream components (dataset builder, model, tests) must use
this registry instead of hardcoding feature names.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FeatureRegistry:
    """
    Central registry of model features.

    The order is intentional and must remain stable because the model
    consumes features in exactly this order.
    """

    FEATURES: Tuple[str, ...] = (

        # ======================================================
        # Frequency
        # ======================================================

        "rate_10",
        "rate_20",
        "rate_50",
        "rate_100",

        # ======================================================
        # Recency
        # ======================================================

        "recency",
        "recency_ratio",

        # ======================================================
        # Trend
        # ======================================================

        "short_vs_long",
        "frequency_volatility",

        # ======================================================
        # Draw statistics
        # ======================================================

        "history_size",
        "average_sum",
        "average_even_count",
        "average_consecutive_pairs",
    )

    @classmethod
    def names(cls) -> list[str]:
        """
        Return the ordered feature list.
        """
        return list(cls.FEATURES)

    @classmethod
    def count(cls) -> int:
        """
        Return the number of registered features.
        """
        return len(cls.FEATURES)

    @classmethod
    def contains(cls, feature_name: str) -> bool:
        """
        True if the feature is registered.
        """
        return feature_name in cls.FEATURES

    @classmethod
    def validate(cls, feature_names: list[str]) -> None:
        """
        Validate that a supplied feature list matches the registry
        exactly.

        Raises
        ------
        ValueError
            If features differ by name or order.
        """

        expected = cls.names()

        if feature_names != expected:
            raise ValueError(
                "Feature registry mismatch.\n"
                f"Expected: {expected}\n"
                f"Received: {feature_names}"
            )