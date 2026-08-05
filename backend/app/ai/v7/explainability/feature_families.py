from __future__ import annotations

from types import MappingProxyType
from typing import Final


FEATURE_FAMILIES: Final[
    MappingProxyType[str, tuple[str, ...]]
] = MappingProxyType(
    {
        "global": (
            "history_size",
            "average_sum",
            "average_even_count",
            "average_consecutive_pairs",
        ),
        "frequency": (
            "rate_10",
            "rate_20",
            "rate_50",
            "rate_100",
        ),
        "recency": (
            "recency",
            "recency_ratio",
        ),
        "trend": (
            "short_vs_long",
        ),
        "volatility": (
            "frequency_volatility",
        ),
    }
)

# Sprint 5 production model contract:
# rate_10 remains engineered but is excluded from active
# explainability families together with MODEL_FEATURES.
FEATURE_FAMILIES = {
    family_name: tuple(
        feature_name
        for feature_name in family_features
        if feature_name != "rate_10"
    )
    for family_name, family_features
    in FEATURE_FAMILIES.items()
}


FEATURE_FAMILY_ORDER: Final[
    tuple[str, ...]
] = (
    "global",
    "frequency",
    "recency",
    "trend",
    "volatility",
)


def all_feature_names() -> tuple[str, ...]:
    """
    Return all configured feature names in family order.
    """

    return tuple(
        feature_name
        for family_name in FEATURE_FAMILY_ORDER
        for feature_name in FEATURE_FAMILIES[
            family_name
        ]
    )


def family_for_feature(
    feature_name: str,
) -> str:
    """
    Return the configured family for one feature.

    Raises
    ------
    ValueError
        If the feature is unknown.
    """

    if not isinstance(
        feature_name,
        str,
    ):
        raise ValueError(
            "feature_name must be a string."
        )

    normalized_feature_name = (
        feature_name.strip()
    )

    if not normalized_feature_name:
        raise ValueError(
            "feature_name cannot be empty."
        )

    for family_name in (
        FEATURE_FAMILY_ORDER
    ):
        if (
            normalized_feature_name
            in FEATURE_FAMILIES[
                family_name
            ]
        ):
            return family_name

    raise ValueError(
        "Unknown feature name: "
        f"{normalized_feature_name}"
    )


def validate_feature_family_configuration(
) -> None:
    """
    Validate the complete feature-family configuration.
    """

    configured_families = set(
        FEATURE_FAMILIES
    )

    ordered_families = set(
        FEATURE_FAMILY_ORDER
    )

    if (
        configured_families
        != ordered_families
    ):
        raise ValueError(
            "FEATURE_FAMILY_ORDER must contain "
            "exactly the configured feature families."
        )

    feature_names = all_feature_names()

    if not feature_names:
        raise ValueError(
            "Feature-family configuration "
            "cannot be empty."
        )

    if (
        len(
            feature_names
        )
        != len(
            set(
                feature_names
            )
        )
    ):
        raise ValueError(
            "Feature names must be unique "
            "across all families."
        )

    for family_name in (
        FEATURE_FAMILY_ORDER
    ):
        features = FEATURE_FAMILIES[
            family_name
        ]

        if not features:
            raise ValueError(
                "Feature family cannot be empty: "
                f"{family_name}"
            )

        if any(
            not isinstance(
                feature_name,
                str,
            )
            or not feature_name.strip()
            for feature_name in features
        ):
            raise ValueError(
                "Feature family contains an invalid "
                f"feature name: {family_name}"
            )


validate_feature_family_configuration()