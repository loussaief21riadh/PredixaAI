from app.ai.v5.feature_engineering import V5FeatureEngineering
from app.models.draw import Draw


class V5BFeatureEngineering:
    """
    Predixa AI V5-B Ablation Feature Engineering.

    V5-A remains untouched.

    V5-B generates the complete V5-A feature set first,
    then removes selected feature families.

    Supported variants:

        full
        no_recency
        no_recency_ratio
        no_short_vs_long
        no_frequency_volatility
        rates_only

    Global features are always preserved.
    """

    VALID_VARIANTS = {
        "full",
        "no_recency",
        "no_recency_ratio",
        "no_short_vs_long",
        "no_frequency_volatility",
        "rates_only",
    }

    GLOBAL_FEATURES = {
        "history_size",
        "average_sum",
        "average_even_count",
        "average_consecutive_pairs",
    }

    @staticmethod
    def build_from_history(
        draws: list[Draw],
        window_size: int = 100,
        variant: str = "full",
    ) -> dict[str, int | float]:

        if variant not in V5BFeatureEngineering.VALID_VARIANTS:
            raise ValueError(
                f"Unknown V5-B variant: {variant}. "
                f"Valid variants: "
                f"{sorted(V5BFeatureEngineering.VALID_VARIANTS)}"
            )

        features = (
            V5FeatureEngineering
            .build_from_history(
                draws,
                window_size=window_size,
            )
        )

        # Full V5-A control.
        if variant == "full":
            return features

        filtered = {}

        for name, value in features.items():

            # Always retain global structural features.
            if name in V5BFeatureEngineering.GLOBAL_FEATURES:
                filtered[name] = value
                continue

            if variant == "no_recency":

                # Remove raw recency only.
                # recency_ratio remains.
                if name.startswith("recency_") and not name.startswith(
                    "recency_ratio_"
                ):
                    continue

            elif variant == "no_recency_ratio":

                if name.startswith("recency_ratio_"):
                    continue

            elif variant == "no_short_vs_long":

                if name.startswith("short_vs_long_"):
                    continue

            elif variant == "no_frequency_volatility":

                if name.startswith("frequency_volatility_"):
                    continue

            elif variant == "rates_only":

                # Keep only:
                # rate_10
                # rate_20
                # rate_50
                # rate_100
                # + global features

                if not name.startswith("rate_"):
                    continue

            filtered[name] = value

        return filtered
