import pandas as pd

from app.ai.v5.feature_engineering_v5b import V5BFeatureEngineering
from app.ai.v5.walk_forward_backtester import V5WalkForwardBacktester


class V5BWalkForwardBacktester(V5WalkForwardBacktester):
    """
    Predixa AI V5-B Ablation Walk-Forward.

    Same temporal protocol as V5-A:
        - target = T
        - prediction features end at T-2
        - T-1 excluded from prediction features
        - T-1 target purged from training
        - strict chronological walk-forward

    The only experimental variable is the feature family
    selected through `variant`.
    """

    EXPECTED_FEATURE_COUNTS = {
        "full": 396,
        "no_recency": 347,
        "no_recency_ratio": 347,
        "no_short_vs_long": 347,
        "no_frequency_volatility": 347,
        "rates_only": 200,
    }

    @staticmethod
    def _build_training_dataset_v5b(
        draws,
        window_size: int,
        max_training_samples: int,
        variant: str,
    ):
        if variant not in V5BWalkForwardBacktester.EXPECTED_FEATURE_COUNTS:
            raise ValueError(
                f"Unknown V5-B variant: {variant}"
            )

        minimum_required = (
            window_size
            + V5BWalkForwardBacktester.LAG_DRAWS
            + 1
        )

        if len(draws) < minimum_required:
            raise ValueError(
                "Not enough historical draws to build "
                "the V5-B training dataset."
            )

        first_target_index = (
            window_size
            + V5BWalkForwardBacktester.LAG_DRAWS
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
                - V5BWalkForwardBacktester.LAG_DRAWS
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

            target_draw = draws[target_index]

            features = (
                V5BFeatureEngineering
                .build_from_history(
                    history,
                    window_size=window_size,
                    variant=variant,
                )
            )

            actual_numbers = set(
                V5BWalkForwardBacktester
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
                for number in range(1, 50)
            }

            feature_rows.append(features)
            target_rows.append(targets)

        X = pd.DataFrame(feature_rows)
        y = pd.DataFrame(target_rows)

        if X.empty or y.empty:
            raise ValueError(
                "V5-B walk-forward training dataset is empty."
            )

        if len(X) != len(y):
            raise ValueError(
                "V5-B X and y sizes do not match."
            )

        if X.isnull().any().any():
            raise ValueError(
                "V5-B training features contain missing values."
            )

        if y.isnull().any().any():
            raise ValueError(
                "V5-B training targets contain missing values."
            )

        expected = (
            V5BWalkForwardBacktester
            .EXPECTED_FEATURE_COUNTS[variant]
        )

        if X.shape[1] != expected:
            raise ValueError(
                f"Unexpected feature count for {variant}. "
                f"Expected {expected}, received {X.shape[1]}."
            )

        return X, y

    @staticmethod
    def run(
        db,
        variant: str = "full",
        test_draws: int = 5,
        window_size: int = 100,
        max_training_samples: int = 1500,
        monte_carlo_simulations: int = 10000,
    ):
        """
        Run V5-B by temporarily substituting the V5-A feature
        builders while preserving the complete V5-A evaluation
        protocol.
        """

        if variant not in V5BWalkForwardBacktester.EXPECTED_FEATURE_COUNTS:
            raise ValueError(
                f"Unknown V5-B variant: {variant}. "
                f"Valid variants: "
                f"{sorted(V5BWalkForwardBacktester.EXPECTED_FEATURE_COUNTS)}"
            )

        original_training_builder = (
            V5WalkForwardBacktester
            ._build_training_dataset
        )

        # Import the V5-A module itself so its global
        # V5FeatureEngineering reference can be substituted.
        import app.ai.v5.walk_forward_backtester as v5a_module

        original_feature_engineering = (
            v5a_module.V5FeatureEngineering
        )

        class VariantFeatureEngineering:
            @staticmethod
            def build_from_history(
                draws,
                window_size=100,
            ):
                return (
                    V5BFeatureEngineering
                    .build_from_history(
                        draws,
                        window_size=window_size,
                        variant=variant,
                    )
                )

        def variant_training_builder(
            draws,
            window_size,
            max_training_samples,
        ):
            return (
                V5BWalkForwardBacktester
                ._build_training_dataset_v5b(
                    draws=draws,
                    window_size=window_size,
                    max_training_samples=max_training_samples,
                    variant=variant,
                )
            )

        try:
            V5WalkForwardBacktester._build_training_dataset = (
                staticmethod(
                    variant_training_builder
                )
            )

            v5a_module.V5FeatureEngineering = (
                VariantFeatureEngineering
            )

            result = V5WalkForwardBacktester.run(
                db=db,
                test_draws=test_draws,
                window_size=window_size,
                max_training_samples=max_training_samples,
                monte_carlo_simulations=monte_carlo_simulations,
            )

        finally:
            V5WalkForwardBacktester._build_training_dataset = (
                staticmethod(
                    original_training_builder
                )
            )

            v5a_module.V5FeatureEngineering = (
                original_feature_engineering
            )

        expected_feature_count = (
            V5BWalkForwardBacktester
            .EXPECTED_FEATURE_COUNTS[variant]
        )

        # Correct V5-A metadata inherited from the parent
        # benchmark.
        result["version"] = (
            f"V5-B-{variant.upper()}-PURGED-T2"
        )

        result["variant"] = variant

        result["feature_count"] = (
            expected_feature_count
        )

        for detail in result.get(
            "details",
            []
        ):
            detail["feature_count"] = (
                expected_feature_count
            )

        return result
