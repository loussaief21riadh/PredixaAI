from __future__ import annotations

from pathlib import Path


class V7Config:
    """
    Predixa AI V7 global configuration.

    V7 starts from the validated V6B-CLEAN baseline.
    New experimental features and models are enabled
    incrementally and must always outperform the
    V6B-CLEAN benchmark before adoption.
    """

    VERSION = "V7.0.0-dev"

    PROJECT_NAME = "Predixa AI"

    AI_NAME = "Predixa AI V7"

    RANDOM_STATE = 42

    # ==================================================
    # DRAWS
    # ==================================================

    NUMBER_MIN = 1

    NUMBER_MAX = 49

    MAIN_NUMBERS = 5

    # ==================================================
    # FEATURE ENGINEERING
    # ==================================================

    HISTORY_SIZE = 100

    FEATURE_COUNT_BASELINE = 396

    ENABLE_PAIR_FEATURES = False

    ENABLE_TRIPLE_FEATURES = False

    ENABLE_NEIGHBOR_FEATURES = False

    ENABLE_GAP_FEATURES = False

    ENABLE_TREND_FEATURES = False

    ENABLE_MOMENTUM_FEATURES = False

    ENABLE_VOLATILITY_FEATURES = False

    ENABLE_CYCLE_FEATURES = False

    ENABLE_EXPERIMENTAL_FEATURES = False

    # ==================================================
    # MODEL
    # ==================================================

    MODEL_NAME = "RandomForest"

    N_ESTIMATORS = 300

    MAX_DEPTH = 12

    TEST_SIZE = 0.20

    TOP_K = 5

    # ==================================================
    # WALK-FORWARD
    # ==================================================

    LAG_DRAWS = 1

    PURGE_DRAWS = 1

    DEFAULT_TEST_DRAWS = 20

    DEFAULT_MAX_TRAINING_TARGETS = 1500

    DEFAULT_MONTE_CARLO_SIMULATIONS = 1000

    # ==================================================
    # MULTI WINDOW
    # ==================================================

    DEFAULT_NUMBER_OF_WINDOWS = 3

    DEFAULT_TEST_DRAWS_PER_WINDOW = 20

    # ==================================================
    # PATHS
    # ==================================================

    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

    APP_DIR = BASE_DIR / "app"

    DATASETS_DIR = BASE_DIR / "datasets"

    MODELS_DIR = BASE_DIR / "trained_models"

    REPORTS_DIR = BASE_DIR / "reports"

    LOGS_DIR = BASE_DIR / "logs"

    # ==================================================
    # BENCHMARK
    # ==================================================

    BASELINE_VERSION = "V6B-CLEAN"

    REQUIRED_BENCHMARK = "WalkForward + MultiWindow"

    REQUIRE_IMPROVEMENT = True

    MIN_REQUIRED_HIT_IMPROVEMENT = 0.0

    # ==================================================
    # DEBUG
    # ==================================================

    VERBOSE = True

    SAVE_REPORTS = True

    SAVE_MODELS = True

    SAVE_PREDICTIONS = True

    # ==================================================
    # HELPERS
    # ==================================================

    @classmethod
    def feature_flags(cls) -> dict[str, bool]:
        return {
            "pair_features": cls.ENABLE_PAIR_FEATURES,
            "triple_features": cls.ENABLE_TRIPLE_FEATURES,
            "neighbor_features": cls.ENABLE_NEIGHBOR_FEATURES,
            "gap_features": cls.ENABLE_GAP_FEATURES,
            "trend_features": cls.ENABLE_TREND_FEATURES,
            "momentum_features": cls.ENABLE_MOMENTUM_FEATURES,
            "volatility_features": cls.ENABLE_VOLATILITY_FEATURES,
            "cycle_features": cls.ENABLE_CYCLE_FEATURES,
            "experimental_features": cls.ENABLE_EXPERIMENTAL_FEATURES,
        }

    @classmethod
    def model_parameters(cls) -> dict[str, int]:
        return {
            "random_state": cls.RANDOM_STATE,
            "n_estimators": cls.N_ESTIMATORS,
            "max_depth": cls.MAX_DEPTH,
        }