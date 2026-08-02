"""
Predixa AI V6B-CLEAN
Configuration module.

This module centralizes all configuration used by the V6B-CLEAN
research architecture.

Design goals
------------
- One source of truth for protocol parameters.
- Same temporal protocol as validated V6.
- Easy experimentation without changing benchmark logic.
"""

from dataclasses import dataclass

from app.core.settings import (
    RANDOM_STATE,
    N_ESTIMATORS,
    MAX_DEPTH,
)


@dataclass(frozen=True)
class ProtocolConfig:
    """
    Parameters that define the benchmark protocol.

    These values should remain identical to V6 so that
    benchmark comparisons remain scientifically valid.
    """

    VERSION: str = "V6B-CLEAN"

    MODERN_LOTO_START_DATE: str = "2008-10-06"

    NUMBER_MIN: int = 1
    NUMBER_MAX: int = 49

    TOP_K: int = 5

    LAG_DRAWS: int = 1
    PURGE_DRAWS: int = 1


@dataclass(frozen=True)
class BenchmarkConfig:
    """
    Default benchmark settings.
    """

    WINDOW_SIZE: int = 100

    TEST_DRAWS: int = 20

    NUMBER_OF_WINDOWS: int = 3

    MAX_TRAINING_TARGETS: int = 1500

    MONTE_CARLO_SIMULATIONS: int = 1000


@dataclass(frozen=True)
class ModelConfig:
    """
    Machine-learning configuration.

    Hyperparameters are imported from the global
    application settings to remain consistent with V6.
    """

    RANDOM_STATE: int = RANDOM_STATE

    N_ESTIMATORS: int = N_ESTIMATORS

    MAX_DEPTH: int | None = MAX_DEPTH

    CLASS_WEIGHT: str = "balanced"

    N_JOBS: int = -1


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Research switches.

    These flags allow future experiments without modifying
    the benchmark protocol.
    """

    MODEL_NAME: str = "RandomForest"

    ENABLE_RELATIONAL_FEATURES: bool = True

    ENABLE_RECENCY_FEATURES: bool = True

    ENABLE_FREQUENCY_FEATURES: bool = True

    ENABLE_VOLATILITY_FEATURES: bool = True

    ENABLE_COOCCURRENCE_FEATURES: bool = True


PROTOCOL = ProtocolConfig()

BENCHMARK = BenchmarkConfig()

MODEL = ModelConfig()

EXPERIMENT = ExperimentConfig()