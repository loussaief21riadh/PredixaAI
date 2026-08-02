from __future__ import annotations

from pathlib import Path

# ==========================================================
# PREDIXA AI V7 CONSTANTS
# ==========================================================

PROJECT_NAME = "Predixa AI"

AI_NAME = "Predixa AI V7"

VERSION = "7.0.0-dev"

BASELINE_VERSION = "V6B-CLEAN"

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

APP_DIR = BASE_DIR / "app"

DATASETS_DIR = BASE_DIR / "datasets"

MODELS_DIR = BASE_DIR / "trained_models"

REPORTS_DIR = BASE_DIR / "reports"

LOGS_DIR = BASE_DIR / "logs"

DATABASE_PATH = BASE_DIR / "sql_app.db"

# ==========================================================
# LOTTO
# ==========================================================

NUMBER_MIN = 1

NUMBER_MAX = 49

MAIN_NUMBERS = 5

TOP_K = 5

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

FEATURE_COUNT_BASELINE = 396

DEFAULT_HISTORY_SIZE = 100

LAG_DRAWS = 1

PURGE_DRAWS = 1

# ==========================================================
# MACHINE LEARNING
# ==========================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

N_ESTIMATORS = 300

MAX_DEPTH = 12

# ==========================================================
# WALK FORWARD
# ==========================================================

DEFAULT_TEST_DRAWS = 20

DEFAULT_MAX_TRAINING_TARGETS = 1500

DEFAULT_MONTE_CARLO_SIMULATIONS = 1000

# ==========================================================
# MULTI WINDOW
# ==========================================================

DEFAULT_NUMBER_OF_WINDOWS = 3

DEFAULT_TEST_DRAWS_PER_WINDOW = 20

# ==========================================================
# BASELINES
# ==========================================================

RANDOM_EXPECTATION = 25 / 49

# ==========================================================
# FEATURE FLAGS
# ==========================================================

ENABLE_PAIR_FEATURES = False

ENABLE_TRIPLE_FEATURES = False

ENABLE_NEIGHBOR_FEATURES = False

ENABLE_GAP_FEATURES = False

ENABLE_TREND_FEATURES = False

ENABLE_MOMENTUM_FEATURES = False

ENABLE_VOLATILITY_FEATURES = False

ENABLE_CYCLE_FEATURES = False

ENABLE_EXPERIMENTAL_FEATURES = False

# ==========================================================
# REPORTS
# ==========================================================

SAVE_MODELS = True

SAVE_REPORTS = True

SAVE_PREDICTIONS = True

VERBOSE = True