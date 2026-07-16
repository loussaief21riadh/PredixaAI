from pathlib import Path

# ===============================
# PROJECT PATHS
# ===============================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

APP_DIR = BASE_DIR / "app"

DATASETS_DIR = BASE_DIR / "datasets"

MODELS_DIR = BASE_DIR / "trained_models"

REPORTS_DIR = BASE_DIR / "reports"

LOGS_DIR = BASE_DIR / "logs"

DATABASE_PATH = BASE_DIR / "sql_app.db"

# ===============================
# PROJECT INFO
# ===============================

PROJECT_NAME = "Predixa AI"

VERSION = "2.1.0"

AUTHOR = "Loussaief Riadh"

OWNER = "Loussaief Riadh"

LICENSE = "Proprietary"

# ===============================
# MACHINE LEARNING
# ===============================

RANDOM_STATE = 42

TEST_SIZE = 0.20

N_ESTIMATORS = 300

MAX_DEPTH = 12