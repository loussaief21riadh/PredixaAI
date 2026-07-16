import logging
from pathlib import Path

from app.core.settings import LOGS_DIR

LOGS_DIR.mkdir(exist_ok=True)

LOG_FILE = LOGS_DIR / "predixa.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("PredixaAI")