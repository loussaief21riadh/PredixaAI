import json
from datetime import date, datetime
from pathlib import Path
from typing import Any


class V5CResultStore:
    """
    Predixa AI V5-C benchmark result storage.

    Saves complete benchmark results as JSON so that
    diagnostics can be rerun without retraining models.
    """

    VERSION = "V5-C-RESULT-STORE"

    DEFAULT_DIRECTORY = Path(
        "data/v5c_results"
    )

    @staticmethod
    def _json_default(
        value: Any,
    ) -> Any:
        if isinstance(
            value,
            (date, datetime),
        ):
            return value.isoformat()

        if hasattr(value, "item"):
            return value.item()

        if isinstance(value, set):
            return sorted(value)

        raise TypeError(
            f"Object of type "
            f"{type(value).__name__} "
            f"is not JSON serializable"
        )

    @classmethod
    def save(
        cls,
        result: dict[str, Any],
        filename: str | None = None,
        directory: str | Path | None = None,
    ) -> Path:
        if not isinstance(result, dict):
            raise ValueError(
                "Result must be a dictionary."
            )

        target_directory = Path(
            directory
            if directory is not None
            else cls.DEFAULT_DIRECTORY
        )

        target_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        if filename is None:
            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            version = str(
                result.get(
                    "version",
                    "v5c",
                )
            )

            safe_version = (
                version
                .lower()
                .replace(" ", "_")
                .replace("/", "_")
            )

            filename = (
                f"{safe_version}_{timestamp}.json"
            )

        if not filename.endswith(".json"):
            filename += ".json"

        path = (
            target_directory
            / filename
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent=2,
                default=cls._json_default,
            )

        return path

    @staticmethod
    def load(
        path: str | Path,
    ) -> dict[str, Any]:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Result file not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            result = json.load(file)

        if not isinstance(result, dict):
            raise ValueError(
                "Stored result is not a JSON object."
            )

        return result
