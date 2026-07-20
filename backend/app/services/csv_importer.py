import csv
from pathlib import Path


class CSVImporter:
    """
    CSV Import Engine for Predixa AI.

    Features:
    - File existence validation
    - Automatic delimiter detection
    - UTF-8 BOM support
    - Empty row filtering
    - Header normalization
    """

    SUPPORTED_DELIMITERS = [",", ";"]

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)

    def validate_file(self) -> None:
        """
        Validate that the file exists and is a CSV file.
        """

        if not self.filepath.exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.filepath}"
            )

        if not self.filepath.is_file():
            raise ValueError(
                "The provided path is not a file."
            )

        if self.filepath.suffix.lower() != ".csv":
            raise ValueError(
                "Only CSV files are supported."
            )

    def detect_delimiter(self) -> str:
        """
        Automatically detect whether CSV uses ',' or ';'.
        """

        self.validate_file()

        with self.filepath.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
        ) as file:
            sample = file.read(4096)

        if not sample.strip():
            raise ValueError(
                "The CSV file is empty."
            )

        try:
            dialect = csv.Sniffer().sniff(
                sample,
                delimiters="".join(
                    self.SUPPORTED_DELIMITERS
                ),
            )
            return dialect.delimiter

        except csv.Error:
            if sample.count(";") > sample.count(","):
                return ";"

            return ","

    @staticmethod
    def normalize_row(row: dict) -> dict:
        """
        Normalize column names and values.
        """

        normalized = {}

        for key, value in row.items():

            if key is None:
                continue

            normalized_key = (
                key.strip()
                .lower()
                .replace(" ", "_")
            )

            if isinstance(value, str):
                value = value.strip()

            normalized[normalized_key] = value

        return normalized

    def load(self) -> list[dict]:
        """
        Load and normalize CSV rows.
        """

        self.validate_file()

        delimiter = self.detect_delimiter()

        with self.filepath.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as file:

            reader = csv.DictReader(
                file,
                delimiter=delimiter,
            )

            if reader.fieldnames is None:
                raise ValueError(
                    "CSV header could not be detected."
                )

            rows = []

            for row in reader:

                if not any(
                    value and str(value).strip()
                    for value in row.values()
                ):
                    continue

                rows.append(
                    self.normalize_row(row)
                )

        if not rows:
            raise ValueError(
                "The CSV file contains no data rows."
            )

        return rows

    def exists(self) -> bool:
        """
        Check if the CSV file exists.
        """

        return self.filepath.exists()

    def count_rows(self) -> int:
        """
        Return number of valid data rows.
        """

        return len(self.load())