import csv
from pathlib import Path


class CSVImporter:
    """
    CSV Import Engine for Predixa AI
    Supports automatic delimiter detection.
    """

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)

    def detect_delimiter(self) -> str:
        """
        Detect whether the CSV uses ';' or ','.
        """

        with self.filepath.open(
            "r",
            encoding="utf-8",
        ) as file:

            sample = file.read(4096)

        if sample.count(";") > sample.count(","):
            return ";"

        return ","

    def load(self) -> list[dict]:
        """
        Load the CSV into a list of dictionaries.
        """

        delimiter = self.detect_delimiter()

        with self.filepath.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:

            reader = csv.DictReader(
                file,
                delimiter=delimiter,
            )

            rows = list(reader)

        return rows

    def exists(self) -> bool:
        """
        Check if the CSV file exists.
        """

        return self.filepath.exists()

    def count_rows(self) -> int:
        """
        Return the number of data rows.
        """

        return len(self.load())