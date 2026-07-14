import csv
from pathlib import Path


class CSVImporter:
    """
    CSV Import Engine for Predixa AI
    """

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def detect_delimiter(self):
        """
        Detect automatically if CSV uses ';' or ','
        """

        with open(self.filepath, "r", encoding="utf-8") as f:
            sample = f.read(2048)

        if sample.count(";") > sample.count(","):
            return ";"

        return ","

    def load(self):
        """
        Load CSV file
        """

        delimiter = self.detect_delimiter()

        with open(
            self.filepath,
            newline="",
            encoding="utf-8",
        ) as csvfile:

            reader = csv.DictReader(csvfile, delimiter=delimiter)

            return list(reader)