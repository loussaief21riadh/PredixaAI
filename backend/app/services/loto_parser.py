from datetime import datetime


class LotoParser:
    """
    Convert an FDJ CSV row into Predixa AI draw format.

    Supports:
    - Legacy Loto format
    - Modern Loto format
    - YYYYMMDD dates
    - DD/MM/YYYY dates
    """

    DATE_FORMATS = (
        "%Y%m%d",
        "%d/%m/%Y",
    )

    @staticmethod
    def _parse_date(value: str):
        """
        Parse FDJ draw dates using supported formats.
        """

        value = str(value).strip()

        for date_format in LotoParser.DATE_FORMATS:
            try:
                return datetime.strptime(
                    value,
                    date_format,
                ).date()

            except ValueError:
                continue

        raise ValueError(
            f"Unsupported draw date format: {value}"
        )

    @staticmethod
    def _optional_int(
        row: dict,
        key: str,
    ):
        """
        Return an integer when the CSV field exists
        and contains a value, otherwise None.
        """

        value = row.get(key)

        if value is None:
            return None

        value = str(value).strip()

        if not value:
            return None

        return int(value)

    @staticmethod
    def parse(row: dict):
        """
        Convert one normalized CSV row into a Draw-compatible dict.
        """

        draw_date_raw = row.get(
            "date_de_tirage"
        )

        if not draw_date_raw:
            raise ValueError(
                "Missing date_de_tirage"
            )

        numbers = []

        for key in (
            "boule_1",
            "boule_2",
            "boule_3",
            "boule_4",
            "boule_5",
        ):
            value = row.get(key)

            if value is None or not str(value).strip():
                raise ValueError(
                    f"Missing required field: {key}"
                )

            numbers.append(
                int(str(value).strip())
            )

        # Validate main numbers
        if len(set(numbers)) != 5:
            raise ValueError(
                "Main lottery numbers must be unique."
            )

        for number in numbers:
            if not 1 <= number <= 49:
                raise ValueError(
                    f"Invalid main lottery number: {number}"
                )

        n6 = LotoParser._optional_int(
            row,
            "boule_6",
        )

        bonus = LotoParser._optional_int(
            row,
            "boule_complementaire",
        )

        chance = LotoParser._optional_int(
            row,
            "numero_chance",
        )

        # Legacy sixth number validation
        if n6 is not None and not 1 <= n6 <= 49:
            raise ValueError(
                f"Invalid sixth number: {n6}"
            )

        # Legacy complementary ball validation
        if bonus is not None and not 1 <= bonus <= 49:
            raise ValueError(
                f"Invalid bonus number: {bonus}"
            )

        # Modern Chance number validation
        if chance is not None and not 1 <= chance <= 10:
            raise ValueError(
                f"Invalid Chance number: {chance}"
            )

        return {
            "game": "Loto FDJ",

            "draw_date": (
                LotoParser._parse_date(
                    draw_date_raw
                )
            ),

            "n1": numbers[0],
            "n2": numbers[1],
            "n3": numbers[2],
            "n4": numbers[3],
            "n5": numbers[4],

            "n6": n6,

            "bonus": bonus,

            "chance": chance,
        }