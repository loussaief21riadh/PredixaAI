from collections import Counter

from sqlalchemy.orm import Session

from app.statistics.analyzer import StatisticsEngine


class ColdNumbersAnalyzer:

    @staticmethod
    def calculate(
        db: Session,
        limit: int = 10,
    ):
        engine = StatisticsEngine(db)

        numbers = engine.all_numbers()

        counter = Counter(numbers)

        # Guarantee that every main Loto number
        # from 1 to 49 is represented.
        frequencies = {
            number: counter.get(number, 0)
            for number in range(1, 50)
        }

        # Sort by frequency, then by number
        # for deterministic results.
        cold_numbers = sorted(
            frequencies.items(),
            key=lambda item: (
                item[1],
                item[0],
            ),
        )[:limit]

        return [
            {
                "number": number,
                "count": count,
            }
            for number, count in cold_numbers
        ]