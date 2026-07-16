from collections import Counter
from sqlalchemy.orm import Session

from app.statistics.analyzer import StatisticsEngine


class HotNumbersAnalyzer:

    @staticmethod
    def calculate(
        db: Session,
        limit: int = 10,
    ):

        engine = StatisticsEngine(db)

        numbers = engine.all_numbers()

        counter = Counter(numbers)

        result = []

        for number, count in counter.most_common(limit):

            result.append(
                {
                    "number": number,
                    "count": count,
                }
            )

        return result