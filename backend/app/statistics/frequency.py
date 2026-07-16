from collections import Counter
from sqlalchemy.orm import Session

from app.statistics.analyzer import StatisticsEngine


class FrequencyAnalyzer:

    @staticmethod
    def calculate(db: Session):

        engine = StatisticsEngine(db)

        numbers = engine.all_numbers()

        counter = Counter(numbers)

        return dict(
            sorted(
                counter.items(),
                key=lambda item: item[0],
            )
        )