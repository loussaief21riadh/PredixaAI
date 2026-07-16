from collections import Counter

from app.statistics.analyzer import StatisticsEngine


class ConsecutiveAnalyzer:

    @staticmethod
    def calculate(db):

        engine = StatisticsEngine(db)

        counter = Counter()

        for draw in engine.all_draws():

            numbers = sorted([
                draw.n1,
                draw.n2,
                draw.n3,
                draw.n4,
                draw.n5,
            ])

            consecutive = 0

            for i in range(4):

                if numbers[i + 1] == numbers[i] + 1:
                    consecutive += 1

            counter[consecutive] += 1

        result = []

        for value in sorted(counter.keys()):

            result.append(
                {
                    "consecutive_pairs": value,
                    "draws": counter[value],
                }
            )

        return result