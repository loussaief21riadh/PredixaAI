from collections import Counter

from app.statistics.analyzer import StatisticsEngine


class SumAnalyzer:

    @staticmethod
    def calculate(db):

        engine = StatisticsEngine(db)

        counter = Counter()

        for draw in engine.all_draws():

            total = (
                draw.n1
                + draw.n2
                + draw.n3
                + draw.n4
                + draw.n5
            )

            counter[total] += 1

        result = []

        for total, count in sorted(counter.items()):

            result.append(
                {
                    "sum": total,
                    "count": count,
                }
            )

        return result