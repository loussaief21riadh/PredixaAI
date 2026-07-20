from collections import Counter

from app.statistics.analyzer import StatisticsEngine


class SumAnalyzer:

    @staticmethod
    def calculate(db):

        engine = StatisticsEngine(db)

        counter = Counter()

        for draw in engine.all_draws():

            numbers = engine.draw_main_numbers(draw)

            # Ignore incomplete draws
            if len(numbers) != 5:
                continue

            total = sum(numbers)

            counter[total] += 1

        return [
            {
                "sum": total,
                "count": count,
            }
            for total, count in sorted(
                counter.items()
            )
        ]