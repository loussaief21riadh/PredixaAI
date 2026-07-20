from collections import Counter

from app.statistics.analyzer import StatisticsEngine


class ConsecutiveAnalyzer:

    @staticmethod
    def calculate(db):

        engine = StatisticsEngine(db)

        counter = Counter()

        for draw in engine.all_draws():

            numbers = sorted(
                engine.draw_main_numbers(draw)
            )

            # Ignore incomplete draws
            if len(numbers) != 5:
                continue

            consecutive_pairs = sum(
                1
                for i in range(len(numbers) - 1)
                if numbers[i + 1] == numbers[i] + 1
            )

            counter[consecutive_pairs] += 1

        return [
            {
                "consecutive_pairs": value,
                "draws": counter[value],
            }
            for value in sorted(counter)
        ]