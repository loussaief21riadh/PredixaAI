from collections import Counter
from itertools import combinations

from app.statistics.analyzer import StatisticsEngine


class TripletAnalyzer:

    @staticmethod
    def calculate(
        db,
        limit: int = 20,
    ):
        engine = StatisticsEngine(db)

        counter = Counter()

        for draw in engine.all_draws():

            numbers = sorted(
                engine.draw_main_numbers(draw)
            )

            for triplet in combinations(numbers, 3):
                counter[triplet] += 1

        return [
            {
                "numbers": list(triplet),
                "count": count,
            }
            for triplet, count in counter.most_common(limit)
        ]