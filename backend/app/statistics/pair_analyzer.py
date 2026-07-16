from collections import Counter
from itertools import combinations

from app.statistics.analyzer import StatisticsEngine


class PairAnalyzer:

    @staticmethod
    def calculate(db, limit: int = 20):

        engine = StatisticsEngine(db)

        counter = Counter()

        for draw in engine.all_draws():

            numbers = [
                draw.n1,
                draw.n2,
                draw.n3,
                draw.n4,
                draw.n5,
            ]

            # On ignore les anciens champs bonus/chance
            # afin de ne comparer que les 5 numéros principaux.
            numbers = sorted(numbers)

            for pair in combinations(numbers, 2):
                counter[pair] += 1

        result = []

        for pair, count in counter.most_common(limit):

            result.append(
                {
                    "numbers": list(pair),
                    "count": count,
                }
            )

        return result