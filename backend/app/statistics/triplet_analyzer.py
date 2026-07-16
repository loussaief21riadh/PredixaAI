from collections import Counter
from itertools import combinations

from app.statistics.analyzer import StatisticsEngine


class TripletAnalyzer:

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

            numbers = sorted(numbers)

            for triplet in combinations(numbers, 3):
                counter[triplet] += 1

        result = []

        for triplet, count in counter.most_common(limit):

            result.append(
                {
                    "numbers": list(triplet),
                    "count": count,
                }
            )

        return result