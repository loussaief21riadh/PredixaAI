from collections import Counter

from app.statistics.analyzer import StatisticsEngine


class DecadeAnalyzer:

    @staticmethod
    def calculate(db):

        engine = StatisticsEngine(db)

        counter = Counter()

        for number in engine.all_numbers():

            if 1 <= number <= 9:
                counter["1-9"] += 1

            elif 10 <= number <= 19:
                counter["10-19"] += 1

            elif 20 <= number <= 29:
                counter["20-29"] += 1

            elif 30 <= number <= 39:
                counter["30-39"] += 1

            elif 40 <= number <= 49:
                counter["40-49"] += 1

        result = []

        for decade in [
            "1-9",
            "10-19",
            "20-29",
            "30-39",
            "40-49",
        ]:

            result.append(
                {
                    "decade": decade,
                    "count": counter.get(decade, 0),
                }
            )

        return result