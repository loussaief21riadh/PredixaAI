from app.statistics.analyzer import StatisticsEngine


class EvenOddAnalyzer:

    @staticmethod
    def calculate(db):

        engine = StatisticsEngine(db)

        statistics = {}

        for draw in engine.all_draws():

            numbers = engine.draw_main_numbers(draw)

            even = sum(
                1
                for number in numbers
                if number % 2 == 0
            )

            odd = len(numbers) - even

            key = f"{even}-{odd}"

            statistics[key] = (
                statistics.get(key, 0) + 1
            )

        return dict(
            sorted(
                statistics.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )