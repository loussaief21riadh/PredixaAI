from app.statistics.analyzer import StatisticsEngine


class OverdueAnalyzer:

    @staticmethod
    def calculate(db):

        engine = StatisticsEngine(db)

        draws = list(reversed(engine.all_draws()))

        last_seen = {}

        for draw_index, draw in enumerate(draws):

            numbers = [
                draw.n1,
                draw.n2,
                draw.n3,
                draw.n4,
                draw.n5,
                draw.n6,
                draw.bonus,
                draw.chance,
            ]

            numbers = [n for n in numbers if n is not None]

            for number in numbers:

                if number not in last_seen:
                    last_seen[number] = draw_index

        result = []

        for number, missed in sorted(
            last_seen.items(),
            key=lambda item: item[1],
            reverse=True,
        ):

            result.append(
                {
                    "number": number,
                    "draws_since_last_seen": missed,
                }
            )

        return result