from app.statistics.analyzer import StatisticsEngine


class OverdueAnalyzer:

    @staticmethod
    def calculate(db):

        engine = StatisticsEngine(db)

        # Most recent draw first
        draws = list(
            reversed(
                engine.all_draws()
            )
        )

        last_seen = {
            number: None
            for number in range(1, 50)
        }

        for draw_index, draw in enumerate(draws):

            numbers = engine.draw_main_numbers(
                draw
            )

            for number in numbers:

                if last_seen[number] is None:
                    last_seen[number] = draw_index

            # Stop early if all 49 numbers
            # have already been found
            if all(
                value is not None
                for value in last_seen.values()
            ):
                break

        result = []

        for number in range(1, 50):

            missed = last_seen[number]

            result.append(
                {
                    "number": number,
                    "draws_since_last_seen": (
                        missed
                        if missed is not None
                        else len(draws)
                    ),
                }
            )

        return sorted(
            result,
            key=lambda item: (
                item["draws_since_last_seen"],
                item["number"],
            ),
            reverse=True,
        )