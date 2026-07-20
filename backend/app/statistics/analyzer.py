from sqlalchemy.orm import Session

from app.models.draw import Draw


class StatisticsEngine:
    """
    Central statistics data engine for Predixa AI.

    This class loads all lottery draws once and exposes
    separated datasets for the different statistical analyzers.

    Main lottery statistics use only n1 to n5.

    Legacy and secondary fields such as:
    - n6
    - bonus
    - chance

    are exposed separately to avoid mixing them with the
    five principal lottery numbers.
    """

    def __init__(self, db: Session):
        self.db = db

        self.draws = (
            db.query(Draw)
            .order_by(Draw.draw_date.asc())
            .all()
        )

    def all_draws(self):
        """
        Return all draws ordered by draw date.
        """
        return self.draws

    def total_draws(self) -> int:
        """
        Return the total number of draws.
        """
        return len(self.draws)

    def draw_main_numbers(self, draw: Draw) -> list[int]:
        """
        Return the five main numbers for a single draw.

        Only n1 to n5 are included.
        """

        return [
            value
            for value in [
                draw.n1,
                draw.n2,
                draw.n3,
                draw.n4,
                draw.n5,
            ]
            if value is not None
        ]

    def main_numbers(self) -> list[int]:
        """
        Return all main lottery numbers from all draws.

        Used for:
        - frequency
        - hot numbers
        - cold numbers
        - overdue numbers
        - pairs
        - triplets
        - even/odd
        - sums
        - decades
        - consecutive numbers
        """

        numbers = []

        for draw in self.draws:
            numbers.extend(
                self.draw_main_numbers(draw)
            )

        return numbers

    def all_numbers(self) -> list[int]:
        """
        Backward-compatible alias.

        Existing analyzers using all_numbers() will continue
        to work, but only the five main numbers are returned.
        """

        return self.main_numbers()

    def legacy_numbers(self) -> list[int]:
        """
        Return legacy sixth numbers (n6).
        """

        return [
            draw.n6
            for draw in self.draws
            if draw.n6 is not None
        ]

    def bonus_numbers(self) -> list[int]:
        """
        Return legacy bonus numbers only.
        """

        return [
            draw.bonus
            for draw in self.draws
            if draw.bonus is not None
        ]

    def chance_numbers(self) -> list[int]:
        """
        Return Chance numbers only.
        """

        return [
            draw.chance
            for draw in self.draws
            if draw.chance is not None
        ]