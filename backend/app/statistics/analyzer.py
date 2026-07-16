from sqlalchemy.orm import Session

from app.models.draw import Draw


class StatisticsEngine:
    """
    Charge tous les tirages une seule fois.
    Tous les analyseurs utiliseront cet objet.
    """

    def __init__(self, db: Session):

        self.db = db

        self.draws = (
            db.query(Draw)
            .order_by(Draw.draw_date)
            .all()
        )

    def all_draws(self):
        return self.draws

    def all_numbers(self):

        numbers = []

        for draw in self.draws:

            values = [
                draw.n1,
                draw.n2,
                draw.n3,
                draw.n4,
                draw.n5,
                draw.n6,
                draw.bonus,
                draw.chance,
            ]

            values = [v for v in values if v is not None]

            numbers.extend(values)

        return numbers

    def total_draws(self):

        return len(self.draws)