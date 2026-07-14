from collections import Counter

from sqlalchemy.orm import Session

from app.models.draw import Draw


class FrequencyAnalyzer:

    @staticmethod
    def calculate(db: Session):

        counter = Counter()

        draws = db.query(Draw).all()

        for draw in draws:

            numbers = [
                draw.n1,
                draw.n2,
                draw.n3,
                draw.n4,
                draw.n5,
            ]

            if draw.n6:
                numbers.append(draw.n6)

            for number in numbers:
                counter[number] += 1

        return dict(sorted(counter.items()))