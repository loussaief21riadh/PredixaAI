from sqlalchemy import Column, Date, DateTime, Integer, String
from sqlalchemy.sql import func

from app.database import Base


class Draw(Base):
    __tablename__ = "draws"

    id = Column(Integer, primary_key=True, index=True)

    game = Column(String(50), nullable=False, default="Loto FDJ")

    draw_date = Column(Date, nullable=False, index=True)

    n1 = Column(Integer, nullable=False)
    n2 = Column(Integer, nullable=False)
    n3 = Column(Integer, nullable=False)
    n4 = Column(Integer, nullable=False)
    n5 = Column(Integer, nullable=False)

    # Ancien Loto (6e numéro)
    n6 = Column(Integer, nullable=True)

    # Ancien Loto (boule complémentaire)
    bonus = Column(Integer, nullable=True)

    # Nouveau Loto (numéro Chance)
    chance = Column(Integer, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )