from sqlalchemy import Column, Date, Integer

from app.database import Base


class Draw(Base):
    __tablename__ = "draws"

    id = Column(Integer, primary_key=True, index=True)
    draw_date = Column(Date, nullable=False, index=True)

    n1 = Column(Integer, nullable=False)
    n2 = Column(Integer, nullable=False)
    n3 = Column(Integer, nullable=False)
    n4 = Column(Integer, nullable=False)
    n5 = Column(Integer, nullable=False)
    n6 = Column(Integer, nullable=False)

    bonus = Column(Integer, nullable=True)