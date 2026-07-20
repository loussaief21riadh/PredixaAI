from typing import Optional

from sqlalchemy.orm import Session

from app.auth.models import User


class UserRepository:
    """
    Repository responsible for all User database operations.
    """

    @staticmethod
    def get_by_id(
        db: Session,
        user_id: int,
    ) -> Optional[User]:
        return (
            db.query(User)
            .filter(User.id == user_id)
            .first()
        )

    @staticmethod
    def get_by_email(
        db: Session,
        email: str,
    ) -> Optional[User]:
        return (
            db.query(User)
            .filter(User.email == email)
            .first()
        )

    @staticmethod
    def get_by_username(
        db: Session,
        username: str,
    ) -> Optional[User]:
        return (
            db.query(User)
            .filter(User.username == username)
            .first()
        )

    @staticmethod
    def count(
        db: Session,
    ) -> int:
        return db.query(User).count()

    @staticmethod
    def create(
        db: Session,
        user: User,
    ) -> User:
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def update(
        db: Session,
        user: User,
    ) -> User:
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def delete(
        db: Session,
        user: User,
    ) -> None:
        db.delete(user)
        db.commit()