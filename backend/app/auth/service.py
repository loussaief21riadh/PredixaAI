from sqlalchemy.orm import Session

from app.auth.models import User
from app.auth.schemas import UserRegister
from app.auth.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from app.repositories.user_repository import UserRepository


class AuthService:
    @staticmethod
    def register(
        db: Session,
        user: UserRegister,
    ) -> User:

        if UserRepository.get_by_email(db, user.email):
            raise ValueError("Email already exists")

        if UserRepository.get_by_username(db, user.username):
            raise ValueError("Username already exists")

        db_user = User(
            username=user.username,
            email=user.email,
            hashed_password=hash_password(user.password),
        )

        return UserRepository.create(db, db_user)

    @staticmethod
    def login(
        db: Session,
        email: str,
        password: str,
    ):

        print("=" * 60)
        print("LOGIN REQUEST")
        print("EMAIL RECEIVED :", repr(email))
        print("PASSWORD       :", repr(password))

        user = UserRepository.get_by_email(db, email)

        print("USER FOUND     :", user)

        if user is None:
            print("USER NOT FOUND")
            return None

        print("DB EMAIL       :", user.email)
        print("HASH           :", user.hashed_password)

        try:
            valid = verify_password(password, user.hashed_password)
            print("VERIFY RESULT  :", valid)
        except Exception as e:
            print("VERIFY ERROR   :", repr(e))
            raise

        if not valid:
            print("PASSWORD INVALID")
            return None

        print("LOGIN SUCCESS")

        token = create_access_token(
            {
                "sub": str(user.id),
                "email": user.email,
                "username": user.username,
                "is_admin": user.is_admin,
            }
        )

        return {
            "access_token": token,
            "token_type": "bearer",
            "user": user,
        }