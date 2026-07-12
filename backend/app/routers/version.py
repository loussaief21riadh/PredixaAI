from fastapi import APIRouter

router = APIRouter()

@router.get("/version")
def get_version():
    """Get the version of the application."""
    return {
        "app": "LottoVisionAI",
        "version": "0.1.0"
    }
