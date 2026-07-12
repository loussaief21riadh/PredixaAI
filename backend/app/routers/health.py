from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health_check():
    """Check if the application is healthy."""
    return {"status": "healthy"}
