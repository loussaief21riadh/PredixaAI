from fastapi import APIRouter

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)

@router.get("/")
def predict():
    """Predict endpoint."""
    return {
        "prediction": "Not implemented yet",
        "success": True
    }
