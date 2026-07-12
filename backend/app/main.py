from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "LottoVisionAI Backend Running"}

@app.get("/health")
def health_check():
    """Check if the application is healthy."""
    return {"status": "healthy"}

@app.get("/version")
def get_version():
    """Get the version of the application."""
    return {
        "app": "LottoVisionAI",
        "version": "0.1.0"
    }
