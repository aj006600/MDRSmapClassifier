"""後端的 FastAPI 進入點。"""

from fastapi import FastAPI

from app.config import settings
from app.logging_setup import configure_logging

configure_logging(settings)

app = FastAPI(title="mdrsmap-classifier backend", version="0.1.0")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/message")
def message():
    return {"message": "Hello from the backend"}
