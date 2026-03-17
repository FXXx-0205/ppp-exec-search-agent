from __future__ import annotations

from fastapi import APIRouter

from app.core.metrics import snapshot

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/metrics")
def metrics() -> dict:
    return snapshot()
