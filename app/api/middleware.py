from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request

from app.core.metrics import observe_request
from app.core.request_context import set_request_id

logger = logging.getLogger("app.request")


def register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:12]}"
        set_request_id(request_id)
        request.state.request_id = request_id

        started_at = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)

        response.headers["x-request-id"] = request_id
        observe_request(duration_ms)
        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response
