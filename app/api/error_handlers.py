from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions import AppError
from app.core.request_context import get_request_id

logger = logging.getLogger(__name__)


def _error_payload(code: str, message: str, details: object = None) -> dict:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "request_id": get_request_id(),
        }
    }


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        logger.warning("Application error", extra={"error_code": exc.code, "status_code": exc.status_code})
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning("Request validation error", extra={"error_code": "request_validation_error"})
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                "request_validation_error",
                "Request validation failed.",
                exc.errors(),
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled application error: %s", exc)
        return JSONResponse(
            status_code=500,
            content=_error_payload("internal_server_error", "An unexpected error occurred."),
        )
