from __future__ import annotations

from typing import Any


class AppError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str = "app_error",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class ValidationError(AppError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(
            message,
            code="validation_error",
            status_code=422,
            details=details,
        )


class NotFoundError(AppError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(
            message,
            code="not_found",
            status_code=404,
            details=details,
        )


class ForbiddenError(AppError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(
            message,
            code="forbidden",
            status_code=403,
            details=details,
        )


class ConflictError(AppError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(
            message,
            code="conflict",
            status_code=409,
            details=details,
        )


class ExternalServiceError(AppError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(
            message,
            code="external_service_error",
            status_code=502,
            details=details,
        )
