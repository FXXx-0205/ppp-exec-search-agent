from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Any

from fastapi import Depends, Header

from app.config import settings
from app.core.exceptions import ForbiddenError, ValidationError
from app.models.auth import ROLE_PERMISSIONS, AccessContext, UserIdentity, UserRole


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValidationError("Malformed bearer token.", details={"reason": "invalid_jwt_parts"})

    header_segment, payload_segment, signature_segment = parts
    try:
        header = json.loads(_b64url_decode(header_segment))
        payload = json.loads(_b64url_decode(payload_segment))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValidationError("Malformed bearer token.", details={"reason": "invalid_jwt_encoding"}) from exc

    if settings.jwt_secret:
        signing_input = f"{header_segment}.{payload_segment}".encode("utf-8")
        expected_signature = base64.urlsafe_b64encode(
            hmac.new(settings.jwt_secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
        ).decode("utf-8").rstrip("=")
        if not hmac.compare_digest(expected_signature, signature_segment):
            raise ForbiddenError("Invalid bearer token signature.")

    issuer = settings.jwt_issuer
    if issuer and payload.get("iss") != issuer:
        raise ForbiddenError("Bearer token issuer mismatch.", details={"expected_issuer": issuer})

    audience = settings.jwt_audience
    if audience and payload.get("aud") != audience:
        raise ForbiddenError("Bearer token audience mismatch.", details={"expected_audience": audience})

    if header.get("alg") not in {None, "HS256", "none"}:
        raise ValidationError("Unsupported bearer token algorithm.", details={"alg": header.get("alg")})

    return payload


def _access_context_from_payload(payload: dict[str, Any]) -> AccessContext:
    role_value = str(payload.get("role") or "researcher").lower()
    try:
        role = UserRole(role_value)
    except ValueError as exc:
        raise ValidationError("Unsupported user role.", details={"role": role_value}) from exc

    return AccessContext(
        tenant_id=str(payload.get("tenant_id") or "demo-tenant"),
        project_id=payload.get("project_id"),
        actor=UserIdentity(
            user_id=str(payload.get("sub") or payload.get("user_id") or "jwt-user"),
            email=str(payload.get("email") or "jwt-user@example.com"),
            display_name=str(payload.get("name") or payload.get("display_name") or "JWT User"),
            role=role,
        ),
    )


def _access_context_from_headers(
    *,
    x_user_id: str | None,
    x_user_email: str | None,
    x_user_name: str | None,
    x_user_role: str | None,
    x_tenant_id: str | None,
    x_project_id: str | None,
) -> AccessContext:
    role_value = (x_user_role or "researcher").lower()
    try:
        role = UserRole(role_value)
    except ValueError as exc:
        raise ValidationError("Unsupported user role.", details={"role": role_value}) from exc

    return AccessContext(
        tenant_id=x_tenant_id or "demo-tenant",
        project_id=x_project_id,
        actor=UserIdentity(
            user_id=x_user_id or "demo-user",
            email=x_user_email or "demo@example.com",
            display_name=x_user_name or "Demo User",
            role=role,
        ),
    )


def get_access_context(
    authorization: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
    x_user_email: str | None = Header(default=None),
    x_user_name: str | None = Header(default=None),
    x_user_role: str | None = Header(default=None),
    x_tenant_id: str | None = Header(default=None),
    x_project_id: str | None = Header(default=None),
) -> AccessContext:
    auth_mode = settings.auth_mode.lower()
    if isinstance(authorization, str) and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
        return _access_context_from_payload(_decode_jwt_payload(token))

    if auth_mode == "jwt":
        raise ForbiddenError("Bearer token required for this environment.")

    return _access_context_from_headers(
        x_user_id=x_user_id,
        x_user_email=x_user_email,
        x_user_name=x_user_name,
        x_user_role=x_user_role,
        x_tenant_id=x_tenant_id,
        x_project_id=x_project_id,
    )


ACCESS_CONTEXT_DEPENDENCY = Depends(get_access_context)


def require_permission(permission: str):
    def dependency(context: AccessContext = ACCESS_CONTEXT_DEPENDENCY) -> AccessContext:
        allowed = ROLE_PERMISSIONS.get(context.actor.role, set())
        if permission not in allowed:
            raise ForbiddenError(
                "You do not have permission to perform this action.",
                details={"required_permission": permission, "role": context.actor.role},
            )
        return context

    return dependency
