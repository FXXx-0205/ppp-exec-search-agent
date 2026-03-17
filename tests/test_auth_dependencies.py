from __future__ import annotations

import base64
import hashlib
import hmac
import json

from app.api.dependencies.auth import get_access_context
from app.config import settings
from app.core.exceptions import ValidationError


def test_invalid_user_role_raises_validation_error() -> None:
    try:
        get_access_context(x_user_role="invalid-role")
    except ValidationError as exc:
        assert exc.code == "validation_error"
    else:
        raise AssertionError("Expected ValidationError for invalid role")


def test_bearer_token_builds_access_context() -> None:
    settings.jwt_secret = "test-secret"
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "sub": "user_jwt",
                "email": "jwt@example.com",
                "name": "JWT User",
                "role": "consultant",
                "tenant_id": "tenant_jwt",
            }
        ).encode()
    ).decode().rstrip("=")
    signing_input = f"{header}.{payload}".encode("utf-8")
    signature = base64.urlsafe_b64encode(
        hmac.new(settings.jwt_secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    ).decode().rstrip("=")
    token = f"{header}.{payload}.{signature}"

    context = get_access_context(authorization=f"Bearer {token}")

    assert context.actor.user_id == "user_jwt"
    assert context.tenant_id == "tenant_jwt"
    assert context.actor.role.value == "consultant"
    settings.jwt_secret = None
