from __future__ import annotations

from app.config import load_settings


def test_load_settings_applies_profile_defaults() -> None:
    settings = load_settings("staging")

    assert settings.app_env == "staging"
    assert settings.storage_backend == "sqlite"
    assert settings.crm_provider == "mock"
