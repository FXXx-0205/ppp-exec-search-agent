from __future__ import annotations

import os
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

PROFILE_DEFAULTS: dict[str, dict[str, str]] = {
    "local": {
        "log_level": "INFO",
        "storage_backend": "file",
        "crm_provider": "mock",
        "ats_provider": "mock",
        "doc_store_provider": "mock",
    },
    "test": {
        "log_level": "WARNING",
        "storage_backend": "file",
        "crm_provider": "mock",
        "ats_provider": "mock",
        "doc_store_provider": "mock",
    },
    "staging": {
        "log_level": "INFO",
        "storage_backend": "sqlite",
        "crm_provider": "mock",
        "ats_provider": "mock",
        "doc_store_provider": "mock",
    },
    "prod": {
        "log_level": "INFO",
        "storage_backend": "sqlite",
        "crm_provider": "mock",
        "ats_provider": "mock",
        "doc_store_provider": "mock",
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env",), env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    log_level: str = "INFO"
    auth_mode: str = "hybrid"

    anthropic_api_key: str | None = None
    jwt_secret: str | None = None
    jwt_issuer: str | None = None
    jwt_audience: str | None = None
    greenhouse_harvest_api_key: str | None = None
    greenhouse_base_url: str = "https://harvest.greenhouse.io/v1"
    greenhouse_per_page: int = 100
    greenhouse_max_pages: int = 10

    database_url: str = "sqlite:///./data/app.db"
    chroma_persist_dir: str = "./data/vector_db"
    demo_data_dir: str = "./data/raw"
    audit_log_path: str = "./data/audit.jsonl"
    brief_storage_dir: str = "./data/processed/briefs"
    integration_backend: str = "mock"
    storage_backend: str = "file"
    crm_provider: str = "mock"
    ats_provider: str = "mock"
    doc_store_provider: str = "mock"

    ranking_strategy_version: str = "v1"
    ranking_weight_skill_match: float = 0.30
    ranking_weight_seniority_match: float = 0.20
    ranking_weight_sector_relevance: float = 0.20
    ranking_weight_functional_similarity: float = 0.15
    ranking_weight_location_alignment: float = 0.10
    ranking_weight_stability_signal: float = 0.05

    def ranking_weights(self) -> dict[str, float]:
        return {
            "skill_match": self.ranking_weight_skill_match,
            "seniority_match": self.ranking_weight_seniority_match,
            "sector_relevance": self.ranking_weight_sector_relevance,
            "functional_similarity": self.ranking_weight_functional_similarity,
            "location_alignment": self.ranking_weight_location_alignment,
            "stability_signal": self.ranking_weight_stability_signal,
        }

    def safe_dump(self) -> dict[str, Any]:
        data = self.model_dump()
        if data.get("anthropic_api_key"):
            data["anthropic_api_key"] = "***"
        return data


def load_settings(app_env: str | None = None) -> Settings:
    env_name = (app_env or os.getenv("APP_ENV") or "local").lower()
    env_file = (".env", f".env.{env_name}")
    loaded = Settings(_env_file=env_file)  # type: ignore[call-arg]
    defaults = PROFILE_DEFAULTS.get(env_name, PROFILE_DEFAULTS["local"])

    for field_name, default_value in defaults.items():
        env_var_name = field_name.upper()
        if env_var_name not in os.environ:
            setattr(loaded, field_name, default_value)

    loaded.app_env = env_name
    return loaded


settings = load_settings()
