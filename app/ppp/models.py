from __future__ import annotations

from pydantic import BaseModel, field_validator

REQUIRED_CSV_COLUMNS = ("full_name", "current_employer", "current_title", "linkedin_url")
ROLE_NAME = "Head of Distribution / National BDM"


class CandidateCSVRow(BaseModel):
    full_name: str
    current_employer: str
    current_title: str
    linkedin_url: str

    @field_validator("full_name", "current_employer", "current_title", "linkedin_url")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty.")
        return normalized
