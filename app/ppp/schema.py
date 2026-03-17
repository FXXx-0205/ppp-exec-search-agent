from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.ppp.models import ROLE_NAME


class CurrentRole(BaseModel):
    title: str
    employer: str
    tenure_years: float

    @field_validator("title", "employer")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty.")
        return normalized


class MobilitySignal(BaseModel):
    score: int = Field(ge=1, le=5)
    rationale: str

    @field_validator("rationale")
    @classmethod
    def _require_rationale(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty.")
        return normalized


class RoleFit(BaseModel):
    role: str = ROLE_NAME
    score: int = Field(ge=1, le=10)
    justification: str

    @field_validator("role")
    @classmethod
    def _validate_role(cls, value: str) -> str:
        normalized = value.strip()
        if normalized != ROLE_NAME:
            raise ValueError(f"role must be '{ROLE_NAME}'.")
        return normalized

    @field_validator("justification")
    @classmethod
    def _require_justification(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty.")
        return normalized


class CandidateBrief(BaseModel):
    candidate_id: str
    full_name: str
    current_role: CurrentRole
    career_narrative: str
    experience_tags: list[str]
    firm_aum_context: str
    mobility_signal: MobilitySignal
    role_fit: RoleFit
    outreach_hook: str

    @field_validator("candidate_id", "full_name", "career_narrative", "firm_aum_context", "outreach_hook")
    @classmethod
    def _validate_text_fields(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty.")
        return normalized

    @field_validator("experience_tags")
    @classmethod
    def _validate_tags(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("experience_tags must contain at least one item.")
        return cleaned


class PPPOutput(BaseModel):
    candidates: list[CandidateBrief]

    @model_validator(mode="after")
    def _validate_candidate_count(self) -> "PPPOutput":
        if len(self.candidates) != 5:
            raise ValueError("output.json must contain exactly five candidates.")
        return self


def validate_candidate_brief(payload: Any) -> CandidateBrief:
    return CandidateBrief.model_validate(payload)


def validate_output_document(payload: Any) -> PPPOutput:
    return PPPOutput.model_validate(payload)


def format_validation_error(exc: ValidationError) -> str:
    issue = exc.errors()[0]
    location = ".".join(str(part) for part in issue.get("loc", ()))
    prefix = f"{location}: " if location else ""
    return f"{prefix}{issue['msg']}"
