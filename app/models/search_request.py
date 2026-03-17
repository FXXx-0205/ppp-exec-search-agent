from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.auth import ApprovalStatus


class IntakeRequest(BaseModel):
    raw_input: str = Field(min_length=1, max_length=8000)
    project_id: str | None = None


class CandidatesRequest(BaseModel):
    project_id: str | None = None
    role_spec: dict
    provider_filters: dict | None = None


class RankRequest(BaseModel):
    role_spec: dict
    candidate_ids: list[str]


class BriefGenerateRequest(BaseModel):
    project_id: str | None = None
    role_spec: dict | None = None
    candidate_ids: list[str] | None = None


class BriefApprovalRequest(BaseModel):
    status: ApprovalStatus = ApprovalStatus.APPROVED
    comment: str | None = Field(default=None, max_length=1000)


class ProjectCreateRequest(BaseModel):
    project_name: str = Field(min_length=1, max_length=255)
    client_name: str | None = Field(default=None, max_length=255)
    role_title: str | None = Field(default=None, max_length=255)
    metadata: dict | None = None


class ProjectSearchRunRequest(BaseModel):
    jd_text: str = Field(min_length=1, max_length=12000)
    candidate_source: str = Field(default="local_first", max_length=100)


class BriefActionRequest(BaseModel):
    notes: str | None = Field(default=None, max_length=1000)


class BriefExportRequest(BaseModel):
    export_format: str = Field(default="md", max_length=20)


class BriefRevisionRequest(BaseModel):
    content: str | None = Field(default=None, max_length=20000)
