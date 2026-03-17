from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class SearchRunStatus(StrEnum):
    DRAFT = "draft"
    PARSED = "parsed"
    SEARCHED = "searched"
    RANKED = "ranked"
    BRIEF_GENERATED = "brief_generated"
    COMPLETED = "completed"
    FAILED = "failed"


class BriefStatus(StrEnum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"
    EXPORTED = "exported"


class MatchType(StrEnum):
    EXPLICIT = "explicit_match"
    INFERRED = "inferred_match"
    UNKNOWN = "unknown"


class SearchProjectModel(BaseModel):
    project_id: str
    tenant_id: str
    project_name: str
    client_name: str | None = None
    role_title: str | None = None
    status: str
    created_by: str
    created_at: datetime
    updated_at: datetime
    metadata: dict | None = None


class SearchRunModel(BaseModel):
    run_id: str
    project_id: str
    tenant_id: str
    input_jd: str
    parsed_role_json: dict | None = None
    candidate_source: str | None = None
    result_count: int | None = None
    run_status: SearchRunStatus
    started_at: datetime
    completed_at: datetime | None = None
    failed_at: datetime | None = None
    error_message: str | None = None
    created_by: str
    metadata: dict | None = None


class BriefVersionModel(BaseModel):
    brief_id: str
    project_id: str
    tenant_id: str
    version: int
    content: str
    status: BriefStatus
    created_by: str
    created_at: datetime
    updated_at: datetime
    submitted_by: str | None = None
    submitted_at: datetime | None = None
    approved_by: str | None = None
    approved_at: datetime | None = None
    approval_notes: str | None = None
    rejection_notes: str | None = None
    exported_by: str | None = None
    exported_at: datetime | None = None
    run_id: str | None = None
    previous_brief_id: str | None = None
    supersedes_brief_id: str | None = None
    change_request_source_brief_id: str | None = None
    change_request_notes: str | None = None
    metadata: dict | None = None


class SearchResultSnapshotModel(BaseModel):
    snapshot_id: str
    run_id: str
    project_id: str
    tenant_id: str
    created_at: datetime
    top_candidates: list[dict]
    ranking_payload: list[dict]
    candidate_count: int
    metadata: dict | None = None


class ProjectSummary(BaseModel):
    project_id: str
    tenant_id: str
    project_name: str
    client_name: str | None = None
    role_title: str | None = None
    project_status: str
    latest_run_id: str | None = None
    latest_run_status: str | None = None
    latest_brief_id: str | None = None
    latest_brief_version: int | None = None
    latest_brief_status: str | None = None
    has_pending_approval: bool = False
    last_exported_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class SearchWorkflowResult(BaseModel):
    project_id: str
    run_id: str
    snapshot_id: str | None = None
    parsed_role: dict
    candidate_count: int
    ranked_candidates: list[dict]
    brief_id: str | None = None
    brief_version: int | None = None
    run_status: SearchRunStatus
    warnings: list[str] = Field(default_factory=list)
