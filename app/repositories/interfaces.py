from __future__ import annotations

import builtins
from dataclasses import dataclass
from typing import Any, Protocol

from app.models.auth import ApprovalStatus
from app.models.workflow import BriefStatus, SearchRunStatus


@dataclass(frozen=True)
class StoredSearchProject:
    project_id: str
    tenant_id: str
    project_name: str
    client_name: str | None
    role_title: str | None
    status: str
    created_by: str
    created_at: str
    updated_at: str
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class StoredSearchRun:
    run_id: str
    project_id: str
    tenant_id: str
    input_jd: str
    parsed_role_json: dict[str, Any] | None
    candidate_source: str | None
    result_count: int | None
    run_status: SearchRunStatus
    started_at: str
    completed_at: str | None
    failed_at: str | None
    error_message: str | None
    created_by: str
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class StoredSearchResultSnapshot:
    snapshot_id: str
    run_id: str
    project_id: str
    tenant_id: str
    created_at: str
    top_candidates: list[dict[str, Any]]
    ranking_payload: list[dict[str, Any]]
    candidate_count: int
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, init=False)
class StoredBrief:
    brief_id: str
    project_id: str
    tenant_id: str
    version: int
    content: str
    status: BriefStatus
    created_by: str
    created_at: str
    updated_at: str
    role_spec: dict[str, Any]
    submitted_by: str | None = None
    submitted_at: str | None = None
    approved_by: str | None = None
    approved_at: str | None = None
    approval_notes: str | None = None
    rejection_notes: str | None = None
    exported_by: str | None = None
    exported_at: str | None = None
    run_id: str | None = None
    previous_brief_id: str | None = None
    supersedes_brief_id: str | None = None
    change_request_source_brief_id: str | None = None
    change_request_notes: str | None = None
    metadata: dict[str, Any] | None = None
    markdown: str = ""
    citations: list[str] | None = None
    generated_at: str | None = None

    def __init__(
        self,
        *,
        brief_id: str,
        project_id: str | None = None,
        tenant_id: str,
        version: int = 1,
        content: str | None = None,
        status: BriefStatus | str | None = None,
        created_by: str,
        created_at: str | None = None,
        updated_at: str | None = None,
        role_spec: dict[str, Any],
        submitted_by: str | None = None,
        submitted_at: str | None = None,
        approved_by: str | None = None,
        approved_at: str | None = None,
        approval_notes: str | None = None,
        rejection_notes: str | None = None,
        exported_by: str | None = None,
        exported_at: str | None = None,
        run_id: str | None = None,
        previous_brief_id: str | None = None,
        supersedes_brief_id: str | None = None,
        change_request_source_brief_id: str | None = None,
        change_request_notes: str | None = None,
        metadata: dict[str, Any] | None = None,
        markdown: str = "",
        citations: list[str] | None = None,
        generated_at: str | None = None,
        approval_status: ApprovalStatus | str | None = None,
    ):
        resolved_status = status
        if resolved_status is None and approval_status is not None:
            approval_value = ApprovalStatus(approval_status)
            resolved_status = {
                ApprovalStatus.DRAFT: BriefStatus.DRAFT,
                ApprovalStatus.PENDING: BriefStatus.PENDING_APPROVAL,
                ApprovalStatus.APPROVED: BriefStatus.APPROVED,
                ApprovalStatus.REJECTED: BriefStatus.REJECTED,
            }[approval_value]
        if resolved_status is None:
            resolved_status = BriefStatus.DRAFT
        created_ts = created_at or generated_at or ""
        updated_ts = updated_at or created_ts
        content_value = content if content is not None else markdown

        object.__setattr__(self, "brief_id", brief_id)
        object.__setattr__(self, "project_id", project_id or "")
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "content", content_value)
        object.__setattr__(self, "status", BriefStatus(resolved_status))
        object.__setattr__(self, "created_by", created_by)
        object.__setattr__(self, "created_at", created_ts)
        object.__setattr__(self, "updated_at", updated_ts)
        object.__setattr__(self, "role_spec", role_spec)
        object.__setattr__(self, "submitted_by", submitted_by)
        object.__setattr__(self, "submitted_at", submitted_at)
        object.__setattr__(self, "approved_by", approved_by)
        object.__setattr__(self, "approved_at", approved_at)
        object.__setattr__(self, "approval_notes", approval_notes)
        object.__setattr__(self, "rejection_notes", rejection_notes)
        object.__setattr__(self, "exported_by", exported_by)
        object.__setattr__(self, "exported_at", exported_at)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "previous_brief_id", previous_brief_id)
        object.__setattr__(self, "supersedes_brief_id", supersedes_brief_id)
        object.__setattr__(self, "change_request_source_brief_id", change_request_source_brief_id)
        object.__setattr__(self, "change_request_notes", change_request_notes)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "markdown", markdown or content_value)
        object.__setattr__(self, "citations", citations)
        object.__setattr__(self, "generated_at", generated_at or created_ts)

    @property
    def approval_status(self) -> ApprovalStatus:
        mapping = {
            BriefStatus.DRAFT: ApprovalStatus.DRAFT,
            BriefStatus.PENDING_APPROVAL: ApprovalStatus.PENDING,
            BriefStatus.APPROVED: ApprovalStatus.APPROVED,
            BriefStatus.REJECTED: ApprovalStatus.REJECTED,
            BriefStatus.CHANGES_REQUESTED: ApprovalStatus.REJECTED,
            BriefStatus.EXPORTED: ApprovalStatus.APPROVED,
        }
        return mapping[self.status]


@dataclass(frozen=True)
class StoredCandidate:
    tenant_id: str
    candidate_id: str
    full_name: str
    current_title: str
    current_company: str | None
    location: str | None
    primary_email: str | None
    summary: str
    evidence: list[str]
    source_system: str
    source_id: str
    application_ids: list[str]
    tag_names: list[str]
    attachment_count: int
    synced_at: str


class BriefRepository(Protocol):
    def save(self, brief: StoredBrief) -> None: ...

    def create_brief_version(self, brief: StoredBrief) -> StoredBrief: ...

    def get(self, brief_id: str) -> StoredBrief | None: ...

    def get_brief(self, brief_id: str) -> StoredBrief | None: ...

    def list(
        self,
        *,
        tenant_id: str,
        project_id: str | None = None,
        approval_status: ApprovalStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> builtins.list[StoredBrief]: ...

    def list_briefs_by_project(self, *, project_id: str, tenant_id: str) -> builtins.list[StoredBrief]: ...

    def get_latest_brief_by_project(self, *, project_id: str, tenant_id: str) -> StoredBrief | None: ...

    def update_brief_status(
        self,
        brief_id: str,
        *,
        status: BriefStatus,
        updated_by: str,
        notes: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StoredBrief | None: ...

    def submit_for_approval(self, brief_id: str, *, submitted_by: str, notes: str | None = None) -> StoredBrief | None: ...

    def approve_brief(self, brief_id: str, *, approved_by: str, notes: str | None = None) -> StoredBrief | None: ...

    def reject_brief(self, brief_id: str, *, rejected_by: str, notes: str | None = None) -> StoredBrief | None: ...

    def request_changes(self, brief_id: str, *, requested_by: str, notes: str | None = None) -> StoredBrief | None: ...

    def mark_exported(self, brief_id: str, *, exported_by: str) -> StoredBrief | None: ...

    def create_revision(
        self,
        source_brief_id: str,
        *,
        new_brief_id: str,
        created_by: str,
        content: str | None = None,
    ) -> StoredBrief | None: ...

    def decide(
        self,
        brief_id: str,
        *,
        status: ApprovalStatus,
        decided_by: str,
        comment: str | None = None,
    ) -> StoredBrief | None: ...


class AuditRepository(Protocol):
    def append(self, event: dict[str, Any]) -> None: ...

    def list_events(
        self,
        *,
        tenant_id: str,
        project_id: str | None = None,
        event_type: str | None = None,
        action: str | None = None,
        resource_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]: ...


class CandidateRepository(Protocol):
    def upsert_many(self, candidates: list[StoredCandidate]) -> None: ...

    def list(
        self,
        *,
        tenant_id: str,
        search_text: str | None = None,
        source_system: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[StoredCandidate]: ...


class ProjectRepository(Protocol):
    def create_project(self, project: StoredSearchProject) -> StoredSearchProject: ...

    def get_project(self, project_id: str) -> StoredSearchProject | None: ...

    def list_projects(self, *, tenant_id: str, limit: int = 50, offset: int = 0) -> list[StoredSearchProject]: ...

    def update_project(self, project: StoredSearchProject) -> StoredSearchProject: ...

    def delete_project(self, project_id: str) -> bool: ...

    def list_project_runs(self, project_id: str, *, tenant_id: str) -> list[StoredSearchRun]: ...

    def list_project_briefs(self, project_id: str, *, tenant_id: str) -> list[StoredBrief]: ...

    def list_project_snapshots(self, project_id: str, *, tenant_id: str) -> list[StoredSearchResultSnapshot]: ...


class SearchRunRepository(Protocol):
    def create_run(self, run: StoredSearchRun) -> StoredSearchRun: ...

    def get_run(self, run_id: str) -> StoredSearchRun | None: ...

    def list_runs_by_project(self, *, project_id: str, tenant_id: str) -> list[StoredSearchRun]: ...

    def update_run(self, run: StoredSearchRun) -> StoredSearchRun: ...

    def mark_run_failed(self, run_id: str, *, error_message: str, failed_at: str) -> StoredSearchRun | None: ...

    def mark_run_completed(
        self,
        run_id: str,
        *,
        run_status: SearchRunStatus,
        completed_at: str,
    ) -> StoredSearchRun | None: ...


class SearchResultSnapshotRepository(Protocol):
    def create_snapshot(self, snapshot: StoredSearchResultSnapshot) -> StoredSearchResultSnapshot: ...

    def get_snapshot(self, snapshot_id: str) -> StoredSearchResultSnapshot | None: ...

    def get_snapshot_by_run(self, *, run_id: str, tenant_id: str) -> StoredSearchResultSnapshot | None: ...

    def list_snapshots_by_project(self, *, project_id: str, tenant_id: str) -> list[StoredSearchResultSnapshot]: ...
