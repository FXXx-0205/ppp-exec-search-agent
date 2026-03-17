from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, Query

from app.api.dependencies.auth import require_permission
from app.api.dependencies.integrations import get_ats_adapter
from app.core.audit import AuditEvent, AuditLogger
from app.core.exceptions import ForbiddenError, NotFoundError
from app.llm.anthropic_client import ClaudeClient
from app.models.auth import AccessContext
from app.models.search_request import ProjectCreateRequest, ProjectSearchRunRequest
from app.models.workflow import BriefStatus, ProjectSummary, SearchRunStatus
from app.repositories.factory import (
    get_audit_repository,
    get_brief_repository,
    get_candidate_repository,
    get_project_repository,
    get_search_result_snapshot_repository,
    get_search_run_repository,
)
from app.repositories.interfaces import StoredBrief, StoredSearchProject, StoredSearchRun
from app.services.brief_service import BriefService
from app.services.candidate_service import CandidateService
from app.services.job_parser_service import JobParserService
from app.services.ranking_service import RankingService
from app.workflows.search_workflow import SearchWorkflow

router = APIRouter()

_project_repo = get_project_repository()
_run_repo = get_search_run_repository()
_brief_repo = get_brief_repository()
_snapshot_repo = get_search_result_snapshot_repository()
_audit_repo = get_audit_repository()
_audit = AuditLogger()
_llm = ClaudeClient()


@router.post("")
def create_project(
    req: ProjectCreateRequest,
    access: AccessContext = Depends(require_permission("project:create")),
) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    project = StoredSearchProject(
        project_id=f"proj_{uuid4().hex[:12]}",
        tenant_id=access.tenant_id,
        project_name=req.project_name,
        client_name=req.client_name,
        role_title=req.role_title,
        status=SearchRunStatus.DRAFT,
        created_by=access.actor.user_id,
        created_at=now,
        updated_at=now,
        metadata=req.metadata,
    )
    _project_repo.create_project(project)
    _audit.log(
        AuditEvent(
            event_type="project_created",
            request_id=project.project_id,
            action="create_project",
            resource_type="search_project",
            resource_id=project.project_id,
            payload={"project_name": project.project_name, "client_name": project.client_name},
            tenant_id=access.tenant_id,
            project_id=project.project_id,
            actor_id=access.actor.user_id,
        )
    )
    return asdict(project)


@router.get("")
def list_projects(
    access: AccessContext = Depends(require_permission("project:view")),
    view: str = Query(default="summary"),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    projects = _project_repo.list_projects(tenant_id=access.tenant_id, limit=limit, offset=offset)
    if view == "raw":
        payload = [asdict(project) for project in projects]
    else:
        payload = [build_project_summary(project.project_id, access.tenant_id).model_dump(mode="json") for project in projects]
    return {"projects": payload, "count": len(projects), "tenant_id": access.tenant_id, "view": view}


@router.get("/{project_id}")
def get_project(
    project_id: str,
    access: AccessContext = Depends(require_permission("project:view")),
    view: str = Query(default="raw"),
) -> dict:
    project = _project_repo.get_project(project_id)
    if project is None:
        raise NotFoundError("Project not found.", details={"project_id": project_id})
    if project.tenant_id != access.tenant_id:
        raise ForbiddenError("Project belongs to a different tenant.", details={"project_id": project_id})
    if view == "summary":
        return build_project_summary(project_id, access.tenant_id).model_dump(mode="json")
    return asdict(project)


@router.get("/{project_id}/summary")
def get_project_summary(
    project_id: str,
    access: AccessContext = Depends(require_permission("project:view")),
) -> dict:
    return build_project_summary(project_id, access.tenant_id).model_dump(mode="json")


@router.get("/{project_id}/runs")
def list_project_runs(
    project_id: str,
    access: AccessContext = Depends(require_permission("project:view")),
) -> dict:
    project = _project_repo.get_project(project_id)
    if project is None:
        raise NotFoundError("Project not found.", details={"project_id": project_id})
    if project.tenant_id != access.tenant_id:
        raise ForbiddenError("Project belongs to a different tenant.", details={"project_id": project_id})
    runs = _run_repo.list_runs_by_project(project_id=project_id, tenant_id=access.tenant_id)
    return {"runs": [asdict(run) for run in runs], "count": len(runs), "project_id": project_id}


@router.get("/{project_id}/briefs")
def list_project_briefs(
    project_id: str,
    access: AccessContext = Depends(require_permission("project:view")),
) -> dict:
    project = _project_repo.get_project(project_id)
    if project is None:
        raise NotFoundError("Project not found.", details={"project_id": project_id})
    if project.tenant_id != access.tenant_id:
        raise ForbiddenError("Project belongs to a different tenant.", details={"project_id": project_id})
    briefs = _brief_repo.list_briefs_by_project(project_id=project_id, tenant_id=access.tenant_id)
    return {"briefs": [asdict(brief) for brief in briefs], "count": len(briefs), "project_id": project_id}


@router.post("/{project_id}/run-search")
def run_search(
    project_id: str,
    req: ProjectSearchRunRequest,
    access: AccessContext = Depends(require_permission("search:run")),
    ats=Depends(get_ats_adapter),
) -> dict:
    workflow = SearchWorkflow(
        parser=JobParserService(llm=_llm),
        candidate_service=CandidateService(ats_adapter=ats, candidate_repository=get_candidate_repository()),
        ranking_service=RankingService(),
        brief_service=BriefService(llm=_llm),
    )
    result = workflow.run(
        project_id=project_id,
        jd_text=req.jd_text,
        user=access,
        candidate_source=req.candidate_source,
        ats_adapter=ats,
    )
    return result.model_dump(mode="json")


@router.get("/{project_id}/runs/{run_id}/results")
def get_run_results(
    project_id: str,
    run_id: str,
    access: AccessContext = Depends(require_permission("project:view")),
) -> dict:
    project = _project_repo.get_project(project_id)
    if project is None or project.tenant_id != access.tenant_id:
        raise NotFoundError("Project not found.", details={"project_id": project_id})
    run = _run_repo.get_run(run_id)
    if run is None or run.project_id != project_id or run.tenant_id != access.tenant_id:
        raise NotFoundError("Search run not found.", details={"run_id": run_id})
    snapshot = _snapshot_repo.get_snapshot_by_run(run_id=run_id, tenant_id=access.tenant_id)
    if snapshot is None:
        raise NotFoundError("Search result snapshot not found.", details={"run_id": run_id})
    return asdict(snapshot)


@router.get("/{project_id}/audit")
def get_project_audit(
    project_id: str,
    access: AccessContext = Depends(require_permission("project:view")),
    action: str | None = Query(default=None),
    resource_type: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict:
    project = _project_repo.get_project(project_id)
    if project is None or project.tenant_id != access.tenant_id:
        raise NotFoundError("Project not found.", details={"project_id": project_id})
    events = _audit_repo.list_events(
        tenant_id=access.tenant_id,
        project_id=project_id,
        action=action,
        resource_type=resource_type,
        limit=limit,
        offset=offset,
    )
    timeline = [
        {
            "audit_id": event.get("audit_id"),
            "timestamp": event.get("ts"),
            "user_id": event.get("actor_id"),
            "action": event.get("action"),
            "resource_type": event.get("resource_type"),
            "resource_id": event.get("resource_id"),
            "project_id": event.get("project_id"),
            "run_id": event.get("run_id"),
            "brief_id": event.get("brief_id"),
            "metadata": event.get("payload", {}),
        }
        for event in events
    ]
    return {"events": timeline, "count": len(timeline), "project_id": project_id, "offset": offset, "limit": limit}


@router.get("/{project_id}/review")
def get_project_review(
    project_id: str,
    access: AccessContext = Depends(require_permission("project:view")),
    audit_limit: int = Query(default=100, ge=1, le=500),
) -> dict:
    project = _project_repo.get_project(project_id)
    if project is None or project.tenant_id != access.tenant_id:
        raise NotFoundError("Project not found.", details={"project_id": project_id})
    summary = build_project_summary(project_id, access.tenant_id)
    runs = _run_repo.list_runs_by_project(project_id=project_id, tenant_id=access.tenant_id)
    latest_run: StoredSearchRun | None = max(runs, key=lambda item: item.started_at) if runs else None
    snapshots = _project_repo.list_project_snapshots(project_id, tenant_id=access.tenant_id)
    latest_snapshot = max(snapshots, key=lambda item: item.created_at) if snapshots else None
    briefs = _brief_repo.list_briefs_by_project(project_id=project_id, tenant_id=access.tenant_id)
    latest_brief: StoredBrief | None = max(briefs, key=lambda item: item.version) if briefs else None
    audit_events = _audit_repo.list_events(
        tenant_id=access.tenant_id,
        project_id=project_id,
        limit=audit_limit,
        offset=0,
    )
    audit_timeline = [
        {
            "audit_id": event.get("audit_id"),
            "timestamp": event.get("ts"),
            "user_id": event.get("actor_id"),
            "action": event.get("action"),
            "resource_type": event.get("resource_type"),
            "resource_id": event.get("resource_id"),
            "project_id": event.get("project_id"),
            "run_id": event.get("run_id"),
            "brief_id": event.get("brief_id"),
            "metadata": event.get("payload", {}),
        }
        for event in audit_events
    ]
    return {
        "project": asdict(project),
        "summary": summary.model_dump(mode="json"),
        "latest_run": asdict(latest_run) if latest_run else None,
        "latest_snapshot": asdict(latest_snapshot) if latest_snapshot else None,
        "briefs": [asdict(brief) for brief in briefs],
        "latest_brief": asdict(latest_brief) if latest_brief else None,
        "audit_timeline": audit_timeline,
    }


def build_project_summary(project_id: str, tenant_id: str) -> ProjectSummary:
    project = _project_repo.get_project(project_id)
    if project is None or project.tenant_id != tenant_id:
        raise NotFoundError("Project not found.", details={"project_id": project_id})
    runs = _run_repo.list_runs_by_project(project_id=project_id, tenant_id=tenant_id)
    briefs = _brief_repo.list_briefs_by_project(project_id=project_id, tenant_id=tenant_id)
    latest_run: StoredSearchRun | None = max(runs, key=lambda item: item.started_at) if runs else None
    latest_brief: StoredBrief | None = max(briefs, key=lambda item: item.version) if briefs else None
    exported_briefs = [brief for brief in briefs if brief.status == BriefStatus.EXPORTED]
    has_pending = latest_brief is not None and latest_brief.status == BriefStatus.PENDING_APPROVAL

    if latest_run is None and latest_brief is None:
        project_status = "draft"
    elif latest_brief is not None and latest_brief.status in {BriefStatus.REJECTED, BriefStatus.CHANGES_REQUESTED}:
        project_status = "attention_required"
    elif latest_brief is not None and latest_brief.status == BriefStatus.PENDING_APPROVAL:
        project_status = "awaiting_approval"
    elif exported_briefs:
        project_status = "exported"
    elif latest_brief is not None and latest_brief.status == BriefStatus.APPROVED:
        project_status = "approved"
    elif latest_brief is not None and latest_brief.status == BriefStatus.DRAFT:
        project_status = "in_progress"
    elif latest_run is not None and latest_run.run_status == SearchRunStatus.FAILED:
        project_status = "attention_required"
    elif latest_run is not None and latest_run.run_status in {
        SearchRunStatus.DRAFT,
        SearchRunStatus.PARSED,
        SearchRunStatus.SEARCHED,
        SearchRunStatus.RANKED,
        SearchRunStatus.BRIEF_GENERATED,
    }:
        project_status = "in_progress"
    elif latest_run is not None and latest_run.run_status == SearchRunStatus.COMPLETED:
        project_status = "completed"
    else:
        project_status = "in_progress"

    last_exported_at_raw = max((brief.exported_at for brief in exported_briefs if brief.exported_at), default=None)
    last_exported_at = datetime.fromisoformat(last_exported_at_raw) if last_exported_at_raw else None
    return ProjectSummary(
        project_id=project.project_id,
        tenant_id=project.tenant_id,
        project_name=project.project_name,
        client_name=project.client_name,
        role_title=project.role_title,
        project_status=project_status,
        latest_run_id=latest_run.run_id if latest_run else None,
        latest_run_status=latest_run.run_status if latest_run else None,
        latest_brief_id=latest_brief.brief_id if latest_brief else None,
        latest_brief_version=latest_brief.version if latest_brief else None,
        latest_brief_status=latest_brief.status if latest_brief else None,
        has_pending_approval=has_pending,
        last_exported_at=last_exported_at,
        created_at=datetime.fromisoformat(project.created_at),
        updated_at=datetime.fromisoformat(project.updated_at),
    )
