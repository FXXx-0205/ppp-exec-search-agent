from __future__ import annotations

from dataclasses import asdict
from uuid import uuid4

from fastapi import APIRouter, Depends, Query

from app.api.dependencies.auth import require_permission
from app.core.audit import AuditEvent, AuditLogger
from app.core.exceptions import ConflictError, ForbiddenError, NotFoundError
from app.llm.anthropic_client import ClaudeClient
from app.models.auth import AccessContext, ApprovalStatus
from app.models.search_request import (
    BriefActionRequest,
    BriefApprovalRequest,
    BriefExportRequest,
    BriefGenerateRequest,
    BriefRevisionRequest,
)
from app.models.workflow import BriefStatus
from app.repositories.factory import get_brief_repository
from app.repositories.interfaces import StoredBrief
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore
from app.services.brief_service import BriefService
from app.services.candidate_service import CandidateService
from app.services.ranking_service import RankingService
from app.workflows.brief_workflow import BriefWorkflow

router = APIRouter()

_llm = ClaudeClient()
_brief = BriefService(llm=_llm)
_candidates = CandidateService()
_ranker = RankingService()
_store = VectorStore()
_retriever = Retriever(store=_store)
_repo = get_brief_repository()
_audit = AuditLogger()
_brief_workflow = BriefWorkflow(brief_repo=_repo, audit_logger=_audit)


@router.post("/generate")
def generate(
    req: BriefGenerateRequest,
    access: AccessContext = Depends(require_permission("brief:generate")),
) -> dict:
    role_spec = req.role_spec or {"title": "Unspecified", "search_keywords": []}
    pool = _candidates.load_demo_candidates()
    by_id = {c.get("candidate_id"): c for c in pool}
    selected = [by_id[cid] for cid in (req.candidate_ids or []) if cid in by_id] or pool
    ranked = _ranker.score_candidates(role_spec, selected)
    retrieval_context = _retriever.retrieve_for_role(role_spec, top_k=5)
    generation_context = {
        "tenant_id": access.tenant_id,
        "project_id": req.project_id or access.project_id,
        "actor_id": access.actor.user_id,
    }
    out = _brief.generate_markdown(
        role_spec=role_spec,
        ranked_candidates=ranked,
        retrieval_context=retrieval_context,
        generation_context=generation_context,
    )
    created_at = out["generated_at"]
    project_id = req.project_id or access.project_id or f"project_ad_hoc_{uuid4().hex[:8]}"
    _repo.create_brief_version(
        StoredBrief(
            brief_id=out["brief_id"],
            project_id=project_id,
            tenant_id=access.tenant_id,
            version=1,
            content=out["markdown"],
            status=BriefStatus.DRAFT,
            created_by=access.actor.user_id,
            created_at=created_at,
            updated_at=created_at,
            role_spec=role_spec,
            markdown=out["markdown"],
            citations=out.get("citations", []),
            generated_at=out["generated_at"],
        )
    )
    _audit.log(
        AuditEvent(
            event_type="brief_generated",
            request_id=out["brief_id"],
            action="brief_generated",
            resource_type="brief",
            resource_id=out["brief_id"],
            payload={"citations": out.get("citations", []), "prompt": out.get("prompt"), "generation_context": generation_context},
            tenant_id=access.tenant_id,
            project_id=project_id,
            brief_id=out["brief_id"],
            actor_id=access.actor.user_id,
        )
    )
    return {**out, "approval_status": ApprovalStatus.DRAFT, "status": BriefStatus.DRAFT}


@router.get("")
def list_briefs(
    access: AccessContext = Depends(require_permission("brief:generate")),
    project_id: str | None = Query(default=None),
    approval_status: ApprovalStatus | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    briefs = _repo.list(
        tenant_id=access.tenant_id,
        project_id=project_id,
        approval_status=approval_status,
        limit=limit,
        offset=offset,
    )
    return {
        "briefs": [
            {
                "brief_id": brief.brief_id,
                "tenant_id": brief.tenant_id,
                "project_id": brief.project_id,
                "approval_status": brief.approval_status,
                "status": brief.status,
                "version": brief.version,
                "created_by": brief.created_by,
                "generated_at": brief.generated_at,
                "approved_by": brief.approved_by,
                "approved_at": brief.approved_at,
            }
            for brief in briefs
        ],
        "count": len(briefs),
        "tenant_id": access.tenant_id,
        "offset": offset,
        "limit": limit,
    }


@router.post("/approve/{brief_id}")
def approve(
    brief_id: str,
    req: BriefApprovalRequest,
    access: AccessContext = Depends(require_permission("brief:approve")),
) -> dict:
    existing = _repo.get(brief_id)
    if not existing:
        raise NotFoundError("Brief not found.", details={"brief_id": brief_id})
    if existing.tenant_id != access.tenant_id:
        raise ForbiddenError("Brief belongs to a different tenant.", details={"brief_id": brief_id})
    b = _repo.decide(brief_id, status=req.status, decided_by=access.actor.user_id, comment=req.comment)
    if not b:
        raise NotFoundError("Brief not found.", details={"brief_id": brief_id})
    _audit.log(
        AuditEvent(
            event_type="brief_decided",
            request_id=brief_id,
            payload={"approval_status": b.approval_status, "approval_notes": b.approval_notes},
            tenant_id=access.tenant_id,
            project_id=b.project_id,
            actor_id=access.actor.user_id,
        )
    )
    return {
        "brief_id": b.brief_id,
        "approval_status": b.approval_status,
        "status": b.status,
        "approved_by": b.approved_by,
        "approved_at": b.approved_at,
        "approval_notes": b.approval_notes,
    }


@router.get("/{brief_id}")
def get_brief(
    brief_id: str,
    access: AccessContext = Depends(require_permission("brief:generate")),
) -> dict:
    b = _repo.get(brief_id)
    if not b:
        raise NotFoundError("Brief not found.", details={"brief_id": brief_id})
    if b.tenant_id != access.tenant_id:
        raise ForbiddenError("Brief belongs to a different tenant.", details={"brief_id": brief_id})
    _audit.log(
        AuditEvent(
            event_type="brief_viewed",
            request_id=brief_id,
            payload={"approval_status": b.approval_status},
            tenant_id=access.tenant_id,
            project_id=b.project_id,
            actor_id=access.actor.user_id,
        )
    )
    return {
        "brief_id": b.brief_id,
        "markdown": b.markdown or b.content,
        "approval_status": b.approval_status,
        "status": b.status,
        "approved_by": b.approved_by,
        "approved_at": b.approved_at,
        "project_id": b.project_id,
        "tenant_id": b.tenant_id,
        "created_by": b.created_by,
        "generated_at": b.generated_at,
        "citations": b.citations or [],
    }


@router.get("/{brief_id}/export")
def export_brief(
    brief_id: str,
    access: AccessContext = Depends(require_permission("brief:export")),
) -> dict:
    """
    Client-facing export gate: must be approved.
    """
    b = _repo.get(brief_id)
    if not b:
        raise NotFoundError("Brief not found.", details={"brief_id": brief_id})
    if b.tenant_id != access.tenant_id:
        raise ForbiddenError("Brief belongs to a different tenant.", details={"brief_id": brief_id})
    if b.approval_status != ApprovalStatus.APPROVED:
        raise ForbiddenError("Brief is not approved for export.", details={"brief_id": brief_id})
    _audit.log(
        AuditEvent(
            event_type="brief_exported",
            request_id=brief_id,
            payload={"approval_status": b.approval_status},
            tenant_id=access.tenant_id,
            project_id=b.project_id,
            actor_id=access.actor.user_id,
        )
    )
    return {"brief_id": b.brief_id, "markdown": b.markdown, "exported": True}


@router.post("/{brief_id}/submit")
def submit_brief(
    brief_id: str,
    req: BriefActionRequest,
    access: AccessContext = Depends(require_permission("brief:submit")),
) -> dict:
    brief = _brief_workflow.submit_for_approval(brief_id, access, req.notes)
    return asdict(brief)


@router.post("/{brief_id}/approve")
def approve_brief(
    brief_id: str,
    req: BriefActionRequest,
    access: AccessContext = Depends(require_permission("brief:approve")),
) -> dict:
    brief = _brief_workflow.approve(brief_id, access, req.notes)
    return asdict(brief)


@router.post("/{brief_id}/reject")
def reject_brief(
    brief_id: str,
    req: BriefActionRequest,
    access: AccessContext = Depends(require_permission("brief:approve")),
) -> dict:
    brief = _brief_workflow.reject(brief_id, access, req.notes)
    return asdict(brief)


@router.post("/{brief_id}/request-changes")
def request_changes(
    brief_id: str,
    req: BriefActionRequest,
    access: AccessContext = Depends(require_permission("brief:approve")),
) -> dict:
    brief = _brief_workflow.request_changes(brief_id, access, req.notes)
    return asdict(brief)


@router.post("/{brief_id}/export")
def export_brief_post(
    brief_id: str,
    req: BriefExportRequest,
    access: AccessContext = Depends(require_permission("brief:export")),
) -> dict:
    return _brief_workflow.export(brief_id, access, req.export_format)


@router.post("/{brief_id}/create-revision")
def create_revision(
    brief_id: str,
    req: BriefRevisionRequest,
    access: AccessContext = Depends(require_permission("brief:generate")),
) -> dict:
    brief = _brief_workflow.create_revision_from_changes_request(brief_id, access, req.content)
    return asdict(brief)


@router.get("/{brief_id}/artifact")
def get_brief_artifact(
    brief_id: str,
    access: AccessContext = Depends(require_permission("brief:generate")),
) -> dict:
    brief = _repo.get_brief(brief_id)
    if not brief:
        raise NotFoundError("Brief not found.", details={"brief_id": brief_id})
    if brief.tenant_id != access.tenant_id:
        raise ForbiddenError("Brief belongs to a different tenant.", details={"brief_id": brief_id})
    if brief.status != BriefStatus.EXPORTED:
        raise ConflictError("Brief artifact is only available after export.", details={"brief_id": brief_id, "status": brief.status})
    return {
        "brief_id": brief.brief_id,
        "project_id": brief.project_id,
        "version": brief.version,
        "status": brief.status,
        "exported_at": brief.exported_at,
        "exported_by": brief.exported_by,
        "content_type": "text/markdown",
        "content": brief.content,
    }
