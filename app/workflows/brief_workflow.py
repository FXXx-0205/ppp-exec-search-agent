from __future__ import annotations

from uuid import uuid4

from app.core.audit import AuditEvent, AuditLogger
from app.core.exceptions import ConflictError, ForbiddenError, NotFoundError
from app.models.auth import MANAGER_ROLES, AccessContext
from app.models.workflow import BriefStatus
from app.repositories.factory import get_brief_repository, get_project_repository, get_search_run_repository
from app.repositories.interfaces import StoredBrief


class BriefWorkflow:
    def __init__(self, *, brief_repo=None, project_repo=None, run_repo=None, audit_logger: AuditLogger | None = None):
        self.brief_repo = brief_repo or get_brief_repository()
        self.project_repo = project_repo or get_project_repository()
        self.run_repo = run_repo or get_search_run_repository()
        self.audit = audit_logger or AuditLogger()

    def submit_for_approval(self, brief_id: str, user: AccessContext, notes: str | None = None):
        brief = self._get_tenant_brief(brief_id, user)
        if brief.status not in {BriefStatus.DRAFT, BriefStatus.CHANGES_REQUESTED}:
            raise ConflictError("Brief cannot be submitted from the current status.", details={"brief_id": brief_id, "status": brief.status})
        updated = self.brief_repo.submit_for_approval(brief_id, submitted_by=user.actor.user_id, notes=notes)
        if updated is None:
            raise NotFoundError("Brief not found.", details={"brief_id": brief_id})
        self._audit("brief_submitted_for_approval", "submit_for_approval", updated, user, notes)
        return updated

    def approve(self, brief_id: str, user: AccessContext, notes: str | None = None):
        self._require_manager(user)
        brief = self._get_tenant_brief(brief_id, user)
        if brief.status != BriefStatus.PENDING_APPROVAL:
            raise ConflictError("Only pending briefs can be approved.", details={"brief_id": brief_id, "status": brief.status})
        updated = self.brief_repo.approve_brief(brief_id, approved_by=user.actor.user_id, notes=notes)
        if updated is None:
            raise NotFoundError("Brief not found.", details={"brief_id": brief_id})
        self._audit("brief_approved", "approve", updated, user, notes)
        return updated

    def reject(self, brief_id: str, user: AccessContext, notes: str | None = None):
        self._require_manager(user)
        brief = self._get_tenant_brief(brief_id, user)
        if brief.status != BriefStatus.PENDING_APPROVAL:
            raise ConflictError("Only pending briefs can be rejected.", details={"brief_id": brief_id, "status": brief.status})
        updated = self.brief_repo.reject_brief(brief_id, rejected_by=user.actor.user_id, notes=notes)
        if updated is None:
            raise NotFoundError("Brief not found.", details={"brief_id": brief_id})
        self._audit("brief_rejected", "reject", updated, user, notes)
        return updated

    def request_changes(self, brief_id: str, user: AccessContext, notes: str | None = None):
        self._require_manager(user)
        brief = self._get_tenant_brief(brief_id, user)
        if brief.status != BriefStatus.PENDING_APPROVAL:
            raise ConflictError(
                "Only pending briefs can move to changes requested.",
                details={"brief_id": brief_id, "status": brief.status},
            )
        updated = self.brief_repo.request_changes(brief_id, requested_by=user.actor.user_id, notes=notes)
        if updated is None:
            raise NotFoundError("Brief not found.", details={"brief_id": brief_id})
        self._audit("brief_changes_requested", "request_changes", updated, user, notes)
        return updated

    def export(self, brief_id: str, user: AccessContext, export_format: str = "md"):
        self._require_manager(user)
        brief = self._get_tenant_brief(brief_id, user)
        if brief.status != BriefStatus.APPROVED:
            raise ConflictError("Only approved briefs can be exported.", details={"brief_id": brief_id, "status": brief.status})
        updated = self.brief_repo.mark_exported(brief_id, exported_by=user.actor.user_id)
        if updated is None:
            raise NotFoundError("Brief not found.", details={"brief_id": brief_id})
        self._audit("brief_exported", "export", updated, user, export_format)
        return {"brief_id": updated.brief_id, "content": updated.content, "export_format": export_format, "status": updated.status}

    def create_revision_from_changes_request(self, source_brief_id: str, user: AccessContext, content: str | None = None):
        source = self._get_tenant_brief(source_brief_id, user)
        if source.status != BriefStatus.CHANGES_REQUESTED:
            raise ConflictError(
                "Only briefs in changes_requested status can create a revision.",
                details={"brief_id": source_brief_id, "status": source.status},
            )
        revision = self.brief_repo.create_revision(
            source_brief_id,
            new_brief_id=f"brief_{uuid4().hex[:12]}",
            created_by=user.actor.user_id,
            content=content,
        )
        if revision is None:
            raise NotFoundError("Brief not found.", details={"brief_id": source_brief_id})
        self._audit("brief_revision_created", "create_revision", revision, user, source.change_request_notes or source.rejection_notes)
        return revision

    def _get_tenant_brief(self, brief_id: str, user: AccessContext) -> StoredBrief:
        brief = self.brief_repo.get_brief(brief_id)
        if brief is None:
            raise NotFoundError("Brief not found.", details={"brief_id": brief_id})
        if brief.tenant_id != user.tenant_id:
            raise ForbiddenError("Brief belongs to a different tenant.", details={"brief_id": brief_id})
        return brief

    def _require_manager(self, user: AccessContext) -> None:
        if user.actor.role not in MANAGER_ROLES:
            raise ForbiddenError("You do not have permission to perform this action.", details={"role": user.actor.role})

    def _audit(self, event_type: str, action: str, brief: StoredBrief, user: AccessContext, notes: str | None) -> None:
        self.audit.log(
            AuditEvent(
                event_type=event_type,
                request_id=brief.brief_id,
                action=action,
                resource_type="brief",
                resource_id=brief.brief_id,
                payload={"status": brief.status, "notes": notes},
                tenant_id=user.tenant_id,
                project_id=brief.project_id,
                run_id=brief.run_id,
                brief_id=brief.brief_id,
                actor_id=user.actor.user_id,
            )
        )
