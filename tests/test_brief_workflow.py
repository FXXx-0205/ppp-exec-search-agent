from __future__ import annotations

import pytest

from app.core.exceptions import ConflictError, ForbiddenError
from app.models.auth import AccessContext, UserIdentity, UserRole
from app.models.workflow import BriefStatus
from app.repositories.brief_repo import BriefRepo
from app.repositories.interfaces import StoredBrief, StoredSearchProject
from app.repositories.project_repo import ProjectRepo
from app.repositories.search_run_repo import SearchRunRepo
from app.workflows.brief_workflow import BriefWorkflow


def _access(role: UserRole, *, tenant_id: str = "tenant_1", user_id: str = "user_1") -> AccessContext:
    return AccessContext(
        tenant_id=tenant_id,
        actor=UserIdentity(user_id=user_id, email=f"{user_id}@example.com", display_name=user_id, role=role),
    )


def _seed(tmp_path) -> tuple[BriefWorkflow, str]:
    brief_repo = BriefRepo(root_dir=str(tmp_path / "briefs"))
    project_repo = ProjectRepo(root_dir=str(tmp_path / "briefs"))
    run_repo = SearchRunRepo(root_dir=str(tmp_path / "briefs"))
    project_repo.create_project(
        StoredSearchProject(
            project_id="proj_1",
            tenant_id="tenant_1",
            project_name="Infra Search",
            client_name="Client",
            role_title="PM",
            status="draft",
            created_by="researcher_1",
            created_at="2026-03-14T00:00:00+00:00",
            updated_at="2026-03-14T00:00:00+00:00",
            metadata=None,
        )
    )
    brief_repo.create_brief_version(
        StoredBrief(
            brief_id="brief_1",
            project_id="proj_1",
            tenant_id="tenant_1",
            version=1,
            content="# Brief",
            status=BriefStatus.DRAFT,
            created_by="researcher_1",
            created_at="2026-03-14T00:00:00+00:00",
            updated_at="2026-03-14T00:00:00+00:00",
            role_spec={"title": "Infra PM"},
        )
    )
    return BriefWorkflow(brief_repo=brief_repo, project_repo=project_repo, run_repo=run_repo), "brief_1"


def test_brief_workflow_submit_approve_reject_and_request_changes(tmp_path) -> None:
    workflow, brief_id = _seed(tmp_path)

    submitted = workflow.submit_for_approval(brief_id, _access(UserRole.RESEARCHER, user_id="researcher_1"), "ready")
    assert submitted.status == BriefStatus.PENDING_APPROVAL

    requested = workflow.request_changes(brief_id, _access(UserRole.CONSULTANT, user_id="manager_1"), "revise")
    assert requested.status == BriefStatus.CHANGES_REQUESTED

    resubmitted = workflow.submit_for_approval(brief_id, _access(UserRole.RESEARCHER, user_id="researcher_1"), "updated")
    approved = workflow.approve(brief_id, _access(UserRole.CONSULTANT, user_id="manager_1"), "approved")
    assert resubmitted.status == BriefStatus.PENDING_APPROVAL
    assert approved.status == BriefStatus.APPROVED

    workflow2, brief_id2 = _seed(tmp_path / "other")
    workflow2.submit_for_approval(brief_id2, _access(UserRole.RESEARCHER, user_id="researcher_1"), "ready")
    rejected = workflow2.reject(brief_id2, _access(UserRole.ADMIN, user_id="admin_1"), "no")
    assert rejected.status == BriefStatus.REJECTED


def test_brief_workflow_rejects_illegal_transitions_and_researcher_permissions(tmp_path) -> None:
    workflow, brief_id = _seed(tmp_path)

    with pytest.raises(ForbiddenError):
        workflow.approve(brief_id, _access(UserRole.RESEARCHER, user_id="researcher_1"), "nope")

    with pytest.raises(ConflictError):
        workflow.export(brief_id, _access(UserRole.CONSULTANT, user_id="manager_1"))

    workflow.submit_for_approval(brief_id, _access(UserRole.RESEARCHER, user_id="researcher_1"), "ready")
    workflow.approve(brief_id, _access(UserRole.CONSULTANT, user_id="manager_1"), "ok")

    with pytest.raises(ConflictError):
        workflow.submit_for_approval(brief_id, _access(UserRole.RESEARCHER, user_id="researcher_1"), "again")


def test_brief_workflow_creates_revision_after_changes_requested(tmp_path) -> None:
    workflow, brief_id = _seed(tmp_path)
    workflow.submit_for_approval(brief_id, _access(UserRole.RESEARCHER, user_id="researcher_1"), "ready")
    changed = workflow.request_changes(brief_id, _access(UserRole.CONSULTANT, user_id="manager_1"), "add more evidence")
    revision = workflow.create_revision_from_changes_request(brief_id, _access(UserRole.RESEARCHER, user_id="researcher_1"), "# Revised Brief")
    resubmitted = workflow.submit_for_approval(revision.brief_id, _access(UserRole.RESEARCHER, user_id="researcher_1"), "revised")
    approved = workflow.approve(revision.brief_id, _access(UserRole.CONSULTANT, user_id="manager_1"), "approved")

    assert changed.status == BriefStatus.CHANGES_REQUESTED
    assert revision.version == 2
    assert revision.previous_brief_id == brief_id
    assert revision.change_request_source_brief_id == brief_id
    assert revision.change_request_notes == "add more evidence"
    assert resubmitted.status == BriefStatus.PENDING_APPROVAL
    assert approved.status == BriefStatus.APPROVED
