from __future__ import annotations

from app.api.routes_projects import build_project_summary
from app.models.workflow import BriefStatus, SearchRunStatus
from app.repositories.brief_repo import BriefRepo
from app.repositories.interfaces import StoredBrief, StoredSearchProject, StoredSearchRun
from app.repositories.project_repo import ProjectRepo
from app.repositories.search_run_repo import SearchRunRepo


def test_project_summary_maps_run_and_brief_statuses(tmp_path, monkeypatch) -> None:
    from app.api import routes_projects

    project_repo = ProjectRepo(root_dir=str(tmp_path / "briefs"))
    run_repo = SearchRunRepo(root_dir=str(tmp_path / "briefs"))
    brief_repo = BriefRepo(root_dir=str(tmp_path / "briefs"))

    monkeypatch.setattr(routes_projects, "_project_repo", project_repo)
    monkeypatch.setattr(routes_projects, "_run_repo", run_repo)
    monkeypatch.setattr(routes_projects, "_brief_repo", brief_repo)

    project_repo.create_project(
        StoredSearchProject(
            project_id="proj_1",
            tenant_id="tenant_1",
            project_name="Infra Search",
            client_name="Client",
            role_title="PM",
            status="draft",
            created_by="user_1",
            created_at="2026-03-14T00:00:00+00:00",
            updated_at="2026-03-14T00:00:00+00:00",
            metadata=None,
        )
    )
    assert build_project_summary("proj_1", "tenant_1").project_status == "draft"

    run_repo.create_run(
        StoredSearchRun(
            run_id="run_1",
            project_id="proj_1",
            tenant_id="tenant_1",
            input_jd="JD",
            parsed_role_json={},
            candidate_source="local",
            result_count=1,
            run_status=SearchRunStatus.RANKED,
            started_at="2026-03-14T01:00:00+00:00",
            completed_at=None,
            failed_at=None,
            error_message=None,
            created_by="user_1",
            metadata=None,
        )
    )
    assert build_project_summary("proj_1", "tenant_1").project_status == "in_progress"

    brief_repo.create_brief_version(
        StoredBrief(
            brief_id="brief_1",
            project_id="proj_1",
            tenant_id="tenant_1",
            version=1,
            content="# Brief",
            status=BriefStatus.PENDING_APPROVAL,
            created_by="user_1",
            created_at="2026-03-14T02:00:00+00:00",
            updated_at="2026-03-14T02:00:00+00:00",
            role_spec={"title": "PM"},
        )
    )
    assert build_project_summary("proj_1", "tenant_1").project_status == "awaiting_approval"

    brief_repo.approve_brief("brief_1", approved_by="manager_1", notes="ok")
    assert build_project_summary("proj_1", "tenant_1").project_status == "approved"

    brief_repo.mark_exported("brief_1", exported_by="manager_1")
    assert build_project_summary("proj_1", "tenant_1").project_status == "exported"


def test_project_summary_ignores_superseded_pending_brief_for_has_pending(tmp_path, monkeypatch) -> None:
    from app.api import routes_projects

    project_repo = ProjectRepo(root_dir=str(tmp_path / "briefs"))
    run_repo = SearchRunRepo(root_dir=str(tmp_path / "briefs"))
    brief_repo = BriefRepo(root_dir=str(tmp_path / "briefs"))

    monkeypatch.setattr(routes_projects, "_project_repo", project_repo)
    monkeypatch.setattr(routes_projects, "_run_repo", run_repo)
    monkeypatch.setattr(routes_projects, "_brief_repo", brief_repo)

    project_repo.create_project(
        StoredSearchProject(
            project_id="proj_2",
            tenant_id="tenant_1",
            project_name="Revision Chain",
            client_name="Client",
            role_title="PM",
            status="draft",
            created_by="user_1",
            created_at="2026-03-14T00:00:00+00:00",
            updated_at="2026-03-14T00:00:00+00:00",
            metadata=None,
        )
    )
    brief_repo.create_brief_version(
        StoredBrief(
            brief_id="brief_pending_v1",
            project_id="proj_2",
            tenant_id="tenant_1",
            version=1,
            content="# Pending",
            status=BriefStatus.PENDING_APPROVAL,
            created_by="user_1",
            created_at="2026-03-14T01:00:00+00:00",
            updated_at="2026-03-14T01:00:00+00:00",
            role_spec={"title": "PM"},
        )
    )
    brief_repo.create_brief_version(
        StoredBrief(
            brief_id="brief_exported_v2",
            project_id="proj_2",
            tenant_id="tenant_1",
            version=2,
            content="# Exported",
            status=BriefStatus.EXPORTED,
            created_by="user_1",
            created_at="2026-03-14T02:00:00+00:00",
            updated_at="2026-03-14T02:10:00+00:00",
            previous_brief_id="brief_pending_v1",
            supersedes_brief_id="brief_pending_v1",
            exported_by="manager_1",
            exported_at="2026-03-14T02:10:00+00:00",
            role_spec={"title": "PM"},
        )
    )

    summary = build_project_summary("proj_2", "tenant_1")

    assert summary.latest_brief_status == "exported"
    assert summary.project_status == "exported"
    assert summary.has_pending_approval is False
