from __future__ import annotations

from app.api import routes_brief, routes_projects
from app.config import settings
from app.repositories.brief_repo import BriefRepo
from app.repositories.project_repo import ProjectRepo
from app.repositories.search_result_snapshot_repo import SearchResultSnapshotRepo
from app.repositories.search_run_repo import SearchRunRepo
from app.workflows.brief_workflow import BriefWorkflow


def test_briefs_plural_is_primary_and_brief_alias_still_works(client, monkeypatch, tmp_path) -> None:
    settings.brief_storage_dir = str(tmp_path / "briefs")
    project_repo = ProjectRepo(root_dir=settings.brief_storage_dir)
    run_repo = SearchRunRepo(root_dir=settings.brief_storage_dir)
    brief_repo = BriefRepo(root_dir=settings.brief_storage_dir)
    snapshot_repo = SearchResultSnapshotRepo(root_dir=settings.brief_storage_dir)
    monkeypatch.setattr(routes_projects, "_project_repo", project_repo)
    monkeypatch.setattr(routes_projects, "_run_repo", run_repo)
    monkeypatch.setattr(routes_projects, "_brief_repo", brief_repo)
    monkeypatch.setattr(routes_projects, "_snapshot_repo", snapshot_repo)
    monkeypatch.setattr(routes_brief, "_repo", brief_repo)
    monkeypatch.setattr(routes_brief, "_brief_workflow", BriefWorkflow(brief_repo=brief_repo, project_repo=project_repo, run_repo=run_repo))

    project_id = client.post(
        "/projects",
        json={"project_name": "Alias Project"},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_alias", "x-user-id": "researcher_1"},
    ).json()["project_id"]
    brief_id = client.post(
        f"/projects/{project_id}/run-search",
        json={"jd_text": "We are seeking a senior infrastructure portfolio manager in Australia"},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_alias", "x-user-id": "researcher_1"},
    ).json()["brief_id"]

    plural = client.get(
        f"/briefs/{brief_id}",
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_alias", "x-user-id": "researcher_1"},
    )
    alias = client.get(
        f"/brief/{brief_id}",
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_alias", "x-user-id": "researcher_1"},
    )

    assert plural.status_code == 200
    assert alias.status_code == 200
    assert plural.json()["brief_id"] == alias.json()["brief_id"]
