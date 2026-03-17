from __future__ import annotations

from app.api import routes_brief, routes_projects
from app.config import settings
from app.repositories.brief_repo import BriefRepo
from app.repositories.project_repo import ProjectRepo
from app.repositories.search_result_snapshot_repo import SearchResultSnapshotRepo
from app.repositories.search_run_repo import SearchRunRepo
from app.workflows.brief_workflow import BriefWorkflow


def _seed_exported_brief(client, monkeypatch, tmp_path, tenant_id: str = "tenant_artifact") -> str:
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
        json={"project_name": "Artifact Project"},
        headers={"x-user-role": "researcher", "x-tenant-id": tenant_id, "x-user-id": "researcher_1"},
    ).json()["project_id"]
    run_body = client.post(
        f"/projects/{project_id}/run-search",
        json={"jd_text": "We are seeking a senior infrastructure portfolio manager in Australia"},
        headers={"x-user-role": "researcher", "x-tenant-id": tenant_id, "x-user-id": "researcher_1"},
    ).json()
    brief_id = run_body["brief_id"]
    client.post(
        f"/briefs/{brief_id}/submit",
        json={"notes": "ready"},
        headers={"x-user-role": "researcher", "x-tenant-id": tenant_id, "x-user-id": "researcher_1"},
    )
    client.post(
        f"/briefs/{brief_id}/approve",
        json={"notes": "approved"},
        headers={"x-user-role": "consultant", "x-tenant-id": tenant_id, "x-user-id": "manager_1"},
    )
    client.post(
        f"/briefs/{brief_id}/export",
        json={"export_format": "md"},
        headers={"x-user-role": "consultant", "x-tenant-id": tenant_id, "x-user-id": "manager_1"},
    )
    return brief_id


def test_exported_brief_artifact_is_readable(client, monkeypatch, tmp_path) -> None:
    brief_id = _seed_exported_brief(client, monkeypatch, tmp_path)

    response = client.get(
        f"/briefs/{brief_id}/artifact",
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_artifact", "x-user-id": "researcher_1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "exported"
    assert body["content_type"] == "text/markdown"
    assert body["brief_id"] == brief_id
    assert body["version"] == 1
    assert body["exported_by"] == "manager_1"
    assert body["exported_at"] is not None
    assert body["content"]


def test_unexported_or_cross_tenant_brief_artifact_is_blocked(client, monkeypatch, tmp_path) -> None:
    brief_id = _seed_exported_brief(client, monkeypatch, tmp_path)

    unexported_project_id = client.post(
        "/projects",
        json={"project_name": "Draft Project"},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_artifact", "x-user-id": "researcher_1"},
    ).json()["project_id"]
    unexported_brief_id = client.post(
        f"/projects/{unexported_project_id}/run-search",
        json={"jd_text": "Need another infrastructure portfolio manager"},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_artifact", "x-user-id": "researcher_1"},
    ).json()["brief_id"]

    draft_response = client.get(
        f"/briefs/{unexported_brief_id}/artifact",
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_artifact", "x-user-id": "researcher_1"},
    )
    tenant_response = client.get(
        f"/briefs/{brief_id}/artifact",
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_other", "x-user-id": "researcher_2"},
    )

    assert draft_response.status_code == 409
    assert tenant_response.status_code == 403
