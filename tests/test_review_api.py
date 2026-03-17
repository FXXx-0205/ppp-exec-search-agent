from __future__ import annotations

from app.api import routes_projects
from app.config import settings
from app.repositories.project_repo import ProjectRepo


def test_project_review_aggregate_handles_empty_project(client, monkeypatch, tmp_path) -> None:
    settings.brief_storage_dir = str(tmp_path / "briefs")
    project_repo = ProjectRepo(root_dir=settings.brief_storage_dir)
    monkeypatch.setattr(routes_projects, "_project_repo", project_repo)
    project_id = client.post(
        "/projects",
        json={"project_name": "Empty Review Project"},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_review", "x-user-id": "user_1"},
    ).json()["project_id"]

    response = client.get(
        f"/projects/{project_id}/review",
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_review", "x-user-id": "user_1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["latest_run"] is None
    assert body["latest_snapshot"] is None
    assert body["latest_brief"] is None
    assert body["briefs"] == []


def test_project_review_aggregate_enforces_tenant_isolation(client, monkeypatch, tmp_path) -> None:
    settings.brief_storage_dir = str(tmp_path / "briefs")
    project_repo = ProjectRepo(root_dir=settings.brief_storage_dir)
    monkeypatch.setattr(routes_projects, "_project_repo", project_repo)
    project_id = client.post(
        "/projects",
        json={"project_name": "Tenant A Project"},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_a", "x-user-id": "user_a"},
    ).json()["project_id"]

    response = client.get(
        f"/projects/{project_id}/review",
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_b", "x-user-id": "user_b"},
    )

    assert response.status_code == 404
