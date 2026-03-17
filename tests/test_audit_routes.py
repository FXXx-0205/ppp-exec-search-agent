from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import routes_projects
from app.config import settings
from app.core.audit import AuditLogger
from app.repositories.audit_repo import JsonlAuditRepository
from app.repositories.brief_repo import BriefRepo
from app.repositories.project_repo import ProjectRepo
from app.repositories.search_result_snapshot_repo import SearchResultSnapshotRepo
from app.repositories.search_run_repo import SearchRunRepo


def test_audit_list_returns_tenant_scoped_events(client: TestClient) -> None:
    client.post(
        "/brief/generate",
        json={"role_spec": {"title": "Role A"}, "candidate_ids": []},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_a", "x-user-id": "user_a"},
    )

    response = client.get("/audit", headers={"x-user-role": "consultant", "x-tenant-id": "tenant_a"})

    body = response.json()
    assert response.status_code == 200
    assert body["count"] >= 1
    assert all(event["tenant_id"] == "tenant_a" for event in body["events"])


def test_audit_list_supports_filters_and_pagination(client: TestClient) -> None:
    client.post(
        "/brief/generate",
        json={"role_spec": {"title": "Role Filter"}, "candidate_ids": []},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_filter", "x-user-id": "user_filter"},
    )

    response = client.get(
        "/audit?event_type=brief_generated&limit=1&offset=0",
        headers={"x-user-role": "consultant", "x-tenant-id": "tenant_filter"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["limit"] == 1
    assert body["offset"] == 0
    assert body["count"] == 1
    assert all(event["event_type"] == "brief_generated" for event in body["events"])


def test_successful_run_records_started_and_completed_audit_events(client: TestClient, monkeypatch, tmp_path) -> None:
    settings.brief_storage_dir = str(tmp_path / "briefs")
    settings.audit_log_path = str(tmp_path / "audit.jsonl")
    project_repo = ProjectRepo(root_dir=settings.brief_storage_dir)
    run_repo = SearchRunRepo(root_dir=settings.brief_storage_dir)
    brief_repo = BriefRepo(root_dir=settings.brief_storage_dir)
    snapshot_repo = SearchResultSnapshotRepo(root_dir=settings.brief_storage_dir)
    audit_repo = JsonlAuditRepository(settings.audit_log_path)
    audit_logger = AuditLogger(repository=audit_repo)

    monkeypatch.setattr(routes_projects, "_project_repo", project_repo)
    monkeypatch.setattr(routes_projects, "_run_repo", run_repo)
    monkeypatch.setattr(routes_projects, "_brief_repo", brief_repo)
    monkeypatch.setattr(routes_projects, "_snapshot_repo", snapshot_repo)
    monkeypatch.setattr(routes_projects, "_audit_repo", audit_repo)
    monkeypatch.setattr(routes_projects, "_audit", audit_logger)

    project_id = client.post(
        "/projects",
        json={"project_name": "Audit Run Project"},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_run_audit", "x-user-id": "researcher_1"},
    ).json()["project_id"]
    client.post(
        f"/projects/{project_id}/run-search",
        json={"jd_text": "We are seeking a senior infrastructure portfolio manager in Australia"},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_run_audit", "x-user-id": "researcher_1"},
    )

    response = client.get(
        f"/projects/{project_id}/audit",
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_run_audit", "x-user-id": "researcher_1"},
    )

    actions = [event["action"] for event in response.json()["events"]]
    assert "start_search_run" in actions
    assert "run_completed" in actions
