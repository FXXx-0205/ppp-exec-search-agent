from __future__ import annotations

from app.api import routes_brief, routes_projects
from app.config import settings
from app.core.audit import AuditLogger
from app.repositories.audit_repo import JsonlAuditRepository
from app.repositories.brief_repo import BriefRepo
from app.repositories.project_repo import ProjectRepo
from app.repositories.search_result_snapshot_repo import SearchResultSnapshotRepo
from app.repositories.search_run_repo import SearchRunRepo
from app.workflows.brief_workflow import BriefWorkflow


def test_project_and_workflow_api_happy_path(client, monkeypatch, tmp_path) -> None:
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

    create_response = client.post(
        "/projects",
        json={"project_name": "Infra PM Search", "client_name": "Example Asset Management", "role_title": "Infrastructure Portfolio Manager", "metadata": {}},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_api", "x-user-id": "researcher_1"},
    )
    assert create_response.status_code == 200
    project_id = create_response.json()["project_id"]

    run_response = client.post(
        f"/projects/{project_id}/run-search",
        json={"jd_text": "We are seeking a senior infrastructure portfolio manager in Australia", "candidate_source": "local_first"},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_api", "x-user-id": "researcher_1"},
    )
    assert run_response.status_code == 200
    brief_id = run_response.json()["brief_id"]
    run_id = run_response.json()["run_id"]
    assert run_response.json()["run_status"] == "completed"

    submit_response = client.post(
        f"/brief/{brief_id}/submit",
        json={"notes": "ready for review"},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_api", "x-user-id": "researcher_1"},
    )
    approve_response = client.post(
        f"/brief/{brief_id}/approve",
        json={"notes": "approved"},
        headers={"x-user-role": "consultant", "x-tenant-id": "tenant_api", "x-user-id": "manager_1"},
    )
    export_response = client.post(
        f"/brief/{brief_id}/export",
        json={"export_format": "md"},
        headers={"x-user-role": "consultant", "x-tenant-id": "tenant_api", "x-user-id": "manager_1"},
    )

    assert submit_response.status_code == 200
    assert submit_response.json()["status"] == "pending_approval"
    assert approve_response.status_code == 200
    assert approve_response.json()["status"] == "approved"
    assert export_response.status_code == 200
    assert export_response.json()["status"] == "exported"

    run_detail = run_repo.get_run(run_id)
    assert run_detail is not None
    assert run_detail.run_status == "completed"

    results_response = client.get(
        f"/projects/{project_id}/runs/{run_id}/results",
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_api", "x-user-id": "researcher_1"},
    )
    summary_response = client.get(
        f"/projects/{project_id}/summary",
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_api", "x-user-id": "researcher_1"},
    )

    assert results_response.status_code == 200
    assert results_response.json()["run_id"] == run_id
    assert summary_response.status_code == 200
    assert summary_response.json()["project_status"] == "exported"


def test_researcher_cannot_approve_or_export_new_brief_endpoints(client, monkeypatch, tmp_path) -> None:
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
        json={"project_name": "Infra PM Search"},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_api", "x-user-id": "researcher_1"},
    ).json()["project_id"]
    brief_id = client.post(
        f"/projects/{project_id}/run-search",
        json={"jd_text": "We are seeking a senior infrastructure portfolio manager in Australia"},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_api", "x-user-id": "researcher_1"},
    ).json()["brief_id"]
    client.post(
        f"/brief/{brief_id}/submit",
        json={"notes": "ready"},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_api", "x-user-id": "researcher_1"},
    )

    approve_response = client.post(
        f"/brief/{brief_id}/approve",
        json={"notes": "no"},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_api", "x-user-id": "researcher_1"},
    )
    export_response = client.post(
        f"/brief/{brief_id}/export",
        json={"export_format": "md"},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_api", "x-user-id": "researcher_1"},
    )

    assert approve_response.status_code == 403
    assert export_response.status_code == 403


def test_brief_revision_and_project_audit_api(client, monkeypatch, tmp_path) -> None:
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
    monkeypatch.setattr(routes_brief, "_repo", brief_repo)
    monkeypatch.setattr(routes_brief, "_audit", audit_logger)
    monkeypatch.setattr(routes_brief, "_brief_workflow", BriefWorkflow(brief_repo=brief_repo, project_repo=project_repo, run_repo=run_repo, audit_logger=audit_logger))

    project_id = client.post(
        "/projects",
        json={"project_name": "Infra PM Search"},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_audit", "x-user-id": "researcher_1"},
    ).json()["project_id"]
    run_body = client.post(
        f"/projects/{project_id}/run-search",
        json={"jd_text": "We are seeking a senior infrastructure portfolio manager in Australia"},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_audit", "x-user-id": "researcher_1"},
    ).json()
    brief_id = run_body["brief_id"]
    client.post(
        f"/brief/{brief_id}/submit",
        json={"notes": "ready"},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_audit", "x-user-id": "researcher_1"},
    )
    client.post(
        f"/brief/{brief_id}/request-changes",
        json={"notes": "tighten evidence"},
        headers={"x-user-role": "consultant", "x-tenant-id": "tenant_audit", "x-user-id": "manager_1"},
    )
    revision_response = client.post(
        f"/brief/{brief_id}/create-revision",
        json={"content": "# Revised"},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_audit", "x-user-id": "researcher_1"},
    )
    audit_response = client.get(
        f"/projects/{project_id}/audit",
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_audit", "x-user-id": "researcher_1"},
    )
    cross_tenant_response = client.get(
        f"/projects/{project_id}/audit",
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_other", "x-user-id": "researcher_2"},
    )

    assert revision_response.status_code == 200
    assert revision_response.json()["version"] == 2
    assert revision_response.json()["previous_brief_id"] == brief_id
    assert audit_response.status_code == 200
    assert audit_response.json()["count"] >= 1
    assert any(event["brief_id"] == brief_id for event in audit_response.json()["events"])
    assert cross_tenant_response.status_code == 404
