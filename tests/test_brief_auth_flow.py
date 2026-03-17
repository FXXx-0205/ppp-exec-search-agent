from __future__ import annotations

from fastapi.testclient import TestClient


def test_researcher_cannot_approve_brief(client: TestClient) -> None:
    generate_response = client.post(
        "/brief/generate",
        json={"role_spec": {"title": "Head of Infra"}, "candidate_ids": []},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_a", "x-user-id": "researcher_1"},
    )
    brief_id = generate_response.json()["brief_id"]

    approve_response = client.post(
        f"/brief/approve/{brief_id}",
        json={"status": "approved", "comment": "ship it"},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_a", "x-user-id": "researcher_1"},
    )

    assert approve_response.status_code == 403
    assert approve_response.json()["error"]["code"] == "forbidden"


def test_consultant_can_approve_and_export_brief(client: TestClient) -> None:
    generate_response = client.post(
        "/brief/generate",
        json={"role_spec": {"title": "Head of Infra"}, "candidate_ids": []},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_b", "x-user-id": "researcher_2"},
    )
    brief_id = generate_response.json()["brief_id"]

    approve_response = client.post(
        f"/brief/approve/{brief_id}",
        json={"status": "approved", "comment": "Approved for client share"},
        headers={"x-user-role": "consultant", "x-tenant-id": "tenant_b", "x-user-id": "consultant_1"},
    )
    export_response = client.get(
        f"/brief/{brief_id}/export",
        headers={"x-user-role": "consultant", "x-tenant-id": "tenant_b", "x-user-id": "consultant_1"},
    )

    assert approve_response.status_code == 200
    assert approve_response.json()["approval_status"] == "approved"
    assert export_response.status_code == 200
