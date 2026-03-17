from __future__ import annotations

from fastapi.testclient import TestClient


def test_brief_list_filters_by_tenant(client: TestClient) -> None:
    client.post(
        "/brief/generate",
        json={"role_spec": {"title": "Role A"}, "candidate_ids": []},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_a", "x-user-id": "user_a"},
    )
    client.post(
        "/brief/generate",
        json={"role_spec": {"title": "Role B"}, "candidate_ids": []},
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_b", "x-user-id": "user_b"},
    )

    response = client.get("/brief", headers={"x-user-role": "researcher", "x-tenant-id": "tenant_a"})

    body = response.json()
    assert response.status_code == 200
    assert body["tenant_id"] == "tenant_a"
    assert all(item["tenant_id"] == "tenant_a" for item in body["briefs"])


def test_brief_list_supports_pagination(client: TestClient) -> None:
    for idx in range(3):
        client.post(
            "/brief/generate",
            json={"role_spec": {"title": f"Role {idx}"}, "candidate_ids": []},
            headers={"x-user-role": "researcher", "x-tenant-id": "tenant_page", "x-user-id": f"user_{idx}"},
        )

    response = client.get(
        "/brief?limit=1&offset=1",
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_page"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["limit"] == 1
    assert body["offset"] == 1
    assert body["count"] == 1


def test_brief_generate_accepts_structured_role_spec_fields(client: TestClient) -> None:
    response = client.post(
        "/briefs/generate",
        json={
            "role_spec": {
                "title": {"title": "Infrastructure Portfolio Manager"},
                "search_keywords": ["infrastructure", "portfolio manager"],
                "sector": {"name": "Infrastructure"},
                "location": {"primary": ["Sydney", "Melbourne"], "country": "Australia"},
            },
            "candidate_ids": [],
        },
        headers={"x-user-role": "researcher", "x-tenant-id": "tenant_structured", "x-user-id": "user_structured"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["brief_id"]
