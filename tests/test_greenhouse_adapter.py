from __future__ import annotations

import json

import httpx

from app.adapters.greenhouse import GreenhouseATSAdapter


def test_greenhouse_adapter_search_candidates_maps_profiles() -> None:
    call_count = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/candidates")
        assert request.url.params["per_page"] == "100"
        call_count["count"] += 1
        return httpx.Response(
            200,
            json=[
                {
                    "id": 123,
                    "first_name": "Jamie",
                    "last_name": "Ng",
                    "title": "Infrastructure Portfolio Manager",
                    "company": "Example Asset Management",
                    "email_addresses": [{"value": "jamie@example.com", "type": "work"}],
                    "addresses": [{"value": "Sydney, Australia", "type": "work"}],
                    "application_ids": [9001, 9002],
                    "attachments": [{"type": "resume", "url": "https://example.com/resume.pdf"}],
                    "tags": ["infrastructure", "super fund"],
                    "created_at": "2026-03-14T00:00:00Z",
                    "updated_at": "2026-03-14T01:00:00Z",
                },
                {
                    "id": 456,
                    "first_name": "Alex",
                    "last_name": "Stone",
                    "title": "Software Engineer",
                    "created_at": "2026-03-14T00:00:00Z",
                    "updated_at": "2026-03-14T01:00:00Z",
                },
            ],
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = GreenhouseATSAdapter(api_token="test-token", client=client)

    profiles = adapter.search_candidates(
        {"tenant_id": "tenant_greenhouse", "keywords": ["infrastructure"], "required_skills": []}
    )

    assert len(profiles) == 1
    assert profiles[0].candidate_id == "123"
    assert profiles[0].source_system == "greenhouse"
    assert profiles[0].current_company == "Example Asset Management"
    assert profiles[0].primary_email == "jamie@example.com"
    assert profiles[0].location == "Sydney, Australia"
    assert profiles[0].attachment_count == 1
    assert call_count["count"] == 1


def test_greenhouse_adapter_candidate_documents_maps_attachments() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/candidates/123")
        return httpx.Response(
            200,
            content=json.dumps(
                {
                    "id": 123,
                    "attachments": [
                        {
                            "filename": "resume.pdf",
                            "url": "https://example.com/resume.pdf",
                            "type": "resume",
                            "created_at": "2026-03-14T01:00:00Z",
                        }
                    ],
                }
            ),
            headers={"Content-Type": "application/json"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = GreenhouseATSAdapter(api_token="test-token", client=client)

    documents = adapter.get_candidate_documents("123")

    assert len(documents) == 1
    assert documents[0].content_type == "resume"


def test_greenhouse_adapter_fetches_multiple_pages_until_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        page = request.url.params.get("page", "1")
        if page == "1":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 101,
                        "first_name": "Taylor",
                        "last_name": "West",
                        "title": "Infrastructure Analyst",
                        "updated_at": "2026-03-14T01:00:00Z",
                    }
                ],
                headers={
                    "Link": '<https://harvest.greenhouse.io/v1/candidates?page=2&per_page=2>; rel="next"'
                },
            )
        return httpx.Response(
            200,
            json=[
                {
                    "id": 102,
                    "first_name": "Morgan",
                    "last_name": "Lee",
                    "title": "Infrastructure Portfolio Manager",
                    "updated_at": "2026-03-14T02:00:00Z",
                }
            ],
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = GreenhouseATSAdapter(api_token="test-token", client=client, per_page=2, max_pages=3)

    profiles = adapter.search_candidates(
        {"tenant_id": "tenant_greenhouse", "keywords": ["infrastructure"], "required_skills": [], "limit": 2}
    )

    assert len(profiles) == 2
    assert [profile.candidate_id for profile in profiles] == ["101", "102"]


def test_greenhouse_adapter_passes_incremental_filters_to_api() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json=[])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = GreenhouseATSAdapter(api_token="test-token", client=client)

    adapter.search_candidates(
        {
            "tenant_id": "tenant_greenhouse",
            "keywords": [],
            "required_skills": [],
            "created_after": "2026-03-01T00:00:00Z",
            "updated_after": "2026-03-10T00:00:00Z",
            "candidate_ids": [1, 2, 3],
        }
    )

    assert captured["created_after"] == "2026-03-01T00:00:00Z"
    assert captured["updated_after"] == "2026-03-10T00:00:00Z"
    assert captured["candidate_ids"] == "1,2,3"
