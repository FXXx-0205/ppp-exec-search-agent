from __future__ import annotations

from app.repositories.interfaces import StoredCandidate
from app.repositories.sqlite_repo import SqliteCandidateRepository


def test_sqlite_candidate_repository_filters_candidates(tmp_path) -> None:
    repo = SqliteCandidateRepository(f"sqlite:///{tmp_path}/candidates.db")
    repo.upsert_many(
        [
            StoredCandidate(
                tenant_id="tenant_repo",
                candidate_id="cand_1",
                full_name="Jamie Ng",
                current_title="Infrastructure Portfolio Manager",
                current_company="Example Asset Management",
                location="Sydney",
                primary_email="jamie@example.com",
                summary="Infrastructure leader",
                evidence=["Primary email on record: jamie@example.com"],
                source_system="greenhouse",
                source_id="cand_1",
                application_ids=["1"],
                tag_names=["infra"],
                attachment_count=1,
                synced_at="2026-03-14T00:00:00+00:00",
            ),
            StoredCandidate(
                tenant_id="tenant_repo",
                candidate_id="cand_2",
                full_name="Alex Stone",
                current_title="Software Engineer",
                current_company="Tech Co",
                location="Melbourne",
                primary_email="alex@example.com",
                summary="Software engineer",
                evidence=[],
                source_system="greenhouse",
                source_id="cand_2",
                application_ids=[],
                tag_names=["software"],
                attachment_count=0,
                synced_at="2026-03-14T00:00:01+00:00",
            ),
        ]
    )

    results = repo.list(tenant_id="tenant_repo", search_text="infrastructure", limit=10)

    assert len(results) == 1
    assert results[0].candidate_id == "cand_1"
