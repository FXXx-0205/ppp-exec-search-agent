from __future__ import annotations

from datetime import datetime, timezone

from app.adapters.ats import CandidateProfile
from app.repositories.sqlite_repo import SqliteCandidateRepository
from app.services.candidate_sync_service import CandidateSyncService


class StubATSAdapter:
    def search_candidates(self, filters, page_token=None):
        return [
            CandidateProfile(
                tenant_id=filters["tenant_id"],
                candidate_id="cand_1",
                full_name="Jamie Ng",
                current_title="Infrastructure Portfolio Manager",
                source_system="greenhouse",
                source_id="cand_1",
                synced_at=datetime.now(timezone.utc),
                current_company="Example Asset Management",
                primary_email="jamie@example.com",
                location="Sydney, Australia",
                application_ids=["9001"],
                tag_names=["infrastructure"],
                attachment_count=1,
            )
        ]

    def get_candidate(self, candidate_id):
        return None

    def get_candidate_documents(self, candidate_id):
        return []

    def upsert_shortlist_assessment(self, candidate_id, project_id, assessment):
        return None


def test_candidate_sync_service_persists_candidates(tmp_path) -> None:
    repo = SqliteCandidateRepository(f"sqlite:///{tmp_path}/sync.db")
    service = CandidateSyncService(ats_adapter=StubATSAdapter(), candidate_repository=repo)

    synced = service.sync_candidates(tenant_id="tenant_sync", provider_filters={"updated_after": "2026-03-10T00:00:00Z"})
    stored = repo.list(tenant_id="tenant_sync")

    assert len(synced) == 1
    assert len(stored) == 1
    assert stored[0].full_name == "Jamie Ng"
