from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


@dataclass(frozen=True)
class CandidateProfile:
    tenant_id: str
    candidate_id: str
    full_name: str
    current_title: str
    source_system: str
    source_id: str
    synced_at: datetime
    current_company: str | None = None
    primary_email: str | None = None
    location: str | None = None
    application_ids: list[str] | None = None
    tag_names: list[str] | None = None
    attachment_count: int = 0


@dataclass(frozen=True)
class CandidateDocument:
    tenant_id: str
    candidate_id: str
    document_id: str
    content_type: str
    source_system: str
    source_id: str
    synced_at: datetime


class ATSAdapter(Protocol):
    def search_candidates(self, filters: dict[str, Any], page_token: str | None = None) -> list[CandidateProfile]: ...

    def get_candidate(self, candidate_id: str) -> CandidateProfile | None: ...

    def get_candidate_documents(self, candidate_id: str) -> list[CandidateDocument]: ...

    def upsert_shortlist_assessment(self, candidate_id: str, project_id: str, assessment: dict[str, Any]) -> None: ...
