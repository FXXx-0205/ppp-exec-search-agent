from __future__ import annotations

from typing import Any

from app.adapters.ats import ATSAdapter
from app.core.audit import AuditEvent, AuditLogger
from app.repositories.interfaces import CandidateRepository, StoredCandidate
from app.services.candidate_service import CandidateService


class CandidateSyncService:
    def __init__(
        self,
        *,
        ats_adapter: ATSAdapter,
        candidate_repository: CandidateRepository,
        audit_logger: AuditLogger | None = None,
    ):
        self.ats_adapter = ats_adapter
        self.candidate_repository = candidate_repository
        self.audit_logger = audit_logger or AuditLogger()
        self.candidate_service = CandidateService()

    def sync_candidates(
        self,
        *,
        tenant_id: str,
        provider_filters: dict[str, Any] | None = None,
        request_id: str = "candidate_sync",
    ) -> list[StoredCandidate]:
        profiles = self.ats_adapter.search_candidates(
            {
                "tenant_id": tenant_id,
                "keywords": [],
                "required_skills": [],
                **(provider_filters or {}),
            }
        )
        normalized = [self.candidate_service._normalize_adapter_candidate(profile) for profile in profiles]
        stored_candidates = [
            StoredCandidate(
                tenant_id=tenant_id,
                candidate_id=item["candidate_id"],
                full_name=item["full_name"],
                current_title=item["current_title"],
                current_company=item.get("current_company"),
                location=item.get("location"),
                primary_email=self._extract_email(item.get("evidence") or []),
                summary=item["summary"],
                evidence=item.get("evidence") or [],
                source_system=item["source_system"],
                source_id=item["candidate_id"],
                application_ids=self._extract_values(item.get("evidence") or [], prefix="Application IDs: "),
                tag_names=self._extract_values(item.get("evidence") or [], prefix="Tags: "),
                attachment_count=self._extract_attachment_count(item.get("evidence") or []),
                synced_at=profiles[idx].synced_at.isoformat(),
            )
            for idx, item in enumerate(normalized)
        ]
        self.candidate_repository.upsert_many(stored_candidates)
        self.audit_logger.log(
            AuditEvent(
                event_type="candidates_synced",
                request_id=request_id,
                payload={"count": len(stored_candidates), "provider_filters": provider_filters or {}},
                tenant_id=tenant_id,
            )
        )
        return stored_candidates

    def _extract_email(self, evidence: list[str]) -> str | None:
        for line in evidence:
            if line.startswith("Primary email on record: "):
                return line.removeprefix("Primary email on record: ")
        return None

    def _extract_values(self, evidence: list[str], *, prefix: str) -> list[str]:
        for line in evidence:
            if line.startswith(prefix):
                return [item.strip() for item in line.removeprefix(prefix).split(",") if item.strip()]
        return []

    def _extract_attachment_count(self, evidence: list[str]) -> int:
        for line in evidence:
            if line.startswith("Attachments available: "):
                try:
                    return int(line.removeprefix("Attachments available: ").strip())
                except ValueError:
                    return 0
        return 0
