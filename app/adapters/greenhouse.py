from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from app.adapters.ats import ATSAdapter, CandidateDocument, CandidateProfile
from app.core.exceptions import ExternalServiceError


class GreenhouseATSAdapter(ATSAdapter):
    def __init__(
        self,
        *,
        api_token: str,
        base_url: str = "https://harvest.greenhouse.io/v1",
        per_page: int = 100,
        max_pages: int = 10,
        client: httpx.Client | None = None,
    ):
        self.api_token = api_token
        self.base_url = base_url.rstrip("/")
        self.per_page = max(1, min(per_page, 500))
        self.max_pages = max(1, max_pages)
        self.client = client or httpx.Client(timeout=15.0)

    def search_candidates(self, filters: dict[str, Any], page_token: str | None = None) -> list[CandidateProfile]:
        tenant_id = str(filters.get("tenant_id") or "greenhouse")
        keywords = [str(item).lower() for item in filters.get("keywords") or []]
        required_skills = [str(item).lower() for item in filters.get("required_skills") or []]
        limit = max(1, min(int(filters.get("limit") or self.per_page), 500))

        current_page = page_token or "1"
        pages_fetched = 0
        matched_profiles: list[CandidateProfile] = []

        while pages_fetched < self.max_pages:
            payload, next_page = self._fetch_candidates_page(filters=filters, page=current_page, per_page=limit)
            page_profiles = [
                candidate
                for candidate in (self._to_candidate_profile(item, tenant_id=tenant_id) for item in payload)
                if self._matches_filters(candidate, keywords=keywords, required_skills=required_skills)
            ]
            matched_profiles.extend(page_profiles)

            if len(matched_profiles) >= limit:
                return matched_profiles[:limit]
            if not next_page:
                break
            current_page = next_page
            pages_fetched += 1

        return matched_profiles[:limit]

    def get_candidate(self, candidate_id: str) -> CandidateProfile | None:
        response = self.client.get(
            f"{self.base_url}/candidates/{candidate_id}",
            auth=(self.api_token, ""),
            headers={"Accept": "application/json"},
        )
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise ExternalServiceError(
                "Greenhouse candidate lookup failed.",
                details={"status_code": response.status_code, "provider": "greenhouse"},
            )
        return self._to_candidate_profile(response.json(), tenant_id="greenhouse")

    def get_candidate_documents(self, candidate_id: str) -> list[CandidateDocument]:
        response = self.client.get(
            f"{self.base_url}/candidates/{candidate_id}",
            auth=(self.api_token, ""),
            headers={"Accept": "application/json"},
        )
        if response.status_code == 404:
            return []
        if response.status_code >= 400:
            raise ExternalServiceError(
                "Greenhouse candidate attachment lookup failed.",
                details={"status_code": response.status_code, "provider": "greenhouse"},
            )
        candidate = response.json()
        documents: list[CandidateDocument] = []
        for attachment in candidate.get("attachments") or []:
            documents.append(
                CandidateDocument(
                    tenant_id="greenhouse",
                    candidate_id=str(candidate.get("id")),
                    document_id=str(attachment.get("filename") or attachment.get("url") or attachment.get("type")),
                    content_type=str(attachment.get("type") or "attachment"),
                    source_system="greenhouse",
                    source_id=str(attachment.get("url") or attachment.get("filename") or attachment.get("type")),
                    synced_at=self._parse_timestamp(attachment.get("created_at")),
                )
            )
        return documents

    def upsert_shortlist_assessment(self, candidate_id: str, project_id: str, assessment: dict[str, Any]) -> None:
        raise NotImplementedError("Greenhouse shortlist write-back is not implemented yet.")

    def _fetch_candidates_page(
        self,
        *,
        filters: dict[str, Any],
        page: str,
        per_page: int,
    ) -> tuple[list[dict[str, Any]], str | None]:
        params: dict[str, Any] = {
            "per_page": per_page,
            "page": page,
            "skip_count": "true",
        }
        if filters.get("job_id"):
            params["job_id"] = filters["job_id"]
        if filters.get("email"):
            params["email"] = filters["email"]
        if filters.get("candidate_ids"):
            candidate_ids = filters["candidate_ids"]
            if isinstance(candidate_ids, list):
                params["candidate_ids"] = ",".join(str(item) for item in candidate_ids[:50])
            else:
                params["candidate_ids"] = str(candidate_ids)
        if filters.get("created_after"):
            params["created_after"] = filters["created_after"]
        if filters.get("created_before"):
            params["created_before"] = filters["created_before"]
        if filters.get("updated_after"):
            params["updated_after"] = filters["updated_after"]
        if filters.get("updated_before"):
            params["updated_before"] = filters["updated_before"]

        response = self.client.get(
            f"{self.base_url}/candidates",
            params=params,
            auth=(self.api_token, ""),
            headers={"Accept": "application/json"},
        )
        if response.status_code >= 400:
            raise ExternalServiceError(
                "Greenhouse candidate search failed.",
                details={"status_code": response.status_code, "provider": "greenhouse"},
            )

        next_page = self._extract_next_page(response.headers.get("Link"))
        return response.json(), next_page

    def _to_candidate_profile(self, item: dict[str, Any], *, tenant_id: str) -> CandidateProfile:
        full_name = " ".join(part for part in [item.get("first_name"), item.get("last_name")] if part).strip()
        return CandidateProfile(
            tenant_id=tenant_id,
            candidate_id=str(item.get("id")),
            full_name=full_name or str(item.get("name") or f"candidate-{item.get('id')}"),
            current_title=str(item.get("title") or "Unknown"),
            current_company=item.get("company"),
            primary_email=self._select_primary_email(item.get("email_addresses") or []),
            location=self._select_primary_location(item.get("addresses") or []),
            application_ids=[str(app_id) for app_id in item.get("application_ids") or []],
            tag_names=[str(tag) for tag in item.get("tags") or []],
            attachment_count=len(item.get("attachments") or []),
            source_system="greenhouse",
            source_id=str(item.get("id")),
            synced_at=self._parse_timestamp(item.get("updated_at") or item.get("created_at")),
        )

    def _matches_filters(
        self,
        candidate: CandidateProfile,
        *,
        keywords: list[str],
        required_skills: list[str],
    ) -> bool:
        if not keywords and not required_skills:
            return True
        blob = " ".join(
            filter(
                None,
                [
                    candidate.full_name,
                    candidate.current_title,
                    candidate.current_company,
                    candidate.primary_email,
                    candidate.location,
                    " ".join(candidate.tag_names or []),
                ],
            )
        ).lower()
        return any(token in blob for token in keywords + required_skills if token)

    def _parse_timestamp(self, value: Any) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)

    def _extract_next_page(self, link_header: str | None) -> str | None:
        if not link_header:
            return None
        for item in link_header.split(","):
            part = item.strip()
            if 'rel="next"' not in part:
                continue
            url_part = part.split(";")[0].strip()
            if not (url_part.startswith("<") and url_part.endswith(">")):
                continue
            parsed = urlparse(url_part[1:-1])
            page_values = parse_qs(parsed.query).get("page")
            if page_values:
                return page_values[0]
        return None

    def _select_primary_email(self, email_addresses: list[dict[str, Any]]) -> str | None:
        if not email_addresses:
            return None
        for email in email_addresses:
            if email.get("type") in {"work", "personal", "other"} and email.get("value"):
                return str(email["value"])
        return str(email_addresses[0].get("value")) if email_addresses[0].get("value") else None

    def _select_primary_location(self, addresses: list[dict[str, Any]]) -> str | None:
        if not addresses:
            return None
        for address in addresses:
            if address.get("value"):
                return str(address["value"])
        return None
