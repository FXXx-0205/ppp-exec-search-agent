from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.adapters.ats import ATSAdapter, CandidateDocument, CandidateProfile
from app.adapters.crm import ClientAccount, CRMAdapter, SearchProject
from app.adapters.doc_store import DocumentStoreAdapter, KnowledgeDocument
from app.demo.demo_candidates import build_mock_ats_profiles


class MockCRMAdapter(CRMAdapter):
    def list_projects(self, tenant_id: str, updated_since: datetime | None = None) -> list[SearchProject]:
        return [
            SearchProject(
                tenant_id=tenant_id,
                project_id="proj_demo",
                account_id="acct_demo",
                title="Demo Search Project",
                source_system="mock-crm",
                source_id="proj_demo",
                synced_at=datetime.now(timezone.utc),
            )
        ]

    def get_project(self, project_id: str) -> SearchProject | None:
        return self.list_projects("demo-tenant")[0] if project_id == "proj_demo" else None

    def get_client_account(self, account_id: str) -> ClientAccount | None:
        if account_id != "acct_demo":
            return None
        return ClientAccount(
            tenant_id="demo-tenant",
            account_id=account_id,
            name="Demo Client",
            source_system="mock-crm",
            source_id=account_id,
            synced_at=datetime.now(timezone.utc),
        )

    def append_project_note(self, project_id: str, markdown: str, metadata: dict[str, Any]) -> None:
        return None


class MockATSAdapter(ATSAdapter):
    def search_candidates(self, filters: dict[str, Any], page_token: str | None = None) -> list[CandidateProfile]:
        tenant_id = str(filters.get("tenant_id") or "demo-tenant")
        return build_mock_ats_profiles(tenant_id=tenant_id)

    def get_candidate(self, candidate_id: str) -> CandidateProfile | None:
        return next((candidate for candidate in self.search_candidates({}) if candidate.candidate_id == candidate_id), None)

    def get_candidate_documents(self, candidate_id: str) -> list[CandidateDocument]:
        return [
            CandidateDocument(
                tenant_id="demo-tenant",
                candidate_id=candidate_id,
                document_id="doc_demo",
                content_type="resume",
                source_system="mock-ats",
                source_id="doc_demo",
                synced_at=datetime.now(timezone.utc),
            )
        ]

    def upsert_shortlist_assessment(self, candidate_id: str, project_id: str, assessment: dict[str, Any]) -> None:
        return None


class MockDocumentStoreAdapter(DocumentStoreAdapter):
    def list_documents(self, collection: str, updated_since: datetime | None = None) -> list[KnowledgeDocument]:
        return [
            KnowledgeDocument(
                tenant_id="demo-tenant",
                document_id="doc_demo",
                title=f"Demo {collection} document",
                source_system="mock-doc-store",
                source_id="doc_demo",
                synced_at=datetime.now(timezone.utc),
            )
        ]

    def get_document(self, document_id: str) -> KnowledgeDocument | None:
        return self.list_documents("default")[0] if document_id == "doc_demo" else None

    def get_document_content(self, document_id: str) -> str:
        return f"Mock content for {document_id}"

    def store_generated_document(
        self,
        project_id: str,
        document_type: str,
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        return None
