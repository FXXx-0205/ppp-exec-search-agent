from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


@dataclass(frozen=True)
class KnowledgeDocument:
    tenant_id: str
    document_id: str
    title: str
    source_system: str
    source_id: str
    synced_at: datetime


class DocumentStoreAdapter(Protocol):
    def list_documents(self, collection: str, updated_since: datetime | None = None) -> list[KnowledgeDocument]: ...

    def get_document(self, document_id: str) -> KnowledgeDocument | None: ...

    def get_document_content(self, document_id: str) -> str: ...

    def store_generated_document(
        self,
        project_id: str,
        document_type: str,
        content: str,
        metadata: dict[str, Any],
    ) -> None: ...
