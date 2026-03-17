from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


@dataclass(frozen=True)
class ClientAccount:
    tenant_id: str
    account_id: str
    name: str
    source_system: str
    source_id: str
    synced_at: datetime


@dataclass(frozen=True)
class SearchProject:
    tenant_id: str
    project_id: str
    account_id: str
    title: str
    source_system: str
    source_id: str
    synced_at: datetime


class CRMAdapter(Protocol):
    def list_projects(self, tenant_id: str, updated_since: datetime | None = None) -> list[SearchProject]: ...

    def get_project(self, project_id: str) -> SearchProject | None: ...

    def get_client_account(self, account_id: str) -> ClientAccount | None: ...

    def append_project_note(self, project_id: str, markdown: str, metadata: dict[str, Any]) -> None: ...
