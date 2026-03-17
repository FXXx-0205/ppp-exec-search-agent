from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import settings


class JsonlAuditRepository:
    def __init__(self, path: str | None = None):
        self.path = Path(path or settings.audit_log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as file_handle:
            file_handle.write(line + "\n")

    def list_events(
        self,
        *,
        tenant_id: str,
        project_id: str | None = None,
        event_type: str | None = None,
        action: str | None = None,
        resource_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []

        events: list[dict[str, Any]] = []
        matched = 0
        for line in reversed(self.path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("tenant_id") != tenant_id:
                continue
            if project_id is not None and event.get("project_id") != project_id:
                continue
            if event_type is not None and event.get("event_type") != event_type:
                continue
            if action is not None and event.get("action") != action:
                continue
            if resource_type is not None and event.get("resource_type") != resource_type:
                continue
            if matched < offset:
                matched += 1
                continue
            normalized = dict(event)
            normalized.setdefault("audit_id", normalized.get("request_id"))
            events.append(normalized)
            if len(events) >= limit:
                break
        return events
