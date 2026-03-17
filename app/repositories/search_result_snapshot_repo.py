from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from app.config import settings
from app.repositories.interfaces import StoredSearchResultSnapshot


class SearchResultSnapshotRepo:
    def __init__(self, root_dir: str | None = None):
        base = Path(root_dir or settings.brief_storage_dir).parent
        self.root = base / "snapshots"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, snapshot_id: str) -> Path:
        return self.root / f"{snapshot_id}.json"

    def create_snapshot(self, snapshot: StoredSearchResultSnapshot) -> StoredSearchResultSnapshot:
        self._path(snapshot.snapshot_id).write_text(json.dumps(asdict(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
        return snapshot

    def get_snapshot(self, snapshot_id: str) -> StoredSearchResultSnapshot | None:
        path = self._path(snapshot_id)
        if not path.exists():
            return None
        return StoredSearchResultSnapshot(**json.loads(path.read_text(encoding="utf-8")))

    def get_snapshot_by_run(self, *, run_id: str, tenant_id: str) -> StoredSearchResultSnapshot | None:
        for path in sorted(self.root.glob("*.json"), reverse=True):
            try:
                snapshot = StoredSearchResultSnapshot(**json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if snapshot.run_id == run_id and snapshot.tenant_id == tenant_id:
                return snapshot
        return None

    def list_snapshots_by_project(self, *, project_id: str, tenant_id: str) -> list[StoredSearchResultSnapshot]:
        snapshots: list[StoredSearchResultSnapshot] = []
        for path in sorted(self.root.glob("*.json"), reverse=True):
            try:
                snapshot = StoredSearchResultSnapshot(**json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if snapshot.project_id == project_id and snapshot.tenant_id == tenant_id:
                snapshots.append(snapshot)
        return snapshots
