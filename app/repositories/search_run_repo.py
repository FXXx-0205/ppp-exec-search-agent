from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from app.config import settings
from app.models.workflow import SearchRunStatus
from app.repositories.interfaces import StoredSearchRun


class SearchRunRepo:
    def __init__(self, root_dir: str | None = None):
        base = Path(root_dir or settings.brief_storage_dir).parent
        self.root = base / "runs"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.json"

    def create_run(self, run: StoredSearchRun) -> StoredSearchRun:
        self._path(run.run_id).write_text(json.dumps(asdict(run), ensure_ascii=False, indent=2), encoding="utf-8")
        return run

    def get_run(self, run_id: str) -> StoredSearchRun | None:
        path = self._path(run_id)
        if not path.exists():
            return None
        return StoredSearchRun(**json.loads(path.read_text(encoding="utf-8")))

    def list_runs_by_project(self, *, project_id: str, tenant_id: str) -> list[StoredSearchRun]:
        runs: list[StoredSearchRun] = []
        for path in sorted(self.root.glob("*.json"), reverse=True):
            try:
                run = StoredSearchRun(**json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if run.tenant_id == tenant_id and run.project_id == project_id:
                runs.append(run)
        return runs

    def update_run(self, run: StoredSearchRun) -> StoredSearchRun:
        return self.create_run(run)

    def mark_run_failed(self, run_id: str, *, error_message: str, failed_at: str) -> StoredSearchRun | None:
        run = self.get_run(run_id)
        if run is None:
            return None
        updated = StoredSearchRun(
            **{
                **asdict(run),
                "run_status": SearchRunStatus.FAILED,
                "failed_at": failed_at,
                "error_message": error_message,
            }
        )
        return self.update_run(updated)

    def mark_run_completed(self, run_id: str, *, run_status: SearchRunStatus, completed_at: str) -> StoredSearchRun | None:
        run = self.get_run(run_id)
        if run is None:
            return None
        updated = StoredSearchRun(
            **{
                **asdict(run),
                "run_status": run_status,
                "completed_at": completed_at,
            }
        )
        return self.update_run(updated)
