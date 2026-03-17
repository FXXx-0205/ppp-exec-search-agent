from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from app.config import settings
from app.repositories.interfaces import StoredBrief, StoredSearchProject, StoredSearchResultSnapshot, StoredSearchRun


class ProjectRepo:
    def __init__(self, root_dir: str | None = None):
        base = Path(root_dir or settings.brief_storage_dir).parent
        self.root = base / "projects"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, project_id: str) -> Path:
        return self.root / f"{project_id}.json"

    def create_project(self, project: StoredSearchProject) -> StoredSearchProject:
        self._path(project.project_id).write_text(json.dumps(asdict(project), ensure_ascii=False, indent=2), encoding="utf-8")
        return project

    def get_project(self, project_id: str) -> StoredSearchProject | None:
        path = self._path(project_id)
        if not path.exists():
            return None
        return StoredSearchProject(**json.loads(path.read_text(encoding="utf-8")))

    def list_projects(self, *, tenant_id: str, limit: int = 50, offset: int = 0) -> list[StoredSearchProject]:
        projects: list[StoredSearchProject] = []
        matched = 0
        for path in sorted(self.root.glob("*.json"), reverse=True):
            try:
                project = StoredSearchProject(**json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if project.tenant_id != tenant_id:
                continue
            if matched < offset:
                matched += 1
                continue
            projects.append(project)
            if len(projects) >= limit:
                break
        return projects

    def update_project(self, project: StoredSearchProject) -> StoredSearchProject:
        return self.create_project(project)

    def delete_project(self, project_id: str) -> bool:
        path = self._path(project_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def list_project_runs(self, project_id: str, *, tenant_id: str) -> list[StoredSearchRun]:
        runs_root = self.root.parent / "runs"
        if not runs_root.exists():
            return []
        runs: list[StoredSearchRun] = []
        for path in sorted(runs_root.glob("*.json"), reverse=True):
            try:
                run = StoredSearchRun(**json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if run.tenant_id == tenant_id and run.project_id == project_id:
                runs.append(run)
        return runs

    def list_project_briefs(self, project_id: str, *, tenant_id: str) -> list[StoredBrief]:
        briefs_root = self.root.parent / "briefs"
        if not briefs_root.exists():
            return []
        briefs: list[StoredBrief] = []
        for path in sorted(briefs_root.glob("*.json"), reverse=True):
            try:
                brief = StoredBrief(**json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if brief.tenant_id == tenant_id and brief.project_id == project_id:
                briefs.append(brief)
        return briefs

    def list_project_snapshots(self, project_id: str, *, tenant_id: str) -> list[StoredSearchResultSnapshot]:
        snapshots_root = self.root.parent / "snapshots"
        if not snapshots_root.exists():
            return []
        snapshots: list[StoredSearchResultSnapshot] = []
        for path in sorted(snapshots_root.glob("*.json"), reverse=True):
            try:
                snapshot = StoredSearchResultSnapshot(**json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if snapshot.tenant_id == tenant_id and snapshot.project_id == project_id:
                snapshots.append(snapshot)
        return snapshots
