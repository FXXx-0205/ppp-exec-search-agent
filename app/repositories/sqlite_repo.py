from __future__ import annotations

import builtins
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.models.auth import ApprovalStatus
from app.models.workflow import BriefStatus, SearchRunStatus
from app.repositories.interfaces import (
    StoredBrief,
    StoredCandidate,
    StoredSearchProject,
    StoredSearchResultSnapshot,
    StoredSearchRun,
)


def _sqlite_path(database_url: str) -> Path:
    if not database_url.startswith("sqlite:///"):
        raise ValueError(f"Unsupported database URL: {database_url}")
    path = Path(database_url.removeprefix("sqlite:///"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class SQLiteRepository:
    def __init__(self, database_url: str):
        self.path = _sqlite_path(database_url)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _table_columns(self, table_name: str) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {row["name"] for row in rows}

    def _ensure_column(self, table_name: str, column_name: str, definition: str) -> None:
        if column_name in self._table_columns(table_name):
            return
        with self._connect() as connection:
            connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


class SqliteBriefRepository(SQLiteRepository):
    def __init__(self, database_url: str):
        super().__init__(database_url)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS briefs (
                    brief_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    content TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'draft',
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT '',
                    submitted_by TEXT,
                    submitted_at TEXT,
                    approved_by TEXT,
                    approved_at TEXT,
                    approval_notes TEXT,
                    rejection_notes TEXT,
                    exported_by TEXT,
                    exported_at TEXT,
                    run_id TEXT,
                    previous_brief_id TEXT,
                    supersedes_brief_id TEXT,
                    change_request_source_brief_id TEXT,
                    change_request_notes TEXT,
                    metadata TEXT,
                    markdown TEXT NOT NULL,
                    role_spec TEXT NOT NULL,
                    citations TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    approval_status TEXT
                )
                """
            )
        self._ensure_column("briefs", "project_id", "TEXT")
        self._ensure_column("briefs", "tenant_id", "TEXT")
        self._ensure_column("briefs", "version", "INTEGER NOT NULL DEFAULT 1")
        self._ensure_column("briefs", "content", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("briefs", "status", "TEXT NOT NULL DEFAULT 'draft'")
        self._ensure_column("briefs", "created_at", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("briefs", "updated_at", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("briefs", "submitted_by", "TEXT")
        self._ensure_column("briefs", "submitted_at", "TEXT")
        self._ensure_column("briefs", "rejection_notes", "TEXT")
        self._ensure_column("briefs", "exported_by", "TEXT")
        self._ensure_column("briefs", "exported_at", "TEXT")
        self._ensure_column("briefs", "run_id", "TEXT")
        self._ensure_column("briefs", "previous_brief_id", "TEXT")
        self._ensure_column("briefs", "supersedes_brief_id", "TEXT")
        self._ensure_column("briefs", "change_request_source_brief_id", "TEXT")
        self._ensure_column("briefs", "change_request_notes", "TEXT")
        self._ensure_column("briefs", "metadata", "TEXT")
        self._ensure_column("briefs", "approval_status", "TEXT")

    def save(self, brief: StoredBrief) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO briefs (
                    brief_id, project_id, tenant_id, version, content, status, created_by, created_at, updated_at,
                    submitted_by, submitted_at, approved_by, approved_at, approval_notes, rejection_notes,
                    exported_by, exported_at, run_id, previous_brief_id, supersedes_brief_id,
                    change_request_source_brief_id, change_request_notes, metadata, markdown, role_spec,
                    citations, generated_at, approval_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    brief.brief_id,
                    brief.project_id,
                    brief.tenant_id,
                    brief.version,
                    brief.content,
                    brief.status,
                    brief.created_by,
                    brief.created_at,
                    brief.updated_at,
                    brief.submitted_by,
                    brief.submitted_at,
                    brief.approved_by,
                    brief.approved_at,
                    brief.approval_notes,
                    brief.rejection_notes,
                    brief.exported_by,
                    brief.exported_at,
                    brief.run_id,
                    brief.previous_brief_id,
                    brief.supersedes_brief_id,
                    brief.change_request_source_brief_id,
                    brief.change_request_notes,
                    json.dumps(brief.metadata or {}, ensure_ascii=False),
                    brief.markdown,
                    json.dumps(brief.role_spec, ensure_ascii=False),
                    json.dumps(brief.citations or [], ensure_ascii=False),
                    brief.generated_at or brief.created_at,
                    brief.approval_status,
                ),
            )

    def create_brief_version(self, brief: StoredBrief) -> StoredBrief:
        self.save(brief)
        return brief

    def get(self, brief_id: str) -> StoredBrief | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM briefs WHERE brief_id = ?", (brief_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_brief(row)

    def get_brief(self, brief_id: str) -> StoredBrief | None:
        return self.get(brief_id)

    def list(
        self,
        *,
        tenant_id: str,
        project_id: str | None = None,
        approval_status: ApprovalStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> builtins.list[StoredBrief]:
        query = "SELECT * FROM briefs WHERE tenant_id = ?"
        params: list[Any] = [tenant_id]
        if project_id is not None:
            query += " AND project_id = ?"
            params.append(project_id)
        if approval_status is not None:
            query += " AND approval_status = ?"
            params.append(approval_status)
        query += " ORDER BY generated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [
            self._row_to_brief(row)
            for row in rows
        ]

    def list_briefs_by_project(self, *, project_id: str, tenant_id: str) -> builtins.list[StoredBrief]:
        return self.list(tenant_id=tenant_id, project_id=project_id, limit=1000, offset=0)

    def get_latest_brief_by_project(self, *, project_id: str, tenant_id: str) -> StoredBrief | None:
        briefs = self.list_briefs_by_project(project_id=project_id, tenant_id=tenant_id)
        if not briefs:
            return None
        return max(briefs, key=lambda item: item.version)

    def update_brief_status(
        self,
        brief_id: str,
        *,
        status: BriefStatus,
        updated_by: str,
        notes: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StoredBrief | None:
        brief = self.get(brief_id)
        if brief is None:
            return None
        now = datetime.now(timezone.utc).isoformat()
        updated = StoredBrief(
            **{
                **asdict(brief),
                "status": status,
                "updated_at": now,
                "metadata": {**(brief.metadata or {}), **(metadata or {})} or None,
                "approved_by": updated_by if status == BriefStatus.APPROVED else brief.approved_by,
                "approved_at": now if status == BriefStatus.APPROVED else brief.approved_at,
                "approval_notes": notes if status == BriefStatus.APPROVED else brief.approval_notes,
                "rejection_notes": notes if status in {BriefStatus.REJECTED, BriefStatus.CHANGES_REQUESTED} else brief.rejection_notes,
                "change_request_notes": notes if status == BriefStatus.CHANGES_REQUESTED else brief.change_request_notes,
            }
        )
        self.save(updated)
        return updated

    def submit_for_approval(self, brief_id: str, *, submitted_by: str, notes: str | None = None) -> StoredBrief | None:
        brief = self.get(brief_id)
        if brief is None:
            return None
        now = datetime.now(timezone.utc).isoformat()
        updated = StoredBrief(
            **{
                **asdict(brief),
                "status": BriefStatus.PENDING_APPROVAL,
                "updated_at": now,
                "submitted_by": submitted_by,
                "submitted_at": now,
                "approval_notes": notes if notes else brief.approval_notes,
            }
        )
        self.save(updated)
        return updated

    def approve_brief(self, brief_id: str, *, approved_by: str, notes: str | None = None) -> StoredBrief | None:
        return self.update_brief_status(brief_id, status=BriefStatus.APPROVED, updated_by=approved_by, notes=notes)

    def reject_brief(self, brief_id: str, *, rejected_by: str, notes: str | None = None) -> StoredBrief | None:
        return self.update_brief_status(brief_id, status=BriefStatus.REJECTED, updated_by=rejected_by, notes=notes)

    def request_changes(self, brief_id: str, *, requested_by: str, notes: str | None = None) -> StoredBrief | None:
        return self.update_brief_status(
            brief_id,
            status=BriefStatus.CHANGES_REQUESTED,
            updated_by=requested_by,
            notes=notes,
        )

    def mark_exported(self, brief_id: str, *, exported_by: str) -> StoredBrief | None:
        brief = self.get(brief_id)
        if brief is None:
            return None
        now = datetime.now(timezone.utc).isoformat()
        updated = StoredBrief(
            **{
                **asdict(brief),
                "status": BriefStatus.EXPORTED,
                "updated_at": now,
                "exported_by": exported_by,
                "exported_at": now,
            }
        )
        self.save(updated)
        return updated

    def create_revision(
        self,
        source_brief_id: str,
        *,
        new_brief_id: str,
        created_by: str,
        content: str | None = None,
    ) -> StoredBrief | None:
        source = self.get(source_brief_id)
        if source is None:
            return None
        now = datetime.now(timezone.utc).isoformat()
        revision = StoredBrief(
            **{
                **asdict(source),
                "brief_id": new_brief_id,
                "version": source.version + 1,
                "content": content or source.content,
                "markdown": content or source.markdown,
                "status": BriefStatus.DRAFT,
                "created_by": created_by,
                "created_at": now,
                "updated_at": now,
                "submitted_by": None,
                "submitted_at": None,
                "approved_by": None,
                "approved_at": None,
                "approval_notes": None,
                "rejection_notes": None,
                "exported_by": None,
                "exported_at": None,
                "previous_brief_id": source.brief_id,
                "supersedes_brief_id": source.brief_id,
                "change_request_source_brief_id": source.brief_id,
                "change_request_notes": source.rejection_notes,
            }
        )
        self.save(revision)
        return revision

    def decide(
        self,
        brief_id: str,
        *,
        status: ApprovalStatus,
        decided_by: str,
        comment: str | None = None,
    ) -> StoredBrief | None:
        if status == ApprovalStatus.APPROVED:
            return self.approve_brief(brief_id, approved_by=decided_by, notes=comment)
        if status == ApprovalStatus.REJECTED:
            return self.reject_brief(brief_id, rejected_by=decided_by, notes=comment)
        return self.submit_for_approval(brief_id, submitted_by=decided_by, notes=comment)

    def _row_to_brief(self, row: sqlite3.Row) -> StoredBrief:
        citations = json.loads(row["citations"]) if row["citations"] else []
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        status_value = row["status"] or BriefStatus.DRAFT
        return StoredBrief(
            brief_id=row["brief_id"],
            project_id=row["project_id"] or "",
            tenant_id=row["tenant_id"],
            version=row["version"] or 1,
            content=row["content"] or row["markdown"],
            status=BriefStatus(status_value),
            created_by=row["created_by"],
            created_at=row["created_at"] or row["generated_at"],
            updated_at=row["updated_at"] or row["generated_at"],
            submitted_by=row["submitted_by"],
            submitted_at=row["submitted_at"],
            approved_by=row["approved_by"],
            approved_at=row["approved_at"],
            approval_notes=row["approval_notes"],
            rejection_notes=row["rejection_notes"],
            exported_by=row["exported_by"],
            exported_at=row["exported_at"],
            run_id=row["run_id"],
            previous_brief_id=row["previous_brief_id"],
            supersedes_brief_id=row["supersedes_brief_id"],
            change_request_source_brief_id=row["change_request_source_brief_id"],
            change_request_notes=row["change_request_notes"],
            metadata=metadata or None,
            markdown=row["markdown"],
            role_spec=json.loads(row["role_spec"]),
            citations=citations,
            generated_at=row["generated_at"],
        )


class SqliteProjectRepository(SQLiteRepository):
    def __init__(self, database_url: str):
        super().__init__(database_url)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS search_projects (
                    project_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    project_name TEXT NOT NULL,
                    client_name TEXT,
                    role_title TEXT,
                    status TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT
                )
                """
            )

    def create_project(self, project: StoredSearchProject) -> StoredSearchProject:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO search_projects (
                    project_id, tenant_id, project_name, client_name, role_title, status,
                    created_by, created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project.project_id,
                    project.tenant_id,
                    project.project_name,
                    project.client_name,
                    project.role_title,
                    project.status,
                    project.created_by,
                    project.created_at,
                    project.updated_at,
                    json.dumps(project.metadata or {}, ensure_ascii=False),
                ),
            )
        return project

    def get_project(self, project_id: str) -> StoredSearchProject | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM search_projects WHERE project_id = ?", (project_id,)).fetchone()
        if row is None:
            return None
        return StoredSearchProject(
            project_id=row["project_id"],
            tenant_id=row["tenant_id"],
            project_name=row["project_name"],
            client_name=row["client_name"],
            role_title=row["role_title"],
            status=row["status"],
            created_by=row["created_by"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
        )

    def list_projects(self, *, tenant_id: str, limit: int = 50, offset: int = 0) -> list[StoredSearchProject]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM search_projects
                WHERE tenant_id = ?
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (tenant_id, limit, offset),
            ).fetchall()
        return [
            StoredSearchProject(
                project_id=row["project_id"],
                tenant_id=row["tenant_id"],
                project_name=row["project_name"],
                client_name=row["client_name"],
                role_title=row["role_title"],
                status=row["status"],
                created_by=row["created_by"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            )
            for row in rows
        ]

    def update_project(self, project: StoredSearchProject) -> StoredSearchProject:
        return self.create_project(project)

    def delete_project(self, project_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM search_projects WHERE project_id = ?", (project_id,))
        return cursor.rowcount > 0

    def list_project_runs(self, project_id: str, *, tenant_id: str) -> list[StoredSearchRun]:
        repo = SqliteSearchRunRepository(f"sqlite:///{self.path}")
        return repo.list_runs_by_project(project_id=project_id, tenant_id=tenant_id)

    def list_project_briefs(self, project_id: str, *, tenant_id: str) -> list[StoredBrief]:
        repo = SqliteBriefRepository(f"sqlite:///{self.path}")
        return repo.list_briefs_by_project(project_id=project_id, tenant_id=tenant_id)

    def list_project_snapshots(self, project_id: str, *, tenant_id: str) -> list[StoredSearchResultSnapshot]:
        repo = SqliteSearchResultSnapshotRepository(f"sqlite:///{self.path}")
        return repo.list_snapshots_by_project(project_id=project_id, tenant_id=tenant_id)


class SqliteSearchRunRepository(SQLiteRepository):
    def __init__(self, database_url: str):
        super().__init__(database_url)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS search_runs (
                    run_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    input_jd TEXT NOT NULL,
                    parsed_role_json TEXT,
                    candidate_source TEXT,
                    result_count INTEGER,
                    run_status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    failed_at TEXT,
                    error_message TEXT,
                    created_by TEXT NOT NULL,
                    metadata TEXT
                )
                """
            )

    def create_run(self, run: StoredSearchRun) -> StoredSearchRun:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO search_runs (
                    run_id, project_id, tenant_id, input_jd, parsed_role_json, candidate_source,
                    result_count, run_status, started_at, completed_at, failed_at, error_message,
                    created_by, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.project_id,
                    run.tenant_id,
                    run.input_jd,
                    json.dumps(run.parsed_role_json or {}, ensure_ascii=False),
                    run.candidate_source,
                    run.result_count,
                    run.run_status,
                    run.started_at,
                    run.completed_at,
                    run.failed_at,
                    run.error_message,
                    run.created_by,
                    json.dumps(run.metadata or {}, ensure_ascii=False),
                ),
            )
        return run

    def get_run(self, run_id: str) -> StoredSearchRun | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM search_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return StoredSearchRun(
            run_id=row["run_id"],
            project_id=row["project_id"],
            tenant_id=row["tenant_id"],
            input_jd=row["input_jd"],
            parsed_role_json=json.loads(row["parsed_role_json"]) if row["parsed_role_json"] else None,
            candidate_source=row["candidate_source"],
            result_count=row["result_count"],
            run_status=SearchRunStatus(row["run_status"]),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            failed_at=row["failed_at"],
            error_message=row["error_message"],
            created_by=row["created_by"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
        )

    def list_runs_by_project(self, *, project_id: str, tenant_id: str) -> list[StoredSearchRun]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM search_runs
                WHERE tenant_id = ? AND project_id = ?
                ORDER BY started_at DESC
                """,
                (tenant_id, project_id),
            ).fetchall()
        return [
            StoredSearchRun(
                run_id=row["run_id"],
                project_id=row["project_id"],
                tenant_id=row["tenant_id"],
                input_jd=row["input_jd"],
                parsed_role_json=json.loads(row["parsed_role_json"]) if row["parsed_role_json"] else None,
                candidate_source=row["candidate_source"],
                result_count=row["result_count"],
                run_status=SearchRunStatus(row["run_status"]),
                started_at=row["started_at"],
                completed_at=row["completed_at"],
                failed_at=row["failed_at"],
                error_message=row["error_message"],
                created_by=row["created_by"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            )
            for row in rows
        ]

    def update_run(self, run: StoredSearchRun) -> StoredSearchRun:
        return self.create_run(run)

    def mark_run_failed(self, run_id: str, *, error_message: str, failed_at: str) -> StoredSearchRun | None:
        run = self.get_run(run_id)
        if run is None:
            return None
        updated = StoredSearchRun(**{**asdict(run), "run_status": SearchRunStatus.FAILED, "failed_at": failed_at, "error_message": error_message})
        return self.update_run(updated)

    def mark_run_completed(self, run_id: str, *, run_status: SearchRunStatus, completed_at: str) -> StoredSearchRun | None:
        run = self.get_run(run_id)
        if run is None:
            return None
        updated = StoredSearchRun(**{**asdict(run), "run_status": run_status, "completed_at": completed_at})
        return self.update_run(updated)


class SqliteSearchResultSnapshotRepository(SQLiteRepository):
    def __init__(self, database_url: str):
        super().__init__(database_url)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS search_result_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    top_candidates TEXT NOT NULL,
                    ranking_payload TEXT NOT NULL,
                    candidate_count INTEGER NOT NULL,
                    metadata TEXT
                )
                """
            )

    def create_snapshot(self, snapshot: StoredSearchResultSnapshot) -> StoredSearchResultSnapshot:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO search_result_snapshots (
                    snapshot_id, run_id, project_id, tenant_id, created_at, top_candidates, ranking_payload, candidate_count, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.snapshot_id,
                    snapshot.run_id,
                    snapshot.project_id,
                    snapshot.tenant_id,
                    snapshot.created_at,
                    json.dumps(snapshot.top_candidates, ensure_ascii=False),
                    json.dumps(snapshot.ranking_payload, ensure_ascii=False),
                    snapshot.candidate_count,
                    json.dumps(snapshot.metadata or {}, ensure_ascii=False),
                ),
            )
        return snapshot

    def get_snapshot(self, snapshot_id: str) -> StoredSearchResultSnapshot | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM search_result_snapshots WHERE snapshot_id = ?", (snapshot_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_snapshot(row)

    def get_snapshot_by_run(self, *, run_id: str, tenant_id: str) -> StoredSearchResultSnapshot | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM search_result_snapshots
                WHERE run_id = ? AND tenant_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (run_id, tenant_id),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_snapshot(row)

    def list_snapshots_by_project(self, *, project_id: str, tenant_id: str) -> list[StoredSearchResultSnapshot]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM search_result_snapshots
                WHERE project_id = ? AND tenant_id = ?
                ORDER BY created_at DESC
                """,
                (project_id, tenant_id),
            ).fetchall()
        return [self._row_to_snapshot(row) for row in rows]

    def _row_to_snapshot(self, row: sqlite3.Row) -> StoredSearchResultSnapshot:
        return StoredSearchResultSnapshot(
            snapshot_id=row["snapshot_id"],
            run_id=row["run_id"],
            project_id=row["project_id"],
            tenant_id=row["tenant_id"],
            created_at=row["created_at"],
            top_candidates=json.loads(row["top_candidates"]),
            ranking_payload=json.loads(row["ranking_payload"]),
            candidate_count=row["candidate_count"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
        )


class SqliteAuditRepository(SQLiteRepository):
    def __init__(self, database_url: str):
        super().__init__(database_url)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    action TEXT,
                    resource_type TEXT,
                    resource_id TEXT,
                    tenant_id TEXT,
                    project_id TEXT,
                    run_id TEXT,
                    brief_id TEXT,
                    actor_id TEXT,
                    payload TEXT NOT NULL,
                    ts TEXT NOT NULL
                )
                """
            )
        self._ensure_column("audit_events", "action", "TEXT")
        self._ensure_column("audit_events", "resource_type", "TEXT")
        self._ensure_column("audit_events", "resource_id", "TEXT")
        self._ensure_column("audit_events", "run_id", "TEXT")
        self._ensure_column("audit_events", "brief_id", "TEXT")

    def append(self, event: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_events (
                    event_type, request_id, action, resource_type, resource_id, tenant_id, project_id, run_id, brief_id, actor_id, payload, ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["event_type"],
                    event["request_id"],
                    event.get("action"),
                    event.get("resource_type"),
                    event.get("resource_id"),
                    event.get("tenant_id"),
                    event.get("project_id"),
                    event.get("run_id"),
                    event.get("brief_id"),
                    event.get("actor_id"),
                    json.dumps(event.get("payload", {}), ensure_ascii=False),
                    event["ts"],
                ),
            )

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
        query = "SELECT * FROM audit_events WHERE tenant_id = ?"
        params: list[Any] = [tenant_id]
        if project_id is not None:
            query += " AND project_id = ?"
            params.append(project_id)
        if event_type is not None:
            query += " AND event_type = ?"
            params.append(event_type)
        if action is not None:
            query += " AND action = ?"
            params.append(action)
        if resource_type is not None:
            query += " AND resource_type = ?"
            params.append(resource_type)
        query += " ORDER BY ts DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()

        return [
            {
                "audit_id": row["id"],
                "event_type": row["event_type"],
                "request_id": row["request_id"],
                "action": row["action"],
                "resource_type": row["resource_type"],
                "resource_id": row["resource_id"],
                "tenant_id": row["tenant_id"],
                "project_id": row["project_id"],
                "run_id": row["run_id"],
                "brief_id": row["brief_id"],
                "actor_id": row["actor_id"],
                "payload": json.loads(row["payload"]),
                "ts": row["ts"],
            }
            for row in rows
        ]


class SqliteCandidateRepository(SQLiteRepository):
    def __init__(self, database_url: str):
        super().__init__(database_url)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS candidates (
                    tenant_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    current_title TEXT NOT NULL,
                    current_company TEXT,
                    location TEXT,
                    primary_email TEXT,
                    summary TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    source_system TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    application_ids TEXT NOT NULL,
                    tag_names TEXT NOT NULL,
                    attachment_count INTEGER NOT NULL,
                    synced_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, candidate_id)
                )
                """
            )

    def upsert_many(self, candidates: list[StoredCandidate]) -> None:
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO candidates (
                    tenant_id, candidate_id, full_name, current_title, current_company, location,
                    primary_email, summary, evidence, source_system, source_id, application_ids,
                    tag_names, attachment_count, synced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        candidate.tenant_id,
                        candidate.candidate_id,
                        candidate.full_name,
                        candidate.current_title,
                        candidate.current_company,
                        candidate.location,
                        candidate.primary_email,
                        candidate.summary,
                        json.dumps(candidate.evidence, ensure_ascii=False),
                        candidate.source_system,
                        candidate.source_id,
                        json.dumps(candidate.application_ids, ensure_ascii=False),
                        json.dumps(candidate.tag_names, ensure_ascii=False),
                        candidate.attachment_count,
                        candidate.synced_at,
                    )
                    for candidate in candidates
                ],
            )

    def list(
        self,
        *,
        tenant_id: str,
        search_text: str | None = None,
        source_system: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[StoredCandidate]:
        query = "SELECT * FROM candidates WHERE tenant_id = ?"
        params: list[Any] = [tenant_id]
        if source_system is not None:
            query += " AND source_system = ?"
            params.append(source_system)
        if search_text:
            query += " AND (lower(full_name) LIKE ? OR lower(current_title) LIKE ? OR lower(summary) LIKE ?)"
            pattern = f"%{search_text.lower()}%"
            params.extend([pattern, pattern, pattern])
        query += " ORDER BY synced_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()

        return [
            StoredCandidate(
                tenant_id=row["tenant_id"],
                candidate_id=row["candidate_id"],
                full_name=row["full_name"],
                current_title=row["current_title"],
                current_company=row["current_company"],
                location=row["location"],
                primary_email=row["primary_email"],
                summary=row["summary"],
                evidence=json.loads(row["evidence"]),
                source_system=row["source_system"],
                source_id=row["source_id"],
                application_ids=json.loads(row["application_ids"]),
                tag_names=json.loads(row["tag_names"]),
                attachment_count=int(row["attachment_count"]),
                synced_at=row["synced_at"],
            )
            for row in rows
        ]
