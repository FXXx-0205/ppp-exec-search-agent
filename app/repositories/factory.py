from __future__ import annotations

from app.config import settings
from app.repositories.audit_repo import JsonlAuditRepository
from app.repositories.brief_repo import BriefRepo
from app.repositories.interfaces import (
    AuditRepository,
    BriefRepository,
    CandidateRepository,
    ProjectRepository,
    SearchResultSnapshotRepository,
    SearchRunRepository,
)
from app.repositories.project_repo import ProjectRepo
from app.repositories.search_result_snapshot_repo import SearchResultSnapshotRepo
from app.repositories.search_run_repo import SearchRunRepo
from app.repositories.sqlite_repo import (
    SqliteAuditRepository,
    SqliteBriefRepository,
    SqliteCandidateRepository,
    SqliteProjectRepository,
    SqliteSearchResultSnapshotRepository,
    SqliteSearchRunRepository,
)


def get_brief_repository() -> BriefRepository:
    if settings.storage_backend == "sqlite":
        return SqliteBriefRepository(settings.database_url)
    return BriefRepo()


def get_audit_repository() -> AuditRepository:
    if settings.storage_backend == "sqlite":
        return SqliteAuditRepository(settings.database_url)
    return JsonlAuditRepository()


def get_candidate_repository() -> CandidateRepository | None:
    if settings.storage_backend == "sqlite":
        return SqliteCandidateRepository(settings.database_url)
    return None


def get_project_repository() -> ProjectRepository:
    if settings.storage_backend == "sqlite":
        return SqliteProjectRepository(settings.database_url)
    return ProjectRepo()


def get_search_run_repository() -> SearchRunRepository:
    if settings.storage_backend == "sqlite":
        return SqliteSearchRunRepository(settings.database_url)
    return SearchRunRepo()


def get_search_result_snapshot_repository() -> SearchResultSnapshotRepository:
    if settings.storage_backend == "sqlite":
        return SqliteSearchResultSnapshotRepository(settings.database_url)
    return SearchResultSnapshotRepo()
