from __future__ import annotations

from app.models.auth import ApprovalStatus
from app.repositories.interfaces import StoredBrief
from app.repositories.sqlite_repo import SqliteAuditRepository, SqliteBriefRepository


def test_sqlite_brief_repository_round_trip(tmp_path) -> None:
    database_url = f"sqlite:///{tmp_path}/app.db"
    repo = SqliteBriefRepository(database_url)
    brief = StoredBrief(
        brief_id="brief_sqlite",
        markdown="# SQLite",
        role_spec={"title": "Infra Lead"},
        citations=["doc-1"],
        generated_at="2026-03-14T00:00:00+00:00",
        tenant_id="tenant_sql",
        project_id="project_sql",
        created_by="user_sql",
        approval_status=ApprovalStatus.PENDING,
    )

    repo.save(brief)
    repo.decide("brief_sqlite", status=ApprovalStatus.APPROVED, decided_by="approver_1", comment="approved")
    loaded = repo.get("brief_sqlite")

    assert loaded is not None
    assert loaded.approval_status == ApprovalStatus.APPROVED
    assert loaded.approved_by == "approver_1"


def test_sqlite_audit_repository_persists_event(tmp_path) -> None:
    database_url = f"sqlite:///{tmp_path}/app.db"
    repo = SqliteAuditRepository(database_url)

    repo.append(
        {
            "event_type": "brief_generated",
            "request_id": "req_1",
            "tenant_id": "tenant_sql",
            "project_id": "project_sql",
            "actor_id": "user_sql",
            "payload": {"ok": True},
            "ts": "2026-03-14T00:00:00+00:00",
        }
    )

    with repo._connect() as connection:
        row = connection.execute("SELECT event_type, tenant_id FROM audit_events").fetchone()

    assert row is not None
    assert row["event_type"] == "brief_generated"
    assert row["tenant_id"] == "tenant_sql"
