from __future__ import annotations

from app.core.audit import AuditEvent


def test_audit_event_serializes_business_context() -> None:
    event = AuditEvent(
        event_type="brief_generated",
        request_id="req_123",
        payload={"ok": True},
        tenant_id="tenant_1",
        project_id="project_1",
        actor_id="user_1",
    )

    data = event.to_dict()

    assert data["tenant_id"] == "tenant_1"
    assert data["project_id"] == "project_1"
    assert data["actor_id"] == "user_1"
