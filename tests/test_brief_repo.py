from __future__ import annotations

from app.models.auth import ApprovalStatus
from app.repositories.brief_repo import BriefRepo
from app.repositories.interfaces import StoredBrief


def test_brief_repo_round_trip(tmp_path) -> None:
    repo = BriefRepo(root_dir=str(tmp_path))
    brief = StoredBrief(
        brief_id="brief_test",
        markdown="# Test",
        role_spec={"title": "Test Role"},
        citations=["source-1"],
        generated_at="2026-03-14T00:00:00+00:00",
        tenant_id="tenant_1",
        project_id="proj_1",
        created_by="user_1",
        approval_status=ApprovalStatus.PENDING,
    )

    repo.save(brief)
    loaded = repo.get("brief_test")
    approved = repo.decide("brief_test", status=ApprovalStatus.APPROVED, decided_by="user_2", comment="Looks good.")

    assert loaded is not None
    assert loaded.markdown == "# Test"
    assert approved is not None
    assert approved.approval_status == ApprovalStatus.APPROVED
    assert approved.approved_by == "user_2"
