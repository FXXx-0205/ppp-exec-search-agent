from __future__ import annotations

from app.models.workflow import SearchRunStatus
from app.repositories.interfaces import StoredSearchRun
from app.repositories.search_run_repo import SearchRunRepo


def test_search_run_repo_create_update_fail_complete(tmp_path) -> None:
    repo = SearchRunRepo(root_dir=str(tmp_path / "briefs"))
    repo.create_run(
        StoredSearchRun(
            run_id="run_1",
            project_id="proj_1",
            tenant_id="tenant_1",
            input_jd="Need infra PM",
            parsed_role_json=None,
            candidate_source="local_first",
            result_count=None,
            run_status=SearchRunStatus.DRAFT,
            started_at="2026-03-14T00:00:00+00:00",
            completed_at=None,
            failed_at=None,
            error_message=None,
            created_by="user_1",
            metadata=None,
        )
    )

    updated = repo.update_run(
        StoredSearchRun(
            run_id="run_1",
            project_id="proj_1",
            tenant_id="tenant_1",
            input_jd="Need infra PM",
            parsed_role_json={"title": "Infrastructure Portfolio Manager"},
            candidate_source="local_first",
            result_count=3,
            run_status=SearchRunStatus.RANKED,
            started_at="2026-03-14T00:00:00+00:00",
            completed_at=None,
            failed_at=None,
            error_message=None,
            created_by="user_1",
            metadata=None,
        )
    )
    failed = repo.mark_run_failed("run_1", error_message="boom", failed_at="2026-03-14T00:01:00+00:00")
    completed = repo.mark_run_completed("run_1", run_status=SearchRunStatus.BRIEF_GENERATED, completed_at="2026-03-14T00:02:00+00:00")

    assert updated.run_status == SearchRunStatus.RANKED
    assert failed is not None
    assert failed.run_status == SearchRunStatus.FAILED
    assert completed is not None
    assert completed.run_status == SearchRunStatus.BRIEF_GENERATED
