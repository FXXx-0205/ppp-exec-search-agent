from __future__ import annotations

import pytest

from app.core.exceptions import NotFoundError
from app.models.auth import AccessContext, UserIdentity, UserRole
from app.models.workflow import BriefStatus, SearchRunStatus
from app.repositories.brief_repo import BriefRepo
from app.repositories.interfaces import StoredSearchProject
from app.repositories.project_repo import ProjectRepo
from app.repositories.search_result_snapshot_repo import SearchResultSnapshotRepo
from app.repositories.search_run_repo import SearchRunRepo
from app.services.ranking_service import RankingService
from app.workflows.search_workflow import SearchWorkflow


class StubParser:
    def __init__(self, role_spec=None, exc: Exception | None = None):
        self.role_spec = role_spec or {
            "title": "Infrastructure Portfolio Manager",
            "required_skills": ["portfolio"],
            "preferred_skills": [],
            "search_keywords": ["portfolio"],
            "location": "Australia",
            "sector": "Infrastructure",
            "seniority": "Senior",
        }
        self.exc = exc

    def parse_role(self, _: str) -> dict:
        if self.exc:
            raise self.exc
        return self.role_spec


class StubCandidateService:
    def load_candidates(self, *_args, **_kwargs):
        return [
            {
                "candidate_id": "cand_1",
                "summary": "Senior infrastructure portfolio leader",
                "current_title": "Senior Portfolio Manager",
                "location": "Australia",
                "sectors": ["Infrastructure"],
            }
        ]

    def filter_candidates(self, _role_spec, candidates):
        return candidates


class StubRetriever:
    def retrieve_for_role(self, *_args, **_kwargs):
        return [{"doc_id": "doc_1", "title": "Market Map", "text": "context", "source": "doc-source"}]


class StubBriefService:
    def __init__(self, exc: Exception | None = None):
        self.exc = exc

    def generate_markdown(self, **_kwargs):
        if self.exc:
            raise self.exc
        return {
            "brief_id": "brief_1",
            "markdown": "# Brief",
            "generated_at": "2026-03-14T00:00:00+00:00",
            "citations": ["doc-source"],
            "prompt": {"id": "brief_generator", "version": "v1"},
        }


def _access() -> AccessContext:
    return AccessContext(
        tenant_id="tenant_1",
        actor=UserIdentity(user_id="researcher_1", email="r@example.com", display_name="Researcher", role=UserRole.RESEARCHER),
    )


def _project_repo(tmp_path):
    repo = ProjectRepo(root_dir=str(tmp_path / "briefs"))
    repo.create_project(
        StoredSearchProject(
            project_id="proj_1",
            tenant_id="tenant_1",
            project_name="Infra Search",
            client_name="Client",
            role_title=None,
            status="draft",
            created_by="researcher_1",
            created_at="2026-03-14T00:00:00+00:00",
            updated_at="2026-03-14T00:00:00+00:00",
            metadata=None,
        )
    )
    return repo


def test_search_workflow_happy_path_persists_run_and_brief(tmp_path) -> None:
    project_repo = _project_repo(tmp_path)
    run_repo = SearchRunRepo(root_dir=str(tmp_path / "briefs"))
    brief_repo = BriefRepo(root_dir=str(tmp_path / "briefs"))
    snapshot_repo = SearchResultSnapshotRepo(root_dir=str(tmp_path / "briefs"))
    workflow = SearchWorkflow(
        parser=StubParser(),
        candidate_service=StubCandidateService(),
        ranking_service=RankingService(),
        brief_service=StubBriefService(),
        retriever=StubRetriever(),
        project_repo=project_repo,
        run_repo=run_repo,
        brief_repo=brief_repo,
        snapshot_repo=snapshot_repo,
    )

    result = workflow.run("proj_1", "Need infra PM", _access())
    stored_run = run_repo.get_run(result.run_id)
    stored_brief = brief_repo.get_brief("brief_1")
    snapshot = snapshot_repo.get_snapshot_by_run(run_id=result.run_id, tenant_id="tenant_1")

    assert result.run_status == SearchRunStatus.COMPLETED
    assert result.candidate_count == 1
    assert stored_run is not None
    assert stored_run.run_status == SearchRunStatus.COMPLETED
    assert stored_brief is not None
    assert stored_brief.status == BriefStatus.DRAFT
    assert snapshot is not None
    assert snapshot.candidate_count == 1
    assert snapshot.top_candidates[0]["candidate_id"] == "cand_1"


def test_search_workflow_marks_failed_on_parse_error(tmp_path) -> None:
    project_repo = _project_repo(tmp_path)
    run_repo = SearchRunRepo(root_dir=str(tmp_path / "briefs"))
    workflow = SearchWorkflow(
        parser=StubParser(exc=ValueError("parse failed")),
        candidate_service=StubCandidateService(),
        ranking_service=RankingService(),
        brief_service=StubBriefService(),
        retriever=StubRetriever(),
        project_repo=project_repo,
        run_repo=run_repo,
        brief_repo=BriefRepo(root_dir=str(tmp_path / "briefs")),
        snapshot_repo=SearchResultSnapshotRepo(root_dir=str(tmp_path / "briefs")),
    )

    with pytest.raises(ValueError):
        workflow.run("proj_1", "bad jd", _access())

    runs = run_repo.list_runs_by_project(project_id="proj_1", tenant_id="tenant_1")
    assert runs[0].run_status == SearchRunStatus.FAILED
    assert runs[0].error_message == "parse failed"


def test_search_workflow_marks_failed_on_brief_generation_error(tmp_path) -> None:
    project_repo = _project_repo(tmp_path)
    run_repo = SearchRunRepo(root_dir=str(tmp_path / "briefs"))
    workflow = SearchWorkflow(
        parser=StubParser(),
        candidate_service=StubCandidateService(),
        ranking_service=RankingService(),
        brief_service=StubBriefService(exc=RuntimeError("brief failed")),
        retriever=StubRetriever(),
        project_repo=project_repo,
        run_repo=run_repo,
        brief_repo=BriefRepo(root_dir=str(tmp_path / "briefs")),
        snapshot_repo=SearchResultSnapshotRepo(root_dir=str(tmp_path / "briefs")),
    )

    with pytest.raises(RuntimeError):
        workflow.run("proj_1", "Need infra PM", _access())

    runs = run_repo.list_runs_by_project(project_id="proj_1", tenant_id="tenant_1")
    assert runs[0].run_status == SearchRunStatus.FAILED
    assert runs[0].error_message == "brief failed"


def test_search_workflow_requires_existing_project(tmp_path) -> None:
    workflow = SearchWorkflow(
        parser=StubParser(),
        candidate_service=StubCandidateService(),
        ranking_service=RankingService(),
        brief_service=StubBriefService(),
        retriever=StubRetriever(),
        project_repo=ProjectRepo(root_dir=str(tmp_path / "briefs")),
        run_repo=SearchRunRepo(root_dir=str(tmp_path / "briefs")),
        brief_repo=BriefRepo(root_dir=str(tmp_path / "briefs")),
        snapshot_repo=SearchResultSnapshotRepo(root_dir=str(tmp_path / "briefs")),
    )

    with pytest.raises(NotFoundError):
        workflow.run("missing", "Need infra PM", _access())
