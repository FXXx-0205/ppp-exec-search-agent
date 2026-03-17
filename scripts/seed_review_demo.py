from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.audit import AuditEvent, AuditLogger
from app.demo.demo_candidates import load_demo_candidates, to_stored_candidates
from app.llm.anthropic_client import ClaudeClient
from app.repositories.audit_repo import JsonlAuditRepository
from app.repositories.brief_repo import BriefRepo
from app.repositories.factory import get_candidate_repository
from app.repositories.interfaces import StoredBrief, StoredSearchProject, StoredSearchResultSnapshot, StoredSearchRun
from app.repositories.project_repo import ProjectRepo
from app.repositories.search_result_snapshot_repo import SearchResultSnapshotRepo
from app.repositories.search_run_repo import SearchRunRepo
from app.models.workflow import BriefStatus, SearchRunStatus
from app.services.brief_service import BriefService
from app.services.ranking_service import RankingService


DEMO_TENANT_ID = "demo-review-tenant"
RESEARCHER_ID = "demo_researcher"
MANAGER_ID = "demo_manager"
HAPPY_PROJECT_ID = "proj_demo_happy"
REVISION_PROJECT_ID = "proj_demo_revision"


def reset_data(root: Path) -> None:
    for relative in [
        "data/processed/briefs",
        "data/processed/projects",
        "data/processed/runs",
        "data/processed/snapshots",
        "data/audit.jsonl",
    ]:
        path = root / relative
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


def seed() -> None:
    root = Path(__file__).resolve().parents[1]
    brief_repo = BriefRepo(root_dir=str(root / "data/processed/briefs"))
    project_repo = ProjectRepo(root_dir=str(root / "data/processed/briefs"))
    run_repo = SearchRunRepo(root_dir=str(root / "data/processed/briefs"))
    snapshot_repo = SearchResultSnapshotRepo(root_dir=str(root / "data/processed/briefs"))
    audit = AuditLogger(repository=JsonlAuditRepository(str(root / "data/audit.jsonl")))
    candidate_pool = load_demo_candidates()
    ranker = RankingService()
    briefer = BriefService(llm=ClaudeClient(api_key=None))
    candidate_repo = get_candidate_repository()
    if candidate_repo is not None:
        candidate_repo.upsert_many(to_stored_candidates(DEMO_TENANT_ID))

    happy_role = {
        "title": "Infrastructure Portfolio Manager",
        "seniority": "Senior",
        "sector": "Infrastructure",
        "location": "Australia",
        "required_skills": ["infrastructure", "portfolio management", "institutional"],
        "preferred_skills": ["real assets", "manager selection"],
        "search_keywords": ["infrastructure", "portfolio manager", "real assets", "institutional"],
    }
    happy_ranked = ranker.score_candidates(happy_role, candidate_pool)
    happy_brief = briefer.generate_markdown(
        role_spec=happy_role,
        ranked_candidates=happy_ranked,
        retrieval_context=[
            {
                "doc_id": "demo_market_map",
                "title": "Infrastructure Investor Market Map",
                "text": "Australian super funds, infrastructure investors, and advisory firms are active sources of senior infrastructure portfolio talent.",
                "source": "demo-market-map",
            }
        ],
        generation_context={"project_id": HAPPY_PROJECT_ID, "tenant_id": DEMO_TENANT_ID, "demo": True},
    )

    project_repo.create_project(
        StoredSearchProject(
            project_id=HAPPY_PROJECT_ID,
            tenant_id=DEMO_TENANT_ID,
            project_name="Happy Path Infrastructure Search",
            client_name="Example Asset Management",
            role_title="Infrastructure Portfolio Manager",
            status="draft",
            created_by=RESEARCHER_ID,
            created_at="2026-03-14T09:00:00+00:00",
            updated_at="2026-03-14T09:00:00+00:00",
            metadata={"demo_path": "happy"},
        )
    )
    run_repo.create_run(
        StoredSearchRun(
            run_id="run_demo_happy_1",
            project_id=HAPPY_PROJECT_ID,
            tenant_id=DEMO_TENANT_ID,
            input_jd="Need a senior infrastructure portfolio manager in Australia.",
            parsed_role_json={"title": "Infrastructure Portfolio Manager", "location": "Australia"},
            candidate_source="local_first",
            result_count=len(happy_ranked),
            run_status=SearchRunStatus.COMPLETED,
            started_at="2026-03-14T09:05:00+00:00",
            completed_at="2026-03-14T09:08:00+00:00",
            failed_at=None,
            error_message=None,
            created_by=RESEARCHER_ID,
            metadata=None,
        )
    )
    snapshot_repo.create_snapshot(
        StoredSearchResultSnapshot(
            snapshot_id="snap_demo_happy_1",
            run_id="run_demo_happy_1",
            project_id=HAPPY_PROJECT_ID,
            tenant_id=DEMO_TENANT_ID,
            created_at="2026-03-14T09:07:00+00:00",
            top_candidates=happy_ranked[:10],
            ranking_payload=happy_ranked,
            candidate_count=len(happy_ranked),
            metadata={"demo": True},
        )
    )
    brief_repo.create_brief_version(
        StoredBrief(
            brief_id="brief_demo_happy_v1",
            project_id=HAPPY_PROJECT_ID,
            tenant_id=DEMO_TENANT_ID,
            version=1,
            content=happy_brief["markdown"],
            status=BriefStatus.EXPORTED,
            created_by=RESEARCHER_ID,
            created_at="2026-03-14T09:08:00+00:00",
            updated_at="2026-03-14T09:12:00+00:00",
            submitted_by=RESEARCHER_ID,
            submitted_at="2026-03-14T09:09:00+00:00",
            approved_by=MANAGER_ID,
            approved_at="2026-03-14T09:10:00+00:00",
            approval_notes="Approved for client share",
            exported_by=MANAGER_ID,
            exported_at="2026-03-14T09:12:00+00:00",
            run_id="run_demo_happy_1",
            role_spec={"title": "Infrastructure Portfolio Manager"},
            markdown=happy_brief["markdown"],
            citations=happy_brief["citations"] or ["demo-market-map"],
            generated_at="2026-03-14T09:08:00+00:00",
            metadata={"demo": True},
        )
    )

    revision_role = {
        "title": "Head of Real Assets",
        "seniority": "Senior",
        "sector": "Real Assets",
        "location": "Australia",
        "required_skills": ["real assets", "portfolio management", "manager selection"],
        "preferred_skills": ["infrastructure", "governance"],
        "search_keywords": ["real assets", "infrastructure", "portfolio", "institutional"],
    }
    revision_ranked = ranker.score_candidates(revision_role, candidate_pool)
    revision_brief = briefer.generate_markdown(
        role_spec=revision_role,
        ranked_candidates=revision_ranked,
        retrieval_context=[
            {
                "doc_id": "demo_real_assets",
                "title": "Real Assets Leadership Notes",
                "text": "Real assets leadership searches need shortlist depth across direct portfolio managers, governance leaders, and adjacent strategy talent.",
                "source": "demo-real-assets-notes",
            }
        ],
        generation_context={"project_id": REVISION_PROJECT_ID, "tenant_id": DEMO_TENANT_ID, "demo": True},
    )

    project_repo.create_project(
        StoredSearchProject(
            project_id=REVISION_PROJECT_ID,
            tenant_id=DEMO_TENANT_ID,
            project_name="Revision Path Real Assets Search",
            client_name="Institutional Super Fund",
            role_title="Head of Real Assets",
            status="draft",
            created_by=RESEARCHER_ID,
            created_at="2026-03-14T10:00:00+00:00",
            updated_at="2026-03-14T10:00:00+00:00",
            metadata={"demo_path": "revision"},
        )
    )
    run_repo.create_run(
        StoredSearchRun(
            run_id="run_demo_revision_1",
            project_id=REVISION_PROJECT_ID,
            tenant_id=DEMO_TENANT_ID,
            input_jd="Need a head of real assets with infrastructure and private markets context.",
            parsed_role_json={"title": "Head of Real Assets"},
            candidate_source="local_first",
            result_count=len(revision_ranked),
            run_status=SearchRunStatus.COMPLETED,
            started_at="2026-03-14T10:05:00+00:00",
            completed_at="2026-03-14T10:08:00+00:00",
            failed_at=None,
            error_message=None,
            created_by=RESEARCHER_ID,
            metadata=None,
        )
    )
    snapshot_repo.create_snapshot(
        StoredSearchResultSnapshot(
            snapshot_id="snap_demo_revision_1",
            run_id="run_demo_revision_1",
            project_id=REVISION_PROJECT_ID,
            tenant_id=DEMO_TENANT_ID,
            created_at="2026-03-14T10:07:00+00:00",
            top_candidates=revision_ranked[:10],
            ranking_payload=revision_ranked,
            candidate_count=len(revision_ranked),
            metadata={"demo": True},
        )
    )
    brief_repo.create_brief_version(
        StoredBrief(
            brief_id="brief_demo_revision_v1",
            project_id=REVISION_PROJECT_ID,
            tenant_id=DEMO_TENANT_ID,
            version=1,
            content=revision_brief["markdown"],
            status=BriefStatus.CHANGES_REQUESTED,
            created_by=RESEARCHER_ID,
            created_at="2026-03-14T10:08:00+00:00",
            updated_at="2026-03-14T10:11:00+00:00",
            submitted_by=RESEARCHER_ID,
            submitted_at="2026-03-14T10:09:00+00:00",
            rejection_notes="Please tighten sector evidence and manager-selection detail.",
            change_request_notes="Please tighten sector evidence and manager-selection detail.",
            run_id="run_demo_revision_1",
            role_spec={"title": "Head of Real Assets"},
            markdown=revision_brief["markdown"],
            citations=revision_brief["citations"] or ["demo-real-assets-notes"],
            generated_at="2026-03-14T10:08:00+00:00",
            metadata={"demo": True},
        )
    )

    events = [
        ("project_created", "create_project", "search_project", HAPPY_PROJECT_ID, HAPPY_PROJECT_ID, None, None, RESEARCHER_ID),
        ("search_run_started", "start_search_run", "search_run", "run_demo_happy_1", HAPPY_PROJECT_ID, "run_demo_happy_1", None, RESEARCHER_ID),
        ("brief_exported", "export", "brief", "brief_demo_happy_v1", HAPPY_PROJECT_ID, "run_demo_happy_1", "brief_demo_happy_v1", MANAGER_ID),
        ("project_created", "create_project", "search_project", REVISION_PROJECT_ID, REVISION_PROJECT_ID, None, None, RESEARCHER_ID),
        ("brief_changes_requested", "request_changes", "brief", "brief_demo_revision_v1", REVISION_PROJECT_ID, "run_demo_revision_1", "brief_demo_revision_v1", MANAGER_ID),
    ]
    for event_type, action, resource_type, resource_id, project_id, run_id, brief_id, actor_id in events:
        audit.log(
            AuditEvent(
                event_type=event_type,
                request_id=resource_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                payload={"demo": True},
                tenant_id=DEMO_TENANT_ID,
                project_id=project_id,
                run_id=run_id,
                brief_id=brief_id,
                actor_id=actor_id,
            )
        )

    print("Review demo seeded successfully.")
    print(f"Demo tenant: {DEMO_TENANT_ID}")
    print(f"Demo candidate pool: {len(candidate_pool)} profiles")
    print(f"Researcher user: x-user-role=researcher, x-user-id={RESEARCHER_ID}")
    print(f"Manager user: x-user-role=consultant, x-user-id={MANAGER_ID}")
    print(f"Happy path project: {HAPPY_PROJECT_ID}")
    print(f"Revision path project: {REVISION_PROJECT_ID}")
    print("Suggested review order: happy project -> revision project -> permission checks")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if args.reset:
        reset_data(root)
    seed()


if __name__ == "__main__":
    main()
