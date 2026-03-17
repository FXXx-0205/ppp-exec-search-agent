from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from app.repositories.brief_repo import BriefRepo
from app.repositories.search_result_snapshot_repo import SearchResultSnapshotRepo
from app.services.candidate_service import CandidateService


def test_seed_review_demo_script_reset_smoke(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/seed_review_demo.py", "--reset"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "demo-review-tenant" in result.stdout
    assert "Demo candidate pool: 30 profiles" in result.stdout
    assert "proj_demo_happy" in result.stdout
    assert "proj_demo_revision" in result.stdout


def test_seed_review_demo_produces_large_candidate_pool_and_multi_candidate_brief() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "scripts/seed_review_demo.py", "--reset"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    service = CandidateService()
    candidates = service.load_demo_candidates()

    assert len(candidates) >= 25
    assert sum(1 for c in candidates if c.get("current_company") not in {None, "", "Unknown"}) >= 25
    assert sum(1 for c in candidates if c.get("location") not in {None, "", "Unknown"}) >= 25
    assert sum(1 for c in candidates if c.get("sectors")) >= 25
    assert sum(1 for c in candidates if c.get("functions")) >= 25

    snapshot_repo = SearchResultSnapshotRepo(root_dir=str(repo_root / "data/processed/briefs"))
    brief_repo = BriefRepo(root_dir=str(repo_root / "data/processed/briefs"))
    happy_snapshot = snapshot_repo.get_snapshot("snap_demo_happy_1")
    happy_brief = brief_repo.get("brief_demo_happy_v1")

    assert happy_snapshot is not None
    assert happy_snapshot.candidate_count >= 10
    assert len(happy_snapshot.top_candidates) >= 3
    assert happy_brief is not None
    assert "Shortlist depth:" in happy_brief.markdown
    assert "Only one candidate profile retrieved" not in happy_brief.markdown
