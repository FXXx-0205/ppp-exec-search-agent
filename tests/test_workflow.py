from __future__ import annotations

from app.adapters.mock import MockATSAdapter
from app.workflows.candidate_search_graph import run_workflow


def test_run_workflow_returns_end_to_end_state() -> None:
    state = run_workflow(
        "Need a senior infrastructure portfolio manager in Australia",
        tenant_id="tenant_workflow",
        ats_adapter=MockATSAdapter(),
    )

    assert state["request_id"].startswith("req_")
    assert state["parsed_role"]["title"]
    assert state["ranking_results"]
    assert state["brief_draft"]
