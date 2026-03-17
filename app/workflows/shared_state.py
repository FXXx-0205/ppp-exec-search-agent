from __future__ import annotations

from typing import TypedDict


class SearchState(TypedDict, total=False):
    request_id: str
    raw_user_input: str
    parsed_role: dict
    retrieval_context: list[dict]
    candidate_pool: list[dict]
    ranking_results: list[dict]
    brief_draft: str
    critique_feedback: list[str]
    audit: list[dict]
    errors: list[str]

