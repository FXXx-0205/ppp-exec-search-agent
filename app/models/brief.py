from __future__ import annotations

from pydantic import BaseModel


class BriefDocument(BaseModel):
    brief_id: str
    role_summary: str
    market_overview: str
    candidate_landscape: str
    shortlist_summary: str
    search_recommendations: str
    citations: list[str]
    generated_at: str
    approved: bool = False

