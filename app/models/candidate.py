from __future__ import annotations

from pydantic import BaseModel


class Candidate(BaseModel):
    candidate_id: str
    full_name: str
    current_title: str
    current_company: str
    location: str
    years_experience: int | None = None
    sectors: list[str] = []
    functions: list[str] = []
    summary: str
    evidence: list[str] = []
    source_urls: list[str] = []
    confidence_score: float | None = None

