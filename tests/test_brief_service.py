from __future__ import annotations

from app.demo.demo_candidates import load_demo_candidates
from app.llm.anthropic_client import ClaudeClient
from app.services.brief_service import BriefService
from app.services.ranking_service import RankingService


def test_brief_service_returns_prompt_metadata_and_context() -> None:
    service = BriefService(llm=ClaudeClient(api_key=None))

    result = service.generate_markdown(
        role_spec={"title": "Head of Infrastructure"},
        ranked_candidates=[{"candidate_id": "cand_1", "fit_score": 91, "reasoning": ["Strong fit"]}],
        retrieval_context=[{"doc_id": "doc_1", "text": "Market context", "source": "doc-src"}],
        generation_context={"tenant_id": "tenant_a"},
    )

    assert result["prompt"]["id"] == "brief_generator"
    assert result["generation_context"]["tenant_id"] == "tenant_a"
    assert "Role Summary" in result["markdown"]


def test_brief_service_reflects_multiple_candidates_in_candidate_landscape() -> None:
    service = BriefService(llm=ClaudeClient(api_key=None))
    ranked = RankingService().score_candidates(
        {
            "title": "Infrastructure Portfolio Manager",
            "required_skills": ["infrastructure", "portfolio management", "institutional"],
            "preferred_skills": ["real assets"],
            "search_keywords": ["infrastructure", "portfolio manager", "real assets", "institutional"],
            "location": "Australia",
            "sector": "Infrastructure",
            "seniority": "senior",
        },
        load_demo_candidates(),
    )

    result = service.generate_markdown(
        role_spec={"title": "Infrastructure Portfolio Manager"},
        ranked_candidates=ranked,
        retrieval_context=[],
        generation_context={"tenant_id": "tenant_demo"},
    )

    assert "Shortlist depth:" in result["markdown"]
    assert "cand_001" in result["markdown"]
    assert "Only one candidate profile retrieved" not in result["markdown"]


def test_brief_service_formats_structured_role_spec_for_prompt() -> None:
    service = BriefService(llm=ClaudeClient(api_key=None))

    result = service.generate_markdown(
        role_spec={
            "title": {"title": "Infrastructure Portfolio Manager"},
            "seniority": "Senior",
            "sector": {"name": "Infrastructure"},
            "location": {"primary": ["Sydney", "Melbourne"], "country": "Australia"},
            "required_skills": ["infrastructure", "portfolio management"],
        },
        ranked_candidates=[{"candidate_id": "cand_1", "fit_score": 91, "reasons": ["Strong fit in Sydney."]}],
        retrieval_context=[],
        generation_context={"tenant_id": "tenant_demo"},
    )

    assert "Role Summary" in result["markdown"]
