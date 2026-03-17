from __future__ import annotations

from app.demo.demo_candidates import load_demo_candidates
from app.services.ranking_service import RankingService


def test_ranking_service_exposes_version_and_weights() -> None:
    service = RankingService(
        weights={
            "skill_match": 0.4,
            "seniority_match": 0.2,
            "sector_relevance": 0.15,
            "functional_similarity": 0.15,
            "location_alignment": 0.05,
            "stability_signal": 0.05,
        },
        strategy_version="v-test",
    )

    ranked = service.score_candidates(
        {"required_skills": ["infrastructure"], "location": "sydney"},
        [
            {
                "candidate_id": "cand_1",
                "summary": "Infrastructure investing leader",
                "sectors": ["Infrastructure"],
                "location": "Sydney",
            }
        ],
    )

    assert ranked[0]["ranking_version"] == "v-test"
    assert ranked[0]["ranking_weights"]["skill_match"] == 0.4
    assert ranked[0]["dimension_scores"]["location_alignment"]["match_type"] == "explicit_match"


def test_ranking_service_dimension_level_explainability() -> None:
    service = RankingService()

    ranked = service.score_candidates(
        {
            "required_skills": ["infrastructure"],
            "location": "sydney",
            "sector": "infrastructure",
            "seniority": "senior",
        },
        [
            {
                "candidate_id": "cand_1",
                "summary": "Infrastructure investing leader",
                "current_title": "Senior Infrastructure Investor",
                "sectors": ["Infrastructure"],
                "location": "Sydney",
            },
            {
                "candidate_id": "cand_2",
                "summary": "",
                "sectors": [],
                "location": None,
            },
        ],
    )

    first = next(item for item in ranked if item["candidate_id"] == "cand_1")
    second = next(item for item in ranked if item["candidate_id"] == "cand_2")

    assert first["dimension_scores"]["skill_match"]["match_type"] == "explicit_match"
    assert first["dimension_scores"]["seniority_match"]["evidence"][0]["source"] == "candidate.current_title"
    assert first["dimension_scores"]["sector_relevance"]["evidence"][0]["field"] == "sector"
    assert second["dimension_scores"]["location_alignment"]["match_type"] == "unknown"
    assert "location" in second["dimension_scores"]["location_alignment"]["missing_information"]
    assert "summary" in second["dimension_scores"]["skill_match"]["missing_information"]


def test_ranking_service_creates_clear_layers_for_demo_jd() -> None:
    service = RankingService()
    role = {
        "title": "Infrastructure Portfolio Manager",
        "required_skills": ["infrastructure", "portfolio management", "institutional"],
        "preferred_skills": ["real assets", "manager selection"],
        "search_keywords": ["infrastructure", "portfolio manager", "real assets", "institutional"],
        "location": "Australia",
        "sector": "Infrastructure",
        "seniority": "senior",
    }

    ranked = service.score_candidates(role, load_demo_candidates())
    top_ids = [candidate["candidate_id"] for candidate in ranked[:3]]
    bottom_ids = [candidate["candidate_id"] for candidate in ranked[-5:]]

    assert len(ranked) >= 25
    assert any(candidate_id in top_ids for candidate_id in {"cand_001", "cand_003", "cand_005", "cand_007"})
    assert any(candidate_id in bottom_ids for candidate_id in {"cand_020", "cand_021", "cand_022", "cand_024"})
    assert ranked[0]["fit_score"] > ranked[9]["fit_score"]
    assert ranked[9]["fit_score"] > ranked[-1]["fit_score"]
    assert ranked[0]["fit_score"] > ranked[0]["raw_fit_score"]
    assert ranked[0]["headline_reason"] != ranked[-1]["headline_reason"]
    assert ranked[0]["summary_reasons"]


def test_ranking_service_produces_candidate_specific_reasons_for_shortlist() -> None:
    service = RankingService()
    role = {
        "title": "Infrastructure Portfolio Manager",
        "required_skills": ["infrastructure", "portfolio management", "institutional"],
        "preferred_skills": ["real assets", "manager selection"],
        "search_keywords": ["infrastructure", "portfolio manager", "real assets", "institutional"],
        "location": "Australia",
        "sector": "Infrastructure",
        "seniority": "senior",
    }

    ranked = service.score_candidates(role, load_demo_candidates())[:10]

    headline_reasons = {candidate["headline_reason"] for candidate in ranked}
    rendered_reasons = {
        " | ".join((candidate.get("summary_reasons") or candidate.get("reasoning") or [""])[:2]) for candidate in ranked
    }
    fit_scores = [candidate["fit_score"] for candidate in ranked]

    assert len(headline_reasons) > 1
    assert len(rendered_reasons) > 1
    assert max(fit_scores) - min(fit_scores) >= 5


def test_ranking_service_supports_structured_location_spec() -> None:
    service = RankingService()

    ranked = service.score_candidates(
        {
            "required_skills": ["infrastructure"],
            "location": {"primary": ["Sydney", "Melbourne"], "country": "Australia"},
            "sector": "infrastructure",
            "seniority": "senior",
            "title": "Infrastructure Portfolio Manager",
        },
        [
            {
                "candidate_id": "cand_1",
                "summary": "Infrastructure investing leader",
                "current_title": "Senior Infrastructure Investor",
                "sectors": ["Infrastructure"],
                "location": "Sydney, Australia",
            }
        ],
    )

    assert ranked[0]["dimension_scores"]["location_alignment"]["score"] >= 0.9
    assert ranked[0]["fit_score"] > 0
