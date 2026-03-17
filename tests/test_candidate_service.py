from __future__ import annotations

from app.adapters.ats import CandidateProfile
from app.demo.demo_candidates import load_demo_candidates
from app.services.candidate_service import CandidateService


def test_candidate_service_normalizes_extended_adapter_fields() -> None:
    service = CandidateService()
    normalized = service._normalize_adapter_candidate(
        CandidateProfile(
            tenant_id="tenant_1",
            candidate_id="cand_1",
            full_name="Jamie Ng",
            current_title="Infrastructure Portfolio Manager",
            current_company="Example Asset Management",
            primary_email="jamie@example.com",
            location="Sydney, Australia",
            application_ids=["9001"],
            tag_names=["infrastructure", "super fund"],
            attachment_count=2,
            source_system="greenhouse",
            source_id="cand_1",
            synced_at=__import__("datetime").datetime.now(),
        )
    )

    assert normalized["current_company"] == "Example Asset Management"
    assert normalized["location"] == "Sydney, Australia"
    assert "Application IDs: 9001" in normalized["evidence"]
    assert "Tags: infrastructure, super fund" in normalized["evidence"]
    assert "Attachments available: 2" in normalized["evidence"]
    assert normalized["confidence_score"] is None


def test_candidate_service_demo_pool_is_rich_and_searchable() -> None:
    service = CandidateService()
    candidates = service.load_demo_candidates()
    filtered = service.filter_candidates(
        {
            "search_keywords": ["infrastructure", "portfolio manager", "institutional"],
            "required_skills": ["portfolio management"],
        },
        candidates,
    )

    assert len(candidates) >= 25
    assert any(candidate["location"].startswith("Sydney") for candidate in candidates)
    assert any("Government" in candidate["sectors"] for candidate in candidates)
    assert any("Distribution" in candidate["functions"] for candidate in candidates)
    assert any(candidate.get("employment_history") for candidate in candidates)
    assert any(candidate.get("skills") for candidate in candidates)
    assert len(filtered) > 5
    assert filtered[0]["candidate_id"] in {candidate["candidate_id"] for candidate in load_demo_candidates()[:10]}


def test_candidate_service_preserves_demo_confidence_when_enriching_adapter_candidates() -> None:
    service = CandidateService()
    demo_pool = service.load_demo_candidates()
    demo_lookup = {candidate["candidate_id"]: candidate for candidate in demo_pool}

    normalized = service._normalize_adapter_candidate(
        CandidateProfile(
            tenant_id="tenant_1",
            candidate_id="cand_005",
            full_name="Grace Wu",
            current_title="Infrastructure Portfolio Manager",
            current_company="IFM Investors",
            primary_email="grace@example.com",
            location="Sydney, Australia",
            application_ids=["9002"],
            tag_names=["infrastructure"],
            attachment_count=1,
            source_system="mock-ats",
            source_id="cand_005",
            synced_at=__import__("datetime").datetime.now(),
        )
    )
    enriched = __import__("app.demo.demo_candidates", fromlist=["enrich_with_demo_fields"]).enrich_with_demo_fields(normalized, demo_lookup)

    assert enriched["confidence_score"] == demo_lookup["cand_005"]["confidence_score"]
