from __future__ import annotations

import json
from typing import Any, cast

import httpx
import pytest
from pydantic import ValidationError

from app.llm.anthropic_client import ClaudeClient
from app.ppp.enrichment import (
    CandidatePublicProfileLookupInput,
    CandidatePublicProfileLookupTool,
    _firm_context_status,
)
from app.ppp.pipeline import (
    PPPTaskError,
    _load_candidates_csv,
    _load_role_spec,
    _resolve_research_mode,
    _stabilize_candidate_brief,
    run_ppp_pipeline,
)
from app.ppp.qa import run_bundle_qa, validate_output_bundle
from app.ppp.research import ResearchClientError, TavilyResearchClient
from app.ppp.role_spec import load_role_spec_json_text, normalize_role_spec, parse_role_spec_text
from app.ppp.schema import validate_candidate_brief, validate_output_document
from app.ppp.validator import parse_candidate_response, validate_and_repair_candidate_payload


def _candidate_from_prompt(payload: dict[str, object]) -> dict[str, str]:
    if "research_package" in payload:
        research_package = payload["research_package"]
        assert isinstance(research_package, dict)
        identity = research_package["candidate_identity"]
        assert isinstance(identity, dict)
        return {
            "full_name": str(identity["full_name"]),
            "current_employer": str(identity["current_employer"]),
            "current_title": str(identity["current_title"]),
        }
    if "candidate_identity" in payload:
        identity = payload["candidate_identity"]
        assert isinstance(identity, dict)
        normalized = payload.get("normalized_evidence", {})
        assert isinstance(normalized, dict)
        return {
            "full_name": str(identity["full_name"]),
            "current_employer": str(normalized.get("verified_employer") or identity.get("current_employer") or ""),
            "current_title": str(normalized.get("verified_title") or identity.get("current_title") or ""),
        }
    minimal_schema = payload["minimal_required_schema"]
    assert isinstance(minimal_schema, dict)
    current_role = minimal_schema["current_role"]
    assert isinstance(current_role, dict)
    return {
        "full_name": str(minimal_schema["full_name"]),
        "current_employer": str(current_role["employer"]),
        "current_title": str(current_role["title"]),
    }


def _allowed_facts_from_prompt(payload: dict[str, object]) -> dict[str, object]:
    if "allowed_facts" in payload:
        allowed_facts = payload["allowed_facts"]
        assert isinstance(allowed_facts, dict)
        return allowed_facts
    if "minimal_required_schema" in payload:
        minimal_schema = payload["minimal_required_schema"]
        assert isinstance(minimal_schema, dict)
        current_role = minimal_schema["current_role"]
        assert isinstance(current_role, dict)
        return {
            "current_role": {
                "title": current_role["title"],
                "employer": current_role["employer"],
            },
            "tenure_years": {
                "value": current_role["tenure_years"],
            },
            "firm_aum_context": {
                "preferred_statement": minimal_schema["firm_aum_context"],
            },
        }
    normalized = payload["normalized_evidence"]
    assert isinstance(normalized, dict)
    current_role = payload["required_output_shape"]["current_role"]
    assert isinstance(current_role, dict)
    return {
        "current_role": {
            "title": current_role["title"],
            "employer": current_role["employer"],
        },
        "tenure_years": {
            "value": normalized["tenure_years"],
        },
        "firm_aum_context": {
            "preferred_statement": payload["firm_aum_context"],
        },
    }


def _allowed_fact_section(allowed_facts: dict[str, object], key: str) -> dict[str, Any]:
    section = allowed_facts[key]
    assert isinstance(section, dict)
    return cast(dict[str, Any], section)


def _candidate_id_from_prompt(payload: dict[str, object]) -> str:
    if "schema_rules" in payload:
        schema_rules = payload["schema_rules"]
        assert isinstance(schema_rules, dict)
        return str(schema_rules["candidate_id"])
    if "candidate_identity" in payload:
        identity = payload["candidate_identity"]
        assert isinstance(identity, dict)
        return str(identity["candidate_id"])
    minimal_schema = payload["minimal_required_schema"]
    assert isinstance(minimal_schema, dict)
    return str(minimal_schema["candidate_id"])


def _fake_normalized_evidence_json(user_prompt: str) -> str:
    payload = json.loads(user_prompt)
    identity = payload["candidate_identity"]
    assert isinstance(identity, dict)
    return json.dumps(
        {
            "match_confidence_state": "verified_match",
            "strongest_title_employer_signal": f"{identity['current_title']} at {identity['current_employer']}",
            "public_profile_confidence_notes": ["Verified in test stub."],
            "firm_context_confidence_notes": ["Firm context available in test stub."],
            "role_relevant_distribution_signals": ["Relevant distribution signal in test stub."],
            "uncertainty_notes": ["Some scope details remain uncertain."],
            "verified_employer": identity["current_employer"],
            "verified_title": identity["current_title"],
            "tenure_years": 2.5,
        }
    )


def _build_enrichment(
    tmp_path,
    *,
    full_name: str = "Example Candidate",
    employer: str = "Example AM",
    title: str = "Head of Distribution",
    fixture: dict[str, Any] | None = None,
):
    fixture_path = tmp_path / f"{full_name.replace(' ', '_')}_fixture.json"
    fixture_path.write_text(json.dumps({full_name: fixture or {}}), encoding="utf-8")
    tool = CandidatePublicProfileLookupTool(fixture_path=str(fixture_path))
    return tool.run(
        CandidatePublicProfileLookupInput(
            candidate_id="candidate_1",
            full_name=full_name,
            current_employer=employer,
            current_title=title,
            linkedin_url="https://example.com/linkedin",
        )
    )


def _build_candidate_payload(
    enrichment,
    *,
    career_narrative: str,
    role_fit_justification: str,
    outreach_hook: str,
) -> dict[str, Any]:
    candidates = [
        {
            "candidate_id": enrichment.candidate_id,
            "full_name": enrichment.full_name,
            "current_role": {
                "title": enrichment.current_title,
                "employer": enrichment.current_employer,
                "tenure_years": enrichment.inferred_tenure_years or 2.0,
            },
            "career_narrative": career_narrative,
            "experience_tags": ["distribution", "client coverage"],
            "firm_aum_context": enrichment.firm_aum_context,
            "mobility_signal": {
                "score": 3,
                "rationale": "Public chronology remains limited. No direct public signal of mobility is visible, so openness should be treated as uncertain pending conversation.",
            },
            "role_fit": {
                "role": "Head of Distribution / National BDM",
                "score": 7,
                "justification": role_fit_justification,
            },
            "outreach_hook": outreach_hook,
        }
    ]
    for idx in range(2, 6):
        candidates.append(
            {
                "candidate_id": f"candidate_{idx}",
                "full_name": f"Placeholder {idx}",
                "current_role": {"title": "Head of Distribution", "employer": f"Firm {idx}", "tenure_years": 2.0},
                "career_narrative": (
                    f"Placeholder {idx} is currently Head of Distribution at Firm {idx}, with public evidence pointing to an institutional lane. "
                    "For the target mandate, the relevance comes from institutional coverage already sitting close to the brief on seniority. "
                    "What remains unclear from public evidence is team leadership scale behind the remit."
                ),
                "experience_tags": ["distribution"],
                "firm_aum_context": f"Firm {idx} context requires verification.",
                "mobility_signal": {"score": 3, "rationale": "Public chronology is limited. Move readiness remains uncertain pending conversation."},
                "role_fit": {
                    "role": "Head of Distribution / National BDM",
                    "score": 6,
                    "justification": f"This profile is in frame because Head of Distribution at Firm {idx} points to institutional coverage at senior level. Key unresolved point: team leadership scale behind the remit, which is not verified publicly. Screening priority: What size team has this role actually led?",
                },
                "outreach_hook": f"We're working on a distribution leadership brief where your institutional coverage at Firm {idx} looks especially relevant.",
            }
        )
    return {"candidates": candidates}


def test_load_candidates_csv_requires_exactly_five_rows(tmp_path) -> None:
    path = tmp_path / "candidates.csv"
    path.write_text(
        "full_name,current_employer,current_title,linkedin_url\n"
        "A,One,Title,https://example.com/1\n",
        encoding="utf-8",
    )

    with pytest.raises(PPPTaskError, match="expected 5 candidates"):
        _load_candidates_csv(path)


def test_load_candidates_csv_requires_columns(tmp_path) -> None:
    path = tmp_path / "candidates.csv"
    path.write_text("full_name,current_employer,current_title\nA,One,Title\n", encoding="utf-8")

    with pytest.raises(PPPTaskError, match="missing required columns"):
        _load_candidates_csv(path)


def test_load_role_spec_requires_object_json(tmp_path) -> None:
    path = tmp_path / "role.json"
    path.write_text('["not-an-object"]', encoding="utf-8")

    with pytest.raises(PPPTaskError, match="must be an object"):
        _load_role_spec(path)


def test_load_role_spec_normalizes_parser_style_payload(tmp_path) -> None:
    path = tmp_path / "role.json"
    path.write_text(
        json.dumps(
            {
                "title": "Head of Distribution / National BDM",
                "required_skills": ["Institutional sales", "Platform relationships"],
                "sector": "Funds Management",
            }
        ),
        encoding="utf-8",
    )

    role_spec = _load_role_spec(path)
    assert role_spec["role"] == "Head of Distribution / National BDM"
    assert role_spec["requirements"] == ["Institutional sales", "Platform relationships"]
    assert role_spec["sector"] == "Funds Management"


def test_role_spec_helpers_normalize_and_parse_text() -> None:
    normalized = normalize_role_spec(
        {
            "title": "Head of Distribution / National BDM",
            "required_skills": ["Institutional sales", "Team leadership"],
            "preferred_skills": ["Alternatives"],
        }
    )
    assert normalized["role"] == "Head of Distribution / National BDM"
    assert normalized["requirements"] == ["Institutional sales", "Team leadership"]

    class FakeClient:
        def generate_text(self, *, system_prompt: str, user_prompt: str, model: str, max_tokens: int, extra=None) -> str:
            assert "structured role specification" in system_prompt
            assert "distribution" in user_prompt.lower()
            return json.dumps(
                {
                    "title": "Head of Distribution / National BDM",
                    "seniority": "Head",
                    "required_skills": ["Institutional sales", "Platform relationships"],
                }
            )

    parsed = parse_role_spec_text(
        text="Head of Distribution mandate focused on institutional sales and platform relationships.",
        client=FakeClient(),  # type: ignore[arg-type]
        model="claude-sonnet-4-5",
    )
    assert parsed["role"] == "Head of Distribution / National BDM"
    assert parsed["requirements"] == ["Institutional sales", "Platform relationships"]

    loaded = load_role_spec_json_text(json.dumps(parsed))
    assert loaded["role"] == "Head of Distribution / National BDM"


def test_claude_client_handles_real_tool_roundtrip() -> None:
    class FakeBlock:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FakeResponse:
        def __init__(self, content):
            self.content = content

    class FakeMessagesAPI:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                assert "tools" in kwargs
                return FakeResponse(
                    [
                        FakeBlock(
                            type="tool_use",
                            id="toolu_1",
                            name="normalize_candidate_evidence",
                            input={"candidate_identity": {"candidate_id": "candidate_1"}, "research_package": {}, "role_spec": {}},
                        )
                    ]
                )
            tool_result_blocks = kwargs["messages"][-1]["content"]
            assert tool_result_blocks[0]["type"] == "tool_result"
            payload = json.loads(tool_result_blocks[0]["content"])
            assert payload["match_confidence_state"] == "verified_match"
            return FakeResponse([FakeBlock(type="text", text='{"candidate_id":"candidate_1"}')])

    client = ClaudeClient(api_key=None)
    client._client = type("FakeAnthropicClient", (), {"messages": FakeMessagesAPI()})()

    tool_calls: list[tuple[str, dict[str, Any]]] = []

    text = client.run_tool_phase_once(
        system_prompt="system",
        user_prompt=json.dumps({"candidate_identity": {"candidate_id": "candidate_1"}, "research_package": {}, "role_spec": {}}),
        model="claude-sonnet-4-5",
        max_tokens=300,
        extra_payload={},
        tool_definitions=[
            {
                "name": "normalize_candidate_evidence",
                "description": "Normalize evidence",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        tool_choice={"type": "any"},
        tool_runner=lambda tool_name, tool_input: (
            tool_calls.append((tool_name, tool_input)) or {"match_confidence_state": "verified_match"}
        ),
    )

    assert json.loads(text)["candidate_id"] == "candidate_1"
    assert tool_calls[0][0] == "normalize_candidate_evidence"


def test_validate_candidate_brief_rejects_null_tenure() -> None:
    with pytest.raises(ValidationError):
        validate_candidate_brief(
            {
                "candidate_id": "candidate_1",
                "full_name": "Example Candidate",
                "current_role": {
                    "title": "Head of Distribution",
                    "employer": "Example AM",
                    "tenure_years": None,
                },
                "career_narrative": "Sentence one. Sentence two. Sentence three.",
                "experience_tags": ["distribution"],
                "firm_aum_context": "Example AM context.",
                "mobility_signal": {"score": 3, "rationale": "Sentence one. Sentence two."},
                "role_fit": {
                    "role": "Head of Distribution / National BDM",
                    "score": 6,
                    "justification": "Sentence one. Sentence two. Sentence three.",
                },
                "outreach_hook": "One sentence only.",
            }
        )


def test_parse_candidate_response_rejects_control_packet() -> None:
    with pytest.raises(ValueError, match="control packet"):
        parse_candidate_response(
            json.dumps(
                {
                    "task": "Generate candidate briefing",
                    "reasoning_contract": {"use_only_supported_claims": True},
                    "schema_rules": {"candidate_id": "candidate_1"},
                    "candidate_identity": {"candidate_id": "candidate_1", "full_name": "Example Candidate"},
                }
            )
        )


def test_resolve_research_mode_defaults_to_live_when_tavily_key_present(monkeypatch) -> None:
    monkeypatch.setattr("app.ppp.pipeline.settings.tavily_api_key", "test-tavily-key")
    monkeypatch.setattr("app.ppp.pipeline.settings.ppp_research_mode", "fixture")

    assert _resolve_research_mode(None) == "live"


def test_run_ppp_pipeline_writes_valid_output(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "candidates.csv"
    input_path.write_text(
        "full_name,current_employer,current_title,linkedin_url\n"
        "A,One,Title,https://example.com/1\n"
        "B,Two,Title,https://example.com/2\n"
        "C,Three,Title,https://example.com/3\n"
        "D,Four,Title,https://example.com/4\n"
        "E,Five,Title,https://example.com/5\n",
        encoding="utf-8",
    )
    role_spec_path = tmp_path / "role.json"
    role_spec_path.write_text(json.dumps({"role": "Head of Distribution / National BDM"}), encoding="utf-8")
    output_path = tmp_path / "output.json"

    monkeypatch.setattr("app.ppp.pipeline.settings.anthropic_api_key", "test-key")
    def fake_generate_text(self, *, system_prompt: str, user_prompt: str, model: str, max_tokens: int, extra=None):
        payload = json.loads(user_prompt)
        candidate = _candidate_from_prompt(payload)
        allowed_facts = _allowed_facts_from_prompt(payload)
        tenure = _allowed_fact_section(allowed_facts, "tenure_years")
        candidate_id = _candidate_id_from_prompt(payload)
        return json.dumps(
            {
                "candidate_id": candidate_id,
                "full_name": candidate["full_name"],
                "current_role": {
                    "title": candidate["current_title"],
                    "employer": candidate["current_employer"],
                    "tenure_years": tenure["value"] or 2.5,
                },
                "career_narrative": f"{candidate['full_name']} is listed in the task input as {candidate['current_title']} at {candidate['current_employer']}. "
                f"The title '{candidate['current_title']}' suggests senior distribution channel responsibility. "
                "Public data remains limited, so some details are framed cautiously.",
                "experience_tags": ["distribution", "leadership"],
                "firm_aum_context": (
                    f"{candidate['current_employer']} appears to be a established market participant funds-management firm; "
                    "exact AUM requires live public verification; distribution exposure inferred from the candidate's title."
                ),
                "mobility_signal": {
                    "score": 3,
                    "rationale": "Public chronology remains limited for this profile. No direct public signal of mobility is visible, so openness should be treated as uncertain pending conversation."
                },
                "role_fit": {
                    "role": "Head of Distribution / National BDM",
                    "score": 7,
                    "justification": f"{candidate['current_employer']} and the title {candidate['current_title']} suggest relevant distribution leadership exposure. "
                    "The available public evidence still leaves channel depth and team scale uncertain. "
                    "Further verification is required before the supported distribution exposure can be treated as complete.",
                },
                "outreach_hook": f"Your experience at {candidate['current_employer']} could make this distribution leadership brief worth discussing.",
            }
        )

    monkeypatch.setattr("app.ppp.pipeline.ClaudeClient.generate_text", fake_generate_text)
    monkeypatch.setattr("app.ppp.pipeline.ClaudeClient.run_tool_phase_once", lambda self, **kwargs: _fake_normalized_evidence_json(kwargs["user_prompt"]))

    result = run_ppp_pipeline(
        input_path=str(input_path),
        output_path=str(output_path),
        role_spec_path=str(role_spec_path),
        model="claude-sonnet-4-5",
        intermediate_dir=str(tmp_path / "intermediate"),
        research_mode="fixture",
    )

    assert output_path.exists()
    body = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(body["candidates"]) == 5
    assert result.candidates[0].candidate_id == "candidate_1"


def test_load_candidates_csv_rejects_empty_field(tmp_path) -> None:
    path = tmp_path / "candidates.csv"
    path.write_text(
        "full_name,current_employer,current_title,linkedin_url\n"
        "A,One,Title,https://example.com/1\n"
        "B,Two,Title,https://example.com/2\n"
        "C,Three,Title,https://example.com/3\n"
        "D,Four,Title,https://example.com/4\n"
        "E,Five,,https://example.com/5\n",
        encoding="utf-8",
    )

    with pytest.raises(PPPTaskError, match="Candidate row 5 is invalid"):
        _load_candidates_csv(path)


def test_validate_and_repair_candidate_payload_repairs_minor_type_issues() -> None:
    candidate = validate_and_repair_candidate_payload(
        {
            "current_role": {"title": "Head of Distribution", "employer": "Example AM", "tenure_years": "2.5 years"},
            "career_narrative": "Candidate has built a strong distribution career. The profile suggests national coverage. "
            "Recent role progression indicates seniority.",
            "experience_tags": "distribution, institutional sales, leadership",
            "firm_aum_context": "Public AUM context remains partially verified.",
            "mobility_signal": {"score": "3", "rationale": "Tenure looks established enough to suggest possible openness."},
            "role_fit": {"score": "8", "justification": "Strong directional fit for the target mandate."},
            "outreach_hook": "Your distribution leadership experience looks highly relevant to a current search mandate.",
        },
        candidate_id="candidate_1",
        full_name="Example Candidate",
    )

    assert candidate.candidate_id == "candidate_1"
    assert candidate.full_name == "Example Candidate"
    assert candidate.current_role.tenure_years == 2.5
    assert candidate.mobility_signal.score == 3
    assert candidate.role_fit.score == 8
    assert candidate.role_fit.role == "Head of Distribution / National BDM"
    assert candidate.experience_tags == ["distribution", "institutional sales", "team leadership"]


def test_validate_and_repair_candidate_payload_rejects_out_of_range_scores() -> None:
    with pytest.raises(ValidationError, match="score"):
        validate_and_repair_candidate_payload(
            {
                "candidate_id": "candidate_1",
                "full_name": "Example Candidate",
                "current_role": {"title": "Head of Distribution", "employer": "Example AM", "tenure_years": 2.5},
                "career_narrative": "Candidate has built a strong distribution career. The profile suggests national coverage. "
                "Recent role progression indicates seniority.",
                "experience_tags": ["distribution"],
                "firm_aum_context": "Public AUM context remains partially verified.",
                "mobility_signal": {"score": 6, "rationale": "Too high for schema."},
                "role_fit": {
                    "role": "Head of Distribution / National BDM",
                    "score": 11,
                    "justification": "Too high for schema.",
                },
                "outreach_hook": "Your distribution leadership experience looks highly relevant to a current search mandate.",
            },
            candidate_id="candidate_1",
            full_name="Example Candidate",
        )


def test_lookup_tool_returns_enrichment_and_saves_artifact(tmp_path) -> None:
    fixture_path = tmp_path / "fixtures.json"
    fixture_path.write_text(
        json.dumps(
            {
                "Example Candidate": {
                    "tenure_years": 2.4,
                    "firm_aum_context": "Example firm context.",
                    "verified_public_snippets": ["Example Candidate is listed as Head of Distribution at Example AM."],
                }
            }
        ),
        encoding="utf-8",
    )
    tool = CandidatePublicProfileLookupTool(fixture_path=str(fixture_path))
    tool_input = CandidatePublicProfileLookupInput(
        candidate_id="candidate_1",
        full_name="Example Candidate",
        current_employer="Example AM",
        current_title="Head of Distribution",
        linkedin_url="https://example.com/linkedin",
    )

    result = tool.run(tool_input)
    saved = tool.save_intermediate(result, output_dir=str(tmp_path / "intermediate"))

    assert result.tool_name == "candidate_public_profile_lookup"
    assert result.inferred_tenure_years == 2.4
    assert result.sources[0].label
    assert result.claims
    assert saved.exists()


def test_lookup_tool_live_mode_uses_research_client(tmp_path) -> None:
    class FakeResearchClient:
        def lookup_candidate(self, tool_input: CandidatePublicProfileLookupInput):
            return type(
                "Payload",
                (),
                {
                    "mode": "live_web_fake",
                    "fixture": {
                        "verified_public_snippets": [f"{tool_input.full_name} appears on a public leadership page."],
                        "tenure_years": 1.8,
                        "firm_aum_context": f"{tool_input.current_employer} appears in public-web results with firm-scale references.",
                        "likely_channel_evidence": ["Public snippets mention wholesale and institutional distribution."],
                        "likely_experience_evidence": ["Public snippets tie the candidate to a senior client coverage remit."],
                        "mobility_evidence": ["Recent public references suggest checking for a current transition window."],
                        "missing_fields": ["verified team size"],
                        "uncertain_fields": ["precise current-role tenure"],
                        "confidence_notes": ["Live public-web research succeeded."],
                        "combined_context": "=== CANDIDATE PROFILE ===\nProfile result\n\n=== FIRM CONTEXT & AUM ===\nFirm result",
                        "sources": [
                            {
                                "label": "Leadership page",
                                "source_type": "company_site",
                                "url": "https://example.com/leadership",
                                "confidence": "medium",
                            }
                        ],
                    },
                },
            )()

    tool = CandidatePublicProfileLookupTool(
        fixture_path=str(tmp_path / "fixtures.json"),
        mode="live",
        research_client=FakeResearchClient(),
    )
    tool_input = CandidatePublicProfileLookupInput(
        candidate_id="candidate_1",
        full_name="Example Candidate",
        current_employer="Example AM",
        current_title="Head of Distribution",
        linkedin_url="https://example.com/linkedin",
    )

    result = tool.run(tool_input)

    assert result.tool_mode == "live_web"
    assert result.claims[0].statement == "Example Candidate appears on a public leadership page."
    assert any("=== FIRM CONTEXT & AUM ===" in note for note in result.verification.confidence_notes)


def test_lookup_tool_auto_mode_falls_back_to_fixture_on_live_error(tmp_path) -> None:
    fixture_path = tmp_path / "fixtures.json"
    fixture_path.write_text(
        json.dumps(
            {
                "Example Candidate": {
                    "verified_public_snippets": ["Fixture snippet."],
                    "firm_aum_context": "Fixture context.",
                }
            }
        ),
        encoding="utf-8",
    )

    class FailingResearchClient:
        def lookup_candidate(self, tool_input: CandidatePublicProfileLookupInput):
            raise ResearchClientError("network down")

    tool = CandidatePublicProfileLookupTool(
        fixture_path=str(fixture_path),
        mode="auto",
        research_client=FailingResearchClient(),
    )
    tool_input = CandidatePublicProfileLookupInput(
        candidate_id="candidate_1",
        full_name="Example Candidate",
        current_employer="Example AM",
        current_title="Head of Distribution",
        linkedin_url="https://example.com/linkedin",
    )

    result = tool.run(tool_input)

    assert result.tool_mode == "fixture_fallback"
    assert result.claims[0].statement == "Fixture snippet."
    assert any("fell back to the controlled fixture" in note for note in result.verification.confidence_notes)


@pytest.mark.parametrize(
    ("title", "fixture", "expected"),
    [
        ("Director, Institutional Sales", {}, "institutional"),
        ("Senior BDM Wholesale", {}, "wholesale"),
        ("Head of Retail Sales", {}, "retail"),
        (
            "Head of Distribution",
            {"likely_channel_evidence": ["Public snippets mention institutional and wholesale distribution coverage."]},
            "mixed",
        ),
        ("Head of Distribution", {"likely_channel_evidence": ["Public snippets mention distribution leadership only."]}, "mixed"),
    ],
)
def test_recruiter_signals_derives_channel_orientation(tmp_path, title: str, fixture: dict[str, Any], expected: str) -> None:
    enrichment = _build_enrichment(tmp_path, title=title, fixture=fixture)
    assert enrichment.recruiter_signals.channel_orientation == expected


@pytest.mark.parametrize(
    ("title", "fixture", "expected"),
    [
        ("Head of Distribution", {}, "direct_match"),
        ("Director, Institutional Sales", {}, "adjacent_match"),
        ("Senior BDM Wholesale", {}, "step_up_candidate"),
        ("Operations Manager", {"likely_channel_evidence": ["Public snippets mention operational leadership only."]}, "unclear_fit"),
    ],
)
def test_recruiter_signals_derives_mandate_similarity(tmp_path, title: str, fixture: dict[str, Any], expected: str) -> None:
    enrichment = _build_enrichment(tmp_path, title=title, fixture=fixture)
    assert enrichment.recruiter_signals.mandate_similarity == expected


def test_deliberate_bad_candidate_is_classified_as_unclear_fit(tmp_path) -> None:
    enrichment = _build_enrichment(
        tmp_path,
        full_name="Daniel Koleth",
        employer="Example Investments",
        title="Senior Investment Analyst",
        fixture={
            "verified_public_snippets": ["Daniel Koleth is listed as Senior Investment Analyst at Example Investments."],
            "likely_channel_evidence": ["Public snippets reference equities research and investment analysis rather than sales or distribution coverage."],
            "likely_experience_evidence": ["Current remit appears research-oriented with no direct evidence of client distribution leadership."],
            "uncertain_fields": ["direct client coverage", "distribution remit", "team leadership scale"],
            "missing_fields": ["platform / IFA / super-fund network depth"],
        },
    )

    assert enrichment.recruiter_signals.channel_orientation == "unclear"
    assert enrichment.recruiter_signals.mandate_similarity == "unclear_fit"


def test_fixture_enrichment_defaults_identity_resolution_to_verified_match(tmp_path) -> None:
    enrichment = _build_enrichment(tmp_path, title="Head of Distribution")

    assert enrichment.identity_resolution.status == "verified_match"
    assert "public sources verify" in enrichment.identity_resolution.rationale.lower()


def test_recruiter_signals_sell_points_and_gaps_are_claim_and_verification_backed(tmp_path) -> None:
    enrichment = _build_enrichment(
        tmp_path,
        title="Director, Institutional Sales",
        fixture={
            "verified_public_snippets": ["Example Candidate is listed as Director, Institutional Sales at Example AM."],
            "likely_channel_evidence": ["Public snippets reference institutional distribution coverage."],
            "likely_experience_evidence": ["Current employer/title pairing suggests relevance to funds-management client coverage at Example AM."],
            "missing_fields": ["verified reporting line / team size"],
            "uncertain_fields": ["direct Australian institutional coverage"],
        },
    )

    assert enrichment.recruiter_signals.key_sell_points
    assert len(enrichment.recruiter_signals.key_sell_points) <= 2
    assert "current remit is anchored in senior institutional coverage" in enrichment.recruiter_signals.key_sell_points
    assert len(enrichment.recruiter_signals.key_gaps) <= 2
    assert enrichment.recruiter_signals.key_gaps[0] == "direct market institutional network depth is not verified publicly"
    assert "team leadership scale behind the remit is not verified publicly" in enrichment.recruiter_signals.key_gaps
    assert enrichment.recruiter_signals.scope_signal == "unclear"
    assert enrichment.recruiter_signals.seniority_signal == "director_level"
    assert enrichment.recruiter_signals.evidence_strength in {"moderate", "strong"}
    assert enrichment.recruiter_signals.screening_priority_question == "How much direct market institutional network depth sits behind the current title?"
    research_package = enrichment.research_package()
    assert research_package["recruiter_signals"]["channel_orientation"] == "institutional"


@pytest.mark.parametrize(
    ("title", "fixture", "expected_scope", "expected_seniority"),
    [
        ("Head of Distribution Wealth Management Australia and New Zealand", {}, "anz", "head_level"),
        ("Global Head of Institutional Distribution", {}, "global", "head_level"),
        ("Regional Director, Wholesale Distribution APAC", {}, "regional", "director_level"),
        ("National BDM Wholesale", {}, "national", "bdm_level"),
    ],
)
def test_recruiter_signals_derives_scope_and_seniority(tmp_path, title: str, fixture: dict[str, Any], expected_scope: str, expected_seniority: str) -> None:
    enrichment = _build_enrichment(tmp_path, title=title, fixture=fixture)
    assert enrichment.recruiter_signals.scope_signal == expected_scope
    assert enrichment.recruiter_signals.seniority_signal == expected_seniority


def test_recruiter_signals_prioritize_commercial_gaps_over_low_value_admin_gaps(tmp_path) -> None:
    enrichment = _build_enrichment(
        tmp_path,
        title="Head of Distribution",
        fixture={
            "uncertain_fields": ["exact current-role start date", "direct Australian institutional coverage", "exact firm AUM"],
            "missing_fields": ["verified reporting line / team size"],
        },
    )

    assert len(enrichment.recruiter_signals.key_gaps) <= 2
    assert any("network depth" in gap for gap in enrichment.recruiter_signals.key_gaps)
    assert any("team leadership scale" in gap for gap in enrichment.recruiter_signals.key_gaps)
    assert all("aum" not in gap.lower() for gap in enrichment.recruiter_signals.key_gaps)


def test_stabilized_output_differentiates_lane_scope_and_hook_across_candidates(tmp_path) -> None:
    institutional = _build_enrichment(
        tmp_path,
        full_name="Institutional Candidate",
        employer="Institutional AM",
        title="Director, Institutional Sales",
        fixture={"likely_channel_evidence": ["Public snippets reference institutional distribution coverage."]},
    )
    wealth = _build_enrichment(
        tmp_path,
        full_name="Wealth Candidate",
        employer="Wealth AM",
        title="Head of Distribution Wealth Management Australia and New Zealand",
        fixture={"likely_channel_evidence": ["Public snippets reference wealth distribution and adviser coverage."]},
    )
    wholesale = _build_enrichment(
        tmp_path,
        full_name="Wholesale Candidate",
        employer="Wholesale AM",
        title="Senior BDM Wholesale",
        fixture={"likely_channel_evidence": ["Public snippets reference wholesale and IFA distribution coverage."]},
    )

    base_payload = {
        "candidate_id": "candidate_1",
        "full_name": "Placeholder",
        "current_role": {"title": "Placeholder", "employer": "Placeholder", "tenure_years": 2.0},
        "career_narrative": "Placeholder one. Placeholder two. Placeholder three.",
        "experience_tags": ["distribution"],
        "firm_aum_context": "Placeholder context.",
        "mobility_signal": {"score": 3, "rationale": "Public chronology remains limited. No direct public signal of mobility is visible, so openness should be treated as uncertain pending conversation."},
        "role_fit": {"role": "Head of Distribution / National BDM", "score": 7, "justification": "Placeholder one. Placeholder two. Placeholder three."},
        "outreach_hook": "Placeholder hook.",
    }

    institutional_brief = _stabilize_candidate_brief(candidate_brief=validate_and_repair_candidate_payload(base_payload, candidate_id="candidate_1", full_name="Institutional Candidate"), enrichment=institutional)
    wealth_brief = _stabilize_candidate_brief(candidate_brief=validate_and_repair_candidate_payload(base_payload, candidate_id="candidate_2", full_name="Wealth Candidate"), enrichment=wealth)
    wholesale_brief = _stabilize_candidate_brief(candidate_brief=validate_and_repair_candidate_payload(base_payload, candidate_id="candidate_3", full_name="Wholesale Candidate"), enrichment=wholesale)

    assert "institutional lane" in institutional_brief.career_narrative.lower()
    assert "wealth lane" in wealth_brief.career_narrative.lower()
    assert "wholesale lane" in wholesale_brief.career_narrative.lower()
    assert "anz scope" in wealth_brief.career_narrative.lower()
    assert institutional_brief.role_fit.justification != wholesale_brief.role_fit.justification
    assert wealth_brief.outreach_hook != wholesale_brief.outreach_hook
    assert "institutional" in institutional_brief.outreach_hook.lower()
    assert "broader channel stretch" in institutional_brief.outreach_hook.lower()
    assert "wealth and intermediary distribution leadership" in wealth_brief.outreach_hook.lower()
    assert "step-up conversation" in wholesale_brief.outreach_hook.lower()


def test_stabilized_output_keeps_deliberate_bad_candidate_low_confidence(tmp_path) -> None:
    enrichment = _build_enrichment(
        tmp_path,
        full_name="Daniel Koleth",
        employer="Example Investments",
        title="Senior Investment Analyst",
        fixture={
            "verified_public_snippets": ["Daniel Koleth is listed as Senior Investment Analyst at Example Investments."],
            "likely_channel_evidence": ["Public snippets reference equities research and investment analysis rather than sales or distribution coverage."],
            "likely_experience_evidence": ["Current remit appears research-oriented with no direct evidence of client distribution leadership."],
            "uncertain_fields": ["direct client coverage", "distribution remit", "team leadership scale"],
            "missing_fields": ["platform / IFA / super-fund network depth"],
        },
    )
    output = validate_output_document(
        _build_candidate_payload(
            enrichment,
            career_narrative=(
                "Daniel Koleth is currently Senior Investment Analyst at Example Investments. "
                "The profile appears relevant to the target mandate because of broader market exposure. "
                "Public evidence remains incomplete."
            ),
            role_fit_justification=(
                "This profile looks highly relevant to the brief because the candidate appears to have a distribution remit. "
                "Direct evidence of team scale is unverified. "
                "Screening priority: how broad is the remit?"
            ),
            outreach_hook="Hi Daniel, we're working on a distribution leadership mandate and your background looks highly relevant.",
        )
    )

    candidate = _stabilize_candidate_brief(candidate_brief=output.candidates[0], enrichment=enrichment)

    assert candidate.role_fit.score <= 3
    assert "distribution remit" not in candidate.role_fit.justification.lower()
    assert "relevant distribution exposure" not in candidate.role_fit.justification.lower()
    assert "distribution leaders" not in candidate.career_narrative.lower()
    assert "distribution leadership mandate" not in candidate.outreach_hook.lower()
    assert "adjacent" in candidate.outreach_hook.lower() or "client-facing" in candidate.outreach_hook.lower()


def test_stabilized_output_handles_not_verified_identity_as_unverified_market_input(tmp_path) -> None:
    enrichment = _build_enrichment(
        tmp_path,
        full_name="Unverified Candidate",
        employer="Example AM",
        title="Head of Distribution",
        fixture={
            "identity_resolution_status": "not_verified",
            "identity_resolution_rationale": "Public search did not verify an exact-match profile for Unverified Candidate at Example AM.",
            "identity_resolution_source_labels": ["Public web search"],
            "verified_public_snippets": [],
            "possible_public_snippets": [],
            "uncertain_fields": ["identity-linked chronology", "identity-specific channel history"],
            "missing_fields": ["exact-match public identity"],
        },
    )
    output = validate_output_document(
        _build_candidate_payload(
            enrichment,
            career_narrative="Placeholder one. Placeholder two. Placeholder three.",
            role_fit_justification="Placeholder one. Placeholder two. Placeholder three.",
            outreach_hook="Placeholder hook.",
        )
    )

    candidate = _stabilize_candidate_brief(candidate_brief=output.candidates[0], enrichment=enrichment)

    assert candidate.role_fit.score <= 2
    assert "does not yet give a reliable current-profile match" in candidate.career_narrative.lower() or "current-profile link is still not reliable enough" in candidate.career_narrative.lower()
    assert "lower-priority mapping lead" in candidate.role_fit.justification.lower()
    assert "distribution brief" in candidate.outreach_hook.lower() or "distribution briefs" in candidate.outreach_hook.lower()


def test_not_verified_career_narrative_still_explains_lane_relevance(tmp_path) -> None:
    enrichment = _build_enrichment(
        tmp_path,
        full_name="Institutional Candidate",
        employer="Example AM",
        title="Director, Institutional Sales",
        fixture={
            "identity_resolution_status": "not_verified",
            "identity_resolution_rationale": "Public search remained ambiguous for Institutional Candidate at Example AM.",
            "identity_resolution_source_labels": ["Public web search"],
            "verified_public_snippets": [],
            "possible_public_snippets": [],
            "channel_orientation": "institutional",
            "scope_signal": "global",
            "mandate_similarity": "unclear_fit",
            "sell_points": [
                "current remit is anchored in senior institutional coverage",
                "public evidence points to global scope",
            ],
            "uncertain_fields": ["identity-linked chronology", "team leadership scale"],
            "missing_fields": ["exact-match public identity"],
        },
    )
    output = validate_output_document(
        _build_candidate_payload(
            enrichment,
            career_narrative="Placeholder one. Placeholder two. Placeholder three.",
            role_fit_justification="Placeholder one. Placeholder two. Placeholder three.",
            outreach_hook="Placeholder hook.",
        )
    )

    candidate = _stabilize_candidate_brief(candidate_brief=output.candidates[0], enrichment=enrichment)
    lowered = candidate.career_narrative.lower()
    assert "institutional" in lowered
    assert "tentative market map entry" in lowered
    assert "action-ready target" in lowered


def test_recruiter_usefulness_qa_accepts_lane_relevance_boundary_structure(tmp_path) -> None:
    enrichment = _build_enrichment(
        tmp_path,
        title="Director, Institutional Sales",
        fixture={
            "verified_public_snippets": ["Example Candidate is listed as Director, Institutional Sales at Example AM."],
            "likely_channel_evidence": ["Public snippets reference institutional distribution coverage."],
            "missing_fields": ["verified reporting line / team size"],
            "uncertain_fields": ["direct Australian institutional coverage"],
        },
    )
    output = validate_output_document(
        _build_candidate_payload(
            enrichment,
            career_narrative=(
                "Example Candidate is currently Director, Institutional Sales at Example AM, with public evidence pointing to an institutional lane. "
                "For the target mandate, the relevance comes from institutional coverage already sitting close to the brief on channel mix and seniority. "
                "What remains unclear from public evidence is how much direct Australian institutional coverage sits behind the remit."
            ),
            role_fit_justification=(
                "This profile is in frame because the current remit already points to institutional coverage at director level. "
                "Key unresolved point: team leadership scale behind the remit, which is not verified publicly. "
                "Screening priority: What size team has this role actually led?"
            ),
            outreach_hook="We're working on a distribution leadership brief where your institutional and super-fund coverage at Example AM looks especially relevant.",
        )
    )

    report = run_bundle_qa(
        output=output,
        candidates={},
        enrichments={"candidate_1": enrichment},
    )

    assert report.passed is True


def test_career_narrative_usefulness_checks_forbid_generic_achievement_language(tmp_path) -> None:
    enrichment = _build_enrichment(tmp_path)
    output = validate_output_document(
        _build_candidate_payload(
            enrichment,
            career_narrative=(
                "Example Candidate is Head of Distribution at Example AM. "
                "The profile suggests relevant distribution coverage. "
                "Public evidence remains limited, but the candidate built the function."
            ),
            role_fit_justification=(
                "This candidate is in frame because current title already signals senior distribution ownership. "
                "The biggest commercial gap is that team scale not verified. "
                "The first screening question should be whether team scale matches the role scope."
            ),
            outreach_hook="I'm reaching out because your background appears relevant to a comparable distribution remit within a Head of Distribution / National BDM search.",
        )
    )

    report = run_bundle_qa(output=output, candidates={}, enrichments={"candidate_1": enrichment})
    assert any(item.check == "career_narrative_forbidden_phrase" for item in report.findings)


def test_role_fit_usefulness_checks_require_screening_angle_and_block_strong_fit_language(tmp_path) -> None:
    enrichment = _build_enrichment(tmp_path, title="Director, Institutional Sales")
    output = validate_output_document(
        _build_candidate_payload(
            enrichment,
            career_narrative=(
                "Example Candidate is currently Director, Institutional Sales at Example AM, with public evidence pointing to an institutional lane. "
                "This profile looks commercially relevant on an adjacent basis because public evidence supports institutional channel relevance. "
                "The main boundary for now is that team scale not verified."
            ),
            role_fit_justification=(
                "This candidate is a strong fit because public evidence supports institutional channel relevance. "
                "The biggest commercial gap is that team scale not verified. "
                "Final fit depends on verification."
            ),
            outreach_hook="I'm reaching out because your background appears relevant to an adjacent but relevant institutional lane within a Head of Distribution / National BDM search.",
        )
    )

    report = run_bundle_qa(output=output, candidates={}, enrichments={"candidate_1": enrichment})
    checks = {item.check for item in report.findings}
    assert "role_fit_forbidden_phrase" in checks
    assert "role_fit_screening_angle" in checks


def test_outreach_hook_usefulness_checks_reject_generic_invitation(tmp_path) -> None:
    enrichment = _build_enrichment(tmp_path, title="Senior BDM Wholesale")
    output = validate_output_document(
        _build_candidate_payload(
            enrichment,
            career_narrative=(
                "Example Candidate is currently Senior BDM Wholesale at Example AM, with public evidence pointing to a wholesale lane. "
                "This profile looks like a potential step-up option because public evidence supports wholesale channel relevance. "
                "The main boundary for now is that team scale not verified."
            ),
            role_fit_justification=(
                "This candidate is in frame because public evidence supports wholesale channel relevance. "
                "The biggest commercial gap is that team scale not verified. "
                "The key point to test in conversation is whether team scale matches the role scope."
            ),
            outreach_hook="Your background could be worth discussing.",
        )
    )

    report = run_bundle_qa(output=output, candidates={}, enrichments={"candidate_1": enrichment})
    checks = {item.check for item in report.findings}
    assert "outreach_hook_generic_invitation" in checks
    assert "outreach_hook_commercial_angle" in checks


def test_firm_aum_context_explicitly_surfaces_unverifiable_context(tmp_path) -> None:
    enrichment = _build_enrichment(
        tmp_path,
        employer="Example AM",
        fixture={
            "firm_aum_context": "Example AM appears to be an established active manager.",
        },
    )
    output = validate_output_document(
        _build_candidate_payload(
            enrichment,
            career_narrative=(
                "Example Candidate is currently Head of Distribution at Example AM, with public evidence pointing to an unclear channel lane from public evidence. "
                "This profile looks only cautiously in frame because public evidence supports a relevant distribution remit. "
                "Public evidence does not fully confirm broader channel depth, and channel breadth remains unclear should be confirmed in conversation."
            ),
            role_fit_justification=(
                "Public evidence supports relevance from Head of Distribution at Example AM because current title already signals senior distribution ownership. "
                "However, direct evidence of channel breadth remains unverified. "
                "This should be tested early in screening conversation, starting with whether channel breadth matches the role scope."
            ),
            outreach_hook="I'm reaching out because your Head of Distribution background at Example AM appears relevant to a comparable distribution remit within a Head of Distribution / National BDM search.",
        )
    )
    candidate = _stabilize_candidate_brief(candidate_brief=output.candidates[0], enrichment=enrichment)
    assert any(
        marker in candidate.firm_aum_context.lower()
        for marker in (
            "unable to verify",
            "public filings do not confirm",
            "public aum remains unverified",
            "exact aum is unavailable",
        )
    )
    assert (
        "firm type and sector context" in candidate.firm_aum_context.lower()
        or "firm profile indicates" in candidate.firm_aum_context.lower()
    )
    assert "$" not in candidate.firm_aum_context


def test_qa_flags_firm_aum_context_that_hides_unverifiable_status(tmp_path) -> None:
    enrichment = _build_enrichment(tmp_path, fixture={"firm_aum_context": "Example AM appears to be an established active manager."})
    output = validate_output_document(
        _build_candidate_payload(
            enrichment,
            career_narrative=(
                "Example Candidate is currently Head of Distribution at Example AM, with public evidence pointing to an unclear channel lane from public evidence. "
                "This profile looks only cautiously in frame because public evidence supports a relevant distribution remit. "
                "Public evidence does not fully confirm broader channel depth, and channel breadth remains unclear should be confirmed in conversation."
            ),
            role_fit_justification=(
                "Public evidence supports relevance from Head of Distribution at Example AM because current title already signals senior distribution ownership. "
                "However, direct evidence of channel breadth remains unverified. "
                "This should be tested early in screening conversation, starting with whether channel breadth matches the role scope."
            ),
            outreach_hook="I'm reaching out because your Head of Distribution background at Example AM appears relevant to a comparable distribution remit within a Head of Distribution / National BDM search.",
        )
    )
    output.candidates[0].firm_aum_context = "Established active manager with mid-tier scale."

    report = run_bundle_qa(output=output, candidates={}, enrichments={"candidate_1": enrichment})
    assert any(item.check == "firm_aum_context_uncertainty" for item in report.findings)


def test_qa_allows_estimated_numeric_aum_with_disclaimer(tmp_path) -> None:
    enrichment = _build_enrichment(
        tmp_path,
        employer="Pendal Group",
        fixture={"firm_aum_context": "Pendal Group appears to be an active asset manager with an estimated AUM of $44.6 billion based on public references."},
    )
    output = validate_output_document(
        _build_candidate_payload(
            enrichment,
            career_narrative=(
                "Example Candidate is currently Head of Distribution at Pendal Group, with public evidence pointing to an unclear channel lane from public evidence. "
                "This profile looks only cautiously in frame because public evidence supports a relevant distribution remit. "
                "Public evidence does not fully confirm broader channel depth, and channel breadth remains unclear should be confirmed in conversation."
            ),
            role_fit_justification=(
                "Public evidence supports relevance from Head of Distribution at Pendal Group because current title already signals senior distribution ownership. "
                "However, direct evidence of channel breadth remains unverified. "
                "This should be tested early in screening conversation, starting with whether channel breadth matches the role scope."
            ),
            outreach_hook="I'm reaching out because your Head of Distribution background at Pendal Group appears relevant to a comparable distribution remit within a Head of Distribution / National BDM search.",
        )
    )
    output.candidates[0].firm_aum_context = (
        "Pendal Group appears to be an active asset manager with an estimated AUM of $44.6 billion based on public references."
    )

    report = run_bundle_qa(output=output, candidates={}, enrichments={"candidate_1": enrichment})
    assert all(item.check != "firm_aum_context_uncertainty" for item in report.findings)
    assert all(item.check != "claim_boundary_firm_aum_context" for item in report.findings)


def test_stabilize_candidate_rewrites_numeric_aum_with_uncertainty_disclaimer(tmp_path) -> None:
    enrichment = _build_enrichment(
        tmp_path,
        employer="Perpetual Limited",
        fixture={"firm_aum_context": "Perpetual Limited reported AUM of $44.6 billion."},
    )
    output = validate_output_document(
        _build_candidate_payload(
            enrichment,
            career_narrative=(
                "Example Candidate is currently Head of Distribution at Perpetual Limited, with public evidence pointing to an unclear channel lane from public evidence. "
                "This profile looks only cautiously in frame because public evidence supports a relevant distribution remit. "
                "Public evidence does not fully confirm broader channel depth, and channel breadth remains unclear should be confirmed in conversation."
            ),
            role_fit_justification=(
                "Public evidence supports relevance from Head of Distribution at Perpetual Limited because current title already signals senior distribution ownership. "
                "However, direct evidence of channel breadth remains unverified. "
                "This should be tested early in screening conversation, starting with whether channel breadth matches the role scope."
            ),
            outreach_hook="I'm reaching out because your Head of Distribution background at Perpetual Limited appears relevant to a comparable distribution remit within a Head of Distribution / National BDM search.",
        )
    )

    candidate = _stabilize_candidate_brief(candidate_brief=output.candidates[0], enrichment=enrichment)
    lowered = candidate.firm_aum_context.lower()
    assert "$44.6 billion" in candidate.firm_aum_context
    assert "estimated" in lowered
    assert "public references" in lowered
    assert "subject to verification" in lowered


def test_firm_context_status_treats_estimated_numeric_aum_as_strongly_inferred() -> None:
    statement = "SG Hiscock & Company appears to be an active asset manager with an estimated AUM of $17 billion."
    assert _firm_context_status(statement) == "strongly_inferred"


def test_mobility_rationale_requires_chronology_uncertainty_and_follow_up(tmp_path) -> None:
    enrichment = _build_enrichment(tmp_path, fixture={"tenure_years": 2.3})
    output = validate_output_document(
        _build_candidate_payload(
            enrichment,
            career_narrative=(
                "Example Candidate is currently Head of Distribution at Example AM, with public evidence pointing to an unclear channel lane from public evidence. "
                "This profile looks only cautiously in frame because public evidence supports a relevant distribution remit. "
                "Public evidence does not fully confirm broader channel depth, and channel breadth remains unclear should be confirmed in conversation."
            ),
            role_fit_justification=(
                "Public evidence supports relevance from Head of Distribution at Example AM because current title already signals senior distribution ownership. "
                "However, direct evidence of channel breadth remains unverified. "
                "This should be tested early in screening conversation, starting with whether channel breadth matches the role scope."
            ),
            outreach_hook="I'm reaching out because your Head of Distribution background at Example AM appears relevant to a comparable distribution remit within a Head of Distribution / National BDM search.",
        )
    )
    candidate = _stabilize_candidate_brief(candidate_brief=output.candidates[0], enrichment=enrichment)
    assert "public chronology suggests approximately" in candidate.mobility_signal.rationale.lower()
    assert "no direct public signal of move readiness is visible" in candidate.mobility_signal.rationale.lower()
    assert "uncertain and checked in follow-up conversation" in candidate.mobility_signal.rationale.lower()


def test_mobility_rationale_is_mode_aware_when_tenure_is_missing(tmp_path) -> None:
    fixture_enrichment = _build_enrichment(tmp_path, fixture={})
    live_enrichment = fixture_enrichment.model_copy(update={"tool_mode": "live_web"})
    output = validate_output_document(
        _build_candidate_payload(
            fixture_enrichment,
            career_narrative=(
                "Example Candidate is currently Head of Distribution at Example AM, with public evidence pointing to an unclear channel lane from public evidence. "
                "This profile looks only cautiously in frame because public evidence supports a relevant distribution remit. "
                "Public evidence does not fully confirm broader channel depth, and channel breadth remains unclear should be confirmed in conversation."
            ),
            role_fit_justification=(
                "Public evidence supports relevance from Head of Distribution at Example AM because current title already signals senior distribution ownership. "
                "However, direct evidence of channel breadth remains unverified. "
                "This should be tested early in screening conversation, starting with whether channel breadth matches the role scope."
            ),
            outreach_hook="I'm reaching out because your Head of Distribution background at Example AM appears relevant to a comparable distribution remit within a Head of Distribution / National BDM search.",
        )
    )

    fixture_candidate = _stabilize_candidate_brief(candidate_brief=output.candidates[0], enrichment=fixture_enrichment)
    live_candidate = _stabilize_candidate_brief(candidate_brief=output.candidates[0], enrichment=live_enrichment)

    assert "fixture-backed run did not provide public chronology for the current role" in fixture_candidate.mobility_signal.rationale.lower()
    assert "live public-web research did not clearly establish public chronology for the current role" in live_candidate.mobility_signal.rationale.lower()


def test_qa_flags_mobility_rationale_that_sounds_like_recruiter_intuition(tmp_path) -> None:
    enrichment = _build_enrichment(tmp_path, fixture={"tenure_years": 2.3})
    output = validate_output_document(
        _build_candidate_payload(
            enrichment,
            career_narrative=(
                "Example Candidate is currently Head of Distribution at Example AM, with public evidence pointing to an unclear channel lane from public evidence. "
                "This profile looks only cautiously in frame because public evidence supports a relevant distribution remit. "
                "Public evidence does not fully confirm broader channel depth, and channel breadth remains unclear should be confirmed in conversation."
            ),
            role_fit_justification=(
                "Public evidence supports relevance from Head of Distribution at Example AM because current title already signals senior distribution ownership. "
                "However, direct evidence of channel breadth remains unverified. "
                "This should be tested early in screening conversation, starting with whether channel breadth matches the role scope."
            ),
            outreach_hook="I'm reaching out because your Head of Distribution background at Example AM appears relevant to a comparable distribution remit within a Head of Distribution / National BDM search.",
        )
    )
    output.candidates[0].mobility_signal.rationale = "The candidate looks settled and likely has low near-term mobility."

    report = run_bundle_qa(output=output, candidates={}, enrichments={"candidate_1": enrichment})
    checks = {item.check for item in report.findings}
    assert "mobility_signal_forbidden_phrase" in checks
    assert "mobility_signal_uncertainty_structure" in checks


def test_tavily_research_client_builds_fixture_from_results() -> None:
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        queries.append(str(payload["query"]))
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Example Candidate appointed to lead distribution",
                        "url": "https://example.com/leadership",
                        "content": "Example Candidate joined Example AM in 2024 to lead wholesale distribution across Australia.",
                    },
                    {
                        "title": "Example AM firm overview",
                        "url": "https://example.com/about",
                        "content": "Example AM is an investment manager with A$10B in assets under management.",
                    },
                ]
            },
        )

    client = TavilyResearchClient(api_key="test-key", client=httpx.Client(transport=httpx.MockTransport(handler)))
    payload = client.lookup_candidate(
        CandidatePublicProfileLookupInput(
            candidate_id="candidate_1",
            full_name="Example Candidate",
            current_employer="Example AM",
            current_title="Head of Distribution",
            linkedin_url="https://example.com/linkedin",
        )
    )

    assert payload.mode == "live_web_tavily"
    assert payload.fixture["tenure_years"] == 2.0
    assert isinstance(payload.fixture["sources"], list)
    assert isinstance(payload.fixture["likely_channel_evidence"], list)
    assert isinstance(payload.fixture["combined_context"], str)
    assert "=== FIRM CONTEXT & AUM ===" in payload.fixture["combined_context"]
    assert len(queries) == 2
    assert "LinkedIn Australia" in queries[0]
    assert "assets under management" in queries[1]


def test_tavily_research_client_filters_same_name_but_wrong_employer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Andrew Swan | Fund manager factsheet",
                        "url": "https://www2.trustnet.com/managers/factsheet/andrew-swan",
                        "content": "Andrew Swan is Head of Asia Equities at Man Group and joined in 2020.",
                    },
                    {
                        "title": "Andrew Swan joins Pendal distribution team",
                        "url": "https://pendalgroup.com/leadership/andrew-swan",
                        "content": "Andrew Swan joined Pendal Group in 2024 as Head of Distribution in Australia.",
                    },
                ]
            },
        )

    client = TavilyResearchClient(api_key="test-key", client=httpx.Client(transport=httpx.MockTransport(handler)))
    payload = client.lookup_candidate(
        CandidatePublicProfileLookupInput(
            candidate_id="candidate_1",
            full_name="Andrew Swan",
            current_employer="Pendal Group",
            current_title="Head of Distribution",
            linkedin_url="https://example.com/linkedin",
        )
    )

    snippets = payload.fixture["verified_public_snippets"]
    assert isinstance(snippets, list)
    assert len(snippets) == 1
    assert "Pendal Group" in snippets[0]


def test_tavily_research_client_marks_possible_match_when_employer_is_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Andrew Swan distribution profile",
                        "url": "https://example.com/profile",
                        "content": "Andrew Swan is a distribution executive in Australian asset management with institutional sales experience.",
                    }
                ]
            },
        )

    client = TavilyResearchClient(api_key="test-key", client=httpx.Client(transport=httpx.MockTransport(handler)))
    payload = client.lookup_candidate(
        CandidatePublicProfileLookupInput(
            candidate_id="candidate_1",
            full_name="Andrew Swan",
            current_employer="Pendal Group",
            current_title="Head of Distribution",
            linkedin_url="https://example.com/linkedin",
        )
    )

    assert payload.fixture["identity_resolution_status"] == "possible_match"
    assert "possible match" in str(payload.fixture["identity_resolution_rationale"]).lower()
    assert isinstance(payload.fixture["possible_public_snippets"], list)


def test_tavily_research_client_rejects_cross_company_firm_material_even_when_target_name_appears() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        query = str(payload["query"])
        if "assets under management" in query:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Perpetual FY23 Financial Results",
                            "url": "https://company-announcements.afr.com/asx/ppt/02dce609-4209-11ee-8af2-5edd2b3cbcdb.pdf",
                            "content": "Perpetual FY23 Financial Results. Total AUM of $212.1 billion. Includes commentary that Pendal refers to the asset management business in Australia.",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Andrew Swan distribution profile",
                        "url": "https://example.com/profile",
                        "content": "Andrew Swan is a distribution executive in Australian asset management with institutional sales experience.",
                    }
                ]
            },
        )

    client = TavilyResearchClient(api_key="test-key", client=httpx.Client(transport=httpx.MockTransport(handler)))
    payload = client.lookup_candidate(
        CandidatePublicProfileLookupInput(
            candidate_id="candidate_1",
            full_name="Andrew Swan",
            current_employer="Pendal Group",
            current_title="Head of Distribution",
            linkedin_url="https://example.com/linkedin",
        )
    )

    firm_context = payload.fixture["firm_aum_context"]
    assert isinstance(firm_context, str)
    assert "$212.1 billion" not in firm_context
    assert "exact AUM" in firm_context


def test_tavily_research_client_history_bio_does_not_count_as_current_role_match() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Cathy Hales - Business News",
                        "url": "https://example.com/business-news/cathy-hales",
                        "content": "Cathy Hales was the global head of Fidante Partners, following senior roles with Deutsche Asset Management, Colonial First State and BT Funds Management.",
                    }
                ]
            },
        )

    client = TavilyResearchClient(api_key="test-key", client=httpx.Client(transport=httpx.MockTransport(handler)))
    payload = client.lookup_candidate(
        CandidatePublicProfileLookupInput(
            candidate_id="candidate_2",
            full_name="Cathy Hales",
            current_employer="Perpetual Limited",
            current_title="Head of Retail Sales",
            linkedin_url="https://example.com/linkedin",
        )
    )

    assert payload.fixture["identity_resolution_status"] == "not_verified"


def test_tavily_research_client_marks_not_verified_when_no_exact_profile_is_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Australian distribution careers overview",
                        "url": "https://example.com/market-overview",
                        "content": "Funds management distribution roles often cover platforms, advisers, and institutional investors across Australia.",
                    }
                ]
            },
        )

    client = TavilyResearchClient(api_key="test-key", client=httpx.Client(transport=httpx.MockTransport(handler)))
    payload = client.lookup_candidate(
        CandidatePublicProfileLookupInput(
            candidate_id="candidate_1",
            full_name="Andrew Swan",
            current_employer="Pendal Group",
            current_title="Head of Distribution",
            linkedin_url="https://example.com/linkedin",
        )
    )

    assert payload.fixture["identity_resolution_status"] == "not_verified"
    assert "did not verify an exact-match profile" in str(payload.fixture["identity_resolution_rationale"]).lower()
    assert payload.fixture["verified_public_snippets"] == []


def test_tavily_research_client_extracts_billion_style_aum_context() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        query = str(payload["query"])
        if "assets under management" in query:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Perpetual FY23 Financial Results Appendix",
                            "url": "https://example.com/perpetual-results",
                            "content": "Assets under Management, Funds under Advice and Funds under Administration Asset Management: AUM and flows 35 AUM by asset class ($b) For the period 30 June 2022 ($b) Pendal AUM (at 11 January 2023) Flows.",
                        },
                        {
                            "title": "Has Pendal acquisition hindered Perpetual?",
                            "url": "https://example.com/perpetual-news",
                            "content": "This meant total Pendal Asset Management AUM was down from $44.6 billion at the end of March to $41 billion.",
                        },
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Anthony Serhan profile",
                        "url": "https://example.com/profile",
                        "content": "Anthony Serhan joined Pendal Group in 2018 as Distribution Director in Australia.",
                    }
                ]
            },
        )

    client = TavilyResearchClient(api_key="test-key", client=httpx.Client(transport=httpx.MockTransport(handler)))
    payload = client.lookup_candidate(
        CandidatePublicProfileLookupInput(
            candidate_id="candidate_1",
            full_name="Anthony Serhan",
            current_employer="Pendal Group",
            current_title="Distribution Director",
            linkedin_url="https://example.com/linkedin",
        )
    )

    firm_context = payload.fixture["firm_aum_context"]
    assert isinstance(firm_context, str)
    assert "$44.6 billion" not in firm_context
    assert "exact AUM figure should still be verified manually" in firm_context


def test_tavily_research_client_extracts_large_currency_amount_aum_context() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        query = str(payload["query"])
        if "assets under management" in query:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Merricks Capital Pty Ltd in Melbourne",
                            "url": "https://example.com/merricks-aum",
                            "content": "Data sourced from SEC public records. Total assets under management $2,104,590,054. More info available on the adviser profile.",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Camelia Seric profile",
                        "url": "https://example.com/profile",
                        "content": "Camelia Seric joined Merricks Capital as Head of Distribution Wealth Management Australia and New Zealand.",
                    }
                ]
            },
        )

    client = TavilyResearchClient(api_key="test-key", client=httpx.Client(transport=httpx.MockTransport(handler)))
    payload = client.lookup_candidate(
        CandidatePublicProfileLookupInput(
            candidate_id="candidate_3",
            full_name="Camelia Seric",
            current_employer="Merricks Capital",
            current_title="Head of Distribution Wealth Management Australia and New Zealand",
            linkedin_url="https://example.com/linkedin",
        )
    )

    firm_context = payload.fixture["firm_aum_context"]
    assert isinstance(firm_context, str)
    assert "$2,104,590,054" in firm_context
    assert "estimated AUM" in firm_context
    assert "still subject to verification" in firm_context


def test_tavily_research_client_accepts_numeric_aum_from_results_style_context() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        query = str(payload["query"])
        if "assets under management" in query:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Perpetual FY23 Financial Results",
                            "url": "https://example.com/perpetual-results",
                            "content": "Perpetual FY23 Financial Results. Total AUM of $212.1 billion as of 30 June 2023 across the asset management platform.",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Cathy Hales profile",
                        "url": "https://example.com/profile",
                        "content": "Cathy Hales is Head of Retail Sales at Perpetual Limited.",
                    }
                ]
            },
        )

    client = TavilyResearchClient(api_key="test-key", client=httpx.Client(transport=httpx.MockTransport(handler)))
    payload = client.lookup_candidate(
        CandidatePublicProfileLookupInput(
            candidate_id="candidate_2",
            full_name="Cathy Hales",
            current_employer="Perpetual Limited",
            current_title="Head of Retail Sales",
            linkedin_url="https://example.com/linkedin",
        )
    )

    firm_context = payload.fixture["firm_aum_context"]
    assert isinstance(firm_context, str)
    assert "$212.1 billion" in firm_context
    assert "estimated AUM" in firm_context


def test_tavily_research_client_rejects_aum_attributed_to_other_entity() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        query = str(payload["query"])
        if "assets under management" in query:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "SG Hiscock Property Fund Fact Sheets",
                            "url": "https://example.com/sgh-factsheet",
                            "content": "The principals were formerly employed at National Asset Management (NAM), a subsidiary of National Australia Bank which had $17 billion funds under management.",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Anthony Cochran profile",
                        "url": "https://example.com/profile",
                        "content": "Anthony Cochran is Head of Distribution at SG Hiscock & Company in Australia.",
                    }
                ]
            },
        )

    client = TavilyResearchClient(api_key="test-key", client=httpx.Client(transport=httpx.MockTransport(handler)))
    payload = client.lookup_candidate(
        CandidatePublicProfileLookupInput(
            candidate_id="candidate_5",
            full_name="Anthony Cochran",
            current_employer="SG Hiscock & Company",
            current_title="Head of Distribution",
            linkedin_url="https://example.com/linkedin",
        )
    )

    firm_context = payload.fixture["firm_aum_context"]
    assert isinstance(firm_context, str)
    assert "$17 billion" not in firm_context
    assert "exact AUM figure should still be verified manually" in firm_context


def test_tavily_research_client_rejects_year_like_billion_match() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        query = str(payload["query"])
        if "assets under management" in query:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "SG Hiscock company overview",
                            "url": "https://example.com/sgh-overview",
                            "content": "SG Hiscock & Company was founded in 2001 B and remains a boutique manager focused on Australia.",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Anthony Cochran profile",
                        "url": "https://example.com/profile",
                        "content": "Anthony Cochran is Head of Distribution at SG Hiscock & Company in Australia.",
                    }
                ]
            },
        )

    client = TavilyResearchClient(api_key="test-key", client=httpx.Client(transport=httpx.MockTransport(handler)))
    payload = client.lookup_candidate(
        CandidatePublicProfileLookupInput(
            candidate_id="candidate_5",
            full_name="Anthony Cochran",
            current_employer="SG Hiscock & Company",
            current_title="Head of Distribution",
            linkedin_url="https://example.com/linkedin",
        )
    )

    firm_context = payload.fixture["firm_aum_context"]
    assert isinstance(firm_context, str)
    assert "2001 B" not in firm_context
    assert "exact AUM still requires direct verification" in firm_context


def test_run_ppp_pipeline_writes_enrichment_artifacts(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "candidates.csv"
    input_path.write_text(
        "full_name,current_employer,current_title,linkedin_url\n"
        "Andrew Swan,Pendal Group,Head of Distribution,https://example.com/1\n"
        "Cathy Hales,Perpetual Limited,Head of Retail Sales,https://example.com/2\n"
        "Jason Ennis,Fidelity International,\"Director, Inst. Sales\",https://example.com/3\n"
        "Deborah Southon,Challenger Limited,Head of Distribution,https://example.com/4\n"
        "Nikki Thomas,Magellan Financial,\"Senior BDM Wholesale\",https://example.com/5\n",
        encoding="utf-8",
    )
    role_spec_path = tmp_path / "role.json"
    role_spec_path.write_text(json.dumps({"role": "Head of Distribution / National BDM"}), encoding="utf-8")
    output_path = tmp_path / "output.json"
    intermediate_dir = tmp_path / "intermediate"
    fixture_path = tmp_path / "fixtures.json"
    fixture_path.write_text(
        json.dumps(
            {
                name: {
                    "tenure_years": 2.0,
                    "firm_aum_context": f"{employer} context from fixture.",
                    "verified_public_snippets": [f"{name} is listed at {employer}."],
                }
                for name, employer in [
                    ("Andrew Swan", "Pendal Group"),
                    ("Cathy Hales", "Perpetual Limited"),
                    ("Jason Ennis", "Fidelity International"),
                    ("Deborah Southon", "Challenger Limited"),
                    ("Nikki Thomas", "Magellan Financial"),
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("app.ppp.pipeline.settings.anthropic_api_key", "test-key")
    def fake_generate_text(self, *, system_prompt: str, user_prompt: str, model: str, max_tokens: int, extra=None):
        payload = json.loads(user_prompt)
        candidate = _candidate_from_prompt(payload)
        allowed_facts = _allowed_facts_from_prompt(payload)
        tenure = _allowed_fact_section(allowed_facts, "tenure_years")
        candidate_id = _candidate_id_from_prompt(payload)
        return json.dumps(
            {
                "candidate_id": candidate_id,
                "full_name": candidate["full_name"],
                "current_role": {
                    "title": candidate["current_title"],
                    "employer": candidate["current_employer"],
                    "tenure_years": tenure["value"] or 2.0,
                },
                "career_narrative": f"{candidate['full_name']} is listed at {candidate['current_employer']} in the research package. "
                "The enrichment payload suggests relevant distribution channel exposure. "
                "The final fit still depends on live verification of team and network depth.",
                "experience_tags": ["distribution", "client coverage"],
                "firm_aum_context": f"{candidate['current_employer']} context from fixture.",
                "mobility_signal": {
                    "score": 3,
                    "rationale": "Public chronology is still limited in the enrichment package. No direct public signal of mobility is visible, so openness should be treated as uncertain pending conversation."
                },
                "role_fit": {
                    "role": "Head of Distribution / National BDM",
                    "score": 7,
                    "justification": f"{candidate['current_employer']} and the supported distribution exposure are relevant to the target role. "
                    "The available public evidence still leaves network depth and leadership scope uncertain. "
                    "Further verification is required before the supported distribution exposure can be treated as complete."
                },
                "outreach_hook": f"Your background at {candidate['current_employer']} could make this leadership search worth a conversation."
            }
        )

    monkeypatch.setattr("app.ppp.pipeline.ClaudeClient.generate_text", fake_generate_text)
    monkeypatch.setattr("app.ppp.pipeline.ClaudeClient.run_tool_phase_once", lambda self, **kwargs: _fake_normalized_evidence_json(kwargs["user_prompt"]))

    result = run_ppp_pipeline(
        input_path=str(input_path),
        output_path=str(output_path),
        role_spec_path=str(role_spec_path),
        model="claude-sonnet-4-5",
        intermediate_dir=str(intermediate_dir),
        research_fixture_path=str(fixture_path),
        research_mode="fixture",
    )

    assert output_path.exists()
    assert len(list(intermediate_dir.glob("*_enriched.json"))) == 5
    assert any(
        marker in result.candidates[0].firm_aum_context.lower()
        for marker in (
            "unable to verify",
            "public filings do not confirm",
            "public aum remains unverified",
            "exact aum is unavailable",
        )
    )


def test_run_ppp_pipeline_retries_after_invalid_first_generation(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "candidates.csv"
    input_path.write_text(
        "full_name,current_employer,current_title,linkedin_url\n"
        "A,One,Title,https://example.com/1\n"
        "B,Two,Title,https://example.com/2\n"
        "C,Three,Title,https://example.com/3\n"
        "D,Four,Title,https://example.com/4\n"
        "E,Five,Title,https://example.com/5\n",
        encoding="utf-8",
    )
    role_spec_path = tmp_path / "role.json"
    role_spec_path.write_text(json.dumps({"role": "Head of Distribution / National BDM"}), encoding="utf-8")
    output_path = tmp_path / "output.json"

    monkeypatch.setattr("app.ppp.pipeline.settings.anthropic_api_key", "test-key")
    call_counter = {"count": 0}

    def fake_generate_text(self, *, system_prompt: str, user_prompt: str, model: str, max_tokens: int, extra=None):
        payload = json.loads(user_prompt)
        candidate = _candidate_from_prompt(payload)
        allowed_facts = _allowed_facts_from_prompt(payload)
        tenure = _allowed_fact_section(allowed_facts, "tenure_years")
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            return '{"candidate_id": "candidate_1"}'
        return json.dumps(
            {
                "candidate_id": _candidate_id_from_prompt(payload),
                "full_name": candidate["full_name"],
                "current_role": {
                    "title": candidate["current_title"],
                    "employer": candidate["current_employer"],
                    "tenure_years": tenure["value"] or 2.0,
                },
                "career_narrative": f"{candidate['full_name']} is listed in the task input as {candidate['current_title']} at {candidate['current_employer']}. "
                f"The title '{candidate['current_title']}' points to relevant distribution exposure. "
                "Public evidence still leaves some scope questions open.",
                "experience_tags": ["distribution", "client coverage"],
                "firm_aum_context": (
                    f"{candidate['current_employer']} appears to be a established market participant funds-management firm; "
                    "exact AUM requires live public verification; distribution exposure inferred from the candidate's title."
                ),
                "mobility_signal": {
                    "score": 3,
                    "rationale": "Public chronology remains limited for this profile. No direct public signal of mobility is visible, so openness should be treated as uncertain pending conversation."
                },
                "role_fit": {
                    "role": "Head of Distribution / National BDM",
                    "score": 7,
                    "justification": f"{candidate['current_employer']} and the title {candidate['current_title']} are relevant to the target distribution role. "
                    "The available public evidence still leaves current scope, channel depth, and leadership scale uncertain. "
                    "Further verification is required before the supported distribution exposure can be treated as complete."
                },
                "outreach_hook": f"Your experience at {candidate['current_employer']} could make this role worth discussing."
            }
        )

    monkeypatch.setattr("app.ppp.pipeline.ClaudeClient.generate_text", fake_generate_text)
    monkeypatch.setattr("app.ppp.pipeline.ClaudeClient.run_tool_phase_once", lambda self, **kwargs: _fake_normalized_evidence_json(kwargs["user_prompt"]))

    run_ppp_pipeline(
        input_path=str(input_path),
        output_path=str(output_path),
        role_spec_path=str(role_spec_path),
        model="claude-sonnet-4-5",
        intermediate_dir=str(tmp_path / "intermediate"),
        research_mode="fixture",
    )

    assert call_counter["count"] == 6


def test_validate_output_bundle_flags_placeholder_content(tmp_path) -> None:
    output_path = tmp_path / "output.json"
    output_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": f"candidate_{idx}",
                        "full_name": name,
                        "current_role": {"title": "Head of Distribution", "employer": employer, "tenure_years": 2.0},
                        "career_narrative": "Unknown candidate. N/A. TBD.",
                        "experience_tags": ["distribution"],
                        "firm_aum_context": f"{employer} context requires verification.",
                        "mobility_signal": {"score": 3, "rationale": "Estimated and unverified."},
                        "role_fit": {
                            "role": "Head of Distribution / National BDM",
                            "score": 7,
                            "justification": f"{employer} distribution background appears relevant."
                        },
                        "outreach_hook": "This could be worth discussing."
                    }
                    for idx, (name, employer) in enumerate(
                        [
                            ("A", "One"),
                            ("B", "Two"),
                            ("C", "Three"),
                            ("D", "Four"),
                            ("E", "Five"),
                        ],
                        start=1,
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    input_path = tmp_path / "candidates.csv"
    input_path.write_text(
        "full_name,current_employer,current_title,linkedin_url\n"
        "A,One,Title,https://example.com/1\n"
        "B,Two,Title,https://example.com/2\n"
        "C,Three,Title,https://example.com/3\n"
        "D,Four,Title,https://example.com/4\n"
        "E,Five,Title,https://example.com/5\n",
        encoding="utf-8",
    )
    intermediate_dir = tmp_path / "intermediate"
    intermediate_dir.mkdir()

    report = validate_output_bundle(
        output_path=str(output_path),
        input_path=str(input_path),
        intermediate_dir=str(intermediate_dir),
    )

    assert report.passed is False
    assert any(item.check == "placeholder_text" for item in report.findings)


def test_validate_output_bundle_returns_report_for_schema_failure(tmp_path) -> None:
    output_path = tmp_path / "output.json"
    output_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": f"candidate_{idx}",
                        "full_name": name,
                        "current_role": {"title": "Head of Distribution", "employer": employer, "tenure_years": 2.0},
                        "career_narrative": "Candidate has strong experience. Candidate has visible market exposure. Candidate remains relevant.",
                        "experience_tags": ["distribution"],
                        "firm_aum_context": f"{employer} context requires verification.",
                        "mobility_signal": {"score": 3, "rationale": "Estimated and unverified."},
                        "role_fit": {
                            "role": "Head of Distribution / National BDM",
                            "score": 12,
                            "justification": f"{employer} distribution background appears relevant."
                        },
                        "outreach_hook": "This could be worth discussing."
                    }
                    for idx, (name, employer) in enumerate(
                        [("A", "One"), ("B", "Two"), ("C", "Three"), ("D", "Four"), ("E", "Five")],
                        start=1,
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    input_path = tmp_path / "candidates.csv"
    input_path.write_text(
        "full_name,current_employer,current_title,linkedin_url\n"
        "A,One,Title,https://example.com/1\n"
        "B,Two,Title,https://example.com/2\n"
        "C,Three,Title,https://example.com/3\n"
        "D,Four,Title,https://example.com/4\n"
        "E,Five,Title,https://example.com/5\n",
        encoding="utf-8",
    )

    report = validate_output_bundle(
        output_path=str(output_path),
        input_path=str(input_path),
        intermediate_dir=str(tmp_path / "intermediate"),
    )

    assert report.passed is False
    assert any(item.check == "output_schema" for item in report.findings)


def test_run_ppp_pipeline_writes_failure_artifact_on_candidate_error(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "candidates.csv"
    input_path.write_text(
        "full_name,current_employer,current_title,linkedin_url\n"
        "A,One,Title,https://example.com/1\n"
        "B,Two,Title,https://example.com/2\n"
        "C,Three,Title,https://example.com/3\n"
        "D,Four,Title,https://example.com/4\n"
        "E,Five,Title,https://example.com/5\n",
        encoding="utf-8",
    )
    role_spec_path = tmp_path / "role.json"
    role_spec_path.write_text(json.dumps({"role": "Head of Distribution / National BDM"}), encoding="utf-8")
    intermediate_dir = tmp_path / "intermediate"

    monkeypatch.setattr("app.ppp.pipeline.settings.anthropic_api_key", "test-key")
    def fake_generate_text(self, *, system_prompt: str, user_prompt: str, model: str, max_tokens: int, extra=None):
        payload = json.loads(user_prompt)
        candidate_id = _candidate_id_from_prompt(payload)
        candidate = _candidate_from_prompt(payload)
        allowed_facts = _allowed_facts_from_prompt(payload)
        tenure = _allowed_fact_section(allowed_facts, "tenure_years")
        if candidate_id == "candidate_1":
            return '{"candidate_id":"candidate_1"}'
        return json.dumps(
            {
                "candidate_id": candidate_id,
                "full_name": candidate["full_name"],
                "current_role": {
                    "title": candidate["current_title"],
                    "employer": candidate["current_employer"],
                    "tenure_years": tenure["value"] or 2.0,
                },
                "career_narrative": f"{candidate['full_name']} is listed in the task input as {candidate['current_title']} at {candidate['current_employer']}. The title suggests relevant client and channel exposure. Public evidence still leaves some scope questions open.",
                "experience_tags": ["distribution", "client coverage"],
                "firm_aum_context": f"{candidate['current_employer']} context requires verification.",
                "mobility_signal": {
                    "score": 3,
                    "rationale": "Public chronology remains limited for this profile. No direct public signal of mobility is visible, so openness should be treated as uncertain pending conversation."
                },
                "role_fit": {
                    "role": "Head of Distribution / National BDM",
                    "score": 7,
                    "justification": f"{candidate['current_employer']} and the title {candidate['current_title']} suggest relevant distribution exposure. The available public evidence still leaves channel depth and leadership scale uncertain. Further verification is required before the supported distribution exposure can be treated as complete."
                },
                "outreach_hook": f"Your experience at {candidate['current_employer']} could make this role worth discussing."
            }
        )

    monkeypatch.setattr("app.ppp.pipeline.ClaudeClient.generate_text", fake_generate_text)
    monkeypatch.setattr("app.ppp.pipeline.ClaudeClient.run_tool_phase_once", lambda self, **kwargs: _fake_normalized_evidence_json(kwargs["user_prompt"]))

    with pytest.raises(PPPTaskError, match="expected exactly 5 validated candidates"):
        run_ppp_pipeline(
            input_path=str(input_path),
            output_path=str(tmp_path / "output.json"),
            role_spec_path=str(role_spec_path),
            model="claude-sonnet-4-5",
            intermediate_dir=str(intermediate_dir),
            research_mode="fixture",
        )

    assert (intermediate_dir / "candidate_1_error.json").exists()
    run_report = json.loads((intermediate_dir / "run_report.json").read_text(encoding="utf-8"))
    assert run_report["failed_candidates"][0]["candidate_id"] == "candidate_1"
    assert run_report["failed_candidates"][0]["stage"] == "generation"


def test_run_ppp_pipeline_raises_when_all_candidates_fail(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "candidates.csv"
    input_path.write_text(
        "full_name,current_employer,current_title,linkedin_url\n"
        "A,One,Title,https://example.com/1\n"
        "B,Two,Title,https://example.com/2\n"
        "C,Three,Title,https://example.com/3\n"
        "D,Four,Title,https://example.com/4\n"
        "E,Five,Title,https://example.com/5\n",
        encoding="utf-8",
    )
    role_spec_path = tmp_path / "role.json"
    role_spec_path.write_text(json.dumps({"role": "Head of Distribution / National BDM"}), encoding="utf-8")
    intermediate_dir = tmp_path / "intermediate"

    monkeypatch.setattr("app.ppp.pipeline.settings.anthropic_api_key", "test-key")
    monkeypatch.setattr("app.ppp.pipeline.ClaudeClient.generate_text", lambda *args, **kwargs: '{"candidate_id":"candidate_1"}')
    monkeypatch.setattr("app.ppp.pipeline.ClaudeClient.run_tool_phase_once", lambda self, **kwargs: _fake_normalized_evidence_json(kwargs["user_prompt"]))

    with pytest.raises(PPPTaskError, match="failed for all candidates"):
        run_ppp_pipeline(
            input_path=str(input_path),
            output_path=str(tmp_path / "output.json"),
            role_spec_path=str(role_spec_path),
            model="claude-sonnet-4-5",
            intermediate_dir=str(intermediate_dir),
            research_mode="fixture",
        )

    assert (intermediate_dir / "run_report.json").exists()
