from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.ppp.enrichment import CandidatePublicProfileLookupInput, CandidatePublicProfileLookupTool
from app.ppp.pipeline import PPPTaskError, _load_candidates_csv, _load_role_spec, run_ppp_pipeline
from app.ppp.qa import validate_output_bundle
from app.ppp.validator import validate_and_repair_candidate_payload


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
    monkeypatch.setattr("app.ppp.pipeline.ClaudeClient.generate_with_tools", lambda *args, **kwargs: "{}")

    def fake_generate_text(self, *, system_prompt: str, user_prompt: str, model: str, max_tokens: int, extra=None):
        payload = json.loads(user_prompt)
        candidate = payload["candidate_input"]
        candidate_id = payload["schema_rules"]["candidate_id"]
        return json.dumps(
            {
                "candidate_id": candidate_id,
                "full_name": candidate["full_name"],
                "current_role": {
                    "title": candidate["current_title"],
                    "employer": candidate["current_employer"],
                    "tenure_years": 2.5,
                },
                "career_narrative": f"{candidate['full_name']} has built a credible distribution career with visible client-facing scope. "
                f"The current role at {candidate['current_employer']} suggests senior market exposure. Public data remains limited, so some details are framed cautiously.",
                "experience_tags": ["distribution", "leadership"],
                "firm_aum_context": f"{candidate['current_employer']} requires live AUM verification.",
                "mobility_signal": {"score": 3, "rationale": "Public tenure indicators suggest a possible but unconfirmed transition window."},
                "role_fit": {
                    "role": "Head of Distribution / National BDM",
                    "score": 7,
                    "justification": f"{candidate['full_name']} appears relevant for funds-management distribution leadership, "
                    "but channel depth still needs live verification.",
                },
                "outreach_hook": f"Your experience at {candidate['current_employer']} could make this distribution leadership brief worth discussing.",
            }
        )

    monkeypatch.setattr("app.ppp.pipeline.ClaudeClient.generate_text", fake_generate_text)

    result = run_ppp_pipeline(
        input_path=str(input_path),
        output_path=str(output_path),
        role_spec_path=str(role_spec_path),
        model="claude-sonnet-4-5",
        intermediate_dir=str(tmp_path / "intermediate"),
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
    assert result.evidence
    assert saved.exists()


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
    monkeypatch.setattr("app.ppp.pipeline.ClaudeClient.generate_with_tools", lambda *args, **kwargs: "{}")

    def fake_generate_text(self, *, system_prompt: str, user_prompt: str, model: str, max_tokens: int, extra=None):
        payload = json.loads(user_prompt)
        candidate = payload["candidate_input"]
        enrichment = payload["enrichment"]
        candidate_id = payload["schema_rules"]["candidate_id"]
        return json.dumps(
            {
                "candidate_id": candidate_id,
                "full_name": candidate["full_name"],
                "current_role": {
                    "title": candidate["current_title"],
                    "employer": candidate["current_employer"],
                    "tenure_years": enrichment["inferred_tenure_years"],
                },
                "career_narrative": f"{candidate['full_name']} has visible distribution experience. "
                "The enrichment payload suggests relevant channel exposure. "
                "The final fit still depends on live verification of team and network depth.",
                "experience_tags": ["distribution", "client coverage"],
                "firm_aum_context": enrichment["firm_aum_context"],
                "mobility_signal": {
                    "score": 3,
                    "rationale": "The enrichment evidence supports a moderate but still unverified mobility case."
                },
                "role_fit": {
                    "role": "Head of Distribution / National BDM",
                    "score": 7,
                    "justification": f"{candidate['full_name']} has evidence of distribution exposure relevant to the target mandate. "
                    "A final recruiter view would still require live verification of network depth and leadership scope."
                },
                "outreach_hook": f"Your background at {candidate['current_employer']} could make this leadership search worth a conversation."
            }
        )

    monkeypatch.setattr("app.ppp.pipeline.ClaudeClient.generate_text", fake_generate_text)

    result = run_ppp_pipeline(
        input_path=str(input_path),
        output_path=str(output_path),
        role_spec_path=str(role_spec_path),
        model="claude-sonnet-4-5",
        intermediate_dir=str(intermediate_dir),
        research_fixture_path=str(fixture_path),
    )

    assert output_path.exists()
    assert len(list(intermediate_dir.glob("*_enriched.json"))) == 5
    assert result.candidates[0].firm_aum_context == "Pendal Group context from fixture."


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
    monkeypatch.setattr("app.ppp.pipeline.ClaudeClient.generate_with_tools", lambda *args, **kwargs: "{}")

    call_counter = {"count": 0}

    def fake_generate_text(self, *, system_prompt: str, user_prompt: str, model: str, max_tokens: int, extra=None):
        payload = json.loads(user_prompt)
        candidate = payload["candidate_input"]
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            return '{"candidate_id": "candidate_1"}'
        return json.dumps(
            {
                "candidate_id": payload["schema_rules"]["candidate_id"],
                "full_name": candidate["full_name"],
                "current_role": {
                    "title": candidate["current_title"],
                    "employer": candidate["current_employer"],
                    "tenure_years": 2.0,
                },
                "career_narrative": f"{candidate['full_name']} currently holds a senior client-facing role. "
                "The profile points to relevant distribution exposure. "
                "Public evidence still leaves some scope questions open.",
                "experience_tags": ["distribution", "client coverage"],
                "firm_aum_context": f"{candidate['current_employer']} context requires live AUM verification.",
                "mobility_signal": {"score": 3, "rationale": "The available evidence supports a moderate but unconfirmed mobility case."},
                "role_fit": {
                    "role": "Head of Distribution / National BDM",
                    "score": 7,
                    "justification": f"{candidate['full_name']} has directional fit for the target distribution mandate, "
                    f"with current experience at {candidate['current_employer']} still requiring scope verification."
                },
                "outreach_hook": f"Your experience at {candidate['current_employer']} could make this role worth discussing."
            }
        )

    monkeypatch.setattr("app.ppp.pipeline.ClaudeClient.generate_text", fake_generate_text)

    run_ppp_pipeline(
        input_path=str(input_path),
        output_path=str(output_path),
        role_spec_path=str(role_spec_path),
        model="claude-sonnet-4-5",
        intermediate_dir=str(tmp_path / "intermediate"),
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
    monkeypatch.setattr("app.ppp.pipeline.ClaudeClient.generate_with_tools", lambda *args, **kwargs: "{}")
    monkeypatch.setattr("app.ppp.pipeline.ClaudeClient.generate_text", lambda *args, **kwargs: '{"candidate_id":"candidate_1"}')

    with pytest.raises(PPPTaskError):
        run_ppp_pipeline(
            input_path=str(input_path),
            output_path=str(tmp_path / "output.json"),
            role_spec_path=str(role_spec_path),
            model="claude-sonnet-4-5",
            intermediate_dir=str(intermediate_dir),
        )

    assert (intermediate_dir / "candidate_1_error.json").exists()
