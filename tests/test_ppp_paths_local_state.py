from __future__ import annotations

import json

from app.ppp.enrichment import CandidatePublicProfileLookupInput, CandidatePublicProfileLookupTool
from app.ppp.local_state import LocalAPIKeyState, clear_local_api_state, load_local_api_state, save_local_api_state
from app.ppp.paths import load_ppp_paths


def test_load_ppp_paths_honors_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("PPP_DATA_DIR", "custom-data/ppp")
    monkeypatch.setenv("PPP_OUTPUT_PATH", "artifacts/final-output.json")
    monkeypatch.setenv("PPP_LOCAL_STATE_FILE", "local-cache/keys.json")

    paths = load_ppp_paths()

    assert paths.data_dir == paths.repo_root / "custom-data" / "ppp"
    assert paths.output_json == paths.repo_root / "artifacts" / "final-output.json"
    assert paths.local_state_file == paths.repo_root / "local-cache" / "keys.json"


def test_local_api_state_round_trip(tmp_path) -> None:
    state_path = tmp_path / "local-state.json"

    save_local_api_state(
        LocalAPIKeyState(anthropic_api_key="anthropic-test", tavily_api_key="tavily-test"),
        path=state_path,
    )

    loaded = load_local_api_state(state_path)
    assert loaded.anthropic_api_key == "anthropic-test"
    assert loaded.tavily_api_key == "tavily-test"

    clear_local_api_state(state_path)
    assert load_local_api_state(state_path) == LocalAPIKeyState()


def test_fixture_claims_are_ingested_into_enrichment(tmp_path) -> None:
    fixture_path = tmp_path / "fixtures.json"
    fixture_path.write_text(
        json.dumps(
            {
                "Example Candidate": {
                    "identity_resolution_status": "possible_match",
                    "possible_public_snippets": [
                        "Local notes tie the candidate to Example AM, but identity still needs live confirmation."
                    ],
                    "claims": [
                        {
                            "category": "scope",
                            "statement": "The local evidence points to ANZ coverage rather than a purely domestic remit.",
                            "verification_status": "strongly_inferred",
                            "confidence": "medium",
                            "supports_output_fields": ["career_narrative", "role_fit"]
                        },
                        {
                            "category": "channel_experience",
                            "statement": "The local evidence points to institutional and wholesale channel ownership.",
                            "verification_status": "strongly_inferred",
                            "confidence": "medium",
                            "supports_output_fields": ["career_narrative", "experience_tags", "role_fit"]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    tool = CandidatePublicProfileLookupTool(fixture_path=str(fixture_path))
    result = tool.run(
        CandidatePublicProfileLookupInput(
            candidate_id="candidate_1",
            full_name="Example Candidate",
            current_employer="Example AM",
            current_title="Director, Institutional Sales",
            linkedin_url="https://example.com/profile",
        )
    )

    assert result.identity_resolution.status == "possible_match"
    assert any(claim.category == "scope" for claim in result.claims)
    assert any(
        "institutional and wholesale" in claim.statement.lower() for claim in result.claims if claim.category == "channel_experience"
    )
    assert result.recruiter_signals.scope_signal == "anz"


def test_scope_signal_does_not_promote_global_from_firm_context_only(tmp_path) -> None:
    fixture_path = tmp_path / "scope_guard.json"
    fixture_path.write_text(
        json.dumps(
            {
                "Scope Guard Candidate": {
                    "identity_resolution_status": "possible_match",
                    "firm_aum_context": "Example Global AM is a global asset manager with international operations and broad product reach.",
                    "likely_channel_evidence": [
                        "Retail-sales leadership points to adviser and platform distribution."
                    ],
                    "likely_experience_evidence": [
                        "Current remit suggests domestic intermediary distribution relevance."
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    tool = CandidatePublicProfileLookupTool(fixture_path=str(fixture_path))
    result = tool.run(
        CandidatePublicProfileLookupInput(
            candidate_id="candidate_1",
            full_name="Scope Guard Candidate",
            current_employer="Example Global AM",
            current_title="Head of Retail Sales",
            linkedin_url="https://example.com/profile",
        )
    )

    assert result.recruiter_signals.scope_signal == "unclear"
