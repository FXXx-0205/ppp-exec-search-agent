from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.ppp.models import CandidateCSVRow


class LookupSource(BaseModel):
    label: str
    source_type: str
    url: str | None = None
    confidence: str = "controlled_fixture"


class EnrichmentEvidence(BaseModel):
    category: str
    signal: str
    snippet: str
    source_labels: list[str]


class CandidatePublicProfileLookupInput(BaseModel):
    candidate_id: str
    full_name: str
    current_employer: str
    current_title: str
    linkedin_url: str

    @classmethod
    def from_candidate(cls, *, candidate_id: str, candidate: CandidateCSVRow) -> "CandidatePublicProfileLookupInput":
        return cls(candidate_id=candidate_id, **candidate.model_dump(mode="json"))


class CandidateEnrichmentResult(BaseModel):
    tool_name: str = "candidate_public_profile_lookup"
    tool_mode: str = "fixture_backed"
    candidate_id: str
    full_name: str
    current_employer: str
    current_title: str
    verified_public_snippets: list[str]
    inferred_tenure_years: float | None = None
    tenure_rationale: str
    likely_channel_evidence: list[str]
    likely_experience_evidence: list[str]
    firm_aum_context: str
    firm_context_clues: list[str]
    mobility_evidence: list[str]
    missing_fields: list[str]
    uncertain_fields: list[str]
    confidence_notes: list[str]
    sources: list[LookupSource]
    evidence: list[EnrichmentEvidence]
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @field_validator(
        "verified_public_snippets",
        "likely_channel_evidence",
        "likely_experience_evidence",
        "firm_context_clues",
        "mobility_evidence",
        "missing_fields",
        "uncertain_fields",
        "confidence_notes",
    )
    @classmethod
    def _strip_list_items(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]


class CandidatePublicProfileLookupTool:
    def __init__(self, *, fixture_path: str = "data/ppp/research_fixtures.json") -> None:
        self.fixture_path = Path(fixture_path)
        self._fixtures = self._load_fixtures(self.fixture_path)

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "name": "candidate_public_profile_lookup",
            "description": (
                "Retrieve controlled public-research fixture data for a candidate, including verified snippets, "
                "tenure clues, firm/AUM context clues, missing fields, and confidence notes."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "full_name": {"type": "string"},
                    "current_employer": {"type": "string"},
                    "current_title": {"type": "string"},
                    "linkedin_url": {"type": "string"},
                },
                "required": ["candidate_id", "full_name", "current_employer", "current_title", "linkedin_url"],
            },
        }

    def run(self, tool_input: CandidatePublicProfileLookupInput) -> CandidateEnrichmentResult:
        fixture = self._fixtures.get(tool_input.full_name, {})
        return self._build_result(tool_input, fixture)

    def run_json(self, tool_input: dict[str, Any]) -> str:
        parsed = CandidatePublicProfileLookupInput.model_validate(tool_input)
        result = self.run(parsed)
        return json.dumps(result.model_dump(mode="json"), ensure_ascii=False)

    def save_intermediate(self, result: CandidateEnrichmentResult, *, output_dir: str = "data/ppp/intermediate") -> Path:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{result.candidate_id}_enriched.json"
        path.write_text(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _build_result(
        self,
        tool_input: CandidatePublicProfileLookupInput,
        fixture: dict[str, Any],
    ) -> CandidateEnrichmentResult:
        verified_public_snippets = fixture.get(
            "verified_public_snippets",
            [
                f"Provided task input lists {tool_input.full_name} as {tool_input.current_title} at {tool_input.current_employer}.",
                "LinkedIn URL in the PPP brief is a placeholder, so live public-profile verification is still pending.",
            ],
        )
        inferred_tenure_years = self._infer_tenure_years(fixture)
        tenure_rationale = self._build_tenure_rationale(fixture, inferred_tenure_years)
        firm_aum_context = self._format_firm_aum_context(tool_input.current_employer, fixture)
        likely_channel_evidence = fixture.get(
            "likely_channel_evidence",
            [f"The title '{tool_input.current_title}' suggests senior distribution channel responsibility."],
        )
        likely_experience_evidence = fixture.get(
            "likely_experience_evidence",
            [f"Current employer/title pairing suggests relevance to funds-management client coverage at {tool_input.current_employer}."],
        )
        mobility_evidence = fixture.get(
            "mobility_evidence",
            [
                "Mobility should be inferred cautiously until live tenure chronology and recent promotions are verified.",
            ],
        )
        missing_fields = fixture.get(
            "missing_fields",
            [
                "exact current-role start date",
                "verified reporting line / team size",
                "direct evidence of platform, IFA, and superannuation coverage",
            ],
        )
        uncertain_fields = fixture.get(
            "uncertain_fields",
            [
                "precise tenure estimate",
                "channel mix breadth",
                "current mobility intent",
            ],
        )
        confidence_notes = fixture.get(
            "confidence_notes",
            [
                "This enrichment run is fixture-backed and should be upgraded with live public-web verification before final submission.",
            ],
        )
        sources = [
            LookupSource(**source)
            for source in fixture.get(
                "sources",
                [
                    {
                        "label": "PPP task input CSV",
                        "source_type": "provided_input",
                        "url": None,
                        "confidence": "high",
                    },
                    {
                        "label": "Task brief LinkedIn placeholder",
                        "source_type": "linkedin_placeholder",
                        "url": tool_input.linkedin_url,
                        "confidence": "low",
                    },
                ],
            )
        ]
        evidence = [
            EnrichmentEvidence(**item)
            for item in fixture.get(
                "evidence",
                [
                    {
                        "category": "current_role",
                        "signal": "provided_input",
                        "snippet": verified_public_snippets[0],
                        "source_labels": [sources[0].label],
                    },
                    {
                        "category": "mobility",
                        "signal": "uncertain",
                        "snippet": mobility_evidence[0],
                        "source_labels": [sources[-1].label],
                    },
                ],
            )
        ]

        return CandidateEnrichmentResult(
            candidate_id=tool_input.candidate_id,
            full_name=tool_input.full_name,
            current_employer=tool_input.current_employer,
            current_title=tool_input.current_title,
            verified_public_snippets=verified_public_snippets,
            inferred_tenure_years=inferred_tenure_years,
            tenure_rationale=tenure_rationale,
            likely_channel_evidence=likely_channel_evidence,
            likely_experience_evidence=likely_experience_evidence,
            firm_aum_context=firm_aum_context,
            firm_context_clues=fixture.get("firm_context_clues", []),
            mobility_evidence=mobility_evidence,
            missing_fields=missing_fields,
            uncertain_fields=uncertain_fields,
            confidence_notes=confidence_notes,
            sources=sources,
            evidence=evidence,
        )

    def _infer_tenure_years(self, fixture: dict[str, Any]) -> float | None:
        raw_tenure = fixture.get("tenure_years")
        if isinstance(raw_tenure, (int, float)):
            return round(float(raw_tenure), 1)

        start_year = fixture.get("current_role_start_year")
        start_month = fixture.get("current_role_start_month", 1)
        if isinstance(start_year, int):
            today = datetime.now(UTC)
            months = max(1, (today.year - start_year) * 12 + (today.month - int(start_month)))
            return round(months / 12.0, 1)
        return None

    def _build_tenure_rationale(self, fixture: dict[str, Any], tenure_years: float | None) -> str:
        if isinstance(fixture.get("tenure_rationale"), str) and fixture["tenure_rationale"].strip():
            return fixture["tenure_rationale"].strip()
        if tenure_years is not None:
            return f"Estimated at approximately {tenure_years:.1f} years based on controlled research fixture chronology."
        return "Exact current-role tenure could not be verified from the controlled fixture and remains a key follow-up item."

    def _format_firm_aum_context(self, employer: str, fixture: dict[str, Any]) -> str:
        if isinstance(fixture.get("firm_aum_context"), str) and fixture["firm_aum_context"].strip():
            return fixture["firm_aum_context"].strip()

        employer_type = fixture.get("employer_type", "funds-management firm")
        market_position = fixture.get("market_position", "established market participant")
        aum_descriptor = fixture.get("aum_descriptor", "exact AUM requires live public verification")
        channel_focus = fixture.get("channel_focus", "distribution exposure inferred from the candidate's title")
        return f"{employer} appears to be a {market_position} {employer_type}; {aum_descriptor}; {channel_focus}."

    def _load_fixtures(self, path: Path) -> dict[str, dict[str, Any]]:
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(raw, dict):
            return {}
        parsed: dict[str, dict[str, Any]] = {}
        for name, payload in raw.items():
            if isinstance(name, str) and isinstance(payload, dict):
                parsed[name] = payload
        return parsed


def validate_enrichment_payload(payload: Any) -> CandidateEnrichmentResult:
    return CandidateEnrichmentResult.model_validate(payload)


def format_enrichment_validation_error(exc: ValidationError) -> str:
    issue = exc.errors()[0]
    location = ".".join(str(part) for part in issue.get("loc", ()))
    prefix = f"{location}: " if location else ""
    return f"{prefix}{issue['msg']}"
