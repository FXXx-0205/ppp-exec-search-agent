from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.ppp.models import CandidateCSVRow
from app.ppp.research import PublicResearchClient, ResearchClientError

VerificationStatus = Literal["verified", "strongly_inferred", "uncertain"]
ConfidenceLevel = Literal["high", "medium", "low"]
ChannelOrientation = Literal["institutional", "wholesale", "wealth", "retail", "mixed", "unclear"]
MandateSimilarity = Literal["direct_match", "adjacent_match", "step_up_candidate", "unclear_fit"]
ScopeSignal = Literal["national", "anz", "global", "regional", "unclear"]
SenioritySignal = Literal["head_level", "director_level", "bdm_level", "unclear"]
EvidenceStrength = Literal["strong", "moderate", "thin"]

QUALIFIER_TERMS = ("unverified", "verification", "estimated", "limited public visibility", "cautious", "approx", "verify")


class LookupSource(BaseModel):
    label: str
    source_type: str
    url: str | None = None
    confidence: ConfidenceLevel = "medium"
    trust_score: float = 0.5


class ResearchClaim(BaseModel):
    claim_id: str
    category: str
    statement: str
    verification_status: VerificationStatus
    confidence: ConfidenceLevel
    source_labels: list[str]
    supports_output_fields: list[str] = Field(default_factory=list)
    numeric_value: float | None = None

    @field_validator("statement")
    @classmethod
    def _require_statement(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty.")
        return normalized

    @field_validator("source_labels", "supports_output_fields")
    @classmethod
    def _strip_list_items(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]


class OutputFieldSupport(BaseModel):
    field_name: str
    supported_by_claim_ids: list[str] = Field(default_factory=list)

    @field_validator("field_name")
    @classmethod
    def _require_field_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty.")
        return normalized

    @field_validator("supported_by_claim_ids")
    @classmethod
    def _strip_claim_ids(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]


class VerificationSummary(BaseModel):
    missing_fields: list[str]
    uncertain_fields: list[str]
    confidence_notes: list[str]

    @field_validator("missing_fields", "uncertain_fields", "confidence_notes")
    @classmethod
    def _strip_list_items(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]


class CandidatePublicProfileLookupInput(BaseModel):
    candidate_id: str
    full_name: str
    current_employer: str
    current_title: str
    linkedin_url: str

    @classmethod
    def from_candidate(cls, *, candidate_id: str, candidate: CandidateCSVRow) -> "CandidatePublicProfileLookupInput":
        return cls(candidate_id=candidate_id, **candidate.model_dump(mode="json"))


class RecruiterSignals(BaseModel):
    channel_orientation: ChannelOrientation
    mandate_similarity: MandateSimilarity
    scope_signal: ScopeSignal
    seniority_signal: SenioritySignal
    evidence_strength: EvidenceStrength
    key_sell_points: list[str] = Field(default_factory=list)
    key_gaps: list[str] = Field(default_factory=list)
    screening_priority_question: str

    @field_validator("key_sell_points", "key_gaps")
    @classmethod
    def _limit_signal_lists(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        return cleaned[:2]

    @field_validator("screening_priority_question")
    @classmethod
    def _require_screening_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field cannot be empty.")
        return normalized


class CandidateEnrichmentResult(BaseModel):
    tool_name: str = "candidate_public_profile_lookup"
    tool_mode: str = "fixture_backed"
    candidate_id: str
    full_name: str
    current_employer: str
    current_title: str
    linkedin_url: str
    sources: list[LookupSource]
    claims: list[ResearchClaim]
    verification: VerificationSummary
    recruiter_signals: RecruiterSignals
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def inferred_tenure_years(self) -> float | None:
        for claim in self.claims:
            if claim.category == "tenure" and claim.numeric_value is not None:
                return claim.numeric_value
        return None

    @property
    def firm_aum_context(self) -> str:
        claim = self.best_claim("firm_context")
        if claim is not None:
            return claim.statement
        return f"{self.current_employer} firm context remains partially verified and should be checked manually."

    def best_claim(self, category: str) -> ResearchClaim | None:
        matches = [claim for claim in self.claims if claim.category == category]
        if not matches:
            return None
        matches.sort(key=lambda claim: (_verification_rank(claim.verification_status), _confidence_rank(claim.confidence)), reverse=True)
        return matches[0]

    def claims_for_output_field(self, field_name: str) -> list[ResearchClaim]:
        return [claim for claim in self.claims if field_name in claim.supports_output_fields]

    def field_support(self, field_name: str) -> OutputFieldSupport:
        claim_ids = [claim.claim_id for claim in self.claims_for_output_field(field_name)]
        return OutputFieldSupport(field_name=field_name, supported_by_claim_ids=claim_ids)

    def output_field_supports(self) -> list[OutputFieldSupport]:
        field_names = {
            field_name
            for claim in self.claims
            for field_name in claim.supports_output_fields
        }
        return [self.field_support(field_name) for field_name in sorted(field_names)]

    def allowed_facts(self) -> dict[str, Any]:
        current_role_claims = self.claims_for_output_field("current_role")
        tenure_claim = self.best_claim("tenure")
        firm_context_claims = self.claims_for_output_field("firm_aum_context")
        role_fit_claims = self.claims_for_output_field("role_fit")
        mobility_claims = self.claims_for_output_field("mobility_signal")
        narrative_claims = self.claims_for_output_field("career_narrative")

        firm_context_claim = self.best_claim("firm_context")
        firm_context_mode = "qualitative_only"
        if any(
            claim.verification_status == "verified" and re.search(r"(\$|aud\s*)\d+(\.\d+)?\s*[bm]", claim.statement.lower())
            for claim in firm_context_claims
        ):
            firm_context_mode = "numeric_or_qualitative"

        tenure_range = None
        if self.inferred_tenure_years is not None and self.inferred_tenure_years >= 0:
            tenure_range = [
                round(max(0.0, self.inferred_tenure_years - 0.5), 1),
                round(self.inferred_tenure_years + 0.5, 1),
            ]

        return {
            "current_role": {
                "title": self.current_title,
                "employer": self.current_employer,
                "supported_by_claim_ids": [claim.claim_id for claim in current_role_claims],
            },
            "tenure_years": {
                "value": self.inferred_tenure_years,
                "confidence": tenure_claim.confidence if tenure_claim is not None else "low",
                "verification_status": tenure_claim.verification_status if tenure_claim is not None else "uncertain",
                "range": tenure_range,
                "supported_by_claim_ids": [claim.claim_id for claim in self.claims_for_output_field("current_role") if claim.category == "tenure"]
                or ([tenure_claim.claim_id] if tenure_claim is not None else []),
            },
            "firm_aum_context": {
                "mode": firm_context_mode,
                "preferred_statement": self.firm_aum_context,
                "allowed_phrases": _allowed_firm_phrases(self.current_employer, firm_context_claim.statement if firm_context_claim is not None else self.firm_aum_context),
                "supported_by_claim_ids": [claim.claim_id for claim in firm_context_claims],
            },
            "career_narrative": {
                "sentence_plan": [
                    "Current role and employer",
                    "Tenure and career direction with explicit uncertainty if needed",
                    "Channel or client exposure supported by claims",
                ],
                "supported_by_claim_ids": [claim.claim_id for claim in narrative_claims],
            },
            "experience_tags": {
                "supported_by_claim_ids": [claim.claim_id for claim in self.claims_for_output_field("experience_tags")],
                "allowed_categories": sorted({claim.category for claim in self.claims_for_output_field("experience_tags")}),
            },
            "mobility_signal": {
                "supported_by_claim_ids": [claim.claim_id for claim in mobility_claims],
                "allowed_reasoning": [
                    "Tenure, trajectory, and explicit public signals only",
                    "Use cautious language when mobility intent is not directly observed",
                    "Do not infer commitment, promotion-cycle psychology, or active-search intent unless explicitly supported",
                ],
                "disallowed_inferences": [
                    "multi-year commitment",
                    "promotion implies low mobility",
                    "active search intent",
                    "natural transition window unless directly evidenced",
                ],
            },
            "role_fit": {
                "supported_by_claim_ids": [claim.claim_id for claim in role_fit_claims],
                "allowed_reasoning": [
                    "Anchor justification to supported employer, title, channel, and experience claims",
                    "Mention limits when public evidence is incomplete",
                    "Prefer gaps and verification needs over unsupported strengths",
                ],
                "disallowed_inferences": [
                    "confirmed team-management scale unless explicitly stated",
                    "confirmed institutional network depth unless explicitly stated",
                    "confirmed geography-specific market embeddedness unless explicitly stated",
                ],
            },
            "outreach_hook": {
                "supported_by_claim_ids": [claim.claim_id for claim in self.claims_for_output_field("outreach_hook")],
                "allowed_reasoning": [
                    "Use one candidate-specific angle already supported by the research package",
                ],
            },
            "recruiter_signals": self.recruiter_signals.model_dump(mode="json"),
        }

    def research_package(self) -> dict[str, Any]:
        return {
            "candidate_identity": {
                "candidate_id": self.candidate_id,
                "full_name": self.full_name,
                "current_employer": self.current_employer,
                "current_title": self.current_title,
                "linkedin_url": self.linkedin_url,
            },
            "sources": [source.model_dump(mode="json") for source in self.sources],
            "claims": [claim.model_dump(mode="json") for claim in self.claims],
            "verification": self.verification.model_dump(mode="json"),
            "recruiter_signals": self.recruiter_signals.model_dump(mode="json"),
            "output_field_supports": [support.model_dump(mode="json") for support in self.output_field_supports()],
            "allowed_facts": self.allowed_facts(),
        }


class CandidatePublicProfileLookupTool:
    def __init__(
        self,
        *,
        fixture_path: str = "data/ppp/research_fixtures.json",
        mode: str = "fixture",
        research_client: PublicResearchClient | None = None,
    ) -> None:
        self.fixture_path = Path(fixture_path)
        self._fixtures = self._load_fixtures(self.fixture_path)
        self.mode = mode
        self.research_client = research_client

    def run(self, tool_input: CandidatePublicProfileLookupInput) -> CandidateEnrichmentResult:
        fixture = self._fixtures.get(tool_input.full_name, {})
        if self.mode == "fixture":
            return self._build_result(tool_input, fixture, tool_mode="fixture_backed")
        if self.mode == "live":
            live_fixture = self._lookup_live(tool_input)
            return self._build_result(tool_input, live_fixture, tool_mode="live_web")
        if self.mode == "auto":
            if self.research_client is not None:
                try:
                    live_fixture = self._lookup_live(tool_input)
                    return self._build_result(tool_input, live_fixture, tool_mode="live_web")
                except ResearchClientError:
                    pass
            merged_fixture = dict(fixture)
            notes = list(merged_fixture.get("confidence_notes", [])) if isinstance(merged_fixture.get("confidence_notes"), list) else []
            notes.append("Live public-web research was unavailable or failed, so the enrichment fell back to the controlled fixture.")
            merged_fixture["confidence_notes"] = notes
            return self._build_result(tool_input, merged_fixture, tool_mode="fixture_fallback")
        raise ResearchClientError(f"Unsupported research mode: {self.mode}")

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
        *,
        tool_mode: str,
    ) -> CandidateEnrichmentResult:
        sources = self._build_sources(tool_input, fixture)
        verification = VerificationSummary(
            missing_fields=self._list_value(
                fixture,
                "missing_fields",
                [
                    "exact current-role start date",
                    "verified reporting line / team size",
                    "direct evidence of platform, IFA, and superannuation coverage",
                ],
            ),
            uncertain_fields=self._list_value(
                fixture,
                "uncertain_fields",
                [
                    "precise tenure estimate",
                    "channel mix breadth",
                    "current mobility intent",
                ],
            ),
            confidence_notes=self._list_value(
                fixture,
                "confidence_notes",
                [
                    "This enrichment run is fixture-backed and should be upgraded with live public-web verification before final submission.",
                ],
            ),
        )
        claims = self._build_claims(tool_input, fixture, sources=sources, verification=verification)
        recruiter_signals = derive_recruiter_signals(
            tool_input=tool_input,
            claims=claims,
            verification=verification,
        )
        return CandidateEnrichmentResult(
            tool_mode=tool_mode,
            candidate_id=tool_input.candidate_id,
            full_name=tool_input.full_name,
            current_employer=tool_input.current_employer,
            current_title=tool_input.current_title,
            linkedin_url=tool_input.linkedin_url,
            sources=sources,
            claims=claims,
            verification=verification,
            recruiter_signals=recruiter_signals,
        )

    def _build_sources(self, tool_input: CandidatePublicProfileLookupInput, fixture: dict[str, Any]) -> list[LookupSource]:
        raw_sources = fixture.get("sources")
        if isinstance(raw_sources, list) and raw_sources:
            return [LookupSource(**source) for source in raw_sources if isinstance(source, dict)]
        return [
            LookupSource(label="PPP task input CSV", source_type="provided_input", url=None, confidence="high", trust_score=0.95),
            LookupSource(
                label="Candidate LinkedIn profile",
                source_type="linkedin_public" if "linkedin.com" in tool_input.linkedin_url.lower() else "provided_profile_url",
                url=tool_input.linkedin_url,
                confidence="medium",
                trust_score=0.9 if "linkedin.com" in tool_input.linkedin_url.lower() else 0.75,
            ),
        ]

    def _build_claims(
        self,
        tool_input: CandidatePublicProfileLookupInput,
        fixture: dict[str, Any],
        *,
        sources: list[LookupSource],
        verification: VerificationSummary,
    ) -> list[ResearchClaim]:
        claims: list[ResearchClaim] = []
        source_labels = [source.label for source in sources]
        primary_source = source_labels[:1] or ["PPP task input CSV"]

        verified_snippets = self._list_value(
            fixture,
            "verified_public_snippets",
            [
                f"Provided task input lists {tool_input.full_name} as {tool_input.current_title} at {tool_input.current_employer}.",
            ],
        )
        claims.extend(
            ResearchClaim(
                claim_id=f"{tool_input.candidate_id}_current_role_{idx}",
                category="current_role",
                statement=snippet,
                verification_status="verified",
                confidence="high" if idx == 1 else "medium",
                source_labels=primary_source,
                supports_output_fields=["career_narrative", "role_fit", "current_role", "outreach_hook"],
            )
            for idx, snippet in enumerate(verified_snippets[:2], start=1)
        )

        inferred_tenure_years = self._infer_tenure_years(fixture)
        tenure_rationale = self._build_tenure_rationale(fixture, inferred_tenure_years)
        claims.append(
            ResearchClaim(
                claim_id=f"{tool_input.candidate_id}_tenure_1",
                category="tenure",
                statement=tenure_rationale,
                verification_status="strongly_inferred" if inferred_tenure_years is not None and inferred_tenure_years >= 0 else "uncertain",
                confidence="medium" if inferred_tenure_years is not None and inferred_tenure_years >= 0 else "low",
                source_labels=primary_source,
                supports_output_fields=["career_narrative", "mobility_signal", "current_role"],
                numeric_value=inferred_tenure_years,
            )
        )

        for idx, statement in enumerate(
            self._list_value(
                fixture,
                "likely_channel_evidence",
                [f"The title '{tool_input.current_title}' suggests senior distribution channel responsibility."],
            ),
            start=1,
        ):
            claims.append(
                ResearchClaim(
                    claim_id=f"{tool_input.candidate_id}_channel_{idx}",
                    category="channel_experience",
                    statement=statement,
                    verification_status="strongly_inferred",
                    confidence="medium",
                    source_labels=source_labels,
                    supports_output_fields=["career_narrative", "experience_tags", "role_fit", "outreach_hook"],
                )
            )

        for idx, statement in enumerate(
            self._list_value(
                fixture,
                "likely_experience_evidence",
                [f"Current employer/title pairing suggests relevance to funds-management client coverage at {tool_input.current_employer}."],
            ),
            start=1,
        ):
            claims.append(
                ResearchClaim(
                    claim_id=f"{tool_input.candidate_id}_experience_{idx}",
                    category="experience",
                    statement=statement,
                    verification_status="strongly_inferred",
                    confidence="medium",
                    source_labels=source_labels,
                    supports_output_fields=["career_narrative", "experience_tags", "role_fit", "outreach_hook"],
                )
            )

        firm_context = self._format_firm_aum_context(tool_input.current_employer, fixture)
        claims.append(
            ResearchClaim(
                claim_id=f"{tool_input.candidate_id}_firm_context_1",
                category="firm_context",
                statement=firm_context,
                verification_status=_firm_context_status(firm_context),
                confidence="medium" if _firm_context_status(firm_context) != "uncertain" else "low",
                source_labels=source_labels,
                supports_output_fields=["firm_aum_context", "career_narrative", "role_fit"],
            )
        )

        for idx, statement in enumerate(
            self._list_value(
                fixture,
                "mobility_evidence",
                ["Mobility should be inferred cautiously until live tenure chronology and recent promotions are verified."],
            ),
            start=1,
        ):
            claims.append(
                ResearchClaim(
                    claim_id=f"{tool_input.candidate_id}_mobility_{idx}",
                    category="mobility",
                    statement=statement,
                    verification_status="uncertain",
                    confidence="low",
                    source_labels=source_labels,
                    supports_output_fields=["mobility_signal", "outreach_hook", "career_narrative"],
                )
            )

        for field_name in verification.uncertain_fields:
            claims.append(
                ResearchClaim(
                    claim_id=f"{tool_input.candidate_id}_verification_{len(claims) + 1}",
                    category="verification_boundary",
                    statement=f"{field_name} remains uncertain and should be framed cautiously.",
                    verification_status="uncertain",
                    confidence="low",
                    source_labels=source_labels,
                    supports_output_fields=["career_narrative", "firm_aum_context", "mobility_signal", "role_fit"],
                )
            )

        for field_name in verification.missing_fields:
            claims.append(
                ResearchClaim(
                    claim_id=f"{tool_input.candidate_id}_verification_{len(claims) + 1}",
                    category="verification_boundary",
                    statement=f"{field_name} is not yet verified from available public research.",
                    verification_status="uncertain",
                    confidence="low",
                    source_labels=source_labels,
                    supports_output_fields=["career_narrative", "firm_aum_context", "mobility_signal", "role_fit"],
                )
            )

        return claims

    def _lookup_live(self, tool_input: CandidatePublicProfileLookupInput) -> dict[str, Any]:
        if self.research_client is None:
            raise ResearchClientError("Live research mode requires a configured public research client.")
        payload = self.research_client.lookup_candidate(tool_input)
        fixture = payload.fixture
        return {str(key): value for key, value in fixture.items()}

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
        return -1.0

    def _build_tenure_rationale(self, fixture: dict[str, Any], tenure_years: float | None) -> str:
        if isinstance(fixture.get("tenure_rationale"), str) and fixture["tenure_rationale"].strip():
            return fixture["tenure_rationale"].strip()
        if tenure_years is not None and tenure_years >= 0:
            return f"Estimated at approximately {tenure_years:.1f} years based on controlled research chronology."
        return "Tenure mapped to placeholder (-1.0) as public chronology is unavailable."

    def _format_firm_aum_context(self, employer: str, fixture: dict[str, Any]) -> str:
        if isinstance(fixture.get("firm_aum_context"), str) and fixture["firm_aum_context"].strip():
            return fixture["firm_aum_context"].strip()

        employer_type = fixture.get("employer_type", "funds-management platform")
        market_position = fixture.get("market_position", "established")
        channel_focus = fixture.get("channel_focus", "distribution exposure inferred from the candidate's title")
        return (
            f"Unable to verify exact AUM from public sources for {employer}; "
            f"based on firm type and sector context, {employer} appears to be an {market_position} {employer_type}; "
            f"{channel_focus}."
        )

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

    def _list_value(self, fixture: dict[str, Any], key: str, fallback: list[str]) -> list[str]:
        value = fixture.get(key)
        if isinstance(value, list):
            return [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return fallback


def validate_enrichment_payload(payload: Any) -> CandidateEnrichmentResult:
    return CandidateEnrichmentResult.model_validate(payload)


def derive_recruiter_signals(
    *,
    tool_input: CandidatePublicProfileLookupInput,
    claims: list[ResearchClaim],
    verification: VerificationSummary,
) -> RecruiterSignals:
    channel_orientation = _derive_channel_orientation(tool_input=tool_input, claims=claims)
    scope_signal = _derive_scope_signal(tool_input=tool_input, claims=claims)
    seniority_signal = _derive_seniority_signal(tool_input=tool_input)
    evidence_strength = _derive_evidence_strength(claims=claims)
    mandate_similarity = _derive_mandate_similarity(
        tool_input=tool_input,
        claims=claims,
        channel_orientation=channel_orientation,
        scope_signal=scope_signal,
        seniority_signal=seniority_signal,
        evidence_strength=evidence_strength,
    )
    key_sell_points = _derive_key_sell_points(
        tool_input=tool_input,
        claims=claims,
        channel_orientation=channel_orientation,
        mandate_similarity=mandate_similarity,
        scope_signal=scope_signal,
        seniority_signal=seniority_signal,
    )
    key_gaps = _derive_key_gaps(
        tool_input=tool_input,
        claims=claims,
        verification=verification,
        channel_orientation=channel_orientation,
        scope_signal=scope_signal,
        seniority_signal=seniority_signal,
        evidence_strength=evidence_strength,
    )
    screening_priority_question = _derive_screening_priority_question(
        key_gaps=key_gaps,
        channel_orientation=channel_orientation,
        scope_signal=scope_signal,
        seniority_signal=seniority_signal,
    )
    return RecruiterSignals(
        channel_orientation=channel_orientation,
        mandate_similarity=mandate_similarity,
        scope_signal=scope_signal,
        seniority_signal=seniority_signal,
        evidence_strength=evidence_strength,
        key_sell_points=key_sell_points,
        key_gaps=key_gaps,
        screening_priority_question=screening_priority_question,
    )


def _firm_context_status(statement: str) -> VerificationStatus:
    lowered = statement.lower()
    if any(term in lowered for term in QUALIFIER_TERMS):
        return "uncertain"
    if re.search(r"(\$|aud\s*)\d+(\.\d+)?\s*[bm]", lowered):
        return "strongly_inferred"
    return "verified"


def _verification_rank(status: VerificationStatus) -> int:
    return {"verified": 3, "strongly_inferred": 2, "uncertain": 1}[status]


def _confidence_rank(confidence: ConfidenceLevel) -> int:
    return {"high": 3, "medium": 2, "low": 1}[confidence]


def _allowed_firm_phrases(employer: str, statement: str) -> list[str]:
    phrases = [employer, "asset manager", "funds-management firm", "institutional platform", "distribution platform"]
    lowered = statement.lower()
    if "global" in lowered:
        phrases.append("global asset manager")
    if "institutional" in lowered:
        phrases.append("institutional platform")
    if "wholesale" in lowered:
        phrases.append("wholesale distribution platform")
    return sorted({phrase for phrase in phrases if phrase})


def _derive_channel_orientation(
    *,
    tool_input: CandidatePublicProfileLookupInput,
    claims: list[ResearchClaim],
) -> ChannelOrientation:
    scores = {key: 0 for key in ("institutional", "wholesale", "wealth", "retail")}
    title = tool_input.current_title.lower()
    relevant_claims = [claim for claim in claims if claim.category in {"current_role", "channel_experience", "experience"}]
    claim_text = " ".join(claim.statement.lower() for claim in relevant_claims)
    explicit_categories: set[ChannelOrientation] = set()

    for category in _extract_channel_categories(title):
        scores[category] += 4
        explicit_categories.add(category)

    for claim in relevant_claims:
        text = claim.statement.lower()
        categories = _extract_channel_categories(text)
        if not categories:
            continue
        weight = 3 if claim.category in {"channel_experience", "current_role"} else 2
        for category in categories:
            scores[category] += weight
            if claim.category in {"channel_experience", "current_role"}:
                explicit_categories.add(category)

    if "ifa" in claim_text or "adviser" in claim_text or "intermediary" in claim_text or "platform" in claim_text:
        scores["wholesale"] += 2
    if "wealth management" in claim_text or "private wealth" in claim_text:
        scores["wealth"] += 2
    if "super" in claim_text or "institutional" in claim_text or "consultant" in claim_text:
        scores["institutional"] += 2
    if "retail" in claim_text:
        scores["retail"] += 2

    if len(explicit_categories) >= 2:
        return "mixed"

    ranked = sorted(((score, category) for category, score in scores.items() if score > 0), reverse=True)
    if ranked:
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] <= 1:
            return "mixed"
        return ranked[0][1]

    if "distribution" in title or "marketing" in title:
        return "mixed"
    return "unclear"


def _extract_channel_categories(text: str) -> list[ChannelOrientation]:
    lowered = text.lower()
    categories: list[ChannelOrientation] = []
    if any(term in lowered for term in ("institutional", "inst. sales", "inst sales")):
        categories.append("institutional")
    if any(term in lowered for term in ("wholesale", "ifa")):
        categories.append("wholesale")
    if "wealth" in lowered:
        categories.append("wealth")
    if "retail" in lowered:
        categories.append("retail")
    return categories


def _derive_mandate_similarity(
    *,
    tool_input: CandidatePublicProfileLookupInput,
    claims: list[ResearchClaim],
    channel_orientation: ChannelOrientation,
    scope_signal: ScopeSignal,
    seniority_signal: SenioritySignal,
    evidence_strength: EvidenceStrength,
) -> MandateSimilarity:
    title = tool_input.current_title.lower()
    claim_text = " ".join(claim.statement.lower() for claim in claims if claim.category in {"current_role", "channel_experience", "experience"})

    has_head_distribution = "head of distribution" in title
    has_national_bdm = "national bdm" in title
    has_head_sales = "head of sales" in title or "head of retail sales" in title or "head of wholesale" in title or "head of wealth" in title
    has_distribution_lead = "distribution" in title and any(term in title for term in ("head", "director", "lead"))
    has_channel_lead = any(term in title for term in ("director", "head", "lead")) and any(
        term in title for term in ("institutional", "wholesale", "wealth", "retail")
    )
    has_bdm = "bdm" in title or "business development manager" in title
    head_scope = scope_signal in {"national", "anz", "global", "regional"}

    if evidence_strength == "thin" and seniority_signal == "unclear":
        return "unclear_fit"
    if has_head_distribution and seniority_signal == "head_level":
        return "direct_match"
    if has_national_bdm:
        return "direct_match"
    if seniority_signal == "head_level" and has_distribution_lead and (channel_orientation != "unclear" or head_scope):
        return "direct_match"
    if seniority_signal == "head_level" and channel_orientation in {"institutional", "wholesale", "wealth", "retail", "mixed"}:
        return "adjacent_match"
    if has_head_sales or has_channel_lead or ("distribution" in claim_text and seniority_signal in {"head_level", "director_level"}):
        return "adjacent_match"
    if has_bdm or seniority_signal == "bdm_level" or ("manager" in title and channel_orientation != "unclear"):
        return "step_up_candidate"
    return "unclear_fit"


def _derive_key_sell_points(
    *,
    tool_input: CandidatePublicProfileLookupInput,
    claims: list[ResearchClaim],
    channel_orientation: ChannelOrientation,
    mandate_similarity: MandateSimilarity,
    scope_signal: ScopeSignal,
    seniority_signal: SenioritySignal,
) -> list[str]:
    sell_points: list[str] = []
    has_supported_firm_context = any(claim.category == "firm_context" and claim.verification_status != "uncertain" for claim in claims)

    if channel_orientation != "unclear":
        if seniority_signal == "head_level":
            sell_points.append(f"current remit sits in {channel_orientation} distribution leadership")
        elif seniority_signal == "director_level":
            sell_points.append(f"current remit is anchored in senior {channel_orientation} coverage")
        else:
            sell_points.append(f"current remit is anchored in {channel_orientation} distribution")
    elif "distribution" in tool_input.current_title.lower():
        sell_points.append("current title points to broad distribution leadership exposure")

    if scope_signal != "unclear":
        sell_points.append(f"public evidence points to {_scope_phrase(scope_signal)} scope")

    if mandate_similarity == "direct_match":
        sell_points.append("the remit already sits close to a head-of-distribution brief")
    elif mandate_similarity == "adjacent_match" and has_supported_firm_context:
        sell_points.append(f"{tool_input.current_employer} platform context is commercially relevant to the brief")
    elif mandate_similarity == "step_up_candidate":
        sell_points.append("the profile suggests stretch potential from channel ownership into broader distribution leadership")

    return _dedupe_preserve_order(sell_points)[:2]


def _derive_key_gaps(
    *,
    tool_input: CandidatePublicProfileLookupInput,
    claims: list[ResearchClaim],
    verification: VerificationSummary,
    channel_orientation: ChannelOrientation,
    scope_signal: ScopeSignal,
    seniority_signal: SenioritySignal,
    evidence_strength: EvidenceStrength,
) -> list[str]:
    prioritized_fields = verification.uncertain_fields + verification.missing_fields
    high_value_gaps: list[str] = []
    low_value_gaps: list[str] = []
    for field in prioritized_fields:
        if "mobility" in field.lower():
            continue
        normalized = _normalize_gap(
            field,
            tool_input=tool_input,
            claims=claims,
            channel_orientation=channel_orientation,
            scope_signal=scope_signal,
            seniority_signal=seniority_signal,
        )
        if not normalized:
            continue
        if _is_low_value_gap(normalized) or _is_deprioritized_gap(normalized, channel_orientation=channel_orientation):
            low_value_gaps.append(normalized)
        else:
            high_value_gaps.append(normalized)

    if not high_value_gaps:
        fallback_gap = _fallback_commercial_gap(
            tool_input=tool_input,
            claims=claims,
            channel_orientation=channel_orientation,
            scope_signal=scope_signal,
            seniority_signal=seniority_signal,
            evidence_strength=evidence_strength,
        )
        if fallback_gap:
            high_value_gaps.append(fallback_gap)

    combined = _dedupe_preserve_order(high_value_gaps + low_value_gaps)
    return (combined or ["commercial scope behind the title remains unclear"])[:2]


def _normalize_gap(
    field: str,
    *,
    tool_input: CandidatePublicProfileLookupInput,
    claims: list[ResearchClaim],
    channel_orientation: ChannelOrientation,
    scope_signal: ScopeSignal,
    seniority_signal: SenioritySignal,
) -> str:
    lowered = field.strip().lower()
    lane = "channel" if channel_orientation == "unclear" else channel_orientation
    geography = _market_phrase(scope_signal)
    title = tool_input.current_title.lower()
    claim_text = " ".join(claim.statement.lower() for claim in claims if claim.category in {"current_role", "channel_experience", "experience"})
    mapping = (
        ("team size", "team leadership scale behind the remit is not verified publicly"),
        ("team scale", "team leadership scale behind the remit is not verified publicly"),
        ("reporting line", "whether the role has included direct team leadership or mainly individual channel ownership is not verified publicly"),
        ("direct evidence of platform, ifa, and superannuation coverage", "how much direct platform, IFA, and super-fund coverage sits behind the remit is not verified publicly"),
        ("superannuation", "super fund coverage depth is not verified publicly"),
        ("family-office", "family-office coverage depth is not verified publicly"),
        ("family office", "family-office coverage depth is not verified publicly"),
        ("ifa", "IFA coverage depth is not verified publicly"),
        ("platform", "platform coverage depth is not verified publicly"),
        ("asset class", "product breadth across asset classes remains unclear"),
        ("product breadth", "product breadth across asset classes remains unclear"),
        ("institutional coverage", f"direct {geography} institutional network depth is not verified publicly"),
        ("wholesale coverage", f"direct {geography} wholesale/intermediary coverage depth is not verified publicly"),
        ("wealth coverage", f"direct {geography} wealth channel coverage depth is not verified publicly"),
        ("retail coverage", f"direct {geography} retail channel coverage depth is not verified publicly"),
        ("channel mix breadth", _mixed_or_lane_gap(channel_orientation=channel_orientation, scope_signal=scope_signal, seniority_signal=seniority_signal)),
        ("channel breadth", _mixed_or_lane_gap(channel_orientation=channel_orientation, scope_signal=scope_signal, seniority_signal=seniority_signal)),
        ("network depth", f"direct {geography} {lane} network depth is not verified publicly"),
        ("market profile", f"{geography.capitalize()} market profile behind the title is not verified publicly"),
        ("tenure", "precise current-role tenure remains unclear"),
        ("start date", "exact current-role start date remains unclear"),
        ("aum", "exact firm AUM remains unclear"),
    )
    for needle, label in mapping:
        if needle in lowered:
            return label
    if seniority_signal in {"head_level", "director_level"} and ("distribution" in title or "distribution" in claim_text):
        if channel_orientation == "wealth":
            return "whether the remit is primarily wealth-led or extends into broader intermediary and institutional coverage remains unclear"
        if channel_orientation == "wholesale":
            return "how much direct platform, IFA, and broader institutional coverage sits behind the remit is not verified publicly"
        if channel_orientation == "institutional":
            return f"how much direct {geography} institutional and super-fund coverage sits behind the remit is not verified publicly"
        if channel_orientation == "mixed":
            return "whether the role is mainly strategic oversight or still hands-on across key channels is not verified publicly"
        return "whether the role is mainly strategic oversight or hands-on channel leadership is not verified publicly"
    if lowered.endswith("verified"):
        return lowered
    if lowered.endswith("unclear"):
        return lowered
    return f"{field.strip()} remains unclear"


def _derive_scope_signal(
    *,
    tool_input: CandidatePublicProfileLookupInput,
    claims: list[ResearchClaim],
) -> ScopeSignal:
    relevant_claims = [
        claim.statement.lower()
        for claim in claims
        if claim.category in {"current_role", "channel_experience", "experience", "firm_context"}
    ]
    text = " ".join([tool_input.current_title.lower(), *relevant_claims])
    if any(term in text for term in ("global ", "global-", "worldwide", "international")):
        return "global"
    if any(term in text for term in ("anz", "australia and new zealand", "australia & new zealand")):
        return "anz"
    if any(term in text for term in ("apac", "asia pacific", "asia-pacific", "emea", "regional")):
        return "regional"
    if any(term in text for term in ("national", "australia", "australian")):
        return "national"
    if any(term in tool_input.current_title.lower() for term in ("head of distribution", "distribution director", "director distribution", "head of sales", "national bdm")):
        return "national"
    return "unclear"


def _derive_seniority_signal(*, tool_input: CandidatePublicProfileLookupInput) -> SenioritySignal:
    title = tool_input.current_title.lower()
    if any(term in title for term in ("chief", "global head", "regional head", "head ")):
        return "head_level"
    if any(term in title for term in ("managing director", "executive director", "director", "lead ")):
        return "director_level"
    if "bdm" in title or "business development manager" in title:
        return "bdm_level"
    if "manager" in title and any(term in title for term in ("distribution", "sales", "coverage", "relationship")):
        return "bdm_level"
    return "unclear"


def _derive_evidence_strength(*, claims: list[ResearchClaim]) -> EvidenceStrength:
    remit_claims = [claim for claim in claims if claim.category in {"current_role", "channel_experience", "experience", "firm_context"}]
    strong_count = sum(claim.verification_status in {"verified", "strongly_inferred"} for claim in remit_claims)
    verified_count = sum(claim.verification_status == "verified" for claim in remit_claims)
    if verified_count >= 1 and strong_count >= 3:
        return "strong"
    if strong_count >= 2:
        return "moderate"
    return "thin"


def _derive_screening_priority_question(
    *,
    key_gaps: list[str],
    channel_orientation: ChannelOrientation,
    scope_signal: ScopeSignal,
    seniority_signal: SenioritySignal,
) -> str:
    primary_gap = key_gaps[0] if key_gaps else ""
    lane = _lane_question_label(channel_orientation)
    geography = _market_phrase(scope_signal)
    if "network depth" in primary_gap:
        return f"How much direct {geography} {lane} network depth sits behind the current title?"
    if "team leadership scale" in primary_gap or "reporting line" in primary_gap:
        return "Has the role included direct leadership of a sales or distribution team, or mainly individual channel ownership?"
    if "super fund" in primary_gap:
        return "How much direct superannuation coverage has sat inside the remit in practice?"
    if "ifa" in primary_gap:
        return "How much direct IFA coverage has sat inside the remit in practice?"
    if "platform" in primary_gap:
        return "How much direct platform coverage has sat inside the remit in practice?"
    if "product breadth" in primary_gap:
        return "How broad is the product set behind the candidate's distribution remit?"
    if "platform, ifa, and super-fund" in primary_gap:
        return "How much direct platform, IFA, and superannuation coverage sits behind the current remit?"
    if "wealth-led" in primary_gap:
        return "Is the candidate's strength primarily wealth and intermediary-led, or does it extend into institutional coverage as well?"
    if "platform, ifa" in primary_gap:
        return "How much direct platform and IFA coverage sits behind the remit, and does it extend into institutional relationships as well?"
    if "institutional and super-fund" in primary_gap:
        return "How much direct institutional and superannuation coverage sits behind the current remit?"
    if "hands-on channel ownership or broader team leadership" in primary_gap:
        return "Has the role been mainly hands-on channel ownership, or has it already included broader team leadership?"
    if "hands-on the role remains versus broad strategic oversight" in primary_gap:
        return "How hands-on is the remit today versus broad strategic oversight across the channel mix?"
    if "strategic oversight" in primary_gap:
        return "Is the role mainly strategic oversight, or is the candidate still hands-on in channel leadership day to day?"
    if "multi-channel coverage" in primary_gap:
        return f"Is the candidate strongest in {lane}, or has the remit genuinely stretched across multiple channels?"
    if "market profile" in primary_gap:
        return f"How established is the candidate's {geography} market profile with allocators and intermediaries?"
    if "scope" in primary_gap:
        return "Was the remit national, ANZ, or narrower in practice?"
    if seniority_signal == "bdm_level":
        return "Has the remit already stretched beyond channel ownership into broader leadership scope?"
    return "What is the first commercial point that needs verifying behind the current remit?"


def _fallback_commercial_gap(
    *,
    tool_input: CandidatePublicProfileLookupInput,
    claims: list[ResearchClaim],
    channel_orientation: ChannelOrientation,
    scope_signal: ScopeSignal,
    seniority_signal: SenioritySignal,
    evidence_strength: EvidenceStrength,
) -> str:
    text = " ".join(claim.statement.lower() for claim in claims if claim.category in {"channel_experience", "experience", "current_role"})
    geography = _market_phrase(scope_signal)
    lane = _lane_question_label(channel_orientation)
    if evidence_strength == "thin":
        if channel_orientation == "wealth":
            return "whether the profile is mainly wealth-led or broader than wealth distribution remains unclear"
        if channel_orientation == "wholesale":
            return "how much direct platform, IFA, and broader institutional coverage sits behind the title is not verified publicly"
        if channel_orientation == "institutional":
            return f"how much direct {geography} institutional and super-fund coverage sits behind the title is not verified publicly"
        if channel_orientation != "unclear":
            return f"how much direct {geography} {lane} coverage sits behind the title is not verified publicly"
        return "commercial scope behind the title remains unclear"
    if seniority_signal in {"head_level", "director_level"} and "team" not in text:
        return "team leadership scale behind the remit is not verified publicly"
    if scope_signal == "unclear":
        return "whether the remit is national, ANZ, or narrower remains unclear"
    return f"how much direct {geography} {lane} coverage sits behind the remit is not verified publicly"


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def _is_low_value_gap(gap: str) -> bool:
    lowered = gap.lower()
    return "aum" in lowered or "start date" in lowered or "tenure" in lowered


def _is_deprioritized_gap(gap: str, *, channel_orientation: ChannelOrientation) -> bool:
    lowered = gap.lower()
    return channel_orientation == "mixed" and "multi-channel coverage" in lowered


def _scope_phrase(scope_signal: ScopeSignal) -> str:
    mapping = {
        "national": "national",
        "anz": "ANZ",
        "global": "global",
        "regional": "regional",
        "unclear": "unclear",
    }
    return mapping[scope_signal]


def _market_phrase(scope_signal: ScopeSignal) -> str:
    mapping = {
        "national": "australian",
        "anz": "anz",
        "global": "global",
        "regional": "regional",
        "unclear": "market",
    }
    return mapping[scope_signal]


def _lane_question_label(channel_orientation: ChannelOrientation) -> str:
    if channel_orientation == "mixed":
        return "distribution"
    if channel_orientation == "unclear":
        return "channel"
    return channel_orientation


def _mixed_or_lane_gap(
    *,
    channel_orientation: ChannelOrientation,
    scope_signal: ScopeSignal,
    seniority_signal: SenioritySignal,
) -> str:
    lane = "channel" if channel_orientation == "unclear" else channel_orientation
    if channel_orientation == "mixed":
        if seniority_signal == "director_level":
            return "whether the remit is mainly hands-on channel ownership or broader team leadership is not verified publicly"
        if seniority_signal == "head_level" and scope_signal == "global":
            return "how hands-on the role remains versus broad strategic oversight is not verified publicly"
        return "how much direct platform, IFA, and super-fund coverage sits behind the remit is not verified publicly"
    return f"how much {lane} versus broader multi-channel coverage sits behind the remit remains unclear"
