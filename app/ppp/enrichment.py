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
    key_sell_points: list[str] = Field(default_factory=list)
    key_gaps: list[str] = Field(default_factory=list)

    @field_validator("key_sell_points", "key_gaps")
    @classmethod
    def _limit_signal_lists(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        return cleaned[:2]


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
        if self.inferred_tenure_years is not None:
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
                verification_status="strongly_inferred" if inferred_tenure_years is not None else "uncertain",
                confidence="medium" if inferred_tenure_years is not None else "low",
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
        return None

    def _build_tenure_rationale(self, fixture: dict[str, Any], tenure_years: float | None) -> str:
        if isinstance(fixture.get("tenure_rationale"), str) and fixture["tenure_rationale"].strip():
            return fixture["tenure_rationale"].strip()
        if tenure_years is not None:
            return f"Estimated at approximately {tenure_years:.1f} years based on controlled research chronology."
        return "Exact current-role tenure could not be verified from available public research and remains a key follow-up item."

    def _format_firm_aum_context(self, employer: str, fixture: dict[str, Any]) -> str:
        if isinstance(fixture.get("firm_aum_context"), str) and fixture["firm_aum_context"].strip():
            return fixture["firm_aum_context"].strip()

        employer_type = fixture.get("employer_type", "funds-management firm")
        market_position = fixture.get("market_position", "established market participant")
        aum_descriptor = fixture.get("aum_descriptor", "unable to verify exact AUM from public sources")
        channel_focus = fixture.get("channel_focus", "distribution exposure inferred from the candidate's title")
        return (
            f"Unable to verify exact AUM from public sources for {employer}; "
            f"based on firm type and sector context, the platform appears broadly comparable to a {market_position} {employer_type}; "
            f"{aum_descriptor}; {channel_focus}."
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
    mandate_similarity = _derive_mandate_similarity(tool_input=tool_input, claims=claims, channel_orientation=channel_orientation)
    key_sell_points = _derive_key_sell_points(
        tool_input=tool_input,
        claims=claims,
        channel_orientation=channel_orientation,
        mandate_similarity=mandate_similarity,
    )
    key_gaps = _derive_key_gaps(verification=verification)
    return RecruiterSignals(
        channel_orientation=channel_orientation,
        mandate_similarity=mandate_similarity,
        key_sell_points=key_sell_points,
        key_gaps=key_gaps,
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
    explicit_categories: list[ChannelOrientation] = []
    inferred_categories: list[ChannelOrientation] = []

    title_categories = _extract_channel_categories(tool_input.current_title)
    if title_categories:
        explicit_categories.extend(title_categories)

    for claim in claims:
        text = claim.statement.lower()
        categories = _extract_channel_categories(text)
        if not categories:
            continue
        if claim.category in {"channel_experience", "current_role"}:
            explicit_categories.extend(categories)
        elif claim.category in {"experience", "firm_context"}:
            inferred_categories.extend(categories)

    chosen = explicit_categories or inferred_categories
    unique = sorted(set(chosen))
    if not unique:
        return "unclear"
    if len(unique) > 1:
        return "mixed"
    return unique[0]


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
) -> MandateSimilarity:
    title = tool_input.current_title.lower()
    claim_text = " ".join(claim.statement.lower() for claim in claims if claim.category in {"current_role", "channel_experience", "experience"})

    has_head_distribution = "head of distribution" in title
    has_national_bdm = "national bdm" in title
    has_head_sales = "head of sales" in title or "head of retail sales" in title or "head of wholesale" in title
    has_distribution_lead = "distribution" in title and any(term in title for term in ("head", "director", "lead"))
    has_channel_lead = any(term in title for term in ("director", "head", "lead")) and any(
        term in title for term in ("institutional", "wholesale", "wealth", "retail")
    )
    has_bdm = "bdm" in title or "business development manager" in title
    seniority_head_or_director = any(term in title for term in ("head", "director", "lead"))

    if has_head_distribution or has_national_bdm:
        return "direct_match"
    if has_distribution_lead and channel_orientation == "mixed":
        return "direct_match"
    if has_head_sales or has_channel_lead or ("distribution" in claim_text and seniority_head_or_director):
        return "adjacent_match"
    if has_bdm or ("manager" in title and channel_orientation != "unclear"):
        return "step_up_candidate"
    return "unclear_fit"


def _derive_key_sell_points(
    *,
    tool_input: CandidatePublicProfileLookupInput,
    claims: list[ResearchClaim],
    channel_orientation: ChannelOrientation,
    mandate_similarity: MandateSimilarity,
) -> list[str]:
    sell_points: list[str] = []
    title = tool_input.current_title.lower()
    claim_text = " ".join(claim.statement.lower() for claim in claims)

    if any(term in title for term in ("head", "director", "lead", "national bdm")):
        sell_points.append("current title already signals senior distribution ownership")

    if channel_orientation != "unclear":
        sell_points.append(f"public evidence supports {channel_orientation} channel relevance")

    if (
        mandate_similarity in {"direct_match", "adjacent_match"}
        and any(claim.category == "firm_context" and claim.verification_status != "uncertain" for claim in claims)
    ):
        sell_points.append("employer context is comparable to the target mandate")

    if (
        mandate_similarity == "step_up_candidate"
        and any(term in claim_text for term in ("distribution", "institutional", "wholesale", "wealth", "retail"))
    ):
        sell_points.append("visible public remit suggests exposure to distribution leadership themes")

    deduped: list[str] = []
    for point in sell_points:
        if point not in deduped:
            deduped.append(point)
    return deduped[:2]


def _derive_key_gaps(*, verification: VerificationSummary) -> list[str]:
    prioritized_fields = verification.uncertain_fields + verification.missing_fields
    gaps: list[str] = []
    for field in prioritized_fields:
        normalized = _normalize_gap(field)
        if normalized and normalized not in gaps:
            gaps.append(normalized)
        if len(gaps) == 2:
            break
    if not gaps:
        return ["public scope details remain unclear"]
    return gaps


def _normalize_gap(field: str) -> str:
    lowered = field.strip().lower()
    mapping = (
        ("team size", "team scale not verified"),
        ("reporting line", "reporting line not verified"),
        ("superannuation", "superannuation coverage depth not verified"),
        ("family-office", "family-office coverage depth not verified"),
        ("family office", "family-office coverage depth not verified"),
        ("ifa", "IFA coverage depth not verified"),
        ("platform", "platform coverage depth not verified"),
        ("asset class", "product breadth across asset classes remains unclear"),
        ("product breadth", "product breadth across asset classes remains unclear"),
        ("institutional coverage", "direct institutional coverage still needs confirmation"),
        ("channel mix breadth", "channel breadth remains unclear"),
        ("channel breadth", "channel breadth remains unclear"),
        ("network depth", "network depth is not verified publicly"),
        ("tenure", "precise current-role tenure remains unclear"),
    )
    for needle, label in mapping:
        if needle in lowered:
            return label
    if lowered.endswith("verified"):
        return lowered
    if lowered.endswith("unclear"):
        return lowered
    return f"{field.strip()} remains unclear"
