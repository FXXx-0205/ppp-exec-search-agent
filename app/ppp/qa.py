from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

from pydantic import BaseModel, ValidationError

from app.ppp.enrichment import CandidateEnrichmentResult, ResearchClaim, validate_enrichment_payload
from app.ppp.models import CandidateCSVRow
from app.ppp.schema import CandidateBrief, PPPOutput, validate_output_document

QUALIFIER_TERMS = (
    "unverified",
    "verification",
    "unable to verify",
    "estimated",
    "limited public visibility",
    "cautious",
    "approx",
    "open question",
    "questions open",
    "public evidence remains limited",
    "not verified",
    "remains unclear",
    "needs confirmation",
    "should be verified",
    "boundary",
    "public evidence does not confirm",
    "public filings do not confirm",
    "public aum remains unverified",
    "exact aum is unavailable",
    "treated here as",
    "pending conversation",
    "should be confirmed in conversation",
    "remains unverified",
)
PLACEHOLDER_TERMS = ("unknown candidate", "n/a", "tbd", "lorem ipsum")
CAREER_NARRATIVE_FORBIDDEN_TERMS = (
    "built the function",
    "transformed",
    "led the build-out",
    "established track record",
    "positioned as",
    "scaled the function",
    "drove growth",
)
MOBILITY_FORBIDDEN_TERMS = (
    "settled",
    "itchy",
    "ready to move",
    "open to move",
    "flight risk",
    "unlikely to move",
    "natural transition",
    "multi-year commitment",
    "mobility appetite",
)
ROLE_FIT_FORBIDDEN_TERMS = (
    "proven",
    "deep network",
    "strong fit",
    "well beyond the role",
    "demonstrated capability",
    "established profile",
)
BAD_PHRASE_TERMS = (
    "a established",
    "distribution distribution",
    "credible angle",
    "adjacent transferability",
    "what stood out was",
    "anchored in",
    "adjacent option",
    "appears relevant",
    "appears relevant to a comparable remit",
    "should be handled as strong evidence until tested",
    "public evidence supports relevance",
    "keeps this profile in frame as a",
    "currently sitting in",
    "from a recruiter lens, the relevance comes from",
    "that puts the profile close to the",
    "the main commercial gap is",
    "the first screening question is",
)
STOPWORDS = {
    "about",
    "across",
    "appears",
    "around",
    "asset",
    "based",
    "build",
    "built",
    "candidate",
    "channel",
    "client",
    "coverage",
    "current",
    "exact",
    "experience",
    "firm",
    "funds",
    "leadership",
    "limited",
    "management",
    "manager",
    "public",
    "remains",
    "still",
    "their",
    "there",
    "would",
    "should",
    "requires",
    "verification",
}
LANE_TERMS = ("institutional", "wholesale", "wealth", "retail", "ifa", "platform", "distribution")
SCOPE_TERMS = ("national", "anz", "global", "regional", "australia", "new zealand", "apac")
SCREENING_ANGLE_TERMS = (
    "first screening question",
    "key point to test",
    "test in conversation",
    "screening question",
    "screening conversation",
    "screening priority",
)
GENERIC_INVITATION_TERMS = (
    "welcome a conversation",
    "caught our attention",
    "worth discussing",
    "worth a conversation",
    "your background appears relevant",
)
COMMERCIAL_ANGLE_TERMS = (
    "institutional",
    "wholesale",
    "wealth",
    "retail",
    "build-out",
    "build out",
    "expansion",
    "step-up",
    "step up",
    "comparable",
    "adjacent",
    "distribution remit",
    "distribution brief",
    "coverage brief",
    "coverage",
    "remit",
    "leadership",
    "anz",
    "national",
)
COMMERCIAL_RELEVANCE_TERMS = ("direct match", "adjacent", "step-up", "commercially", "mandate", "in frame", "relevance")
LOW_VALUE_GAP_TERMS = ("aum", "start date", "tenure")
HIGH_VALUE_GAP_TERMS = ("network depth", "team", "platform", "ifa", "super", "institutional coverage", "product breadth", "market profile", "channel")


class QAFinding(BaseModel):
    candidate_id: str | None = None
    severity: str
    check: str
    message: str


class QAReport(BaseModel):
    passed: bool
    findings: list[QAFinding]


def run_bundle_qa(
    *,
    output: PPPOutput,
    candidates: dict[str, CandidateCSVRow],
    enrichments: dict[str, CandidateEnrichmentResult],
) -> QAReport:
    findings: list[QAFinding] = []
    for candidate in output.candidates:
        input_candidate = candidates.get(candidate.candidate_id)
        enrichment = enrichments.get(candidate.candidate_id)
        findings.extend(_check_candidate(candidate, input_candidate, enrichment))
    findings.extend(_check_bundle_differentiation(output=output, enrichments=enrichments))
    return QAReport(passed=not any(item.severity == "error" for item in findings), findings=findings)


def validate_output_bundle(
    *,
    output_path: str,
    input_path: str,
    intermediate_dir: str,
) -> QAReport:
    findings: list[QAFinding] = []

    try:
        raw_output = Path(output_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return QAReport(passed=False, findings=[_error(None, "output_file", f"Output file not found: {output_path}")])

    try:
        output_payload = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        return QAReport(passed=False, findings=[_error(None, "output_json", f"Output JSON is not parseable: {exc}")])

    try:
        output = validate_output_document(output_payload)
    except ValidationError as exc:
        return QAReport(passed=False, findings=[_error(None, "output_schema", str(exc.errors()[0]["msg"]))])

    try:
        candidates = _load_candidates(input_path)
    except Exception as exc:
        findings.append(_error(None, "input_csv", f"Input CSV validation failed: {exc}"))
        return QAReport(passed=False, findings=findings)

    try:
        enrichments = _load_enrichments(intermediate_dir)
    except Exception as exc:
        findings.append(_error(None, "enrichment_artifacts", f"Intermediate enrichment validation failed: {exc}"))
        return QAReport(passed=False, findings=findings)

    report = run_bundle_qa(output=output, candidates=candidates, enrichments=enrichments)
    return report


def write_qa_report(report: QAReport, *, path: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _check_candidate(
    candidate: CandidateBrief,
    input_candidate: CandidateCSVRow | None,
    enrichment: CandidateEnrichmentResult | None,
) -> list[QAFinding]:
    findings: list[QAFinding] = []

    if _sentence_count(candidate.career_narrative) < 3:
        findings.append(_error(candidate.candidate_id, "career_narrative_sentences", "career_narrative must contain at least 3 sentences."))

    if _sentence_count(candidate.outreach_hook) != 1:
        findings.append(_error(candidate.candidate_id, "outreach_hook_sentences", "outreach_hook must contain exactly 1 sentence."))

    mobility_sentences = _sentence_count(candidate.mobility_signal.rationale)
    if mobility_sentences < 1 or mobility_sentences > 2:
        findings.append(
            _error(candidate.candidate_id, "mobility_signal_sentences", "mobility_signal.rationale must contain 1 to 2 sentences.")
        )

    role_fit_sentences = _sentence_count(candidate.role_fit.justification)
    if role_fit_sentences != 3:
        findings.append(
            _error(candidate.candidate_id, "role_fit_sentences", "role_fit.justification must contain exactly 3 sentences.")
        )

    combined = " ".join(
        [
            candidate.career_narrative,
            candidate.firm_aum_context,
            candidate.mobility_signal.rationale,
            candidate.role_fit.justification,
            candidate.outreach_hook,
        ]
    ).lower()
    for term in PLACEHOLDER_TERMS:
        if term in combined:
            findings.append(_error(candidate.candidate_id, "placeholder_text", f"Output contains placeholder text: '{term}'."))

    for term in CAREER_NARRATIVE_FORBIDDEN_TERMS:
        if term in candidate.career_narrative.lower():
            findings.append(
                _error(candidate.candidate_id, "career_narrative_forbidden_phrase", f"career_narrative contains forbidden phrasing: '{term}'.")
            )

    for term in MOBILITY_FORBIDDEN_TERMS:
        if term in candidate.mobility_signal.rationale.lower():
            findings.append(
                _error(candidate.candidate_id, "mobility_signal_forbidden_phrase", f"mobility_signal contains forbidden phrasing: '{term}'.")
            )

    for term in ROLE_FIT_FORBIDDEN_TERMS:
        if term in candidate.role_fit.justification.lower():
            findings.append(
                _error(candidate.candidate_id, "role_fit_forbidden_phrase", f"role_fit.justification contains forbidden phrasing: '{term}'.")
            )
    for term in BAD_PHRASE_TERMS:
        if term in combined:
            findings.append(_error(candidate.candidate_id, "bad_phrase_suppression", f"Output contains banned phrasing: '{term}'."))

    findings.extend(_check_recruiter_usefulness(candidate, enrichment))
    findings.extend(_check_uncertainty_expression(candidate, enrichment))

    if not _justification_mentions_evidence(candidate, input_candidate, enrichment):
        findings.append(
            _error(
                candidate.candidate_id,
                "claim_boundary_role_fit",
                "role_fit.justification should stay anchored to employer, title, or supported research claims.",
            )
        )

    if _mentions_specific_aum_without_support(candidate.firm_aum_context, enrichment):
        findings.append(
            _error(
                candidate.candidate_id,
                "claim_boundary_firm_aum_context",
                "firm_aum_context mentions a specific AUM figure without verified claim support or qualifying language.",
            )
        )
    if enrichment is not None and not _firm_aum_context_handles_uncertainty(candidate.firm_aum_context, enrichment):
        findings.append(
            _error(
                candidate.candidate_id,
                "firm_aum_context_uncertainty",
                "firm_aum_context should explicitly state when exact AUM cannot be verified and keep any estimate qualitative.",
            )
        )

    if enrichment is not None:
        if enrichment.inferred_tenure_years is None and not _contains_qualifier(combined):
            findings.append(
                _error(
                    candidate.candidate_id,
                    "claim_boundary_tenure_confidence",
                    "tenure_years is not grounded in a supported tenure claim and the narrative should signal uncertainty more clearly.",
                )
            )
        if enrichment.inferred_tenure_years is not None and abs(candidate.current_role.tenure_years - enrichment.inferred_tenure_years) > 0.25:
            findings.append(
                _error(
                    candidate.candidate_id,
                    "claim_boundary_tenure_alignment",
                    "current_role.tenure_years deviates materially from the supported tenure claim.",
                )
            )
        if enrichment.verification.uncertain_fields and not _contains_qualifier(combined):
            findings.append(
                _error(
                    candidate.candidate_id,
                    "claim_boundary_uncertainty_language",
                    "The research package marks fields as uncertain, but the final output does not clearly signal verification limits.",
                )
            )
        unsupported_output_fields = _unsupported_output_fields(candidate, enrichment)
        for field_name in unsupported_output_fields:
            findings.append(
                _error(
                    candidate.candidate_id,
                    "claim_boundary_output_field",
                    f"{field_name} appears to go beyond the supported research claim set.",
                )
            )

    return findings


def _load_candidates(input_path: str) -> dict[str, CandidateCSVRow]:
    from app.ppp.pipeline import _load_candidates_csv

    rows = _load_candidates_csv(Path(input_path))
    return {f"candidate_{index}": row for index, row in enumerate(rows, start=1)}


def _load_enrichments(intermediate_dir: str) -> dict[str, CandidateEnrichmentResult]:
    results: dict[str, CandidateEnrichmentResult] = {}
    base = Path(intermediate_dir)
    if not base.exists():
        return results
    for path in base.glob("*_enriched.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = validate_enrichment_payload(payload)
        results[result.candidate_id] = result
    return results


def _justification_mentions_evidence(
    candidate: CandidateBrief,
    input_candidate: CandidateCSVRow | None,
    enrichment: CandidateEnrichmentResult | None,
) -> bool:
    text = candidate.role_fit.justification.lower()
    if input_candidate is not None:
        if input_candidate.current_employer.lower() in text:
            return True
        title_tokens = [part.strip().lower() for part in input_candidate.current_title.replace("/", " ").split() if len(part.strip()) > 3]
        if any(token in text for token in title_tokens):
            return True
    if enrichment is not None:
        signal_phrases = [
            *enrichment.recruiter_signals.key_sell_points,
            *enrichment.recruiter_signals.key_gaps,
            enrichment.recruiter_signals.channel_orientation,
        ]
        if any(phrase.lower() in text for phrase in signal_phrases if phrase and phrase != "unclear"):
            return True
        claims = enrichment.claims_for_output_field("role_fit")
        if _text_grounded_in_claims(text, claims, field_name="role_fit"):
            return True
    return any(tag.lower() in text for tag in candidate.experience_tags)


def _mentions_specific_aum_without_support(text: str, enrichment: CandidateEnrichmentResult | None) -> bool:
    has_amount = bool(re.search(r"(\$|aud\s*)\d+(\.\d+)?\s*[bm]", text.lower()))
    if not has_amount:
        return False
    if _contains_qualifier(text):
        return False
    if enrichment is None:
        return True
    for claim in enrichment.claims_for_output_field("firm_aum_context"):
        if claim.verification_status == "verified" and re.search(r"(\$|aud\s*)\d+(\.\d+)?\s*[bm]", claim.statement.lower()):
            return False
    return True


def _contains_qualifier(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in QUALIFIER_TERMS)


def _unsupported_output_fields(candidate: CandidateBrief, enrichment: CandidateEnrichmentResult) -> list[str]:
    unsupported: list[str] = []
    checks = {
        "career_narrative": candidate.career_narrative,
        "mobility_signal": candidate.mobility_signal.rationale,
        "firm_aum_context": candidate.firm_aum_context,
        "role_fit": candidate.role_fit.justification,
    }
    for field_name, text in checks.items():
        support = enrichment.field_support(field_name)
        claims = enrichment.claims_for_output_field(field_name)
        if field_name == "role_fit" and _role_fit_grounded_in_claims_or_signals(text, enrichment):
            continue
        if support.supported_by_claim_ids and _field_exceeds_claim_boundary(text, claims, field_name=field_name):
            unsupported.append(field_name)
    return unsupported


def _field_exceeds_claim_boundary(text: str, claims: list[ResearchClaim], *, field_name: str) -> bool:
    if not claims:
        return False
    return not _text_grounded_in_claims(text, claims, field_name=field_name)


def _text_grounded_in_claims(text: str, claims: list[ResearchClaim], *, field_name: str) -> bool:
    sentences = [sentence for sentence in _split_sentences(text) if sentence]
    if not sentences:
        return False

    if field_name == "firm_aum_context":
        return _firm_context_is_grounded(text, claims)

    if field_name == "role_fit":
        return _all_sentences_supported(sentences, claims, min_overlap=1, min_similarity=0.48)

    if field_name == "mobility_signal":
        return _all_sentences_supported(sentences, claims, min_overlap=1, min_similarity=0.42)

    if field_name == "outreach_hook":
        return _all_sentences_supported(sentences, claims, min_overlap=1, min_similarity=0.38)

    return _all_sentences_supported(sentences, claims, min_overlap=2, min_similarity=0.45)


def _firm_context_is_grounded(text: str, claims: list[ResearchClaim]) -> bool:
    if not claims:
        return True
    if _contains_qualifier(text):
        return True

    claim_numbers = {number for claim in claims for number in _extract_numbers(claim.statement)}
    text_numbers = _extract_numbers(text)
    if text_numbers and not text_numbers.issubset(claim_numbers):
        return False

    return _all_sentences_supported(_split_sentences(text), claims, min_overlap=1, min_similarity=0.4)


def _all_sentences_supported(
    sentences: list[str],
    claims: list[ResearchClaim],
    *,
    min_overlap: int,
    min_similarity: float,
) -> bool:
    return all(
        _sentence_supported_by_claims(sentence, claims, min_overlap=min_overlap, min_similarity=min_similarity)
        for sentence in sentences
    )


def _sentence_supported_by_claims(
    sentence: str,
    claims: list[ResearchClaim],
    *,
    min_overlap: int,
    min_similarity: float,
) -> bool:
    lowered = sentence.lower()
    if _contains_qualifier(lowered):
        return True

    sentence_tokens = _meaningful_tokens(lowered)
    if not sentence_tokens:
        return True

    for claim in claims:
        claim_tokens = _meaningful_tokens(claim.statement.lower())
        overlap = len(sentence_tokens & claim_tokens)
        similarity = difflib.SequenceMatcher(a=lowered, b=claim.statement.lower()).ratio()
        if overlap >= min_overlap or similarity >= min_similarity:
            return True
        if _extract_numbers(sentence).issubset(_extract_numbers(claim.statement)) and overlap >= 1:
            return True
    return False


def _meaningful_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z]{4,}", text.lower())
        if token not in STOPWORDS
    }


def _extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"(?:aud\s*)?\$?\d+(?:\.\d+)?\s*[bm]?", text.lower()))


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def _sentence_count(text: str) -> int:
    return len(_split_sentences(text))


def _error(candidate_id: str | None, check: str, message: str) -> QAFinding:
    return QAFinding(candidate_id=candidate_id, severity="error", check=check, message=message)


def _check_recruiter_usefulness(
    candidate: CandidateBrief,
    enrichment: CandidateEnrichmentResult | None,
) -> list[QAFinding]:
    findings: list[QAFinding] = []

    career_text = candidate.career_narrative.lower()
    if not _contains_lane_signal(candidate.career_narrative, enrichment):
        findings.append(
            _error(
                candidate.candidate_id,
                "career_narrative_lane_signal",
                "career_narrative must include a candidate-specific lane, channel, or remit signal.",
            )
        )
    if not _contains_qualifier(candidate.career_narrative):
        findings.append(
            _error(
                candidate.candidate_id,
                "career_narrative_boundary",
                "career_narrative must include an explicit verification boundary or uncertainty signal.",
            )
        )
    if _looks_like_title_plus_caution_only(candidate.career_narrative):
        findings.append(
            _error(
                candidate.candidate_id,
                "career_narrative_generic",
                "career_narrative cannot stop at title summary plus generic caution; it should also explain lane relevance.",
            )
        )
    if not _contains_commercial_relevance_signal(candidate.career_narrative, enrichment):
        findings.append(
            _error(
                candidate.candidate_id,
                "career_narrative_commercial_relevance",
                "career_narrative must include at least one commercial relevance signal tied to lane, scope, or fit type.",
            )
        )

    role_fit_text = candidate.role_fit.justification.lower()
    if not _contains_supported_strength(candidate.role_fit.justification, enrichment):
        findings.append(
            _error(
                candidate.candidate_id,
                "role_fit_supported_strength",
                "role_fit.justification must include at least one supported strength.",
            )
        )
    if not _contains_concrete_gap(candidate.role_fit.justification, enrichment):
        findings.append(
            _error(
                candidate.candidate_id,
                "role_fit_concrete_gap",
                "role_fit.justification must name at least one concrete commercial gap.",
            )
        )
    if not any(term in role_fit_text for term in SCREENING_ANGLE_TERMS):
        findings.append(
            _error(
                candidate.candidate_id,
                "role_fit_screening_angle",
                "role_fit.justification must include an explicit screening angle or what-to-test sentence.",
            )
        )
    if not _contains_lane_or_scope_signal(candidate.role_fit.justification, enrichment):
        findings.append(
            _error(
                candidate.candidate_id,
                "role_fit_lane_or_scope_signal",
                "role_fit.justification must reference at least one lane, scope, or market-specific signal.",
            )
        )
    if enrichment is not None and _has_only_low_value_gaps(enrichment.recruiter_signals.key_gaps):
        findings.append(
            _error(
                candidate.candidate_id,
                "low_value_gap_suppression",
                "recruiter_signals.key_gaps cannot collapse to only exact AUM or date-style admin gaps when stronger commercial gaps are available.",
            )
        )

    hook_text = candidate.outreach_hook.lower()
    if any(term in hook_text for term in GENERIC_INVITATION_TERMS) and not _contains_specific_commercial_angle(candidate.outreach_hook, enrichment):
        findings.append(
            _error(
                candidate.candidate_id,
                "outreach_hook_generic_invitation",
                "outreach_hook cannot be just a generic invitation; it needs a concrete commercial angle.",
            )
        )
    if not _contains_specific_commercial_angle(candidate.outreach_hook, enrichment):
        findings.append(
            _error(
                candidate.candidate_id,
                "outreach_hook_commercial_angle",
                "outreach_hook must include one specific commercial angle grounded in recruiter_signals or supported claims.",
            )
        )

    return findings


def _check_uncertainty_expression(
    candidate: CandidateBrief,
    enrichment: CandidateEnrichmentResult | None,
) -> list[QAFinding]:
    findings: list[QAFinding] = []

    if not _mobility_has_uncertainty_structure(candidate.mobility_signal.rationale):
        findings.append(
            _error(
                candidate.candidate_id,
                "mobility_signal_uncertainty_structure",
                "mobility_signal.rationale must separate chronology observation from absent move-readiness evidence and conversation follow-up.",
            )
        )

    if not _contains_qualifier(candidate.career_narrative):
        findings.append(
            _error(
                candidate.candidate_id,
                "career_narrative_uncertainty_marker",
                "career_narrative must make unverified scope or trajectory limits explicit.",
            )
        )

    role_fit_text = candidate.role_fit.justification.lower()
    if (
        "unverified" not in role_fit_text
        and "does not confirm" not in role_fit_text
        and "remains unclear" not in role_fit_text
        and "not verified" not in role_fit_text
    ):
        findings.append(
            _error(
                candidate.candidate_id,
                "role_fit_unverified_gap_language",
                "role_fit.justification must distinguish supported relevance from unverified requirement coverage.",
            )
        )

    return findings


def _contains_lane_signal(text: str, enrichment: CandidateEnrichmentResult | None) -> bool:
    lowered = text.lower()
    if any(term in lowered for term in LANE_TERMS):
        return True
    if enrichment is None:
        return False
    orientation = enrichment.recruiter_signals.channel_orientation
    return orientation != "unclear" and orientation in lowered


def _looks_like_title_plus_caution_only(text: str) -> bool:
    sentences = _split_sentences(text.lower())
    if len(sentences) < 3:
        return False
    second_sentence = sentences[1]
    third_sentence = sentences[2]
    has_relevance_signal = any(term in second_sentence for term in LANE_TERMS) or any(
        term in second_sentence for term in ("relevant", "mandate", "match", "remit", "frame", "commercial")
    )
    return not has_relevance_signal and _contains_qualifier(third_sentence)


def _contains_supported_strength(text: str, enrichment: CandidateEnrichmentResult | None) -> bool:
    lowered = text.lower()
    if enrichment is not None:
        sell_points = [point.lower() for point in enrichment.recruiter_signals.key_sell_points]
        if any(point in lowered for point in sell_points):
            return True
        if enrichment.current_employer.lower() in lowered and any(term in lowered for term in ("remit", "relevance", "in frame", "distribution")):
            return True
        return _text_grounded_in_claims(text, enrichment.claims_for_output_field("role_fit"), field_name="role_fit")
    return any(term in lowered for term in ("title", "employer", "distribution", "institutional", "wholesale", "wealth", "retail"))


def _contains_concrete_gap(text: str, enrichment: CandidateEnrichmentResult | None) -> bool:
    lowered = text.lower()
    if enrichment is not None:
        gaps = [gap.lower() for gap in enrichment.recruiter_signals.key_gaps]
        if any(gap in lowered for gap in gaps):
            return True
    concrete_gap_terms = (
        "team scale",
        "team leadership scale",
        "reporting line",
        "channel breadth",
        "network depth",
        "coverage depth",
        "market profile",
        "product breadth",
        "super fund",
        "platform",
        "ifa",
        "not verified",
        "remains unclear",
        "needs confirmation",
    )
    return any(term in lowered for term in concrete_gap_terms)


def _contains_specific_commercial_angle(text: str, enrichment: CandidateEnrichmentResult | None) -> bool:
    lowered = text.lower()
    if any(term in lowered for term in COMMERCIAL_ANGLE_TERMS):
        return True
    if enrichment is None:
        return False
    signals = enrichment.recruiter_signals
    if signals.channel_orientation != "unclear" and signals.channel_orientation in lowered:
        return True
    if signals.mandate_similarity.replace("_", " ") in lowered:
        return True
    if signals.scope_signal != "unclear" and signals.scope_signal in lowered:
        return True
    return any(point.lower() in lowered for point in signals.key_sell_points)


def _firm_aum_context_handles_uncertainty(text: str, enrichment: CandidateEnrichmentResult) -> bool:
    has_verified_numeric = any(
        claim.verification_status == "verified" and bool(re.search(r"(\$|aud\s*)\d+(\.\d+)?\s*[bm]", claim.statement.lower()))
        for claim in enrichment.claims_for_output_field("firm_aum_context")
    )
    if has_verified_numeric:
        return True
    lowered = text.lower()
    if bool(re.search(r"(\$|aud\s*)\d+(\.\d+)?\s*[bm]", lowered)):
        return False
    required_markers = (
        "unable to verify",
        "public evidence does not confirm",
        "public filings do not confirm",
        "public aum remains unverified",
        "exact aum is unavailable",
        "estimated",
        "treated here as",
        "based on firm type and sector context",
        "firm profile indicates",
    )
    return any(marker in lowered for marker in required_markers)


def _mobility_has_uncertainty_structure(text: str) -> bool:
    lowered = text.lower()
    chronology_markers = ("public chronology", "approximately", "current-role tenure", "years in the current role")
    uncertainty_markers = ("uncertain", "no direct public signal", "pending conversation", "requires follow-up")
    follow_up_markers = ("pending conversation", "follow-up", "should be treated as uncertain")
    return (
        any(marker in lowered for marker in chronology_markers)
        and any(marker in lowered for marker in uncertainty_markers)
        and any(marker in lowered for marker in follow_up_markers)
    )


def _role_fit_grounded_in_claims_or_signals(text: str, enrichment: CandidateEnrichmentResult) -> bool:
    lowered = text.lower()
    signals = enrichment.recruiter_signals
    supported_phrases = [
        *[point.lower() for point in signals.key_sell_points],
        *[gap.lower() for gap in signals.key_gaps],
        signals.screening_priority_question.lower(),
        signals.channel_orientation.lower(),
        signals.scope_signal.lower(),
        enrichment.current_employer.lower(),
        enrichment.current_title.lower(),
    ]
    if any(phrase and phrase in lowered for phrase in supported_phrases if phrase != "unclear"):
        return True
    return _text_grounded_in_claims(text, enrichment.claims_for_output_field("role_fit"), field_name="role_fit")


def _contains_commercial_relevance_signal(text: str, enrichment: CandidateEnrichmentResult | None) -> bool:
    lowered = text.lower()
    if any(term in lowered for term in COMMERCIAL_RELEVANCE_TERMS):
        return True
    if enrichment is None:
        return False
    if enrichment.recruiter_signals.mandate_similarity.replace("_", " ") in lowered:
        return True
    if any(point.lower() in lowered for point in enrichment.recruiter_signals.key_sell_points):
        return True
    return _contains_lane_or_scope_signal(text, enrichment)


def _contains_lane_or_scope_signal(text: str, enrichment: CandidateEnrichmentResult | None) -> bool:
    lowered = text.lower()
    if any(term in lowered for term in LANE_TERMS + SCOPE_TERMS):
        return True
    if enrichment is None:
        return False
    signals = enrichment.recruiter_signals
    if signals.channel_orientation != "unclear" and signals.channel_orientation in lowered:
        return True
    if signals.scope_signal != "unclear" and signals.scope_signal in lowered:
        return True
    return False


def _has_only_low_value_gaps(gaps: list[str]) -> bool:
    cleaned = [gap.lower() for gap in gaps if gap.strip()]
    if not cleaned:
        return False
    has_high_value = any(any(term in gap for term in HIGH_VALUE_GAP_TERMS) for gap in cleaned)
    if has_high_value:
        return False
    return all(any(term in gap for term in LOW_VALUE_GAP_TERMS) for gap in cleaned)


def _check_bundle_differentiation(
    *,
    output: PPPOutput,
    enrichments: dict[str, CandidateEnrichmentResult],
) -> list[QAFinding]:
    findings: list[QAFinding] = []
    for field_name in ("career_narrative", "role_fit.justification", "outreach_hook"):
        masked_seen: dict[str, tuple[str, str]] = {}
        for candidate in output.candidates:
            enrichment = enrichments.get(candidate.candidate_id)
            masked = _mask_candidate_specific_terms(_extract_field(candidate, field_name), enrichment)
            normalized = re.sub(r"\s+", " ", masked.strip().lower())
            signal_fingerprint = _signal_fingerprint(enrichment)
            if normalized in masked_seen and masked_seen[normalized][1] != signal_fingerprint:
                findings.append(
                    _error(
                        candidate.candidate_id,
                        f"{field_name}_template_similarity",
                        f"{field_name} is still too similar to {masked_seen[normalized][0]} once candidate-specific names and employers are removed.",
                    )
                )
            else:
                masked_seen[normalized] = (candidate.candidate_id, signal_fingerprint)

    question_counts: dict[str, list[tuple[str, str]]] = {}
    gap_counts: dict[str, list[tuple[str, str]]] = {}
    for candidate_id, enrichment in enrichments.items():
        question = re.sub(r"\s+", " ", enrichment.recruiter_signals.screening_priority_question.strip().lower())
        fingerprint = _signal_fingerprint(enrichment)
        question_counts.setdefault(question, []).append((candidate_id, fingerprint))
        for gap in enrichment.recruiter_signals.key_gaps:
            normalized_gap = re.sub(r"\s+", " ", gap.strip().lower())
            gap_counts.setdefault(normalized_gap, []).append((candidate_id, fingerprint))

    for question, candidate_entries in question_counts.items():
        if len(candidate_entries) >= 3 and len({fingerprint for _, fingerprint in candidate_entries}) >= 2:
            findings.append(
                _error(
                    None,
                    "screening_question_template_similarity",
                    f"screening_priority_question is repeating across too many candidates: '{question}'.",
                )
            )
    for gap, candidate_entries in gap_counts.items():
        if len(candidate_entries) >= 3 and len({fingerprint for _, fingerprint in candidate_entries}) >= 2 and (
            "multi-channel" in gap or "aum" in gap or "start date" in gap or "mobility" in gap
        ):
            findings.append(
                _error(
                    None,
                    "key_gap_template_similarity",
                    f"key_gaps are repeating as a low-value template across too many candidates: '{gap}'.",
                )
            )
    return findings


def _mask_candidate_specific_terms(text: str, enrichment: CandidateEnrichmentResult | None) -> str:
    masked = text
    if enrichment is None:
        return masked
    replacements = [
        enrichment.full_name,
        enrichment.current_employer,
        enrichment.current_title,
        "Head of Distribution / National BDM",
    ]
    for value in replacements:
        if value:
            masked = re.sub(re.escape(value), "<masked>", masked, flags=re.IGNORECASE)
    return masked


def _signal_fingerprint(enrichment: CandidateEnrichmentResult | None) -> str:
    if enrichment is None:
        return "no-enrichment"
    signals = enrichment.recruiter_signals
    return "|".join(
        [
            signals.channel_orientation,
            signals.scope_signal,
            signals.mandate_similarity,
            signals.seniority_signal,
            ",".join(signals.key_gaps),
            signals.screening_priority_question,
        ]
    )


def _extract_field(candidate: CandidateBrief, field_name: str) -> str:
    if field_name == "career_narrative":
        return candidate.career_narrative
    if field_name == "role_fit.justification":
        return candidate.role_fit.justification
    if field_name == "outreach_hook":
        return candidate.outreach_hook
    raise ValueError(f"Unsupported field differentiation check: {field_name}")
