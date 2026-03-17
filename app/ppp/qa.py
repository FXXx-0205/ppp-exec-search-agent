from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ValidationError

from app.ppp.enrichment import CandidateEnrichmentResult, validate_enrichment_payload
from app.ppp.models import CandidateCSVRow
from app.ppp.schema import CandidateBrief, PPPOutput, validate_output_document

QUALIFIER_TERMS = ("unverified", "verification", "estimated", "limited public visibility", "cautious", "approx")
PLACEHOLDER_TERMS = ("unknown candidate", "n/a", "tbd", "lorem ipsum")


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

    if not _justification_mentions_evidence(candidate, input_candidate):
        findings.append(
            _error(
                candidate.candidate_id,
                "justification_evidence",
                "role_fit.justification should reference employer, title, or evidence-backed experience tags.",
            )
        )

    if _mentions_specific_aum_without_qualifier(candidate.firm_aum_context):
        findings.append(
            _error(
                candidate.candidate_id,
                "firm_aum_context_confidence",
                "firm_aum_context mentions a specific AUM figure without a qualifying phrase such as estimated or to verify.",
            )
        )

    if enrichment is not None:
        if enrichment.inferred_tenure_years is None and not _contains_qualifier(combined):
            findings.append(
                _error(
                    candidate.candidate_id,
                    "tenure_confidence",
                    "tenure_years is not strongly grounded in enrichment and the narrative should signal uncertainty more clearly.",
                )
            )
        if enrichment.inferred_tenure_years is not None and abs(candidate.current_role.tenure_years - enrichment.inferred_tenure_years) > 0.25:
            findings.append(
                _error(
                    candidate.candidate_id,
                    "tenure_alignment",
                    "current_role.tenure_years deviates materially from the enrichment estimate.",
                )
            )
        if enrichment.uncertain_fields and not _contains_qualifier(combined):
            findings.append(
                _error(
                    candidate.candidate_id,
                    "uncertainty_language",
                    "Enrichment marks fields as uncertain, but the final output does not clearly signal verification limits.",
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


def _justification_mentions_evidence(candidate: CandidateBrief, input_candidate: CandidateCSVRow | None) -> bool:
    text = candidate.role_fit.justification.lower()
    if input_candidate is not None:
        if input_candidate.current_employer.lower() in text:
            return True
        title_tokens = [part.strip().lower() for part in input_candidate.current_title.replace("/", " ").split() if len(part.strip()) > 3]
        if any(token in text for token in title_tokens):
            return True
    return any(tag.lower() in text for tag in candidate.experience_tags)


def _mentions_specific_aum_without_qualifier(text: str) -> bool:
    has_amount = bool(re.search(r"(\$|aud\s*)\d+(\.\d+)?\s*[bm]", text.lower()))
    return has_amount and not _contains_qualifier(text)


def _contains_qualifier(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in QUALIFIER_TERMS)


def _sentence_count(text: str) -> int:
    return len([item for item in re.split(r"[.!?]+", text) if item.strip()])


def _error(candidate_id: str | None, check: str, message: str) -> QAFinding:
    return QAFinding(candidate_id=candidate_id, severity="error", check=check, message=message)
