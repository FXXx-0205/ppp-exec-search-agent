from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.config import settings
from app.llm.anthropic_client import ClaudeClient
from app.ppp.enrichment import (
    CandidateEnrichmentResult,
    CandidatePublicProfileLookupInput,
    CandidatePublicProfileLookupTool,
)
from app.ppp.models import REQUIRED_CSV_COLUMNS, CandidateCSVRow
from app.ppp.prompts import (
    build_generation_system_prompt,
    build_generation_user_prompt,
)
from app.ppp.qa import run_bundle_qa, write_qa_report
from app.ppp.quality import validate_output_quality
from app.ppp.research import PublicResearchClient, TavilyResearchClient
from app.ppp.schema import CandidateBrief, MobilitySignal, PPPOutput, RoleFit
from app.ppp.validator import (
    describe_validation_failure,
    parse_candidate_response,
    validate_and_repair_candidate_payload,
    validate_output_payload,
)

logger = logging.getLogger(__name__)


class PPPTaskError(Exception):
    pass


def run_ppp_pipeline(
    *,
    input_path: str,
    output_path: str,
    role_spec_path: str,
    model: str,
    intermediate_dir: str = "data/ppp/intermediate",
    research_fixture_path: str = "data/ppp/research_fixtures.json",
    research_mode: str | None = None,
) -> PPPOutput:
    api_key = settings.anthropic_api_key
    if not api_key:
        raise PPPTaskError("ANTHROPIC_API_KEY is missing. Set it in your environment or .env before running the PPP task.")

    candidates = _load_candidates_csv(Path(input_path))
    role_spec = _load_role_spec(Path(role_spec_path))
    client = ClaudeClient(api_key=api_key)
    effective_research_mode = (research_mode or settings.ppp_research_mode).strip().lower()
    lookup_tool = CandidatePublicProfileLookupTool(
        fixture_path=research_fixture_path,
        mode=effective_research_mode,
        research_client=_build_research_client(effective_research_mode),
    )
    candidate_inputs_by_id: dict[str, CandidateCSVRow] = {}
    enrichments_by_id: dict[str, CandidateEnrichmentResult] = {}

    output_candidates: list[CandidateBrief] = []
    for idx, candidate in enumerate(candidates, start=1):
        candidate_id = f"candidate_{idx}"
        candidate_inputs_by_id[candidate_id] = candidate
        logger.info("Enriching %s (%s/5)", candidate.full_name, idx)
        try:
            enrichment = _run_enrichment_stage(
                candidate_id=candidate_id,
                candidate=candidate,
                lookup_tool=lookup_tool,
            )
            enrichments_by_id[candidate_id] = enrichment
            artifact_path = lookup_tool.save_intermediate(enrichment, output_dir=intermediate_dir)
            logger.info("Saved enrichment artifact to %s", artifact_path)
            logger.info("Generating PPP briefing for %s (%s/5)", candidate.full_name, idx)
            output_candidates.append(
                _generate_candidate_brief(
                    client=client,
                    candidate_id=candidate_id,
                    candidate=candidate,
                    enrichment=enrichment,
                    role_spec=role_spec,
                    model=model,
                )
            )
        except Exception as exc:
            logger.exception("PPP candidate failed for %s at candidate_id=%s", candidate.full_name, candidate_id)
            _write_failure_artifact(
                candidate_id=candidate_id,
                candidate=candidate,
                intermediate_dir=intermediate_dir,
                error_message=str(exc),
                enrichment=enrichments_by_id.get(candidate_id),
            )
            raise

    output = validate_output_payload({"candidates": [candidate.model_dump(mode="json") for candidate in output_candidates]})
    try:
        validate_output_quality(output)
    except ValueError as exc:
        draft_output_path = _write_draft_output(output=output, intermediate_dir=intermediate_dir)
        raise PPPTaskError(f"Output quality check failed: {exc}. Review {draft_output_path}.") from exc
    qa_report = run_bundle_qa(output=output, candidates=candidate_inputs_by_id, enrichments=enrichments_by_id)
    qa_report_path = write_qa_report(qa_report, path=str(Path(intermediate_dir) / "qa_report.json"))
    logger.info("QA report written to %s", qa_report_path)
    if not qa_report.passed:
        draft_output_path = _write_draft_output(output=output, intermediate_dir=intermediate_dir)
        raise PPPTaskError(f"Post-generation QA failed. Review {qa_report_path} and {draft_output_path} for details.")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("PPP output written to %s", output_file)
    return output


def _build_research_client(research_mode: str) -> PublicResearchClient | None:
    if research_mode == "fixture":
        return None

    provider = settings.ppp_research_provider.strip().lower()
    if provider != "tavily":
        raise PPPTaskError(f"Unsupported PPP research provider: {settings.ppp_research_provider}")

    if not settings.tavily_api_key:
        if research_mode == "live":
            raise PPPTaskError("Live PPP research mode requires TAVILY_API_KEY to be set.")
        return None

    return TavilyResearchClient(
        api_key=settings.tavily_api_key,
        timeout_seconds=settings.ppp_research_timeout_seconds,
    )


def _load_candidates_csv(path: Path) -> list[CandidateCSVRow]:
    if not path.exists():
        raise PPPTaskError(f"CSV file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise PPPTaskError("CSV format error: header row is missing.")

        missing_columns = [column for column in REQUIRED_CSV_COLUMNS if column not in reader.fieldnames]
        if missing_columns:
            raise PPPTaskError(f"CSV format error: missing required columns: {', '.join(missing_columns)}")

        rows = list(reader)

    if len(rows) != 5:
        raise PPPTaskError(f"Candidate count error: expected 5 candidates, found {len(rows)}.")

    parsed_rows: list[CandidateCSVRow] = []
    for idx, row in enumerate(rows, start=1):
        try:
            parsed_rows.append(CandidateCSVRow(**{field: row.get(field, "") for field in REQUIRED_CSV_COLUMNS}))
        except ValidationError as exc:
            raise PPPTaskError(f"Candidate row {idx} is invalid: {exc.errors()[0]['msg']}") from exc
    return parsed_rows


def _load_role_spec(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PPPTaskError(f"Role spec file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PPPTaskError(f"Role spec JSON is invalid: {exc}") from exc

    if not isinstance(data, dict):
        raise PPPTaskError("Role spec JSON must be an object.")
    return data


def _generate_candidate_brief(
    *,
    client: ClaudeClient,
    candidate_id: str,
    candidate: CandidateCSVRow,
    enrichment: CandidateEnrichmentResult,
    role_spec: dict[str, Any],
    model: str,
) -> CandidateBrief:
    system_prompt = build_generation_system_prompt()
    previous_error: str | None = None
    previous_output: str | None = None

    for attempt in range(2):
        user_prompt = build_generation_user_prompt(
            candidate_id=candidate_id,
            candidate=candidate,
            enrichment=enrichment,
            role_spec=role_spec,
            previous_error=previous_error,
            previous_output=previous_output,
        )
        raw = client.generate_text(system_prompt=system_prompt, user_prompt=user_prompt, model=model, max_tokens=1400)
        previous_output = raw

        try:
            parsed = parse_candidate_response(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            previous_error = "Return a single valid JSON object only, with no markdown fences or commentary."
            if attempt == 1:
                raise PPPTaskError(
                    f"Claude returned non-JSON output for {candidate.full_name}. Review the prompt or response handling."
                ) from exc
            continue

        try:
            candidate_brief = validate_and_repair_candidate_payload(
                parsed,
                candidate_id=candidate_id,
                full_name=candidate.full_name,
                firm_aum_context=enrichment.firm_aum_context,
                inferred_tenure_years=enrichment.inferred_tenure_years,
            )
            return _stabilize_candidate_brief(candidate_brief=candidate_brief, enrichment=enrichment)
        except ValidationError as exc:
            previous_error = describe_validation_failure(exc)
            if attempt == 1:
                raise PPPTaskError(
                    f"Claude returned invalid schema output for {candidate.full_name}: {previous_error}"
                ) from exc

    raise PPPTaskError(f"Candidate generation failed for {candidate.full_name}.")


def _run_enrichment_stage(
    *,
    candidate_id: str,
    candidate: CandidateCSVRow,
    lookup_tool: CandidatePublicProfileLookupTool,
) -> CandidateEnrichmentResult:
    tool_input = CandidatePublicProfileLookupInput.from_candidate(candidate_id=candidate_id, candidate=candidate)
    return lookup_tool.run(tool_input)


def _write_failure_artifact(
    *,
    candidate_id: str,
    candidate: CandidateCSVRow,
    intermediate_dir: str,
    error_message: str,
    enrichment: CandidateEnrichmentResult | None,
) -> None:
    path = Path(intermediate_dir) / f"{candidate_id}_error.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "candidate_input": candidate.model_dump(mode="json"),
                "enrichment": enrichment.model_dump(mode="json") if enrichment is not None else None,
                "error_message": error_message,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_draft_output(*, output: PPPOutput, intermediate_dir: str) -> Path:
    path = Path(intermediate_dir) / "draft_output.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _stabilize_candidate_brief(
    *,
    candidate_brief: CandidateBrief,
    enrichment: CandidateEnrichmentResult,
) -> CandidateBrief:
    return candidate_brief.model_copy(
        update={
            "career_narrative": _safe_career_narrative(candidate_brief=candidate_brief, enrichment=enrichment),
            "firm_aum_context": _safe_firm_aum_context(enrichment=enrichment),
            "mobility_signal": MobilitySignal(
                score=_safe_mobility_score(enrichment),
                rationale=_safe_mobility_rationale(enrichment),
            ),
            "role_fit": RoleFit(
                role=candidate_brief.role_fit.role,
                score=candidate_brief.role_fit.score,
                justification=_safe_role_fit_justification(candidate_brief=candidate_brief, enrichment=enrichment),
            ),
            "outreach_hook": _safe_outreach_hook(candidate_brief=candidate_brief, enrichment=enrichment),
        }
    )


def _safe_career_narrative(*, candidate_brief: CandidateBrief, enrichment: CandidateEnrichmentResult) -> str:
    signals = enrichment.recruiter_signals
    lane_phrase = _lane_phrase(signals.channel_orientation)
    sentence_one = f"{candidate_brief.full_name} is currently {_sentence_safe_title(enrichment.current_title)} at {enrichment.current_employer}, with public evidence pointing to {lane_phrase}."
    sentence_two = f"This profile looks {_mandate_phrase(signals.mandate_similarity)} for a {candidate_brief.role_fit.role} brief because {_sell_point_phrase(signals)}."
    sentence_three = f"Public evidence does not fully confirm the broader scope here, and {_gap_phrase(signals)} should be confirmed in conversation."
    return " ".join([sentence_one, sentence_two, sentence_three])


def _safe_firm_aum_context(*, enrichment: CandidateEnrichmentResult) -> str:
    verified_numeric_claim = next(
        (
            claim
            for claim in enrichment.claims_for_output_field("firm_aum_context")
            if claim.verification_status == "verified" and _contains_numeric_aum(claim.statement)
        ),
        None,
    )
    if verified_numeric_claim is not None:
        return verified_numeric_claim.statement

    statement = enrichment.firm_aum_context
    lowered = statement.lower()
    if _contains_numeric_aum(statement):
        return (
            f"Unable to verify exact AUM from public sources for {enrichment.current_employer}; "
            f"public evidence does not confirm exact AUM, so the platform is treated here as broadly comparable on firm type and sector context."
        )
    if any(term in lowered for term in ("unable to verify", "public evidence does not confirm", "estimated", "treated here as")):
        return statement
    return (
        f"Unable to verify exact AUM from public sources for {enrichment.current_employer}; "
        f"based on firm type and sector context, the platform appears broadly comparable to an established funds-management platform."
    )


def _safe_mobility_score(enrichment: CandidateEnrichmentResult) -> int:
    tenure = enrichment.inferred_tenure_years
    if tenure is None:
        return 3
    if tenure < 1.5:
        return 2
    if tenure >= 5.0:
        return 3
    return 3


def _safe_mobility_rationale(enrichment: CandidateEnrichmentResult) -> str:
    tenure = enrichment.inferred_tenure_years
    if tenure is not None:
        sentence_one = f"Public chronology suggests approximately {tenure:.1f} years in the current role."
    else:
        sentence_one = "Public chronology does not clearly establish current-role tenure."
    sentence_two = "No direct public signal of move readiness is visible, so mobility should therefore be treated as uncertain pending conversation."
    return " ".join([sentence_one, sentence_two])


def _safe_role_fit_justification(*, candidate_brief: CandidateBrief, enrichment: CandidateEnrichmentResult) -> str:
    signals = enrichment.recruiter_signals
    sentence_one = (
        f"Public evidence supports relevance from {_sentence_safe_title(enrichment.current_title)} at {enrichment.current_employer} because {_sell_point_phrase(signals)}."
    )
    sentence_two = f"However, direct evidence of {_gap_phrase(signals)} remains unverified."
    sentence_three = f"This should be tested early in screening conversation, starting with whether {_screening_question_topic(signals)}."
    return " ".join([sentence_one, sentence_two, sentence_three])


def _safe_outreach_hook(*, candidate_brief: CandidateBrief, enrichment: CandidateEnrichmentResult) -> str:
    signals = enrichment.recruiter_signals
    angle = _hook_angle(signals)
    return (
        f"I'm reaching out because your {_sentence_safe_title(enrichment.current_title)} background at {enrichment.current_employer} appears relevant to a {angle} within a {candidate_brief.role_fit.role} search."
    )


def _supported_remit_phrase(enrichment: CandidateEnrichmentResult) -> str:
    claim_text = " ".join(claim.statement.lower() for claim in enrichment.claims if claim.category in {"channel_experience", "experience", "current_role"})
    if "institutional" in claim_text and "wholesale" in claim_text:
        return "institutional and wholesale channel exposure"
    if "institutional" in claim_text:
        return "institutional channel exposure"
    if "wholesale" in claim_text:
        return "wholesale distribution exposure"
    if "distribution" in claim_text:
        return "distribution-facing remit"
    if "client relationship" in claim_text or "client relationships" in claim_text:
        return "client relationship coverage"
    return "relevant client and distribution coverage"


def _verification_summary_phrase(enrichment: CandidateEnrichmentResult) -> str:
    uncertain_fields = [field for field in enrichment.verification.uncertain_fields[:2] if field.strip()]
    if uncertain_fields:
        joined = ", ".join(uncertain_fields)
        return f"Public evidence remains incomplete and {joined} still require verification"
    return "Public evidence remains incomplete and additional scope details still require verification"


def _verification_gap_phrase(enrichment: CandidateEnrichmentResult) -> str:
    exact_gaps = [field.strip() for field in enrichment.verification.uncertain_fields if field.strip()]
    if exact_gaps:
        return ", ".join(exact_gaps[:3])

    fallback_gaps = [field.strip() for field in enrichment.verification.missing_fields if field.strip()]
    if fallback_gaps:
        return ", ".join(fallback_gaps[:3])

    return "the open verification items already flagged in the research package"


def _lane_phrase(channel_orientation: str) -> str:
    mapping = {
        "institutional": "an institutional distribution lane",
        "wholesale": "a wholesale distribution lane",
        "wealth": "a wealth distribution lane",
        "retail": "a retail distribution lane",
        "mixed": "a mixed distribution lane across multiple channels",
        "unclear": "an unclear channel lane from public evidence",
    }
    return mapping.get(channel_orientation, "an unclear distribution lane")


def _mandate_phrase(mandate_similarity: str) -> str:
    mapping = {
        "direct_match": "like a direct match",
        "adjacent_match": "commercially relevant on an adjacent basis",
        "step_up_candidate": "like a potential step-up option",
        "unclear_fit": "only cautiously in frame",
    }
    return mapping.get(mandate_similarity, "cautiously in frame")


def _mandate_label(mandate_similarity: str) -> str:
    mapping = {
        "direct_match": "direct match candidate",
        "adjacent_match": "adjacent match candidate",
        "step_up_candidate": "step-up candidate",
        "unclear_fit": "unclear-fit profile",
    }
    return mapping.get(mandate_similarity, "unclear-fit profile")


def _sell_point_phrase(enrichment_signals) -> str:
    if enrichment_signals.key_sell_points:
        if len(enrichment_signals.key_sell_points) == 1:
            return enrichment_signals.key_sell_points[0]
        return f"{enrichment_signals.key_sell_points[0]}, and {enrichment_signals.key_sell_points[1]}"
    return "public evidence supports a relevant distribution remit"


def _gap_phrase(enrichment_signals) -> str:
    if enrichment_signals.key_gaps:
        if len(enrichment_signals.key_gaps) == 1:
            return enrichment_signals.key_gaps[0]
        return f"{enrichment_signals.key_gaps[0]} and {enrichment_signals.key_gaps[1]}"
    return "the public record still leaves commercial scope unclear"


def _screening_question_topic(enrichment_signals) -> str:
    if enrichment_signals.key_gaps:
        gap = enrichment_signals.key_gaps[0]
        if gap.endswith("not verified"):
            return gap.removesuffix("not verified").strip()
        if gap.endswith("remains unclear"):
            return gap.removesuffix("remains unclear").strip()
        if gap.endswith("still needs confirmation"):
            return gap.removesuffix("still needs confirmation").strip()
        return gap
    return "the current public remit maps directly to the role scope"


def _hook_angle(enrichment_signals) -> str:
    channel = enrichment_signals.channel_orientation
    mandate = enrichment_signals.mandate_similarity
    if mandate == "direct_match":
        if channel == "institutional":
            return "comparable institutional distribution remit"
        if channel == "wholesale":
            return "comparable wholesale expansion remit"
        if channel == "wealth":
            return "comparable wealth distribution remit"
        if channel == "retail":
            return "comparable retail distribution remit"
        return "comparable distribution remit"
    if mandate == "adjacent_match":
        return f"an adjacent but relevant {_lane_label(channel)} lane"
    if mandate == "step_up_candidate":
        return "leadership step-up conversation"
    return f"a {_lane_label(channel)} coverage brief"


def _lane_label(channel_orientation: str) -> str:
    mapping = {
        "institutional": "institutional",
        "wholesale": "wholesale",
        "wealth": "wealth",
        "retail": "retail",
        "mixed": "multi-channel",
        "unclear": "distribution",
    }
    return mapping.get(channel_orientation, "distribution")


def _sentence_safe_title(title: str) -> str:
    return title.replace(".", "")


def _contains_numeric_aum(text: str) -> bool:
    return bool(re.search(r"(\$|aud\s*)\d+(\.\d+)?\s*[bm]", text.lower()))
