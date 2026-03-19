from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from app.config import settings
from app.llm.anthropic_client import ClaudeClient
from app.ppp.enrichment import (
    CandidateEnrichmentResult,
    CandidatePublicProfileLookupInput,
    CandidatePublicProfileLookupTool,
    _is_non_distribution_title,
)
from app.ppp.models import REQUIRED_CSV_COLUMNS, CandidateCSVRow
from app.ppp.prompts import (
    build_generation_system_prompt,
    build_generation_user_prompt,
)
from app.ppp.qa import run_bundle_qa, run_candidate_qa, write_qa_report
from app.ppp.quality import validate_output_quality
from app.ppp.research import PublicResearchClient, TavilyResearchClient
from app.ppp.role_spec import load_role_spec_file
from app.ppp.schema import CandidateBrief, CandidateRunFailure, MobilitySignal, PPPOutput, PPPRunResult, RoleFit
from app.ppp.style import choose_variant, polish_join, polish_text
from app.ppp.validator import (
    describe_validation_failure,
    parse_candidate_response,
    validate_and_repair_candidate_payload,
    validate_output_payload,
)

logger = logging.getLogger(__name__)


class PPPTaskError(Exception):
    pass


def _resolve_research_mode(research_mode: str | None) -> str:
    if research_mode is not None:
        return research_mode.strip().lower()
    if settings.tavily_api_key:
        return "live"
    return settings.ppp_research_mode.strip().lower()


def run_ppp_pipeline(
    *,
    input_path: str,
    output_path: str,
    role_spec_path: str,
    model: str,
    intermediate_dir: str = "data/ppp/intermediate",
    research_fixture_path: str = "data/ppp/research_fixtures.json",
    research_mode: str | None = None,
) -> PPPRunResult:
    api_key = settings.anthropic_api_key
    if not api_key:
        raise PPPTaskError("ANTHROPIC_API_KEY is missing. Set it in your environment or .env before running the PPP task.")

    candidates = _load_candidates_csv(Path(input_path))
    role_spec = _load_role_spec(Path(role_spec_path))
    client = ClaudeClient(api_key=api_key)
    effective_research_mode = _resolve_research_mode(research_mode)
    lookup_tool = CandidatePublicProfileLookupTool(
        fixture_path=research_fixture_path,
        mode=effective_research_mode,
        research_client=_build_research_client(effective_research_mode),
    )
    candidate_inputs_by_id: dict[str, CandidateCSVRow] = {}
    enrichments_by_id: dict[str, CandidateEnrichmentResult] = {}
    failed_candidates: list[CandidateRunFailure] = []

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
        except Exception as exc:
            logger.exception("PPP candidate failed for %s at candidate_id=%s", candidate.full_name, candidate_id)
            _write_failure_artifact(
                candidate_id=candidate_id,
                candidate=candidate,
                intermediate_dir=intermediate_dir,
                stage="enrichment",
                error_message=str(exc),
                enrichment=enrichments_by_id.get(candidate_id),
            )
            failed_candidates.append(
                CandidateRunFailure(
                    candidate_id=candidate_id,
                    full_name=candidate.full_name,
                    stage="enrichment",
                    error_message=str(exc),
                    artifact_path=str(Path(intermediate_dir) / f"{candidate_id}_error.json"),
                )
            )
            continue

    enrichments_by_id = _diversify_recruiter_signals(enrichments_by_id)

    output_candidates: list[CandidateBrief] = []
    successful_candidate_inputs: dict[str, CandidateCSVRow] = {}
    successful_enrichments: dict[str, CandidateEnrichmentResult] = {}
    for idx, candidate in enumerate(candidates, start=1):
        candidate_id = f"candidate_{idx}"
        if candidate_id not in enrichments_by_id:
            continue
        enrichment = enrichments_by_id[candidate_id]
        artifact_path = lookup_tool.save_intermediate(enrichment, output_dir=intermediate_dir)
        logger.info("Saved enrichment artifact to %s", artifact_path)
        logger.info("Generating PPP briefing for %s (%s/5)", candidate.full_name, idx)
        try:
            candidate_brief = _generate_candidate_brief(
                client=client,
                candidate_id=candidate_id,
                candidate=candidate,
                enrichment=enrichment,
                role_spec=role_spec,
                model=model,
            )
        except Exception as exc:
            logger.exception("PPP generation failed for %s at candidate_id=%s", candidate.full_name, candidate_id)
            _write_failure_artifact(
                candidate_id=candidate_id,
                candidate=candidate,
                intermediate_dir=intermediate_dir,
                stage="generation",
                error_message=str(exc),
                enrichment=enrichment,
            )
            failed_candidates.append(
                CandidateRunFailure(
                    candidate_id=candidate_id,
                    full_name=candidate.full_name,
                    stage="generation",
                    error_message=str(exc),
                    artifact_path=str(Path(intermediate_dir) / f"{candidate_id}_error.json"),
                )
            )
            continue

        candidate_report = run_candidate_qa(candidate=candidate_brief, input_candidate=candidate, enrichment=enrichment)
        if not candidate_report.passed:
            error_message = "; ".join(finding.message for finding in candidate_report.findings if finding.severity == "error")
            logger.warning("PPP candidate QA failed for %s at candidate_id=%s: %s", candidate.full_name, candidate_id, error_message)
            _write_failure_artifact(
                candidate_id=candidate_id,
                candidate=candidate,
                intermediate_dir=intermediate_dir,
                stage="candidate_qa",
                error_message=error_message,
                enrichment=enrichment,
                candidate_brief=candidate_brief,
            )
            failed_candidates.append(
                CandidateRunFailure(
                    candidate_id=candidate_id,
                    full_name=candidate.full_name,
                    stage="candidate_qa",
                    error_message=error_message,
                    artifact_path=str(Path(intermediate_dir) / f"{candidate_id}_error.json"),
                )
            )
            continue

        output_candidates.append(candidate_brief)
        successful_candidate_inputs[candidate_id] = candidate
        successful_enrichments[candidate_id] = enrichment

    if not output_candidates:
        run_report_path = _write_run_report(
            failed_candidates=failed_candidates,
            warnings=["No candidate briefings passed generation and QA."],
            output_candidates=[],
            successful_enrichments={},
            output_path=output_path,
            intermediate_dir=intermediate_dir,
        )
        raise PPPTaskError(f"PPP task failed for all candidates. Review {run_report_path} for details.")

    output = validate_output_payload({"candidates": [candidate.model_dump(mode="json") for candidate in output_candidates]})
    warnings: list[str] = []
    try:
        validate_output_quality(output)
    except ValueError as exc:
        draft_output_path = _write_draft_output(output=output, intermediate_dir=intermediate_dir)
        warnings.append(f"Output quality warning: {exc}. Review {draft_output_path}.")
    qa_report = run_bundle_qa(output=output, candidates=successful_candidate_inputs, enrichments=successful_enrichments)
    qa_report_path = write_qa_report(qa_report, path=str(Path(intermediate_dir) / "qa_report.json"))
    logger.info("QA report written to %s", qa_report_path)
    if not qa_report.passed:
        warnings.append(f"Bundle QA flagged issues. Review {qa_report_path} for details.")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("PPP output written to %s", output_file)
    delivery_status: Literal["success", "partial_success"] = (
        "success" if not failed_candidates and not warnings and len(output_candidates) == len(candidates) else "partial_success"
    )
    run_report_path = _write_run_report(
        failed_candidates=failed_candidates,
        warnings=warnings,
        output_candidates=output_candidates,
        successful_enrichments=successful_enrichments,
        output_path=str(output_file),
        intermediate_dir=intermediate_dir,
        qa_report_path=str(qa_report_path),
        delivery_status=delivery_status,
    )
    return PPPRunResult(
        output=output,
        failed_candidates=failed_candidates,
        delivery_status=delivery_status,
        warnings=warnings,
        output_path=str(output_file),
        qa_report_path=str(qa_report_path),
        run_report_path=str(run_report_path),
    )


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
    try:
        return load_role_spec_file(path)
    except ValueError as exc:
        raise PPPTaskError(str(exc)) from exc


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
    stage: str,
    error_message: str,
    enrichment: CandidateEnrichmentResult | None,
    candidate_brief: CandidateBrief | None = None,
) -> Path:
    path = Path(intermediate_dir) / f"{candidate_id}_error.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "stage": stage,
                "candidate_input": candidate.model_dump(mode="json"),
                "enrichment": enrichment.model_dump(mode="json") if enrichment is not None else None,
                "error_message": error_message,
                "candidate_brief": candidate_brief.model_dump(mode="json") if candidate_brief is not None else None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _write_draft_output(*, output: PPPOutput, intermediate_dir: str) -> Path:
    path = Path(intermediate_dir) / "draft_output.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_run_report(
    *,
    failed_candidates: list[CandidateRunFailure],
    warnings: list[str],
    output_candidates: list[CandidateBrief],
    successful_enrichments: dict[str, CandidateEnrichmentResult],
    output_path: str,
    intermediate_dir: str,
    qa_report_path: str | None = None,
    delivery_status: str = "partial_success",
) -> Path:
    path = Path(intermediate_dir) / "run_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "delivery_status": delivery_status,
        "successful_candidate_count": len(output_candidates),
        "failed_candidate_count": len(failed_candidates),
        "warnings": warnings,
        "output_path": output_path,
        "qa_report_path": qa_report_path,
        "successful_candidate_ids": [candidate.candidate_id for candidate in output_candidates],
        "successful_candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "full_name": candidate.full_name,
                "identity_resolution": successful_enrichments[candidate.candidate_id].identity_resolution.status,
                "verification_posture": _verification_posture(successful_enrichments[candidate.candidate_id]),
                "inclusion_reason": _inclusion_reason(candidate, successful_enrichments[candidate.candidate_id]),
            }
            for candidate in output_candidates
            if candidate.candidate_id in successful_enrichments
        ],
        "failed_candidates": [failure.model_dump(mode="json") for failure in failed_candidates],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _verification_posture(enrichment: CandidateEnrichmentResult) -> str:
    status = enrichment.identity_resolution.status
    if status == "verified_match":
        return "verified public match"
    if status == "possible_match":
        return "possible public match with explicit caveats"
    return "unverified market-map input"


def _inclusion_reason(candidate: CandidateBrief, enrichment: CandidateEnrichmentResult) -> str:
    if candidate.role_fit.score >= 7:
        return "Included as a credible shortlist candidate based on commercially relevant lane and scope evidence."
    if candidate.role_fit.score >= 5:
        return "Included as an adjacent step-up or directionally strong possible match that still merits consultant review."
    if enrichment.identity_resolution.status == "possible_match":
        return "Included as a possible match because the commercial shape is useful even though identity still needs confirmation."
    return "Included as a market-map reference point because the remit appears commercially plausible but remains verification-sensitive."


def _diversify_recruiter_signals(enrichments_by_id: dict[str, CandidateEnrichmentResult]) -> dict[str, CandidateEnrichmentResult]:
    diversified = dict(enrichments_by_id)
    for rank, candidate_id in enumerate(sorted(diversified), start=1):
        enrichment = diversified[candidate_id]
        diversified[candidate_id] = _rebalance_gap_profile(enrichment=enrichment, rank=rank)
    return diversified


def _rebalance_gap_profile(*, enrichment: CandidateEnrichmentResult, rank: int) -> CandidateEnrichmentResult:
    signals = enrichment.recruiter_signals
    key_gaps: list[str] = []
    seen: set[str] = set()
    for gap in [*_gap_priority_pool(enrichment=enrichment, rank=rank), *signals.key_gaps]:
        normalized = _normalize_gap(gap)
        if not normalized or normalized in seen:
            continue
        if _is_template_or_low_value_gap(gap) and key_gaps:
            continue
        seen.add(normalized)
        key_gaps.append(gap)
        if len(key_gaps) == 2:
            break
    if not key_gaps:
        key_gaps = [_diversified_primary_gap(signals=signals, rank=rank)]
    screening_question = _screening_question_from_gaps(key_gaps)
    return enrichment.model_copy(
        update={
            "recruiter_signals": signals.model_copy(
                update={
                    "key_gaps": key_gaps,
                    "screening_priority_question": screening_question,
                }
            )
        }
    )


def _diversified_primary_gap(*, signals, rank: int) -> str:
    channel = signals.channel_orientation
    scope = signals.scope_signal
    seniority = signals.seniority_signal
    if channel == "wealth":
        return "whether the role is mainly wealth-led or extends into broader intermediary and institutional coverage"
    if channel == "institutional":
        return "how much direct institutional and super-fund coverage sits behind the remit"
    if channel == "wholesale":
        return "how much direct IFA and platform penetration sits behind the remit"
    if seniority == "director_level":
        return "whether the remit is mainly hands-on channel ownership or broader team leadership"
    if seniority == "head_level" and scope == "global":
        return "how hands-on the role remains versus broad strategic oversight"
    if seniority == "head_level" and scope in {"national", "anz"}:
        return "how much direct platform, IFA, and super-fund coverage sits behind the remit"
    if rank % 2 == 0:
        return "team leadership scale behind the remit"
    return "hands-on versus strategic ownership behind the remit"


def _gap_priority_pool(*, enrichment: CandidateEnrichmentResult, rank: int) -> list[str]:
    signals = enrichment.recruiter_signals
    channel = signals.channel_orientation
    scope = signals.scope_signal
    seniority = signals.seniority_signal
    pool: list[str] = []

    if channel == "institutional":
        pool.extend(
            [
                "how much direct institutional and super-fund coverage sits behind the remit",
                "product breadth across the current remit",
            ]
        )
    elif channel == "wholesale":
        pool.extend(
            [
                "how much direct IFA and platform penetration sits behind the remit",
                "team leadership scale behind the remit",
            ]
        )
    elif channel == "wealth":
        pool.extend(
            [
                "whether the role is mainly wealth-led or extends into broader intermediary and institutional coverage",
                "how much direct IFA and platform penetration sits behind the remit",
            ]
        )
    elif channel == "mixed":
        mixed_rotations = [
            [
                "whether the remit is mainly hands-on channel ownership or broader strategic leadership",
                "product breadth across the current remit",
            ],
            [
                "team leadership scale behind the remit",
                "whether the exposure is mainly ANZ, global, or local in practice",
            ],
            [
                "product breadth across the current remit",
                "whether the remit is genuinely national or concentrated in a narrower market segment",
            ],
            [
                "whether the remit is genuinely national or concentrated in a narrower market segment",
                "team leadership scale behind the remit",
            ],
            [
                "how much direct IFA and platform penetration sits behind the remit",
                "how much direct institutional and super-fund coverage sits behind the remit",
            ],
        ]
        pool.extend(mixed_rotations[(rank - 1) % len(mixed_rotations)])

    if seniority == "head_level":
        pool.append("how hands-on the role remains versus broad strategic oversight")
    elif seniority in {"director_level", "bdm_level"}:
        pool.append("team leadership scale behind the remit")

    if scope in {"global", "regional", "anz"}:
        pool.append("whether the exposure is mainly ANZ, global, or local in practice")
    elif scope == "national":
        pool.append("whether the remit is genuinely national or concentrated in a narrower market segment")

    rotated_fallbacks = [
        "team leadership scale behind the remit",
        "how hands-on the role remains versus broad strategic oversight",
        "product breadth across the current remit",
        "whether the exposure is mainly ANZ, global, or local in practice",
        "how much direct IFA and platform penetration sits behind the remit",
        "how much direct institutional and super-fund coverage sits behind the remit",
    ]
    shift = (rank - 1) % len(rotated_fallbacks)
    pool.extend(rotated_fallbacks[shift:] + rotated_fallbacks[:shift])
    return pool


def _normalize_gap(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _is_template_or_low_value_gap(text: str) -> bool:
    lowered = text.lower()
    return any(
        term in lowered
        for term in (
            "channel versus broader multi-channel coverage",
            "mobility",
            "aum",
            "start date",
            "tenure",
        )
    )


def _stabilize_candidate_brief(
    *,
    candidate_brief: CandidateBrief,
    enrichment: CandidateEnrichmentResult,
) -> CandidateBrief:
    return candidate_brief.model_copy(
        update={
            "career_narrative": polish_text(_safe_career_narrative(candidate_brief=candidate_brief, enrichment=enrichment)),
            "firm_aum_context": _safe_firm_aum_context(enrichment=enrichment),
            "mobility_signal": MobilitySignal(
                score=_safe_mobility_score(enrichment),
                rationale=polish_text(_safe_mobility_rationale(enrichment)),
            ),
            "role_fit": RoleFit(
                role=candidate_brief.role_fit.role,
                score=_safe_role_fit_score(candidate_brief=candidate_brief, enrichment=enrichment),
                justification=polish_text(_safe_role_fit_justification(candidate_brief=candidate_brief, enrichment=enrichment)),
            ),
            "outreach_hook": polish_text(_safe_outreach_hook(candidate_brief=candidate_brief, enrichment=enrichment)),
        }
    )


def _safe_career_narrative(*, candidate_brief: CandidateBrief, enrichment: CandidateEnrichmentResult) -> str:
    sentence_one = _career_opening(candidate_brief=candidate_brief, enrichment=enrichment)
    sentence_two = _career_relevance(candidate_brief=candidate_brief, enrichment=enrichment)
    sentence_three = _career_boundary(enrichment=enrichment)
    return polish_join(sentence_one, sentence_two, sentence_three)


def _safe_firm_aum_context(*, enrichment: CandidateEnrichmentResult) -> str:
    verified_numeric_claim = next(
        (
            claim
            for claim in enrichment.claims_for_output_field("firm_aum_context")
            if claim.verification_status in {"verified", "strongly_inferred"} and _contains_numeric_aum(claim.statement)
        ),
        None,
    )
    if verified_numeric_claim is not None:
        return _normalize_numeric_aum_context(verified_numeric_claim.statement, employer=enrichment.current_employer)

    statement = enrichment.firm_aum_context
    lowered = statement.lower()
    if _contains_numeric_aum(statement):
        return _normalize_numeric_aum_context(statement, employer=enrichment.current_employer)
    if any(term in lowered for term in ("unable to verify", "public evidence does not confirm", "estimated", "treated here as")):
        return _clean_firm_aum_context(statement, employer=enrichment.current_employer)
    return _fallback_firm_aum_context(enrichment.current_employer)


def _safe_mobility_score(enrichment: CandidateEnrichmentResult) -> int:
    if enrichment.identity_resolution.status == "not_verified":
        return 2 if enrichment.recruiter_signals.mandate_similarity == "unclear_fit" else 3
    tenure = enrichment.inferred_tenure_years
    if tenure is None or tenure < 0:
        return 3
    if tenure < 1.5:
        return 2
    if tenure >= 5.0:
        return 3
    return 3


def _safe_mobility_rationale(enrichment: CandidateEnrichmentResult) -> str:
    trajectory = _trajectory_signal_sentence(enrichment)
    if enrichment.identity_resolution.status == "not_verified":
        sentence_one = trajectory or "Public chronology could not be confirmed because public search did not verify an exact-match profile."
        sentence_two = "No direct public signal of move readiness is visible, so mobility should be treated as uncertain and checked in follow-up conversation."
        return polish_join(sentence_one, sentence_two)
    if enrichment.identity_resolution.status == "possible_match" and enrichment.inferred_tenure_years is None:
        sentence_one = trajectory or "Public chronology remains incomplete because search surfaced only a possible match, so current-role tenure is not yet confirmed."
        sentence_two = "No direct public signal of move readiness is visible, so mobility should be treated as uncertain and checked in follow-up conversation."
        return polish_join(sentence_one, sentence_two)
    tenure = enrichment.inferred_tenure_years
    if tenure is not None:
        sentence_one = trajectory or f"Public chronology suggests approximately {tenure:.1f} years in the current role."
    elif enrichment.tool_mode in {"fixture_backed", "fixture_fallback"}:
        sentence_one = "Fixture-backed run did not provide public chronology for the current role."
    else:
        sentence_one = "Live public-web research did not clearly establish public chronology for the current role."
    sentence_two = "No direct public signal of move readiness is visible, so mobility should be treated as uncertain and checked in follow-up conversation."
    return polish_join(sentence_one, sentence_two)


def _safe_role_fit_score(*, candidate_brief: CandidateBrief, enrichment: CandidateEnrichmentResult) -> int:
    score = candidate_brief.role_fit.score
    signals = enrichment.recruiter_signals
    if enrichment.identity_resolution.status == "not_verified":
        return min(score, 3 if signals.mandate_similarity in {"direct_match", "adjacent_match"} else 2)
    if enrichment.identity_resolution.status == "possible_match":
        cap = 6 if signals.mandate_similarity in {"direct_match", "adjacent_match"} and signals.evidence_strength in {"strong", "moderate"} else 5
        score = min(score, cap)
    if _is_non_distribution_title(enrichment.current_title.lower()):
        return min(score, 3)
    if signals.mandate_similarity == "unclear_fit":
        return min(score, 5)
    if signals.mandate_similarity == "step_up_candidate":
        return min(score, 6)
    if signals.mandate_similarity == "adjacent_match":
        return min(score, 7)
    return score


def _safe_role_fit_justification(*, candidate_brief: CandidateBrief, enrichment: CandidateEnrichmentResult) -> str:
    signals = enrichment.recruiter_signals
    sentence_one = _role_fit_strength(enrichment=enrichment)
    sentence_two = _role_fit_gap(enrichment=enrichment)
    sentence_three = f"Screening priority: {signals.screening_priority_question}"
    return polish_join(sentence_one, sentence_two, sentence_three)


def _safe_outreach_hook(*, candidate_brief: CandidateBrief, enrichment: CandidateEnrichmentResult) -> str:
    signals = enrichment.recruiter_signals
    angle = _hook_angle(signals)
    employer = enrichment.current_employer
    first_name = _first_name(candidate_brief.full_name)
    if enrichment.identity_resolution.status == "not_verified":
        bucket = [
            f"Hi {first_name}, we're mapping the market around a distribution brief and could not yet verify an exact public profile, but wanted to compare whether your remit at {employer} overlaps with {angle}.",
            f"Hi {first_name}, we're working from a partially verified market map around a distribution search and wanted to sense-check whether your work at {employer} includes {angle}.",
            f"Hi {first_name}, we have your remit at {employer} pencilled in as a possible market-map input and wanted to confirm whether it really covers {angle}.",
            f"Hi {first_name}, before we treat your profile as action-ready for a distribution search, I wanted to lightly check whether your work at {employer} includes {angle}.",
        ]
        return choose_variant(bucket, candidate_brief.full_name, employer, angle, "not_verified")
    if enrichment.identity_resolution.status == "possible_match":
        bucket = [
            f"Hi {first_name}, we're mapping a distribution brief and your public profile looks like a possible overlap with {angle} at {employer}; worth a quick sense-check?",
            f"Hi {first_name}, public search surfaced a possible match to your remit at {employer}, especially around {angle}, and I wanted to compare notes briefly.",
            f"Hi {first_name}, we're speaking with a small number of possible market matches around a distribution search and your work at {employer} appears directionally relevant on {angle}.",
            f"Hi {first_name}, your remit at {employer} looks close enough to our current distribution search to justify an exploratory check, particularly around {angle}.",
        ]
        return choose_variant(bucket, candidate_brief.full_name, employer, angle, "possible_match")
    if _is_non_distribution_title(enrichment.current_title.lower()):
        bucket = [
            f"Hi {first_name}, we're mapping adjacent profiles around a distribution search and wanted to understand how client-facing your remit at {employer} actually is, especially around {angle}.",
            f"Hi {first_name}, your background at {employer} looks more adjacent than like-for-like to a distribution mandate, but I wanted to compare notes on any investor-facing overlap, particularly {angle}.",
            f"Hi {first_name}, we're pressure-testing adjacent investment-side profiles against a distribution brief and your remit at {employer} raised a question around {angle}; open to a quick compare-and-contrast?",
        ]
        return choose_variant(bucket, candidate_brief.full_name, employer, angle, "adjacent")
    variants = {
        "direct_match": [
            f"Hi {first_name}, we're partnering with an active manager on a distribution search and your current remit at {employer} looks highly relevant, especially around {angle}; open to a brief chat?",
            f"Hi {first_name}, we're running a Head of Distribution mandate that maps well to your work at {employer}, especially around {angle}; worth connecting?",
            f"Hi {first_name}, reaching out on a senior distribution brief, and your coverage at {employer} stood out straight away, particularly {angle}.",
            f"Hi {first_name}, we're speaking with a small group for a distribution leadership role and your remit at {employer} looks very much in the mix, particularly around {angle}; open to a quick conversation?",
        ],
        "adjacent_match": [
            f"Hi {first_name}, we're working on a senior distribution search and your remit at {employer} caught my eye, particularly the overlap with {angle}; worth a quick chat?",
            f"Hi {first_name}, reaching out on a Head of Distribution brief, and your coverage at {employer} looks adjacent in a useful way, especially around {angle}.",
            f"Hi {first_name}, we're partnering on a distribution mandate and your current remit at {employer} feels relevant, particularly where it touches {angle}; open to connecting?",
            f"Hi {first_name}, one of our active distribution searches has some overlap with what you're doing at {employer}, especially {angle}; would a short intro be worthwhile?",
        ],
        "step_up_candidate": [
            f"Hi {first_name}, we're running a broader distribution leadership search and your background at {employer} suggests an interesting step-up conversation, particularly around {angle}.",
            f"Hi {first_name}, reaching out on a senior distribution mandate; your work at {employer} looks like the kind of stretch profile we should compare, especially {angle}.",
            f"Hi {first_name}, we're mapping a Head of Distribution search and your remit at {employer} looks like a credible progression conversation, particularly around {angle}.",
            f"Hi {first_name}, one of our current distribution briefs could be a genuine stretch from your role at {employer}, especially given the overlap with {angle}; open to a quick chat?",
        ],
        "unclear_fit": [
            f"Hi {first_name}, we're mapping adjacent profiles around a distribution search and wanted to compare notes on the scope of your remit at {employer}, particularly around {angle}.",
            f"Hi {first_name}, reaching out on a senior distribution brief; your background at {employer} looks directionally relevant in places, especially where it touches {angle}.",
            f"Hi {first_name}, we're speaking with a small number of adjacent profiles around a distribution mandate and your current remit at {employer} caught my eye, particularly {angle}.",
            f"Hi {first_name}, one of our active mandates has some adjacency to your remit at {employer}, especially around {angle}; worth a brief introduction?",
        ],
    }
    bucket = variants.get(signals.mandate_similarity, variants["unclear_fit"])
    return choose_variant(bucket, candidate_brief.full_name, employer, angle, signals.mandate_similarity)


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
        "institutional": "an institutional lane",
        "wholesale": "a wholesale lane",
        "wealth": "a wealth lane",
        "retail": "a retail lane",
        "mixed": "a mixed distribution lane",
        "unclear": "a broad distribution brief with no clean single-channel read",
    }
    return mapping.get(channel_orientation, "a broad distribution brief with no clean single-channel read")


def _mandate_phrase(mandate_similarity: str) -> str:
    mapping = {
        "direct_match": "close to a direct match",
        "adjacent_match": "commercially adjacent rather than like-for-like",
        "step_up_candidate": "more of a step-up than a de-risked replica",
        "unclear_fit": "only cautiously in frame",
    }
    return mapping.get(mandate_similarity, "cautiously in frame")


def _mandate_label(mandate_similarity: str) -> str:
    mapping = {
        "direct_match": "a direct option",
        "adjacent_match": "an adjacent option",
        "step_up_candidate": "a step-up option",
        "unclear_fit": "a cautious longlist option",
    }
    return mapping.get(mandate_similarity, "a cautious longlist option")


def _sell_point_phrase(enrichment_signals) -> str:
    if enrichment_signals.key_sell_points:
        if len(enrichment_signals.key_sell_points) == 1:
            return _naturalize_sell_point(enrichment_signals.key_sell_points[0])
        return f"{_naturalize_sell_point(enrichment_signals.key_sell_points[0])}, and {_naturalize_sell_point(enrichment_signals.key_sell_points[1])}"
    return "the public record points to a relevant distribution remit"


def _gap_phrase(enrichment_signals) -> str:
    if enrichment_signals.key_gaps:
        if len(enrichment_signals.key_gaps) == 1:
            return enrichment_signals.key_gaps[0]
        return f"{enrichment_signals.key_gaps[0]} and {enrichment_signals.key_gaps[1]}"
    return "the public record still leaves commercial scope unclear"


def _hook_angle(enrichment_signals) -> str:
    channel = enrichment_signals.channel_orientation
    mandate = enrichment_signals.mandate_similarity
    scope = enrichment_signals.scope_signal
    primary_gap = enrichment_signals.key_gaps[0].lower() if enrichment_signals.key_gaps else ""
    scope_prefix = _scope_label(scope)
    if mandate == "direct_match":
        if channel == "institutional":
            return _join_angle(scope_prefix, "institutional and super-fund coverage")
        if channel == "wholesale":
            return _join_angle(scope_prefix, "wholesale, IFA, and platform coverage")
        if channel == "wealth":
            return _join_angle(scope_prefix, "wealth and intermediary distribution leadership")
        if channel == "retail":
            return _join_angle(scope_prefix, "retail distribution leadership")
        if channel == "mixed":
            if "team leadership scale" in primary_gap:
                return _join_angle(scope_prefix, "distribution leadership with real team scale")
            if "product breadth" in primary_gap:
                return _join_angle(scope_prefix, "distribution leadership across a broader product set")
            if "genuinely national" in primary_gap:
                return _join_angle(scope_prefix, "genuinely national distribution leadership")
            if "ifa and platform" in primary_gap:
                return _join_angle(scope_prefix, "multi-channel distribution with platform and intermediary reach")
            return _join_angle(scope_prefix, "distribution leadership with hands-on commercial ownership")
        return _join_angle(scope_prefix, "distribution leadership with real commercial scope")
    if mandate == "adjacent_match":
        if channel == "institutional":
            return _join_angle(scope_prefix, "institutional coverage with broader channel stretch")
        if channel == "wholesale":
            return _join_angle(scope_prefix, "intermediary coverage with leadership scale")
        if channel == "wealth":
            return _join_angle(scope_prefix, "wealth distribution with scope beyond pure wealth channels")
        if channel == "mixed":
            if "genuinely national" in primary_gap:
                return _join_angle(scope_prefix, "distribution exposure with true national breadth")
            if "team leadership scale" in primary_gap:
                return _join_angle(scope_prefix, "senior distribution exposure with team scale")
            if "product breadth" in primary_gap:
                return _join_angle(scope_prefix, "senior distribution exposure across a broader product set")
            return _join_angle(scope_prefix, "senior multi-channel distribution exposure")
        if "product breadth" in primary_gap:
            return _join_angle(scope_prefix, "multi-channel distribution across a broader product set")
        return _join_angle(scope_prefix, "senior distribution exposure with broader remit stretch")
    if mandate == "step_up_candidate":
        return f"{_lane_label(channel)} coverage stepping toward broader distribution leadership"
    if "team leadership scale" in primary_gap:
        return "team-leadership scale behind a senior distribution remit"
    if "product breadth" in primary_gap:
        return "product breadth behind a senior distribution remit"
    if "institutional and super-fund" in primary_gap:
        return "institutional and super-fund coverage behind the remit"
    if "ifa and platform" in primary_gap:
        return "IFA and platform coverage behind the remit"
    return "the commercial scope behind a senior distribution remit"


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
    lowered = text.lower()
    scaled_match = re.search(
        r"(?i)(?:a\$|aud|usd|\$|£|€)?\s*\d+(?:\.\d+)?\s*(?:b|m|bn|mn|billion|million|trillion)\b",
        lowered,
    )
    large_amount_match = re.search(
        r"(?i)(?:a\$|aud|usd|\$|£|€)\s*\d{1,3}(?:,\d{3}){2,}(?:\.\d+)?\b",
        lowered,
    )
    return bool(scaled_match or large_amount_match)


def _scope_phrase(scope_signal: str) -> str:
    mapping = {
        "national": "national scope",
        "anz": "ANZ scope",
        "global": "global scope",
        "regional": "regional scope",
        "unclear": "unclear public scope",
    }
    return mapping.get(scope_signal, "unclear public scope")


def _scope_label(scope_signal: str) -> str:
    mapping = {
        "national": "national",
        "anz": "ANZ",
        "global": "global",
        "regional": "regional",
        "unclear": "",
    }
    return mapping.get(scope_signal, "")


def _trajectory_signal_sentence(enrichment: CandidateEnrichmentResult) -> str | None:
    signals = enrichment.recruiter_signals
    tenure = enrichment.inferred_tenure_years
    scope = _scope_phrase(signals.scope_signal)
    seniority = _seniority_phrase(signals.seniority_signal)
    if tenure is not None:
        variants = [
            f"Public chronology suggests approximately {tenure:.1f} years in the current role, which reads as an established {scope} remit at {seniority}.",
            f"Public chronology suggests roughly {tenure:.1f} years in the current role, so the visible remit looks more established than newly stepped into.",
            f"Public chronology points to about {tenure:.1f} years in seat, with {scope} and {seniority} already visible in the public record.",
        ]
        return choose_variant(variants, enrichment.full_name, enrichment.current_employer, "tenure")
    if enrichment.identity_resolution.status == "possible_match":
        variants = [
            f"Public chronology remains incomplete because search surfaced only a possible match, although the visible remit already reads as {scope} at {seniority}.",
            f"Public chronology remains incomplete because only a possible match surfaced, but the public record still points to {scope} and {seniority}.",
            f"Public chronology is not fully pinned down because search found only a possible match, even though the remit appears to sit at {scope} and {seniority}.",
        ]
        return choose_variant(variants, enrichment.full_name, enrichment.current_employer, "possible_trajectory")
    if enrichment.identity_resolution.status == "not_verified":
        variants = [
            f"Public chronology could not be confirmed because public search did not verify an exact-match profile, even though the task input points to {scope} and {seniority}.",
            f"Public chronology remains unverified because no exact-match profile was confirmed, despite the task input reading like {scope} at {seniority}.",
            f"Public chronology could not be established from public search, although the task input still suggests {scope} and {seniority}.",
        ]
        return choose_variant(variants, enrichment.full_name, enrichment.current_employer, "not_verified_trajectory")
    heuristics = [
        f"Public chronology is still incomplete, but the visible remit reads like {scope} at {seniority} rather than a newly expanded assignment.",
        f"Public chronology is incomplete, although the public record does suggest {scope} and {seniority} in the current remit.",
        f"Public chronology is not fully established, but the visible role signals {scope} and {seniority}.",
    ]
    return choose_variant(heuristics, enrichment.full_name, enrichment.current_employer, "heuristic_trajectory")


def _seniority_phrase(seniority_signal: str) -> str:
    mapping = {
        "head_level": "head-level seniority",
        "director_level": "director-level seniority",
        "bdm_level": "BDM-level seniority",
        "unclear": "unclear seniority from title alone",
    }
    return mapping.get(seniority_signal, "unclear seniority from title alone")


def _evidence_phrase(evidence_strength: str) -> str:
    mapping = {
        "strong": "strong",
        "moderate": "moderate",
        "thin": "thin",
    }
    return mapping.get(evidence_strength, "thin")


def _career_opening(*, candidate_brief: CandidateBrief, enrichment: CandidateEnrichmentResult) -> str:
    signals = enrichment.recruiter_signals
    scope = _scope_phrase(signals.scope_signal)
    seniority = _seniority_phrase(signals.seniority_signal)
    title = _sentence_safe_title(enrichment.current_title)
    employer = enrichment.current_employer
    if enrichment.identity_resolution.status == "not_verified":
        variants = [
            f"Task input lists {candidate_brief.full_name} as {title} at {employer}, but public search did not verify an exact-match profile.",
            f"{candidate_brief.full_name} appears in the task input as {title} at {employer}, although public search did not verify an exact-match profile.",
            f"The task input places {candidate_brief.full_name} at {employer} as {title}, but that identity could not be verified from public search.",
        ]
        return choose_variant(variants, candidate_brief.full_name, employer, title, "career_opening")
    if enrichment.identity_resolution.status == "possible_match":
        variants = [
            f"Public search surfaced a possible match for {candidate_brief.full_name} as {title} at {employer}, but the identity remains only partially confirmed across {scope} and {seniority} signals.",
            f"Public search returned a plausible match for {candidate_brief.full_name} at {employer} as {title}, though the identity is still only partially confirmed across {scope} and {seniority} signals.",
            f"{candidate_brief.full_name} looks like a possible public match for {title} at {employer}, but the identity is still only partially confirmed across {scope} and {seniority} signals.",
        ]
        return choose_variant(variants, candidate_brief.full_name, employer, title, "possible_opening")
    if signals.channel_orientation == "unclear":
        if signals.mandate_similarity == "unclear_fit":
            return (
                f"{candidate_brief.full_name} is currently {title} at {employer}. "
                f"Public evidence does not yet establish a clear distribution lane, and {scope} with {seniority} should be treated cautiously."
            )
        return f"{candidate_brief.full_name} is currently {title} at {employer}, with a broad distribution remit signalled at {scope} and {seniority}."
    if signals.seniority_signal == "head_level":
        return f"{candidate_brief.full_name} holds {_lane_phrase(signals.channel_orientation)} at {employer}, with {scope} and {seniority} signalled by the title {title}."
    if signals.scope_signal in {"anz", "global", "regional"}:
        return f"Public evidence places {candidate_brief.full_name} in {_lane_phrase(signals.channel_orientation)} at {employer}, with {scope} and {seniority} attached to the remit."
    return f"The current remit at {employer} appears weighted toward {_lane_phrase(signals.channel_orientation)}, with {seniority} and {scope} signalled by the title {title}."


def _career_relevance(*, candidate_brief: CandidateBrief, enrichment: CandidateEnrichmentResult) -> str:
    signals = enrichment.recruiter_signals
    sell_points = _sell_point_phrase(signals)
    if enrichment.identity_resolution.status == "not_verified":
        lane_scope = _join_angle(_lane_label(signals.channel_orientation), _scope_label(signals.scope_signal))
        if lane_scope:
            variants = [
                f"Even with identity still unverified, the task input points to {lane_scope} coverage, so this is commercially plausible but best treated as a tentative market map rather than a confirmed target; {sell_points}.",
                f"Commercially, the task input still reads like {lane_scope} coverage, but without identity verification this remains a tentative market map and is not yet safe to action as a confirmed target; {sell_points}.",
                f"The task input still suggests {lane_scope} exposure, which keeps the profile directionally relevant, but it should stay as a tentative market map rather than the actionable shortlist until identity is verified; {sell_points}.",
            ]
            return choose_variant(variants, candidate_brief.full_name, enrichment.current_employer, lane_scope, "not_verified_relevance")
        return f"Even with identity still unverified, the task input points to adjacent client-facing coverage, so this is commercially plausible but not yet safe to action; {sell_points}."
    if enrichment.identity_resolution.status == "possible_match":
        variants = [
            f"There is enough directional overlap to keep this as a possible match pending verification; {sell_points}, but the identity still needs confirmation.",
            f"Commercial relevance looks directionally strong enough to keep this in the search, especially as {sell_points}, although the identity still needs confirmation.",
            f"This remains a possible match pending verification because {sell_points}; the commercial shape is useful even though the identity is not fully confirmed.",
        ]
        return choose_variant(variants, candidate_brief.full_name, enrichment.current_employer, "possible_relevance")
    if signals.mandate_similarity == "direct_match":
        return f"Strong alignment with the {candidate_brief.role_fit.role} brief on visible lane and scope; {sell_points}."
    if signals.mandate_similarity == "adjacent_match":
        return f"Strong commercial relevance here: {sell_points}; not a clean like-for-like match, but clearly adjacent."
    if signals.mandate_similarity == "step_up_candidate":
        return f"More of a progression case than a replica remit; even so, {sell_points}."
    if _is_non_distribution_title(enrichment.current_title.lower()):
        return "This looks more like an adjacent investment-platform profile than a proven distribution candidate, so any relevance to the brief should be treated as tentative."
    return f"Enough substance here to justify a screening call; {sell_points}."


def _career_boundary(*, enrichment: CandidateEnrichmentResult) -> str:
    return _gap_boundary_sentence(enrichment.recruiter_signals.key_gaps)


def _role_fit_strength(*, enrichment: CandidateEnrichmentResult) -> str:
    signals = enrichment.recruiter_signals
    title = _sentence_safe_title(enrichment.current_title)
    employer = enrichment.current_employer
    sell_points = _sell_point_phrase(signals)
    lane_scope = _join_angle(_scope_label(signals.scope_signal), f"{_lane_label(signals.channel_orientation)} coverage").strip()
    if enrichment.identity_resolution.status == "not_verified":
        return f"Market map only as an unverified market input; {title} at {employer} looks commercially plausible, but public search did not confirm that it maps to an exact-match profile."
    if enrichment.identity_resolution.status == "possible_match":
        return f"Possible match pending verification; {title} at {employer} may bring relevant {lane_scope or 'distribution'} exposure, and {sell_points}, but identity confirmation still comes first."
    if signals.mandate_similarity == "direct_match":
        return f"Credible shortlist candidate; current remit at {employer} brings relevant {lane_scope or 'distribution'}, and {sell_points}."
    if signals.mandate_similarity == "adjacent_match":
        return f"Possible match pending verification; {title} at {employer} brings relevant {lane_scope or 'distribution'} exposure, and {sell_points}."
    if signals.mandate_similarity == "step_up_candidate":
        return f"Adjacent step-up conversation; {title} at {employer} brings relevant {lane_scope or 'distribution'} exposure, and {sell_points}."
    if _is_non_distribution_title(enrichment.current_title.lower()):
        return f"Market map only; {title} at {employer} points more to investment-platform exposure than to a verified client-facing commercial remit."
    return f"Possible match pending verification; {title} at {employer} leaves the distribution relevance only partially established, although {sell_points}."


def _role_fit_gap(*, enrichment: CandidateEnrichmentResult) -> str:
    return _role_gap_sentence(enrichment.recruiter_signals.key_gaps)


def _clean_firm_aum_context(statement: str, *, employer: str) -> str:
    cleaned = re.sub(r"\s+", " ", statement.strip())
    cleaned = re.sub(r"(unable to verify exact aum from public sources(?: for [^;,.]+)?)[;,:]\s*\1", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("a established", "an established")
    cleaned = cleaned.replace("market participant funds-management firm", "funds-management platform")
    cleaned = cleaned.replace("market participant", "established")
    cleaned = re.sub(r";\s*unable to verify exact aum from public sources\b", "", cleaned, flags=re.IGNORECASE)
    if cleaned.count("Unable to verify exact AUM from public sources") > 1:
        cleaned = _fallback_firm_aum_context(employer)
    if not cleaned.endswith("."):
        cleaned += "."
    return cleaned


def _normalize_numeric_aum_context(statement: str, *, employer: str) -> str:
    cleaned = _clean_firm_aum_context(statement, employer=employer)
    lowered = cleaned.lower()
    if not _contains_numeric_aum(cleaned):
        return cleaned
    if any(
        marker in lowered
        for marker in (
            "estimated",
            "subject to verification",
            "public references",
            "public-web references",
            "approximately",
            "approx",
            "treated here as",
        )
    ):
        return cleaned
    return re.sub(
        r"\.\s*$",
        " and should be treated as an estimated figure based on public references that remains subject to verification.",
        cleaned,
    )


def _fallback_firm_aum_context(employer: str) -> str:
    variants = [
        f"Public AUM remains unverified. However, {employer} is recognized as an established funds-management platform in the market.",
        f"Exact AUM is unavailable for {employer}; assessing scale based on sector footprint and firm profile.",
        f"Public filings do not confirm exact AUM for {employer}; firm profile indicates an established funds-management platform.",
        f"AUM remains unverified from public sources. {employer} operates as an established platform based on firm type and sector context.",
    ]
    index = _stable_variant_index(employer, modulo=len(variants))
    return variants[index]


def _stable_variant_index(*parts: str, modulo: int) -> int:
    joined = "|".join(part for part in parts if part)
    return sum(ord(char) for char in joined) % modulo if joined else 0


def _first_name(full_name: str) -> str:
    parts = [part.strip() for part in full_name.split() if part.strip()]
    return parts[0] if parts else "there"


def _dedupe_phrase(text: str) -> str:
    return re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE).strip()


def _join_angle(prefix: str, base: str) -> str:
    joined = f"{prefix} {base}".strip()
    return _dedupe_phrase(joined)


def _gap_boundary_sentence(gaps: list[str]) -> str:
    if not gaps:
        return "What remains unclear from public evidence is the exact commercial scope behind the remit."
    if len(gaps) == 1:
        return f"What remains unclear from public evidence is {_strip_gap_qualifier(gaps[0])}."
    return f"What remains unclear from public evidence is {_strip_gap_qualifier(gaps[0])}; {_strip_gap_qualifier(gaps[1])} also needs testing."


def _role_gap_sentence(gaps: list[str]) -> str:
    if not gaps:
        return "Key unresolved point: the exact scope behind the remit."
    if len(gaps) == 1:
        return f"Key unresolved point: {_strip_gap_qualifier(gaps[0])}."
    return f"Key unresolved point: {_strip_gap_qualifier(gaps[0])}; {_strip_gap_qualifier(gaps[1])} also remains unverified."


def _strip_gap_qualifier(gap: str) -> str:
    text = gap.strip()
    suffixes = (
        " is not verified publicly",
        " remains unclear",
        " also remains unverified",
        " still needs confirmation",
    )
    lowered = text.lower()
    for suffix in suffixes:
        if lowered.endswith(suffix):
            text = text[: -len(suffix)].strip()
            break
    text = text.replace("distribution remit", "client-facing remit")
    return text


def _naturalize_sell_point(point: str) -> str:
    mapping = {
        "current remit sits in mixed distribution leadership": "the remit already sits at broad distribution-leadership level",
        "current remit sits in retail distribution leadership": "the remit already sits in retail distribution leadership",
        "current remit sits in wealth distribution leadership": "the remit already sits in wealth distribution leadership",
        "current remit sits in institutional distribution leadership": "the remit already sits in institutional distribution leadership",
        "current remit sits in wholesale distribution leadership": "the remit already sits in wholesale distribution leadership",
        "current remit is anchored in mixed distribution": "the current role already sits across more than one distribution channel",
        "current remit is anchored in retail distribution": "the current role carries clear retail distribution exposure",
        "current remit is anchored in wealth distribution": "the current role carries clear wealth distribution exposure",
        "current remit is anchored in institutional distribution": "the current role carries clear institutional distribution exposure",
        "current remit is anchored in wholesale distribution": "the current role carries clear wholesale distribution exposure",
        "current remit is anchored in senior mixed coverage": "the current role is senior enough to sit credibly in the frame",
        "current remit is anchored in senior institutional coverage": "the current role carries clear institutional exposure at senior level",
        "current remit is anchored in senior wholesale coverage": "the current role carries clear wholesale exposure at senior level",
        "current title points to broad distribution leadership exposure": "the title points to broad distribution leadership exposure",
        "the remit already sits close to a head-of-distribution brief": "the remit already sits close to a head-of-distribution brief",
        "the profile suggests stretch potential from channel ownership into broader distribution leadership": "the profile suggests stretch potential beyond individual channel ownership",
    }
    return mapping.get(point, point)


def _screening_question_from_gaps(gaps: list[str]) -> str:
    primary_gap = gaps[0] if gaps else ""
    lowered = primary_gap.lower()
    if "institutional and super-fund" in lowered:
        return "How much direct institutional and superannuation coverage sits within the role today?"
    if "ifa and platform penetration" in lowered:
        return "How much direct IFA and platform penetration sits behind the remit today?"
    if "platform, ifa, and super-fund" in lowered:
        return "How much direct platform, IFA, and superannuation coverage sits behind the remit today?"
    if "wealth-led or extends into broader intermediary and institutional coverage" in lowered:
        return "Is the candidate's strength primarily wealth and intermediary-led, or does it extend into institutional coverage as well?"
    if "hands-on channel ownership or broader team leadership" in lowered:
        return "Has the role been mainly hands-on channel ownership, or has it already included broader team leadership?"
    if "hands-on channel ownership or broader strategic leadership" in lowered:
        return "Has the remit been mainly hands-on channel ownership, or does it already carry broader strategic leadership responsibility?"
    if "hands-on versus strategic ownership" in lowered or "broad strategic oversight" in lowered:
        return "How hands-on is the remit today versus broader strategic oversight?"
    if "team leadership scale" in lowered:
        return "What size team has this role actually led?"
    if "anz, global, or local in practice" in lowered or "anz vs global vs local exposure" in lowered:
        return "Is the remit genuinely ANZ in practice, or more global or local than the title suggests?"
    if "narrower market segment" in lowered:
        return "Is the remit genuinely national, or is it concentrated in a narrower market segment?"
    if "product breadth" in lowered:
        return "How broad is the product set behind the current remit?"
    return "What is the first commercial point that needs confirming in the role?"
