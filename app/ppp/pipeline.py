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
from app.ppp.agent_tools import build_candidate_normalization_tool, run_candidate_tool
from app.ppp.enrichment import (
    CandidateEnrichmentResult,
    CandidatePublicProfileLookupInput,
    CandidatePublicProfileLookupTool,
    NormalizedEvidence,
    _is_non_distribution_title,
    validate_normalized_evidence_payload,
)
from app.ppp.models import REQUIRED_CSV_COLUMNS, CandidateCSVRow
from app.ppp.paths import DEFAULT_PATHS
from app.ppp.prompts import (
    build_candidate_repair_system_prompt,
    build_candidate_repair_user_prompt,
    build_final_brief_system_prompt,
    build_final_brief_user_prompt,
    build_research_system_prompt,
    build_research_user_prompt,
)
from app.ppp.qa import run_bundle_qa, run_candidate_qa, write_qa_report
from app.ppp.quality import validate_output_quality
from app.ppp.research import PublicResearchClient, TavilyResearchClient
from app.ppp.role_spec import load_role_spec_file
from app.ppp.schema import (
    CandidateBrief,
    CandidateRunFailure,
    CurrentRole,
    MobilitySignal,
    PPPOutput,
    PPPRunResult,
    RoleFit,
)
from app.ppp.style import choose_variant, polish_join, polish_text
from app.ppp.validator import (
    describe_validation_failure,
    parse_candidate_response,
    parse_json_response,
    parse_normalized_evidence_response,
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
    intermediate_dir: str = str(DEFAULT_PATHS.intermediate_dir),
    research_fixture_path: str = str(DEFAULT_PATHS.research_fixtures),
    research_mode: str | None = None,
) -> PPPRunResult:
    api_key = settings.anthropic_api_key
    if not api_key:
        raise PPPTaskError("ANTHROPIC_API_KEY is missing. Set it in your environment or .env before running the PPP task.")

    candidates = _load_candidates_csv(Path(input_path))
    _reset_intermediate_artifacts(intermediate_dir=intermediate_dir)
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
    if len(output_candidates) != 5:
        run_report_path = _write_run_report(
            failed_candidates=failed_candidates,
            warnings=["PPP export blocked because the final output did not contain exactly five candidates."],
            output_candidates=output_candidates,
            successful_enrichments=successful_enrichments,
            output_path=output_path,
            intermediate_dir=intermediate_dir,
        )
        raise PPPTaskError(
            f"PPP export blocked: expected exactly 5 validated candidates, found {len(output_candidates)}. Review {run_report_path}."
        )

    output = validate_output_payload({"candidates": [candidate.model_dump(mode="json") for candidate in output_candidates]})
    _ensure_export_ready_output(output)
    warnings: list[str] = []
    try:
        validate_output_quality(output)
    except ValueError as exc:
        draft_output_path = _write_draft_output(output=output, intermediate_dir=intermediate_dir)
        raise PPPTaskError(f"PPP export blocked by output quality validation: {exc}. Review {draft_output_path}.") from exc
    qa_report = run_bundle_qa(output=output, candidates=successful_candidate_inputs, enrichments=successful_enrichments)
    qa_report_path = write_qa_report(qa_report, path=str(Path(intermediate_dir) / "qa_report.json"))
    logger.info("QA report written to %s", qa_report_path)
    if not qa_report.passed:
        raise PPPTaskError(f"PPP export blocked by bundle QA. Review {qa_report_path} for details.")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    _ensure_export_ready_output(output)
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


def _ensure_export_ready_output(output: PPPOutput) -> None:
    if len(output.candidates) != 5:
        raise PPPTaskError(f"PPP schema validation failed before export: expected exactly 5 candidates, found {len(output.candidates)}.")
    for candidate in output.candidates:
        if not isinstance(candidate.current_role.tenure_years, (int, float)):
            raise PPPTaskError(f"PPP schema validation failed before export: {candidate.candidate_id}.current_role.tenure_years must be numeric.")


def _reset_intermediate_artifacts(*, intermediate_dir: str) -> None:
    base = Path(intermediate_dir)
    if not base.exists():
        return
    for pattern in (
        "candidate_*_enriched.json",
        "candidate_*_error.json",
        "qa_report.json",
        "run_report.json",
        "draft_output.json",
    ):
        for path in base.glob(pattern):
            try:
                path.unlink()
            except OSError:
                logger.warning("Unable to remove stale intermediate artifact: %s", path)


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
    # Bug note: the previous implementation mixed tool use, final brief generation, and repair retries
    # inside one giant control packet. That let orchestration metadata leak back into the model output,
    # so validation sometimes received a prompt/control object instead of a CandidateBrief.
    normalized_evidence = _generate_normalized_evidence(
        client=client,
        candidate_id=candidate_id,
        candidate=candidate,
        enrichment=enrichment,
        role_spec=role_spec,
        model=model,
    )

    logger.info("Entering final brief generation phase for %s", candidate.full_name)
    raw = client.generate_text(
        system_prompt=build_final_brief_system_prompt(),
        user_prompt=build_final_brief_user_prompt(
            candidate_id=candidate_id,
            candidate=candidate,
            normalized_evidence=normalized_evidence,
            enrichment=enrichment,
            role_spec=role_spec,
        ),
        model=model,
        max_tokens=900,
        extra={"allow_fallback": False},
    )

    try:
        parsed = parse_candidate_response(raw)
        candidate_brief = validate_and_repair_candidate_payload(
            parsed,
            candidate_id=candidate_id,
            full_name=candidate.full_name,
            firm_aum_context=enrichment.firm_aum_context,
            inferred_tenure_years=normalized_evidence.tenure_years,
        )
        return _stabilize_candidate_brief(candidate_brief=candidate_brief, enrichment=enrichment)
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        if "control packet" in str(exc).lower():
            logger.warning("Detected control-packet output instead of CandidateBrief for %s", candidate.full_name)
        logger.info("Entering repair phase for %s", candidate.full_name)
        repaired = _repair_candidate_brief(
            client=client,
            candidate_id=candidate_id,
            candidate=candidate,
            invalid_raw_output=raw,
            validation_error=str(exc),
            enrichment=enrichment,
            normalized_evidence=normalized_evidence,
            model=model,
        )
        return _stabilize_candidate_brief(candidate_brief=repaired, enrichment=enrichment)


def _generate_normalized_evidence(
    *,
    client: ClaudeClient,
    candidate_id: str,
    candidate: CandidateCSVRow,
    enrichment: CandidateEnrichmentResult,
    role_spec: dict[str, Any],
    model: str,
) -> NormalizedEvidence:
    logger.info("Entering research phase for %s", candidate.full_name)
    raw = client.run_tool_phase_once(
        system_prompt=build_research_system_prompt(),
        user_prompt=build_research_user_prompt(
            candidate_id=candidate_id,
            candidate=candidate,
            enrichment=enrichment,
            role_spec=role_spec,
        ),
        model=model,
        max_tokens=550,
        extra_payload={"allow_fallback": False},
        tool_definitions=[build_candidate_normalization_tool()],
        tool_choice={"type": "any"},
        tool_runner=lambda tool_name, tool_input: run_candidate_tool(
            tool_name,
            tool_input,
            enrichment=enrichment,
            role_spec=role_spec,
        ),
    )
    try:
        parsed = parse_normalized_evidence_response(raw)
        return validate_normalized_evidence_payload(parsed)
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        raise PPPTaskError(f"Research phase failed for {candidate.full_name}: {exc}") from exc


def _repair_candidate_brief(
    *,
    client: ClaudeClient,
    candidate_id: str,
    candidate: CandidateCSVRow,
    invalid_raw_output: str,
    validation_error: str,
    enrichment: CandidateEnrichmentResult,
    normalized_evidence: NormalizedEvidence,
    model: str,
) -> CandidateBrief:
    try:
        invalid_payload = parse_json_response(invalid_raw_output)
    except json.JSONDecodeError:
        invalid_payload = {"raw_output": invalid_raw_output}

    raw = client.generate_text(
        system_prompt=build_candidate_repair_system_prompt(),
        user_prompt=build_candidate_repair_user_prompt(
            invalid_payload=invalid_payload if isinstance(invalid_payload, dict) else {"raw_output": invalid_raw_output},
            validation_error=validation_error,
            candidate_id=candidate_id,
            full_name=candidate.full_name,
            firm_aum_context=enrichment.firm_aum_context,
            tenure_years=normalized_evidence.tenure_years,
        ),
        model=model,
        max_tokens=700,
        extra={"allow_fallback": False},
    )
    try:
        parsed = parse_candidate_response(raw)
        return validate_and_repair_candidate_payload(
            parsed,
            candidate_id=candidate_id,
            full_name=candidate.full_name,
            firm_aum_context=enrichment.firm_aum_context,
            inferred_tenure_years=normalized_evidence.tenure_years,
        )
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        if "control packet" in str(exc).lower():
            logger.warning("Detected control-packet output during repair for %s", candidate.full_name)
        raise PPPTaskError(f"Claude repair failed for {candidate.full_name}: {exc}") from exc


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
            "current_role": CurrentRole(
                title=candidate_brief.current_role.title,
                employer=candidate_brief.current_role.employer,
                tenure_years=enrichment.output_tenure_years,
            ),
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
    if _contains_numeric_aum(statement):
        return _normalize_numeric_aum_context(statement, employer=enrichment.current_employer)
    return _qualitative_firm_context(enrichment=enrichment, statement=statement)


def _safe_mobility_score(enrichment: CandidateEnrichmentResult) -> int:
    match_state = enrichment.normalized_evidence().match_confidence_state
    if match_state == "no_reliable_match":
        return 2
    tenure = enrichment.inferred_tenure_years
    if tenure is None or tenure < 0:
        return 3
    if tenure < 1.5:
        return 2
    if tenure >= 4.0 and match_state in {"verified_match", "likely_match"}:
        return 4
    if tenure >= 5.0:
        return 3
    return 3


def _safe_mobility_rationale(enrichment: CandidateEnrichmentResult) -> str:
    match_state = enrichment.normalized_evidence().match_confidence_state
    trajectory = _trajectory_signal_sentence(enrichment)
    # Important trust-calibration boundary: once the pipeline decides a current-profile
    # match is too weak to trust for precise tenure output, mobility copy must not reintroduce
    # that precision via a chronology sentence. Keep chronology, tenure precision, and move
    # readiness wording aligned in the final exported artifact.
    if match_state == "no_reliable_match":
        sentence_one = trajectory or "Public chronology could not be validated against a reliable current-profile match."
        sentence_two = _mobility_follow_up_sentence(match_state=match_state)
        return polish_join(sentence_one, sentence_two)
    if match_state in {"likely_match", "partial_match"} and enrichment.inferred_tenure_years is None:
        sentence_one = trajectory or "Public chronology is only partially established, so current-role tenure should be treated as directional rather than locked."
        sentence_two = _mobility_follow_up_sentence(match_state=match_state)
        return polish_join(sentence_one, sentence_two)
    tenure = enrichment.inferred_tenure_years
    if tenure is not None:
        sentence_one = trajectory or f"Public chronology suggests approximately {tenure:.1f} years in the current role."
    elif enrichment.tool_mode in {"fixture_backed", "fixture_fallback"}:
        sentence_one = "Fixture-backed run did not provide public chronology for the current role."
    else:
        sentence_one = "Live public-web research did not clearly establish public chronology for the current role."
    sentence_two = _mobility_follow_up_sentence(match_state=match_state)
    return polish_join(sentence_one, sentence_two)


def _safe_role_fit_score(*, candidate_brief: CandidateBrief, enrichment: CandidateEnrichmentResult) -> int:
    score = candidate_brief.role_fit.score
    signals = enrichment.recruiter_signals
    match_state = enrichment.normalized_evidence().match_confidence_state
    bucket = _priority_bucket(enrichment=enrichment)
    if bucket == "mapping_lead":
        if match_state == "no_reliable_match":
            return 2
        if signals.mandate_similarity == "step_up_candidate":
            return 4
        return 3
    if bucket == "credible_adjacent_screen":
        if signals.seniority_signal == "head_level":
            score = min(max(score, 5), 6)
        elif signals.seniority_signal == "director_level":
            score = min(max(score, 4), 5)
        else:
            score = min(max(score, 4), 4)
        if signals.channel_orientation == "retail":
            score = min(score, 5)
        return score

    # Shortlist-style targets should separate clearly from adjacent screens even when
    # identity is still "likely" rather than fully verified.
    if match_state == "verified_match" and signals.mandate_similarity == "direct_match":
        return 8
    if signals.channel_orientation == "wholesale":
        return 8 if match_state == "verified_match" else 7
    return 7


def _safe_role_fit_justification(*, candidate_brief: CandidateBrief, enrichment: CandidateEnrichmentResult) -> str:
    signals = enrichment.recruiter_signals
    sentence_one = _soften_role_fit_chain(_role_fit_strength(enrichment=enrichment))
    sentence_two = _role_fit_gap(enrichment=enrichment)
    screening_question = signals.screening_priority_question.strip()
    variants = [
        f"The screening call should focus first on this point: {screening_question}",
        f"The quickest diligence question is this: {screening_question}",
        f"That is the first issue to pressure-test on a call: {screening_question}",
    ]
    sentence_three = choose_variant(variants, enrichment.full_name, enrichment.current_employer, screening_question, "screening_sentence")
    return polish_join(sentence_one, sentence_two, sentence_three)


def _safe_outreach_hook(*, candidate_brief: CandidateBrief, enrichment: CandidateEnrichmentResult) -> str:
    signals = enrichment.recruiter_signals
    angle = _hook_angle(signals)
    employer = enrichment.current_employer
    first_name = _first_name(candidate_brief.full_name)
    match_state = enrichment.normalized_evidence().match_confidence_state
    bucket = _priority_bucket(enrichment=enrichment)
    if match_state == "no_reliable_match":
        bucket = [
            f"Hi {first_name}, we're mapping a distribution brief and wanted to sense-check whether your remit at {employer} genuinely covers {angle}.",
            f"Hi {first_name}, we're keeping a market map around a distribution search and wanted to confirm whether your role at {employer} really includes {angle}.",
            f"Hi {first_name}, one of our distribution briefs has some adjacency to your remit at {employer}, and I wanted to check how much it really reaches into {angle}.",
        ]
        return choose_variant(bucket, candidate_brief.full_name, employer, angle, "not_verified")
    if bucket == "strong_shortlist":
        shortlist_bucket = [
            f"Hi {first_name}, we're already prioritising a small shortlist for a senior distribution mandate, and your remit at {employer} looks firmly in range, especially around {angle}; open to a brief conversation?",
            f"Hi {first_name}, we're moving quickly on a Head of Distribution search and your current remit at {employer} looks close enough to warrant an early call, particularly around {angle}.",
            f"Hi {first_name}, your work at {employer} reads like one of the more relevant profiles on our distribution shortlist, especially where it touches {angle}; would a quick intro be worthwhile?",
        ]
        return choose_variant(shortlist_bucket, candidate_brief.full_name, employer, angle, "strong_shortlist")
    if bucket == "credible_adjacent_screen":
        if signals.mandate_similarity == "step_up_candidate":
            step_up_bucket = [
                f"Hi {first_name}, we're mapping a senior distribution mandate and your remit at {employer} looks like a genuine step-up conversation, especially around {angle}.",
                f"Hi {first_name}, your work at {employer} is not a straight replica of our brief, but it does look like the kind of stretch profile worth testing, particularly around {angle}.",
                f"Hi {first_name}, we're speaking with a few progression-style candidates for a broader distribution brief, and your remit at {employer} stands out as a plausible step-up conversation, especially around {angle}.",
            ]
            return choose_variant(step_up_bucket, candidate_brief.full_name, employer, angle, "step_up_bucket")
        adjacent_bucket = [
            f"Hi {first_name}, we're screening a handful of adjacent distribution profiles and your remit at {employer} looks worth pressure-testing, especially around {angle}.",
            f"Hi {first_name}, your coverage at {employer} is not a straight title match to our brief, but there is enough overlap around {angle} to justify a quick screening conversation.",
            f"Hi {first_name}, we're speaking with a few adjacent candidates for a senior distribution role, and your remit at {employer} looks commercially relevant, especially around {angle}; open to a brief chat?",
        ]
        return choose_variant(adjacent_bucket, candidate_brief.full_name, employer, angle, "credible_adjacent_screen")
    if match_state in {"likely_match", "partial_match"}:
        bucket = [
            f"Hi {first_name}, we're running a distribution search and your remit at {employer} looks close enough to warrant a screening chat, especially around {angle}.",
            f"Hi {first_name}, your current remit at {employer} looks relevant to a live distribution brief, particularly where it touches {angle}; open to a quick conversation?",
            f"Hi {first_name}, we're speaking with a short list of relevant distribution profiles and your work at {employer} stood out on {angle}; would a brief chat be worthwhile?",
            f"Hi {first_name}, one of our active distribution mandates overlaps with your remit at {employer}, especially around {angle}; keen to compare notes?",
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


def _uncertain_scope_phrase(scope_signal: str) -> str:
    mapping = {
        "national": "reported national-scope cues",
        "anz": "reported ANZ-scope cues",
        "global": "reported global-scope cues",
        "regional": "reported regional-scope cues",
        "unclear": "limited public scope cues",
    }
    return mapping.get(scope_signal, "limited public scope cues")


def _trajectory_signal_sentence(enrichment: CandidateEnrichmentResult) -> str | None:
    signals = enrichment.recruiter_signals
    tenure = enrichment.inferred_tenure_years
    scope = _scope_phrase(signals.scope_signal)
    seniority = _seniority_phrase(signals.seniority_signal)
    match_state = enrichment.normalized_evidence().match_confidence_state
    if match_state == "no_reliable_match":
        variants = [
            f"Public chronology remains unreliable because the current-profile match is not trusted, even though the task input still suggests {_uncertain_scope_phrase(signals.scope_signal)} and {seniority}.",
            f"Chronology should be treated cautiously because the underlying current-profile match is still unreliable, despite the task input reading like {_uncertain_scope_phrase(signals.scope_signal)} at {seniority}.",
            f"The visible chronology is not strong enough to trust against the current-profile match, although the task input still points toward {_uncertain_scope_phrase(signals.scope_signal)} and {seniority}.",
        ]
        return choose_variant(variants, enrichment.full_name, enrichment.current_employer, "not_verified_trajectory")
    if match_state == "partial_match":
        variants = [
            f"Public chronology remains incomplete because search surfaced only a possible match, although the visible remit still shows {_uncertain_scope_phrase(signals.scope_signal)} and {seniority}.",
            f"Public chronology remains incomplete because only a possible match surfaced, but the public record still points to {_uncertain_scope_phrase(signals.scope_signal)} and {seniority}.",
            f"Public chronology is not fully pinned down because search found only a possible match, even though the remit appears to sit within {_uncertain_scope_phrase(signals.scope_signal)} and {seniority}.",
        ]
        return choose_variant(variants, enrichment.full_name, enrichment.current_employer, "possible_trajectory")
    if tenure is not None:
        variants = [
            f"Public chronology suggests approximately {tenure:.1f} years in the current role, which reads as an established {scope} remit at {seniority}.",
            f"Public chronology suggests roughly {tenure:.1f} years in the current role, so the visible remit looks more established than newly stepped into.",
            f"Public chronology points to about {tenure:.1f} years in seat, with {scope} and {seniority} already visible in the public record.",
        ]
        return choose_variant(variants, enrichment.full_name, enrichment.current_employer, "tenure")
    if match_state == "likely_match":
        variants = [
            f"Chronology is not fully pinned down, but the visible profile still reads as {scope} and {seniority} rather than a speculative match.",
            f"Public chronology remains directional rather than exact, although the current remit still reads as {scope} and {seniority}.",
        ]
        return choose_variant(variants, enrichment.full_name, enrichment.current_employer, "likely_trajectory")
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
    match_state = enrichment.normalized_evidence().match_confidence_state
    if match_state == "no_reliable_match":
        variants = [
            f"Task input lists {candidate_brief.full_name} as {title} at {employer}, but the current-profile match remains unreliable in public evidence.",
            f"{candidate_brief.full_name} appears in the task input as {title} at {employer}, although public evidence does not yet give a reliable current-profile match.",
            f"The task input places {candidate_brief.full_name} at {employer} as {title}, but that current-profile link is still not reliable enough to treat as verified.",
        ]
        return choose_variant(variants, candidate_brief.full_name, employer, title, "career_opening")
    if match_state == "partial_match":
        variants = [
            f"Public search surfaced a partial match for {candidate_brief.full_name} as {title} at {employer}, with {_uncertain_scope_phrase(signals.scope_signal)} and {seniority} visible but not fully locked.",
            f"Public search returned a plausible partial match for {candidate_brief.full_name} at {employer} as {title}, though the current-profile link is still only partially confirmed.",
            f"{candidate_brief.full_name} looks like a partial public match for {title} at {employer}, with enough remit evidence to screen but not enough to treat as fully verified.",
        ]
        return choose_variant(variants, candidate_brief.full_name, employer, title, "possible_opening")
    if match_state == "likely_match":
        variants = [
            f"Public evidence most likely places {candidate_brief.full_name} in the {title} remit at {employer}, with {_uncertain_scope_phrase(signals.scope_signal)} and {seniority} already visible.",
            f"{candidate_brief.full_name} reads as a likely current {title} at {employer}, with public signals pointing to {_uncertain_scope_phrase(signals.scope_signal)} and {seniority}.",
        ]
        return choose_variant(variants, candidate_brief.full_name, employer, title, "likely_opening")
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
    match_state = enrichment.normalized_evidence().match_confidence_state
    bucket = _priority_bucket(enrichment=enrichment)
    if match_state == "no_reliable_match":
        lane_scope = _join_angle(_lane_label(signals.channel_orientation), _scope_label(signals.scope_signal))
        if lane_scope:
            variants = [
                f"The lane still looks commercially relevant on the face of the input, especially around {lane_scope}, but this should stay as a tentative market map entry until the current-profile match is confirmed; {sell_points}.",
                f"There is enough lane overlap to keep this as a tentative market map entry, particularly around {lane_scope}, but not enough profile certainty yet to treat it as an action-ready target; {sell_points}.",
            ]
            return choose_variant(variants, candidate_brief.full_name, enrichment.current_employer, lane_scope, "not_verified_relevance")
        return f"Even with the current-profile match still unreliable, the remit is commercially adjacent enough to keep as a tentative market map entry; {sell_points}."
    if match_state == "partial_match":
        variants = [
            f"There is enough here to justify a screening call because {sell_points}, even though the profile should still be treated as partially matched rather than fully verified.",
            f"Commercial relevance is strong enough to keep this in the active screen, especially as {sell_points}, with profile certainty still to be tightened.",
            f"This is worth screening rather than shelving because {sell_points}; the remit is useful even though the profile match is only partial.",
        ]
        return choose_variant(variants, candidate_brief.full_name, enrichment.current_employer, "possible_relevance")
    if bucket == "strong_shortlist":
        variants = [
            f"This should sit in the early-call shortlist because {sell_points}; the diligence point is remit depth rather than basic relevance.",
            f"This reads more like a first-wave call than a mapping name because {sell_points}; the remaining work is to test scope depth, not lane fit.",
            f"This looks actionable for shortlist discussion because {sell_points}; the real question is how deep the remit runs rather than whether the lane is relevant.",
        ]
        return choose_variant(variants, candidate_brief.full_name, enrichment.current_employer, "strong_shortlist_relevance")
    if bucket == "credible_adjacent_screen":
        variants = [
            f"This is better treated as an adjacent screen than a first-wave shortlist name because {sell_points}; the overlap is real, but the transferability still needs pressure-testing.",
            f"This belongs in the adjacent-screen bucket rather than the first shortlist cut because {sell_points}; there is enough overlap to test, but not enough to assume transferability.",
            f"This looks directionally relevant enough for a screening call because {sell_points}, even if it still sits a step away from a clean shortlist profile.",
        ]
        return choose_variant(variants, candidate_brief.full_name, enrichment.current_employer, "adjacent_screen_relevance")
    if match_state == "likely_match":
        return f"This looks worth calling rather than simply mapping because {sell_points}, with the remaining uncertainty sitting more around scope detail than identity."
    if signals.mandate_similarity == "direct_match":
        return f"Visible lane and scope line up well with the {candidate_brief.role_fit.role} brief; {sell_points}."
    if signals.mandate_similarity == "adjacent_match":
        return f"This is an adjacent but credible screen because {sell_points}; not like-for-like, but close enough to test quickly."
    if signals.mandate_similarity == "step_up_candidate":
        return f"This is more of a progression case than a replica remit, but still worth testing because {sell_points}."
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
    match_state = enrichment.normalized_evidence().match_confidence_state
    bucket = _priority_bucket(enrichment=enrichment)
    if match_state == "no_reliable_match":
        return f"Lower-priority mapping lead; the task input suggests commercially relevant {lane_scope or 'distribution'} exposure at {employer}, but the current-profile match is still unreliable."
    if bucket == "strong_shortlist":
        return f"Shortlist-style target; {title} at {employer} brings one of the cleaner overlaps with the brief, especially around {lane_scope or 'distribution'}, and {sell_points}."
    if bucket == "credible_adjacent_screen":
        return f"Adjacent screen rather than first-wave shortlist; {title} at {employer} brings useful overlap with the brief, especially around {lane_scope or 'distribution'}, and {sell_points}."
    if match_state == "partial_match":
        return f"Worth a screening call; {title} at {employer} may bring relevant {lane_scope or 'distribution'} exposure, and {sell_points}, even though the profile match is only partial."
    if match_state == "likely_match":
        return f"Screen-worthy likely match; {title} at {employer} appears to bring relevant {lane_scope or 'distribution'} exposure, and {sell_points}."
    if signals.mandate_similarity == "direct_match":
        return f"Credible shortlist candidate; current remit at {employer} brings relevant {lane_scope or 'distribution'}, and {sell_points}."
    if signals.mandate_similarity == "adjacent_match":
        return f"Adjacent but credible target; {title} at {employer} brings relevant {lane_scope or 'distribution'} exposure, and {sell_points}."
    if signals.mandate_similarity == "step_up_candidate":
        return f"Adjacent step-up conversation; {title} at {employer} brings relevant {lane_scope or 'distribution'} exposure, and {sell_points}."
    if _is_non_distribution_title(enrichment.current_title.lower()):
        return f"Market map only; {title} at {employer} points more to investment-platform exposure than to a verified client-facing commercial remit."
    return f"Possible match pending verification; {title} at {employer} leaves the distribution relevance only partially established, although {sell_points}."


def _priority_bucket(*, enrichment: CandidateEnrichmentResult) -> str:
    # Internal recruiter-priority bucket used to separate shortlist, adjacent screen,
    # and mapping profiles without changing the PPP output schema.
    signals = enrichment.recruiter_signals
    match_state = enrichment.normalized_evidence().match_confidence_state
    title = enrichment.current_title.lower()

    if match_state == "no_reliable_match" or signals.mandate_similarity == "unclear_fit":
        return "mapping_lead"
    if _is_non_distribution_title(title):
        return "mapping_lead"

    has_head_distribution_title = "head of distribution" in title
    head_like_scope = signals.seniority_signal == "head_level" and signals.scope_signal in {"national", "anz", "regional", "global"}
    channel_close = signals.channel_orientation in {"wholesale", "institutional", "mixed"}

    if (
        signals.mandate_similarity == "direct_match"
        or (has_head_distribution_title and head_like_scope and match_state in {"verified_match", "likely_match", "partial_match"})
        or (signals.channel_orientation == "wholesale" and head_like_scope and signals.mandate_similarity == "adjacent_match")
    ):
        return "strong_shortlist"

    if signals.mandate_similarity in {"adjacent_match", "step_up_candidate"}:
        return "credible_adjacent_screen"

    return "mapping_lead"


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


def _qualitative_firm_context(*, enrichment: CandidateEnrichmentResult, statement: str) -> str:
    employer = enrichment.current_employer
    lowered = statement.lower()
    firm_descriptor = _firm_descriptor_from_context(employer=employer, statement=statement, channel_orientation=enrichment.recruiter_signals.channel_orientation)
    market_descriptor = _firm_market_context_phrase(lowered)
    channel_descriptor = _firm_channel_context_phrase(enrichment)
    uncertainty_descriptor = choose_variant(
        [
            "Based on firm type and sector context, exact AUM is unavailable here, so the better recruiter read is firm type, market position, and channel context.",
            "Exact AUM is unavailable here; the useful takeaway is firm profile, market position, and channel mix rather than a precise figure.",
            "Firm profile indicates useful market context here even though exact AUM is unavailable, so the note stays qualitative.",
        ],
        employer,
        firm_descriptor,
        "firm_context_uncertainty",
    )

    openings = [
        f"{employer} looks like {firm_descriptor}",
        f"{employer} reads as {firm_descriptor}",
        f"{employer} appears to be {firm_descriptor}",
    ]
    parts = [choose_variant(openings, employer, firm_descriptor, "firm_context_opening")]
    if market_descriptor:
        parts.append(market_descriptor)
    if channel_descriptor:
        parts.append(channel_descriptor)
    parts.append(uncertainty_descriptor)
    return polish_join(*parts)


def _firm_descriptor_from_context(*, employer: str, statement: str, channel_orientation: str) -> str:
    lowered = statement.lower()
    if "global asset manager" in lowered:
        return "a large global asset manager"
    if "wealth and asset-management platform" in lowered:
        return "an established Australian wealth and asset-management platform"
    if "active manager" in lowered:
        return "an established Australian active manager"
    if "financial-services platform" in lowered:
        return "an established Australian financial-services platform"
    if "asset-management brand" in lowered:
        return "a visible Australian asset-management brand"
    if "funds-management platform" in lowered:
        return "an established diversified funds-management platform"
    if channel_orientation == "wholesale":
        return "an established funds-management platform with intermediary distribution relevance"
    if channel_orientation == "institutional":
        return "an established asset manager with institutional distribution relevance"
    return "an established diversified funds-management platform"


def _firm_market_context_phrase(lowered_statement: str) -> str:
    if "retirement-income" in lowered_statement or "annuities" in lowered_statement:
        return "with a visible retirement-income and specialist product footprint."
    if "institutional credibility" in lowered_statement:
        return "with obvious institutional credibility in market."
    if "visible market scale" in lowered_statement:
        return "with visible market scale in Australian funds management."
    if "market footprint" in lowered_statement:
        return "with a clear market footprint."
    if "market presence" in lowered_statement:
        return "with a recognizable market presence."
    if "distribution visibility" in lowered_statement:
        return "with clear distribution visibility."
    return "with identifiable market relevance in funds management."


def _firm_channel_context_phrase(enrichment: CandidateEnrichmentResult) -> str:
    channel = enrichment.recruiter_signals.channel_orientation
    if channel == "wholesale":
        variants = [
            "The role sits in wholesale and intermediary distribution rather than a pure institutional-only lane.",
            "The visible channel mix is intermediary and wholesale-led rather than institutional-only.",
            "The remit is closer to intermediary and wholesale distribution than to a pure institutional coverage role.",
        ]
        return choose_variant(variants, enrichment.full_name, enrichment.current_employer, "firm_channel_wholesale")
    if channel == "institutional":
        variants = [
            "The remit is closer to institutional coverage and allocator-facing distribution than to retail-only sales.",
            "The visible lane is institutional and allocator-facing rather than adviser-led retail sales.",
            "The role context is more institutional and client-allocation facing than retail-only distribution.",
        ]
        return choose_variant(variants, enrichment.full_name, enrichment.current_employer, "firm_channel_institutional")
    if channel == "wealth":
        variants = [
            "The visible channel context is wealth and adviser-led rather than a pure institutional remit.",
            "The channel mix reads more wealth and intermediary-led than institutional-only.",
            "The visible remit sits closer to adviser and wealth distribution than to a pure institutional lane.",
        ]
        return choose_variant(variants, enrichment.full_name, enrichment.current_employer, "firm_channel_wealth")
    if channel == "mixed":
        variants = [
            "The visible channel context spans multiple distribution lanes, which is useful for a broad head-of-distribution mandate.",
            "The remit appears to cut across more than one distribution lane, which matters for a broad commercial brief.",
            "The visible channel mix is broader than a single-lane sales remit, which is relevant to a head-of-distribution search.",
        ]
        return choose_variant(variants, enrichment.full_name, enrichment.current_employer, "firm_channel_mixed")
    return ""


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
        gap = _strip_gap_qualifier(gaps[0])
        variants = [
            f"The main point to test on a first call is {gap}, which still needs confirmation.",
            f"The first diligence point is {gap}, which should be confirmed in conversation.",
            f"The initial screening focus should be {gap}, because that still needs confirmation.",
        ]
        return choose_variant(variants, gap, "single_gap_boundary")
    primary_gap = _strip_gap_qualifier(gaps[0])
    secondary_gap = _strip_gap_qualifier(gaps[1])
    variants = [
        f"The main point to test on a first call is {primary_gap}; {secondary_gap} also still needs confirmation.",
        f"The first diligence point is {primary_gap}; {secondary_gap} should be confirmed in conversation straight after that.",
        f"The initial screening focus should be {primary_gap}, with {secondary_gap} as the next point that still needs confirmation.",
    ]
    return choose_variant(variants, primary_gap, secondary_gap, "multi_gap_boundary")


def _role_gap_sentence(gaps: list[str]) -> str:
    if not gaps:
        return "Key unresolved point: the exact scope behind the remit."
    if len(gaps) == 1:
        gap = _strip_gap_qualifier(gaps[0])
        variants = [
            f"The main fit gap is {gap}.",
            f"The unresolved commercial question is {gap}.",
            f"The biggest diligence gap is {gap}.",
        ]
        return choose_variant(variants, gap, "single_role_gap")
    primary_gap = _strip_gap_qualifier(gaps[0])
    secondary_gap = _strip_gap_qualifier(gaps[1])
    variants = [
        f"The main fit gap is {primary_gap}; {secondary_gap} also remains to be tested.",
        f"The biggest diligence gap is {primary_gap}; {secondary_gap} should be checked alongside it.",
        f"The unresolved commercial issue is {primary_gap}, with {secondary_gap} as the next point to test.",
    ]
    return choose_variant(variants, primary_gap, secondary_gap, "multi_role_gap")


def _mobility_follow_up_sentence(*, match_state: str) -> str:
    variants = {
        "no_reliable_match": [
            "No direct public signal of move readiness is visible, so mobility should be treated as uncertain and checked in follow-up conversation, with outreach framed as exploratory rather than timely.",
            "No direct public signal of move readiness is visible, so mobility should be treated as uncertain and checked in follow-up conversation, not inferred from the task input alone.",
            "No direct public signal of move readiness is visible, so mobility should be treated as uncertain and checked in follow-up conversation before any priority call is assumed.",
        ],
        "partial_match": [
            "No direct public signal of move readiness is visible, so mobility should be treated as uncertain and checked in follow-up conversation, even if the commercial overlap looks usable.",
            "No direct public signal of move readiness is visible, so mobility should be treated as uncertain and checked in follow-up conversation rather than inferred from partial chronology.",
            "No direct public signal of move readiness is visible, so mobility should be treated as uncertain and checked in follow-up conversation once profile certainty is tightened.",
        ],
        "likely_match": [
            "No direct public signal of move readiness is visible, so mobility should be treated as uncertain and checked in follow-up conversation rather than assumed from tenure alone.",
            "No direct public signal of move readiness is visible, so mobility should be treated as uncertain and checked in follow-up conversation even where chronology looks directionally solid.",
            "No direct public signal of move readiness is visible, so mobility should be treated as uncertain and checked in follow-up conversation before call priority is calibrated too tightly.",
        ],
        "verified_match": [
            "No direct public signal of move readiness is visible, so mobility should be treated as uncertain and checked in follow-up conversation rather than inferred from a settled remit.",
            "No direct public signal of move readiness is visible, so mobility should be treated as uncertain and checked in follow-up conversation even though the role chronology is clearer.",
            "No direct public signal of move readiness is visible, so mobility should be treated as uncertain and checked in follow-up conversation instead of being read off tenure alone.",
        ],
    }
    bucket = variants.get(match_state, variants["likely_match"])
    return choose_variant(bucket, match_state, "mobility_follow_up")


def _soften_role_fit_chain(text: str) -> str:
    # Keep the same evidence and ranking logic, but break up stacked "..., and ..., and ..."
    # patterns so the exported note reads more like recruiter judgment than prompt residue.
    softened = text
    softened = re.sub(r", and (the remit already sits\b)", r"; \1", softened)
    softened = re.sub(r", and (the current role is senior enough\b)", r"; \1", softened)
    softened = re.sub(r", and (the visible remit reads\b)", r"; \1", softened)
    softened = re.sub(r", and (public evidence points to\b)", r"; \1", softened)
    return softened


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
        "public evidence points to national scope": "the visible remit reads as national in scope",
        "public evidence points to anz scope": "the visible remit reads as ANZ in scope",
        "public evidence points to global scope": "the visible remit reads as global in scope",
        "public evidence points to regional scope": "the visible remit reads as regional in scope",
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
