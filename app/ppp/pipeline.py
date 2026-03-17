from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.config import settings
from app.llm.anthropic_client import ClaudeClient
from app.ppp.enrichment import (
    CandidateEnrichmentResult,
    CandidatePublicProfileLookupInput,
    CandidatePublicProfileLookupTool,
    format_enrichment_validation_error,
    validate_enrichment_payload,
)
from app.ppp.models import REQUIRED_CSV_COLUMNS, CandidateCSVRow
from app.ppp.prompts import (
    build_enrichment_system_prompt,
    build_enrichment_user_prompt,
    build_generation_system_prompt,
    build_generation_user_prompt,
)
from app.ppp.qa import run_bundle_qa, write_qa_report
from app.ppp.quality import validate_output_quality
from app.ppp.schema import CandidateBrief, PPPOutput
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
) -> PPPOutput:
    api_key = settings.anthropic_api_key
    if not api_key:
        raise PPPTaskError("ANTHROPIC_API_KEY is missing. Set it in your environment or .env before running the PPP task.")

    candidates = _load_candidates_csv(Path(input_path))
    role_spec = _load_role_spec(Path(role_spec_path))
    client = ClaudeClient(api_key=api_key)
    lookup_tool = CandidatePublicProfileLookupTool(fixture_path=research_fixture_path)
    candidate_inputs_by_id: dict[str, CandidateCSVRow] = {}
    enrichments_by_id: dict[str, CandidateEnrichmentResult] = {}

    output_candidates: list[CandidateBrief] = []
    for idx, candidate in enumerate(candidates, start=1):
        candidate_id = f"candidate_{idx}"
        candidate_inputs_by_id[candidate_id] = candidate
        logger.info("Enriching %s (%s/5)", candidate.full_name, idx)
        try:
            enrichment = _run_enrichment_stage(
                client=client,
                candidate_id=candidate_id,
                candidate=candidate,
                model=model,
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
        raise PPPTaskError(f"Output quality check failed: {exc}") from exc
    qa_report = run_bundle_qa(output=output, candidates=candidate_inputs_by_id, enrichments=enrichments_by_id)
    qa_report_path = write_qa_report(qa_report, path=str(Path(intermediate_dir) / "qa_report.json"))
    logger.info("QA report written to %s", qa_report_path)
    if not qa_report.passed:
        raise PPPTaskError(f"Post-generation QA failed. Review {qa_report_path} for details.")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("PPP output written to %s", output_file)
    return output


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
            return validate_and_repair_candidate_payload(
                parsed,
                candidate_id=candidate_id,
                full_name=candidate.full_name,
                firm_aum_context=enrichment.firm_aum_context,
                inferred_tenure_years=enrichment.inferred_tenure_years,
            )
        except ValidationError as exc:
            previous_error = describe_validation_failure(exc)
            if attempt == 1:
                raise PPPTaskError(
                    f"Claude returned invalid schema output for {candidate.full_name}: {previous_error}"
                ) from exc

    raise PPPTaskError(f"Candidate generation failed for {candidate.full_name}.")


def _run_enrichment_stage(
    *,
    client: ClaudeClient,
    candidate_id: str,
    candidate: CandidateCSVRow,
    model: str,
    lookup_tool: CandidatePublicProfileLookupTool,
) -> CandidateEnrichmentResult:
    tool_input = CandidatePublicProfileLookupInput.from_candidate(candidate_id=candidate_id, candidate=candidate)
    system_prompt = build_enrichment_system_prompt()
    user_prompt = build_enrichment_user_prompt(candidate_id=candidate_id, candidate=candidate)

    if client._client is None:
        return lookup_tool.run(tool_input)

    raw = client.generate_with_tools(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        max_tokens=1600,
        tools=[lookup_tool.definition],
        tool_handler=lambda name, payload: _handle_lookup_tool(lookup_tool, name, payload),
    )

    try:
        parsed = json.loads(raw)
        return validate_enrichment_payload(parsed)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning(
            "Tool-assisted enrichment failed for %s; falling back to direct fixture lookup: %s",
            candidate.full_name,
            exc if not isinstance(exc, ValidationError) else format_enrichment_validation_error(exc),
        )
        return lookup_tool.run(tool_input)


def _handle_lookup_tool(
    lookup_tool: CandidatePublicProfileLookupTool,
    tool_name: str,
    payload: dict[str, Any],
) -> str:
    if tool_name != "candidate_public_profile_lookup":
        raise PPPTaskError(f"Unsupported tool requested: {tool_name}")
    return lookup_tool.run_json(payload)


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
