from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from app.ppp.models import ROLE_NAME
from app.ppp.prompts import CONTROL_PACKET_FIELDS
from app.ppp.schema import (
    CandidateBrief,
    PPPOutput,
    format_validation_error,
    validate_candidate_brief,
    validate_output_document,
)

REQUIRED_CANDIDATE_BRIEF_FIELDS = {
    "candidate_id",
    "full_name",
    "current_role",
    "career_narrative",
    "experience_tags",
    "firm_aum_context",
    "mobility_signal",
    "role_fit",
    "outreach_hook",
}


def parse_json_response(raw_text: str) -> Any:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    return json.loads(cleaned)


def parse_candidate_response(raw_text: str) -> dict[str, Any]:
    payload = parse_json_response(raw_text)
    if isinstance(payload, dict) and "candidates" in payload and isinstance(payload["candidates"], list):
        if len(payload["candidates"]) != 1:
            raise ValueError("candidate response must contain exactly one candidate object when wrapped in candidates[].")
        candidate_payload = payload["candidates"][0]
        if not isinstance(candidate_payload, dict):
            raise ValueError("wrapped candidate payload must be an object.")
        _ensure_candidate_brief_shape(candidate_payload)
        return candidate_payload
    if not isinstance(payload, dict):
        raise ValueError("candidate response must be a JSON object.")
    _ensure_candidate_brief_shape(payload)
    return payload


def parse_normalized_evidence_response(raw_text: str) -> dict[str, Any]:
    payload = parse_json_response(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("normalized evidence response must be a JSON object.")
    if CONTROL_PACKET_FIELDS.intersection(payload.keys()):
        raise ValueError("Model returned control packet instead of normalized evidence.")
    return payload


def repair_candidate_payload(
    payload: dict[str, Any],
    *,
    candidate_id: str,
    full_name: str,
    firm_aum_context: str | None = None,
    inferred_tenure_years: float | None = None,
) -> dict[str, Any]:
    repaired = dict(payload)
    repaired.setdefault("candidate_id", candidate_id)
    repaired.setdefault("full_name", full_name)

    experience_tags = repaired.get("experience_tags")
    if isinstance(experience_tags, str):
        repaired["experience_tags"] = [_normalize_tag(item) for item in experience_tags.split(",") if item.strip()]
    elif isinstance(experience_tags, list):
        repaired["experience_tags"] = [_normalize_tag(item) for item in experience_tags if isinstance(item, str) and item.strip()]

    current_role = repaired.get("current_role")
    if isinstance(current_role, dict):
        role_copy = dict(current_role)
        tenure_years = role_copy.get("tenure_years")
        if isinstance(tenure_years, str):
            stripped = tenure_years.strip().replace("years", "").replace("year", "").strip()
            if stripped:
                role_copy["tenure_years"] = stripped
        if role_copy.get("tenure_years") in {None, ""}:
            role_copy["tenure_years"] = inferred_tenure_years if inferred_tenure_years is not None else 0.0
        repaired["current_role"] = role_copy

    mobility_signal = repaired.get("mobility_signal")
    if isinstance(mobility_signal, dict):
        mobility_copy = dict(mobility_signal)
        if isinstance(mobility_copy.get("score"), str):
            mobility_copy["score"] = _coerce_numeric_string(mobility_copy["score"])
        repaired["mobility_signal"] = mobility_copy

    role_fit = repaired.get("role_fit")
    if isinstance(role_fit, dict):
        role_fit_copy = dict(role_fit)
        role_fit_copy.setdefault("role", ROLE_NAME)
        if isinstance(role_fit_copy.get("score"), str):
            role_fit_copy["score"] = _coerce_numeric_string(role_fit_copy["score"])
        repaired["role_fit"] = role_fit_copy

    if not repaired.get("firm_aum_context") and firm_aum_context:
        repaired["firm_aum_context"] = firm_aum_context

    outreach_hook = repaired.get("outreach_hook")
    if isinstance(outreach_hook, str):
        repaired["outreach_hook"] = _first_sentence(outreach_hook)

    for field_name in ("career_narrative", "firm_aum_context"):
        value = repaired.get(field_name)
        if isinstance(value, str):
            repaired[field_name] = _clean_text(value)

    if isinstance(repaired.get("mobility_signal"), dict):
        rationale = repaired["mobility_signal"].get("rationale")
        if isinstance(rationale, str):
            repaired["mobility_signal"]["rationale"] = _clean_text(rationale)
    if isinstance(repaired.get("role_fit"), dict):
        justification = repaired["role_fit"].get("justification")
        if isinstance(justification, str):
            repaired["role_fit"]["justification"] = _clean_text(justification)

    return repaired


def validate_and_repair_candidate_payload(
    payload: dict[str, Any],
    *,
    candidate_id: str,
    full_name: str,
    firm_aum_context: str | None = None,
    inferred_tenure_years: float | None = None,
) -> CandidateBrief:
    repaired = repair_candidate_payload(
        payload,
        candidate_id=candidate_id,
        full_name=full_name,
        firm_aum_context=firm_aum_context,
        inferred_tenure_years=inferred_tenure_years,
    )
    return validate_candidate_brief(repaired)


def looks_like_candidate_brief_payload(payload: dict[str, Any]) -> bool:
    return REQUIRED_CANDIDATE_BRIEF_FIELDS.issubset(payload.keys())


def is_control_packet_payload(payload: dict[str, Any]) -> bool:
    return bool(CONTROL_PACKET_FIELDS.intersection(payload.keys()))


def validate_output_payload(payload: Any) -> PPPOutput:
    return validate_output_document(payload)


def describe_validation_failure(exc: ValidationError) -> str:
    return format_validation_error(exc)


def _coerce_numeric_string(value: str) -> int | float | str:
    stripped = value.strip()
    if not stripped:
        return value
    try:
        number = float(stripped)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def _normalize_tag(value: str) -> str:
    cleaned = _clean_text(value).lower()
    tag_map = {
        "inst. sales": "institutional sales",
        "institutional": "institutional sales",
        "wholesale": "wholesale distribution",
        "bdm": "business development",
        "leadership": "team leadership",
        "team lead": "team leadership",
    }
    return tag_map.get(cleaned, cleaned)


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _first_sentence(value: str) -> str:
    cleaned = _clean_text(value)
    match = re.match(r"(.+?[.!?])(?:\s|$)", cleaned)
    return match.group(1).strip() if match else cleaned


def _ensure_candidate_brief_shape(payload: dict[str, Any]) -> None:
    if is_control_packet_payload(payload):
        raise ValueError("Model returned control packet instead of CandidateBrief.")
    if not looks_like_candidate_brief_payload(payload):
        raise ValueError("Model returned JSON object that does not match CandidateBrief shape.")
