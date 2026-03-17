from __future__ import annotations

import re

from app.ppp.schema import PPPOutput

GENERIC_TEMPLATE_PHRASES = (
    "directionally aligned",
    "highly relevant to a current search mandate",
    "looks relevant to a leadership search we are currently running",
)


def validate_output_quality(output: PPPOutput) -> None:
    _ensure_unique_candidate_field(output, field_name="career_narrative")
    _ensure_unique_candidate_field(output, field_name="role_fit.justification")
    _ensure_unique_candidate_field(output, field_name="outreach_hook")
    _ensure_non_template_language(output)


def _ensure_unique_candidate_field(output: PPPOutput, *, field_name: str) -> None:
    seen: dict[str, str] = {}
    for candidate in output.candidates:
        value = _extract_field(candidate, field_name)
        normalized = _normalize_text(value)
        if normalized in seen:
            raise ValueError(
                f"{field_name} is duplicated for {candidate.candidate_id} and {seen[normalized]}; output must be candidate-specific."
            )
        seen[normalized] = candidate.candidate_id


def _ensure_non_template_language(output: PPPOutput) -> None:
    for candidate in output.candidates:
        combined = " ".join(
            [
                candidate.career_narrative,
                candidate.role_fit.justification,
                candidate.outreach_hook,
            ]
        ).lower()
        for phrase in GENERIC_TEMPLATE_PHRASES:
            if phrase in combined:
                raise ValueError(
                    f"{candidate.candidate_id} contains template-like phrasing ('{phrase}') that should be rewritten more specifically."
                )


def _extract_field(candidate, field_name: str) -> str:
    if field_name == "career_narrative":
        return candidate.career_narrative
    if field_name == "role_fit.justification":
        return candidate.role_fit.justification
    if field_name == "outreach_hook":
        return candidate.outreach_hook
    raise ValueError(f"Unsupported field quality check: {field_name}")


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
