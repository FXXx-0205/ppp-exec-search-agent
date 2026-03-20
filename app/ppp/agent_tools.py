from __future__ import annotations

from typing import Any

from app.ppp.enrichment import CandidateEnrichmentResult


def build_candidate_normalization_tool() -> dict[str, Any]:
    return {
        "name": "normalize_candidate_evidence",
        "description": (
            "Normalize the candidate research package into a recruiter-safe evidence summary with a stable "
            "match-confidence state, validated role signals, firm context notes, and visible uncertainty boundaries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_identity": {"type": "object"},
                "research_package": {"type": "object"},
                "role_spec": {"type": "object"},
            },
            "required": ["candidate_identity", "research_package", "role_spec"],
        },
    }


def run_candidate_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    enrichment: CandidateEnrichmentResult,
    role_spec: dict[str, Any],
) -> dict[str, Any]:
    if tool_name != "normalize_candidate_evidence":
        raise ValueError(f"Unsupported Claude tool: {tool_name}")

    # Return the exact normalized-evidence contract used downstream so the research
    # phase can fail over to the real tool result without inventing another format.
    return enrichment.normalized_evidence().model_dump(mode="json")
