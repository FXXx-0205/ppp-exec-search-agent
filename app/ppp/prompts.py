from __future__ import annotations

import json
from typing import Any

from app.ppp.enrichment import CandidateEnrichmentResult, NormalizedEvidence
from app.ppp.models import ROLE_NAME, CandidateCSVRow

CONTROL_PACKET_FIELDS = {
    "task",
    "tooling_contract",
    "reasoning_contract",
    "schema_rules",
    "generation_plan",
    "correction_request",
    "research_package",
}


def build_research_system_prompt() -> str:
    return (
        "You are a recruiter-assist agent preparing a normalized evidence packet for executive-search briefing. "
        "You must call the normalize_candidate_evidence tool exactly once before responding. "
        "After the tool result is available, return only that normalized evidence JSON object in valid JSON. "
        "Do not return a candidate brief in this phase. Do not return wrapper objects, task metadata, or reasoning notes."
    )


def build_research_user_prompt(
    *,
    candidate_id: str,
    candidate: CandidateCSVRow,
    enrichment: CandidateEnrichmentResult,
    role_spec: dict[str, Any],
) -> str:
    payload = {
        "candidate_identity": {
            "candidate_id": candidate_id,
            "full_name": candidate.full_name,
            "current_employer": candidate.current_employer,
            "current_title": candidate.current_title,
        },
        "research_package": enrichment.research_package(),
        "role_spec": {
            "role": role_spec.get("role", ROLE_NAME),
            "focus": role_spec.get("focus", []),
            "requirements": role_spec.get("requirements", []),
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def build_final_brief_system_prompt() -> str:
    return (
        "You are a senior executive recruiter writing a PPP shortlist note. "
        "Return only one CandidateBrief JSON object and nothing else. "
        "Do not use tools. Do not return wrapper objects, instructions, contracts, reasoning, schema commentary, or markdown. "
        "Use the normalized evidence object as the primary evidence contract. "
        "Be decisive but evidence-bounded: identify lane, seniority, why worth calling or not, the main fit, the main gap, and the first thing to test on a call. "
        "Differentiate clearly between shortlist-style targets, adjacent screens, and mapping leads. "
        "Do not flatten scores around the mid-point by default. "
        "State one strongest fit factor and one main test or gap per candidate, and keep uncertainty concise and specific rather than generic. "
        "For role_fit.justification, prefer this logic: sentence 1 = why in frame, sentence 2 = why not clean yet, sentence 3 = first call question. "
        "For outreach_hook, use one specific candidate angle only; do not stack three different selling points in one sentence. "
        "Write like a recruiter deciding whether the call is worth taking now, not like a résumé summarizer. "
        "Avoid reusing the same structural phrasing across every candidate note."
    )


def build_final_brief_user_prompt(
    *,
    candidate_id: str,
    candidate: CandidateCSVRow,
    normalized_evidence: NormalizedEvidence,
    enrichment: CandidateEnrichmentResult,
    role_spec: dict[str, Any],
) -> str:
    payload = {
        "candidate_identity": {
            "candidate_id": candidate_id,
            "full_name": candidate.full_name,
            "current_employer": candidate.current_employer,
            "current_title": candidate.current_title,
        },
        "normalized_evidence": normalized_evidence.model_dump(mode="json"),
        "recruiter_signals": enrichment.recruiter_signals.model_dump(mode="json"),
        "role_spec": {
            "role": role_spec.get("role", ROLE_NAME),
            "firm": role_spec.get("firm", ""),
            "focus": role_spec.get("focus", []),
            "requirements": role_spec.get("requirements", []),
            "ideal_background": role_spec.get("ideal_background", ""),
        },
        "firm_aum_context": enrichment.firm_aum_context,
        "writing_requirements": {
            "career_narrative": "exactly 3 sentences",
            "mobility_signal.rationale": "exactly 2 sentences",
            "role_fit.justification": "exactly 3 sentences",
            "outreach_hook": "exactly 1 sentence",
            "experience_tags": "non-empty array of strings",
            "role_fit.role": ROLE_NAME,
            "tenure_years_rule": "must be numeric; use normalized_evidence.tenure_years exactly",
            "recruiter_calibration": (
                "Make the priority legible: shortlist-style target vs adjacent screen vs mapping lead. "
                "Avoid giving near-identical role_fit scores unless the evidence truly supports it."
            ),
            "role_fit_sentence_plan": [
                "Sentence 1: strongest supported fit reason tied to employer, lane, scope, or seniority",
                "Sentence 2: main commercial gap or transfer risk with explicit uncertainty language",
                "Sentence 3: the first call question or screening angle",
            ],
            "outreach_hook_plan": (
                "Use one angle only. Prefer one of: direct-match angle, adjacent-transfer angle, gap-led angle, or verify-first angle. "
                "The hook should sound like a consultant opening a call, not a generic invitation."
            ),
        },
        "required_output_shape": {
            "candidate_id": candidate_id,
            "full_name": candidate.full_name,
            "current_role": {
                "title": normalized_evidence.verified_title,
                "employer": normalized_evidence.verified_employer,
                "tenure_years": normalized_evidence.tenure_years,
            },
            "career_narrative": "string",
            "experience_tags": ["string"],
            "firm_aum_context": "string",
            "mobility_signal": {"score": 3, "rationale": "string"},
            "role_fit": {"role": ROLE_NAME, "score": 6, "justification": "string"},
            "outreach_hook": "string",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def build_candidate_repair_system_prompt() -> str:
    return (
        "Repair one invalid CandidateBrief JSON object. "
        "Return only the repaired CandidateBrief JSON object. "
        "Do not include wrapper objects, instructions, contracts, reasoning, or schema commentary."
    )


def build_candidate_repair_user_prompt(
    *,
    invalid_payload: dict[str, Any],
    validation_error: str,
    candidate_id: str,
    full_name: str,
    firm_aum_context: str,
    tenure_years: float,
) -> str:
    payload = {
        "invalid_candidate_brief": invalid_payload,
        "validation_error": validation_error,
        "minimal_required_schema": {
            "candidate_id": candidate_id,
            "full_name": full_name,
            "current_role": {
                "title": "string",
                "employer": "string",
                "tenure_years": tenure_years,
            },
            "career_narrative": "string",
            "experience_tags": ["string"],
            "firm_aum_context": firm_aum_context,
            "mobility_signal": {"score": "integer 1-5", "rationale": "string"},
            "role_fit": {"role": ROLE_NAME, "score": "integer 1-10", "justification": "string"},
            "outreach_hook": "string",
        },
        "instruction": (
            "Return only the repaired CandidateBrief JSON object. "
            "Do not include wrapper objects, instructions, contracts, reasoning, or schema commentary."
        ),
    }
    return json.dumps(payload, ensure_ascii=False)
