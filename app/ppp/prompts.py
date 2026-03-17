from __future__ import annotations

import json

from app.ppp.enrichment import CandidateEnrichmentResult
from app.ppp.models import ROLE_NAME, CandidateCSVRow


def build_enrichment_system_prompt() -> str:
    return (
        "You are coordinating public-profile enrichment for an executive-search workflow. "
        "You must call the candidate_public_profile_lookup tool exactly once, then return only a JSON object matching "
        "the enrichment schema. Do not invent live-public facts beyond the tool result."
    )


def build_enrichment_user_prompt(*, candidate_id: str, candidate: CandidateCSVRow) -> str:
    return json.dumps(
        {
            "task": "Enrich this candidate for downstream briefing generation.",
            "candidate": {
                "candidate_id": candidate_id,
                **candidate.model_dump(mode="json"),
            },
            "required_fields": {
                "tool_name": "candidate_public_profile_lookup",
                "tool_mode": "fixture_backed or live_public_research",
                "verified_public_snippets": ["string"],
                "inferred_tenure_years": "number or null",
                "tenure_rationale": "string",
                "likely_channel_evidence": ["string"],
                "likely_experience_evidence": ["string"],
                "firm_aum_context": "string",
                "firm_context_clues": ["string"],
                "mobility_evidence": ["string"],
                "missing_fields": ["string"],
                "uncertain_fields": ["string"],
                "confidence_notes": ["string"],
                "sources": [{"label": "string", "source_type": "string", "url": "string or null", "confidence": "string"}],
                "evidence": [{"category": "string", "signal": "string", "snippet": "string", "source_labels": ["string"]}],
            },
        },
        ensure_ascii=False,
    )


def build_generation_system_prompt() -> str:
    return (
        "You are an executive-search research analyst writing candidate briefings for a specialist search firm. "
        "Return only valid JSON for one candidate. Never use markdown fences. Never invent facts beyond the supplied "
        "enrichment package. When evidence is uncertain, say so in a recruiter-usable way instead of overstating. "
        "Write like a search consultant: specific, concise, commercially useful, and candidate-specific. "
        "career_narrative must be 3-4 sentences, experience_tags must come from evidence, mobility_signal must be "
        "grounded in tenure/trajectory evidence, role_fit.justification must explain both strengths and limits, and "
        "outreach_hook must sound like a real consultant opening line."
    )


def build_generation_user_prompt(
    *,
    candidate_id: str,
    candidate: CandidateCSVRow,
    enrichment: CandidateEnrichmentResult,
    role_spec: dict,
    previous_error: str | None = None,
    previous_output: str | None = None,
) -> str:
    payload = {
        "task": "Generate a single candidate briefing object that matches the required schema exactly.",
        "writing_standards": {
            "career_narrative": [
                "3 to 4 sentences",
                "include current role, seniority, channel/functional direction, and a concise career arc",
                "must mention uncertainty when evidence is incomplete",
            ],
            "experience_tags": [
                "use evidence-backed tags only",
                "normalize to recruiter-friendly labels such as institutional sales, wholesale distribution, team leadership",
            ],
            "firm_aum_context": [
                "describe firm type and AUM context",
                "if exact AUM is not verified, say limited public visibility or verification still required",
            ],
            "mobility_signal": [
                "score 1 to 5",
                "base rationale on tenure, trajectory, and public signals only",
            ],
            "role_fit": [
                f"role must be {ROLE_NAME}",
                "score 1 to 10",
                "justify fit against the provided role spec with both strengths and gaps",
            ],
            "outreach_hook": [
                "exactly 1 sentence",
                "sound like a consultant, not marketing copy",
                "must be candidate-specific",
            ],
        },
        "schema_rules": {
            "candidate_id": candidate_id,
            "role_fit.role": ROLE_NAME,
            "career_narrative": "3 to 4 sentences",
            "mobility_signal.score": "integer 1 to 5",
            "role_fit.score": "integer 1 to 10",
            "outreach_hook": "1 sentence only",
            "experience_tags": "non-empty array of strings",
        },
        "candidate_input": candidate.model_dump(mode="json"),
        "enrichment": enrichment.model_dump(mode="json"),
        "role_spec": role_spec,
        "required_output_shape": {
            "candidate_id": candidate_id,
            "full_name": candidate.full_name,
            "current_role": {
                "title": candidate.current_title,
                "employer": candidate.current_employer,
                "tenure_years": enrichment.inferred_tenure_years or "number",
            },
            "career_narrative": "string",
            "experience_tags": ["string"],
            "firm_aum_context": enrichment.firm_aum_context,
            "mobility_signal": {"score": 3, "rationale": "string"},
            "role_fit": {
                "role": ROLE_NAME,
                "score": 7,
                "justification": "string",
            },
            "outreach_hook": "string",
        },
    }

    if previous_error:
        payload["correction_request"] = {
            "previous_error": previous_error,
            "instruction": "Fix the schema or quality issue and return corrected JSON only.",
            "previous_output": previous_output or "",
        }

    return json.dumps(payload, ensure_ascii=False)
