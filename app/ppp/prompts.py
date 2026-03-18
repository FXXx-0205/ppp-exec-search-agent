from __future__ import annotations

import json

from app.ppp.enrichment import CandidateEnrichmentResult
from app.ppp.models import ROLE_NAME, CandidateCSVRow


def build_generation_system_prompt() -> str:
    return (
        "You are an executive-search research analyst writing candidate briefings for a specialist search firm. "
        "Return only valid JSON for one candidate. Never use markdown fences. Never invent facts beyond the supplied "
        "structured research package. Treat claims marked verified as hard evidence, claims marked strongly_inferred "
        "as directional but still bounded evidence, and claims marked uncertain as follow-up items that require cautious wording. "
        "Do not promote uncertain claims into hard facts. You MUST stay within the allowed_facts contract and use safe degradation "
        "when evidence is incomplete. If a field is qualitative_only, do not introduce numbers. If a fact is uncertain, state the "
        "uncertainty explicitly. If a detail cannot be supported safely, use minimal recruiter-usable wording rather than inventing "
        "specificity. If a data point cannot be directly verified from public evidence, you must say so explicitly. You may provide a cautious contextual estimate, "
        "but you must not present inferred information as confirmed fact. A useful recruiter-facing brief makes uncertainty visible rather than hiding it behind confident language. "
        "Write like a search consultant: specific, concise, commercially useful, and candidate-specific. "
        "Avoid generic recruiter-template language such as 'directionally aligned', 'highly relevant to a current search mandate', "
        "'strong directional fit', 'directional fit', or 'looks relevant to a leadership search we are currently running'. "
        "career_narrative must be exactly 3 sentences, experience_tags must come from supported claims, "
        "mobility_signal must be grounded in tenure/trajectory claims and written in a cautious two-sentence structure, "
        "role_fit.justification must explain both strengths and limits in exactly three sentences, "
        "and outreach_hook must be exactly one sentence with a specific commercial reason-for-call. recruiter_signals are provided only to organize "
        "recruiter-useful writing; they are not a new source of facts and must remain inside the same claims/confidence boundary. Treat the task as constrained slot filling plus polished language, "
        "not open-ended generation."
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
        "reasoning_contract": {
            "use_only_supported_claims": True,
            "output_must_be_subset_of_allowed_facts": True,
            "verified_claims": "May be stated directly when they are relevant to the recruiter-facing brief.",
            "strongly_inferred_claims": "May be used with directional language but should not be overstated as guaranteed facts.",
            "uncertain_claims": "Must be framed with caution or verification language rather than presented as settled fact.",
            "recruiter_signals_usage": [
                "recruiter_signals help organize recruiter-useful wording only",
                "do not treat recruiter_signals as independent facts beyond the supplied claims and verification summary",
                "all sell points and gaps must stay inside the claim/confidence boundary",
                "if mandate_similarity is adjacent_match or step_up_candidate, do not rewrite it as direct fit or proven fit",
                "if a data point cannot be directly verified, say that explicitly rather than smoothing it into recruiter confidence",
                "contextual estimates are allowed only when they are clearly marked as estimated, broad, or treated here as",
            ],
        },
        "failure_policy": {
            "priority": "Prefer safe degradation over failure where possible, but never cross evidence boundaries.",
            "soft_degrade": [
                "If AUM is not numerically supported, use qualitative firm context only.",
                "If exact AUM cannot be directly verified, say unable to verify from public sources and use only cautious contextual positioning.",
                "If tenure is directional rather than verified, use approximate or cautious wording.",
                "If role-fit support is limited, keep justification high-level and mention the limitation.",
            ],
            "hard_fail_conditions": [
                "Do not add unsupported numbers, employers, titles, chronology, or channel claims.",
                "Do not compress uncertain research into certain-sounding statements.",
                "Do not present contextual estimates as confirmed facts.",
            ],
        },
        "writing_standards": {
            "career_narrative": [
                "exactly 3 sentences",
                "sentence 1: current lane using title + employer + recruiter_signals.channel_orientation",
                "sentence 2: commercial relevance using recruiter_signals.mandate_similarity plus recruiter_signals.key_sell_points",
                "sentence 3: explicit verification boundary using recruiter_signals.key_gaps",
                "avoid generic template phrases",
                "do not use achievement framing unless directly stated in a supported claim",
                "do not use built, established track record, led transformation, scaled the function, positioned as, or drove growth unless a claim directly supports that wording",
            ],
            "experience_tags": [
                "use claim-backed tags only",
                "normalize to recruiter-friendly labels such as institutional sales, wholesale distribution, team leadership",
            ],
            "firm_aum_context": [
                "describe firm type and AUM context",
                "if exact AUM is not supported by a verified claim, explicitly say unable to verify from public sources or public evidence does not confirm exact AUM",
                "if exact AUM is not verified, you may give only a cautious qualitative estimate based on firm type and sector context",
                "if allowed_facts marks firm_aum_context as qualitative_only, do not use numbers",
                "do not invent a specific AUM number when direct verification is absent",
            ],
            "mobility_signal": [
                "score 1 to 5",
                "base rationale on tenure, trajectory, and supported public signals only",
                "write exactly 2 sentences",
                "sentence 1: observable timing or chronology only",
                "sentence 2: direct mobility evidence absent + explicit uncertainty + need for follow-up",
                "do not infer motivation, commitment horizon, or active-search intent unless explicitly supported by the research package",
                "when evidence is thin, prefer neutral wording such as uncertain, difficult to assess, or requires follow-up",
                "do not use settled, ready to move, open to move, low near-term mobility, likely to leave, or natural transition point unless directly evidenced",
            ],
            "role_fit": [
                f"role must be {ROLE_NAME}",
                "score 1 to 10",
                "write exactly 3 sentences",
                "sentence 1: why the candidate is in frame using recruiter_signals.mandate_similarity plus recruiter_signals.key_sell_points",
                "sentence 2: biggest commercial gap using recruiter_signals.key_gaps with explicit unverified wording",
                "sentence 3: explicit screening angle such as 'The first screening question should be...' or 'The key point to test in conversation is...'",
                "justify fit against the provided role spec with both strengths and gaps",
                "avoid generic template phrases and make the reasoning candidate-specific",
                "do not state a capability as proven unless a supported claim directly anchors it",
                "when in doubt, describe the gap or need for verification instead of upgrading the candidate's strength",
                "do not use phrases like directionally aligned, directional fit, or strong directional fit",
                "allowed language includes suggests, appears relevant, likely exposure, and public evidence supports",
                "do not use strong fit, proven, deep network, fully de-risked, or well beyond the role unless directly supported",
            ],
            "outreach_hook": [
                "exactly 1 sentence",
                "sound like a consultant, not marketing copy",
                "must be candidate-specific",
                "avoid generic search-template wording",
                "must include a specific commercial angle such as institutional build-out, wholesale expansion, leadership step-up, comparable distribution remit, or adjacent but relevant lane",
                "must be based on recruiter_signals rather than a generic invitation",
            ],
        },
        "schema_rules": {
            "candidate_id": candidate_id,
            "role_fit.role": ROLE_NAME,
            "career_narrative": "exactly 3 sentences",
            "mobility_signal.score": "integer 1 to 5",
            "role_fit.score": "integer 1 to 10",
            "outreach_hook": "1 sentence only",
            "experience_tags": "non-empty array of strings",
        },
        "candidate_identity": {
            "candidate_id": candidate_id,
            "full_name": candidate.full_name,
        },
        "research_package": enrichment.research_package(),
        "allowed_facts": enrichment.allowed_facts(),
        "role_spec": role_spec,
        "required_output_shape": {
            "candidate_id": candidate_id,
            "full_name": candidate.full_name,
            "current_role": {
                "title": "string supported by the research package",
                "employer": "string supported by the research package",
                "tenure_years": enrichment.inferred_tenure_years or "number",
            },
            "career_narrative": "string",
            "experience_tags": ["string"],
            "firm_aum_context": "string grounded in firm_context claims",
            "mobility_signal": {"score": 3, "rationale": "string"},
            "role_fit": {
                "role": ROLE_NAME,
                "score": 7,
                "justification": "string",
            },
            "outreach_hook": "string",
        },
        "generation_plan": {
            "mode": "constrained_slot_filling",
            "field_order": [
                "current_role",
                "career_narrative",
                "experience_tags",
                "firm_aum_context",
                "mobility_signal",
                "role_fit",
                "outreach_hook",
            ],
        },
        "field_specific_guardrails": {
            "mobility_signal": {
                "good_example": "Public chronology suggests approximately 1.9 years in the current role. No direct public signal of move readiness is visible, so mobility should be treated as uncertain pending conversation.",
                "avoid": [
                    "recent promotion implies multi-year commitment",
                    "candidate is unlikely to move",
                    "candidate is in a natural transition window",
                    "settled",
                    "open to move",
                    "ready to move",
                    "flight risk",
                ],
            },
            "firm_aum_context": {
                "good_example": "Unable to verify exact AUM from public sources; based on firm type and sector context, the platform appears broadly comparable to a mid-tier active manager.",
                "avoid": [
                    "invented AUM numbers without direct support",
                    "exact AUM stated as fact when public evidence does not confirm it",
                ],
            },
            "role_fit": {
                "good_example": "Public evidence supports relevance on current title and distribution remit. However, direct evidence of team scale and channel breadth remains unverified. This should be tested early in screening conversation.",
                "avoid": [
                    "demonstrated team leadership unless explicitly supported",
                    "deep Australian market networks unless explicitly supported",
                    "proven track record claims without direct evidence",
                    "strong fit",
                    "well beyond the role",
                    "established profile",
                ],
            },
            "career_narrative": {
                "good_example": "The candidate is currently in an institutional lane at the named employer. That looks commercially relevant to the brief because public evidence supports institutional channel relevance. Public evidence does not fully confirm broader channel depth, and this should be confirmed in conversation.",
                "avoid": [
                    "built the function",
                    "transformed",
                    "led the build-out",
                    "established track record",
                    "positioned as",
                    "scaled the function",
                    "drove growth",
                ],
            },
            "outreach_hook": {
                "avoid": [
                    "would welcome a conversation",
                    "caught our attention",
                    "worth discussing",
                    "current search mandate",
                ],
            },
        },
    }

    if previous_error:
        payload["correction_request"] = {
            "previous_error": previous_error,
            "instruction": "Fix the schema or quality issue and return corrected JSON only.",
            "previous_output": previous_output or "",
        }

    return json.dumps(payload, ensure_ascii=False)
