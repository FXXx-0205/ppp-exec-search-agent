from __future__ import annotations

import json

from app.ppp.enrichment import CandidateEnrichmentResult
from app.ppp.models import ROLE_NAME, CandidateCSVRow


def build_generation_system_prompt() -> str:
    return (
        "You are a senior executive recruiter writing shortlist notes for a specialist search firm. "
        "Write with a crisp, direct, commercially grounded tone. Sound like a senior recruiter, not an analyst explaining reasoning. "
        "Return only valid JSON for one candidate. Never use markdown fences. Never invent facts beyond the supplied "
        "structured research package. Treat claims marked verified as hard evidence, claims marked strongly_inferred "
        "as directional but still bounded evidence, and claims marked uncertain as follow-up items that require cautious wording. "
        "Do not promote uncertain claims into hard facts. You MUST stay within the allowed_facts contract and use safe degradation "
        "when evidence is incomplete. If a field is qualitative_only, do not introduce numbers. If a fact is uncertain, state the "
        "uncertainty explicitly. If a detail cannot be supported safely, use minimal recruiter-usable wording rather than inventing "
        "specificity. If a data point cannot be directly verified from public evidence, you must say so explicitly. You may provide a cautious contextual estimate, "
        "but you must not present inferred information as confirmed fact. A useful recruiter-facing brief makes uncertainty visible rather than hiding it behind confident language. "
        "Write like a search consultant: specific, concise, commercially useful, and candidate-specific. State conclusions directly rather than narrating your thought process. "
        "Avoid generic recruiter-template language such as 'directionally aligned', 'highly relevant to a current search mandate', "
        "'strong directional fit', 'directional fit', or 'looks relevant to a leadership search we are currently running'. "
        "Avoid stiff phrasing such as 'credible angle', 'adjacent transferability', 'appears relevant to a comparable remit', "
        "'keeps this profile in frame', 'public evidence supports relevance', or 'should be handled as strong evidence until tested'. "
        "Do not use these phrases: 'Currently sitting in a [X] lane', 'From a recruiter lens, the relevance comes from', "
        "'That puts the profile close to the', 'The main commercial gap is', or 'The first screening question is'. "
        "career_narrative must be exactly 3 sentences, experience_tags must come from supported claims, "
        "mobility_signal must be grounded in tenure/trajectory claims and written in a cautious two-sentence structure, "
        "role_fit.justification must explain both strengths and limits in exactly three sentences, "
        "and outreach_hook must be exactly one sentence with a specific commercial reason-for-call. recruiter_signals are provided only to organize "
        "recruiter-useful writing; they are recruiter-oriented interpretations of the existing claims, not a new source of facts, and they must remain inside the same claims/confidence boundary. "
        "Use recruiter_signals to sharpen lane, scope, fit type, commercial reason-to-call, and commercial gap, but never to introduce facts not already supported in claims or verification. Treat the task as constrained slot filling plus polished language, "
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
                "recruiter_signals are not independent facts beyond the supplied claims and verification summary",
                "all sell points and gaps must stay inside the claim/confidence boundary",
                "if mandate_similarity is adjacent_match or step_up_candidate, do not rewrite it as direct fit or proven fit",
                "use lane, scope, seniority, evidence strength, and screening question to make the writing recruiter-usable without adding new facts",
                "if a data point cannot be directly verified, say that explicitly rather than smoothing it into recruiter confidence",
                "identity_resolution.status governs how firmly you may talk about the candidate as a verified public match",
                "if identity_resolution.status is not_verified, describe the task input as unverified rather than implying the public profile was confirmed",
                "contextual estimates are allowed only when they are clearly marked as estimated, broad, or treated here as",
                "state the commercial conclusion directly rather than narrating how you reached it",
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
                "if identity_resolution.status is not_verified, sentence 1 must say the task input was not verified from public sources",
                "if identity_resolution.status is possible_match, sentence 1 must use possible-match wording rather than verified-profile wording",
                "sentence 1: current lane + scope using title + employer + recruiter_signals.channel_orientation + recruiter_signals.scope_signal + recruiter_signals.seniority_signal",
                "sentence 2: commercial relevance using recruiter_signals.mandate_similarity plus recruiter_signals.key_sell_points",
                "sentence 3: explicit verification boundary using recruiter_signals.key_gaps and recruiter_signals.evidence_strength",
                "must answer which lane the candidate sits in, why that matters commercially, and what business boundary still needs testing",
                "vary the phrasing across candidates; do not reuse the same sentence skeleton every time",
                "bounded directional language such as leans toward, weighted toward, or broad distribution remit is allowed when it stays inside the claim boundary",
                "do not write your reasoning process; give the conclusion directly",
                "avoid generic template phrases",
                "do not use achievement framing unless directly stated in a supported claim",
                "do not use built, established track record, led transformation, scaled the function, positioned as, or drove growth unless a claim directly supports that wording",
                "do not use 'Currently sitting in a [X] lane', 'From a recruiter lens, the relevance comes from', or 'That puts the profile close to the'",
            ],
            "experience_tags": [
                "use claim-backed tags only",
                "normalize to recruiter-friendly labels such as institutional sales, wholesale distribution, team leadership",
            ],
            "firm_aum_context": [
                "describe firm type and AUM context",
                "if exact AUM is not supported by a verified claim, state the uncertainty clearly but vary the phrasing naturally",
                "if exact AUM is not verified, you may give only a cautious qualitative estimate based on firm type and sector context",
                "if you include a numeric AUM from public materials, explicitly frame it as an estimate based on public references and note that it remains subject to verification",
                "if allowed_facts marks firm_aum_context as qualitative_only, do not use numbers",
                "do not invent a specific AUM number when direct verification is absent",
                "do not always start with 'Unable to verify exact AUM from public sources'",
            ],
            "mobility_signal": [
                "score 1 to 5",
                "base rationale on tenure, trajectory, and supported public signals only",
                "write exactly 2 sentences",
                "sentence 1: observable timing or chronology only",
                "sentence 2: direct mobility evidence absent + explicit uncertainty + need for follow-up",
                "if tenure_years is null or unavailable, sentence 1 must say clearly that current-role tenure is not established",
                "if research_package.tool_mode is fixture_backed or fixture_fallback and tenure is unavailable, make clear the fixture-backed run did not provide tenure evidence",
                "do not infer motivation, commitment horizon, or active-search intent unless explicitly supported by the research package",
                "when evidence is thin, prefer neutral wording such as uncertain, difficult to assess, or requires follow-up",
                "do not use settled, ready to move, open to move, low near-term mobility, likely to leave, or natural transition point unless directly evidenced",
            ],
            "role_fit": [
                f"role must be {ROLE_NAME}",
                "score 1 to 10",
                "if identity_resolution.status is not_verified, keep score low and make the identity uncertainty explicit",
                "write exactly 3 sentences",
                "sentence 1: why the candidate is in frame using recruiter_signals.mandate_similarity plus recruiter_signals.channel_orientation, recruiter_signals.seniority_signal, and recruiter_signals.key_sell_points",
                "sentence 2: biggest commercial gap using recruiter_signals.key_gaps with explicit unverified wording",
                "sentence 3: explicit screening angle using recruiter_signals.screening_priority_question",
                "must answer why in frame, the main commercial gap, and the first thing to test on the phone",
                "justify fit against the provided role spec with both strengths and gaps",
                "avoid generic template phrases and make the reasoning candidate-specific",
                "write like a shortlist note rather than a system label",
                "do not narrate the logic; state the commercial judgment directly",
                "prefer direct constructions such as 'Strong alignment on X, but Y remains unverified. Screening priority: Z.'",
                "do not state a capability as proven unless a supported claim directly anchors it",
                "when in doubt, describe the gap or need for verification instead of upgrading the candidate's strength",
                "do not use phrases like directionally aligned, directional fit, or strong directional fit",
                "allowed language includes suggests, appears relevant, likely exposure, and public evidence supports",
                "do not use strong fit, proven, deep network, fully de-risked, or well beyond the role unless directly supported",
                "do not use rigid labels like direct match candidate or adjacent match candidate as standalone phrasing",
                "prefer recruiter-relevant commercial gaps such as network depth, team scale, super/platform/IFA/institutional coverage, product breadth, or market profile over low-value admin gaps",
                "do not use 'The main commercial gap is' or 'The first screening question is'",
            ],
            "outreach_hook": [
                "exactly 1 sentence",
                "sound like a real LinkedIn InMail opener from a recruiter",
                "must be candidate-specific",
                "avoid generic search-template wording",
                "must include a specific commercial angle such as institutional build-out, wholesale expansion, ANZ wealth channel leadership, national distribution leadership, or adjacent but relevant lane transferability",
                "must be based on recruiter_signals rather than a generic invitation",
                "keep it conversational and natural, in one short sentence",
                "do not simply repeat title + employer or stack multiple specialist labels",
                "it may open with 'Hi [Name], ...' and may close with 'Worth a brief chat?'",
                "do not use what stood out was, credible angle, appears relevant, or adjacent transferability",
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
        "identity_resolution": enrichment.identity_resolution.model_dump(mode="json"),
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
                "good_example": "Exact AUM remains unverified publicly. Firm profile indicates an established active-management platform rather than a start-up.",
                "avoid": [
                    "invented AUM numbers without direct support",
                    "exact AUM stated as fact when public evidence does not confirm it",
                    "numeric AUM without calling it an estimate based on public references",
                    "repeating the same uncertainty opener every time",
                    "a established",
                ],
            },
            "role_fit": {
                "good_example": "Good alignment on lane and remit. Direct evidence of team scale and channel breadth remains unverified. Screening priority: how much team leadership and channel depth sit behind the title?",
                "avoid": [
                    "demonstrated team leadership unless explicitly supported",
                    "deep Australian market networks unless explicitly supported",
                    "proven track record claims without direct evidence",
                    "strong fit",
                    "well beyond the role",
                    "established profile",
                    "keeps this profile in frame as a",
                    "public evidence supports a recruiter-relevant remit",
                    "the main commercial gap is",
                    "the first screening question is",
                ],
            },
            "career_narrative": {
                "good_example": "Holds an institutional distribution remit at the named employer with visible ANZ scope. The profile is commercially relevant because the lane is close to the mandate. Broader team scale and network depth remain unverified and should be tested early.",
                "avoid": [
                    "built the function",
                    "transformed",
                    "led the build-out",
                    "established track record",
                    "positioned as",
                    "scaled the function",
                    "drove growth",
                    "this profile looks like",
                    "should be handled as strong evidence until tested",
                    "currently sitting in a",
                    "from a recruiter lens, the relevance comes from",
                    "that puts the profile close to the",
                ],
            },
            "outreach_hook": {
                "avoid": [
                    "would welcome a conversation",
                    "caught our attention",
                    "worth discussing",
                    "current search mandate",
                    "your background appears relevant",
                    "credible angle",
                    "adjacent transferability",
                    "comparable remit",
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
