# Design Note

I chose a three-stage architecture because PPP’s task is not just “generate some text from a CSV”; it is closer to a recruiter workflow where evidence quality matters as much as output shape. In a single prompt, research, reasoning, and formatting tend to get mixed together, which makes hallucination harder to control. The current system therefore separates those responsibilities explicitly.

Stage 1 is a research adapter, not a Claude enrichment pass. It gathers public evidence and normalizes it into a structured research package with three layers: `sources`, `claims`, and `verification`. Each claim is labelled with a verification status (`verified`, `strongly_inferred`, or `uncertain`), a confidence level, and the recruiter-facing output fields that the claim is allowed to support. This is a better fit for executive search than a loose prose enrichment blob because it makes the boundary between “observed,” “directionally inferred,” and “still unverified” explicit in the data model.

Stage 2 uses Claude only for reasoning over that bounded package and formatting the final recruiter-facing JSON. Claude does not perform the research loop itself. Instead, it receives the candidate record, the structured `research_package`, the role specification, and strict instructions about how each claim type may be used in the output. Verified claims can be stated directly, strongly inferred claims can guide cautious directional language, and uncertain claims must not be promoted into settled facts.

Stage 3 is a boundary-aware QA layer. I enforced strict schema validation and post-generation checks because recruiter-facing outputs should fail loudly when they are malformed, generic, or overconfident, rather than quietly passing through as “close enough.” The QA layer still checks formatting rules like sentence count and placeholder text, but it now also checks whether the final writing stays inside source, claim, and confidence boundaries. That includes whether role-fit justification is anchored to supported claims, whether AUM language exceeds verified evidence, whether tenure aligns to the underlying tenure claim, and whether uncertainty is still visible when the research package is incomplete.

The system is intentionally designed to handle incomplete candidate inputs. Where public evidence is partial or ambiguous, the output surfaces this uncertainty explicitly rather than masking it behind recruiter-style confidence.

## Uncertainty Framework

The current pipeline uses three internal identity states:

- `verified_match`: public evidence is strong enough to treat the profile as the intended person
- `possible_match`: the commercial shape looks directionally right, but identity still needs confirmation
- `not_verified`: the task input may still be commercially plausible, but it is not yet safe to action as a confirmed target

Those states feed deterministic downgrade rules rather than a separate LLM pass. In practice that means:

- `career_narrative` can still explain why a profile is commercially relevant, but it must also state when the identity is only possible or not verified
- `mobility_signal` prioritises observed chronology, then inferred trajectory, then heuristic role-shape cues, before falling back to generic uncertainty
- `role_fit.justification` opens with a consultant-style category such as `credible shortlist candidate`, `possible match pending verification`, `adjacent step-up conversation`, or `market map only`
- `outreach_hook` tone changes with confidence: direct for shortlist candidates, exploratory for possible matches, and light-touch for market-map inputs

This distinction matters because executive search consultants often need to preserve commercial judgment even when public identity data is incomplete. The system therefore separates "commercially interesting" from "safe to action" rather than collapsing both into a single confidence number.

This matters because `output.json` is treated as the final deliverable, not a preview artifact. The goal is not merely to produce convincing prose; it is to produce submission-safe recruiter output that remains auditable against the research package that generated it.

With more time, I would improve the system in three specific ways. First, I would tighten Stage 2 wording calibration even further, especially for firm context, AUM language, and cases where uncertainty gets compressed too aggressively into smooth recruiter prose. Second, I would make the QA layer more semantic and field-specific so support checks are less heuristic, particularly for `firm_aum_context`, `mobility_signal.rationale`, and `role_fit.justification`. Third, I would add a citation-aware reviewer layer so a consultant could see which claim in the briefing came from which source and quickly understand why a candidate passed or failed QA.

If I joined PPP, one additional automation I would build is a “search mandate to longlist triage assistant.” In practice, a search does not stop at candidate briefing; the harder operational problem is deciding who deserves immediate partner attention, who should stay in the broader map, and who is adjacent but not ready. I would build a workflow that takes a live mandate, applies role-specific criteria, scores candidate evidence quality, flags missing proof points, and produces a prioritised outreach queue for consultants. For PPP’s funds management distribution work, that would be especially useful because channel relevance, institutional versus wholesale depth, leadership scope, and market credibility are often unevenly visible in public profiles. A triage assistant would help consultants move faster without giving up judgment, and it would fit naturally with the candidate briefing system delivered here.
