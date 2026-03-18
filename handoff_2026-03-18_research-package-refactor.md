# Codex Handoff: Research Package Refactor

Date:
- 2026-03-18

Repo:
- `/Users/fangxixix/CursorProject/exec-search-ai-workflow-ppp-task`

## 1. What Changed

This window completed a structural refactor of the PPP candidate briefing pipeline.

The previous mental model was partially mixed:
- Stage 1 looked like "enrichment"
- some old code paths still treated Claude as part of the enrichment/tool stage
- QA mainly checked style + some business rules, but not whether output crossed evidence boundaries

The new mental model is now:

`Stage 1 = research adapter`
`Stage 2 = Claude reasoning + formatting`
`Stage 3 = boundary-aware QA`

In other words:
- Stage 1 now collects and normalizes public evidence
- Stage 2 now receives a structured research package instead of mixed prose
- QA now checks whether output stays inside source/claim/confidence boundaries

## 2. Core Architectural Decision

The team aligned on three principles:

1. Stage 1 is a research adapter, not "Claude enrichment"
2. `output.json` is a submission-safe final deliverable, not a preview artifact
3. internal enrichment data should be explicitly layered into:
   - `sources`
   - `claims`
   - `verification/confidence`

This refactor implemented those principles in code.

## 3. Main Code Changes

### A. Enrichment schema was redefined

Primary file:
- `/Users/fangxixix/CursorProject/exec-search-ai-workflow-ppp-task/app/ppp/enrichment.py`

`CandidateEnrichmentResult` is now an explicit research package.

It now contains:
- candidate identity fields
- `sources: list[LookupSource]`
- `claims: list[ResearchClaim]`
- `verification: VerificationSummary`

Important supporting concepts:
- `ResearchClaim.verification_status`
  - `verified`
  - `strongly_inferred`
  - `uncertain`
- `ResearchClaim.confidence`
  - `high`
  - `medium`
  - `low`
- `ResearchClaim.supports_output_fields`
  This is important because QA now uses it to determine whether output fields are supported by the research package.

Compatibility notes:
- `inferred_tenure_years` still exists as a property on the enrichment object
- `firm_aum_context` still exists as a property on the enrichment object
- these are now derived from the structured claim set rather than stored as top-level freeform fields

### B. Stage 2 input contract was tightened

Primary file:
- `/Users/fangxixix/CursorProject/exec-search-ai-workflow-ppp-task/app/ppp/prompts.py`

The generation prompt now tells Claude to consume:
- `research_package`
- `role_spec`
- strict output rules

It explicitly instructs Claude how to handle:
- `verified` claims
- `strongly_inferred` claims
- `uncertain` claims

Old mixed input style was removed:
- no more relying on a loose prose-style enrichment blob as the main reasoning substrate

### C. QA was upgraded to boundary-aware checks

Primary file:
- `/Users/fangxixix/CursorProject/exec-search-ai-workflow-ppp-task/app/ppp/qa.py`

QA still checks:
- sentence counts
- placeholder text
- schema-related conditions

But it now also checks boundary logic:
- whether `role_fit.justification` is anchored to supported claims
- whether `firm_aum_context` exceeds verified/supportable evidence
- whether `tenure_years` aligns with tenure claims
- whether uncertainty in the research package is reflected in the final writing
- whether certain output fields appear to drift beyond the claim set

This is the biggest conceptual change in QA.

### D. Old enrichment/tool-use path was removed

Primary files:
- `/Users/fangxixix/CursorProject/exec-search-ai-workflow-ppp-task/app/ppp/pipeline.py`
- `/Users/fangxixix/CursorProject/exec-search-ai-workflow-ppp-task/app/llm/anthropic_client.py`
- `/Users/fangxixix/CursorProject/exec-search-ai-workflow-ppp-task/app/ppp/prompts.py`

What was removed:
- old Claude tool-use enrichment roundtrip
- old enrichment prompt functions
- old unused tool JSON helper methods
- old unused tool-response plumbing

Current behavior:
- Stage 1 directly runs the research adapter
- Stage 2 uses Claude only for final candidate brief generation

This removed a major source of redundancy and JSON-convergence fragility.

## 4. Current Research Adapter Behavior

Primary file:
- `/Users/fangxixix/CursorProject/exec-search-ai-workflow-ppp-task/app/ppp/research.py`

Current live research path:
- provider: Tavily
- mode support:
  - `fixture`
  - `live`
  - `auto`

Current filtering improvements already in place:
- stronger result scoring
- some source-type prioritization
- filtering for same-name wrong-employer cases
- exclusion of clearly low-value hosts

Important real-world observation from this window:
- when candidate data is realistic and LinkedIn URLs are real, live research quality improves materially
- however, some candidates may still fall back if filtered live results are too weak or ambiguous

## 5. Files Most Changed

High-impact files:
- `/Users/fangxixix/CursorProject/exec-search-ai-workflow-ppp-task/app/ppp/enrichment.py`
- `/Users/fangxixix/CursorProject/exec-search-ai-workflow-ppp-task/app/ppp/prompts.py`
- `/Users/fangxixix/CursorProject/exec-search-ai-workflow-ppp-task/app/ppp/qa.py`
- `/Users/fangxixix/CursorProject/exec-search-ai-workflow-ppp-task/app/ppp/pipeline.py`
- `/Users/fangxixix/CursorProject/exec-search-ai-workflow-ppp-task/app/ppp/research.py`
- `/Users/fangxixix/CursorProject/exec-search-ai-workflow-ppp-task/tests/test_ppp_pipeline.py`

Also touched:
- `/Users/fangxixix/CursorProject/exec-search-ai-workflow-ppp-task/app/llm/anthropic_client.py`
- `/Users/fangxixix/CursorProject/exec-search-ai-workflow-ppp-task/docs/ppp_implementation.md`

## 6. Verification Status

Checks run successfully after refactor:
- `pytest -q tests/test_ppp_pipeline.py`
- `ruff check --no-cache app/ppp tests/test_ppp_pipeline.py`
- `mypy app/ppp tests/test_ppp_pipeline.py`

Status at end of window:
- tests passed
- lint passed
- mypy passed

## 7. Real Execution Observations

Using realistic public candidate data with real LinkedIn URLs:
- live research hit rate improved significantly
- the previous redundant enrichment fallback issue is gone
- pipeline can now reach live enrichment directly without the removed tool-use failure loop

But one practical issue still remains in real runs:
- final output can still fail QA if Claude overstates AUM context or similar business-sensitive fields
- this is now being caught correctly by the stricter boundary-aware QA

That means:
- the new contract is doing its job
- the next work should likely focus on improving Stage 2 adherence and wording calibration, not weakening QA

## 8. Safe Cleanup Already Done

This window also removed high-certainty dead code:
- obsolete enrichment prompt functions
- obsolete tool-use roundtrip logic
- obsolete helper methods that only existed for the removed path
- unused `LLMResponse` in the PPP-relevant Claude client path

## 9. Important Remaining Work

### A. Update docs to match the new architecture

The code now reflects:
- research adapter
- structured research package
- boundary-aware QA

But top-level documentation is not fully synchronized yet.

Likely next docs to update:
- `/Users/fangxixix/CursorProject/exec-search-ai-workflow-ppp-task/README.md`
- `/Users/fangxixix/CursorProject/exec-search-ai-workflow-ppp-task/design_note.md`

These should be rewritten to describe:
- explicit source/claim/verification layering
- Stage 2 consuming a structured research package
- QA guarding evidence boundaries

### B. Tighten Stage 2 generation behavior

The prompt contract is stronger now, but it is still possible for Claude to:
- write overly specific AUM language
- compress uncertainty too aggressively
- produce recruiter-usable prose that is stylistically good but slightly too assertive

The next window should probably improve:
- wording calibration
- claim-to-output grounding
- possibly explicit structured "allowed facts" summaries passed into the final prompt

### C. Make QA more semantically precise

Current boundary-aware QA is materially better than before, but still heuristic.

Possible next step:
- move from token-overlap style support checks toward richer field-specific claim matching
- especially for:
  - `firm_aum_context`
  - `mobility_signal.rationale`
  - `role_fit.justification`

### D. Consider exposing the research package in intermediate review tooling

Because the enrichment object is now a proper research package, it may be worth improving intermediate human review UX later.

## 10. New / Notable Data Files

Added during this broader workstream:
- `/Users/fangxixix/CursorProject/exec-search-ai-workflow-ppp-task/data/ppp/candidates_alt_institutional.csv`
- `/Users/fangxixix/CursorProject/exec-search-ai-workflow-ppp-task/data/ppp/candidates_alt_wealth.csv`
- `/Users/fangxixix/CursorProject/exec-search-ai-workflow-ppp-task/data/ppp/candidates_realistic_public.csv`

Notes:
- `candidates_realistic_public.csv` was manually updated to use real LinkedIn profile URLs
- this improved live-research behavior

## 11. Working Assumptions For The Next Window

Assume the following are now true unless changed intentionally:

1. Stage 1 should remain deterministic research normalization, not Claude-led enrichment
2. Stage 2 should only consume structured research data
3. QA should block outputs that cross claim/confidence boundaries
4. final `output.json` should stay submission-safe even if that means failing hard on overclaim

## 12. Recommended Next Step

Best next move:

1. update `README.md` and `design_note.md` so they describe the new architecture accurately
2. run the realistic public CSV through the refactored pipeline again
3. inspect any QA failures as prompt-calibration issues rather than schema issues
4. tighten Stage 2 wording around firm context / AUM / uncertainty

## 13. One-Sentence Mental Model

This PPP path is now a research-adapter-driven candidate briefing system where structured public evidence is collected first, then Claude writes the final recruiter-facing JSON under explicit claim and confidence boundaries.
