# PPP AI Assessment Submission

Claude-powered candidate briefing agent for executive search, built for the Platinum Pacific Partners AI & Automation Intern task.

This submission focuses on a real PPP-style workflow: take a `candidates.csv`, enrich each profile with structured public-research evidence, generate recruiter-usable candidate briefings in strict JSON, and run a final QA layer before submission.

## What It Does

- Accepts the PPP candidate CSV input format with exactly 5 rows.
- Runs a two-stage pipeline:
  1. `candidate_public_profile_lookup` enrichment
  2. Claude-based structured briefing generation
- Produces `output.json` matching the required schema.
- Saves intermediate enrichment artifacts for review and debugging.
- Runs schema, content, and business-rule QA before treating the output as complete.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

## Example Command

```bash
python3 scripts/run_ppp_task.py \
  --input data/ppp/candidates.csv \
  --output data/ppp/output.json \
  --role-spec data/ppp/role_spec.json \
  --research-fixtures data/ppp/research_fixtures.json \
  --intermediate-dir data/ppp/intermediate \
  --model claude-sonnet-4-5
```

Optional non-technical runner:

```bash
streamlit run app/ui/ppp_task_app.py
```

Pre-submission validation:

```bash
python3 scripts/run_ppp_task.py \
  --input data/ppp/candidates.csv \
  --output data/ppp/output.json \
  --intermediate-dir data/ppp/intermediate \
  --validate-only
```

## Output Format

Primary deliverable:

- `data/ppp/output.json`

Intermediate artifacts:

- `data/ppp/intermediate/candidate_1_enriched.json` to `candidate_5_enriched.json`
- `data/ppp/intermediate/qa_report.json`
- `data/ppp/intermediate/candidate_*_error.json` on candidate-level failure

The final JSON contains:

- `candidate_id`
- `full_name`
- `current_role`
- `career_narrative`
- `experience_tags`
- `firm_aum_context`
- `mobility_signal`
- `role_fit`
- `outreach_hook`

## Architecture Overview

The pipeline uses a two-stage pattern because it is more reliable than asking one prompt to both research and format:

1. **Enrichment / tool stage**
   The system gathers structured evidence for each candidate: verified snippets, tenure clues, channel indicators, firm context, missing fields, uncertain fields, sources, and confidence notes.
2. **Structured generation stage**
   Claude receives the candidate input, enrichment payload, role spec, and strict output requirements, then returns a single candidate JSON object.
3. **Validation and QA stage**
   The system repairs minor formatting issues, validates the schema with Pydantic, and runs content and business-rule checks before accepting the bundle.

This makes the output more robust, auditable, and recruiter-usable.

## Tool Design

Implemented tool:

- `candidate_public_profile_lookup`

Input:

- `full_name`
- `current_employer`
- `current_title`
- `linkedin_url`

Output:

- verified public snippets
- inferred tenure clues
- likely channel / experience evidence
- firm / AUM context clues
- missing / uncertain fields
- source list
- confidence notes

Current mode:

- The tool is implemented as a fixture-backed public-research runner using `data/ppp/research_fixtures.json`.
- This keeps the interface realistic while preventing unsupported claims in an environment where live public-web verification is not always available.
- The code is intentionally structured so the fixture-backed runner can be replaced by a live web research connector later without rewriting the rest of the pipeline.

## Validation And QA

This submission does more than generate JSON. It checks whether the output looks safe and usable for an actual recruiter workflow.

Checks include:

- JSON parseability and schema compliance
- exactly 5 candidates
- score bounds and required fields
- `career_narrative` sentence count
- `outreach_hook` single-sentence rule
- placeholder text detection
- evidence-aware `role_fit.justification`
- cautious AUM and tenure language
- uncertainty signalling when evidence is incomplete

## Known Limitations

- The enrichment tool currently uses controlled fixtures rather than a live public-web connector.
- LinkedIn URLs in the PPP brief are placeholders, so final public-profile verification still needs to happen against real sources.
- Source citation is captured in enrichment artifacts, but not yet surfaced directly in the final `output.json` schema.
- Confidence logic is rule-based and conservative rather than statistically calibrated.

## What I Would Build Next For PPP

- A live public research connector that searches company pages, bios, and public profiles with source capture.
- A citation-aware recruiter view that shows which sentence in the briefing is supported by which source.
- A reviewer feedback loop so consultants can mark a briefing as accurate, weak, or misleading and improve future generations.
- A candidate brief QA + outreach prioritisation layer that ranks who should actually be called first based on fit, evidence confidence, and mobility.

## Existing Background

This repository evolved from my broader executive-search workflow system, which models role intake, candidate search, explainable ranking, briefing, approval, and review as a project-centric workflow rather than a chatbot. For this PPP submission, I narrowed that broader system into a task-focused candidate briefing agent so the assessment deliverable stays clear and directly runnable.
