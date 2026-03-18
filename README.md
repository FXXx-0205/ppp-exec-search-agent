# PPP AI Assessment Submission

Claude-powered candidate briefing agent for executive search, built for the Platinum Pacific Partners AI & Automation Intern task.

This submission focuses on a real PPP-style workflow: take a `candidates.csv`, collect and normalize structured public-research evidence for each profile, generate recruiter-usable candidate briefings in strict JSON, and block outputs that cross evidence boundaries before submission.

It is intentionally designed to handle partial or unverified candidate inputs. When public evidence is incomplete, the system prefers explicit uncertainty over fabricated precision and treats unverified AUM, tenure, mobility, and remit details as recruiter-facing caveats rather than hidden model assumptions.

## What It Does

- Accepts the PPP candidate CSV input format with exactly 5 rows.
- Runs a three-stage pipeline:
  1. research adapter
  2. Claude-based reasoning and recruiter-facing formatting
  3. boundary-aware QA
- Produces `output.json` as the submission-safe final deliverable.
- Saves intermediate research-package artifacts for review and debugging.
- Runs schema, content, business-rule, and evidence-boundary QA before treating the output as complete.

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

Optional for real public-web research:

```bash
export PPP_RESEARCH_MODE=auto
export TAVILY_API_KEY=your_tavily_key_here
```

For the realistic public-data rerun used during manual review:

```bash
streamlit run app/ui/ppp_task_app.py
```

## Example Command

```bash
python3 scripts/run_ppp_task.py \
  --input data/ppp/candidates.csv \
  --output data/ppp/output.json \
  --role-spec data/ppp/role_spec.json \
  --research-fixtures data/ppp/research_fixtures.json \
  --research-mode auto \
  --intermediate-dir data/ppp/intermediate \
  --model claude-sonnet-4-5
```

Realistic public-data example:

```bash
python3 scripts/run_ppp_task.py \
  --input data/ppp/candidates_realistic_public.csv \
  --output data/ppp/output.json \
  --role-spec data/ppp/role_spec.json \
  --research-fixtures data/ppp/research_fixtures.json \
  --research-mode auto \
  --intermediate-dir data/ppp/intermediate_realistic_public \
  --model claude-sonnet-4-5
```

Optional non-technical runner:

```bash
streamlit run app/ui/ppp_task_app.py
```

Streamlit runner flow:

1. Start the app with the command above.
2. In **Step 1: Configure Inputs**, set:
   - `Candidate CSV`: `data/ppp/candidates_realistic_public.csv`
   - `Research Mode`: `auto` or `live`
   - `Intermediate Directory`: `data/ppp/intermediate_realistic_public`
3. Keep the default role spec, output path, and fixtures unless you are intentionally testing alternates.
4. Review the **Current Paths** block so you can see exactly which files will be read and written.
5. Click **Run PPP Task**.
6. Wait for the app to generate:
   - `data/ppp/output.json`
   - `data/ppp/intermediate_realistic_public/candidate_*_enriched.json`
   - `data/ppp/intermediate_realistic_public/qa_report.json`
7. Review the **Output Preview** in the app.
8. Click **Validate Existing Output** to rerun validation without regenerating the briefings.
9. Review the **QA Report** section in the app and confirm it passes before submission.

What a non-technical reviewer should expect to see:

- one button to generate the deliverable
- one button to validate the latest deliverable
- a preview of the final JSON
- a preview of the QA report
- no need to edit Python files or use the terminal after Streamlit is running

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

The pipeline now uses a three-stage pattern with an explicit research package between collection and writing:

1. **Stage 1: Research adapter**
   The system runs `candidate_public_profile_lookup` and normalizes public evidence into a structured research package. This stage is deterministic and is not a Claude enrichment pass.
2. **Stage 2: Claude reasoning + formatting**
   Claude receives the candidate input, `research_package`, role spec, and strict output rules, then writes a single candidate JSON object. It is instructed to treat `verified`, `strongly_inferred`, and `uncertain` claims differently, and to say explicitly when a point cannot be verified from public sources.
3. **Stage 3: Boundary-aware QA**
   The system validates the schema and checks whether the final writing stays within source, claim, and confidence boundaries before accepting the bundle.

This makes the output more robust, auditable, recruiter-usable, and safer for submission.

## Research Package Design

The Stage 1 output is an explicit research package rather than a loose prose enrichment blob.

Core layers:

- `sources`: normalized public lookup sources
- `claims`: recruiter-relevant statements with verification status and confidence
- `verification`: summary of evidence completeness and unresolved gaps

Each claim carries output-boundary metadata:

- `verification_status`: `verified`, `strongly_inferred`, or `uncertain`
- `confidence`: `high`, `medium`, or `low`
- `supports_output_fields`: which recruiter-facing fields that claim is allowed to support

Compatibility helpers such as `inferred_tenure_years` and `firm_aum_context` still exist on the enrichment object, but they are now derived from the structured claim set rather than stored as freeform top-level fields.

Implemented research adapter:

- `candidate_public_profile_lookup`

Input:

- `full_name`
- `current_employer`
- `current_title`
- `linkedin_url`

Output:

- source records
- structured research claims
- verification summary
- derived tenure and firm-context helpers for downstream compatibility

Current mode:

- `fixture`: uses `data/ppp/research_fixtures.json` only.
- `live`: uses a real public-web research connector and requires `TAVILY_API_KEY`.
- `auto`: tries live public-web research first, then falls back to fixtures if the provider is unavailable or returns no usable evidence.
- This keeps the interface realistic while preserving a conservative fallback path when live public-web verification is unavailable.

Current live provider:

- Tavily

## Validation And QA

This submission does more than generate JSON. It checks whether the output is structurally correct and whether the writing stays inside the evidence the system actually collected.

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
- explicit “unable to verify” style wording for unverifiable firm context and other bounded inferences
- field-level claim boundary checks against the research package

In practice, this means `output.json` is treated as the final deliverable, not as a preview artifact. If the generated writing overclaims beyond verified or supportable evidence, QA should fail loudly rather than quietly letting the JSON pass.

## Manual Review With Streamlit

For the current architecture, manual review is not just a UI check. It is a boundary check.

Recommended review flow for `data/ppp/candidates_realistic_public.csv`:

1. Launch Streamlit with `streamlit run app/ui/ppp_task_app.py`.
2. Set `Candidate CSV` to `data/ppp/candidates_realistic_public.csv`.
3. Set `Research Mode` to `auto` unless you specifically want to force live-only behavior.
4. Set `Intermediate Directory` to `data/ppp/intermediate_realistic_public`.
5. Run the task, then immediately run **Validate Existing Output**.
6. Confirm `qa_report.json` shows `passed: true` before treating the bundle as usable.

What to inspect manually:

- `output.json` should read like confident recruiter prose without sounding more certain than the evidence supports.
- `firm_aum_context` should avoid precise AUM statements unless the intermediate research package contains a verified supporting claim, and should say explicitly when exact AUM cannot be verified from public sources.
- `current_role.tenure_years` should align with the tenure claim in the enrichment artifact, or the prose should signal uncertainty.
- `mobility_signal.rationale` should separate observed chronology from absent move-readiness evidence and should end in a clear uncertainty or follow-up cue.
- `role_fit.justification` should be anchored to employer, title, channel exposure, or other supported claims rather than generic recruiter inference, and should distinguish supported relevance from unverified requirement coverage.
- If a candidate fails QA, treat that as a Stage 2 wording-calibration issue first, not a reason to weaken QA.

## Known Limitations

- Source citation is captured in intermediate research artifacts, but not surfaced directly in the final `output.json` schema.
- Stage 2 can still occasionally compress uncertainty too aggressively, especially around firm context or business-sensitive fields, even though the current prompt and stabilizers now push much harder toward explicit “unable to verify” wording.
- The current QA boundary checks are materially better than before, but some support checks are still heuristic rather than fully semantic.
- Confidence logic is rule-based and conservative rather than statistically calibrated.

## What I Would Build Next For PPP

- Stronger Stage 2 wording calibration so final prose stays closer to claim and confidence boundaries.
- More semantic field-specific QA for `firm_aum_context`, `mobility_signal.rationale`, and `role_fit.justification`.
- A citation-aware recruiter view that shows which sentence in the briefing is supported by which source.
- A reviewer feedback loop so consultants can mark a briefing as accurate, weak, or misleading and improve future generations.

## Existing Background

This repository evolved from my broader executive-search workflow system, which models role intake, candidate search, explainable ranking, briefing, approval, and review as a project-centric workflow rather than a chatbot. For this PPP submission, I narrowed that broader system into a task-focused candidate briefing agent so the assessment deliverable stays clear and directly runnable.
