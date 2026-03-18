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

For the new non-technical PPP briefing UI:

```bash
streamlit run streamlit_app.py
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

## Streamlit UI Usage

`streamlit_app.py` is a direct UI wrapper around the existing PPP pipeline.

Important:

- You do **not** need to start the FastAPI backend before running `streamlit_app.py`.
- The Streamlit page imports and calls `app.ppp.run_ppp_pipeline(...)` directly inside the same Python process.
- Only start the backend separately if you want to use the repository's API routes for other workflows. It is not required for the PPP briefing UI.

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the Streamlit app

From the repository root:

```bash
streamlit run streamlit_app.py
```

After startup, Streamlit will print a local URL such as `http://localhost:8501`. Open that URL in your browser.

### 3. Prepare what you need before clicking Generate

- An Anthropic API key
- A candidate CSV file in PPP format
- Exactly 5 candidate rows

Required CSV columns:

- `full_name`
- `current_employer`
- `current_title`
- `linkedin_url`

### 4. Use the UI

1. Open the app in the browser.
2. In the left sidebar, enter your Anthropic API key in **Anthropic API Key**.
3. In the main area, upload the PPP candidate CSV with **Upload candidate CSV**.
4. Click **🚀 Generate Briefings**.
5. Wait while the existing pipeline runs. The page will show a loading spinner during processing.
6. Review the generated candidate briefings:
   - each candidate appears in an expandable panel
   - the panel title shows `full_name - current_role.title @ current_role.employer`
   - the top row shows `Role Fit Score` and `Mobility Score`
   - the highlighted box shows the recruiter-ready `outreach_hook`
   - the narrative, fit justification, tags, AUM context, and tenure note appear below
7. Click **Download output.json** at the bottom to save the generated result bundle.

### 5. What the app does behind the scenes

When you click **Generate Briefings**, the UI:

1. saves the uploaded CSV to a temporary file
2. injects the sidebar API key into the current process environment
3. calls the existing PPP pipeline directly:

```python
run_ppp_pipeline(
    input_path=...,
    output_path=...,
    role_spec_path="data/ppp/role_spec.json",
    model="claude-sonnet-4-5",
    intermediate_dir=...,
    research_fixture_path="data/ppp/research_fixtures.json",
    research_mode="fixture",
)
```

4. renders the returned JSON in a recruiter-friendly layout

No PPP pipeline business logic is reimplemented in the Streamlit layer.

### 6. Common troubleshooting

- `ANTHROPIC_API_KEY is missing`
  Enter the API key in the Streamlit sidebar, then click generate again.
- `CSV format error: missing required columns`
  Check that the uploaded CSV contains all four required header names exactly.
- `Candidate count error: expected 5 candidates`
  The PPP pipeline only accepts CSV files with exactly 5 rows.
- The app opens but generation fails
  Make sure your virtual environment is activated and dependencies from `requirements.txt` are installed.
- You are wondering whether to run `uvicorn` or `python app/main.py`
  You do not need either of those for `streamlit_app.py`.

Pre-submission validation:

```bash
python3 scripts/run_ppp_task.py \
  --input data/ppp/candidates.csv \
  --output data/ppp/output.json \
  --intermediate-dir data/ppp/intermediate \
  --validate-only
```

## How Uncertainty Is Handled

The PPP task explicitly allows partial, noisy, or illustrative candidate inputs. This implementation therefore treats uncertainty as a first-class product feature rather than as an exception path.

The pipeline uses three identity states internally:

- `verified_match`: public evidence supports an exact or near-exact profile match
- `possible_match`: public evidence is directionally consistent, but identity still needs confirmation
- `not_verified`: the task input remains commercially plausible, but public evidence does not yet support treating it as an action-ready target

Those states do not change the required output schema, but they do change how the system writes and scores each candidate:

- verified profiles can be presented more directly
- possible matches can still score meaningfully when the commercial fit is strong, but the wording stays caveated
- not verified profiles are kept low-scoring and framed as market-map inputs rather than confirmed outreach targets

Field-specific downgrade rules keep the output submission-safe:

- `firm_aum_context` uses numeric AUM only when it is explicitly framed as estimated and based on public references
- `mobility_signal` separates visible chronology from missing move-readiness evidence
- `role_fit.justification` distinguishes commercial relevance from verification confidence
- `outreach_hook` shifts tone from direct to exploratory to light-touch depending on evidence quality

Outputs explicitly prefer phrases such as `unable to verify`, `appears to`, `based on public evidence`, and `subject to verification`.

## Output Format

Primary deliverable:

- `data/ppp/output.json`

Intermediate artifacts:

- `data/ppp/intermediate/candidate_1_enriched.json` to `candidate_5_enriched.json`
- `data/ppp/intermediate/qa_report.json`
- `data/ppp/intermediate/candidate_*_error.json` on candidate-level failure
- `data/ppp/intermediate/run_report.json` with delivery status, identity resolution, verification posture, and inclusion reasoning

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

1. Launch the legacy reviewer with `streamlit run app/ui/ppp_task_app.py` if you want the older validation-oriented control panel.
2. For the new non-technical briefing experience, launch `streamlit run streamlit_app.py`.
3. If you use the legacy reviewer, set `Candidate CSV` to `data/ppp/candidates_realistic_public.csv`.
4. Set `Research Mode` to `auto` unless you specifically want to force live-only behavior.
5. Set `Intermediate Directory` to `data/ppp/intermediate_realistic_public`.
6. Run the task, then immediately run **Validate Existing Output**.
7. Confirm `qa_report.json` shows `passed: true` before treating the bundle as usable.

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
