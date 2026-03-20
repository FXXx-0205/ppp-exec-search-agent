# PPP Executive-Search Briefing Agent

Claude-powered candidate-briefing workflow for the Platinum Pacific Partners practical task.

The repository is built around the actual PPP deliverable: take a five-row `candidates.csv`, enrich each profile with structured public-profile evidence, and export a recruiter-usable `output.json` with exactly five PPP-schema candidate briefs. The writing is tuned for executive-search judgment rather than generic summarisation, so the output is designed to help a consultant distinguish shortlist-style targets, adjacent screens, and mapping leads while staying inside public-evidence boundaries.

## What This Submission Does

- Accepts the PPP candidate CSV format with exactly 5 candidates.
- Uses the Anthropic Claude API with real API-level tool calling.
- Follows a two-step generation pattern:
  1. research and evidence normalisation through Claude tool use
  2. final strict CandidateBrief generation with no tools exposed
- Validates the final bundle against the PPP output schema before export.
- Exports `data/ppp/output.json` as the primary submission artifact.
- Produces QA and run artifacts in `data/ppp/intermediate/`.

## Run It

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the Anthropic key:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

Optional for live public-web research:

```bash
export TAVILY_API_KEY=your_tavily_key_here
```

### Reproducible PPP Run

This is the main command for a reproducible run using the included research fixtures plus live Claude generation:

```bash
python3 scripts/run_ppp_task.py \
  --input data/ppp/candidates.csv \
  --output data/ppp/output.json \
  --role-spec data/ppp/role_spec.json \
  --research-fixtures data/ppp/research_fixtures.json \
  --research-mode fixture \
  --intermediate-dir data/ppp/intermediate \
  --model claude-sonnet-4-5
```

### Validate Existing Output

```bash
python3 scripts/run_ppp_task.py \
  --input data/ppp/candidates.csv \
  --output data/ppp/output.json \
  --intermediate-dir data/ppp/intermediate \
  --validate-only
```

### Non-Technical UI

For a non-technical user path, the repo also includes a Streamlit wrapper around the same pipeline:

```bash
streamlit run streamlit_app.py
```

The UI does not reimplement the business logic. It calls the same PPP pipeline and still writes the same structured `output.json`.

By default, the UI:

- saves uploaded CSVs to `data/ppp/uploaded_candidates.csv`
- writes the exported briefing bundle to `data/ppp/output.json`
- writes QA and run artifacts to `data/ppp/intermediate/`
- uses fixture mode as the safest demo path unless the user deliberately switches to live public-web research

## Architecture

The pipeline is intentionally narrow and acceptance-driven.

### 1. Research and Evidence Normalisation

The system first builds a candidate research package from fixture or live public-web lookup. Claude is then called with a real Anthropic tool:

- `messages.create(..., tools=[...])`
- Claude is instructed to call `normalize_candidate_evidence`
- Python executes the tool
- the tool result is returned as a `tool_result`
- Claude completes the research phase by returning a normalized evidence object

This phase does not produce the final candidate brief. Its job is to stabilize evidence quality and confidence handling before writing.

### 2. Final Candidate Brief Generation

The second Claude call does not expose tools. It takes:

- candidate identity
- normalized evidence
- recruiter signals
- role spec

and returns exactly one `CandidateBrief` JSON object. This separation keeps enrichment, judgment, and strict formatting from bleeding into one another.

### 3. Validation and QA

Before export, the system performs:

- candidate-level parsing and validation
- strict PPP schema validation
- bundle-level QA for quality, repetition, and evidence discipline

If the bundle is malformed, export is blocked. Bundle-level QA also flags repetition and evidence-discipline issues before submission.

## Output Contract

`data/ppp/output.json` is the primary deliverable.

It contains:

- a top-level `candidates` array
- exactly 5 candidate objects
- only the PPP-required fields

Each candidate includes:

- `candidate_id`
- `full_name`
- `current_role`
- `career_narrative`
- `experience_tags`
- `firm_aum_context`
- `mobility_signal`
- `role_fit`
- `outreach_hook`

Schema rules that are enforced in code:

- exactly 5 candidates, not fewer and not more
- `current_role.tenure_years` is always numeric
- `mobility_signal.score` is an integer from 1 to 5
- `role_fit.score` is an integer from 1 to 10
- extra fields are rejected
- missing required fields are rejected

## Evidence and Confidence Handling

The system is designed for imperfect public inputs. It does not assume every candidate row can be verified cleanly.

Internally, evidence is normalized into confidence-aware states that drive how the writing behaves. In practice:

- strong or likely matches can be written more directly
- adjacent but imperfectly supported profiles can still be screened when the commercial shape is relevant
- low-confidence identity cases are treated conservatively

That shows up in the exported output in a few concrete ways:

- `current_role.tenure_years` is always numeric for schema compliance
- low-confidence identity cases use conservative tenure handling, including `0.0` where current-role chronology is not reliable enough to trust
- mobility wording does not overclaim move readiness when chronology or identity is weak
- `firm_aum_context` prioritises useful qualitative firm framing over empty AUM disclaimers
- `role_fit.justification` is written to support recruiter prioritisation, not just résumé paraphrase

## Current Output Style

The current output is tuned to help a PPP consultant answer:

- who looks like a first-wave shortlist target
- who is an adjacent screen worth testing
- who is mainly a market-map lead
- what the main fit is
- what the main gap is
- what should be tested first on a call

The final artifact therefore aims to read like a recruiter note, not like a generic LLM summary.

## Intermediate Artifacts

Useful generated files include:

- `data/ppp/output.json`
- `data/ppp/intermediate/qa_report.json`
- `data/ppp/intermediate/run_report.json`
- `data/ppp/intermediate/candidate_*_enriched.json`

The current checked bundle reports:

- `run_report.json`: 5 successful candidates, 0 failures
- `qa_report.json`: `passed: true`

## Known Limitations

- Public evidence is uneven. The system can frame likely remit shape and firm context, but it does not claim hidden data such as exact team size, direct reports, or move intent unless that is actually supported.
- `firm_aum_context` is usually qualitative rather than numeric. That is intentional when exact AUM cannot be defended safely from the available evidence.
- The recruiter scoring and phrasing are calibrated for this PPP brief, not for every search mandate.
- Fixture mode is the most reproducible path. Live web research can add evidence, but results depend on external provider availability and public-profile coverage.

## What I Would Build Next For PPP

- A citation-aware reviewer layer so consultants can see which public signal supports each sentence in the brief.
- A mandate-specific triage mode that ranks longlists into immediate call priority, adjacent screen, and mapping-only lanes.
- A tighter consultant feedback loop so PPP recruiters can mark a brief as helpful, too cautious, or misleading and feed that back into prompt and QA calibration.
