# PPP Executive-Search Briefing Agent

This repository is my submission for the Platinum Pacific Partners practical task.

I approached it as a small internal search-execution tool, not as a generic “AI recruiter” demo. The job is to take a five-row `candidates.csv`, normalize uneven public evidence with real Claude tool use, and produce an `output.json` that helps a consultant decide who belongs in the first-call group, who deserves an adjacent screen, and who should stay in the market map without overstating what the evidence can support.

For review purposes, the fixture-backed path is the canonical submission-review path. The committed [data/ppp/output.json](/Users/fangxixix/CursorProject/ppp-exec-search-agent/data/ppp/output.json) is the primary artifact for evaluation and represents the promoted best fixture-backed submission artifact. Live mode is still included because the task calls for a real Claude tool and a real public-research-capable workflow, but live results can vary as public sources and search-provider coverage change.

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

Live mode is optional. It demonstrates the public-web research path, but it is not the canonical submission-review path because external evidence can change even when the code and prompts do not.

### Reproducible PPP Run

This is the main command for the recommended submission-review path. It uses the included research fixtures, keeps the review path stable, and reproduces the output shape PPP is evaluating:

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
- uses fixture mode as the safest path for submission review and demo use

The committed [data/ppp/output.json](/Users/fangxixix/CursorProject/ppp-exec-search-agent/data/ppp/output.json) remains the correct artifact to review because it is the promoted best fixture-backed output. Re-running fixture mode uses the same inputs, research fixtures, schema checks, and QA path, but the final briefs are still model-generated, so wording may vary slightly between runs. Live mode uses the same interface and schema guarantees, but the available public evidence may vary over time.

## Architecture

The pipeline is intentionally narrow. I preferred a smaller, clearer flow that matches the PPP acceptance criteria over a more ambitious agent design.

### 1. Research and Evidence Normalisation

The system first builds a candidate research package from fixture or live public-web lookup. Claude is then called with a real Anthropic tool:

- `messages.create(..., tools=[...])`
- Claude is instructed to call `normalize_candidate_evidence`
- Python executes the tool
- the tool result is returned as a `tool_result`
- Claude completes the research phase by returning a normalized evidence object

I kept this phase separate because I did not want research, confidence handling, and final writing to collapse into one prompt. The purpose of this step is to stabilize evidence before any recruiter-style synthesis happens.

In practice, that gives the repository two honest operating modes: a fixture-backed path for stable submission review and a live public-web path for external enrichment. Both use the same two-step architecture and the same schema contract. The fixture-backed path is the recommended review path because it keeps the evidence pack stable and schema-safe, even though the final brief wording may still vary slightly across runs. The live path is intentionally more variable because it depends on changing external evidence.

### Why The Tool Step Matters

The tool step is there to improve judgment quality, not just to satisfy the assignment requirement. Without the normalization pass, several candidates would collapse into the same generic note shape: senior distribution title, plausible relevance, uncertain scope, worth a call. The tool-assisted research phase makes the differences more usable by separating profile confidence, remit shape, channel orientation, and chronology quality before the final shortlist note is written.

That matters commercially. It is the difference between:

- a bundle of five similar summaries
- a slate that separates first-wave calls from calibration screens and mapping-only names

In other words, the tool is doing workflow work. It is not decorative infrastructure around the final JSON.

### 2. Final Candidate Brief Generation

The second Claude call does not expose tools. It takes:

- candidate identity
- normalized evidence
- recruiter signals
- role spec

and returns exactly one `CandidateBrief` JSON object. This separation keeps enrichment, judgment, and strict formatting from bleeding into one another.
That trade-off made the system less flashy, but more reliable.

### 3. Validation and QA

Before export, the system performs:

- candidate-level parsing and validation
- strict PPP schema validation
- bundle-level QA for quality, repetition, and evidence discipline

If the bundle is malformed, export is blocked. Bundle-level QA also flags repetition and evidence-discipline issues before submission so the final artifact is not just valid JSON, but a more coherent briefing set.

## Output Contract

`data/ppp/output.json` is the primary deliverable and the canonical submission artifact.

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

The system assumes public evidence will be uneven. I did not want low-confidence profiles to read with the same certainty as cleanly matched ones.

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

If productised further for PPP, the immediate week-one unlock would be faster mandate triage: consultants could move from raw names to call-priority, adjacent-screen, and map-only lanes with a tighter audit trail around what the public evidence actually supports.

## Intermediate Artifacts

Useful generated files include:

- `data/ppp/output.json`
- `data/ppp/intermediate/qa_report.json`
- `data/ppp/intermediate/run_report.json`
- `data/ppp/intermediate/candidate_*_enriched.json`

The current checked bundle reports:

- `run_report.json`: 5 successful candidates, 0 failures
- `qa_report.json`: `passed: true`

For PPP review, [data/ppp/output.json](/Users/fangxixix/CursorProject/ppp-exec-search-agent/data/ppp/output.json) is the main file to inspect. It is the promoted best fixture-backed artifact selected because it gave the strongest submission-safe balance of recruiter usefulness, evidence discipline, and QA cleanliness. The intermediate artifacts support QA and traceability, but they do not replace the final deliverable.

## Known Limitations

- Public evidence is uneven. The system can frame likely remit shape and firm context, but it should not be read as knowing hidden details such as exact team size, direct reports, or move intent unless those are actually supported.
- `firm_aum_context` is usually qualitative rather than numeric. That is deliberate when exact AUM cannot be defended from the available evidence.
- The scoring and phrasing are tuned for this PPP brief, not as a universal executive-search scoring model.
- Fixture mode is the recommended submission-review path because it keeps the evidence pack, schema checks, and QA path stable. The final candidate briefs are still model-generated, so wording may vary slightly between fixture-backed runs; the committed [data/ppp/output.json](/Users/fangxixix/CursorProject/ppp-exec-search-agent/data/ppp/output.json) is therefore the canonical artifact to inspect. Live web research can add evidence, but results are inherently more variable because provider availability, public-profile coverage, and source material can change over time.

## What I Would Build Next For PPP

- A citation-aware reviewer layer so consultants can see which public signal supports each sentence in the brief.
- A mandate-specific triage mode that helps split longlists into immediate call priority, adjacent screen, and mapping-only lanes.
- A tighter consultant feedback loop so PPP recruiters can mark a brief as helpful, too cautious, or misleading and improve future calibration.
