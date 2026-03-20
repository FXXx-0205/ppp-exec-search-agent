# PPP Executive-Search Briefing Agent

This repository is my submission for the Platinum Pacific Partners practical task.

## Quick Start 

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
export ANTHROPIC_API_KEY=your_key_here
```

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
## Non-Technical UI

For a reviewer who prefers a UI:

```bash
streamlit run streamlit_app.py
```

The Streamlit app uses the same pipeline and writes the same structured outputs:

- `data/ppp/output.json`
- `data/ppp/intermediate/qa_report.json`
- `data/ppp/intermediate/run_report.json`

## What This Does

- Turns a five-candidate PPP input list into a call-priority output for consultant review.
- Produces structured `output.json` rather than free-form summaries.
- Uses evidence-aware judgment so verified, adjacent, and low-confidence cases are handled differently.
- Separates shortlist, screen, and mapping value without changing the recruiter workflow.
- Writes QA and run artifacts alongside the final export.

## Example Outcome

After running the pipeline, a consultant does not get five similar summaries.

They get:
- a clear call order (who to call first, second, third)
- explicit differences between candidates (not just tone, but prioritisation)
- a starting point for screening conversations (what to test, what is unclear)

This turns the output from a reading task into a decision-making tool.

## How To Read The Output

- `role_fit.score` is designed to reflect call priority, not just generic candidate quality.
- Score differences are intentional: they help show who should get the first call, who belongs in the next screening wave, and who is mainly map value.
- The output is built for recruiter decision-making under uncertainty, not for producing polished profile summaries.
- The goal is not to eliminate judgment, but to make it faster and more consistent.

## Recommended Review Path

The fixture-backed path is the canonical review path. It keeps the evidence pack stable and makes the generated bundle easier to assess fairly. The committed [data/ppp/output.json](/Users/fangxixix/CursorProject/ppp-exec-search-agent/data/ppp/output.json) is the primary artifact to inspect.

To validate an existing export without regenerating it:

```bash
python3 scripts/run_ppp_task.py \
  --input data/ppp/candidates.csv \
  --output data/ppp/output.json \
  --intermediate-dir data/ppp/intermediate \
  --validate-only
```

## Fixture vs Live

Fixture mode is recommended for submission review because it is reproducible and keeps the evidence pack stable. Live mode is also supported for real public-web research, but results can vary as source coverage changes. If you want live enrichment, set `TAVILY_API_KEY` and switch `--research-mode live`.

## Architecture

The architecture is intentionally narrow to keep the workflow clear and reviewable:

1. mandate-aware evidence normalization
2. candidate brief generation into strict PPP JSON
3. QA and schema validation before export

The system uses real Claude tool calling, but the important business point is not the API mechanics. The purpose of the architecture is to stabilize uneven public evidence before writing recruiter-facing output, so consultants get a more consistent call-priority artifact rather than five loosely comparable summaries.

## Output Contract

`data/ppp/output.json` is the main deliverable. It contains:

- exactly 5 candidates
- only the PPP-required fields
- integer scores in schema-safe ranges
- structured recruiter-facing fields such as `career_narrative`, `role_fit`, and `outreach_hook`

## Evidence, Auditability, and QA

The pipeline is designed to be conservative with public evidence:

- evidence is normalized before final writing
- confidence states affect tone and score calibration
- lower-confidence cases are written more cautiously
- schema validation and QA run before export
- fixture-backed mode makes runs reproducible for review

That audit trail is part of the product value. The JSON output is not just an export format; it is a review artifact that makes uncertainty handling visible.

## Why This Is Useful For PPP

This tool does not ask consultants to change how they work. It compresses the existing workflow: less time on first-pass research, less effort comparing uneven profiles, and more consistency in deciding who should get the next call. Judgment remains with the consultant, but the evidence is shaped into a more usable starting point.

## Intermediate Artifacts

Useful files include:

- [data/ppp/output.json](/Users/fangxixix/CursorProject/ppp-exec-search-agent/data/ppp/output.json)
- [data/ppp/intermediate/qa_report.json](/Users/fangxixix/CursorProject/ppp-exec-search-agent/data/ppp/intermediate/qa_report.json)
- [data/ppp/intermediate/run_report.json](/Users/fangxixix/CursorProject/ppp-exec-search-agent/data/ppp/intermediate/run_report.json)
- `data/ppp/intermediate/candidate_*_enriched.json`

## Known Limitations

- Public evidence remains uneven, so some remit details still need live validation on a call.
- Exact AUM is often treated qualitatively rather than forced into weak numeric claims.
- The scoring logic is tuned for this PPP brief, not as a universal executive-search model.
- Even fixture-backed runs can vary slightly in phrasing because the final briefs are model-generated.
