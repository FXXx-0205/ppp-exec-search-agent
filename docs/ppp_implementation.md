# PPP Task Implementation Notes

This repository now includes a dedicated PPP assessment path that sits alongside the existing executive-search workflow system.

## Directory structure

- `app/ppp/`
  PPP-specific models and pipeline logic.
- `scripts/run_ppp_task.py`
  CLI entrypoint for non-technical execution.
- `app/ui/ppp_task_app.py`
  Minimal Streamlit runner for upload-and-run usage.
- `data/ppp/candidates.csv`
  Default candidate input file.
- `data/ppp/role_spec.json`
  Default role specification for role-fit evaluation.

## Current scope

Phase 1 focused on the runnable interface:

- input path handling
- CSV validation
- role spec loading
- API key checks
- candidate-by-candidate JSON generation
- output writing
- clear user-facing errors

Phase 2 adds a dedicated enrichment/tool layer:

- `candidate_public_profile_lookup` tool contract
- fixture-backed public-research runner
- evidence, source, missing-field, and confidence-note capture
- firm/AUM context formatting and tenure estimation helpers
- intermediate artifacts under `data/ppp/intermediate/`
- a two-stage flow: enrichment first, structured briefing second

## Tooling mode

Because the current implementation environment does not guarantee live public-web access, the enrichment stage runs in a controlled fixture-backed mode by default using `data/ppp/research_fixtures.json`.

This is deliberate:

- it preserves a real tool interface and two-stage architecture
- it prevents unsupported claims from creeping into the final brief
- it keeps the system ready for a later live-web connector without rewriting the pipeline

When Anthropic tool use is available, the pipeline can ask Claude to call `candidate_public_profile_lookup`; when it is not, the pipeline falls back to direct local execution of the same tool contract.

## QA layer

The PPP path now also includes a post-generation QA and validation layer:

- schema validation before any output is written
- candidate-level content checks such as sentence counts and placeholder detection
- business checks around justification evidence, AUM confidence language, and tenure alignment
- candidate-specific failure artifacts for partial debug
- `qa_report.json` generation for one-click pre-submission validation
