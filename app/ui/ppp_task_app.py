from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ppp import PPPTaskError, run_ppp_pipeline
from app.ppp.qa import validate_output_bundle, write_qa_report

st.set_page_config(page_title="PPP Task Runner", layout="centered")

DEFAULT_INPUT = "data/ppp/candidates.csv"
DEFAULT_ROLE_SPEC = "data/ppp/role_spec.json"
DEFAULT_OUTPUT = "data/ppp/output.json"
DEFAULT_FIXTURES = "data/ppp/research_fixtures.json"
DEFAULT_INTERMEDIATE = "data/ppp/intermediate"

st.title("PPP Candidate Briefing Runner")
st.caption("Run the PPP assessment pipeline from a simple interface without editing code.")

st.markdown("### Workflow")
st.markdown(
    """
1. Confirm the default input paths or upload a replacement `candidates.csv`.
2. Run the briefing pipeline to generate `output.json` plus intermediate enrichment artifacts.
3. Run validation to confirm the output bundle passes schema and QA checks.
4. Review the JSON output and QA report before submission.
"""
)

with st.expander("Step 1: Configure Inputs", expanded=True):
    input_path = st.text_input("Candidate CSV", value=DEFAULT_INPUT)
    role_spec_path = st.text_input("Role Spec JSON", value=DEFAULT_ROLE_SPEC)
    output_path = st.text_input("Output JSON", value=DEFAULT_OUTPUT)
    model = st.text_input("Claude Model", value="claude-sonnet-4-5")
    research_fixture_path = st.text_input("Research Fixtures", value=DEFAULT_FIXTURES)
    research_mode = st.selectbox("Research Mode", options=["fixture", "auto", "live"], index=0)
    intermediate_dir = st.text_input("Intermediate Directory", value=DEFAULT_INTERMEDIATE)

with st.expander("Step 2: Optional CSV Upload", expanded=True):
    uploaded_csv = st.file_uploader("Upload a replacement candidates.csv", type=["csv"])
    if uploaded_csv is not None:
        upload_target = Path("data/ppp/uploaded_candidates.csv")
        upload_target.parent.mkdir(parents=True, exist_ok=True)
        upload_target.write_bytes(uploaded_csv.getvalue())
        input_path = str(upload_target)
        st.info(f"Using uploaded CSV: {input_path}")

st.markdown("### Current Paths")
st.code(
    "\n".join(
        [
            f"input: {input_path}",
            f"role_spec: {role_spec_path}",
            f"output: {output_path}",
            f"fixtures: {research_fixture_path}",
            f"research_mode: {research_mode}",
            f"intermediate: {intermediate_dir}",
        ]
    ),
    language="text",
)

run_col, validate_col = st.columns(2)

if run_col.button("Run PPP Task", use_container_width=True):
    try:
        result = run_ppp_pipeline(
            input_path=input_path,
            output_path=output_path,
            role_spec_path=role_spec_path,
            model=model,
            intermediate_dir=intermediate_dir,
            research_fixture_path=research_fixture_path,
            research_mode=research_mode,
        )
    except PPPTaskError as exc:
        st.error(str(exc))
    except Exception as exc:  # pragma: no cover - Streamlit display path
        st.exception(exc)
    else:
        st.success(f"Generated {len(result.candidates)} candidate briefings.")
        st.caption(f"Enrichment artifacts saved under {intermediate_dir}")
        st.markdown("### Output Preview")
        st.code(Path(output_path).read_text(encoding="utf-8"), language="json")

if validate_col.button("Validate Existing Output", use_container_width=True):
    try:
        report = validate_output_bundle(
            output_path=output_path,
            input_path=input_path,
            intermediate_dir=intermediate_dir,
        )
        report_path = write_qa_report(report, path=str(Path(intermediate_dir) / "qa_report.json"))
    except Exception as exc:  # pragma: no cover - Streamlit display path
        st.exception(exc)
    else:
        if report.passed:
            st.success(f"Validation passed. QA report written to {report_path}")
        else:
            st.error(f"Validation failed. QA report written to {report_path}")
        st.markdown("### QA Report")
        st.code(Path(report_path).read_text(encoding="utf-8"), language="json")

st.markdown("### What To Review Before Submission")
st.markdown(
    """
- Confirm `output.json` contains exactly 5 candidates.
- Open `qa_report.json` and confirm `passed` is `true`.
- Spot-check one or two `candidate_*_enriched.json` files to verify the evidence and confidence notes look sensible.
- If validation fails, fix the issue and rerun before submitting the repository.
"""
)
