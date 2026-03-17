from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.ppp import PPPTaskError, run_ppp_pipeline

st.set_page_config(page_title="PPP Task Runner", layout="centered")

DEFAULT_INPUT = "data/ppp/candidates.csv"
DEFAULT_ROLE_SPEC = "data/ppp/role_spec.json"
DEFAULT_OUTPUT = "data/ppp/output.json"
DEFAULT_FIXTURES = "data/ppp/research_fixtures.json"
DEFAULT_INTERMEDIATE = "data/ppp/intermediate"

st.title("PPP Candidate Briefing Runner")
st.caption("Run the PPP assessment pipeline from a simple interface without editing code.")

st.markdown("### Inputs")
input_path = st.text_input("Candidate CSV", value=DEFAULT_INPUT)
role_spec_path = st.text_input("Role Spec JSON", value=DEFAULT_ROLE_SPEC)
output_path = st.text_input("Output JSON", value=DEFAULT_OUTPUT)
model = st.text_input("Claude Model", value="claude-sonnet-4-5")
research_fixture_path = st.text_input("Research Fixtures", value=DEFAULT_FIXTURES)
intermediate_dir = st.text_input("Intermediate Directory", value=DEFAULT_INTERMEDIATE)

st.markdown("### Optional Upload")
uploaded_csv = st.file_uploader("Upload a replacement candidates.csv", type=["csv"])
if uploaded_csv is not None:
    upload_target = Path("data/ppp/uploaded_candidates.csv")
    upload_target.parent.mkdir(parents=True, exist_ok=True)
    upload_target.write_bytes(uploaded_csv.getvalue())
    input_path = str(upload_target)
    st.info(f"Using uploaded CSV: {input_path}")

if st.button("Run PPP Task", use_container_width=True):
    try:
        result = run_ppp_pipeline(
            input_path=input_path,
            output_path=output_path,
            role_spec_path=role_spec_path,
            model=model,
            intermediate_dir=intermediate_dir,
            research_fixture_path=research_fixture_path,
        )
    except PPPTaskError as exc:
        st.error(str(exc))
    except Exception as exc:  # pragma: no cover - Streamlit display path
        st.exception(exc)
    else:
        st.success(f"Generated {len(result.candidates)} candidate briefings.")
        st.caption(f"Enrichment artifacts saved under {intermediate_dir}")
        st.code(Path(output_path).read_text(encoding="utf-8"), language="json")
