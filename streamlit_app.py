from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import settings
from app.llm.anthropic_client import ClaudeClient
from app.ppp import PPPTaskError, run_ppp_pipeline
from app.ppp.local_state import LocalAPIKeyState, clear_local_api_state, load_local_api_state, save_local_api_state
from app.ppp.paths import DEFAULT_PATHS
from app.ppp.role_spec import (
    dump_role_spec_json,
    load_role_spec_file,
    load_role_spec_json_text,
    parse_role_spec_text,
)
from app.ppp.schema import PPPOutput, PPPRunResult

APP_TITLE = "PPP Executive Search Agent"
DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_ROLE_SPEC = DEFAULT_PATHS.role_spec
UI_ROLE_SPEC_OVERRIDE = DEFAULT_PATHS.role_spec_ui_override
DEFAULT_FIXTURES = DEFAULT_PATHS.research_fixtures
DEFAULT_RESEARCH_MODE = "fixture"
DEFAULT_UPLOADED_CSV = DEFAULT_PATHS.uploaded_candidates_csv
DEFAULT_OUTPUT_PATH = DEFAULT_PATHS.output_json
DEFAULT_INTERMEDIATE_DIR = DEFAULT_PATHS.intermediate_dir
RESEARCH_MODE_OPTIONS = {
    "Fixture (recommended for submission/demo)": "fixture",
    "Live API (optional, requires API keys)": "live",
}


def inject_api_key(api_key: str) -> None:
    normalized = api_key.strip()
    settings.anthropic_api_key = normalized or None
    if normalized:
        os.environ["ANTHROPIC_API_KEY"] = normalized
    else:
        os.environ.pop("ANTHROPIC_API_KEY", None)


def inject_tavily_api_key(api_key: str) -> None:
    normalized = api_key.strip()
    settings.tavily_api_key = normalized or None
    if normalized:
        os.environ["TAVILY_API_KEY"] = normalized
    else:
        os.environ.pop("TAVILY_API_KEY", None)


def _save_local_keys(anthropic_api_key: str, tavily_api_key: str) -> Path:
    return save_local_api_state(
        LocalAPIKeyState(
            anthropic_api_key=anthropic_api_key.strip(),
            tavily_api_key=tavily_api_key.strip(),
        )
    )


def save_uploaded_csv(uploaded_file, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / (Path(uploaded_file.name).name or "uploaded_candidates.csv")
    target_path.write_bytes(uploaded_file.getvalue())
    return target_path


def persist_role_spec(role_spec: dict) -> Path:
    UI_ROLE_SPEC_OVERRIDE.parent.mkdir(parents=True, exist_ok=True)
    UI_ROLE_SPEC_OVERRIDE.write_text(dump_role_spec_json(role_spec), encoding="utf-8")
    return UI_ROLE_SPEC_OVERRIDE


def _read_artifact_text(path_str: str | None) -> str | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _load_failure_artifact(path_str: str | None) -> dict | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _friendly_failure_title(stage: str) -> str:
    return {
        "enrichment": "Research could not be completed",
        "generation": "Briefing draft could not be finalized",
        "candidate_qa": "Held back from final briefing",
        "bundle_qa": "Held back at final review",
    }.get(stage, "Needs review")


def _friendly_failure_reasons(error_message: str) -> list[str]:
    raw_parts = [part.strip() for part in error_message.split(";") if part.strip()]
    reasons: list[str] = []
    replacements = {
        "career_narrative cannot stop at title summary plus generic caution; it should also explain lane relevance.": "The draft stayed too generic and did not explain why the candidate's lane or remit is relevant to the search.",
        "mobility_signal.rationale must separate chronology observation from absent move-readiness evidence and conversation follow-up.": "The mobility note did not clearly separate what public chronology shows from what still needs to be tested in conversation.",
        "firm_aum_context should either stay qualitative when AUM is unverified or clearly frame any numeric AUM as an estimate based on public references.": "The AUM context was not framed cautiously enough for an unverified public data point.",
    }
    for part in raw_parts:
        reasons.append(replacements.get(part, part))
    return reasons or ["This candidate needs manual review before being included in the final briefing set."]


def _recommended_next_check(failure, artifact: dict | None) -> str:
    if artifact is not None:
        enrichment = artifact.get("enrichment") or {}
        signals = enrichment.get("recruiter_signals") or {}
        screening_question = signals.get("screening_priority_question")
        if isinstance(screening_question, str) and screening_question.strip():
            return screening_question.strip()
    message = failure.error_message.lower()
    if "mobility_signal" in message:
        return "Confirm current-role tenure and whether there is any real move trigger before outreach."
    if "career_narrative" in message:
        return "Confirm the candidate's actual lane, scope, and channel relevance before positioning them to the client."
    return "Review the research package and confirm the missing commercial detail before using this profile."


def render_failure_card(failure) -> None:
    artifact = _load_failure_artifact(failure.artifact_path)
    title = _friendly_failure_title(failure.stage)
    st.markdown(f"**{failure.full_name}**")
    st.caption(title)
    st.markdown("**Why it was held back**")
    for reason in _friendly_failure_reasons(failure.error_message):
        st.write(f"- {reason}")

    next_check = _recommended_next_check(failure, artifact)
    st.markdown("**Recommended next check**")
    st.write(next_check)

    if artifact is not None:
        candidate_brief = artifact.get("candidate_brief") or {}
        if candidate_brief:
            st.markdown("**Draft wording that was held back**")
            if candidate_brief.get("career_narrative"):
                st.caption(f"Career Narrative: {candidate_brief['career_narrative']}")
            mobility = candidate_brief.get("mobility_signal") or {}
            if mobility.get("rationale"):
                st.caption(f"Mobility Note: {mobility['rationale']}")

    if failure.artifact_path:
        st.caption(f"Review artifact: {failure.artifact_path}")
    st.divider()


def render_candidate_card(candidate) -> None:
    label = (
        f"{candidate.full_name} | {candidate.current_role.title} @ {candidate.current_role.employer} | "
        f"Role Fit {candidate.role_fit.score}/10 | Mobility {candidate.mobility_signal.score}/5"
    )
    with st.expander(label, expanded=False):
        st.markdown("**Outreach Hook**")
        st.write(candidate.outreach_hook)

        st.markdown("**Career Narrative**")
        st.write(candidate.career_narrative)

        st.markdown("**Experience Tags**")
        st.write(", ".join(candidate.experience_tags))

        st.markdown("**Firm Context**")
        st.write(candidate.firm_aum_context)

        st.markdown("**Mobility Rationale**")
        st.write(candidate.mobility_signal.rationale)

        st.markdown("**Role Fit Justification**")
        st.write(candidate.role_fit.justification)


def _candidate_bucket(candidate) -> str:
    opening = candidate.role_fit.justification.split(".", 1)[0].lower()
    if "credible shortlist candidate" in opening or candidate.role_fit.score >= 7:
        return "priority"
    if "possible match pending verification" in opening or candidate.role_fit.score >= 4:
        return "possible"
    return "verify_first"


def render_output_preview(output: PPPOutput) -> None:
    st.markdown("## Review the generated briefings")
    st.caption("Open each candidate to review the note before downloading the final bundle.")
    for candidate in output.candidates:
        render_candidate_card(candidate)


def render_candidate_diagnostics(output: PPPOutput) -> None:
    grouped = {
        "priority": [candidate for candidate in output.candidates if _candidate_bucket(candidate) == "priority"],
        "possible": [candidate for candidate in output.candidates if _candidate_bucket(candidate) == "possible"],
        "verify_first": [candidate for candidate in output.candidates if _candidate_bucket(candidate) == "verify_first"],
    }
    sections = [
        ("Priority Conversations", "Most actionable profiles for immediate consultant outreach.", grouped["priority"]),
        ("Possible Matches", "Directionally relevant profiles that still need some verification or scope checks.", grouped["possible"]),
        ("Verify First", "Commercially interesting profiles that should stay in the market map until proof points improve.", grouped["verify_first"]),
    ]
    for heading, caption, candidates in sections:
        if not candidates:
            continue
        st.markdown(f"### {heading}")
        st.caption(caption)
        for candidate in candidates:
            st.write(f"- {candidate.full_name} — {candidate.current_role.title} @ {candidate.current_role.employer}")


def _qa_summary(qa_report_json: str | None) -> tuple[str, str]:
    if not qa_report_json:
        return ("QA status unavailable", "No QA report was loaded into the UI.")
    try:
        payload = json.loads(qa_report_json)
    except json.JSONDecodeError:
        return ("QA report unreadable", "The QA report file could not be parsed.")
    passed = bool(payload.get("passed"))
    findings = payload.get("findings") or []
    if passed:
        return ("QA passed", "Bundle-level QA passed for the current output.")
    return ("QA needs review", f"Bundle-level QA reported {len(findings)} finding(s).")


def render_result_summary(result: PPPRunResult, qa_report_json: str | None) -> None:
    qa_status, qa_caption = _qa_summary(qa_report_json)
    st.markdown("## Result summary")
    summary_col_one, summary_col_two, summary_col_three = st.columns(3)
    summary_col_one.metric("Candidate briefings", str(result.successful_candidate_count))
    summary_col_two.metric("Delivery status", result.delivery_status.replace("_", " ").title())
    summary_col_three.metric("QA status", qa_status)
    st.caption(qa_caption)
    st.caption(f"Output saved to `{DEFAULT_PATHS.display(DEFAULT_OUTPUT_PATH)}`")
    st.caption(f"QA and run artifacts saved to `{DEFAULT_PATHS.display(DEFAULT_INTERMEDIATE_DIR)}`")

def render_run_report(result: PPPRunResult, run_report_json: str | None, qa_report_json: str | None) -> None:
    if result.delivery_status == "partial_success":
        st.warning(
            f"Generated {result.successful_candidate_count} candidate briefings; "
            f"{result.failed_candidate_count} candidate(s) need review."
        )
    if result.warnings:
        for warning in result.warnings:
            st.info(warning)
    if result.failed_candidates:
        st.markdown("## Held Back From Final Briefing")
        st.caption("These profiles completed processing but were not promoted into the final briefing set because the draft remained too uncertain or not client-ready.")
        for failure in result.failed_candidates:
            render_failure_card(failure)
    download_col_one, download_col_two = st.columns(2)
    if run_report_json is not None:
        with download_col_one:
            st.download_button(
                label="Download run_report.json",
                data=run_report_json,
                file_name="run_report.json",
                mime="application/json",
                use_container_width=True,
                key="download_run_report_json",
                on_click="ignore",
            )
    if qa_report_json is not None:
        with download_col_two:
            st.download_button(
                label="Download qa_report.json",
                data=qa_report_json,
                file_name="qa_report.json",
                mime="application/json",
                use_container_width=True,
                key="download_qa_report_json",
                on_click="ignore",
            )


st.set_page_config(page_title=APP_TITLE, page_icon="🔍", layout="wide")

cached_keys = load_local_api_state()
st.session_state.setdefault("ppp_output", None)
st.session_state.setdefault("ppp_output_json", None)
st.session_state.setdefault("ppp_run_result", None)
st.session_state.setdefault("ppp_run_report_json", None)
st.session_state.setdefault("ppp_qa_report_json", None)
st.session_state.setdefault("ppp_role_spec_text", dump_role_spec_json(load_role_spec_file(DEFAULT_ROLE_SPEC)))
st.session_state.setdefault("ppp_role_spec_source_text", "")
st.session_state.setdefault("ppp_anthropic_api_key", cached_keys.anthropic_api_key)
st.session_state.setdefault("ppp_tavily_api_key", cached_keys.tavily_api_key)
st.session_state.setdefault(
    "ppp_remember_api_keys",
    bool(cached_keys.anthropic_api_key or cached_keys.tavily_api_key),
)

st.title("PPP Candidate Briefing Generator")
st.write("Upload a five-row PPP candidate CSV and generate the final `output.json` briefing bundle.")
st.caption(
    "For submission or demo use, fixture mode is recommended. Live mode is an optional public-web enrichment path if you want to test external research."
)
st.caption("1. Choose fixture mode for submission/demo. 2. Upload the CSV. 3. Generate and review the output bundle.")
st.markdown(
    """
    <style>
    .stApp {
        background: #ffffff;
    }
    header[data-testid="stHeader"] {
        background: transparent;
        border-bottom: none;
    }
    div[data-testid="stToolbar"] {
        right: 0.75rem;
    }
    .block-container {
        padding-top: 1.5rem;
    }
    div[data-testid="stVerticalBlock"] div[data-testid="stExpander"] {
        border: 1px solid #e2d4cd;
        border-radius: 12px;
        background-color: rgba(255, 255, 255, 0.72);
    }
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.78);
        border: 1px solid #e4d7cf;
        border-radius: 12px;
        padding: 0.4rem 0.75rem;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #ead9d2;
        color: #5a3b32;
        border: 1px solid #d3bbb1;
        border-radius: 10px;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #e0ccc4;
        border-color: #c8aa9d;
        color: #4d3028;
    }
    div.stDownloadButton > button {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

api_key = st.session_state["ppp_anthropic_api_key"]
tavily_api_key = st.session_state["ppp_tavily_api_key"]

with st.expander("Advanced: local key storage", expanded=False):
    remember_keys = st.checkbox(
        "Remember keys locally on this machine",
        key="ppp_remember_api_keys",
        help=f"Saved only to {DEFAULT_PATHS.display(DEFAULT_PATHS.local_state_file)} and ignored by git.",
    )
    save_keys_col, clear_keys_col = st.columns(2)
    with save_keys_col:
        save_keys_clicked = st.button("Save Keys", use_container_width=True)
    with clear_keys_col:
        clear_keys_clicked = st.button("Clear Saved", use_container_width=True)
    if save_keys_clicked:
        saved_path = _save_local_keys(api_key, tavily_api_key)
        st.success(f"Saved locally: {DEFAULT_PATHS.display(saved_path)}")
    if clear_keys_clicked:
        clear_local_api_state()
        st.session_state["ppp_anthropic_api_key"] = ""
        st.session_state["ppp_tavily_api_key"] = ""
        st.session_state["ppp_remember_api_keys"] = False
        api_key = ""
        tavily_api_key = ""
        remember_keys = False
        inject_api_key("")
        inject_tavily_api_key("")
        st.success("Cleared locally saved API keys.")

parse_role_spec_clicked = False
reset_role_spec_clicked = False

has_output = st.session_state["ppp_output"] is not None and st.session_state["ppp_output_json"] is not None

if has_output and st.session_state["ppp_run_result"] is not None:
    render_result_summary(
        result=st.session_state["ppp_run_result"],
        qa_report_json=st.session_state["ppp_qa_report_json"],
    )

    download_col_one, download_col_two, download_col_three = st.columns(3)
    with download_col_one:
        st.download_button(
            label="Download output.json",
            data=st.session_state["ppp_output_json"],
            file_name="output.json",
            mime="application/json",
            use_container_width=True,
            key="download_output_json",
            on_click="ignore",
        )
    with download_col_two:
        if st.session_state["ppp_run_report_json"] is not None:
            st.download_button(
                label="Download run_report.json",
                data=st.session_state["ppp_run_report_json"],
                file_name="run_report.json",
                mime="application/json",
                use_container_width=True,
                key="download_run_report_top",
                on_click="ignore",
            )
    with download_col_three:
        if st.session_state["ppp_qa_report_json"] is not None:
            st.download_button(
                label="Download qa_report.json",
                data=st.session_state["ppp_qa_report_json"],
                file_name="qa_report.json",
                mime="application/json",
                use_container_width=True,
                key="download_qa_report_top",
                on_click="ignore",
            )

    render_output_preview(st.session_state["ppp_output"])

st.markdown("## Generate the final output bundle" if not has_output else "## Run the task again")
run_panel = st.container(border=True)
with run_panel:
    primary_left, primary_right = st.columns(2)
    with primary_left:
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            key="ppp_anthropic_api_key",
            help="Required for the Claude steps in the pipeline.",
        )
        research_mode_label = st.radio(
            "Mode",
            options=list(RESEARCH_MODE_OPTIONS.keys()),
            index=0 if DEFAULT_RESEARCH_MODE == "fixture" else 1,
            help="Fixture mode is the safest path for a submission demo.",
        )
        selected_research_mode = RESEARCH_MODE_OPTIONS[research_mode_label]
        st.caption("Fixture mode is sufficient to review the workflow and output quality. Live mode uses changing public evidence, so results may vary over time.")
    with primary_right:
        if selected_research_mode == "live":
            tavily_api_key = st.text_input(
                "TAVILY_API_KEY",
                type="password",
                key="ppp_tavily_api_key",
                help="Only required for live mode.",
            )
            st.caption("Only required for live mode. Live output may vary with public-source coverage and search-provider availability.")
        else:
            tavily_api_key = st.session_state["ppp_tavily_api_key"]
        uploaded_csv = st.file_uploader("Upload candidate CSV", type=["csv"])
        st.caption(
            f"Uploaded CSVs are saved to `{DEFAULT_PATHS.display(DEFAULT_UPLOADED_CSV)}`. "
            f"The generated bundle is written to `{DEFAULT_PATHS.display(DEFAULT_OUTPUT_PATH)}`."
        )

    st.caption("This will generate the final candidate briefing bundle (`output.json`).")
    generate_clicked = st.button("Generate output bundle (output.json)", use_container_width=True)

inject_api_key(api_key)
inject_tavily_api_key(tavily_api_key)

if generate_clicked:
    if not api_key.strip():
        st.error("Please enter an Anthropic API Key.")
    elif selected_research_mode == "live" and not tavily_api_key.strip():
        st.error("Please enter a TAVILY_API_KEY when using Live API mode.")
    elif uploaded_csv is None:
        st.error("Please upload a CSV file before generating briefings.")
    else:
        try:
            role_spec = load_role_spec_json_text(st.session_state["ppp_role_spec_text"])
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        spinner_text = (
            "Connecting to live research and generating the briefings..."
            if selected_research_mode == "live"
            else "Running in fixture mode..."
        )
        with st.spinner(spinner_text):
            temp_csv_path = save_uploaded_csv(uploaded_csv, DEFAULT_UPLOADED_CSV.parent)
            role_spec_path = persist_role_spec(role_spec)
            if remember_keys:
                _save_local_keys(api_key, tavily_api_key)
            try:
                result = run_ppp_pipeline(
                    input_path=str(temp_csv_path),
                    output_path=str(DEFAULT_OUTPUT_PATH),
                    role_spec_path=str(role_spec_path),
                    model=DEFAULT_MODEL,
                    intermediate_dir=str(DEFAULT_INTERMEDIATE_DIR),
                    research_fixture_path=str(DEFAULT_FIXTURES),
                    research_mode=selected_research_mode,
                )
            except PPPTaskError as exc:
                st.error(str(exc))
            except Exception as exc:  # pragma: no cover - UI exception path
                st.error("The PPP pipeline failed unexpectedly.")
                st.exception(exc)
            else:
                output_json = json.dumps(result.output.model_dump(mode="json"), ensure_ascii=False, indent=2)
                run_report_json = json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2)
                qa_report_json = _read_artifact_text(result.qa_report_path) or _read_artifact_text(
                    str(DEFAULT_INTERMEDIATE_DIR / "qa_report.json")
                )
                st.session_state["ppp_run_result"] = result
                st.session_state["ppp_output"] = result.output
                st.session_state["ppp_output_json"] = output_json
                st.session_state["ppp_run_report_json"] = run_report_json
                st.session_state["ppp_qa_report_json"] = qa_report_json
                st.session_state["ppp_last_saved_role_spec"] = DEFAULT_PATHS.display(role_spec_path)
                st.rerun()

with st.expander("Advanced: role specification override", expanded=False):
    st.caption(
        f"The app loads the default role specification from `{DEFAULT_PATHS.display(DEFAULT_ROLE_SPEC)}`. "
        "You can leave this as-is, paste a different brief to parse, or edit the JSON directly."
    )
    role_spec_text_col, role_spec_json_col = st.columns(2)
    with role_spec_text_col:
        st.text_area(
            "Role specification text (optional)",
            key="ppp_role_spec_source_text",
            height=260,
            placeholder="Paste a hiring brief or role description here, then click Parse Role Spec.",
        )
    with role_spec_json_col:
        st.text_area(
            "Structured role specification JSON",
            key="ppp_role_spec_text",
            height=260,
        )

    role_spec_action_col, role_spec_reset_col = st.columns(2)
    with role_spec_action_col:
        parse_role_spec_clicked = st.button("Parse Role Spec", use_container_width=True)
    with role_spec_reset_col:
        reset_role_spec_clicked = st.button("Reset Role Spec To Default", use_container_width=True)

if parse_role_spec_clicked:
    try:
        parsed_role_spec = parse_role_spec_text(
            text=st.session_state["ppp_role_spec_source_text"],
            client=ClaudeClient(api_key=settings.anthropic_api_key),
            model=DEFAULT_MODEL,
        )
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.session_state["ppp_role_spec_text"] = dump_role_spec_json(parsed_role_spec)
        st.success("Role specification parsed into structured JSON.")

if reset_role_spec_clicked:
    st.session_state["ppp_role_spec_text"] = dump_role_spec_json(load_role_spec_file(DEFAULT_ROLE_SPEC))
    st.session_state["ppp_role_spec_source_text"] = ""

if has_output and st.session_state["ppp_run_result"] is not None:
    with st.expander("Advanced: QA and run diagnostics", expanded=False):
        render_run_report(
            result=st.session_state["ppp_run_result"],
            run_report_json=st.session_state["ppp_run_report_json"],
            qa_report_json=st.session_state["ppp_qa_report_json"],
        )
    with st.expander("Advanced: additional candidate diagnostics", expanded=False):
        render_candidate_diagnostics(
            output=st.session_state["ppp_output"],
        )
    with st.expander("Advanced: view raw JSON", expanded=False):
        st.code(st.session_state["ppp_output_json"], language="json")
