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
DEFAULT_RESEARCH_MODE = "live" if settings.tavily_api_key else "fixture"
DEFAULT_UPLOADED_CSV = DEFAULT_PATHS.uploaded_candidates_csv
DEFAULT_OUTPUT_PATH = DEFAULT_PATHS.output_json
DEFAULT_INTERMEDIATE_DIR = DEFAULT_PATHS.intermediate_dir
RESEARCH_MODE_OPTIONS = {
    "Fixture (Local test data)": "fixture",
    "Live API (Public web research)": "live",
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
    subtitle = f"{failure.full_name} - {title}"
    with st.expander(subtitle, expanded=False):
        st.warning(title)
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


def render_candidate_card(candidate) -> None:
    tenure_unknown = candidate.current_role.tenure_years == -1
    expander_label = (
        f"{candidate.full_name} - {candidate.current_role.title} @ {candidate.current_role.employer}"
    )

    with st.expander(expander_label, expanded=False):
        score_col, mobility_col = st.columns(2)
        score_col.metric("Role Fit Score", f"{candidate.role_fit.score} / 10")
        mobility_col.metric("Mobility Score", f"{candidate.mobility_signal.score} / 5")

        st.success(candidate.outreach_hook)

        st.markdown("**Career Narrative**")
        st.write(candidate.career_narrative)

        st.markdown("**Role Fit Justification**")
        st.write(candidate.role_fit.justification)

        tags_text = " ".join(f"`{tag}`" for tag in candidate.experience_tags)
        st.caption(f"Experience Tags: {tags_text}")
        st.caption(f"Firm AUM Context: {candidate.firm_aum_context}")

        if tenure_unknown:
            st.caption(f"Tenure Note: {candidate.mobility_signal.rationale}")


def _candidate_bucket(candidate) -> str:
    opening = candidate.role_fit.justification.split(".", 1)[0].lower()
    if "credible shortlist candidate" in opening or candidate.role_fit.score >= 7:
        return "priority"
    if "possible match pending verification" in opening or candidate.role_fit.score >= 4:
        return "possible"
    return "verify_first"


def render_results(output: PPPOutput, output_json: str) -> None:
    st.markdown("## Candidate Briefings")
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
            render_candidate_card(candidate)

    st.download_button(
        label="Download output.json",
        data=output_json,
        file_name="output.json",
        mime="application/json",
        use_container_width=True,
        key="download_output_json",
        on_click="ignore",
    )


def render_run_report(result: PPPRunResult, run_report_json: str | None) -> None:
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
    if run_report_json is not None:
        st.download_button(
            label="Download run_report.json",
            data=run_report_json,
            file_name="run_report.json",
            mime="application/json",
            use_container_width=True,
            key="download_run_report_json",
            on_click="ignore",
        )


st.set_page_config(page_title=APP_TITLE, page_icon="🔍", layout="wide")

cached_keys = load_local_api_state()
st.session_state.setdefault("ppp_output", None)
st.session_state.setdefault("ppp_output_json", None)
st.session_state.setdefault("ppp_run_result", None)
st.session_state.setdefault("ppp_run_report_json", None)
st.session_state.setdefault("ppp_role_spec_text", dump_role_spec_json(load_role_spec_file(DEFAULT_ROLE_SPEC)))
st.session_state.setdefault("ppp_role_spec_source_text", "")
st.session_state.setdefault("ppp_anthropic_api_key", cached_keys.anthropic_api_key)
st.session_state.setdefault("ppp_tavily_api_key", cached_keys.tavily_api_key)
st.session_state.setdefault(
    "ppp_remember_api_keys",
    bool(cached_keys.anthropic_api_key or cached_keys.tavily_api_key),
)

with st.sidebar:
    st.title("🔍 PPP Executive Search Agent")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        key="ppp_anthropic_api_key",
        help="Used for the current run. You can also cache it locally on this machine.",
    )
    tavily_api_key = st.text_input(
        "TAVILY_API_KEY",
        type="password",
        key="ppp_tavily_api_key",
        help="Required only when using Live API mode. Can be cached locally on this machine.",
    )
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
    research_mode_label = st.radio(
        "Research Mode",
        options=list(RESEARCH_MODE_OPTIONS.keys()),
        index=0 if DEFAULT_RESEARCH_MODE == "fixture" else 1,
    )
    selected_research_mode = RESEARCH_MODE_OPTIONS[research_mode_label]
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
    st.caption("Live mode uses public web research and API credits. Fixture mode uses local mock research for fast testing.")
    inject_api_key(api_key)
    inject_tavily_api_key(tavily_api_key)
    st.caption("Upload a five-row candidate CSV, run the existing PPP pipeline, and review client-ready briefings.")
    st.caption(f"Results overwrite the latest saved files under `{DEFAULT_PATHS.display(DEFAULT_OUTPUT_PATH.parent)}`.")

st.markdown("<h1 style='text-align: center;'>Candidate Briefing Generator</h1>", unsafe_allow_html=True)
st.markdown("## Role Specification")
st.caption(
    f"The app loads the default role specification from `{DEFAULT_PATHS.display(DEFAULT_ROLE_SPEC)}`. "
    "You can paste a new mandate below to parse it, or edit the structured JSON directly before uploading candidates."
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

st.markdown("## Candidate CSV")
uploaded_csv = st.file_uploader("Upload candidate CSV", type=["csv"])

generate_clicked = st.button("🚀 Generate Briefings", type="primary", use_container_width=True)

if generate_clicked:
    if not api_key.strip():
        st.error("Please enter an Anthropic API Key in the sidebar.")
    elif selected_research_mode == "live" and not tavily_api_key.strip():
        st.error("Please enter a TAVILY_API_KEY in the sidebar when using Live API mode.")
    elif uploaded_csv is None:
        st.error("Please upload a CSV file before generating briefings.")
    else:
        try:
            role_spec = load_role_spec_json_text(st.session_state["ppp_role_spec_text"])
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        spinner_text = (
            "🌐 Connecting to Live API and researching candidates (This may take a minute)..."
            if selected_research_mode == "live"
            else "⚡ Running in fast fixture mode..."
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
                st.session_state["ppp_run_result"] = result
                st.session_state["ppp_output"] = result.output
                st.session_state["ppp_output_json"] = output_json
                st.session_state["ppp_run_report_json"] = run_report_json
                if result.delivery_status == "success":
                    st.success(f"Generated {len(result.candidates)} candidate briefings.")
                else:
                    st.warning(
                        f"Generated {result.successful_candidate_count} candidate briefings; "
                        f"{result.failed_candidate_count} candidate(s) need review."
                    )
                st.caption(f"Saved output: `{DEFAULT_PATHS.display(DEFAULT_OUTPUT_PATH)}`")
                st.caption(f"Saved QA / run artifacts: `{DEFAULT_PATHS.display(DEFAULT_INTERMEDIATE_DIR)}`")
                st.caption(f"Saved role spec: `{DEFAULT_PATHS.display(role_spec_path)}`")

if st.session_state["ppp_output"] is not None and st.session_state["ppp_output_json"] is not None:
    if st.session_state["ppp_run_result"] is not None:
        render_run_report(
            result=st.session_state["ppp_run_result"],
            run_report_json=st.session_state["ppp_run_report_json"],
        )
    render_results(
        output=st.session_state["ppp_output"],
        output_json=st.session_state["ppp_output_json"],
    )
