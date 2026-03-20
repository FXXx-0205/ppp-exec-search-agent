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
from app.ppp.schema import PPPOutput, PPPRunResult, validate_output_document

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


def _load_saved_preview_bundle() -> tuple[PPPOutput, PPPRunResult, str, str | None, str | None]:
    output_text = DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8")
    output = validate_output_document(json.loads(output_text))

    run_report_text = _read_artifact_text(str(DEFAULT_INTERMEDIATE_DIR / "run_report.json"))
    qa_report_text = _read_artifact_text(str(DEFAULT_INTERMEDIATE_DIR / "qa_report.json"))

    delivery_status = "success"
    warnings: list[str] = []
    if run_report_text:
        try:
            run_payload = json.loads(run_report_text)
        except json.JSONDecodeError:
            run_payload = {}
        if isinstance(run_payload, dict):
            raw_status = str(run_payload.get("delivery_status", "success"))
            if raw_status in {"success", "partial_success"}:
                delivery_status = raw_status
            raw_warnings = run_payload.get("warnings")
            if isinstance(raw_warnings, list):
                warnings = [str(item) for item in raw_warnings]

    result = PPPRunResult(
        output=output,
        delivery_status=delivery_status,
        warnings=warnings,
        failed_candidates=[],
        output_path=str(DEFAULT_OUTPUT_PATH),
        qa_report_path=str(DEFAULT_INTERMEDIATE_DIR / "qa_report.json"),
        run_report_path=str(DEFAULT_INTERMEDIATE_DIR / "run_report.json"),
    )
    return output, result, output_text, run_report_text, qa_report_text


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


def _bucket_meta(candidate) -> tuple[str, str]:
    bucket = _candidate_bucket(candidate)
    mapping = {
        "priority": ("Priority Call", "#c76b4f"),
        "possible": ("Screen Next", "#b78b3f"),
        "verify_first": ("Map / Verify", "#7f8b94"),
    }
    return mapping[bucket]


def _tenure_display(candidate) -> str:
    tenure = candidate.current_role.tenure_years
    if tenure <= 0:
        return "Unverified"
    return f"{tenure:.1f} yrs"


def _extract_screening_focus(candidate) -> str:
    justification = candidate.role_fit.justification.strip()
    parts = [part.strip() for part in justification.split(".") if part.strip()]
    return parts[-1] if parts else justification


def _render_tag_row(tags: list[str]) -> None:
    tag_html = "".join(
        f"<span style='display:inline-block;margin:0 0.4rem 0.45rem 0;padding:0.28rem 0.62rem;border-radius:999px;"
        f"background:#f5eee9;border:1px solid #e4d7cf;color:#6a493f;font-size:0.82rem;'>{tag}</span>"
        for tag in tags
    )
    st.markdown(tag_html, unsafe_allow_html=True)


def _status_badge_html(label: str, tone: str = "success") -> str:
    styles = {
        "success": ("#e8f5ec", "#147a3f"),
        "warning": ("#fff4e5", "#9a5b12"),
        "neutral": ("#eef2f6", "#506070"),
    }
    bg, fg = styles.get(tone, styles["neutral"])
    return (
        f"<span style='display:inline-block;padding:0.28rem 0.6rem;border-radius:999px;"
        f"background:{bg};color:{fg};font-size:0.82rem;font-weight:600;'>{label}</span>"
    )


def render_candidate_card(candidate) -> None:
    bucket_label, bucket_color = _bucket_meta(candidate)
    label = (
        f"{candidate.full_name} | {candidate.current_role.title} @ {candidate.current_role.employer} | "
        f"{bucket_label} | Fit {candidate.role_fit.score}/10"
    )
    with st.expander(label, expanded=False):
        st.markdown(
            (
                "<div style='padding:1rem 1.1rem 0.9rem 1.1rem;border:1px solid #ead9d2;border-radius:14px;"
                "background:linear-gradient(180deg,#fffaf7 0%,#fff 100%);margin-bottom:0.9rem;'>"
                f"<div style='display:inline-block;padding:0.24rem 0.62rem;border-radius:999px;background:{bucket_color};"
                "color:white;font-size:0.78rem;font-weight:600;letter-spacing:0.01em;'>"
                f"{bucket_label}</div>"
                f"<div style='margin-top:0.8rem;font-size:1.02rem;line-height:1.55;color:#3f2f2a;'><strong>Outreach hook</strong><br>{candidate.outreach_hook}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        metric_one, metric_two, metric_three = st.columns(3)
        metric_one.metric("Role Fit", f"{candidate.role_fit.score} / 10")
        metric_two.metric("Mobility", f"{candidate.mobility_signal.score} / 5")
        metric_three.metric("Tenure", _tenure_display(candidate))

        review_tab, detail_tab = st.tabs(["Recruiter View", "Detail View"])
        with review_tab:
            review_left, review_right = st.columns([1.1, 0.9])
            with review_left:
                st.markdown("**Career Narrative**")
                st.write(candidate.career_narrative)
                st.markdown("**Role Fit Call**")
                st.write(candidate.role_fit.justification)
            with review_right:
                st.markdown("**What To Test First**")
                st.info(_extract_screening_focus(candidate))
                st.markdown("**Experience Tags**")
                _render_tag_row(candidate.experience_tags)

        with detail_tab:
            st.markdown("**Firm Context**")
            st.write(candidate.firm_aum_context)
            st.markdown("**Mobility Rationale**")
            st.write(candidate.mobility_signal.rationale)


def _candidate_bucket(candidate) -> str:
    opening = candidate.role_fit.justification.split(".", 1)[0].lower()
    if "credible shortlist candidate" in opening or candidate.role_fit.score >= 7:
        return "priority"
    if "possible match pending verification" in opening or candidate.role_fit.score >= 4:
        return "possible"
    return "verify_first"


def render_output_preview(output: PPPOutput) -> None:
    # review section
    st.markdown("<section class='ppp-section-card ppp-review-section'>", unsafe_allow_html=True)
    st.markdown("## Review the generated briefings")
    st.caption("Start with the call order below, then open a candidate card only when you need the full note.")

    grouped = {
        "priority": [candidate for candidate in output.candidates if _candidate_bucket(candidate) == "priority"],
        "possible": [candidate for candidate in output.candidates if _candidate_bucket(candidate) == "possible"],
        "verify_first": [candidate for candidate in output.candidates if _candidate_bucket(candidate) == "verify_first"],
    }

    sections = [
        ("Priority Conversations", "Highest-urgency recruiter calls for this slate.", grouped["priority"]),
        ("Screen Next", "Relevant profiles worth testing, but not yet first-call names.", grouped["possible"]),
        ("Map / Verify", "Useful names to hold in the market map until proof points improve.", grouped["verify_first"]),
    ]
    for heading, caption, candidates in sections:
        if not candidates:
            continue
        st.markdown("<div class='ppp-subsection-card'>", unsafe_allow_html=True)
        st.markdown(f"### {heading}")
        st.caption(caption)
        for candidate in candidates:
            render_candidate_card(candidate)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</section>", unsafe_allow_html=True)


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
    status_label = result.delivery_status.replace("_", " ").title()
    qa_label = qa_status.replace("QA ", "")
    status_tone = "success" if result.delivery_status == "success" else "warning"
    qa_tone = "success" if "passed" in qa_status.lower() else "warning"

    # run summary
    st.markdown("<section class='ppp-section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ppp-section-heading'>Run Summary</div>", unsafe_allow_html=True)
    summary_col_one, summary_col_two, summary_col_three = st.columns(3)
    with summary_col_one:
        st.markdown(
            "<div class='ppp-summary-card'>"
            "<div class='ppp-summary-label'>Candidates</div>"
            f"<div class='ppp-summary-value'>{result.successful_candidate_count}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with summary_col_two:
        st.markdown(
            "<div class='ppp-summary-card'>"
            "<div class='ppp-summary-label'>Status</div>"
            f"<div class='ppp-summary-badge'>{_status_badge_html(status_label, status_tone)}</div>"
            f"<div class='ppp-summary-value ppp-summary-value--status'>{status_label}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with summary_col_three:
        st.markdown(
            "<div class='ppp-summary-card'>"
            "<div class='ppp-summary-label'>QA</div>"
            f"<div class='ppp-summary-badge'>{_status_badge_html(qa_label, qa_tone)}</div>"
            f"<div class='ppp-summary-value ppp-summary-value--status'>{qa_label}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    banner_tone = "success" if result.delivery_status == "success" else "warning"
    banner_text = (
        "Bundle generated successfully and passed bundle-level QA."
        if result.delivery_status == "success"
        else "Run completed with issues. Review held-back profiles before using the bundle."
    )
    st.markdown(
        f"<div class='ppp-status-banner ppp-status-banner--{banner_tone}'>{banner_text}</div>",
        unsafe_allow_html=True,
    )

    meta_left, meta_right = st.columns([1.1, 0.9])
    with meta_left:
        st.caption(qa_caption)
    with meta_right:
        st.caption(
            f"Saved to `{DEFAULT_PATHS.display(DEFAULT_OUTPUT_PATH)}` "
            f"and `{DEFAULT_PATHS.display(DEFAULT_INTERMEDIATE_DIR)}`"
        )
    st.markdown("</section>", unsafe_allow_html=True)

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

# hero header
st.markdown("<section class='ppp-hero'>", unsafe_allow_html=True)
st.title("PPP Candidate Briefing Generator")
st.write("Upload a five-row PPP candidate CSV and generate the final `output.json` briefing bundle.")
st.caption("Fixture mode is the clean review path. Live mode is optional when you want public-web enrichment.")
st.markdown("</section>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .stApp {
        background: #f7f8fa;
    }
    header[data-testid="stHeader"] {
        background: transparent;
        border-bottom: none;
    }
    div[data-testid="stToolbar"] {
        right: 0.75rem;
    }
    .block-container {
        max-width: 1280px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    div[data-testid="stVerticalBlock"] div[data-testid="stExpander"] {
        border: 1px solid #dfe4ea;
        border-radius: 14px;
        background-color: #ffffff;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #dfe4ea;
        border-radius: 14px;
        padding: 0.4rem 0.75rem;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.84rem;
        color: #697586;
    }
    div[data-testid="stMetricValue"] {
        font-weight: 700;
    }
    .ppp-hero {
        margin-bottom: 2.1rem;
    }
    .ppp-section-card {
        background: #ffffff;
        border: 1px solid #dfe4ea;
        border-radius: 18px;
        padding: 1.25rem 1.25rem 1.1rem 1.25rem;
        margin: 0 0 1.6rem 0;
    }
    .ppp-review-section {
        margin-top: 2rem;
        padding-top: 1.4rem;
    }
    .ppp-subsection-card {
        padding: 0.35rem 0 0.15rem 0;
    }
    .ppp-section-heading {
        font-size: 1.5rem;
        font-weight: 700;
        color: #212938;
        margin-bottom: 1rem;
    }
    .ppp-summary-card {
        min-height: 146px;
        border: 1px solid #dfe4ea;
        border-radius: 16px;
        padding: 1rem 1rem 0.9rem 1rem;
        background: #fbfcfd;
    }
    .ppp-summary-label {
        font-size: 0.85rem;
        color: #697586;
        margin-bottom: 0.75rem;
    }
    .ppp-summary-value {
        font-size: 2rem;
        line-height: 1.05;
        font-weight: 700;
        color: #212938;
        margin-top: 1rem;
    }
    .ppp-summary-value--status {
        font-size: 1.75rem;
    }
    .ppp-summary-badge {
        margin-bottom: 0.35rem;
    }
    .ppp-status-banner {
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin-top: 1rem;
        margin-bottom: 0.7rem;
        font-size: 0.96rem;
        font-weight: 500;
    }
    .ppp-status-banner--success {
        background: #eaf6ee;
        color: #17663d;
        border: 1px solid #cce8d5;
    }
    .ppp-status-banner--warning {
        background: #fff5e9;
        color: #8c5312;
        border: 1px solid #f1d7b5;
    }
    .ppp-kpi-card {
        min-height: 112px;
        border: 1px solid #dfe4ea;
        border-radius: 16px;
        padding: 0.9rem 1rem 0.8rem 1rem;
        background: #ffffff;
    }
    .ppp-kpi-label {
        font-size: 0.82rem;
        color: #697586;
        margin-bottom: 0.8rem;
    }
    .ppp-kpi-value {
        font-size: 1.85rem;
        line-height: 1;
        font-weight: 700;
        color: #212938;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #ffffff;
        color: #344054;
        border: 1px solid #d0d5dd;
        border-radius: 12px;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #f8fafc;
        border-color: #c4ccd6;
        color: #1f2937;
    }
    div.stDownloadButton > button {
        width: 100%;
        min-height: 44px;
        border-radius: 12px;
    }
    div.stDownloadButton > button[kind="primary"] {
        background: #222c3d;
        border: 1px solid #222c3d;
        color: #ffffff;
    }
    div.stDownloadButton > button[kind="secondary"] {
        background: #ffffff;
        border: 1px solid #d0d5dd;
        color: #344054;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

api_key = st.session_state["ppp_anthropic_api_key"]
tavily_api_key = st.session_state["ppp_tavily_api_key"]

parse_role_spec_clicked = False
reset_role_spec_clicked = False

has_output = st.session_state["ppp_output"] is not None and st.session_state["ppp_output_json"] is not None

if has_output and st.session_state["ppp_run_result"] is not None:
    render_result_summary(
        result=st.session_state["ppp_run_result"],
        qa_report_json=st.session_state["ppp_qa_report_json"],
    )

    # action bar
    st.markdown("<section class='ppp-section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ppp-section-heading'>Action Bar</div>", unsafe_allow_html=True)
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
            type="primary",
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
    st.markdown("</section>", unsafe_allow_html=True)

    bucket_counts = {
        "priority": len([candidate for candidate in st.session_state["ppp_output"].candidates if _candidate_bucket(candidate) == "priority"]),
        "possible": len([candidate for candidate in st.session_state["ppp_output"].candidates if _candidate_bucket(candidate) == "possible"]),
        "verify_first": len([candidate for candidate in st.session_state["ppp_output"].candidates if _candidate_bucket(candidate) == "verify_first"]),
    }

    # pipeline snapshot
    st.markdown("<section class='ppp-section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ppp-section-heading'>Pipeline Snapshot</div>", unsafe_allow_html=True)
    count_one, count_two, count_three = st.columns(3)
    with count_one:
        st.markdown(
            "<div class='ppp-kpi-card'>"
            "<div class='ppp-kpi-label'>Priority Calls</div>"
            f"<div class='ppp-kpi-value'>{bucket_counts['priority']}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with count_two:
        st.markdown(
            "<div class='ppp-kpi-card'>"
            "<div class='ppp-kpi-label'>Screen Next</div>"
            f"<div class='ppp-kpi-value'>{bucket_counts['possible']}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with count_three:
        st.markdown(
            "<div class='ppp-kpi-card'>"
            "<div class='ppp-kpi-label'>Map / Verify</div>"
            f"<div class='ppp-kpi-value'>{bucket_counts['verify_first']}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</section>", unsafe_allow_html=True)

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
            if st.session_state.get("ppp_remember_api_keys", False):
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
