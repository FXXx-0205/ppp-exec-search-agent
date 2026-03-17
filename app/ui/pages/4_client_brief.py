from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st


st.session_state["current_step"] = 4
st.header("4) Client Brief  ·  Ranking and Briefing")

api_base = st.session_state.get("api_base") or os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
role_spec = st.session_state.get("role_spec")
candidates = st.session_state.get("candidates") or []

if not role_spec:
    st.info("Complete Step 1 (Role Intake) first.")
    st.stop()

candidate_ids = [c.get("candidate_id") for c in candidates if c.get("candidate_id")]
default_pick = candidate_ids[:3]

st.markdown("**Select candidates for ranking and brief generation (leave empty to use the full pool)**")
picked = st.multiselect("Candidate IDs", options=candidate_ids, default=default_pick)

col1, col2 = st.columns([1, 1])
with col1:
    run_rank = st.button("Run Ranking", type="secondary", use_container_width=True)
with col2:
    run_brief = st.button("Generate Brief (Markdown)", type="primary", use_container_width=True)

if run_rank:
    with st.spinner("Ranking..."):
        r = httpx.post(
            f"{api_base}/search/rank",
            json={"role_spec": role_spec, "candidate_ids": picked or candidate_ids},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        st.session_state["ranked_candidates"] = data["ranked_candidates"]

ranked = st.session_state.get("ranked_candidates") or []
if ranked:
    st.subheader("Ranking Results（Top 10）")
    table = [
        {
            "ID": r.get("candidate_id"),
            "Fit": r.get("fit_score"),
            "Skill": r.get("skill_match_score"),
            "Sector": r.get("sector_relevance_score"),
            "Location": r.get("location_score"),
            "Reason": " | ".join((r.get("summary_reasons") or r.get("reasoning") or [""])[:2]),
        }
        for r in ranked[:10]
    ]
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)

    with st.expander("View full ranking JSON (debug)", expanded=False):
        st.json(ranked[:10])

if run_brief:
    with st.spinner("Generating brief..."):
        r = httpx.post(
            f"{api_base}/briefs/generate",
            json={"role_spec": role_spec, "candidate_ids": picked or None, "project_id": "demo"},
            timeout=90,
        )
        r.raise_for_status()
        data = r.json()
        st.session_state["brief_md"] = data["markdown"]
        st.session_state["brief_id"] = data.get("brief_id")
        st.session_state["brief_approved"] = False

if "brief_md" in st.session_state:
    st.subheader("Brief Generation and Approval")
    brief_id = st.session_state.get("brief_id")

    status_col, action_col = st.columns([1, 2])
    with status_col:
        approved = bool(st.session_state.get("brief_approved"))
        if approved:
            st.success(f"Approved · brief_id = {brief_id}")
        else:
            st.warning(f"Pending approval · brief_id = {brief_id}")

    with action_col:
        cols = st.columns([1, 1, 2])
        with cols[0]:
            approve = st.button("Approve (Enable External Export)", type="primary", disabled=not bool(brief_id))
        with cols[1]:
            refresh = st.button("Refresh Approval Status", disabled=not bool(brief_id))

        if approve and brief_id:
            r = httpx.post(f"{api_base}/briefs/approve/{brief_id}", timeout=60)
            r.raise_for_status()
            st.session_state["brief_approved"] = True
            st.success("Brief approved. Export is now enabled.")

        if refresh and brief_id:
            r = httpx.get(f"{api_base}/briefs/{brief_id}", timeout=60)
            r.raise_for_status()
            st.session_state["brief_approved"] = bool(r.json().get("approved"))

    approved = bool(st.session_state.get("brief_approved"))

    if approved:
        st.download_button(
            "Download Markdown (Approved)",
            data=st.session_state["brief_md"],
            file_name=f"{brief_id or 'brief'}.md",
            use_container_width=True,
        )
    else:
        st.info("This brief has not been approved yet. Export remains gated to simulate a regulated review workflow.")

    st.markdown("---")
    st.markdown("#### Brief Preview")
    st.markdown(st.session_state["brief_md"])

    st.session_state["step_4_done"] = True

st.markdown("---")
col_prev, col_next = st.columns([1, 1])
with col_prev:
    if st.button("← Back: Market Map", use_container_width=True):
        try:
            st.switch_page("pages/3_market_map.py")
        except Exception:
            st.warning("Automatic navigation is unavailable. Please select “3) Market Map” from the sidebar.")
with col_next:
    if st.button("Return to Role Intake →", use_container_width=True):
        try:
            st.switch_page("pages/1_role_intake.py")
        except Exception:
            st.warning("Automatic navigation is unavailable. Please select “1) Role Intake” from the sidebar.")
