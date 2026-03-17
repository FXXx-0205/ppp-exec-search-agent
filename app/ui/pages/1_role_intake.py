from __future__ import annotations

import os

import httpx
import streamlit as st


def _format_location(location: object) -> str:
    if isinstance(location, str):
        return location or "-"
    if isinstance(location, dict):
        primary = location.get("primary") or []
        country = location.get("country")
        remote_flexibility = location.get("remote_flexibility")

        parts: list[str] = []
        if isinstance(primary, list) and primary:
            parts.append(", ".join(str(item) for item in primary if item))
        elif primary:
            parts.append(str(primary))
        if country:
            parts.append(str(country))
        if remote_flexibility and str(remote_flexibility).lower() not in {"not specified", "unknown"}:
            parts.append(str(remote_flexibility))
        return " / ".join(parts) or "-"
    return "-"


st.session_state["current_step"] = 1
st.header("1) Role Intake  ·  Role Parsing and Knowledge Retrieval")
st.caption("Paste a client brief or JD below. The system will turn it into a structured role spec and retrieve relevant market context.")

api_base = st.session_state.get("api_base") or os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

with st.container():
    col_left, col_right = st.columns([2, 1])

    with col_left:
        raw = st.text_area(
            "Paste Client Brief / JD",
            value="Find senior infrastructure portfolio managers in Australia with institutional funds management experience",
            height=180,
            help="Paste a full English JD or internal search brief.",
        )
        run = st.button("Parse Role and Retrieve Context", type="primary", use_container_width=True)

    with col_right:
        st.markdown("**API Settings**")
        st.text_input("API_BASE_URL", value=api_base, key="api_base", help="You can usually keep the default value.")
        st.markdown("---")
        st.markdown("**Tips**")
        st.markdown(
            "- Include: client type, geography, role scope, must-have requirements, and preferred attributes\n"
            "- Example: *Senior infrastructure portfolio manager for Australian super funds...*"
        )

if run:
    with st.spinner("Running intake..."):
        r = httpx.post(f"{api_base}/search/intake", json={"raw_input": raw}, timeout=60)
        r.raise_for_status()
        data = r.json()
        st.session_state["role_spec"] = data["role_spec"]
        st.session_state["retrieval_context"] = data.get("retrieval_context", [])
        st.session_state["vector_store_mode"] = data.get("vector_store_mode")
        st.session_state["step_1_done"] = True

if "role_spec" in st.session_state:
    role_spec = st.session_state["role_spec"]
    retrieval_context = st.session_state.get("retrieval_context", [])

    tab_summary, tab_raw = st.tabs(["Parsed Summary", "Raw JSON"])

    with tab_summary:
        st.subheader("Structured Role Spec")
        top_cols = st.columns(4)
        top_cols[0].metric("Title", role_spec.get("title") or "-")
        top_cols[1].metric("Seniority", role_spec.get("seniority") or "-")
        top_cols[2].metric("Sector", role_spec.get("sector") or "-")
        top_cols[3].metric("Location", _format_location(role_spec.get("location")))

        col_req, col_pref = st.columns(2)
        col_req.markdown("**Must-have skills**")
        col_req.write(", ".join(role_spec.get("required_skills") or []) or "No required skills detected.")
        col_pref.markdown("**Nice-to-have skills**")
        col_pref.write(", ".join(role_spec.get("preferred_skills") or []) or "No preferred skills detected.")

        if role_spec.get("disqualifiers"):
            st.markdown("**Disqualifiers**")
            st.write(", ".join(role_spec["disqualifiers"]))

        st.markdown("---")
        st.subheader("Retrieved Context (RAG)")
        st.caption(f"vector_store_mode = {st.session_state.get('vector_store_mode')}")

        if retrieval_context:
            for idx, doc in enumerate(retrieval_context, start=1):
                title = doc.get("title") or doc.get("doc_id") or f"doc-{idx}"
                meta = doc.get("metadata") or {}
                sector = meta.get("sector") or "-"
                region = meta.get("region") or "-"
                with st.expander(f"{idx}. {title}  ·  {sector} · {region}", expanded=(idx == 1)):
                    st.markdown(doc.get("text", ""))
                    if meta:
                        st.caption(f"metadata: {meta}")
        else:
            st.info("No context documents were retrieved. Check whether `scripts/ingest_documents.py` has been run.")

    with tab_raw:
        st.markdown("#### Role Spec JSON")
        st.json(role_spec)
        st.markdown("#### Retrieval Context JSON")
        st.json(retrieval_context)

st.markdown("---")
col_prev, col_next = st.columns([1, 1])
with col_next:
    disabled = not bool(st.session_state.get("step_1_done"))
    help_text = "Complete role parsing first." if disabled else None
    if st.button("Next: Candidate Search →", type="primary", disabled=disabled, help=help_text, use_container_width=True):
        try:
            st.switch_page("pages/2_candidate_search.py")
        except Exception:
            st.warning("Automatic navigation is unavailable. Please select “2) Candidate Search” from the sidebar.")
