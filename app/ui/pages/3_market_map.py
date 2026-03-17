from __future__ import annotations

import pandas as pd
import streamlit as st


st.session_state["current_step"] = 3
st.header("3) Market Map  ·  Firm and Market View")
st.caption("A lightweight view of firm distribution and market context based on the current candidate pool and retrieved company profiles.")

candidates = st.session_state.get("candidates") or []
retrieval_context = st.session_state.get("retrieval_context") or []

if not candidates and not retrieval_context:
    st.info("Complete Step 1 (Role Intake) and Step 2 (Candidate Search) first.")
    st.stop()

col_firms, col_context = st.columns([1.2, 1])

with col_firms:
    st.subheader("Top Firms (by Candidate Count)")
    firms = {}
    for c in candidates:
        firm = c.get("current_company") or "Unknown"
        firms[firm] = firms.get(firm, 0) + 1

    if firms:
        df_firms = pd.DataFrame(
            sorted([{"Firm": k, "Candidates": v} for k, v in firms.items()], key=lambda x: x["Candidates"], reverse=True)
        )
        candidate_counts = df_firms["Candidates"].tolist()
        has_meaningful_distribution = len(set(candidate_counts)) > 1
        if has_meaningful_distribution:
            st.bar_chart(df_firms.set_index("Firm"))
        else:
            st.caption("Candidate counts are nearly identical across firms, so the bar chart has been hidden to avoid low-information visual noise.")
        st.dataframe(df_firms, hide_index=True, use_container_width=True)
    else:
        st.write("No candidate data is available yet.")

with col_context:
    st.subheader("RAG Grounding Context (Firms / Institutions)")
    if retrieval_context:
        for d in retrieval_context:
            meta = d.get("metadata") or {}
            title = d.get("title") or d.get("doc_id") or meta.get("company") or "firm"
            sector = meta.get("sector") or "-"
            region = meta.get("region") or "-"
            with st.expander(f"{title} · {sector} · {region}", expanded=False):
                st.write(d.get("text", ""))
                if meta:
                    st.caption(f"metadata: {meta}")
    else:
        st.write("No retrieval context is currently available. `scripts/ingest_documents.py` may not have been run, or the retrieval query may be too narrow.")

st.session_state["step_3_done"] = True

st.markdown("---")
col_prev, col_next = st.columns([1, 1])
with col_prev:
    if st.button("← Back: Candidate Search", use_container_width=True):
        try:
            st.switch_page("pages/2_candidate_search.py")
        except Exception:
            st.warning("Automatic navigation is unavailable. Please select “2) Candidate Search” from the sidebar.")
with col_next:
    if st.button("Next: Client Brief →", type="primary", use_container_width=True):
        try:
            st.switch_page("pages/4_client_brief.py")
        except Exception:
            st.warning("Automatic navigation is unavailable. Please select “4) Client Brief” from the sidebar.")
