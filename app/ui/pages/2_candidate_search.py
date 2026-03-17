from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st


st.session_state["current_step"] = 2
st.header("2) Candidate Search  ·  Candidate Retrieval and Review")

api_base = st.session_state.get("api_base") or os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
role_spec = st.session_state.get("role_spec")

if not role_spec:
    st.info("Complete Step 1 (Role Intake) first.")
    st.stop()

run = st.button("Retrieve Candidates from Demo Pool", type="primary")

if run:
    with st.spinner("Searching candidates..."):
        r = httpx.post(f"{api_base}/search/candidates", json={"role_spec": role_spec, "project_id": "demo"}, timeout=60)
        r.raise_for_status()
        data = r.json()
        st.session_state["candidates"] = data["candidates"]

candidates = st.session_state.get("candidates") or []
st.subheader(f"Candidate List ({len(candidates)})")

if candidates:
    all_locations = sorted({c.get("location") for c in candidates if c.get("location")})
    all_sectors = sorted({sector for c in candidates for sector in (c.get("sectors") or [])})
    all_functions = sorted({function for c in candidates for function in (c.get("functions") or [])})

    st.markdown("#### Filters")
    filter_col_1, filter_col_2, filter_col_3 = st.columns(3)
    with filter_col_1:
        keyword = st.text_input("Keyword", placeholder="name / title / company")
        selected_locations = st.multiselect("Location", options=all_locations)
    with filter_col_2:
        selected_sectors = st.multiselect("Sector", options=all_sectors)
        selected_functions = st.multiselect("Function", options=all_functions)
    with filter_col_3:
        min_confidence = st.slider("Minimum confidence", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    filtered_candidates = []
    keyword_lower = keyword.strip().lower()
    for candidate in candidates:
        haystack = " ".join(
            [
                str(candidate.get("full_name") or ""),
                str(candidate.get("current_title") or ""),
                str(candidate.get("current_company") or ""),
            ]
        ).lower()
        if keyword_lower and keyword_lower not in haystack:
            continue
        if selected_locations and candidate.get("location") not in selected_locations:
            continue
        if selected_sectors and not set(selected_sectors).intersection(candidate.get("sectors") or []):
            continue
        if selected_functions and not set(selected_functions).intersection(candidate.get("functions") or []):
            continue
        if (candidate.get("confidence_score") or 0.0) < min_confidence:
            continue
        filtered_candidates.append(candidate)

    table_data = [
        {
            "ID": c.get("candidate_id"),
            "Name": c.get("full_name"),
            "Title": c.get("current_title"),
            "Company": c.get("current_company"),
            "Location": c.get("location"),
            "Sectors": ", ".join(c.get("sectors") or []),
            "Functions": ", ".join(c.get("functions") or []),
            "Confidence": round((c.get("confidence_score") or 0.0) * 100, 0),
        }
        for c in filtered_candidates
    ]

    st.caption(f"{len(filtered_candidates)} of {len(candidates)} candidates shown after filtering.")
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("#### Candidate Detail")
    if not table_data:
        st.info("No candidates match the current filters. Try broadening the criteria.")
    else:
        selected_id = st.selectbox(
            "Select a candidate to inspect",
            options=[row["ID"] for row in table_data],
            format_func=lambda cid: next((r["Name"] for r in table_data if r["ID"] == cid), cid),
        )

        detail = next((c for c in filtered_candidates if c.get("candidate_id") == selected_id), None)
    if table_data and detail:
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.markdown(f"**{detail.get('full_name')}**")
            st.caption(f"{detail.get('current_title')} @ {detail.get('current_company')}")
            st.markdown(f"Location: `{detail.get('location')}`")
            st.markdown(f"Sectors: {', '.join(detail.get('sectors') or []) or '-'}")
            st.markdown(f"Functions: {', '.join(detail.get('functions') or []) or '-'}")
            st.markdown("**Summary**")
            st.write(detail.get("summary") or "-")
            st.markdown(f"Years of experience: `{detail.get('years_experience') or '-'}`")
            st.markdown(f"Skills: {', '.join(detail.get('skills') or []) or '-'}")
            if detail.get("employment_history"):
                st.markdown("**Employment history**")
                for item in detail["employment_history"]:
                    end_year = item.get("end_year") or "Present"
                    st.markdown(
                        f"- {item.get('title')} | {item.get('company')} | {item.get('start_year')} - {end_year} | {item.get('location')}"
                    )

        with col_right:
            st.metric("Confidence score", f"{(detail.get('confidence_score') or 0) * 100:.0f}")
            st.metric("Source", detail.get("source_type") or "-")
            if detail.get("source_urls"):
                st.markdown("**Source URLs**")
                for url in detail["source_urls"]:
                    st.markdown(f"- `{url}`")

        with st.expander("View full JSON (debug)", expanded=False):
            st.json(detail)

    st.session_state["step_2_done"] = True

st.markdown("---")
col_prev, col_next = st.columns([1, 1])
with col_prev:
    if st.button("← Back: Role Intake", use_container_width=True):
        try:
            st.switch_page("pages/1_role_intake.py")
        except Exception:
            st.warning("Automatic navigation is unavailable. Please select “1) Role Intake” from the sidebar.")
with col_next:
    disabled = not bool(st.session_state.get("step_2_done"))
    if st.button("Next: Market Map →", type="primary", disabled=disabled, use_container_width=True):
        try:
            st.switch_page("pages/3_market_map.py")
        except Exception:
            st.warning("Automatic navigation is unavailable. Please select “3) Market Map” from the sidebar.")
