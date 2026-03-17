from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Review Console", layout="wide")

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def api_headers() -> dict[str, str]:
    return {
        "x-tenant-id": st.session_state["tenant_id"],
        "x-user-role": st.session_state["user_role"],
        "x-user-id": st.session_state["user_id"],
    }


def api_get(path: str, **params: Any) -> dict[str, Any]:
    response = httpx.get(f"{API_BASE_URL}{path}", headers=api_headers(), params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    response = httpx.post(f"{API_BASE_URL}{path}", headers=api_headers(), json=payload or {}, timeout=90)
    response.raise_for_status()
    return response.json()


def refresh_review(project_id: str) -> dict[str, Any]:
    review = api_get(f"/projects/{project_id}/review")
    st.session_state["selected_project_id"] = project_id
    st.session_state["review_payload"] = review
    return review


def run_action(path: str, payload: dict[str, Any] | None = None, *, success_message: str) -> None:
    try:
        api_post(path, payload)
        st.session_state["flash_message"] = success_message
        st.session_state["selected_project_id"] = st.session_state.get("selected_project_id")
        st.rerun()
    except httpx.HTTPStatusError as exc:
        st.error(exc.response.text)


st.title("Project Review Console")
st.caption("Internal reviewer console for project summary, run results, brief versions, approvals, exports, and audit.")

with st.sidebar:
    st.header("Review Context")

    st.session_state.setdefault("tenant_id", "demo-review-tenant")
    st.session_state.setdefault("user_role", "researcher")
    st.session_state.setdefault("user_id", "demo_researcher")
    st.session_state.setdefault("_last_user_role", st.session_state["user_role"])

    tenant_id = st.text_input("Tenant ID", key="tenant_id")
    user_role = st.selectbox("Role", options=["researcher", "consultant", "admin"], key="user_role")

    expected_default_user_id = "demo_researcher" if user_role == "researcher" else "demo_manager"

    if st.session_state["_last_user_role"] != user_role:
        current_user_id = st.session_state.get("user_id", "")
        previous_default = "demo_researcher" if st.session_state["_last_user_role"] == "researcher" else "demo_manager"

        # Only switch automatically when the current user_id is still the previous default.
        if current_user_id == previous_default or not current_user_id:
            st.session_state["user_id"] = expected_default_user_id

        st.session_state["_last_user_role"] = user_role

    st.text_input("User ID", key="user_id")

    if st.button("Refresh Project List", use_container_width=True):
        st.session_state.pop("review_payload", None)


projects_body = api_get("/projects", view="summary")
projects = projects_body.get("projects", [])
status_options = ["all"] + sorted({item.get("project_status") for item in projects if item.get("project_status")})
status_filter = st.selectbox("Project Status Filter", options=status_options)
filtered_projects = [item for item in projects if status_filter == "all" or item.get("project_status") == status_filter]

st.subheader("Project List")
if filtered_projects:
    project_df = pd.DataFrame(filtered_projects)[
        [
            "project_id",
            "project_name",
            "client_name",
            "role_title",
            "project_status",
            "latest_run_status",
            "latest_brief_status",
            "has_pending_approval",
            "last_exported_at",
        ]
    ]
    st.dataframe(project_df, use_container_width=True, hide_index=True)
else:
    st.info("No projects available for the current tenant.")

project_ids = [item["project_id"] for item in filtered_projects]
selected_project_id = st.selectbox("Open Project", options=project_ids, index=0 if project_ids else None)

if selected_project_id:
    try:
        review = refresh_review(selected_project_id)
    except httpx.HTTPStatusError as exc:
        st.error(exc.response.text)
        st.stop()

    if st.session_state.get("flash_message"):
        st.success(st.session_state.pop("flash_message"))

    summary = review["summary"]
    latest_run = review["latest_run"]
    latest_snapshot = review["latest_snapshot"]
    briefs = review["briefs"]
    latest_brief = review["latest_brief"]
    audit_timeline = review["audit_timeline"]

    st.subheader("Project Detail")
    top_cols = st.columns(4)
    top_cols[0].metric("Project Status", summary["project_status"])
    top_cols[1].metric("Latest Run", summary["latest_run_status"] or "-")
    top_cols[2].metric("Latest Brief", summary["latest_brief_status"] or "-")
    top_cols[3].metric("Pending Approval", "Yes" if summary["has_pending_approval"] else "No")

    info_col, action_col = st.columns([2, 1])
    with info_col:
        project_payload = dict(review["project"])
        stored_status = project_payload.pop("status", None)
        st.json(project_payload)
        with st.expander("Legacy Stored Project Status", expanded=False):
            st.caption("This is the raw persisted field and may lag behind the computed business status.")
            st.write({"stored_status": stored_status})
        st.json(summary)
    with action_col:
        jd_text = st.text_area("JD Text", value="We are seeking a senior infrastructure portfolio manager in Australia", height=140)
        if st.button("Run Search", use_container_width=True):
            run_action(
                f"/projects/{selected_project_id}/run-search",
                {"jd_text": jd_text, "candidate_source": "local_first"},
                success_message="Search run completed.",
            )

        if latest_brief:
            brief_id = latest_brief["brief_id"]
            action_map = [
                ("Submit For Approval", f"/briefs/{brief_id}/submit", {"notes": "Submitted from review console"}),
                ("Approve", f"/briefs/{brief_id}/approve", {"notes": "Approved from review console"}),
                ("Reject", f"/briefs/{brief_id}/reject", {"notes": "Rejected from review console"}),
                ("Request Changes", f"/briefs/{brief_id}/request-changes", {"notes": "Changes requested from review console"}),
                ("Export", f"/briefs/{brief_id}/export", {"export_format": "md"}),
            ]
            for label, path, payload in action_map:
                disabled = False
                if label == "Submit For Approval":
                    disabled = latest_brief["status"] not in {"draft", "changes_requested"}
                elif label in {"Approve", "Reject", "Request Changes"}:
                    disabled = latest_brief["status"] != "pending_approval" or st.session_state["user_role"] == "researcher"
                elif label == "Export":
                    disabled = latest_brief["status"] != "approved" or st.session_state["user_role"] == "researcher"
                if st.button(label, use_container_width=True, disabled=disabled, key=label):
                    run_action(path, payload, success_message=f"{label} succeeded.")

            revision_content = st.text_area("Revision Content", value=(latest_brief or {}).get("content") or "", height=120)
            revision_disabled = latest_brief["status"] != "changes_requested"
            if st.button("Create Revision", use_container_width=True, disabled=revision_disabled):
                run_action(
                    f"/briefs/{brief_id}/create-revision",
                    {"content": revision_content},
                    success_message="Revision created.",
                )

    st.markdown("---")
    snap_col, brief_col = st.columns(2)
    with snap_col:
        st.markdown("### Latest Snapshot")
        if latest_snapshot:
            st.write(f"Snapshot: `{latest_snapshot['snapshot_id']}`")
            st.dataframe(pd.DataFrame(latest_snapshot["top_candidates"]), use_container_width=True, hide_index=True)
        else:
            st.info("No snapshot yet.")
    with brief_col:
        st.markdown("### Latest Brief")
        if latest_brief:
            st.write(f"Brief: `{latest_brief['brief_id']}` v{latest_brief['version']} [{latest_brief['status']}]")
            st.markdown(latest_brief["content"])
            if latest_brief["status"] == "exported":
                try:
                    artifact = api_get(f"/briefs/{latest_brief['brief_id']}/artifact")
                    st.markdown("#### Artifact")
                    st.json(
                        {
                            "brief_id": artifact["brief_id"],
                            "project_id": artifact["project_id"],
                            "version": artifact["version"],
                            "status": artifact["status"],
                            "exported_by": artifact["exported_by"],
                            "exported_at": artifact["exported_at"],
                            "content_type": artifact["content_type"],
                        }
                    )
                    st.code(artifact["content"], language="markdown")
                except httpx.HTTPStatusError as exc:
                    st.warning(exc.response.text)
        else:
            st.info("No brief yet.")

    st.markdown("### Brief Versions")
    if briefs:
        st.dataframe(pd.DataFrame(briefs), use_container_width=True, hide_index=True)
    else:
        st.info("No brief versions yet.")

    st.markdown("### Audit Timeline")
    action_filter = st.selectbox("Audit Action Filter", options=["all"] + sorted({item["action"] for item in audit_timeline if item.get("action")}))
    resource_filter = st.selectbox(
        "Audit Resource Filter",
        options=["all"] + sorted({item["resource_type"] for item in audit_timeline if item.get("resource_type")}),
    )
    filtered_audit = [
        item
        for item in audit_timeline
        if (action_filter == "all" or item.get("action") == action_filter)
        and (resource_filter == "all" or item.get("resource_type") == resource_filter)
    ]
    st.dataframe(pd.DataFrame(filtered_audit), use_container_width=True, hide_index=True)
