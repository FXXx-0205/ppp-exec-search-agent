## Manual Review Checklist

### Happy Path
1. Run `python3 scripts/seed_review_demo.py --reset`.
   Expected: demo tenant `demo-review-tenant` and project `proj_demo_happy` exist.
2. Start API with `uvicorn app.main:app --reload`.
3. Start review console with `streamlit run app/ui/review_console.py`.
4. In the console sidebar, use tenant `demo-review-tenant` and role `researcher`.
5. Open project `proj_demo_happy`.
   Expected: project summary shows an exported/approved happy-path project, latest run exists, latest snapshot exists.
6. Inspect latest snapshot and latest brief.
   Expected: top candidates, latest brief content, and audit timeline are visible.
7. Switch role to `consultant`.
8. Open `/briefs/brief_demo_happy_v1/artifact` through the UI artifact panel or API.
   Expected: artifact returns markdown content, `status=exported`, and exported metadata.
9. Check project audit timeline.
   Expected: create/run/export events are visible with action/resource metadata.

### Revision Path
1. In the review console, open `proj_demo_revision`.
2. Confirm latest brief is `changes_requested`.
   Expected: brief v1 is still preserved and audit shows `request_changes`.
3. Click `Create Revision`.
   Expected: a new brief version is created as `draft` with version incremented.
4. Review the new revision content.
5. Click `Submit For Approval`.
   Expected: new version moves to `pending_approval`.
6. Switch role to `consultant` and click `Approve`.
   Expected: new version moves to `approved`.
7. Click `Export`.
   Expected: new version moves to `exported`.
8. Verify both old and new versions remain visible.

### Permission Path
1. In the review console, switch to role `researcher`.
2. Open a brief in `pending_approval` or `approved`.
3. Try `Approve` or `Export`.
   Expected: button is disabled or API returns 403 if forced directly.
4. Switch to role `consultant`.
5. Repeat `Approve` or `Export`.
   Expected: action succeeds and UI refresh shows updated brief/project state.
