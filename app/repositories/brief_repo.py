from __future__ import annotations

import builtins
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.models.auth import ApprovalStatus
from app.models.workflow import BriefStatus
from app.repositories.interfaces import StoredBrief


class BriefRepo:
    """
    MVP: file-backed storage under data/processed/briefs.
    """

    def __init__(self, root_dir: str | None = None):
        self.root = Path(root_dir or settings.brief_storage_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, brief_id: str) -> Path:
        return self.root / f"{brief_id}.json"

    def save(self, brief: StoredBrief) -> None:
        self._path(brief.brief_id).write_text(json.dumps(asdict(brief), ensure_ascii=False, indent=2), encoding="utf-8")

    def create_brief_version(self, brief: StoredBrief) -> StoredBrief:
        self.save(brief)
        return brief

    def get(self, brief_id: str) -> StoredBrief | None:
        p = self._path(brief_id)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return StoredBrief(**data)

    def get_brief(self, brief_id: str) -> StoredBrief | None:
        return self.get(brief_id)

    def list(
        self,
        *,
        tenant_id: str,
        project_id: str | None = None,
        approval_status: ApprovalStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> builtins.list[StoredBrief]:
        briefs: builtins.list[StoredBrief] = []
        matched = 0
        for path in sorted(self.root.glob("*.json"), reverse=True):
            try:
                brief = StoredBrief(**json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if brief.tenant_id != tenant_id:
                continue
            if project_id is not None and brief.project_id != project_id:
                continue
            if approval_status is not None and brief.approval_status != approval_status:
                continue
            if matched < offset:
                matched += 1
                continue
            briefs.append(brief)
            if len(briefs) >= limit:
                break
        return briefs

    def list_briefs_by_project(self, *, project_id: str, tenant_id: str) -> builtins.list[StoredBrief]:
        return self.list(tenant_id=tenant_id, project_id=project_id, limit=1000, offset=0)

    def get_latest_brief_by_project(self, *, project_id: str, tenant_id: str) -> StoredBrief | None:
        briefs = self.list_briefs_by_project(project_id=project_id, tenant_id=tenant_id)
        if not briefs:
            return None
        return max(briefs, key=lambda item: item.version)

    def update_brief_status(
        self,
        brief_id: str,
        *,
        status: BriefStatus,
        updated_by: str,
        notes: str | None = None,
        metadata: dict | None = None,
    ) -> StoredBrief | None:
        brief = self.get(brief_id)
        if not brief:
            return None
        now = datetime.now(timezone.utc).isoformat()
        updated = StoredBrief(
            **{
                **asdict(brief),
                "status": status,
                "updated_at": now,
                "metadata": {**(brief.metadata or {}), **(metadata or {})} or None,
                "approval_notes": notes if status == BriefStatus.APPROVED else brief.approval_notes,
                "rejection_notes": notes if status in {BriefStatus.REJECTED, BriefStatus.CHANGES_REQUESTED} else brief.rejection_notes,
                "change_request_notes": notes if status == BriefStatus.CHANGES_REQUESTED else brief.change_request_notes,
                "approved_by": updated_by if status == BriefStatus.APPROVED else brief.approved_by,
                "approved_at": now if status == BriefStatus.APPROVED else brief.approved_at,
            }
        )
        self.save(updated)
        return updated

    def submit_for_approval(self, brief_id: str, *, submitted_by: str, notes: str | None = None) -> StoredBrief | None:
        brief = self.get(brief_id)
        if not brief:
            return None
        now = datetime.now(timezone.utc).isoformat()
        updated = StoredBrief(
            **{
                **asdict(brief),
                "status": BriefStatus.PENDING_APPROVAL,
                "updated_at": now,
                "submitted_by": submitted_by,
                "submitted_at": now,
                "approval_notes": notes if notes else brief.approval_notes,
            }
        )
        self.save(updated)
        return updated

    def approve_brief(self, brief_id: str, *, approved_by: str, notes: str | None = None) -> StoredBrief | None:
        return self.update_brief_status(brief_id, status=BriefStatus.APPROVED, updated_by=approved_by, notes=notes)

    def reject_brief(self, brief_id: str, *, rejected_by: str, notes: str | None = None) -> StoredBrief | None:
        brief = self.get(brief_id)
        if not brief:
            return None
        now = datetime.now(timezone.utc).isoformat()
        updated = StoredBrief(
            **{
                **asdict(brief),
                "status": BriefStatus.REJECTED,
                "updated_at": now,
                "rejection_notes": notes,
            }
        )
        self.save(updated)
        return updated

    def request_changes(self, brief_id: str, *, requested_by: str, notes: str | None = None) -> StoredBrief | None:
        brief = self.get(brief_id)
        if not brief:
            return None
        now = datetime.now(timezone.utc).isoformat()
        updated = StoredBrief(
            **{
                **asdict(brief),
                "status": BriefStatus.CHANGES_REQUESTED,
                "updated_at": now,
                "rejection_notes": notes,
                "change_request_notes": notes,
            }
        )
        self.save(updated)
        return updated

    def mark_exported(self, brief_id: str, *, exported_by: str) -> StoredBrief | None:
        brief = self.get(brief_id)
        if not brief:
            return None
        now = datetime.now(timezone.utc).isoformat()
        updated = StoredBrief(
            **{
                **asdict(brief),
                "status": BriefStatus.EXPORTED,
                "updated_at": now,
                "exported_by": exported_by,
                "exported_at": now,
            }
        )
        self.save(updated)
        return updated

    def create_revision(
        self,
        source_brief_id: str,
        *,
        new_brief_id: str,
        created_by: str,
        content: str | None = None,
    ) -> StoredBrief | None:
        source = self.get(source_brief_id)
        if not source:
            return None
        now = datetime.now(timezone.utc).isoformat()
        revision = StoredBrief(
            **{
                **asdict(source),
                "brief_id": new_brief_id,
                "version": source.version + 1,
                "content": content or source.content,
                "markdown": content or source.markdown,
                "status": BriefStatus.DRAFT,
                "created_by": created_by,
                "created_at": now,
                "updated_at": now,
                "submitted_by": None,
                "submitted_at": None,
                "approved_by": None,
                "approved_at": None,
                "approval_notes": None,
                "rejection_notes": None,
                "exported_by": None,
                "exported_at": None,
                "previous_brief_id": source.brief_id,
                "supersedes_brief_id": source.brief_id,
                "change_request_source_brief_id": source.brief_id,
                "change_request_notes": source.rejection_notes,
            }
        )
        self.save(revision)
        return revision

    def decide(
        self,
        brief_id: str,
        *,
        status: ApprovalStatus,
        decided_by: str,
        comment: str | None = None,
    ) -> StoredBrief | None:
        b = self.get(brief_id)
        if not b:
            return None
        if status == ApprovalStatus.APPROVED:
            return self.approve_brief(brief_id, approved_by=decided_by, notes=comment)
        if status == ApprovalStatus.REJECTED:
            return self.reject_brief(brief_id, rejected_by=decided_by, notes=comment)
        return self.submit_for_approval(brief_id, submitted_by=decided_by, notes=comment)
