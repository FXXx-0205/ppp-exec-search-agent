from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.adapters.ats import ATSAdapter
from app.core.audit import AuditEvent, AuditLogger
from app.core.exceptions import NotFoundError
from app.models.auth import AccessContext
from app.models.workflow import BriefStatus, SearchRunStatus, SearchWorkflowResult
from app.repositories.factory import (
    get_brief_repository,
    get_candidate_repository,
    get_project_repository,
    get_search_result_snapshot_repository,
    get_search_run_repository,
)
from app.repositories.interfaces import StoredBrief, StoredSearchResultSnapshot, StoredSearchRun
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore
from app.services.candidate_service import CandidateService
from app.services.ranking_service import RankingService


class SearchWorkflow:
    def __init__(
        self,
        *,
        parser: Any,
        candidate_service: Any,
        ranking_service: RankingService,
        brief_service: Any,
        retriever: Any | None = None,
        project_repo=None,
        run_repo=None,
        brief_repo=None,
        snapshot_repo=None,
        audit_logger: AuditLogger | None = None,
    ):
        self.parser = parser
        self.candidate_service = candidate_service
        self.ranking_service = ranking_service
        self.brief_service = brief_service
        self.retriever = retriever or Retriever(store=VectorStore())
        self.project_repo = project_repo or get_project_repository()
        self.run_repo = run_repo or get_search_run_repository()
        self.brief_repo = brief_repo or get_brief_repository()
        self.snapshot_repo = snapshot_repo or get_search_result_snapshot_repository()
        self.audit = audit_logger or AuditLogger()

    def run(
        self,
        project_id: str,
        jd_text: str,
        user: AccessContext,
        *,
        candidate_source: str = "local_first",
        ats_adapter: ATSAdapter | None = None,
    ) -> SearchWorkflowResult:
        project = self.project_repo.get_project(project_id)
        if project is None or project.tenant_id != user.tenant_id:
            raise NotFoundError("Project not found.", details={"project_id": project_id})

        now = datetime.now(timezone.utc).isoformat()
        run = StoredSearchRun(
            run_id=f"run_{uuid4().hex[:12]}",
            project_id=project_id,
            tenant_id=user.tenant_id,
            input_jd=jd_text,
            parsed_role_json=None,
            candidate_source=candidate_source,
            result_count=None,
            run_status=SearchRunStatus.DRAFT,
            started_at=now,
            completed_at=None,
            failed_at=None,
            error_message=None,
            created_by=user.actor.user_id,
            metadata=None,
        )
        self.run_repo.create_run(run)
        self._audit(
            event_type="search_run_started",
            action="start_search_run",
            resource_type="search_run",
            resource_id=run.run_id,
            user=user,
            project_id=project_id,
            run_id=run.run_id,
            payload={"candidate_source": candidate_source},
        )

        try:
            parsed_role = self.parser.parse_role(jd_text)
            run = self._update_run(run, run_status=SearchRunStatus.PARSED, parsed_role_json=parsed_role)
            self._audit(
                event_type="role_parsed",
                action="parse_role_success",
                resource_type="search_run",
                resource_id=run.run_id,
                user=user,
                project_id=project_id,
                run_id=run.run_id,
                payload={"parsed_role": {k: v for k, v in parsed_role.items() if k != "_prompt"}},
            )

            retrieval_context = self.retriever.retrieve_for_role(parsed_role, top_k=5)
            service = self.candidate_service
            if ats_adapter is not None:
                service = CandidateService(ats_adapter=ats_adapter, candidate_repository=get_candidate_repository())
            pool = service.load_candidates(parsed_role, tenant_id=user.tenant_id)
            filtered = service.filter_candidates(parsed_role, pool)
            run = self._update_run(run, run_status=SearchRunStatus.SEARCHED, result_count=len(filtered))
            self._audit(
                event_type="candidate_search_completed",
                action="candidate_search_success",
                resource_type="search_run",
                resource_id=run.run_id,
                user=user,
                project_id=project_id,
                run_id=run.run_id,
                payload={"result_count": len(filtered), "candidate_source": candidate_source},
            )

            ranked = self.ranking_service.score_candidates(parsed_role, filtered)
            run = self._update_run(run, run_status=SearchRunStatus.RANKED)
            self._audit(
                event_type="candidates_ranked",
                action="ranking_success",
                resource_type="search_run",
                resource_id=run.run_id,
                user=user,
                project_id=project_id,
                run_id=run.run_id,
                payload={"candidate_count": len(ranked), "top_candidate_ids": [item.get("candidate_id") for item in ranked[:5]]},
            )
            snapshot = StoredSearchResultSnapshot(
                snapshot_id=f"snap_{uuid4().hex[:12]}",
                run_id=run.run_id,
                project_id=project_id,
                tenant_id=user.tenant_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                top_candidates=ranked[:10],
                ranking_payload=ranked,
                candidate_count=len(ranked),
                metadata={"candidate_source": candidate_source},
            )
            self.snapshot_repo.create_snapshot(snapshot)

            brief_out = self.brief_service.generate_markdown(
                role_spec=parsed_role,
                ranked_candidates=ranked,
                retrieval_context=retrieval_context,
                generation_context={"tenant_id": user.tenant_id, "project_id": project_id, "run_id": run.run_id},
            )
            existing = self.brief_repo.get_latest_brief_by_project(project_id=project_id, tenant_id=user.tenant_id)
            version = 1 if existing is None else existing.version + 1
            created_at = brief_out["generated_at"]
            brief = StoredBrief(
                brief_id=brief_out["brief_id"],
                project_id=project_id,
                tenant_id=user.tenant_id,
                version=version,
                content=brief_out["markdown"],
                status=BriefStatus.DRAFT,
                created_by=user.actor.user_id,
                created_at=created_at,
                updated_at=created_at,
                role_spec=parsed_role,
                run_id=run.run_id,
                metadata={"citations": brief_out.get("citations", []), "prompt": brief_out.get("prompt")},
                markdown=brief_out["markdown"],
                citations=brief_out.get("citations", []),
                generated_at=created_at,
            )
            self.brief_repo.create_brief_version(brief)
            run = self._update_run(run, run_status=SearchRunStatus.BRIEF_GENERATED)
            self._audit(
                event_type="brief_generated",
                action="brief_generated",
                resource_type="brief",
                resource_id=brief.brief_id,
                user=user,
                project_id=project_id,
                run_id=run.run_id,
                brief_id=brief.brief_id,
                payload={"version": version, "candidate_count": len(ranked), "snapshot_id": snapshot.snapshot_id},
            )
            run = self._update_run(run, run_status=SearchRunStatus.COMPLETED, completed=True)
            self._audit(
                event_type="run_completed",
                action="run_completed",
                resource_type="search_run",
                resource_id=run.run_id,
                user=user,
                project_id=project_id,
                run_id=run.run_id,
                brief_id=brief.brief_id,
                payload={"status": run.run_status, "snapshot_id": snapshot.snapshot_id, "brief_id": brief.brief_id},
            )

            return SearchWorkflowResult(
                project_id=project_id,
                run_id=run.run_id,
                snapshot_id=snapshot.snapshot_id,
                parsed_role=parsed_role,
                candidate_count=len(ranked),
                ranked_candidates=ranked,
                brief_id=brief.brief_id,
                brief_version=version,
                run_status=run.run_status,
                warnings=[],
            )
        except Exception as exc:
            failed_at = datetime.now(timezone.utc).isoformat()
            self.run_repo.mark_run_failed(run.run_id, error_message=str(exc), failed_at=failed_at)
            self._audit(
                event_type="search_run_failed",
                action="workflow_failed",
                resource_type="search_run",
                resource_id=run.run_id,
                user=user,
                project_id=project_id,
                run_id=run.run_id,
                payload={"error_message": str(exc)},
            )
            raise

    def _update_run(
        self,
        run: StoredSearchRun,
        *,
        run_status: SearchRunStatus,
        parsed_role_json: dict | None = None,
        result_count: int | None = None,
        completed: bool = False,
    ) -> StoredSearchRun:
        data = asdict(run)
        data["run_status"] = run_status
        if parsed_role_json is not None:
            data["parsed_role_json"] = parsed_role_json
        if result_count is not None:
            data["result_count"] = result_count
        if completed:
            data["completed_at"] = datetime.now(timezone.utc).isoformat()
        updated = StoredSearchRun(**data)
        return self.run_repo.update_run(updated)

    def _audit(
        self,
        *,
        event_type: str,
        action: str,
        resource_type: str,
        resource_id: str,
        user: AccessContext,
        project_id: str,
        payload: dict,
        run_id: str | None = None,
        brief_id: str | None = None,
    ) -> None:
        self.audit.log(
            AuditEvent(
                event_type=event_type,
                request_id=run_id or resource_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                payload=payload,
                tenant_id=user.tenant_id,
                project_id=project_id,
                run_id=run_id,
                brief_id=brief_id,
                actor_id=user.actor.user_id,
            )
        )
