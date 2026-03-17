from __future__ import annotations

import uuid

from app.adapters.ats import ATSAdapter
from app.agents.critique_agent import CritiqueAgent
from app.agents.planner_agent import PlannerAgent
from app.core.audit import AuditEvent, AuditLogger
from app.core.pii import mask_pii
from app.llm.anthropic_client import ClaudeClient
from app.repositories.factory import get_candidate_repository
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore
from app.services.brief_service import BriefService
from app.services.candidate_service import CandidateService
from app.services.job_parser_service import JobParserService
from app.services.ranking_service import RankingService
from app.workflows.shared_state import SearchState


def run_workflow(raw_input: str, *, tenant_id: str = "demo-tenant", ats_adapter: ATSAdapter | None = None) -> SearchState:
    """
    LangGraph 可用时走图；否则走顺序执行。
    """
    request_id = f"req_{uuid.uuid4().hex[:10]}"
    llm = ClaudeClient()
    parser = JobParserService(llm=llm)
    store = VectorStore()
    retriever = Retriever(store=store)
    candidate_svc = CandidateService(ats_adapter=ats_adapter, candidate_repository=get_candidate_repository())
    ranker = RankingService()
    brief_svc = BriefService(llm=llm)
    planner = PlannerAgent()
    critic = CritiqueAgent()
    audit = AuditLogger()

    state: SearchState = {"request_id": request_id, "raw_user_input": raw_input}
    audit.log(
        AuditEvent(
            event_type="request_received",
            request_id=request_id,
            payload={"raw_input": mask_pii(raw_input)},
            tenant_id=tenant_id,
        )
    )

    role_spec = parser.parse_role(raw_input)
    state["parsed_role"] = role_spec
    audit.log(
        AuditEvent(
            event_type="role_parsed",
            request_id=request_id,
            payload={"role_spec": {k: v for k, v in role_spec.items() if k != "_prompt"}, "prompt": role_spec.get("_prompt")},
            tenant_id=tenant_id,
        )
    )

    plan = planner.plan(role_spec)
    audit.log(
        AuditEvent(
            event_type="plan_created",
            request_id=request_id,
            payload={"mode": plan.mode, "steps": plan.steps},
            tenant_id=tenant_id,
        )
    )

    retrieval_context = retriever.retrieve_for_role(role_spec, top_k=5)
    state["retrieval_context"] = retrieval_context
    audit.log(
        AuditEvent(
            event_type="context_retrieved",
            request_id=request_id,
            payload={"doc_ids": [d.get("doc_id") for d in retrieval_context], "store_mode": store.mode},
            tenant_id=tenant_id,
        )
    )

    pool = candidate_svc.load_candidates(role_spec, tenant_id=tenant_id)
    filtered = candidate_svc.filter_candidates(role_spec, pool)
    state["candidate_pool"] = filtered
    audit.log(
        AuditEvent(
            event_type="candidates_collected",
            request_id=request_id,
            payload={"count": len(filtered), "tenant_id": tenant_id, "external_adapter": bool(ats_adapter)},
            tenant_id=tenant_id,
        )
    )

    ranked = ranker.score_candidates(role_spec, filtered)
    state["ranking_results"] = ranked
    audit.log(
        AuditEvent(
            event_type="candidates_ranked",
            request_id=request_id,
            payload={"top": ranked[:5]},
            tenant_id=tenant_id,
        )
    )

    brief_out = brief_svc.generate_markdown(
        role_spec=role_spec,
        ranked_candidates=ranked,
        retrieval_context=retrieval_context,
        generation_context={"tenant_id": tenant_id},
    )
    brief_md = brief_out["markdown"]
    state["brief_draft"] = brief_md
    audit.log(
        AuditEvent(
            event_type="brief_generated",
            request_id=request_id,
            payload={
                "brief_id": brief_out["brief_id"],
                "citations": brief_out.get("citations", []),
                "prompt": brief_out.get("prompt"),
                "generation_context": brief_out.get("generation_context", {}),
            },
            tenant_id=tenant_id,
        )
    )

    critique = critic.critique(brief_md)
    state["critique_feedback"] = critique.issues
    audit.log(
        AuditEvent(
            event_type="brief_critique",
            request_id=request_id,
            payload={"ok": critique.ok, "issues": critique.issues},
            tenant_id=tenant_id,
        )
    )

    return state
