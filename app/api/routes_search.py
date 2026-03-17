from __future__ import annotations

from fastapi import APIRouter, Depends

from app.adapters.ats import ATSAdapter
from app.adapters.crm import CRMAdapter
from app.adapters.doc_store import DocumentStoreAdapter
from app.api.dependencies.auth import require_permission
from app.api.dependencies.integrations import get_ats_adapter, get_crm_adapter, get_document_store_adapter
from app.llm.anthropic_client import ClaudeClient
from app.models.auth import AccessContext
from app.models.search_request import CandidatesRequest, IntakeRequest, RankRequest
from app.repositories.factory import get_candidate_repository
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore
from app.services.brief_service import BriefService
from app.services.candidate_service import CandidateService
from app.services.job_parser_service import JobParserService
from app.services.ranking_service import RankingService
from app.workflows.candidate_search_graph import run_workflow
from app.workflows.search_workflow import SearchWorkflow

router = APIRouter()

_llm = ClaudeClient()
_parser = JobParserService(llm=_llm)
_candidates = CandidateService()
_ranker = RankingService()
_store = VectorStore()
_retriever = Retriever(store=_store)


@router.post("/intake")
def intake(
    req: IntakeRequest,
    access: AccessContext = Depends(require_permission("search:run")),
    doc_store: DocumentStoreAdapter = Depends(get_document_store_adapter),
    crm: CRMAdapter = Depends(get_crm_adapter),
) -> dict:
    role_spec = _parser.parse_role(req.raw_input)
    retrieval_context = _retriever.retrieve_for_role(role_spec, top_k=5)
    project = crm.get_project(req.project_id) if req.project_id else None
    knowledge_docs = doc_store.list_documents("firm_profiles")
    return {
        "role_spec": role_spec,
        "retrieval_context": retrieval_context,
        "vector_store_mode": _store.mode,
        "tenant_id": access.tenant_id,
        "project": project.__dict__ if project else None,
        "knowledge_sources": [doc.__dict__ for doc in knowledge_docs[:5]],
    }


@router.post("/candidates")
def candidates(
    req: CandidatesRequest,
    access: AccessContext = Depends(require_permission("search:run")),
    ats: ATSAdapter = Depends(get_ats_adapter),
) -> dict:
    service = CandidateService(ats_adapter=ats, candidate_repository=get_candidate_repository())
    pool = service.load_candidates(req.role_spec, tenant_id=access.tenant_id, provider_filters=req.provider_filters)
    filtered = _candidates.filter_candidates(req.role_spec, pool)
    return {"candidates": filtered, "count": len(filtered), "tenant_id": access.tenant_id}


@router.post("/rank")
def rank(
    req: RankRequest,
    access: AccessContext = Depends(require_permission("search:run")),
    ats: ATSAdapter = Depends(get_ats_adapter),
) -> dict:
    service = CandidateService(ats_adapter=ats, candidate_repository=get_candidate_repository())
    pool = service.load_candidates(req.role_spec, tenant_id=access.tenant_id)
    by_id = {c.get("candidate_id"): c for c in pool}
    selected = [by_id[cid] for cid in req.candidate_ids if cid in by_id]
    ranked = _ranker.score_candidates(req.role_spec, selected or pool)
    return {"ranked_candidates": ranked, "count": len(ranked), "tenant_id": access.tenant_id}


@router.post("/run")
def run(
    req: IntakeRequest,
    access: AccessContext = Depends(require_permission("search:run")),
    ats: ATSAdapter = Depends(get_ats_adapter),
) -> dict:
    """
    Agentic demo entrypoint: run end-to-end workflow and return a compact result.
    """
    if req.project_id:
        workflow = SearchWorkflow(
            parser=_parser,
            candidate_service=CandidateService(ats_adapter=ats, candidate_repository=get_candidate_repository()),
            ranking_service=_ranker,
            brief_service=BriefService(llm=_llm),
        )
        result = workflow.run(req.project_id, req.raw_input, access, ats_adapter=ats)
        return result.model_dump(mode="json")
    state = run_workflow(req.raw_input, tenant_id=access.tenant_id, ats_adapter=ats)
    return {
        "request_id": state.get("request_id"),
        "tenant_id": access.tenant_id,
        "role_spec": state.get("parsed_role"),
        "retrieval_context": state.get("retrieval_context", []),
        "ranked_candidates": (state.get("ranking_results") or [])[:10],
        "brief_markdown": state.get("brief_draft"),
        "critique_feedback": state.get("critique_feedback", []),
    }
