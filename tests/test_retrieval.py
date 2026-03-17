from __future__ import annotations

from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore


def test_retriever_returns_sector_filtered_results() -> None:
    store = VectorStore()
    store._mode = "memory"
    store._memory_docs = [
        {"doc_id": "doc_1", "text": "Infrastructure investor in Australia", "metadata": {"sector": "Infra"}},
        {"doc_id": "doc_2", "text": "Healthcare operator", "metadata": {"sector": "Health"}},
    ]
    retriever = Retriever(store=store)

    results = retriever.retrieve_for_role({"search_keywords": ["infrastructure"], "sector": "Infra"})

    assert len(results) == 1
    assert results[0]["doc_id"] == "doc_1"
