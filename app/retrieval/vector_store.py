from __future__ import annotations

from typing import Any

from app.config import settings


class VectorStore:
    """
    MVP：优先用 Chroma（如可用）；否则用内存简化实现，保证 demo 可跑。
    """

    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self._mode = "memory"
        self._memory_docs: list[dict[str, Any]] = []
        self._chroma = None
        self._collection = None

        try:
            import chromadb

            self._chroma = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._chroma.get_or_create_collection(name="docs")
            self._mode = "chroma"
        except Exception:
            self._mode = "memory"

    @property
    def mode(self) -> str:
        return self._mode

    def upsert(self, docs: list[dict[str, Any]]) -> None:
        if self._mode == "chroma" and self._collection is not None:
            ids = [d["doc_id"] for d in docs]
            texts = [d["text"] for d in docs]
            metadatas = [d.get("metadata") or {} for d in docs]
            self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
            return

        self._memory_docs.extend(docs)

    def query(self, query_text: str, top_k: int = 5, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if self._mode == "chroma" and self._collection is not None:
            try:
                res = self._collection.query(query_texts=[query_text], n_results=top_k, where=where)
                out: list[dict[str, Any]] = []
                ids = res.get("ids") or [[]]
                documents = res.get("documents") or [[]]
                metadatas = res.get("metadatas") or [[{}]]
                for i in range(len(ids[0])):
                    out.append(
                        {
                            "doc_id": ids[0][i],
                            "text": documents[0][i],
                            "metadata": metadatas[0][i] or {},
                            "source": (metadatas[0][i] or {}).get("source"),
                            "title": (metadatas[0][i] or {}).get("title"),
                        }
                    )
                return out
            except Exception:
                # 如果底层 Chroma 出错，不让请求失败，退化为 memory / 空结果
                return []

        q = query_text.lower()
        scored = []
        for d in self._memory_docs:
            if where and any((d.get("metadata") or {}).get(key) != value for key, value in where.items()):
                continue
            text = (d.get("text") or "").lower()
            s = sum(1 for token in q.split() if token and token in text)
            scored.append((s, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for s, d in scored[:top_k] if s > 0]
